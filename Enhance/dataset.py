import torch
from torchvision.datasets import VisionDataset
from PIL import Image
from glob import glob
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
# from pit.util import instantiate_from_config
from functools import partial
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from einops import rearrange
from transformers import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
import numpy as np
from torch.utils.data import Dataset
from vision_process import process_vision_info_my
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
MODEL_ID = "/mnt/workspace/zhangzf/ICML2026/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

def format_data(image_directory_path, entry):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_directory_path + "/" + entry["image"],
                },
                {
                    "type": "text",
                    "text": entry["prefix"],
                },
            ],
        },
        { 
            "role": "assistant",
            "content": [{"type": "text", "text": entry["suffix"]}],
        },
    ]

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()
    def _load_entries(self):
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)
        return image, entry, format_data(self.image_directory_path, entry)
    
class CompressedDataset(Dataset):
    def __init__(self, compressed_paths, original_paths, transform=None):
        self.compressed_paths = compressed_paths
        self.original_paths = original_paths
        compressed_splitdir = Path(compressed_paths)
        original_splitdir = Path(original_paths)
        self.compressed_samples = sorted(f for f in compressed_splitdir.iterdir() if f.is_file())
        self.original_samples = sorted(f for f in original_splitdir.iterdir() if f.is_file())
        self.transform = transform

    def __len__(self):
        return len(self.compressed_samples)

    def __getitem__(self, idx):
        compressed_img = Image.open(self.compressed_samples[idx]).convert("RGB")
        original_img = Image.open(self.original_samples[idx]).convert("RGB")

        if self.transform:
            compressed_img = self.transform(compressed_img)
            original_img = self.transform(original_img)

        return compressed_img, original_img




class CompressedDatasetWithAnno(Dataset):
    def __init__(self, annotation_file, transform=None):

        self.annotation_file = annotation_file
        self.transform = transform

        self.data = self._load_annotations(annotation_file)

    def _load_annotations(self, annotation_file):
        data = []
        with open(annotation_file, 'r') as file:
            for line in file.readlines():
                # 解析每一行的数据
                original_path, compressed_path, compression_type, compression_level = line.strip().split(', ')
                data.append((original_path, compressed_path, compression_type, int(compression_level)))
        return data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取索引为idx的图片及相关信息
        
        Args:
            idx (int): 索引

        Returns:
            dict: 一个包含原始图像、压缩图像、压缩类型、压缩强度的字典
        """
        
        try:
            original_path, compressed_path, compression_type, compression_level = self.data[idx]
            original_image = Image.open(original_path).convert('RGB')
            compressed_image = Image.open(compressed_path).convert('RGB')
        except Exception as e:
            original_path, compressed_path, compression_type, compression_level = self.data[0]
            original_image = Image.open(original_path).convert('RGB')
            compressed_image = Image.open(compressed_path).convert('RGB')


        if self.transform:
            original_image = self.transform(original_image)
            compressed_image = self.transform(compressed_image)

        # 返回图像数据和标签
        return original_image, compressed_image, compression_type, compression_level





class GreedyCompressedDatasetWithAnno(torch.utils.data.IterableDataset):
    def __init__(self, root, max_seq_len, patch_size, rank, t_patch_size=2):
        super().__init__()
        self.t = t_patch_size # temporal_patch_size
        self.p = patch_size
        self.pm = self.p * 2
        assert(max_seq_len % (2 ** 2) == 0)
        self.rank = rank
        self.max_seq_len = max_seq_len

        self.data = self._load_annotations(root)

        assert len(self.data) > 0, "File list is empty. Check the root."
        self.len = len(self.data)
        self.cnt = 0
        self.round = 0
        self.rank = rank
        self.seeding()

    def _load_annotations(self, annotation_file):
        data = []
        with open(annotation_file, 'r') as file:
            for line in file.readlines():
                # 解析每一行的数据
                original_path, compressed_path, compression_type, compression_level = line.strip().split(', ')
                data.append((original_path, compressed_path, compression_type, int(compression_level)))
        return data
    
    
    def seeding(self):
        prev_rgn = np.random.get_state()
        np.random.seed(self.rank + self.round)
        self.idxs = np.random.shuffle(np.arange(0,self.len))
        np.random.set_state(prev_rgn)

    def __iter__(self):
        while True:
            yield self.getseq()

    def getseq(self):
        imgs = []
        ori_images = []
        codec_infos = []
        sizes = []
        ori_sizes = []
        vals = []
        cur_len = 0
        while True:
            nex_out = self.getone()
            nex_len = cur_len + nex_out["img"].shape[0]
            if nex_len > self.max_seq_len:
                break
            else:
                cur_len = nex_len 
                imgs.append(nex_out["img"])
                sizes.append(nex_out["size"])
                vals.append(nex_out["val"])
                ori_images.append(nex_out["ori_img"])
                ori_sizes.append(nex_out["ori_size"])
                codec_infos.append(nex_out["codec_label"])
        if cur_len < self.max_seq_len:
            pad_len = self.max_seq_len - cur_len
            pad_h = self.pm
            pad_img = torch.zeros([pad_len, 3 * self.p * self.p * self.t])
            pad_size = torch.tensor([1, pad_h // self.p, pad_len // (pad_h // self.p)], dtype=torch.long)[None]
            pad_val = torch.zeros([pad_len])

            imgs.append(pad_img)
            ori_images.append(pad_img)
            sizes.append(pad_size)
            ori_sizes.append(pad_size)
            vals.append(pad_val)
            codec_infos.append(('JPEG', 0))

        imgs = torch.cat(imgs, dim=0)
        ori_images = torch.cat(ori_images, dim=0)
        sizes = torch.cat(sizes, dim=0)
        ori_sizes = torch.cat(ori_sizes, dim=0)
        vals = torch.cat(vals, dim=0)

        return { "img": imgs, "ori_img": ori_images, "size": sizes, "val": vals, "codec_label": codec_infos, "ori_size": ori_sizes }


    def getone(self):
        out = self.getidx_zzf(self.cnt)
        self.cnt += 1
        if self.cnt >= self.len:
            self.cnt = 0
            self.round += 1
            self.seeding()
        return out


    def getidx_zzf(self, index: int):
        # fpath = self.fpaths[index]
        original_path, compressed_path, compression_type, compression_level = self.data[index]
        compressed_image = Image.open(compressed_path).convert("RGB")
        compressed_image_input = process_vision_info_my([{"image": compressed_image}])

        original_image = Image.open(original_path).convert('RGB')
        original_image_input = process_vision_info_my([{"image": original_image}])

    
        output_kwargs = processor._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
        )
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        codec_label = (compression_type, compression_level)

        compressed_image_output = processor.image_processor(images=compressed_image_input, **output_kwargs["images_kwargs"])
        original_image_output = processor.image_processor(images=original_image_input, **output_kwargs["images_kwargs"])


        compressed_image_output = BatchFeature(data={**compressed_image_output}, tensor_type=return_tensors)
        original_image_output = BatchFeature(data={**original_image_output}, tensor_type=return_tensors)
        compressed_pixel_value = compressed_image_output["pixel_values"]
        compressed_image_grid_thw = compressed_image_output["image_grid_thw"]
        original_pixel_value = original_image_output["pixel_values"]
        original_image_grid_thw = original_image_output["image_grid_thw"]

        val = torch.ones([compressed_pixel_value.shape[0]])    
        return {
            "img": compressed_pixel_value,
            "size": compressed_image_grid_thw,
            "ori_img": original_pixel_value,
            "ori_size": original_image_grid_thw,
            "val": val, 
            "codec_label": codec_label,
        }
        

# IMPORTANT: collate function
def sp_collate(original_batch):
    imgs = []
    sizes = []
    vals = []
    ori_imgs = []
    codec_labels = []
    ori_sizes = []
    for ob in original_batch:
        imgs.append(ob["img"])
        sizes.append(ob["size"])
        vals.append(ob["val"])
        ori_imgs.append(ob["ori_img"])
        ori_sizes.append(ob["ori_size"])
        codec_labels.append(ob["codec_label"])
        
    ori_imgs = torch.cat(ori_imgs, dim=0)
    imgs = torch.cat(imgs, dim=0)
    sizes = torch.cat(sizes, dim=0)
    ori_sizes = torch.cat(ori_sizes, dim=0)
    vals = torch.cat(vals, dim=0)
    codec_labels = codec_labels[0]
    return imgs, sizes, ori_imgs, ori_sizes, codec_labels, vals


if __name__ == "__main__":
    # max_seq_len % spatial_merge_size ** 2 == 0
    dataset = GreedyCompressedDatasetWithAnno("/data11/zhangzf2505/COCODataset/COCO_JPEG_ELIC_ILLM.txt", max_seq_len=(2**2) * 1024, rank=0, patch_size=14, t_patch_size=2)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=sp_collate, num_workers=0)
    config = Qwen2_5_VLVisionConfig()
    model = Qwen2_5_VisionTransformerPretrainedModel._from_config(config).cuda()
    for bi, data in enumerate(loader):
        x_compress = data[0].cuda() # source 
        grid_thw = data[1].cuda()
        val_mask = data[2].cuda()
        x_ori = data[3].cuda() # original
        codec_labels = data[4]
        breakpoint()
        x_out = model(x_ori, grid_thw)
        print(x_ori.shape, x_out.shape, grid_thw.shape, val_mask.shape, val_mask[:1], val_mask[-1:], val_mask[::4].shape)
        # for output use every spatial merger size ** 2, which is 4
        # only allow valid output to be involved in computation
        x_out = x_out * val_mask[::4, None]
        breakpoint()

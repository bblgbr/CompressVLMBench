# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from torchvision.transforms import ToTensor
from tqdm import tqdm

from neuralcompression.metrics import (
    MultiscaleStructuralSimilarity,
    calc_psnr,
    pickle_size_of,
    update_patch_fid,
)


def rescale_image(image: Tensor, back_to_float: bool = True) -> Tensor:
    dtype = image.dtype
    image = (image * 255 + 0.5).to(torch.uint8)

    if back_to_float:
        image = image.to(dtype)

    return image


def main():
    parser = ArgumentParser()

    parser.add_argument("clic_path", type=str, help="path to CLIC2020 directory")
    parser.add_argument("--model", type=str, help="msillm_quality", default="msillm_quality_3")
    parser.add_argument("--save_path", type=str, help="save image path")

    args = parser.parse_args()
    clic_path = Path(args.clic_path)

    device = torch.device("cuda")

    # 设置本地缓存目录路径（假设已经存在）
    local_hub_path = os.path.expanduser("~/.cache/torch/hub")  # 或者指定你实际的路径
    # 模型文件所在路径
    model_path = os.path.join(local_hub_path, "facebookresearch_NeuralCompression_main")
    # 假设 args.model 是你要加载的模型名称
    model = torch.hub.load(model_path, args.model, source='local')

    # model = torch.hub.load("facebookresearch/NeuralCompression", args.model)
    model = model.to(device)
    model = model.eval()
    model.update()
    model.update_tensor_devices("compress")

    totensor = ToTensor()

    msssim_metric = MultiscaleStructuralSimilarity(data_range=255.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance().to(device)

    psnr_vals = []
    bpp_vals = []

    for image_path in tqdm(list(clic_path.glob("*.jpg"))):
        with open(image_path, "rb") as f:
            image_pil = Image.open(f)
            image_pil = image_pil.convert("RGB")

        image = totensor(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            compressed = model.compress(image, force_cpu=False)
            decompressed = model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)

        num_bytes = pickle_size_of(compressed)
        bpp = num_bytes * 8 / (image.shape[0] * image.shape[-2] * image.shape[-1])
        bpp_vals.append(float(bpp))

        orig_image = rescale_image(image)
        pred_image = rescale_image(decompressed)
        save_image = pred_image.squeeze(0)
        save_image = save_image.permute(1, 2, 0).to(torch.uint8)

        save_image_pil = Image.fromarray(save_image.cpu().numpy())

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # 保存图像到指定路径
        save_path = os.path.join(args.save_path, os.path.basename(image_path))
        save_image_pil.save(save_path)

        # with torch.no_grad():
        #     # update_patch_fid(image, decompressed, fid_metric)

        #     orig_image = rescale_image(image)
        #     pred_image = rescale_image(decompressed)

            # psnr_val = calc_psnr(pred_image, orig_image)
            # psnr_vals.append(float(psnr_val))
            # msssim_metric(pred_image, orig_image)
            # lpips_metric(decompressed, image)

    bpp_total = sum(bpp_vals) / len(bpp_vals)
    # psnr_total = sum(psnr_vals) / len(psnr_vals)
    # msssim_total = float(msssim_metric.compute())
    # lpips_total = float(lpips_metric.compute())
    # fid_total = float(fid_metric.compute())

    print("Compression complete")
    print(f"Rate: {bpp_total}")
    # print(f"PSNR: {psnr_total}")
    # print(f"MS-SSIM: {msssim_total}")
    # print(f"LPIPS: {lpips_total}")
    # print(f"FID: {fid_total}")


if __name__ == "__main__":
    main()

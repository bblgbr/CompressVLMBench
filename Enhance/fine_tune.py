import os
import logging
import pathlib
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from unifiedencoder_re_elic import Qwen2_5_VisionTransformerPretrainedModelCodec
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
from dataset import JSONLDataset, CompressedDataset, CompressedDatasetWithAnno
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
from vision_process import process_vision_info_my
from transformers.feature_extraction_utils import BatchFeature

BATCH_SIZE = 20
NUM_WORKERS = 10
MODEL_ID = "/data11/zhangzf2505/Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
processor = Qwen2_5_VLProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

 # 注意ddp fix seed，确保初始化一样
# early_stopping_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

class SaveCheckpoint(Callback):
    def __init__(self, result_path):
        self.result_path = result_path
        self.epoch = 0

    def convert_to_float32(self, model):
        state_dict = model.state_dict()
        float32_state_dict = {
            k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }
        return float32_state_dict

    
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            checkpoint_path = f"{self.result_path}/{self.epoch}"
            os.makedirs(checkpoint_path, exist_ok=True)
            state_dict = self.convert_to_float32(pl_module.model)
            torch.save(state_dict, f"{checkpoint_path}/pytorch_model.bin")
            # pl_module.model.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            self.epoch += 1
    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            checkpoint_path = f"{self.result_path}/latest"
            os.makedirs(checkpoint_path, exist_ok=True)
            # pl_module.model.save_pretrained(checkpoint_path)
            state_dict = self.convert_to_float32(pl_module.model)
            torch.save(state_dict, f"{checkpoint_path}/pytorch_model.bin")
            print(f"Saved checkpoint to {checkpoint_path}")

def train_collate_fn(batch):
    original_images, compressed_images, compression_types, compression_levels = zip(*batch)
    compressed_image_inputs = [
        process_vision_info_my([{"image": example, 'resized_height': 336, 'resized_width': 336}])
        for example
        in compressed_images
    ]
    original_image_inputs = [
        process_vision_info_my([{"image": example, 'resized_height': 336, 'resized_width': 336}])
        for example
        in original_images
    ]
    output_kwargs = processor._merge_kwargs(
        Qwen2_5_VLProcessorKwargs,
        tokenizer_init_kwargs=processor.tokenizer.init_kwargs,
    )
    return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
    codec_labels = list(zip(compression_types, compression_levels))

    compressed_image_outputs = processor.image_processor(images=compressed_image_inputs, **output_kwargs["images_kwargs"])
    original_images_outputs = processor.image_processor(images=original_image_inputs, **output_kwargs["images_kwargs"])


    compressed_image_outputs = BatchFeature(data={**compressed_image_outputs}, tensor_type=return_tensors)
    original_images_outputs = BatchFeature(data={**original_images_outputs}, tensor_type=return_tensors)
    compressed_pixel_values = compressed_image_outputs["pixel_values"]
    compressed_image_grid_thw = compressed_image_outputs["image_grid_thw"]
    original_pixel_values = original_images_outputs["pixel_values"]
    original_image_grid_thw = original_images_outputs["image_grid_thw"]

    return compressed_pixel_values, compressed_image_grid_thw, original_pixel_values, original_image_grid_thw, codec_labels


class QwenEncoderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vision_config_path = "/data11/zhangzf2505/Qwen/Qwen2.5-VL-3B-Instruct/config.json"
        with open(vision_config_path, 'r') as f:
            config = json.load(f)
            config['vision_config']['dtype'] = 'bfloat16'
            vision_config = config['vision_config']
            config = Qwen2_5_VLConfig(vision_config=vision_config)
        vit_standard_encoder = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config, dtype=torch.bfloat16)
        state_dict = torch.load("/data11/zhangzf2505/Finetune-Qwen/weight/qwen2_5VL3B_vit_encoder_state_dict.pth", weights_only=False)
        vit_standard_encoder.load_state_dict(state_dict, strict=True)
        self.standard_model = vit_standard_encoder.eval()
        for param in self.standard_model.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()
    def forward(self, compressed_outputs, original_hidden_states, original_image_grid_thw):
        with torch.no_grad():
            original_outputs = self.standard_model(
                hidden_states=original_hidden_states,
                grid_thw=original_image_grid_thw
            )
        loss = self.criterion(compressed_outputs, original_outputs)
        return loss

class Qwen2_5_Trainer(L.LightningModule):
    def __init__(self, config, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        vision_config_path = "/data11/zhangzf2505/Qwen/Qwen2.5-VL-3B-Instruct/config.json"
        with open(vision_config_path, 'r') as f:
            config = json.load(f)
            config['vision_config']['dtype'] = 'bfloat16'
            vision_config = config['vision_config']
            config = Qwen2_5_VLConfig(vision_config=vision_config)
        vit_encoder = Qwen2_5_VisionTransformerPretrainedModelCodec._from_config(config.vision_config, dtype=torch.bfloat16)
        # vit_encoder = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config, dtype=torch.bfloat16)
        state_dict = torch.load("/data11/zhangzf2505/Finetune-Qwen/weight/qwen2_5VL3B_vit_encoder_state_dict.pth", weights_only=False)
        vit_encoder.load_state_dict(state_dict, strict=False)
        self.model = vit_encoder
        self.train_dataset = CompressedDatasetWithAnno(
            annotation_file = '/data11/zhangzf2505/COCODataset/COCO_ELIC.txt'
            # annotation_file = '/data11/zhangzf2505/COCODataset/KODAK_JPEG.txt'
        )
        self.valid_dataset = CompressedDatasetWithAnno(
            annotation_file = '/data11/zhangzf2505/COCODataset/KODAK_ELIC.txt'
        )
        self.criterion = QwenEncoderLoss()
        
    def training_step(self, batch, batch_idx):
        compressed_hidden_states, compressed_image_grid_thw, original_hidden_states, original_image_grid_thw, codec_labels = batch
        compressed_outputs = self.model(
            hidden_states=compressed_hidden_states,
            grid_thw=compressed_image_grid_thw,
            codec_label= codec_labels
        )
       
        loss = self.criterion(compressed_outputs, original_hidden_states, original_image_grid_thw)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        compressed_hidden_states, compressed_image_grid_thw, original_hidden_states, original_image_grid_thw, codec_labels = batch
        
        compressed_outputs = self.model(
            hidden_states=compressed_hidden_states,
            grid_thw=compressed_image_grid_thw,
            codec_label= codec_labels
        )
       
        loss = self.criterion(compressed_outputs, original_hidden_states, original_image_grid_thw)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=train_collate_fn,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=1,
            collate_fn=train_collate_fn,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
    
if __name__ == "__main__":
    config = {
        "max_epochs": 100,
        "lr": 1e-4,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "qwen2.5-3b-instruct-ft-Rebuttal-ELIC4points"
    }
    L.seed_everything(42, workers=True)
    # 初始化 WandbLogger
    wandb_logger = WandbLogger(
        project="FintuneQwen-Rebuttal",  # 替换为你的项目名
        name="ELIC",         # 可选：每次运行的名称
        log_model=True                # 是否记录模型 checkpoint
    )
    trainer = L.Trainer(
        accelerator="cuda",
        strategy="ddp",
        # devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        limit_val_batches=1,
        num_sanity_val_steps=0,
        log_every_n_steps=20,
        callbacks=[SaveCheckpoint(result_path=config["result_path"])],
        # precision='16-mixed',
        sync_batchnorm=True,
        logger=wandb_logger
    )
    model_module = Qwen2_5_Trainer(config, processor)
    trainer.fit(model_module)
        





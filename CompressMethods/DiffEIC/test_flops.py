from typing import List, Tuple, Optional
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from calflops import calculate_flops
from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.ddim_sampler import DDIMSampler
from model.diffeic import DiffEIC
from utils.image import pad
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from time import time
import torch.nn as nn
from thop import profile, clever_format

@torch.no_grad()
def process(
    model: DiffEIC,
    imgs,
    sampler: str,
    steps: int,
    stream_path: str
) -> Tuple[List[np.ndarray], float]:
    """
    Apply DiffEIC model on a list of images.
    
    Args:
        model (DiffEIC): Model.
        imgs (List[np.ndarray]): A list of images (HWC, RGB, range in [0, 255])
        sampler (str): Sampler name.
        steps (int): Sampling steps.
        stream_path (str): Savedir of bitstream
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        bpp
    """
    # breakpoint()
    n_samples = 1
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    # control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(imgs, "n h w c -> n c h w").contiguous()
    
    height, width = control.size(-2), control.size(-1)
    start = time()
    bpp = model.apply_condition_compress(control, stream_path, height, width)
    end = time()
    print(f"Compression Enc time: {end - start:.2f} seconds")
    cond = {
        "c_latent": [model.apply_condition_decompress(stream_path)],
        "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]
    }
    start = time()
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            cond_fn=None, x_T=x_T
        )
    else:
        sampler: DDIMSampler
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            x_T=x_T, eta=0
        )
    
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    end = time()
    print(f"Decoding time: {end - start:.2f} seconds")
    
    preds = [x_samples[i] for i in range(n_samples)]
    breakpoint()
    
    return preds, bpp


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt_sd", default='./weight/v2-1_512-ema-pruned.ckpt', type=str, help="checkpoint path of stable diffusion")
    parser.add_argument("--ckpt_lc", default='./weight/1_2_1/lc.ckpt', type=str, help="checkpoint path of lfgcm and control module")
    parser.add_argument("--config", default='configs/model/diffeic.yaml', type=str, help="model config path")
    
    parser.add_argument("--input", type=str, default='path to input images')
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=50, type=int)
    
    parser.add_argument("--output", type=str, default='results/')
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    return parser.parse_args()


class DiffEIC_MY(nn.Module):
    def __init__(self):
        super().__init__()
        args = parse_args()
        self.args = args
        pl.seed_everything(args.seed)
        
        if args.device == "cpu":
            disable_xformers()

        model: DiffEIC = instantiate_from_config(OmegaConf.load(args.config))
        self.model = model
        ckpt_sd = torch.load(args.ckpt_sd, map_location="cpu")['state_dict']
        ckpt_lc = torch.load(args.ckpt_lc, map_location="cpu")['state_dict']
        ckpt_sd.update(ckpt_lc)
        load_state_dict(model, ckpt_sd, strict=False)
        # update preprocess model
        self.model.preprocess_model.update(force=True)
        self.model.freeze()
        self.model.to(args.device)
        bpps = []
        # assert os.path.isdir(args.input)
        print(f"sampling {args.steps} steps using {args.sampler} sampler")

    def forward(self, input) -> None:
            
        preds, bpp = process(
            self.model, input, steps=self.args.steps, sampler=self.args.sampler,
            stream_path='./test_flops_stream.bin'
        )
        pred = preds[0]

            

            

if __name__ == "__main__":
    net = DiffEIC_MY()
    # model = Cheng()
    # model = Mbt()
    net.eval()
    net.cuda()

    # # # Calculate FLOPs
    # flops = calculate_flops(net, input_shape=(1, 512, 384, 3), output_as_string=True, output_precision=4)
    # print(f"FLOPs: {flops}")

    # Parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # FLOPs
    dummy_input = torch.randn(1, 512, 768, 3).cuda()
    with torch.no_grad():
        macs, _ = profile(net, inputs=(dummy_input,), verbose=False)

    macs_str, params_str = clever_format([macs, total_params], "%.3f")

    print(f"Total Parameters:     {params_str} ({total_params})")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"MACs:                 {macs_str}")
    print(f"FLOPs (≈2×MACs):     {clever_format([macs * 2], '%.3f')[0]}")


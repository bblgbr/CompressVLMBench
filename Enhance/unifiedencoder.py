import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Any, Callable, Optional, Union
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel, Qwen2_5_VLTextModel, Qwen2_5_VLModelOutputWithPast
from transformers.utils import auto_docstring, TransformersKwargs, is_torchdynamo_compiling
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from diffusers.models.embeddings import TimestepEmbedding

def one_hot_encode(labels, num_points):
    # label example: ['JPEG', 1]
    codec_dict = {'JPEG': 0, 'ELIC': 1, 'ILLM': 2}
    index = []
    for label in labels:
        index.append(int(codec_dict[label[0]] * (num_points // len(codec_dict)) + label[1]))
    return F.one_hot(torch.tensor(index), num_classes=num_points)


class Qwen2_5_VisionTransformerPretrainedModelCodec(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        codec_types = 3
        rd_points = 4
        self.codec_classes = codec_types * rd_points
        codec_ebmed_dim = 40

        # 这里注意一下, 看一下embed_dim到底应该多少
        self.codec_emb = TimestepEmbedding(
            self.codec_classes, 
            codec_ebmed_dim,
        )

        # 将 linear_2 的权重和偏置初始化为 0
        torch.nn.init.constant_(self.codec_emb.linear_2.weight, 0.0)
        if self.codec_emb.linear_2.bias is not None:
            torch.nn.init.constant_(self.codec_emb.linear_2.bias, 0.0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, codec_label, **kwargs): 
        
        codec_info = one_hot_encode(codec_label, self.codec_classes).to(hidden_states.device).to(self.dtype)
        codec_bitlevl_emb = self.codec_emb(codec_info)
        # codec_bitlevl_emb = codec_bitlevl_emb.repeat_interleave(grid_thw[0][1] * grid_thw[0][2], dim=0)
        codec_bitlevl_emb_list = []

        for i in range(grid_thw.size(0)):  # Iterate over the grid_thw
            th, w, h = grid_thw[i]  # Extract th, w, h values for each entry
            codec_bitlevl_emb_list.append(codec_bitlevl_emb[i].unsqueeze(0).repeat(th * w * h, 1))
        codec_bitlevl_emb = torch.cat(codec_bitlevl_emb_list, dim=0)
    
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # add codec embedding
        rotary_pos_emb = rotary_pos_emb + codec_bitlevl_emb

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states
    


class Qwen2_5_VLModelCodec(Qwen2_5_VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModelCodec._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here
        self.codec_label = (config.compress_type, config.compress_level)
        # Initialize weights and apply final processing
        self.post_init()

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None) -> Any:
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw, codec_label=self.codec_label)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds



class Qwen2_5_VLForConditionalGenerationCodec(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModelCodec(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

if __name__ == "__main__":
    model = Qwen2_5_VLForConditionalGenerationCodec.from_pretrained("/data11/zhangzf2505/Finetune-Qwen/qwen2_5VL3B-Adapter")
    print("done")
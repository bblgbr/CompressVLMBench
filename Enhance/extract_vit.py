import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

def extract_vit_encoder():
    MODEL_ID = "/data11/zhangzf2505/Qwen/Qwen2.5-VL-7B-Instruct"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto")


    torch.save(model.model.visual.state_dict(), "qwen2_5VL7B_vit_encoder_state_dict.pth")


extract_vit_encoder()
vision_config_path = "/data11/zhangzf2505/Qwen/Qwen2.5-VL-7B-Instruct/config.json"
with open(vision_config_path, 'r') as f:
    config = json.load(f)
    config['vision_config']['dtype'] = 'bfloat16'
    vision_config = config['vision_config']
    config = Qwen2_5_VLConfig(vision_config=vision_config)
vit_encoder = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config, dtype=torch.bfloat16)
state_dict = torch.load("/data11/zhangzf2505/Finetune-Qwen/qwen2_5VL7B_vit_encoder_state_dict.pth", weights_only=False)
vit_encoder.load_state_dict(state_dict, strict=True)

## 微调vit_encoder

print("ViT encoder loaded successfully.")



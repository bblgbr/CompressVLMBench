import os
import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from unifiedencoder import Qwen2_5_VLForConditionalGenerationCodec



def copy_model(target_path):
    root_path = "/mnt/workspace/zhangzf/ICML2026/Qwen2.5-VL-3B-Instruct"
    # copy json to target_path
    os.system(f"cp {root_path}/chat_template.json {target_path}/chat_template.json")
    os.system(f"cp {root_path}/preprocessor_config.json {target_path}/preprocessor_config.json")
    os.system(f"cp {root_path}/tokenizer.json {target_path}/tokenizer.json")
    os.system(f"cp {root_path}/tokenizer_config.json {target_path}/tokenizer_config.json")
    os.system(f"cp {root_path}/vocab.json {target_path}/vocab.json")
    os.system(f"cp {root_path}/merges.txt {target_path}/merges.txt")

def save_codec_model(save_path):
    MODEL_ID = "/mnt/workspace/zhangzf/ICML2026/Qwen2.5-VL-3B-Instruct"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGenerationCodec.from_pretrained(
        MODEL_ID,
        device_map="auto", dtype=torch.bfloat16)
    state_dict = torch.load("/mnt/workspace/zhangzf/ICML2026/Finetune-Qwen/qwen2.5-3b-instruct-ft/12/pytorch_model.bin", weights_only=False)
    model.model.visual.load_state_dict(state_dict, strict=True)
    model.save_pretrained(save_path, type)



if __name__ == "__main__":
    save_path = "/mnt/workspace/zhangzf/ICML2026/Finetune-Qwen-ali/qwen2_5VL3B-Adapter-epoch12"
    save_codec_model(save_path)
    copy_model(save_path)
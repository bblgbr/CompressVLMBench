import torch

def load_state_dict(path):
    obj = torch.load(path, map_location="cpu")
    # 如果是包装结构，提取 state_dict
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    elif isinstance(obj, dict):
        return obj
    else:
        raise TypeError(f"{path} 不包含有效的 state_dict")

def check_structure_and_dtype(path1, path2):
    sd1 = load_state_dict(path1)
    sd2 = load_state_dict(path2)

    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    # 检查参数名称是否一致
    if keys1 != keys2:
        print("❌ 参数名称不一致")
        print("仅在第一个文件中存在的参数:", keys1 - keys2)
        print("仅在第二个文件中存在的参数:", keys2 - keys1)
        return False

    all_match = True
    for key in sorted(keys1):
        param1 = sd1[key]
        param2 = sd2[key]

        shape_match = param1.shape == param2.shape
        dtype_match = param1.dtype == param2.dtype

        if not shape_match or not dtype_match:
            print(f"⚠️ 参数 {key} 不一致:")
            if not shape_match:
                print(f"  - 形状不同: {param1.shape} vs {param2.shape}")
            if not dtype_match:
                print(f"  - 类型不同: {param1.dtype} vs {param2.dtype}")
            all_match = False

    if all_match:
        print("✅ 两个 .pth 文件结构和参数类型完全一致")
    else:
        print("❌ 存在不一致项")

    return all_match

# 示例用法
check_structure_and_dtype("/data11/zhangzf2505/Finetune-Qwen/qwen2.5-3b-instruct-ft/1/pytorch_model.bin", "/data11/zhangzf2505/Finetune-Qwen/qwen2_5VL3B_vit_encoder_state_dict.pth")

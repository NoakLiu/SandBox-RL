#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置LoRA目录结构
"""

import os
import json
import shutil
from pathlib import Path

# 配置
CPFS_BASE = "/cpfs04/shared/kilab/liudong"

def create_mock_lora_checkpoint(lora_id: int, output_dir: str):
    """创建模拟的LoRA checkpoint"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建adapter_config.json
    config = {
        "base_model_name_or_path": "qwen-2",
        "bias": "none",
        "enable_lora": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": 16.0,
        "lora_dropout": 0.1,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM"
    }
    
    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 创建模拟的adapter_model.bin（实际应该是真实的权重文件）
    with open(output_path / "adapter_model.bin", "wb") as f:
        f.write(b"mock_lora_weights_for_demo")
    
    print(f"✅ 创建LoRA {lora_id}: {output_path}")

def setup_lora_directories():
    """设置LoRA目录结构"""
    print(f"🔧 设置LoRA目录结构: {CPFS_BASE}")
    
    # 创建基础目录
    base_path = Path(CPFS_BASE)
    base_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ 创建基础目录: {base_path}")
    
    # 为每个LoRA创建目录和checkpoint
    for i in range(1, 9):
        lora_dir = base_path / f"lora{i}"
        create_mock_lora_checkpoint(i, str(lora_dir))
    
    print(f"\n🎉 LoRA目录设置完成！")
    print(f"📁 目录结构:")
    for i in range(1, 9):
        lora_dir = base_path / f"lora{i}"
        print(f"   lora{i}: {lora_dir}")
        if (lora_dir / "adapter_config.json").exists():
            print(f"     ✅ adapter_config.json")
        if (lora_dir / "adapter_model.bin").exists():
            print(f"     ✅ adapter_model.bin")

if __name__ == "__main__":
    setup_lora_directories()

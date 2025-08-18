#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Llama-2模型到本地的Python脚本
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, login
import argparse


def download_llama_model(model_name="meta-llama/Llama-2-7b-hf", 
                        local_dir="/cpfs04/shared/kilab/hf-hub/Llama-2-7b-hf",
                        token=None):
    """
    下载Llama-2模型到本地
    
    Args:
        model_name: Hugging Face模型名称
        local_dir: 本地保存目录
        token: Hugging Face token（如果需要）
    """
    
    print(f"🚀 开始下载模型: {model_name}")
    print(f"📁 保存到: {local_dir}")
    
    # 创建本地目录
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        # 如果提供了token，先登录
        if token:
            print("🔐 使用token登录Hugging Face...")
            login(token=token)
        
        # 下载模型
        print("📥 正在下载模型文件...")
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=True,  # 支持断点续传
            max_workers=4  # 并发下载数量
        )
        
        print(f"✅ 模型下载完成!")
        print(f"📂 模型路径: {model_path}")
        
        # 检查下载的文件
        model_files = list(Path(model_path).glob("*"))
        print(f"📊 下载的文件数量: {len(model_files)}")
        
        # 显示主要文件
        print("\n📋 主要文件:")
        for file_path in model_files:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name}: {size_mb:.1f} MB")
        
        return model_path
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确认Hugging Face访问权限")
        print("3. 检查磁盘空间")
        print("4. 尝试使用token登录")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载Llama-2模型到本地")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf", 
                       help="模型名称 (默认: meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--local-dir", default="/cpfs04/shared/kilab/hf-hub/Llama-2-7b-hf",
                       help="本地保存目录")
    parser.add_argument("--token", help="Hugging Face token")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 Llama-2模型下载工具")
    print("=" * 60)
    
    # 检查磁盘空间
    local_path = Path(args.local_dir)
    if local_path.exists():
        free_space = os.statvfs(local_path).f_frsize * os.statvfs(local_path).f_bavail
        free_space_gb = free_space / (1024**3)
        print(f"💾 可用磁盘空间: {free_space_gb:.1f} GB")
        
        if free_space_gb < 15:
            print("⚠️  警告: 磁盘空间可能不足，Llama-2-7b需要约13GB空间")
    
    # 下载模型
    model_path = download_llama_model(
        model_name=args.model,
        local_dir=args.local_dir,
        token=args.token
    )
    
    if model_path:
        print("\n🎉 下载成功!")
        print(f"📝 在您的代码中使用以下路径:")
        print(f"   model_path = '{model_path}'")
        
        # 创建使用示例
        example_code = f'''
# 使用示例
from vllm import LLM, SamplingParams

llm = LLM(
    model="{model_path}",
    enable_lora=True,
    max_lora_rank=64,
    max_loras=2,
    tensor_parallel_size=4
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256
)

outputs = llm.generate(["Hello, how are you?"], sampling_params)
print(outputs[0].outputs[0].text)
'''
        
        print("\n💻 使用示例:")
        print(example_code)
    else:
        print("\n❌ 下载失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()

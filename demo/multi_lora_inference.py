# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use the multi-LoRA functionality
for offline inference with Qwen2.5-7B-Instruct model.
"""

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    """Main function that sets up and runs the prompt processing."""
    
    # 1. 下载LoRA适配器
    print("正在下载LoRA适配器...")
    sql_lora_path = snapshot_download(repo_id="ngxson/LoRA-Human-Like-Qwen2.5-7B-Instruct")
    print(f"SQL LoRA下载完成: {sql_lora_path}")
    
    # 2. 实例化基础模型，启用LoRA
    print("初始化vLLM模型...")
    llm = LLM(
        model="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct", 
        enable_lora=True,
        tensor_parallel_size=4  # 使用4个GPU，因为Qwen2.5-7B有28个注意力头
    )
    
    # 3. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        stop=["[/assistant]"]
    )
    
    # 4. 准备提示
    prompts = [
        "你好，请介绍一下自己。",
        "请用友好的语气问候一下。",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",
        "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_11 (nationality VARCHAR, elector VARCHAR)\n\n question: When Anchero Pantaleone was the elector what is under nationality? [/user] [assistant]",
    ]
    
    # 5. 生成输出 - 前两个使用基础模型，后两个使用LoRA
    print("\n" + "="*50)
    print("基础模型输出:")
    print("="*50)
    
    # 基础模型请求（不使用LoRA）
    outputs = llm.generate(prompts[:2], sampling_params)
    for i, output in enumerate(outputs):
        print(f"Prompt {i+1}: {prompts[i]}")
        print(f"Output: {output.outputs[0].text}")
        print("-" * 30)
    
    print("\n" + "="*50)
    print("LoRA模型输出:")
    print("="*50)
    
    # LoRA请求
    outputs = llm.generate(
        prompts[2:], 
        sampling_params,
        lora_request=LoRARequest("sql_adapter", 1, sql_lora_path)
    )
    
    for i, output in enumerate(outputs):
        print(f"Prompt {i+3}: {prompts[i+2]}")
        print(f"Output: {output.outputs[0].text}")
        print("-" * 30)
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
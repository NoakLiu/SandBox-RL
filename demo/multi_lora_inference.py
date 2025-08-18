# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use the multi-LoRA functionality
for offline inference with Qwen2.5-7B-Instruct model.
"""

from typing import Optional
import os

from huggingface_hub import snapshot_download
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
    lora_paths: dict,
) -> list[tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 4 requests for the LoRA. We define 2
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=1`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    return [
        (
            "你好，请介绍一下自己。",
            SamplingParams(
                temperature=0.0, logprobs=1, prompt_logprobs=1, max_tokens=128
            ),
            None,
        ),
        (
            "请用友好的语气问候一下。",
            SamplingParams(
                temperature=0.8, top_k=5, presence_penalty=0.2, max_tokens=128
            ),
            None,
        ),
        (
            "请用专业的语气回答问题：什么是人工智能？",
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
            ),
            LoRARequest("sql-lora", 1, lora_paths.get("sql", "")),
        ),
        (
            "请用幽默的方式表达：今天天气怎么样？",
            SamplingParams(
                temperature=0.0,
                logprobs=1,
                prompt_logprobs=1,
                max_tokens=128,
            ),
            LoRARequest("code-lora", 2, lora_paths.get("code", "")),
        ),
    ]


def process_requests(
    engine: LLMEngine,
    test_prompts: list[tuple[str, SamplingParams, Optional[LoRARequest]]],
):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    print("-" * 50)
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(
                str(request_id), prompt, sampling_params, lora_request=lora_request
            )
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
                print("-" * 50)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(
        model="/cpfs04/shared/kilab/hf-hub/Qwen2.5-7B-Instruct",
        enable_lora=True,
        max_loras=2,
        max_lora_rank=8,
        max_cpu_loras=2,
        max_num_seqs=256,
        tensor_parallel_size=4,  # 使用4个GPU，因为Qwen2.5-7B有28个注意力头
    )
    return LLMEngine.from_engine_args(engine_args)


def download_loras() -> dict:
    """下载LoRA模型"""
    print("正在下载LoRA模型...")
    
    lora_paths = {}
    
    try:
        # 下载SQL LoRA
        print("下载SQL LoRA...")
        sql_lora_path = snapshot_download(
            repo_id="yard1/llama-2-7b-sql-lora-test",
            cache_dir="/cpfs04/shared/kilab/liudong/loras"
        )
        lora_paths["sql"] = sql_lora_path
        print(f"SQL LoRA下载完成: {sql_lora_path}")
    except Exception as e:
        print(f"SQL LoRA下载失败: {e}")
        # 使用备用LoRA
        try:
            sql_lora_path = snapshot_download(
                repo_id="microsoft/DialoGPT-medium",
                cache_dir="/cpfs04/shared/kilab/liudong/loras/sql"
            )
            lora_paths["sql"] = sql_lora_path
            print(f"备用SQL LoRA下载完成: {sql_lora_path}")
        except Exception as e2:
            print(f"备用SQL LoRA也失败: {e2}")
    
    try:
        # 下载代码LoRA
        print("下载代码LoRA...")
        code_lora_path = snapshot_download(
            repo_id="microsoft/DialoGPT-small",
            cache_dir="/cpfs04/shared/kilab/liudong/loras/code"
        )
        lora_paths["code"] = code_lora_path
        print(f"代码LoRA下载完成: {code_lora_path}")
    except Exception as e:
        print(f"代码LoRA下载失败: {e}")
        # 使用备用LoRA
        try:
            code_lora_path = snapshot_download(
                repo_id="microsoft/DialoGPT-medium",
                cache_dir="/cpfs04/shared/kilab/liudong/loras/code"
            )
            lora_paths["code"] = code_lora_path
            print(f"备用代码LoRA下载完成: {code_lora_path}")
        except Exception as e2:
            print(f"备用代码LoRA也失败: {e2}")
    
    return lora_paths


def main():
    """Main function that sets up and runs the prompt processing."""
    # 下载LoRA模型
    lora_paths = download_loras()
    
    if not lora_paths:
        print("❌ 没有可用的LoRA模型，退出")
        return
    
    print(f"✅ 可用的LoRA模型: {lora_paths}")
    
    # 初始化引擎
    engine = initialize_engine()
    
    # 创建测试提示
    test_prompts = create_test_prompts(lora_paths)
    
    # 处理请求
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    main()
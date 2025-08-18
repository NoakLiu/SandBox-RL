#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的LoRA测试脚本 - 根据vLLM官方文档
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Any

# 配置
BASE_URL = "http://127.0.0.1:8001"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_health():
    """测试健康检查"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    logger.info("✅ vLLM实例健康")
                    return True
                else:
                    logger.error(f"❌ vLLM实例异常: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 无法连接到vLLM: {e}")
            return False


async def test_models():
    """测试模型列表"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    logger.info(f"✅ 可用模型: {len(models)} 个")
                    
                    for model in models:
                        model_id = model.get("id", "unknown")
                        logger.info(f"   - {model_id}")
                    
                    # 检查是否有LoRA模型
                    lora_models = [m for m in models if "lora" in m.get("id", "").lower()]
                    if lora_models:
                        logger.info(f"✅ 找到 {len(lora_models)} 个LoRA模型")
                        return True
                    else:
                        logger.warning("⚠️ 没有找到LoRA模型")
                        return False
                else:
                    logger.error(f"❌ 获取模型列表失败: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ 测试模型列表失败: {e}")
            return False


async def test_chat_with_lora(lora_name: str, prompt: str) -> str:
    """测试带LoRA的聊天（使用正确的模型名称）"""
    url = f"{BASE_URL}/v1/chat/completions"
    
    # 直接使用LoRA模型名称
    payload = {
        "model": lora_name,  # 直接使用LoRA模型名称
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logger.info(f"✅ LoRA {lora_name} 响应: {content[:100]}...")
                    return content
                else:
                    error_text = await response.text()
                    logger.error(f"❌ LoRA {lora_name} 请求失败: {response.status} - {error_text}")
                    return f"错误: {response.status}"
        except Exception as e:
            logger.error(f"❌ LoRA {lora_name} 请求异常: {e}")
            return f"异常: {e}"


async def test_concurrent_requests():
    """测试并发请求"""
    logger.info("🔄 测试并发LoRA请求...")
    
    # 准备8个不同的请求
    requests_data = [
        ("lora1", "请用简短的话介绍一下自己。"),
        ("lora2", "请用友好的语气问候一下。"),
        ("lora3", "请用专业的语气回答问题。"),
        ("lora4", "请用幽默的方式表达。"),
        ("lora5", "请用正式的语气说话。"),
        ("lora6", "请用轻松的语气聊天。"),
        ("lora7", "请用严谨的态度分析。"),
        ("lora8", "请用温暖的语言表达。"),
    ]
    
    start_time = time.time()
    
    # 并发执行
    tasks = [
        test_chat_with_lora(lora_name, prompt) 
        for lora_name, prompt in requests_data
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"⏱️ 并发请求完成，耗时: {duration:.2f}秒")
    
    # 统计结果
    success_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"❌ 请求 {i+1} 失败: {result}")
        else:
            success_count += 1
            logger.info(f"✅ 请求 {i+1} 成功")
    
    logger.info(f"📊 并发测试结果: {success_count}/{len(results)} 成功")
    return success_count == len(results)


async def main():
    """主测试函数"""
    logger.info("🧪 开始正确的LoRA测试...")
    
    # 1. 健康检查
    if not await test_health():
        logger.error("❌ 健康检查失败，请确保vLLM实例正在运行")
        return
    
    # 2. 检查模型列表
    if not await test_models():
        logger.error("❌ 模型列表检查失败")
        return
    
    # 3. 单个LoRA测试
    logger.info("🔍 测试单个LoRA...")
    await test_chat_with_lora("lora1", "你好，请介绍一下自己。")
    
    # 4. 并发测试
    await test_concurrent_requests()
    
    logger.info("🎉 测试完成！")


if __name__ == "__main__":
    asyncio.run(main())

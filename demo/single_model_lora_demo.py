#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型+8GPU LoRA完整演示

演示功能：
1. 启动单模型+8GPU vLLM实例
2. 发布8个LoRA权重
3. 自动热更新
4. 并发测试
"""

import asyncio
import time
import json
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SingleModelLoRADemo:
    """单模型+8GPU LoRA演示"""
    
    def __init__(self):
        self.vllm_process = None
        self.updater_process = None
        self.publisher_process = None
        
    def start_vllm(self):
        """启动vLLM实例"""
        logger.info("🚀 启动单模型+8GPU vLLM实例...")
        
        # 使用启动脚本
        script_path = "demo/launch_single_model_8gpu.sh"
        if not os.path.exists(script_path):
            logger.error(f"❌ 启动脚本不存在: {script_path}")
            return False
        
        try:
            # 给脚本执行权限
            os.chmod(script_path, 0o755)
            
            # 启动vLLM
            self.vllm_process = subprocess.Popen(
                [f"./{script_path}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info("✅ vLLM启动命令已执行")
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动vLLM失败: {e}")
            return False
    
    def start_lora_updater(self):
        """启动LoRA热更新管理器"""
        logger.info("🔄 启动LoRA热更新管理器...")
        
        try:
            self.updater_process = subprocess.Popen(
                [sys.executable, "demo/lora_updater_single.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            logger.info("✅ LoRA热更新管理器已启动")
            return True
            
        except Exception as e:
            logger.error(f"❌ 启动LoRA热更新管理器失败: {e}")
            return False
    
    def publish_loras(self):
        """发布LoRA权重"""
        logger.info("📤 发布LoRA权重...")
        
        try:
            result = subprocess.run(
                [sys.executable, "demo/lora_publisher.py"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info("✅ LoRA权重发布成功")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ LoRA权重发布失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 发布LoRA权重异常: {e}")
            return False
    
    async def test_concurrent_requests(self):
        """测试并发请求"""
        logger.info("🧪 测试并发LoRA请求...")
        
        # 导入测试函数
        try:
            from demo.test_single_model_lora import test_concurrent_requests
            success = await test_concurrent_requests()
            if success:
                logger.info("✅ 并发测试成功")
            else:
                logger.warning("⚠️ 并发测试部分失败")
            return success
        except Exception as e:
            logger.error(f"❌ 并发测试失败: {e}")
            return False
    
    def check_vllm_status(self):
        """检查vLLM状态"""
        try:
            import requests
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ vLLM实例运行正常")
                return True
            else:
                logger.warning(f"⚠️ vLLM实例异常: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ 无法连接到vLLM: {e}")
            return False
    
    def wait_for_vllm(self, timeout=120):
        """等待vLLM启动"""
        logger.info(f"⏳ 等待vLLM启动 (最多{timeout}秒)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_vllm_status():
                return True
            time.sleep(5)
        
        logger.error("❌ vLLM启动超时")
        return False
    
    async def run_demo(self):
        """运行完整演示"""
        logger.info("🎬 开始单模型+8GPU LoRA演示...")
        
        try:
            # 1. 启动vLLM
            if not self.start_vllm():
                logger.error("❌ 无法启动vLLM，演示终止")
                return
            
            # 2. 等待vLLM启动
            if not self.wait_for_vllm():
                logger.error("❌ vLLM启动失败，演示终止")
                return
            
            # 3. 启动LoRA热更新管理器
            if not self.start_lora_updater():
                logger.warning("⚠️ LoRA热更新管理器启动失败，继续演示")
            
            # 4. 等待一段时间让热更新管理器启动
            logger.info("⏳ 等待LoRA热更新管理器启动...")
            await asyncio.sleep(10)
            
            # 5. 发布LoRA权重
            if not self.publish_loras():
                logger.warning("⚠️ LoRA权重发布失败，继续演示")
            
            # 6. 等待热更新
            logger.info("⏳ 等待LoRA热更新...")
            await asyncio.sleep(15)
            
            # 7. 测试并发请求
            await self.test_concurrent_requests()
            
            # 8. 显示最终状态
            logger.info("📊 演示完成！")
            logger.info("💡 系统现在支持:")
            logger.info("   - 单模型+8GPU张量并行")
            logger.info("   - 8个LoRA独立热更新")
            logger.info("   - 并发请求处理")
            logger.info("   - RL策略集成")
            
        except KeyboardInterrupt:
            logger.info("🛑 用户中断演示")
        except Exception as e:
            logger.error(f"❌ 演示运行失败: {e}")
        finally:
            # 清理
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理资源...")
        
        if self.updater_process:
            self.updater_process.terminate()
            logger.info("✅ LoRA热更新管理器已停止")
        
        if self.vllm_process:
            self.vllm_process.terminate()
            logger.info("✅ vLLM实例已停止")


async def main():
    """主函数"""
    print("🎬 单模型+8GPU LoRA演示")
    print("="*60)
    print("这个演示将展示:")
    print("1. 单模型+8GPU张量并行")
    print("2. 8个LoRA独立热更新")
    print("3. 并发请求处理")
    print("4. RL策略集成")
    print("="*60)
    
    # 检查必要文件
    required_files = [
        "demo/launch_single_model_8gpu.sh",
        "demo/lora_updater_single.py",
        "demo/lora_publisher.py",
        "demo/test_single_model_lora.py"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            return
    
    print("✅ 所有必要文件存在")
    
    # 运行演示
    demo = SingleModelLoRADemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())

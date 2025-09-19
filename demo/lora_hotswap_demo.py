#!/usr/bin/env python3
"""
LoRA热更新演示脚本

演示功能：
1. 启动8GPU分布式LoRA调度器
2. 模拟RL策略更新LoRA权重
3. 实时监控LoRA热更新过程
4. 展示完整的LoRA生命周期管理
"""

import asyncio
import time
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox_rl.core.distributed_lora_scheduler import (
    create_distributed_lora_scheduler,
    LoRARLStrategy,
    LoRAUpdateEvent
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoRAHotSwapDemo:
    """LoRA热更新演示"""
    
    def __init__(self):
        self.scheduler = None
        self.rl_strategy = None
        self.demo_running = False
        
        # 演示配置
        self.cpfs_base = "/cpfs04/shared/kilab/lora_ckpts"
        self.base_port = 8001
        self.num_gpus = 8
        
        # 事件统计
        self.update_events = []
        self.failed_events = []
    
    async def setup(self):
        """设置演示环境"""
        logger.info("🚀 设置LoRA热更新演示环境...")
        
        # 创建调度器
        self.scheduler = create_distributed_lora_scheduler(
            base_port=self.base_port,
            num_gpus=self.num_gpus,
            cpfs_base=self.cpfs_base,
            poll_interval=3.0,  # 快速轮询用于演示
            enable_probe=True
        )
        
        # 创建RL策略
        self.rl_strategy = LoRARLStrategy(self.scheduler)
        
        # 设置事件回调
        self.scheduler.on_lora_updated = self._on_lora_updated
        self.scheduler.on_lora_failed = self._on_lora_failed
        
        # 创建CPFS目录结构
        await self._setup_cpfs_structure()
        
        logger.info("✅ 演示环境设置完成")
    
    async def _setup_cpfs_structure(self):
        """设置CPFS目录结构"""
        cpfs_path = Path(self.cpfs_base)
        cpfs_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个LoRA创建目录
        for i in range(1, self.num_gpus + 1):
            lora_dir = cpfs_path / f"lora{i}"
            lora_dir.mkdir(exist_ok=True)
            logger.info(f"创建LoRA目录: {lora_dir}")
    
    def _on_lora_updated(self, event: LoRAUpdateEvent):
        """LoRA更新成功回调"""
        self.update_events.append(event)
        logger.info(f"🎉 LoRA {event.lora_id} 热更新成功: {event.timestamp}")
        logger.info(f"   元数据: {event.metadata}")
    
    def _on_lora_failed(self, event: LoRAUpdateEvent):
        """LoRA更新失败回调"""
        self.failed_events.append(event)
        logger.error(f"❌ LoRA {event.lora_id} 热更新失败: {event.error_message}")
    
    async def start_demo(self):
        """启动演示"""
        logger.info("🎬 启动LoRA热更新演示...")
        
        # 启动调度器
        await self.scheduler.start()
        self.demo_running = True
        
        logger.info("✅ 演示已启动，开始模拟RL策略更新...")
    
    async def simulate_rl_updates(self):
        """模拟RL策略更新"""
        logger.info("🤖 开始模拟RL策略更新...")
        
        # 模拟8个LoRA的更新序列
        update_sequence = [
            # (lora_id, reward, delay)
            (1, 0.85, 2),
            (3, 0.92, 3),
            (5, 0.78, 2),
            (7, 0.95, 4),
            (2, 0.88, 3),
            (4, 0.91, 2),
            (6, 0.83, 3),
            (8, 0.89, 2),
        ]
        
        for lora_id, reward, delay in update_sequence:
            if not self.demo_running:
                break
            
            try:
                # 模拟生成新的权重
                new_weights = {
                    "rank": 8,
                    "alpha": 16.0,
                    "dropout": 0.1,
                    "learning_rate": 1e-4,
                    "target_modules": ["q_proj", "v_proj"]
                }
                
                # 创建模拟checkpoint目录
                checkpoint_dir = await self._create_mock_checkpoint(lora_id, new_weights)
                
                # 发布更新
                timestamp = await self.scheduler.publish_lora_update(
                    lora_id, checkpoint_dir, {
                        "reward": reward,
                        "training_step": len(self.update_events),
                        "weights_info": new_weights
                    }
                )
                
                logger.info(f"📤 发布LoRA {lora_id} 更新: reward={reward}, timestamp={timestamp}")
                
                # 等待一段时间
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"模拟更新LoRA {lora_id} 失败: {e}")
        
        logger.info("✅ RL策略更新模拟完成")
    
    async def _create_mock_checkpoint(self, lora_id: int, weights: Dict[str, Any]) -> str:
        """创建模拟checkpoint"""
        import tempfile
        import shutil
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix=f"lora_{lora_id}_")
        
        # 创建adapter_config.json
        config = {
            "base_model_name_or_path": "qwen-2",
            "bias": "none",
            "enable_lora": None,
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": weights["alpha"],
            "lora_dropout": weights["dropout"],
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": weights["rank"],
            "target_modules": weights["target_modules"],
            "task_type": "CAUSAL_LM"
        }
        
        with open(os.path.join(temp_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # 创建模拟的adapter_model.bin（实际应该是真实的权重文件）
        # 这里创建一个小的占位文件用于演示
        with open(os.path.join(temp_dir, "adapter_model.bin"), "wb") as f:
            f.write(b"mock_lora_weights_for_demo")
        
        return temp_dir
    
    async def monitor_status(self):
        """监控系统状态"""
        logger.info("📊 开始监控系统状态...")
        
        while self.demo_running:
            try:
                # 获取系统状态
                status = await self.scheduler.get_system_status()
                
                # 打印关键信息
                print("\n" + "="*60)
                print("📊 系统状态报告")
                print("="*60)
                
                # LoRA状态
                print("🔧 LoRA状态:")
                for lora_id in range(1, 9):
                    lora_key = f"lora{lora_id}"
                    if lora_key in status["lora_details"]:
                        lora_status = status["lora_details"][lora_key]
                        current = lora_status.get("current_version", "未加载")
                        ready = "✅" if lora_status.get("is_ready") else "❌"
                        print(f"   LoRA {lora_id}: {ready} {current}")
                
                # 更新历史
                print(f"\n📈 更新历史: {len(status['update_history'])} 次")
                for event in status["update_history"][-5:]:  # 最近5次
                    status_icon = "✅" if event["success"] else "❌"
                    print(f"   {status_icon} LoRA {event['lora_id']}: {event['timestamp']}")
                
                # 等待一段时间
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"监控状态失败: {e}")
                await asyncio.sleep(5)
    
    async def run_demo(self, duration: int = 60):
        """运行完整演示"""
        try:
            # 设置环境
            await self.setup()
            
            # 启动演示
            await self.start_demo()
            
            # 启动监控任务
            monitor_task = asyncio.create_task(self.monitor_status())
            
            # 启动RL更新模拟
            rl_task = asyncio.create_task(self.simulate_rl_updates())
            
            # 等待演示完成
            logger.info(f"⏰ 演示将运行 {duration} 秒...")
            await asyncio.sleep(duration)
            
            # 停止演示
            self.demo_running = False
            
            # 等待任务完成
            await asyncio.gather(monitor_task, rl_task, return_exceptions=True)
            
            # 显示最终统计
            await self._show_final_stats()
            
        except KeyboardInterrupt:
            logger.info("🛑 用户中断演示")
        except Exception as e:
            logger.error(f"演示运行失败: {e}")
        finally:
            # 清理
            if self.scheduler:
                await self.scheduler.shutdown()
    
    async def _show_final_stats(self):
        """显示最终统计"""
        print("\n" + "="*60)
        print("🎯 演示最终统计")
        print("="*60)
        
        print(f"📊 成功更新: {len(self.update_events)} 次")
        print(f"❌ 失败更新: {len(self.failed_events)} 次")
        
        if self.update_events:
            print("\n📈 成功更新详情:")
            for event in self.update_events:
                print(f"   LoRA {event.lora_id}: {event.timestamp} (reward: {event.metadata.get('reward', 'N/A')})")
        
        if self.failed_events:
            print("\n❌ 失败更新详情:")
            for event in self.failed_events:
                print(f"   LoRA {event.lora_id}: {event.error_message}")
        
        # 获取最终系统状态
        if self.scheduler:
            final_status = await self.scheduler.get_system_status()
            print(f"\n🔧 最终LoRA状态:")
            for lora_id in range(1, 9):
                lora_key = f"lora{lora_id}"
                if lora_key in final_status["lora_details"]:
                    lora_status = final_status["lora_details"][lora_key]
                    current = lora_status.get("current_version", "未加载")
                    ready = "✅" if lora_status.get("is_ready") else "❌"
                    print(f"   LoRA {lora_id}: {ready} {current}")


async def main():
    """主函数"""
    print("🎬 LoRA热更新演示")
    print("="*60)
    print("这个演示将展示:")
    print("1. 8GPU分布式LoRA调度器")
    print("2. RL策略动态更新LoRA权重")
    print("3. 实时热插拔过程")
    print("4. 完整的LoRA生命周期管理")
    print("="*60)
    
    # 检查vLLM实例是否运行
    print("🔍 检查vLLM实例状态...")
    import requests
    
    vllm_running = True
    for i in range(8):
        try:
            port = 8001 + i
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"   ✅ GPU {i} (端口 {port}): 运行中")
            else:
                print(f"   ❌ GPU {i} (端口 {port}): 异常")
                vllm_running = False
        except Exception:
            print(f"   ❌ GPU {i} (端口 {port}): 未响应")
            vllm_running = False
    
    if not vllm_running:
        print("\n⚠️  警告: 部分vLLM实例未运行")
        print("请先启动8GPU vLLM实例:")
        print("   ./demo/launch_8gpu_no_compile.sh")
        print("\n是否继续演示? (y/N): ", end="")
        
        response = input().strip().lower()
        if response != 'y':
            print("演示已取消")
            return
    
    # 运行演示
    demo = LoRAHotSwapDemo()
    await demo.run_demo(duration=120)  # 运行2分钟


if __name__ == "__main__":
    asyncio.run(main())

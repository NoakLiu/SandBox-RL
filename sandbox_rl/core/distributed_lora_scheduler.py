#!/usr/bin/env python3
"""
集成LoRA热更新的分布式调度器

功能：
1. 集成LoRA热更新管理器
2. 支持RL策略动态更新LoRA权重
3. 与分布式多模型调度器协同工作
4. 提供完整的LoRA生命周期管理
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path

from .distributed_multi_model_scheduler import (
    DistributedMultiModelScheduler, 
    DistributedVLLMClient,
    create_distributed_scheduler
)
from .lora_hotswap_manager import (
    LoRAHotSwapManager, 
    LoRAPublisher,
    create_lora_hotswap_manager,
    create_lora_publisher
)

logger = logging.getLogger(__name__)


@dataclass
class LoRAUpdateEvent:
    """LoRA更新事件"""
    lora_id: int
    timestamp: str
    success: bool
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class DistributedLoRAScheduler:
    """集成LoRA热更新的分布式调度器"""
    
    def __init__(self,
                 base_port: int = 8001,
                 num_gpus: int = 8,
                 model_name: str = "qwen-2",
                 cpfs_base: str = "/cpfs04/shared/kilab/lora_ckpts",
                 poll_interval: float = 5.0,
                 enable_probe: bool = True):
        
        # 基础调度器
        self.base_scheduler = create_distributed_scheduler(
            base_port=base_port,
            num_gpus=num_gpus,
            model_name=model_name
        )
        
        # LoRA热更新管理器
        self.lora_manager = create_lora_hotswap_manager(
            cpfs_base=cpfs_base,
            base_port=base_port,
            num_loras=num_gpus,
            poll_interval=poll_interval,
            enable_probe=enable_probe
        )
        
        # LoRA发布器
        self.lora_publisher = create_lora_publisher(cpfs_base)
        
        # 事件回调
        self.on_lora_updated: Optional[Callable[[LoRAUpdateEvent], None]] = None
        self.on_lora_failed: Optional[Callable[[LoRAUpdateEvent], None]] = None
        
        # 状态跟踪
        self.is_running = False
        self.update_history: List[LoRAUpdateEvent] = []
        
        logger.info(f"分布式LoRA调度器初始化完成: {num_gpus}个GPU, CPFS: {cpfs_base}")
    
    async def start(self):
        """启动调度器"""
        if self.is_running:
            logger.warning("调度器已在运行")
            return
        
        self.is_running = True
        
        # 设置LoRA更新回调
        self.lora_manager.on_lora_updated = self._on_lora_updated
        self.lora_manager.on_lora_failed = self._on_lora_failed
        
        # 启动LoRA热更新管理器
        await self.lora_manager.start()
        
        logger.info("分布式LoRA调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止LoRA热更新管理器
        await self.lora_manager.stop()
        
        logger.info("分布式LoRA调度器已停止")
    
    def _on_lora_updated(self, lora_id: int, version):
        """LoRA更新成功回调"""
        event = LoRAUpdateEvent(
            lora_id=lora_id,
            timestamp=version.timestamp,
            success=True,
            metadata=version.metadata
        )
        
        self.update_history.append(event)
        
        if self.on_lora_updated:
            self.on_lora_updated(event)
        
        logger.info(f"LoRA {lora_id} 更新成功: {version.timestamp}")
    
    def _on_lora_failed(self, lora_id: int, version):
        """LoRA更新失败回调"""
        event = LoRAUpdateEvent(
            lora_id=lora_id,
            timestamp=version.timestamp,
            success=False,
            metadata=version.metadata,
            error_message="热更新失败"
        )
        
        self.update_history.append(event)
        
        if self.on_lora_failed:
            self.on_lora_failed(event)
        
        logger.error(f"LoRA {lora_id} 更新失败: {version.timestamp}")
    
    async def publish_lora_update(self, lora_id: int, src_ckpt_dir: str, 
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        发布LoRA更新（供RL策略调用）
        
        Args:
            lora_id: LoRA ID (1-8)
            src_ckpt_dir: 源checkpoint目录
            metadata: 元数据信息
            
        Returns:
            发布的版本时间戳
        """
        try:
            timestamp = self.lora_publisher.publish_lora(lora_id, src_ckpt_dir, metadata)
            logger.info(f"发布LoRA {lora_id} 更新: {timestamp}")
            return timestamp
        except Exception as e:
            logger.error(f"发布LoRA {lora_id} 更新失败: {e}")
            raise
    
    async def get_lora_status(self, lora_id: int) -> Dict[str, Any]:
        """获取LoRA状态"""
        try:
            # 获取当前版本
            current_version = self.lora_manager.current_versions.get(lora_id)
            
            # 获取所有版本
            versions = self.lora_publisher.list_versions(lora_id)
            
            return {
                "lora_id": lora_id,
                "current_version": current_version,
                "available_versions": [v.timestamp for v in versions],
                "latest_version": versions[0].timestamp if versions else None,
                "is_ready": bool(current_version)
            }
        except Exception as e:
            logger.error(f"获取LoRA {lora_id} 状态失败: {e}")
            return {"lora_id": lora_id, "error": str(e)}
    
    async def rollback_lora(self, lora_id: int, target_timestamp: str) -> bool:
        """回滚LoRA到指定版本"""
        try:
            success = self.lora_publisher.rollback_to_version(lora_id, target_timestamp)
            if success:
                logger.info(f"回滚LoRA {lora_id} 到版本 {target_timestamp}")
            return success
        except Exception as e:
            logger.error(f"回滚LoRA {lora_id} 失败: {e}")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        base_status = self.base_scheduler.get_system_statistics()
        lora_status = self.lora_manager.get_status()
        
        # 获取所有LoRA的状态
        lora_details = {}
        for lora_id in range(1, 9):
            lora_details[f"lora{lora_id}"] = await self.get_lora_status(lora_id)
        
        return {
            "base_scheduler": base_status,
            "lora_manager": lora_status,
            "lora_details": lora_details,
            "update_history": [
                {
                    "lora_id": event.lora_id,
                    "timestamp": event.timestamp,
                    "success": event.success,
                    "error": event.error_message
                }
                for event in self.update_history[-10:]  # 最近10次更新
            ]
        }
    
    # 代理基础调度器的方法
    async def submit_task(self, task, selected_models=None):
        """提交任务（代理到基础调度器）"""
        return await self.base_scheduler.submit_task(task, selected_models)
    
    def register_model(self, model_id: str, gpu_id: int, initial_capabilities=None):
        """注册模型（代理到基础调度器）"""
        return self.base_scheduler.register_model(model_id, gpu_id, initial_capabilities)
    
    async def health_check_all(self):
        """健康检查（代理到基础调度器）"""
        return await self.base_scheduler.health_check_all()
    
    async def shutdown(self):
        """关闭调度器"""
        await self.stop()
        await self.base_scheduler.shutdown()


# 工厂函数
def create_distributed_lora_scheduler(
    base_port: int = 8001,
    num_gpus: int = 8,
    model_name: str = "qwen-2",
    cpfs_base: str = "/cpfs04/shared/kilab/lora_ckpts",
    poll_interval: float = 5.0,
    enable_probe: bool = True
) -> DistributedLoRAScheduler:
    """创建分布式LoRA调度器"""
    return DistributedLoRAScheduler(
        base_port=base_port,
        num_gpus=num_gpus,
        model_name=model_name,
        cpfs_base=cpfs_base,
        poll_interval=poll_interval,
        enable_probe=enable_probe
    )


# RL策略集成示例
class LoRARLStrategy:
    """LoRA RL策略集成示例"""
    
    def __init__(self, scheduler: DistributedLoRAScheduler):
        self.scheduler = scheduler
        self.training_history: List[Dict[str, Any]] = []
    
    async def update_lora_weights(self, lora_id: int, new_weights: Dict[str, Any], 
                                reward: float, metadata: Optional[Dict[str, Any]] = None):
        """
        根据RL策略更新LoRA权重
        
        Args:
            lora_id: LoRA ID
            new_weights: 新的权重数据
            reward: RL奖励
            metadata: 元数据
        """
        try:
            # 这里应该调用你的RL训练框架生成LoRA checkpoint
            # 示例：假设已经生成了checkpoint目录
            checkpoint_dir = f"/tmp/lora_{lora_id}_checkpoint"
            
            # 构建元数据
            update_metadata = {
                "reward": reward,
                "training_step": len(self.training_history),
                "timestamp": time.time(),
                "weights_info": {
                    "rank": new_weights.get("rank", 8),
                    "alpha": new_weights.get("alpha", 16.0),
                    "dropout": new_weights.get("dropout", 0.1)
                }
            }
            
            if metadata:
                update_metadata.update(metadata)
            
            # 发布更新
            timestamp = await self.scheduler.publish_lora_update(
                lora_id, checkpoint_dir, update_metadata
            )
            
            # 记录训练历史
            self.training_history.append({
                "lora_id": lora_id,
                "timestamp": timestamp,
                "reward": reward,
                "metadata": update_metadata
            })
            
            logger.info(f"RL策略更新LoRA {lora_id}: reward={reward}, timestamp={timestamp}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"RL策略更新LoRA {lora_id} 失败: {e}")
            raise
    
    async def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        return {
            "total_updates": len(self.training_history),
            "recent_updates": self.training_history[-10:],
            "reward_stats": {
                "min": min([h["reward"] for h in self.training_history]) if self.training_history else 0,
                "max": max([h["reward"] for h in self.training_history]) if self.training_history else 0,
                "avg": sum([h["reward"] for h in self.training_history]) / len(self.training_history) if self.training_history else 0
            }
        }

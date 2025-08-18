#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA发布器 - 供SandGraph RL策略使用

功能：
1. 发布新的LoRA权重到CPFS
2. 支持原子发布（先写文件，再写READY）
3. 版本化管理
4. 元数据记录
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

# 配置
CPFS_BASE = "/cpfs04/shared/kilab/liudong"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LoRAPublisher:
    """LoRA发布器"""
    
    def __init__(self, cpfs_base: str = CPFS_BASE):
        self.cpfs_base = Path(cpfs_base)
        self.cpfs_base.mkdir(parents=True, exist_ok=True)
        logger.info(f"LoRA发布器初始化: {self.cpfs_base}")
    
    def publish_lora(self, lora_id: int, src_ckpt_dir: str,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        发布新的LoRA权重
        
        Args:
            lora_id: LoRA ID (1-8)
            src_ckpt_dir: 源checkpoint目录
            metadata: 元数据信息
            
        Returns:
            发布的版本时间戳
        """
        # 创建目标目录
        lora_base = self.cpfs_base / f"lora{lora_id}"
        lora_base.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        dst_dir = lora_base / timestamp
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制LoRA文件
        src_path = Path(src_ckpt_dir)
        required_files = ["adapter_model.bin", "adapter_config.json"]
        
        for filename in required_files:
            src_file = src_path / filename
            dst_file = dst_dir / filename
            
            if not src_file.exists():
                raise FileNotFoundError(f"缺少必要文件: {filename}")
            
            shutil.copy2(src_file, dst_file)
            logger.debug(f"复制文件: {src_file} -> {dst_file}")
        
        # 写入元数据
        if metadata:
            with open(dst_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 创建READY标志（原子发布）
        (dst_dir / "READY").touch()
        
        logger.info(f"发布LoRA {lora_id} -> {dst_dir}")
        return timestamp
    
    def list_versions(self, lora_id: int) -> list:
        """列出LoRA的所有版本"""
        lora_base = self.cpfs_base / f"lora{lora_id}"
        if not lora_base.exists():
            return []
        
        versions = []
        for child in lora_base.iterdir():
            if child.is_dir() and (child / "READY").exists():
                metadata = {}
                metadata_file = child / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                
                versions.append({
                    "timestamp": child.name,
                    "path": str(child),
                    "metadata": metadata
                })
        
        versions.sort(key=lambda v: v["timestamp"], reverse=True)
        return versions
    
    def rollback_to_version(self, lora_id: int, target_timestamp: str) -> bool:
        """回滚到指定版本"""
        try:
            # 创建新的时间戳目录，复制目标版本的内容
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            self.publish_lora(lora_id, str(self.cpfs_base / f"lora{lora_id}" / target_timestamp))
            return True
        except Exception as e:
            logger.error(f"回滚LoRA {lora_id} 到版本 {target_timestamp} 失败: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取发布器状态"""
        status = {}
        for i in range(1, 9):
            versions = self.list_versions(i)
            status[f"lora{i}"] = {
                "versions": len(versions),
                "latest": versions[0]["timestamp"] if versions else None
            }
        return status


def create_mock_checkpoint(lora_id: int, weights: Dict[str, Any]) -> str:
    """创建模拟checkpoint（用于演示）"""
    import tempfile
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix=f"lora_{lora_id}_")
    
    # 创建adapter_config.json
    config = {
        "base_model_name_or_path": "qwen-2",
        "bias": "none",
        "enable_lora": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": weights.get("alpha", 16.0),
        "lora_dropout": weights.get("dropout", 0.1),
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": weights.get("rank", 8),
        "target_modules": weights.get("target_modules", ["q_proj", "v_proj"]),
        "task_type": "CAUSAL_LM"
    }
    
    with open(os.path.join(temp_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 创建模拟的adapter_model.bin（实际应该是真实的权重文件）
    with open(os.path.join(temp_dir, "adapter_model.bin"), "wb") as f:
        f.write(b"mock_lora_weights_for_demo")
    
    return temp_dir


def demo_publish():
    """演示发布功能"""
    publisher = LoRAPublisher()
    
    # 模拟发布8个LoRA
    for i in range(1, 9):
        # 创建模拟权重
        weights = {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        # 创建模拟checkpoint
        checkpoint_dir = create_mock_checkpoint(i, weights)
        
        # 发布
        metadata = {
            "reward": 0.8 + i * 0.02,  # 模拟奖励
            "training_step": i * 100,
            "weights_info": weights
        }
        
        timestamp = publisher.publish_lora(i, checkpoint_dir, metadata)
        print(f"发布LoRA {i}: {timestamp}")
        
        # 清理临时目录
        shutil.rmtree(checkpoint_dir)
    
    # 显示状态
    status = publisher.get_status()
    print("\n发布状态:")
    for lora_id, info in status.items():
        print(f"  {lora_id}: {info['versions']} 个版本, 最新: {info['latest']}")


if __name__ == "__main__":
    demo_publish()

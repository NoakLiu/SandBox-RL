#!/usr/bin/env python3
"""
LoRA热更新管理器 - 集成到Sandbox-RL核心模块

功能：
1. 监控CPFS上的LoRA checkpoint更新
2. 自动热插拔LoRA权重到8个vLLM实例
3. 与Sandbox-RL的RL策略集成
4. 支持8个LoRA独立更新
"""

import os
import time
import json
import threading
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA配置"""
    lora_id: int
    port: int
    cpfs_root: str
    adapter_name: str
    adapter_id: int
    base_url: str = ""
    
    def __post_init__(self):
        self.base_url = f"http://127.0.0.1:{self.port}"


@dataclass
class LoRAVersion:
    """LoRA版本信息"""
    version_path: Path
    timestamp: str
    adapter_path: str
    is_ready: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoRAHotSwapManager:
    """LoRA热更新管理器"""
    
    def __init__(self, 
                 lora_configs: Dict[int, LoRAConfig],
                 poll_interval: float = 5.0,
                 enable_probe: bool = True,
                 probe_prompt: str = "请用简短两句话问候一下。"):
        
        self.lora_configs = lora_configs
        self.poll_interval = poll_interval
        self.enable_probe = enable_probe
        self.probe_prompt = probe_prompt
        
        # 状态跟踪
        self.current_versions: Dict[int, str] = {}
        self.api_styles: Dict[int, Dict] = {}
        self.is_running = False
        self.workers: List[threading.Thread] = []
        
        # HTTP会话
        self.session = None
        
        # 回调函数
        self.on_lora_updated = None
        self.on_lora_failed = None
        
        logger.info(f"LoRA热更新管理器初始化完成: {len(lora_configs)}个LoRA")
    
    async def start(self):
        """启动热更新管理器"""
        if self.is_running:
            logger.warning("热更新管理器已在运行")
            return
        
        self.is_running = True
        self.session = aiohttp.ClientSession()
        
        # 启动工作线程
        for lora_id, config in self.lora_configs.items():
            worker = threading.Thread(
                target=self._worker_loop,
                args=(lora_id, config),
                daemon=True,
                name=f"LoRA-{lora_id}"
            )
            self.workers.append(worker)
            worker.start()
        
        logger.info("LoRA热更新管理器已启动")
    
    async def stop(self):
        """停止热更新管理器"""
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        # 等待工作线程结束
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        logger.info("LoRA热更新管理器已停止")
    
    def _worker_loop(self, lora_id: int, config: LoRAConfig):
        """工作线程循环"""
        logger.info(f"[LoRA-{lora_id}] 开始监控 {config.cpfs_root} -> {config.base_url}")
        
        # 探测API风格
        api_style = self._detect_api_style(config.base_url)
        self.api_styles[lora_id] = api_style
        
        current_version = None
        
        while self.is_running:
            try:
                # 检查新版本
                latest_version = self._get_latest_ready_version(config.cpfs_root)
                
                if latest_version and latest_version.timestamp != current_version:
                    logger.info(f"[LoRA-{lora_id}] 检测到新版本: {latest_version.timestamp}")
                    
                    # 执行热更新
                    success = self._perform_hot_swap(lora_id, config, api_style, latest_version)
                    
                    if success:
                        current_version = latest_version.timestamp
                        self.current_versions[lora_id] = current_version
                        
                        # 调用回调
                        if self.on_lora_updated:
                            self.on_lora_updated(lora_id, latest_version)
                        
                        logger.info(f"[LoRA-{lora_id}] 热更新成功: {current_version}")
                    else:
                        if self.on_lora_failed:
                            self.on_lora_failed(lora_id, latest_version)
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"[LoRA-{lora_id}] 工作循环错误: {e}")
                time.sleep(self.poll_interval)
    
    def _detect_api_style(self, base_url: str) -> Dict:
        """探测vLLM LoRA API风格"""
        try:
            import requests
            r = requests.get(f"{base_url}/openapi.json", timeout=5)
            r.raise_for_status()
            paths = set(r.json().get("paths", {}).keys())
        except Exception:
            paths = set()
        
        # 两种常见API风格
        legacy = {
            "style": "legacy",
            "apply": "/v1/load_lora_adapter",
            "list": "/v1/lora_adapters", 
            "remove": "/v1/unload_lora_adapter",
        }
        new = {
            "style": "new",
            "apply": "/v1/lora/apply",
            "list": "/v1/lora/adapters",
            "remove": "/v1/lora/adapters/{name}",
        }
        
        if legacy["apply"] in paths or legacy["list"] in paths:
            return legacy
        if new["apply"] in paths or new["list"] in paths:
            return new
        
        # 默认使用legacy风格
        return legacy
    
    def _get_latest_ready_version(self, cpfs_root: str) -> Optional[LoRAVersion]:
        """获取最新的就绪版本"""
        root_path = Path(cpfs_root)
        if not root_path.exists():
            return None
        
        candidates = []
        for child in root_path.iterdir():
            if child.is_dir() and (child / "READY").exists():
                # 检查必要的文件
                if (child / "adapter_model.bin").exists() and (child / "adapter_config.json").exists():
                    timestamp = child.name
                    candidates.append(LoRAVersion(
                        version_path=child,
                        timestamp=timestamp,
                        adapter_path=str(child),
                        is_ready=True
                    ))
        
        if not candidates:
            return None
        
        # 按时间戳排序，返回最新的
        candidates.sort(key=lambda v: v.timestamp, reverse=True)
        return candidates[0]
    
    def _perform_hot_swap(self, lora_id: int, config: LoRAConfig, 
                         api_style: Dict, new_version: LoRAVersion) -> bool:
        """执行热插拔"""
        try:
            # 1. 卸载旧版本
            self._remove_lora(config.base_url, api_style, config.adapter_name)
            logger.debug(f"[LoRA-{lora_id}] 卸载旧版本完成")
            
            # 2. 加载新版本
            self._apply_lora(config.base_url, api_style, config.adapter_name, 
                           new_version.adapter_path, config.adapter_id)
            logger.debug(f"[LoRA-{lora_id}] 加载新版本完成")
            
            # 3. 冒烟测试
            if self.enable_probe:
                probe_result = self._probe_lora(config.base_url, config.adapter_name)
                logger.info(f"[LoRA-{lora_id}] 冒烟测试: {probe_result[:100]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"[LoRA-{lora_id}] 热插拔失败: {e}")
            return False
    
    def _apply_lora(self, base_url: str, api_style: Dict, adapter_name: str, 
                   adapter_path: str, adapter_id: int):
        """应用LoRA"""
        import requests
        
        if api_style["style"] == "legacy":
            url = base_url + api_style["apply"]
            payload = {
                "adapter_name": adapter_name,
                "adapter_path": adapter_path,
                "adapter_id": adapter_id
            }
            r = requests.post(url, json=payload, timeout=60)
        else:
            url = base_url + api_style["apply"]
            payload = {
                "adapter_name": adapter_name,
                "adapter_path": adapter_path,
                "adapter_id": adapter_id
            }
            r = requests.post(url, json=payload, timeout=60)
        
        r.raise_for_status()
        return r.json() if r.text else {"ok": True}
    
    def _remove_lora(self, base_url: str, api_style: Dict, adapter_name: str):
        """移除LoRA"""
        import requests
        
        try:
            if api_style["style"] == "legacy":
                url = base_url + api_style["remove"]
                r = requests.post(url, json={"adapter_name": adapter_name}, timeout=30)
            else:
                url = base_url + api_style["remove"].replace("{name}", adapter_name)
                r = requests.delete(url, timeout=30)
            
            if r.status_code in (200, 204):
                return {"ok": True}
            r.raise_for_status()
            return r.json() if r.text else {"ok": True}
        except Exception as e:
            # 如果移除失败（可能本来就没有），忽略错误
            logger.debug(f"移除LoRA {adapter_name} 时忽略错误: {e}")
            return {"ok": True}
    
    def _probe_lora(self, base_url: str, adapter_name: str) -> str:
        """冒烟测试LoRA"""
        import requests
        
        url = f"{base_url}/v1/chat/completions"
        
        # 尝试不同的请求格式
        payloads = [
            {
                "model": "qwen-2",
                "messages": [{"role": "user", "content": self.probe_prompt}],
                "extra_body": {"lora_request": {"lora_name": adapter_name}},
            },
            {
                "model": "qwen-2", 
                "messages": [{"role": "user", "content": self.probe_prompt}],
                "lora_request": {"lora_name": adapter_name},
            },
            {
                "model": adapter_name,
                "messages": [{"role": "user", "content": self.probe_prompt}],
            },
        ]
        
        for payload in payloads:
            try:
                r = requests.post(url, json=payload, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return msg
            except Exception:
                continue
        
        return "冒烟测试失败"
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            "is_running": self.is_running,
            "current_versions": self.current_versions.copy(),
            "api_styles": {k: v["style"] for k, v in self.api_styles.items()},
            "lora_configs": {
                lora_id: {
                    "port": config.port,
                    "cpfs_root": config.cpfs_root,
                    "adapter_name": config.adapter_name
                }
                for lora_id, config in self.lora_configs.items()
            }
        }


class LoRAPublisher:
    """LoRA发布器 - 用于Sandbox-RL RL策略发布新权重"""
    
    def __init__(self, cpfs_base: str = "/cpfs04/shared/kilab/lora_ckpts"):
        self.cpfs_base = Path(cpfs_base)
        self.cpfs_base.mkdir(parents=True, exist_ok=True)
    
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
            
            import shutil
            shutil.copy2(src_file, dst_file)
        
        # 写入元数据
        if metadata:
            with open(dst_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 创建READY标志（原子发布）
        (dst_dir / "READY").touch()
        
        logger.info(f"发布LoRA {lora_id} -> {dst_dir}")
        return timestamp
    
    def list_versions(self, lora_id: int) -> List[LoRAVersion]:
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
                
                versions.append(LoRAVersion(
                    version_path=child,
                    timestamp=child.name,
                    adapter_path=str(child),
                    is_ready=True,
                    metadata=metadata
                ))
        
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        return versions
    
    def rollback_to_version(self, lora_id: int, target_timestamp: str) -> bool:
        """回滚到指定版本"""
        try:
            # 创建新的时间戳目录，复制目标版本的内容
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
            return self.publish_lora(lora_id, str(self.cpfs_base / f"lora{lora_id}" / target_timestamp))
        except Exception as e:
            logger.error(f"回滚LoRA {lora_id} 到版本 {target_timestamp} 失败: {e}")
            return False


# 工厂函数
def create_lora_hotswap_manager(
    cpfs_base: str = "/cpfs04/shared/kilab/lora_ckpts",
    base_port: int = 8001,
    num_loras: int = 8,
    poll_interval: float = 5.0,
    enable_probe: bool = True
) -> LoRAHotSwapManager:
    """创建LoRA热更新管理器"""
    
    lora_configs = {}
    for i in range(1, num_loras + 1):
        lora_configs[i] = LoRAConfig(
            lora_id=i,
            port=base_port + i - 1,
            cpfs_root=f"{cpfs_base}/lora{i}",
            adapter_name=f"lora{i}",
            adapter_id=i
        )
    
    return LoRAHotSwapManager(
        lora_configs=lora_configs,
        poll_interval=poll_interval,
        enable_probe=enable_probe
    )


def create_lora_publisher(
    cpfs_base: str = "/cpfs04/shared/kilab/lora_ckpts"
) -> LoRAPublisher:
    """创建LoRA发布器"""
    return LoRAPublisher(cpfs_base)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型+8GPU的LoRA热更新管理器

功能：
1. 监听8个LoRA目录 (lora1-lora8)
2. 自动检测新版本并热更新
3. 支持原子发布（READY文件）
4. 自动探测vLLM LoRA API风格
"""

import os
import time
import json
import requests
import logging
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# 配置
BASE_URL = "http://127.0.0.1:8001"     # 单实例端口
SERVED_MODEL = "qwen-2"
POLL_INTERVAL = 5.0
CPFS_BASE = "/cpfs04/shared/kilab/liudong"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# LoRA -> 目录映射
LORA_DIRS: Dict[str, str] = {
    f"lora{i}": f"{CPFS_BASE}/lora{i}" for i in range(1, 9)
}

# HTTP会话
sess = requests.Session()
sess.headers.update({"Content-Type": "application/json"})


def detect_api():
    """探测vLLM LoRA API风格"""
    try:
        r = sess.get(f"{BASE_URL}/openapi.json", timeout=5)
        r.raise_for_status()
        paths = set(r.json().get("paths", {}).keys())
    except Exception as e:
        logger.warning(f"无法获取OpenAPI: {e}")
        paths = set()
    
    # 两种常见API风格
    legacy = {
        "style": "legacy",
        "apply": "/v1/load_lora_adapter",
        "list": "/v1/lora_adapters", 
        "remove": "/v1/unload_lora_adapter"
    }
    new = {
        "style": "new",
        "apply": "/v1/lora/apply",
        "list": "/v1/lora/adapters",
        "remove": "/v1/lora/adapters/{name}"
    }
    
    if legacy["apply"] in paths or legacy["list"] in paths:
        return legacy
    if new["apply"] in paths or new["list"] in paths:
        return new
    
    # 默认使用legacy风格
    return legacy


def latest_ready(root: str) -> Optional[Path]:
    """获取最新的就绪版本"""
    p = Path(root)
    if not p.exists():
        return None
    
    candidates = []
    for child in p.iterdir():
        if child.is_dir() and (child / "READY").exists():
            # 检查必要的文件
            if (child / "adapter_model.bin").exists() and (child / "adapter_config.json").exists():
                candidates.append(child)
    
    if not candidates:
        return None
    
    # 按修改时间排序，返回最新的
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def lora_apply(name: str, path: str, adapter_id: int):
    """应用LoRA"""
    url = BASE_URL + API["apply"]
    payload = {
        "adapter_name": name, 
        "adapter_path": path, 
        "adapter_id": adapter_id
    }
    
    logger.info(f"应用LoRA: {name} -> {path}")
    r = sess.post(url, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.json() if r.text else {"ok": True}


def lora_remove(name: str):
    """移除LoRA"""
    try:
        if API["style"] == "legacy":
            url = BASE_URL + API["remove"]
            r = sess.post(url, data=json.dumps({"adapter_name": name}), timeout=30)
        else:
            url = BASE_URL + API["remove"].replace("{name}", name)
            r = sess.delete(url, timeout=30)
        
        if r.status_code in (200, 204):
            logger.info(f"移除LoRA: {name}")
            return {"ok": True}
        r.raise_for_status()
        return r.json() if r.text else {"ok": True}
    except Exception as e:
        logger.debug(f"移除LoRA {name} 时忽略错误: {e}")
        return {"ok": True}


def lora_list():
    """列出当前加载的LoRA"""
    try:
        url = BASE_URL + API["list"]
        r = sess.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"获取LoRA列表失败: {e}")
        return {"data": []}


def probe_lora(name: str) -> str:
    """冒烟测试LoRA"""
    url = f"{BASE_URL}/v1/chat/completions"
    
    # 尝试不同的请求格式
    payloads = [
        {
            "model": "qwen-2",
            "messages": [{"role": "user", "content": "请用简短两句话问候一下。"}],
            "extra_body": {"lora_request": {"lora_name": name}},
        },
        {
            "model": "qwen-2", 
            "messages": [{"role": "user", "content": "请用简短两句话问候一下。"}],
            "lora_request": {"lora_name": name},
        },
        {
            "model": name,
            "messages": [{"role": "user", "content": "请用简短两句话问候一下。"}],
        },
    ]
    
    for payload in payloads:
        try:
            r = sess.post(url, data=json.dumps(payload), timeout=60)
            if r.status_code == 200:
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return msg
        except Exception:
            continue
    
    return "冒烟测试失败"


def main():
    """主循环"""
    global API
    
    logger.info("🔍 探测LoRA API风格...")
    API = detect_api()
    logger.info(f"LoRA API -> {API['style']}")
    
    logger.info("🚀 启动LoRA热更新管理器...")
    logger.info(f"监听目录: {CPFS_BASE}")
    logger.info(f"vLLM地址: {BASE_URL}")
    logger.info(f"轮询间隔: {POLL_INTERVAL}秒")
    
    # 检查vLLM是否运行
    try:
        r = sess.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            logger.info("✅ vLLM实例运行正常")
        else:
            logger.error("❌ vLLM实例异常")
            return
    except Exception as e:
        logger.error(f"❌ 无法连接到vLLM: {e}")
        return
    
    # 显示当前LoRA状态
    current_loras = lora_list()
    logger.info(f"当前加载的LoRA: {current_loras}")
    
    current_ver: Dict[str, str] = {}
    
    logger.info("🔄 开始监控LoRA更新...")
    
    while True:
        try:
            for name, root in LORA_DIRS.items():
                try:
                    latest = latest_ready(root)
                    if not latest:
                        continue
                    
                    ver = latest.name
                    if current_ver.get(name) == ver:
                        continue
                    
                    logger.info(f"[{name}] 检测到新版本 -> {latest}")
                    
                    # 尝试卸载旧版本（第一次可能不存在，忽略错误）
                    try:
                        lora_remove(name)
                        logger.info(f"[{name}] 卸载旧版本完成")
                    except Exception as e:
                        logger.debug(f"[{name}] 卸载跳过: {e}")
                    
                    # 加载新版本
                    adapter_id = int(name[4:])  # lora1 -> 1, lora2 -> 2, ...
                    lora_apply(name, str(latest), adapter_id)
                    logger.info(f"[{name}] 加载新版本完成: {latest}")
                    
                    # 冒烟测试
                    try:
                        probe_result = probe_lora(name)
                        logger.info(f"[{name}] 冒烟测试: {probe_result[:100]}...")
                    except Exception as e:
                        logger.warning(f"[{name}] 冒烟测试失败: {e}")
                    
                    current_ver[name] = ver
                    
                except Exception as e:
                    logger.error(f"[{name}] 更新错误: {e}")
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("🛑 用户中断，退出...")
            break
        except Exception as e:
            logger.error(f"主循环错误: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

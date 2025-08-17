#!/usr/bin/env python3
"""
最小测试脚本 - 直连某张卡/端口
"""

import requests
import sys
import json

def test_port(port):
    """测试指定端口"""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "qwen-2",
        "messages": [{"role": "user", "content": f"Hello from port {port}!"}],
    }
    
    try:
        r = requests.post(url, json=payload, timeout=20)
        print(f"Status: {r.status_code}")
        print(json.dumps(r.json(), indent=2, ensure_ascii=False)[:800])
        return r.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"Testing port {port}...")
    success = test_port(port)
    print(f"Test {'PASSED' if success else 'FAILED'}")

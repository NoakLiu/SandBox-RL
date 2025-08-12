#!/usr/bin/env python3
"""
生成测试数据用于可视化演示
"""

import json
import random
from datetime import datetime
from typing import List, Dict, Any

def generate_test_data() -> List[Dict[str, Any]]:
    """生成测试数据"""
    data = []
    
    # 生成模型性能数据
    model_roles = ["leader", "follower", "competitor", "teammate", "neutral"]
    team_ids = ["team_alpha", "team_beta", "team_gamma", None]
    
    for i in range(15):
        role = random.choice(model_roles)
        team_id = random.choice(team_ids) if role in ["leader", "teammate"] else None
        
        performance = {
            "model_id": f"model_{i:03d}",
            "task_id": f"task_{random.randint(1, 20):03d}",
            "completion_time": random.uniform(0.5, 5.0),
            "accuracy": random.uniform(0.6, 0.95),
            "efficiency": random.uniform(0.5, 0.9),
            "cooperation_score": random.uniform(0.0, 1.0),
            "reward_earned": random.uniform(10.0, 100.0),
            "weight_updates": random.randint(1, 20),
            "lora_adaptations": random.randint(0, 10),
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "team_id": team_id,
            "total_tasks": random.randint(5, 25),
            "avg_accuracy": random.uniform(0.6, 0.95),
            "avg_efficiency": random.uniform(0.5, 0.9),
            "avg_cooperation": random.uniform(0.0, 1.0),
            "total_reward": random.uniform(20.0, 150.0)
        }
        data.append(performance)
    
    return data

def main():
    """主函数"""
    print("📊 生成测试数据...")
    
    # 生成数据
    test_data = generate_test_data()
    
    # 保存到文件
    filename = "multi_model_training_simple_results.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 测试数据已保存到: {filename}")
    print(f"📈 生成了 {len(test_data)} 条模型性能记录")
    
    # 显示数据统计
    roles = [item["role"] for item in test_data]
    role_counts = {}
    for role in roles:
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print("\n📊 数据统计:")
    print(f"   总记录数: {len(test_data)}")
    print(f"   角色分布: {role_counts}")
    print(f"   平均准确率: {sum(item['accuracy'] for item in test_data) / len(test_data):.3f}")
    print(f"   平均效率: {sum(item['efficiency'] for item in test_data) / len(test_data):.3f}")
    print(f"   总奖励: {sum(item['reward_earned'] for item in test_data):.2f}")

if __name__ == "__main__":
    main()

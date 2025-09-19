#!/usr/bin/env python3
"""
Core On-Policy RL Demo

测试core模块中的合作因子和能力因子功能
"""

import numpy as np
import time
import logging
import json
from typing import Dict, List, Any

# Sandbox-RL Core imports
try:
    from sandbox_rl.core.rl_algorithms import (
        CooperationType,
        CompetenceType,
        CooperationFactor,
        CompetenceFactor,
        RLConfig,
        TrajectoryStep,
        OnPolicyRLAgent,
        MultiAgentOnPolicyRL
    )
    HAS_SANDGRAPH = True
    print("✅ Sandbox-RL core RL algorithms imported successfully")
except ImportError as e:
    HAS_SANDGRAPH = False
    print(f"❌ Sandbox-RL core RL algorithms not available: {e}")
    print("Will use mock implementations")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_cooperation_factors():
    """演示不同的合作因子配置"""
    print("\n🔗 合作因子演示")
    print("=" * 50)
    
    # 创建不同的合作配置
    cooperation_configs = [
        ("无合作", CooperationFactor(
            cooperation_type=CooperationType.NONE,
            cooperation_strength=0.0
        )),
        ("团队合作", CooperationFactor(
            cooperation_type=CooperationType.TEAM_BASED,
            cooperation_strength=0.3,
            team_size=4,
            shared_reward_ratio=0.6
        )),
        ("共享奖励", CooperationFactor(
            cooperation_type=CooperationType.SHARED_REWARDS,
            cooperation_strength=0.2,
            shared_reward_ratio=0.8
        )),
        ("知识转移", CooperationFactor(
            cooperation_type=CooperationType.KNOWLEDGE_TRANSFER,
            cooperation_strength=0.4,
            knowledge_transfer_rate=0.15
        ))
    ]
    
    for i, (description, config) in enumerate(cooperation_configs):
        print(f"\n{i+1}. {description}")
        print(f"   - 合作类型: {config.cooperation_type.value}")
        print(f"   - 合作强度: {config.cooperation_strength}")
        print(f"   - 团队大小: {config.team_size}")
        print(f"   - 共享奖励比例: {config.shared_reward_ratio}")
        print(f"   - 知识转移率: {config.knowledge_transfer_rate}")

def demonstrate_competence_factors():
    """演示不同的能力因子配置"""
    print("\n🎯 能力因子演示")
    print("=" * 50)
    
    # 创建不同的能力配置
    competence_configs = [
        ("新手智能体", CompetenceFactor(
            competence_type=CompetenceType.NOVICE,
            base_capability=0.3,
            learning_rate=0.01,
            adaptation_speed=0.05
        )),
        ("通用智能体", CompetenceFactor(
            competence_type=CompetenceType.GENERAL,
            base_capability=0.5,
            learning_rate=0.02,
            adaptation_speed=0.1
        )),
        ("专业智能体", CompetenceFactor(
            competence_type=CompetenceType.SPECIALIZED,
            base_capability=0.6,
            learning_rate=0.03,
            specialization_level=0.4
        )),
        ("自适应智能体", CompetenceFactor(
            competence_type=CompetenceType.ADAPTIVE,
            base_capability=0.4,
            learning_rate=0.025,
            adaptation_speed=0.15
        )),
        ("专家智能体", CompetenceFactor(
            competence_type=CompetenceType.EXPERT,
            base_capability=0.8,
            learning_rate=0.01,
            specialization_level=0.6
        ))
    ]
    
    for i, (description, config) in enumerate(competence_configs):
        print(f"\n{i+1}. {description}")
        print(f"   - 能力类型: {config.competence_type.value}")
        print(f"   - 基础能力: {config.base_capability}")
        print(f"   - 学习率: {config.learning_rate}")
        print(f"   - 适应速度: {config.adaptation_speed}")
        print(f"   - 专业化水平: {config.specialization_level}")

def run_multi_agent_demo():
    """运行多智能体演示"""
    print("\n🚀 多智能体On-Policy RL演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL不可用，跳过训练演示")
        return
    
    # 创建合作配置
    cooperation_configs = []
    for i in range(8):
        if i < 4:
            # 前4个智能体使用团队合作
            cooperation_configs.append(CooperationFactor(
                cooperation_type=CooperationType.TEAM_BASED,
                cooperation_strength=0.3 + 0.1 * (i % 2),
                team_size=4,
                shared_reward_ratio=0.6
            ))
        else:
            # 后4个智能体使用共享奖励
            cooperation_configs.append(CooperationFactor(
                cooperation_type=CooperationType.SHARED_REWARDS,
                cooperation_strength=0.2,
                shared_reward_ratio=0.7
            ))
    
    # 创建能力配置
    competence_configs = []
    for i in range(8):
        competence_configs.append(CompetenceFactor(
            competence_type=CompetenceType.ADAPTIVE,
            base_capability=0.4 + 0.1 * (i % 3),
            learning_rate=0.02 + 0.01 * (i % 2),
            adaptation_speed=0.15 + 0.05 * (i % 2)
        ))
    
    # 创建多智能体系统
    multi_agent_system = MultiAgentOnPolicyRL(
        num_agents=8,
        state_dim=64,
        action_dim=10,
        cooperation_configs=cooperation_configs,
        competence_configs=competence_configs
    )
    
    print(f"创建了包含{len(multi_agent_system.agents)}个智能体的系统")
    
    # 模拟训练过程
    num_episodes = 20
    max_steps_per_episode = 50
    
    print(f"开始{num_episodes}个episode的训练...")
    
    episode_rewards = []
    agent_stats_history = []
    
    for episode in range(num_episodes):
        episode_reward = 0.0
        
        for step in range(max_steps_per_episode):
            # 随机选择智能体
            agent_id = f"agent_{step % 8}"
            
            # 创建模拟状态
            state = {
                "position": np.random.randn(3),
                "velocity": np.random.randn(3),
                "energy": np.random.uniform(0.5, 1.0),
                "step": step
            }
            
            # 智能体执行动作
            action, log_prob, value = multi_agent_system.step(agent_id, state)
            
            # 模拟奖励
            reward = np.sin(step * 0.1) + np.random.normal(0, 0.1)
            reward += 0.1 * (int(action.split('_')[1]) / 10)  # 动作奖励
            
            # 创建轨迹步骤
            trajectory_step = TrajectoryStep(
                state=state,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=(step == max_steps_per_episode - 1)
            )
            
            # 更新智能体
            multi_agent_system.update_agent(agent_id, trajectory_step)
            
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        
        # 每5个episode收集一次统计信息
        if episode % 5 == 0:
            agent_stats = multi_agent_system.get_agent_stats()
            agent_stats_history.append({
                'episode': episode,
                'stats': agent_stats
            })
            print(f"Episode {episode + 1}: 总奖励 = {episode_reward:.3f}")
    
    print(f"训练完成！平均奖励: {np.mean(episode_rewards):.3f}")
    
    # 显示最终统计信息
    final_stats = multi_agent_system.get_agent_stats()
    print("\n📊 最终智能体统计:")
    for agent_id, stats in final_stats.items():
        print(f"  {agent_id}:")
        print(f"    - 能力: {stats['capability']:.3f}")
        print(f"    - 经验数: {stats['experience_count']}")
        print(f"    - 合作类型: {stats['cooperation_type']}")
        print(f"    - 能力类型: {stats['competence_type']}")
        print(f"    - 平均奖励: {stats['avg_reward']:.3f}")
    
    # 显示团队信息
    team_stats = multi_agent_system.get_team_stats()
    print("\n👥 团队信息:")
    for team_id, members in team_stats.items():
        print(f"  {team_id}: {members}")
    
    # 保存结果
    results = {
        'episode_rewards': episode_rewards,
        'final_agent_stats': final_stats,
        'team_stats': team_stats,
        'training_config': {
            'num_agents': 8,
            'num_episodes': num_episodes,
            'max_steps_per_episode': max_steps_per_episode
        }
    }
    
    with open('core_on_policy_rl_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n✅ 演示完成！")
    print("📁 结果已保存到: core_on_policy_rl_results.json")
    
    return multi_agent_system, episode_rewards, agent_stats_history

def main():
    """主演示函数"""
    print("🚀 Core On-Policy RL 演示")
    print("=" * 60)
    print("本演示展示core模块中的:")
    print("- 合作因子：控制多智能体协作行为")
    print("- 能力因子：控制个体智能体能力和学习")
    print("- On-Policy RL：支持合作和能力的强化学习")
    
    # 演示合作因子
    demonstrate_cooperation_factors()
    
    # 演示能力因子
    demonstrate_competence_factors()
    
    # 运行多智能体演示
    if HAS_SANDGRAPH:
        multi_agent_system, episode_rewards, agent_stats_history = run_multi_agent_demo()
        
        print(f"\n📈 训练结果摘要:")
        print(f"  - 总episode数: {len(episode_rewards)}")
        print(f"  - 平均奖励: {np.mean(episode_rewards):.3f}")
        print(f"  - 最高奖励: {max(episode_rewards):.3f}")
        print(f"  - 最低奖励: {min(episode_rewards):.3f}")
        print(f"  - 奖励标准差: {np.std(episode_rewards):.3f}")
        
        print("\n✅ 演示成功完成！")
    else:
        print("\n⚠️ 多智能体演示因缺少依赖而跳过")
        print("   确保Sandbox-RL可用以获得完整演示")

if __name__ == "__main__":
    main()

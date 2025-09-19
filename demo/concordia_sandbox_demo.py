#!/usr/bin/env python3
"""
Concordia Contest Sandbox Demo

演示Concordia Contest沙盒在Sandbox-RLX中的使用
"""

import os
import time
import json
import logging
from typing import Dict, List, Any

# Sandbox-RL Core imports
try:
    from sandbox_rl.core.concordia_sandbox import (
        ConcordiaScenario,
        ConcordiaRole,
        ConcordiaConfig,
        ConcordiaSandbox,
        create_concordia_sandbox,
        create_trading_scenario,
        create_public_goods_scenario,
        create_negotiation_scenario
    )
    from sandbox_rl.core.rl_algorithms import (
        CooperationType, CompetenceType,
        CooperationFactor, CompetenceFactor
    )
    HAS_SANDGRAPH = True
    print("✅ Sandbox-RL Concordia sandbox imported successfully")
except ImportError as e:
    HAS_SANDGRAPH = False
    print(f"❌ Sandbox-RL Concordia sandbox not available: {e}")
    print("Will use mock implementations")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_trading_scenario():
    """演示交易场景"""
    print("\n💰 交易场景演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL不可用，跳过演示")
        return
    
    # 创建交易场景
    sandbox = create_trading_scenario("trader_a")
    
    print(f"创建了交易场景: {sandbox.scenario} - {sandbox.role}")
    
    # 运行几个回合
    num_turns = 5
    print(f"\n运行{num_turns}个回合...")
    
    total_reward = 0.0
    
    for turn in range(num_turns):
        # 生成案例
        case = sandbox.case_generator()
        print(f"\n回合 {turn + 1}:")
        print(f"  观察: {case['obs'][:100]}...")
        
        # 生成提示
        prompt = sandbox.prompt_func(case)
        print(f"  提示长度: {len(prompt)} 字符")
        
        # 模拟动作（这里用简单的基线动作）
        if turn == 0:
            action = "I offer to trade my apple for your orange."
        elif turn == 1:
            action = "I accept your offer and propose trading banana for grape."
        elif turn == 2:
            action = "Let's agree on a fair price for both items."
        elif turn == 3:
            action = "I think we can both benefit from this trade."
        else:
            action = "Thank you for the successful trade."
        
        print(f"  动作: {action}")
        
        # 验证动作并获取奖励
        reward = sandbox.verify_score(action, case)
        total_reward += reward.reward
        
        print(f"  奖励: {reward.reward:.3f}")
        print(f"  完成: {reward.done}")
        
        if reward.done:
            print("  场景结束")
            break
    
    print(f"\n总奖励: {total_reward:.3f}")
    
    # 显示最终状态
    final_state = sandbox.get_state()
    print(f"\n最终状态:")
    print(f"  回合数: {final_state['turn']}")
    print(f"  记忆条数: {len(final_state['memory'])}")
    print(f"  指标: {final_state['metrics']}")


def demonstrate_public_goods_scenario():
    """演示公共物品场景"""
    print("\n🏛️ 公共物品场景演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL不可用，跳过演示")
        return
    
    # 创建公共物品场景
    config = ConcordiaConfig(
        scenario=ConcordiaScenario.PUBLIC_GOODS,
        role=ConcordiaRole.CONTRIBUTOR,
        max_turns=10,
        cooperation_factor=CooperationFactor(
            cooperation_type=CooperationType.TEAM_BASED,
            cooperation_strength=0.8,
            team_size=2
        )
    )
    
    sandbox = create_concordia_sandbox("public_goods", "contributor", config)
    
    print(f"创建了公共物品场景: {sandbox.scenario} - {sandbox.role}")
    
    # 运行几个回合
    num_turns = 5
    print(f"\n运行{num_turns}个回合...")
    
    total_reward = 0.0
    
    for turn in range(num_turns):
        # 生成案例
        case = sandbox.case_generator()
        print(f"\n回合 {turn + 1}:")
        print(f"  观察: {case['obs'][:100]}...")
        
        # 生成提示
        prompt = sandbox.prompt_func(case)
        print(f"  提示长度: {len(prompt)} 字符")
        
        # 模拟动作
        if turn == 0:
            action = "I will contribute 10 resources to the public pool."
        elif turn == 1:
            action = "I contribute another 15 resources for the common good."
        elif turn == 2:
            action = "Let me contribute 20 more resources to maximize social welfare."
        elif turn == 3:
            action = "I contribute 5 resources to maintain cooperation."
        else:
            action = "I contribute my remaining resources to the public pool."
        
        print(f"  动作: {action}")
        
        # 验证动作并获取奖励
        reward = sandbox.verify_score(action, case)
        total_reward += reward.reward
        
        print(f"  奖励: {reward.reward:.3f}")
        print(f"  完成: {reward.done}")
        
        if reward.done:
            print("  场景结束")
            break
    
    print(f"\n总奖励: {total_reward:.3f}")
    
    # 显示最终状态
    final_state = sandbox.get_state()
    print(f"\n最终状态:")
    print(f"  回合数: {final_state['turn']}")
    print(f"  记忆条数: {len(final_state['memory'])}")
    print(f"  指标: {final_state['metrics']}")


def demonstrate_negotiation_scenario():
    """演示协商场景"""
    print("\n🤝 协商场景演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL不可用，跳过演示")
        return
    
    # 创建协商场景
    config = ConcordiaConfig(
        scenario=ConcordiaScenario.NEGOTIATION,
        role=ConcordiaRole.NEGOTIATOR,
        max_turns=8,
        cooperation_factor=CooperationFactor(
            cooperation_type=CooperationType.KNOWLEDGE_TRANSFER,
            cooperation_strength=0.6
        )
    )
    
    sandbox = create_concordia_sandbox("negotiation", "negotiator_a", config)
    
    print(f"创建了协商场景: {sandbox.scenario} - {sandbox.role}")
    
    # 运行几个回合
    num_turns = 5
    print(f"\n运行{num_turns}个回合...")
    
    total_reward = 0.0
    
    for turn in range(num_turns):
        # 生成案例
        case = sandbox.case_generator()
        print(f"\n回合 {turn + 1}:")
        print(f"  观察: {case['obs'][:100]}...")
        
        # 生成提示
        prompt = sandbox.prompt_func(case)
        print(f"  提示长度: {len(prompt)} 字符")
        
        # 模拟动作
        if turn == 0:
            action = "I propose we split the stakes 60-40 in my favor."
        elif turn == 1:
            action = "I can compromise to a 55-45 split."
        elif turn == 2:
            action = "Let's find a middle ground at 50-50."
        elif turn == 3:
            action = "I agree to the 50-50 split for mutual benefit."
        else:
            action = "Thank you for reaching this agreement."
        
        print(f"  动作: {action}")
        
        # 验证动作并获取奖励
        reward = sandbox.verify_score(action, case)
        total_reward += reward.reward
        
        print(f"  奖励: {reward.reward:.3f}")
        print(f"  完成: {reward.done}")
        
        if reward.done:
            print("  场景结束")
            break
    
    print(f"\n总奖励: {total_reward:.3f}")
    
    # 显示最终状态
    final_state = sandbox.get_state()
    print(f"\n最终状态:")
    print(f"  回合数: {final_state['turn']}")
    print(f"  记忆条数: {len(final_state['memory'])}")
    print(f"  指标: {final_state['metrics']}")


def demonstrate_multi_scenario_comparison():
    """演示多场景比较"""
    print("\n📊 多场景比较演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL不可用，跳过演示")
        return
    
    # 创建不同场景
    scenarios = [
        ("trading", "trader_a", "交易场景"),
        ("public_goods", "contributor", "公共物品场景"),
        ("negotiation", "negotiator_a", "协商场景")
    ]
    
    results = {}
    
    for scenario, role, name in scenarios:
        print(f"\n测试 {name}...")
        
        # 创建沙盒
        sandbox = create_concordia_sandbox(scenario, role)
        
        # 运行3个回合
        total_reward = 0.0
        for turn in range(3):
            case = sandbox.case_generator()
            prompt = sandbox.prompt_func(case)
            
            # 简单的基线动作
            if "trading" in scenario:
                action = "I propose a fair trade."
            elif "public_goods" in scenario:
                action = "I contribute resources."
            else:
                action = "I propose a compromise."
            
            reward = sandbox.verify_score(action, case)
            total_reward += reward.reward
            
            if reward.done:
                break
        
        results[name] = {
            "total_reward": total_reward,
            "final_state": sandbox.get_state()
        }
        
        print(f"  {name} 总奖励: {total_reward:.3f}")
    
    # 比较结果
    print(f"\n📈 场景比较结果:")
    for name, result in results.items():
        print(f"  {name}:")
        print(f"    总奖励: {result['total_reward']:.3f}")
        print(f"    最终回合: {result['final_state']['turn']}")
        print(f"    协作率: {result['final_state']['metrics'].get('collaboration_rate', 0):.3f}")


def demonstrate_integration_with_rl():
    """演示与RL系统的集成"""
    print("\n🤖 与RL系统集成演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ Sandbox-RL不可用，跳过演示")
        return
    
    try:
        from sandbox_rl.core.rl_algorithms import MultiAgentOnPolicyRL
        
        # 创建多个角色的沙盒
        sandboxes = {}
        roles = ["trader_a", "contributor", "negotiator_a"]
        scenarios = ["trading", "public_goods", "negotiation"]
        
        for role, scenario in zip(roles, scenarios):
            sandboxes[role] = create_concordia_sandbox(scenario, role)
        
        # 创建多智能体RL系统
        cooperation_configs = []
        competence_configs = []
        
        for i, role in enumerate(roles):
            # 合作配置
            cooperation_config = CooperationFactor(
                cooperation_type=CooperationType.TEAM_BASED if i < 2 else CooperationType.SHARED_REWARDS,
                cooperation_strength=0.5 + i * 0.1,
                team_size=2,
                shared_reward_ratio=0.6 + i * 0.1
            )
            cooperation_configs.append(cooperation_config)
            
            # 能力配置
            competence_config = CompetenceFactor(
                competence_type=CompetenceType.ADAPTIVE,
                base_capability=0.4 + i * 0.2,
                learning_rate=0.02 + i * 0.01,
                adaptation_speed=0.15 + i * 0.05
            )
            competence_configs.append(competence_config)
        
        # 创建多智能体RL系统
        multi_agent_rl = MultiAgentOnPolicyRL(
            num_agents=len(roles),
            cooperation_configs=cooperation_configs,
            competence_configs=competence_configs
        )
        
        print(f"创建了包含{len(multi_agent_rl.agents)}个RL智能体的系统")
        print(f"创建了包含{len(sandboxes)}个沙盒的场景")
        
        # 模拟集成场景
        print("\n模拟RL智能体与沙盒的协作...")
        
        for i in range(3):
            for j, role in enumerate(roles):
                # 获取沙盒状态
                sandbox = sandboxes[role]
                case = sandbox.case_generator()
                
                # RL智能体选择动作
                agent_id = f"agent_{j}"
                state = {
                    "position": [i, j, 0],
                    "energy": 1.0,
                    "observation": case['obs'][:50]
                }
                
                try:
                    action, log_prob, value = multi_agent_rl.step(agent_id, state)
                    print(f"  智能体 {agent_id} 选择动作: {action}")
                    
                    # 在沙盒中执行动作
                    reward = sandbox.verify_score(f"RL action: {action}", case)
                    print(f"  沙盒奖励: {reward.reward:.3f}")
                    
                except Exception as e:
                    print(f"  智能体 {agent_id} 处理失败: {e}")
        
        print("✅ RL集成演示完成")
        
    except ImportError as e:
        print(f"❌ RL模块不可用: {e}")


def main():
    """主演示函数"""
    print("🚀 Concordia Contest沙盒演示")
    print("=" * 60)
    print("本演示展示:")
    print("- Concordia Contest场景适配")
    print("- 文本交互环境的case → prompt → y → verify(r)闭环")
    print("- 不同场景的协作机制")
    print("- 与RL系统的集成")
    
    # 演示不同场景
    demonstrate_trading_scenario()
    demonstrate_public_goods_scenario()
    demonstrate_negotiation_scenario()
    
    # 演示多场景比较
    demonstrate_multi_scenario_comparison()
    
    # 演示RL集成
    demonstrate_integration_with_rl()
    
    print("\n✅ 所有演示完成！")
    print("\n📝 总结:")
    print("- 成功实现了Concordia Contest的沙盒适配器")
    print("- 支持多种场景：交易、公共物品、协商")
    print("- 实现了完整的case → prompt → y → verify(r)闭环")
    print("- 可以与RL系统集成，实现智能协作")
    print("- 提供了丰富的协作机制和奖励形状")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sandbox-RL 交互日志演示脚本

展示增强的日志功能，包括：
1. LLM思考过程记录
2. 沙盒状态和动作记录
3. 奖励计算详情记录
4. 动作序列记录
5. 交互过程统计分析
"""

import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径以便导入
sys.path.insert(0, '.')

from demo import TrainingLogger

def demonstrate_interaction_logging():
    """演示详细的交互日志功能"""
    print("🔍 Sandbox-RL Enhanced Interaction Logging Demo")
    print("=" * 60)
    
    # 创建日志记录器
    logger = TrainingLogger("interaction_demo_logs")
    
    # 1. 演示LLM交互记录
    print("\n1. 📝 LLM Interaction Logging Demo")
    print("-" * 40)
    
    llm_nodes = ["task_analyzer", "strategy_planner", "math_solver", "text_processor"]
    
    for i, node_id in enumerate(llm_nodes):
        thinking_process = f"""
        Step {i+1} Analysis:
        - Current task: {node_id.replace('_', ' ').title()}
        - Context: Processing complex workflow
        - Strategy: Apply specialized reasoning for {node_id}
        - Expected outcome: High-quality response with confidence > 0.8
        """
        
        prompt = f"Process task for {node_id} with optimization focus"
        response = f"Optimized response from {node_id} with enhanced reasoning and improved performance metrics"
        
        logger.log_llm_interaction(
            node_id=node_id,
            prompt=prompt,
            response=response,
            thinking_process=thinking_process,
            confidence=0.7 + i * 0.05,
            metadata={
                "step": i + 1,
                "task_type": "workflow_processing",
                "optimization_level": "high"
            }
        )
        time.sleep(0.1)  # 模拟处理时间
    
    # 2. 演示沙盒交互记录
    print("\n2. 🏝️ Sandbox Interaction Logging Demo")
    print("-" * 40)
    
    sandbox_nodes = ["game24_sandbox", "summary_sandbox"]
    
    for i, sandbox_id in enumerate(sandbox_nodes):
        # 模拟沙盒交互序列
        states = ["ready", "processing", "completed"]
        actions = ["initialize", "execute_task", "finalize_result"]
        rewards = [0.1, 0.8, 0.3]  # 执行阶段奖励最高
        
        for j, (state, action, reward) in enumerate(zip(states, actions, rewards)):
            next_state = states[j + 1] if j < len(states) - 1 else "idle"
            
            logger.log_sandbox_interaction(
                sandbox_id=sandbox_id,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=(j == len(states) - 1),
                case_data={
                    "task_id": f"{sandbox_id}_task_{i+1}",
                    "complexity": 3 + i,
                    "input_size": 100 + i * 50
                },
                result_data={
                    "success": True,
                    "quality_score": 0.8 + i * 0.1,
                    "execution_time": 0.5 + j * 0.2
                },
                metadata={
                    "sandbox_type": sandbox_id.split('_')[0],
                    "step": j + 1,
                    "total_steps": len(states)
                }
            )
            time.sleep(0.05)
    
    # 3. 演示奖励计算记录
    print("\n3. 🎁 Reward Calculation Logging Demo")
    print("-" * 40)
    
    for i, node_id in enumerate(llm_nodes):
        raw_score = 0.6 + i * 0.1
        reward_components = {
            "base_reward": raw_score,
            "performance_bonus": raw_score * 0.2,
            "efficiency_bonus": 0.1,
            "quality_bonus": raw_score * 0.15,
            "consistency_penalty": -0.05 if i % 2 == 0 else 0
        }
        total_reward = sum(reward_components.values())
        
        logger.log_reward_calculation(
            node_id=node_id,
            raw_score=raw_score,
            reward_components=reward_components,
            total_reward=total_reward,
            calculation_details={
                "calculation_method": "weighted_sum",
                "bonus_multiplier": 0.2,
                "penalty_applied": i % 2 == 0,
                "normalization": "none"
            }
        )
    
    # 4. 演示动作序列记录
    print("\n4. 🔄 Action Sequence Logging Demo")
    print("-" * 40)
    
    # 创建复杂的动作序列
    workflow_actions = []
    for i in range(3):  # 3个工作流步骤
        step_actions = []
        for j, node_id in enumerate(llm_nodes[:2]):  # 每步2个节点
            action = {
                "action_type": "llm_processing",
                "node_id": node_id,
                "step": i + 1,
                "substep": j + 1,
                "input": f"Step {i+1} input for {node_id}",
                "output": f"Step {i+1} output from {node_id}",
                "duration": 0.3 + j * 0.1,
                "success": True
            }
            step_actions.append(action)
        workflow_actions.extend(step_actions)
    
    logger.log_action_sequence(
        sequence_id="complex_workflow_demo",
        actions=workflow_actions,
        sequence_type="multi_step_workflow",
        metadata={
            "total_steps": 3,
            "nodes_per_step": 2,
            "workflow_type": "parallel_processing"
        }
    )
    
    # 5. 显示交互统计
    print("\n5. 📊 Interaction Statistics")
    print("-" * 40)
    
    summary = logger.get_interaction_summary()
    
    print("LLM Interactions:")
    llm_stats = summary["llm_interactions"]
    print(f"  - Total interactions: {llm_stats['total_count']}")
    print(f"  - Nodes involved: {llm_stats['nodes_involved']}")
    print(f"  - Average confidence: {llm_stats['avg_confidence']:.3f}")
    print(f"  - Total tokens processed: {llm_stats['total_tokens_processed']}")
    
    print("\nSandbox Interactions:")
    sandbox_stats = summary["sandbox_interactions"]
    print(f"  - Total interactions: {sandbox_stats['total_count']}")
    print(f"  - Sandboxes involved: {sandbox_stats['sandboxes_involved']}")
    print(f"  - Total reward: {sandbox_stats['total_reward']:.3f}")
    print(f"  - Average reward: {sandbox_stats['avg_reward']:.3f}")
    
    print("\nReward Calculations:")
    reward_stats = summary["reward_calculations"]
    print(f"  - Total calculations: {reward_stats['total_count']}")
    print(f"  - Total reward distributed: {reward_stats['total_reward']:.3f}")
    print(f"  - Average components per calculation: {reward_stats['avg_components_per_calc']:.1f}")
    
    print("\nAction Sequences:")
    action_stats = summary["action_sequences"]
    print(f"  - Total sequences: {action_stats['total_sequences']}")
    print(f"  - Total actions: {action_stats['total_actions']}")
    
    # 6. 保存日志
    print("\n6. 💾 Saving Interaction Logs")
    print("-" * 40)
    
    timestamp = logger.save_logs()
    
    print("✅ Interaction logs saved successfully!")
    print(f"📁 Log directory: interaction_demo_logs/")
    print(f"📝 Files created:")
    print(f"   - text_logs_{timestamp}.json")
    print(f"   - llm_interactions_{timestamp}.json")
    print(f"   - sandbox_interactions_{timestamp}.json")
    print(f"   - reward_details_{timestamp}.json")
    print(f"   - action_sequences_{timestamp}.json")
    print(f"   - interaction_summary_{timestamp}.json")
    
    # 7. 展示部分日志内容
    print("\n7. 🔍 Sample Log Content Preview")
    print("-" * 40)
    
    if logger.llm_interactions:
        print("Sample LLM Interaction:")
        sample_llm = logger.llm_interactions[0]
        print(f"  Node: {sample_llm['node_id']}")
        print(f"  Confidence: {sample_llm['confidence']:.3f}")
        print(f"  Thinking: {sample_llm['thinking_process'][:100]}...")
        print(f"  Response: {sample_llm['response'][:80]}...")
    
    if logger.sandbox_interactions:
        print("\nSample Sandbox Interaction:")
        sample_sandbox = logger.sandbox_interactions[0]
        print(f"  Sandbox: {sample_sandbox['sandbox_id']}")
        print(f"  State Transition: {sample_sandbox['state']} -> {sample_sandbox['next_state']}")
        print(f"  Action: {sample_sandbox['action']}")
        print(f"  Reward: {sample_sandbox['reward']:.3f}")
    
    if logger.reward_details:
        print("\nSample Reward Calculation:")
        sample_reward = logger.reward_details[0]
        print(f"  Node: {sample_reward['node_id']}")
        print(f"  Raw Score: {sample_reward['raw_score']:.3f}")
        print(f"  Components: {list(sample_reward['reward_components'].keys())}")
        print(f"  Total Reward: {sample_reward['total_reward']:.3f}")
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Interaction Logging Demo Completed!")
    print("✅ All interaction types successfully logged and analyzed")
    print("✅ Detailed logs saved for further analysis")
    print("✅ Statistics and summaries generated")
    
    return logger, timestamp

if __name__ == "__main__":
    logger, timestamp = demonstrate_interaction_logging()
    print(f"\n📋 Demo completed at: {timestamp}") 
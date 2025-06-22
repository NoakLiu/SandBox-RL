#!/usr/bin/env python3
"""
SandGraph 社交网络环境演示 - 基于RL的LLM决策架构

新的架构设计：
1. Sandbox作为环境节点
2. LLM作为决策器（不是节点）
3. RL算法更新LLM权重
4. 状态转移由LLM决策驱动
"""

import sys
import os
import time
import json
import argparse
import random
import re
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from sandgraph.sandbox_implementations import SocialNetworkSandbox


def print_section(title: str):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


class LLMDecisionMaker:
    """LLM决策器 - 不是节点，而是决策引擎"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.decision_count = 0
        
        # 历史数据管理
        self.decision_history = []  # 决策历史
        self.network_history = []   # 网络状态历史
        self.user_history = []      # 用户行为历史
        self.performance_history = [] # 表现历史
        
        # 注册决策节点
        self.llm_manager.register_node("social_decision", {
            "role": "社交网络策略专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
    
    def make_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态做出决策"""
        self.decision_count += 1
        
        # 构建决策提示
        prompt = self._construct_decision_prompt(current_state)
        
        print("=" * 80)
        print(f"Decision {self.decision_count} - Complete Prompt Content:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        try:
            # 生成LLM响应
            response = self.llm_manager.generate_for_node(
                "social_decision",
                prompt,
                temperature=0.7,
                max_new_tokens=256,  # 增加token数量
                do_sample=True
            )
            
            print(f"LLM Response Status: {response.status if hasattr(response, 'status') else 'unknown'}")
            print(f"LLM Complete Response: {response.text}")
            
            # 解析响应
            decision = self._parse_decision_response(response.text)
            
            # 如果解析失败，使用更宽松的解析
            if decision is None:
                decision = self._parse_decision_fallback(response.text)
            
            # 如果还是失败，使用默认决策
            if decision is None:
                decision = {
                    "action": "CREATE_POST",
                    "target": "N/A",
                    "reasoning": "Fallback decision due to parsing failure"
                }
            
            # 更新历史数据
            self._update_history(current_state, decision, response.text)
            
            return {
                "decision": decision,
                "llm_response": response.text,
                "prompt": prompt,
                "decision_count": self.decision_count
            }
            
        except Exception as e:
            print(f"❌ Decision generation failed: {e}")
            fallback_decision = {
                "action": "CREATE_POST",
                "target": "N/A", 
                "reasoning": f"Error in decision generation: {str(e)}"
            }
            
            # 更新历史数据（即使失败）
            self._update_history(current_state, fallback_decision, f"Error: {str(e)}")
            
            return {
                "decision": fallback_decision,
                "llm_response": f"Error: {str(e)}",
                "prompt": prompt,
                "decision_count": self.decision_count
            }
    
    def _update_history(self, state: Dict[str, Any], decision: Dict[str, Any], llm_response: str):
        """更新历史数据"""
        # 记录决策历史
        decision_record = {
            "step": self.decision_count,
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "llm_response": llm_response,
            "network_state": state.get("network_state", {}),
            "user_behavior": state.get("user_behavior", {}),
            "content_metrics": state.get("content_metrics", {})
        }
        self.decision_history.append(decision_record)
        
        # 保持历史记录在合理范围内
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]
        
        # 记录网络状态历史
        network_record = {
            "step": self.decision_count,
            "network_state": state.get("network_state", {}),
            "user_behavior": state.get("user_behavior", {})
        }
        self.network_history.append(network_record)
        
        # 记录用户行为历史
        user_record = {
            "step": self.decision_count,
            "user_behavior": state.get("user_behavior", {}),
            "engagement_score": self._calculate_engagement_score(state)
        }
        self.user_history.append(user_record)
        
        # 保持历史记录在合理范围内
        if len(self.network_history) > 30:
            self.network_history = self.network_history[-30:]
        if len(self.user_history) > 30:
            self.user_history = self.user_history[-30:]
    
    def _calculate_engagement_score(self, state: Dict[str, Any]) -> float:
        """计算用户参与度分数"""
        user_behavior = state.get("user_behavior", {})
        
        # 计算各种参与度指标
        total_posts = user_behavior.get("posts_created", 0)
        total_likes = user_behavior.get("likes_given", 0)
        total_comments = user_behavior.get("comments_made", 0)
        total_shares = user_behavior.get("shares_made", 0)
        active_users = user_behavior.get("active_users", 0)
        
        # 综合参与度分数
        engagement_score = (
            total_posts * 0.3 +
            total_likes * 0.2 +
            total_comments * 0.3 +
            total_shares * 0.2
        ) * (active_users / 100.0)  # 归一化
        
        return min(1.0, engagement_score)
    
    def _construct_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构造决策提示（包含历史数据）"""
        network_state = state.get("network_state", {})
        user_behavior = state.get("user_behavior", {})
        content_metrics = state.get("content_metrics", {})
        
        # 构建网络状态摘要
        network_summary = []
        for user_id, user_data in network_state.items():
            network_summary.append(
                f"User {user_id}: "
                f"Followers={user_data.get('followers', 0)}, "
                f"Following={user_data.get('following', 0)}, "
                f"Posts={user_data.get('posts', 0)}, "
                f"Engagement={user_data.get('engagement_rate', 0):.2f}%"
            )
        
        # 构建用户行为摘要
        behavior_summary = f"""
Current User Behavior:
- Active Users: {user_behavior.get('active_users', 0)}
- Posts Created: {user_behavior.get('posts_created', 0)}
- Likes Given: {user_behavior.get('likes_given', 0)}
- Comments Made: {user_behavior.get('comments_made', 0)}
- Shares Made: {user_behavior.get('shares_made', 0)}
- Average Session Time: {user_behavior.get('avg_session_time', 0):.1f} minutes
"""
        
        # 构建内容指标摘要
        content_summary = f"""
Content Performance:
- Viral Posts: {content_metrics.get('viral_posts', 0)}
- Trending Topics: {content_metrics.get('trending_topics', 0)}
- Content Quality Score: {content_metrics.get('quality_score', 0):.2f}
- User Satisfaction: {content_metrics.get('satisfaction_score', 0):.2f}
"""
        
        # 构建历史决策摘要
        history_summary = ""
        if self.decision_history:
            recent_decisions = self.decision_history[-5:]  # 最近5个决策
            history_summary = "\nRecent Decision History:\n"
            for record in recent_decisions:
                decision = record["decision"]
                history_summary += f"- Step {record['step']}: {decision['action']} - {decision.get('reasoning', 'No reasoning')}\n"
        
        # 构建历史表现摘要
        performance_summary = ""
        if self.user_history:
            recent_performance = self.user_history[-5:]  # 最近5个表现记录
            performance_summary = "\nRecent Performance:\n"
            for record in recent_performance:
                performance_summary += f"- Step {record['step']}: Engagement Score = {record['engagement_score']:.3f}\n"
        
        # 构建完整提示
        prompt = f"""
You are a social network strategy expert in a simulation game. This is NOT real social media management - it's a simulation game for testing AI strategies.

IMPORTANT: You MUST respond in ENGLISH only. Do NOT use Chinese or any other language.

Your goal is to maximize user engagement and network growth through strategic content and user interaction decisions.

Current Network State:
{chr(10).join(network_summary)}

{behavior_summary}

{content_summary}

{history_summary}

{performance_summary}

Based on the current state and history, you must make a specific decision to improve the social network. You can choose from these actions:

1. CREATE_POST - Create engaging content to increase user activity
2. ENCOURAGE_INTERACTION - Promote likes, comments, and shares
3. FEATURE_USER - Highlight active users to increase engagement
4. LAUNCH_CAMPAIGN - Start a viral marketing campaign
5. IMPROVE_ALGORITHM - Optimize content recommendation algorithm
6. ADD_FEATURE - Introduce new social features
7. MODERATE_CONTENT - Improve content quality and safety
8. EXPAND_NETWORK - Grow user base through targeted outreach

CRITICAL REQUIREMENTS:
- You MUST respond in ENGLISH only
- You MUST provide a specific action with clear reasoning
- No vague responses, no cautious strategies, no holding back
- This is a simulation game - be bold and specific!
- You MUST use the exact format below

REQUIRED FORMAT (copy and fill in):
ACTION: [specific action name from the list above]
TARGET: [specific target if applicable, or "N/A"]
REASONING: [clear explanation of why this action will improve the network]

Example response:
ACTION: CREATE_POST
TARGET: N/A
REASONING: The network has low content creation activity, creating engaging posts will increase user engagement and attract more active users.

What specific action will you take to improve this social network? Respond in the exact format above.

IMPORTANT: Your response must start with "ACTION:" and follow the exact format shown above. Do not provide any other analysis or explanation outside of this format.
"""
        
        return prompt
    
    def _parse_decision_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM决策响应"""
        response = response.strip()
        
        print(f"🔍 解析响应: {response[:200]}...")  # 打印前200个字符用于调试
        
        # 尝试解析标准格式
        try:
            # 查找ACTION行 - 使用更宽松的正则表达式
            action_patterns = [
                r'ACTION:\s*([A-Z_]+)',  # 标准格式
                r'action:\s*([A-Z_]+)',  # 小写
                r'Action:\s*([A-Z_]+)',  # 首字母大写
                r'ACTION\s*:\s*([A-Z_]+)',  # 无冒号空格
                r'ACTION\s*=\s*([A-Z_]+)',  # 等号格式
            ]
            
            action = None
            for pattern in action_patterns:
                action_match = re.search(pattern, response, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).upper()
                    print(f"✅ 找到ACTION: {action}")
                    break
            
            if not action:
                print("❌ 未找到ACTION字段")
                return None
            
            # 查找TARGET行 - 使用更宽松的正则表达式
            target_patterns = [
                r'TARGET:\s*(.+?)(?:\n|$)',  # 标准格式
                r'target:\s*(.+?)(?:\n|$)',  # 小写
                r'Target:\s*(.+?)(?:\n|$)',  # 首字母大写
                r'TARGET\s*:\s*(.+?)(?:\n|$)',  # 无冒号空格
                r'TARGET\s*=\s*(.+?)(?:\n|$)',  # 等号格式
            ]
            
            target = "N/A"
            for pattern in target_patterns:
                target_match = re.search(pattern, response, re.IGNORECASE)
                if target_match:
                    target = target_match.group(1).strip()
                    print(f"✅ 找到TARGET: {target}")
                    break
            
            # 查找REASONING行 - 使用更宽松的正则表达式
            reasoning_patterns = [
                r'REASONING:\s*(.+?)(?:\n|$)',  # 标准格式
                r'reasoning:\s*(.+?)(?:\n|$)',  # 小写
                r'Reasoning:\s*(.+?)(?:\n|$)',  # 首字母大写
                r'REASONING\s*:\s*(.+?)(?:\n|$)',  # 无冒号空格
                r'REASONING\s*=\s*(.+?)(?:\n|$)',  # 等号格式
            ]
            
            reasoning = "No reasoning provided"
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, response, re.IGNORECASE)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    print(f"✅ 找到REASONING: {reasoning[:50]}...")
                    break
            
            # 验证动作是否有效
            valid_actions = [
                "CREATE_POST", "ENCOURAGE_INTERACTION", "FEATURE_USER", 
                "LAUNCH_CAMPAIGN", "IMPROVE_ALGORITHM", "ADD_FEATURE", 
                "MODERATE_CONTENT", "EXPAND_NETWORK"
            ]
            
            if action not in valid_actions:
                print(f"❌ 无效的ACTION: {action}")
                return None
            
            print(f"✅ 解析成功: {action} | {target} | {reasoning[:30]}...")
            
            return {
                "action": action,
                "target": target,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"❌ Decision parsing failed: {e}")
            return None
    
    def _parse_decision_fallback(self, response: str) -> Optional[Dict[str, Any]]:
        """备用决策解析逻辑"""
        # 尝试从响应中提取任何可能的动作
        response_upper = response.upper()
        
        # 检查是否包含任何有效动作
        valid_actions = [
            "CREATE_POST", "ENCOURAGE_INTERACTION", "FEATURE_USER", 
            "LAUNCH_CAMPAIGN", "IMPROVE_ALGORITHM", "ADD_FEATURE", 
            "MODERATE_CONTENT", "EXPAND_NETWORK"
        ]
        
        for action in valid_actions:
            if action in response_upper:
                return {
                    "action": action,
                    "target": "N/A",
                    "reasoning": f"Extracted action '{action}' from response"
                }
        
        # 如果没有找到有效动作，返回None
        return None


def create_rl_social_workflow(llm_manager) -> tuple[SG_Workflow, RLTrainer, LLMDecisionMaker]:
    """创建基于RL的LLM决策社交网络工作流"""
    
    # 创建RL配置
    rl_config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=32,
        mini_batch_size=8,
        ppo_epochs=4,
        target_kl=0.01
    )
    
    # 创建RL训练器
    rl_trainer = RLTrainer(rl_config, llm_manager)
    
    # 创建LLM决策器
    decision_maker = LLMDecisionMaker(llm_manager)
    
    # 创建工作流
    workflow = SG_Workflow("rl_social_workflow", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建社交网络沙盒
    sandbox = SocialNetworkSandbox(
        initial_users=100,
        max_users=1000
    )
    
    # 创建社交网络环境节点
    def social_env_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """社交网络环境节点函数"""
        # 获取当前状态
        case = sandbox.case_generator()
        current_state = case["state"]
        
        # 使用LLM做出决策
        decision_result = decision_maker.make_decision(current_state)
        decision = decision_result["decision"]  # 从结果中提取决策
        
        # 执行社交网络决策
        try:
            # 验证和执行决策
            score = sandbox.verify_score(
                f"{decision['action']} {decision.get('target', '')}",
                case
            )
            
            # 计算奖励
            reward = score * 10
            
            # 构建状态特征
            state_features = {
                "active_users": current_state["user_behavior"]["active_users"],
                "engagement_rate": _calculate_engagement_rate(current_state),
                "content_quality": current_state["content_metrics"]["quality_score"],
                "network_growth": _calculate_network_growth(current_state),
                "decision_type": _encode_decision_type(decision["action"])
            }
            
            # 添加到RL训练器
            rl_trainer.add_experience(
                state=state_features,
                action=json.dumps(decision),
                reward=reward,
                done=False
            )
            
            # 更新策略
            update_result = rl_trainer.update_policy()
            
            return {
                "state": current_state,
                "decision": decision,
                "llm_response": decision_result["llm_response"],
                "score": score,
                "reward": reward,
                "rl_update": update_result,
                "sandbox_id": sandbox.sandbox_id
            }
            
        except Exception as e:
            print(f"社交网络执行错误: {e}")
            return {
                "state": current_state,
                "decision": {"action": "CREATE_POST", "reasoning": f"执行错误: {e}"},
                "score": 0.0,
                "reward": 0.0,
                "error": str(e)
            }
    
    # 添加社交网络环境节点
    social_env_node = EnhancedWorkflowNode(
        "social_environment",
        NodeType.SANDBOX,
        sandbox=sandbox,
        condition=NodeCondition(),
        limits=NodeLimits(max_visits=10, resource_cost={"energy": 10, "tokens": 5})
    )
    workflow.add_node(social_env_node)
    
    return workflow, rl_trainer, decision_maker


def _calculate_engagement_rate(state: Dict[str, Any]) -> float:
    """计算用户参与率"""
    user_behavior = state.get("user_behavior", {})
    active_users = user_behavior.get("active_users", 0)
    total_users = len(state.get("network_state", {}))
    
    if total_users == 0:
        return 0.0
    
    return active_users / total_users


def _calculate_network_growth(state: Dict[str, Any]) -> float:
    """计算网络增长率"""
    network_state = state.get("network_state", {})
    total_users = len(network_state)
    
    # 基于用户数量和连接度计算增长率
    total_connections = 0
    for user_data in network_state.values():
        total_connections += user_data.get("followers", 0) + user_data.get("following", 0)
    
    if total_users == 0:
        return 0.0
    
    avg_connections = total_connections / total_users
    return min(1.0, avg_connections / 100.0)  # 归一化


def _encode_decision_type(action: str) -> int:
    """编码决策类型"""
    action_map = {
        "CREATE_POST": 1,
        "ENCOURAGE_INTERACTION": 2,
        "FEATURE_USER": 3,
        "LAUNCH_CAMPAIGN": 4,
        "IMPROVE_ALGORITHM": 5,
        "ADD_FEATURE": 6,
        "MODERATE_CONTENT": 7,
        "EXPAND_NETWORK": 8
    }
    return action_map.get(action, 0)


def run_rl_social_demo(steps: int = 5, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    """运行基于RL的LLM决策社交网络演示"""
    
    print_section("RL-based LLM Decision Social Network Demo")
    
    # 1. 创建LLM管理器
    print(f"\n1. Creating LLM Manager with model: {model_name}")
    llm_manager = create_shared_llm_manager(
        model_name=model_name,
        backend="huggingface",
        temperature=0.7,
        max_length=512,
        device="auto",
        torch_dtype="float16"
    )
    
    # 2. 创建工作流和RL训练器
    print("\n2. Creating RL Social Network Workflow")
    workflow, rl_trainer, decision_maker = create_rl_social_workflow(llm_manager)
    
    # 3. 执行多步社交网络管理
    print(f"\n3. Executing {steps} Social Network Management Steps")
    
    results = []
    for step in range(steps):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        try:
            # 直接执行社交网络环境节点
            node = workflow.nodes.get("social_environment")
            if node and node.sandbox:
                # 获取当前状态
                case = node.sandbox.case_generator()
                current_state = case["state"]
                
                # 使用LLM做出决策
                decision_result = decision_maker.make_decision(current_state)
                decision = decision_result["decision"]  # 从结果中提取决策
                
                # 执行社交网络决策
                try:
                    # 验证和执行决策
                    score = node.sandbox.verify_score(
                        f"{decision['action']} {decision.get('target', '')}",
                        case
                    )
                    
                    # 计算奖励
                    reward = score * 10
                    
                    # 构建状态特征
                    state_features = {
                        "active_users": current_state["user_behavior"]["active_users"],
                        "engagement_rate": _calculate_engagement_rate(current_state),
                        "content_quality": current_state["content_metrics"]["quality_score"],
                        "network_growth": _calculate_network_growth(current_state),
                        "decision_type": _encode_decision_type(decision["action"])
                    }
                    
                    # 添加到RL训练器
                    rl_trainer.add_experience(
                        state=state_features,
                        action=json.dumps(decision),
                        reward=reward,
                        done=False
                    )
                    
                    # 更新策略
                    update_result = rl_trainer.update_policy()
                    
                    result = {
                        "step": step + 1,
                        "state": current_state,
                        "decision": decision,
                        "llm_response": decision_result["llm_response"],
                        "score": score,
                        "reward": reward,
                        "rl_update": update_result
                    }
                    
                    results.append(result)
                    
                    # 打印结果
                    print(f"Decision: {decision['action']}")
                    print(f"Target: {decision.get('target', 'N/A')}")
                    print(f"Reasoning: {decision['reasoning']}")
                    print(f"Score: {score:.2f}")
                    print(f"Reward: {reward:.2f}")
                    print(f"Active Users: {current_state['user_behavior']['active_users']}")
                    print(f"Engagement Rate: {_calculate_engagement_rate(current_state):.3f}")
                    
                except Exception as e:
                    print(f"执行错误: {e}")
                    result = {
                        "step": step + 1,
                        "error": str(e),
                        "decision": {"action": "CREATE_POST", "reasoning": f"执行错误: {e}"}
                    }
                    results.append(result)
            
        except Exception as e:
            print(f"步骤 {step + 1} 执行失败: {e}")
            result = {
                "step": step + 1,
                "error": str(e)
            }
            results.append(result)
    
    # 4. 显示最终结果
    print_section("Final Results")
    
    successful_steps = [r for r in results if "error" not in r]
    if successful_steps:
        avg_score = sum(r["score"] for r in successful_steps) / len(successful_steps)
        avg_reward = sum(r["reward"] for r in successful_steps) / len(successful_steps)
        final_engagement = _calculate_engagement_rate(successful_steps[-1]["state"])
        
        print(f"Average Score: {avg_score:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Final Engagement Rate: {final_engagement:.3f}")
        print(f"Successful Steps: {len(successful_steps)}/{steps}")
    
    # 5. 显示RL训练统计
    print_section("RL Training Statistics")
    training_stats = rl_trainer.get_training_stats()
    print(f"Training Steps: {training_stats['training_step']}")
    print(f"Algorithm: {training_stats['algorithm']}")
    print(f"Recent Updates: {len(training_stats['recent_updates'])}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SandGraph Social Network Demo")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM model to use")
    
    args = parser.parse_args()
    
    print("🔥 SandGraph Social Network Demo")
    print("=" * 60)
    print(f"Steps: {args.steps}")
    print(f"Model: {args.model}")
    
    # 支持的模型列表
    supported_models = [
        "Qwen/Qwen-7B-Chat",
        "Qwen/Qwen-1_8B-Chat", 
        "microsoft/Phi-2",
        "google/gemma-2b-it",
        "01-ai/Yi-6B-Chat",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "THUDM/chatglm3-6b",
        "baichuan-inc/Baichuan2-7B-Chat"
    ]
    
    if args.model not in supported_models:
        print(f"\n⚠️  Warning: Model {args.model} not in supported list.")
        print(f"Supported models: {', '.join(supported_models)}")
        print("Continuing anyway...")
    
    try:
        results = run_rl_social_demo(args.steps, args.model)
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
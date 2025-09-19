#!/usr/bin/env python3
"""
Sandbox-RL 社交网络环境演示 - 基于RL的LLM决策架构

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

from sandbox_rl.core.llm_interface import create_shared_llm_manager
from sandbox_rl.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandbox_rl.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from sandbox_rl.sandbox_implementations import SocialNetworkSandbox


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
            # 生成LLM响应 - 添加系统消息强制格式
            system_message = """You are a social network strategy expert. You MUST respond with EXACTLY 3 lines in this format:
ACTION: [ACTION_NAME]
TARGET: [TARGET_VALUE] 
REASONING: [REASONING_TEXT]

Do not add any other text, explanations, or formatting. Only the 3 lines above."""
            
            response = self.llm_manager.generate_for_node(
                "social_decision",
                prompt,
                temperature=0.3,  # 降低温度以获得更一致的格式
                max_new_tokens=128,  # 减少token数量
                do_sample=True,
                system_message=system_message
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
        network_dynamics = state.get("network_dynamics", {})
        
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
- Bounce Rate: {user_behavior.get('bounce_rate', 0):.2f}
- Retention Rate: {user_behavior.get('retention_rate', 0):.2f}
"""
        
        # 构建内容指标摘要
        content_summary = f"""
Content Performance:
- Viral Posts: {content_metrics.get('viral_posts', 0)}
- Trending Topics: {content_metrics.get('trending_topics', 0)}
- Content Quality Score: {content_metrics.get('quality_score', 0):.2f}
- User Satisfaction: {content_metrics.get('satisfaction_score', 0):.2f}
- Content Diversity: {content_metrics.get('diversity_score', 0):.2f}
- Controversy Level: {content_metrics.get('controversy_level', 0):.2f}
"""
        
        # 构建网络动态摘要
        dynamics_summary = f"""
Network Dynamics:
- Network Mood: {network_dynamics.get('mood', 0):.2f} (-1=Negative, 1=Positive)
- Competition Level: {network_dynamics.get('competition_level', 0):.2f}
- Innovation Rate: {network_dynamics.get('innovation_rate', 0):.2f}
- Crisis Level: {network_dynamics.get('crisis_level', 0):.2f}
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
        
        # 构建策略建议
        strategy_advice = self._generate_strategy_advice(state)
        
        # 重构后的增强提示 - 更严格的格式要求
        prompt = f"""You are a social network strategy expert. You MUST respond with EXACTLY 3 lines:

ACTION: [CREATE_POST|ENCOURAGE_INTERACTION|FEATURE_USER|LAUNCH_CAMPAIGN|IMPROVE_ALGORITHM|ADD_FEATURE|MODERATE_CONTENT|EXPAND_NETWORK]
TARGET: [specific target or "N/A"]
REASONING: [brief explanation]

Current State:
- Active Users: {user_behavior.get('active_users', 0)}
- Posts Created: {user_behavior.get('posts_created', 0)}
- Content Quality: {content_metrics.get('quality_score', 0):.2f}
- Network Mood: {network_dynamics.get('mood', 0):.2f}
- Crisis Level: {network_dynamics.get('crisis_level', 0):.2f}
- Bounce Rate: {user_behavior.get('bounce_rate', 0):.2f}

Strategy Guidelines:
- Negative mood → LAUNCH_CAMPAIGN or ENCOURAGE_INTERACTION
- High crisis → MODERATE_CONTENT or LAUNCH_CAMPAIGN  
- Low innovation → ADD_FEATURE or IMPROVE_ALGORITHM
- High controversy → MODERATE_CONTENT
- High bounce rate → IMPROVE_ALGORITHM

Respond with EXACTLY 3 lines in the format above. No other text."""
        
        return prompt
    
    def _generate_strategy_advice(self, state: Dict[str, Any]) -> str:
        """根据当前状态生成策略建议"""
        network_dynamics = state.get("network_dynamics", {})
        user_behavior = state.get("user_behavior", {})
        content_metrics = state.get("content_metrics", {})
        
        advice = []
        
        # 基于网络情绪的建议
        mood = network_dynamics.get("mood", 0)
        if mood < -0.3:
            advice.append("⚠️ Network mood is negative - consider LAUNCH_CAMPAIGN or ENCOURAGE_INTERACTION to boost morale")
        elif mood > 0.3:
            advice.append("✅ Network mood is positive - good time for EXPAND_NETWORK or ADD_FEATURE")
        
        # 基于危机程度的建议
        crisis_level = network_dynamics.get("crisis_level", 0)
        if crisis_level > 0.4:
            advice.append("🚨 Crisis detected - prioritize MODERATE_CONTENT or LAUNCH_CAMPAIGN to address issues")
        
        # 基于争议程度的建议
        controversy_level = content_metrics.get("controversy_level", 0)
        if controversy_level > 0.4:
            advice.append("⚠️ High controversy - use MODERATE_CONTENT to control the situation")
        
        # 基于创新率的建议
        innovation_rate = network_dynamics.get("innovation_rate", 0)
        if innovation_rate < 0.3:
            advice.append("💡 Low innovation - consider ADD_FEATURE or IMPROVE_ALGORITHM to drive innovation")
        
        # 基于用户体验的建议
        bounce_rate = user_behavior.get("bounce_rate", 0)
        if bounce_rate > 0.5:
            advice.append("📉 High bounce rate - use IMPROVE_ALGORITHM to improve user experience")
        
        # 基于内容质量的建议
        quality_score = content_metrics.get("quality_score", 0)
        if quality_score < 0.6:
            advice.append("📝 Low content quality - consider CREATE_POST or IMPROVE_ALGORITHM")
        
        # 基于用户参与的建议
        active_users = user_behavior.get("active_users", 0)
        if active_users < 50:
            advice.append("👥 Low active users - consider FEATURE_USER or ENCOURAGE_INTERACTION")
        
        if not advice:
            advice.append("✅ Network is stable - any action can help maintain growth")
        
        return "\n".join(advice)
    
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
                r'^([A-Z_]+)\s*$',  # 单独一行
            ]
            
            action = None
            for pattern in action_patterns:
                action_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
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
                target_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
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
                reasoning_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
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
        
        # 按优先级排序动作（基于当前状态）
        action_priority = []
        for action in valid_actions:
            if action in response_upper:
                action_priority.append(action)
        
        if action_priority:
            # 选择第一个找到的动作
            selected_action = action_priority[0]
            
            # 尝试从响应中提取一些上下文作为推理
            reasoning = "Action extracted from response"
            
            # 查找包含动作的句子
            sentences = response.split('.')
            for sentence in sentences:
                if selected_action.lower().replace('_', ' ') in sentence.lower():
                    reasoning = sentence.strip()
                    break
            
            return {
                "action": selected_action,
                "target": "N/A",
                "reasoning": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
            }
        
        # 如果没有找到有效动作，返回None
        return None


def create_rl_social_workflow(llm_manager) -> tuple[SG_Workflow, RLTrainer, LLMDecisionMaker]:
    """创建基于RL的LLM决策社交网络工作流"""
    
    # 创建RL配置 - 减小batch size以在少量步骤后开始训练
    rl_config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=4,  # 从32减小到4
        mini_batch_size=2,  # 从8减小到2
        ppo_epochs=2,  # 从4减小到2
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
            
            # 显示RL更新状态
            print(f"RL Update Status: {update_result.get('status', 'unknown')}")
            if update_result.get('status') == 'insufficient_data':
                print(f"  Trajectory Count: {update_result.get('trajectory_count', 0)}")
                print(f"  Required Batch Size: {update_result.get('required_batch_size', 0)}")
            elif update_result.get('status') == 'updated':
                print(f"  Training Step: {update_result.get('training_step', 0)}")
                print(f"  Algorithm: {update_result.get('algorithm', 'unknown')}")
            
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
    network_state = state.get("network_state", {})
    
    # 获取基础数据
    active_users = user_behavior.get("active_users", 0)
    total_users = len(network_state)
    
    if total_users == 0:
        return 0.0
    
    # 计算基础参与率
    base_engagement = active_users / total_users
    
    # 考虑用户行为质量
    posts_created = user_behavior.get("posts_created", 0)
    likes_given = user_behavior.get("likes_given", 0)
    comments_made = user_behavior.get("comments_made", 0)
    shares_made = user_behavior.get("shares_made", 0)
    
    # 计算互动质量分数
    total_interactions = posts_created + likes_given + comments_made + shares_made
    interaction_quality = min(1.0, total_interactions / (total_users * 10))  # 每个用户平均10次互动为满分
    
    # 考虑会话时间和留存率
    avg_session_time = user_behavior.get("avg_session_time", 0)
    retention_rate = user_behavior.get("retention_rate", 0)
    bounce_rate = user_behavior.get("bounce_rate", 0)
    
    # 会话质量分数
    session_quality = min(1.0, avg_session_time / 30.0)  # 30分钟为满分
    
    # 留存质量分数
    retention_quality = retention_rate * (1 - bounce_rate)
    
    # 综合参与率计算
    engagement_rate = (
        base_engagement * 0.4 +           # 基础参与率权重40%
        interaction_quality * 0.3 +       # 互动质量权重30%
        session_quality * 0.2 +           # 会话质量权重20%
        retention_quality * 0.1           # 留存质量权重10%
    )
    
    return min(1.0, max(0.0, engagement_rate))


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
    parser = argparse.ArgumentParser(description="Sandbox-RL Social Network Demo")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="LLM model to use")
    
    args = parser.parse_args()
    
    print("🔥 Sandbox-RL Social Network Demo")
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
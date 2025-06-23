#!/usr/bin/env python3
"""
SandGraph 虚假信息传播演示 - 基于RL的LLM决策架构

模拟虚假信息在社交网络中的传播：
1. 信息传播机制
2. 虚假信息检测
3. 干预策略
4. 用户行为建模
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
from enum import Enum

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm


class InformationType(Enum):
    """信息类型"""
    TRUE = "true"
    FALSE = "false"
    MISLEADING = "misleading"
    UNVERIFIED = "unverified"


class UserBelief(Enum):
    """用户信念状态"""
    BELIEVER = "believer"      # 相信
    SKEPTIC = "skeptic"        # 怀疑
    NEUTRAL = "neutral"        # 中立
    DISBELIEVER = "disbeliever" # 不相信


class InterventionType(Enum):
    """干预类型"""
    FACT_CHECK = "fact_check"           # 事实核查
    WARNING_LABEL = "warning_label"     # 警告标签
    DOWNRANK = "downrank"               # 降权
    REMOVE = "remove"                   # 删除
    EDUCATE = "educate"                 # 教育用户
    PROMOTE_TRUTH = "promote_truth"     # 推广真实信息


class MisinformationSandbox:
    """虚假信息传播沙盒"""
    
    def __init__(self, 
                 initial_users: int = 100,
                 max_users: int = 1000,
                 initial_misinfo_count: int = 5,
                 spread_probability: float = 0.3,
                 belief_change_prob: float = 0.2):
        
        self.sandbox_id = f"misinfo_spread_{int(time.time())}"
        self.initial_users = initial_users
        self.max_users = max_users
        self.initial_misinfo_count = initial_misinfo_count
        self.spread_probability = spread_probability
        self.belief_change_prob = belief_change_prob
        
        # 网络状态
        self.users = {}
        self.connections = []
        self.information_pieces = []
        self.spread_history = []
        self.interventions = []
        
        # 初始化网络
        self._initialize_network()
        self._initialize_misinformation()
    
    def _initialize_network(self):
        """初始化社交网络"""
        # 创建用户
        for i in range(self.initial_users):
            user_id = f"user_{i}"
            self.users[user_id] = {
                "id": user_id,
                "followers": [],
                "following": [],
                "posts": [],
                "belief_state": random.choice(list(UserBelief)),
                "credibility": random.uniform(0.1, 1.0),
                "susceptibility": random.uniform(0.1, 1.0),  # 对虚假信息的易感性
                "fact_checking_habit": random.uniform(0.0, 1.0),  # 事实核查习惯
                "created_at": datetime.now()
            }
        
        # 创建连接
        for user_id in self.users:
            # 每个用户随机关注5-15个其他用户
            following_count = random.randint(5, 15)
            potential_follows = [uid for uid in self.users if uid != user_id]
            follows = random.sample(potential_follows, min(following_count, len(potential_follows)))
            
            for follow_id in follows:
                self.users[user_id]["following"].append(follow_id)
                self.users[follow_id]["followers"].append(user_id)
                self.connections.append((user_id, follow_id))
    
    def _initialize_misinformation(self):
        """初始化虚假信息"""
        misinfo_templates = [
            {
                "title": "震惊！科学家发现新的健康风险",
                "content": "最新研究表明，日常用品中存在未知的健康风险...",
                "type": InformationType.FALSE,
                "virality_factor": 0.8,
                "credibility_impact": -0.3
            },
            {
                "title": "政府隐瞒重要信息",
                "content": "据内部消息，政府正在隐瞒一项重要发现...",
                "type": InformationType.MISLEADING,
                "virality_factor": 0.9,
                "credibility_impact": -0.4
            },
            {
                "title": "特效药治愈率100%",
                "content": "某公司声称其新药治愈率达到100%...",
                "type": InformationType.FALSE,
                "virality_factor": 0.7,
                "credibility_impact": -0.5
            },
            {
                "title": "外星人接触证据曝光",
                "content": "匿名人士提供的照片显示外星人接触证据...",
                "type": InformationType.UNVERIFIED,
                "virality_factor": 0.6,
                "credibility_impact": -0.2
            },
            {
                "title": "经济崩溃预警",
                "content": "专家预测经济即将崩溃，建议立即行动...",
                "type": InformationType.MISLEADING,
                "virality_factor": 0.85,
                "credibility_impact": -0.35
            }
        ]
        
        for i in range(self.initial_misinfo_count):
            template = random.choice(misinfo_templates)
            info_id = f"misinfo_{i}"
            
            # 随机选择一个用户作为信息源
            source_user = random.choice(list(self.users.keys()))
            
            self.information_pieces.append({
                "id": info_id,
                "title": template["title"],
                "content": template["content"],
                "type": template["type"],
                "source_user": source_user,
                "virality_factor": template["virality_factor"],
                "credibility_impact": template["credibility_impact"],
                "spread_count": 0,
                "believers": [],
                "skeptics": [],
                "created_at": datetime.now(),
                "is_verified": False,
                "verification_score": 0.0
            })
            
            # 初始传播
            self._spread_information(info_id, source_user)
    
    def _spread_information(self, info_id: str, source_user: str):
        """传播信息"""
        info = next((i for i in self.information_pieces if i["id"] == info_id), None)
        if not info:
            return
        
        # 获取源用户的关注者
        followers = self.users[source_user]["followers"]
        
        for follower_id in followers:
            # 计算传播概率
            base_prob = self.spread_probability
            user_susceptibility = self.users[follower_id]["susceptibility"]
            virality = info["virality_factor"]
            
            spread_prob = base_prob * user_susceptibility * virality
            
            if random.random() < spread_prob:
                # 信息传播成功
                info["spread_count"] += 1
                
                # 更新用户信念
                self._update_user_belief(follower_id, info)
                
                # 记录传播历史
                self.spread_history.append({
                    "info_id": info_id,
                    "from_user": source_user,
                    "to_user": follower_id,
                    "timestamp": datetime.now(),
                    "belief_state": self.users[follower_id]["belief_state"]
                })
    
    def _update_user_belief(self, user_id: str, info: Dict[str, Any]):
        """更新用户信念状态"""
        user = self.users[user_id]
        current_belief = user["belief_state"]
        
        # 基于信息类型和用户特征计算信念变化概率
        if info["type"] == InformationType.FALSE:
            change_prob = self.belief_change_prob * user["susceptibility"]
        elif info["type"] == InformationType.MISLEADING:
            change_prob = self.belief_change_prob * user["susceptibility"] * 0.8
        else:  # UNVERIFIED
            change_prob = self.belief_change_prob * user["susceptibility"] * 0.5
        
        if random.random() < change_prob:
            # 信念发生变化
            if current_belief == UserBelief.NEUTRAL:
                new_belief = UserBelief.BELIEVER if random.random() < 0.6 else UserBelief.SKEPTIC
            elif current_belief == UserBelief.SKEPTIC:
                new_belief = UserBelief.BELIEVER if random.random() < 0.3 else UserBelief.NEUTRAL
            elif current_belief == UserBelief.BELIEVER:
                new_belief = UserBelief.SKEPTIC if random.random() < 0.2 else UserBelief.NEUTRAL
            else:  # DISBELIEVER
                new_belief = UserBelief.SKEPTIC if random.random() < 0.4 else UserBelief.NEUTRAL
            
            user["belief_state"] = new_belief
            
            # 更新信息统计
            if new_belief == UserBelief.BELIEVER:
                info["believers"].append(user_id)
            elif new_belief == UserBelief.SKEPTIC:
                info["skeptics"].append(user_id)
    
    def case_generator(self) -> Dict[str, Any]:
        """生成当前状态"""
        # 计算统计信息
        total_users = len(self.users)
        total_connections = len(self.connections)
        total_misinfo = len(self.information_pieces)
        
        # 计算传播统计
        total_spreads = sum(info["spread_count"] for info in self.information_pieces)
        total_believers = sum(len(info["believers"]) for info in self.information_pieces)
        total_skeptics = sum(len(info["skeptics"]) for info in self.information_pieces)
        
        # 计算信念分布
        belief_distribution = {}
        for belief in UserBelief:
            count = sum(1 for user in self.users.values() if user["belief_state"] == belief)
            belief_distribution[belief.value] = count
        
        # 计算干预效果
        intervention_stats = {
            "total_interventions": len(self.interventions),
            "successful_interventions": sum(1 for i in self.interventions if i.get("success", False)),
            "recent_interventions": len([i for i in self.interventions if (datetime.now() - i["timestamp"]).seconds < 3600])
        }
        
        return {
            "state": {
                "network_state": {
                    "total_users": total_users,
                    "total_connections": total_connections,
                    "avg_followers": total_connections / total_users if total_users > 0 else 0,
                    "network_density": total_connections / (total_users * (total_users - 1)) if total_users > 1 else 0
                },
                "misinformation_state": {
                    "total_pieces": total_misinfo,
                    "total_spreads": total_spreads,
                    "avg_spreads_per_piece": total_spreads / total_misinfo if total_misinfo > 0 else 0,
                    "believers_count": total_believers,
                    "skeptics_count": total_skeptics,
                    "belief_ratio": total_believers / (total_believers + total_skeptics) if (total_believers + total_skeptics) > 0 else 0
                },
                "user_beliefs": belief_distribution,
                "intervention_stats": intervention_stats,
                "recent_spreads": self.spread_history[-10:] if self.spread_history else [],
                "active_misinfo": [info for info in self.information_pieces if info["spread_count"] > 0]
            },
            "metadata": {
                "sandbox_id": self.sandbox_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def verify_score(self, action: str, case: Dict[str, Any]) -> float:
        """验证干预行动的效果"""
        try:
            # 解析行动
            action_parts = action.split()
            if len(action_parts) < 2:
                return 0.0
            
            intervention_type = action_parts[0].upper()
            target = action_parts[1] if len(action_parts) > 1 else "general"
            
            # 基础分数
            base_score = 0.0
            
            # 根据干预类型评分
            if intervention_type == "FACT_CHECK":
                base_score = 0.7
            elif intervention_type == "WARNING_LABEL":
                base_score = 0.6
            elif intervention_type == "DOWNRANK":
                base_score = 0.5
            elif intervention_type == "REMOVE":
                base_score = 0.8
            elif intervention_type == "EDUCATE":
                base_score = 0.4
            elif intervention_type == "PROMOTE_TRUTH":
                base_score = 0.6
            else:
                base_score = 0.3
            
            # 根据目标调整分数
            if target == "high_spread":
                base_score *= 1.2
            elif target == "high_belief":
                base_score *= 1.1
            elif target == "general":
                base_score *= 1.0
            
            # 考虑当前状态
            current_state = case["state"]
            belief_ratio = current_state["misinformation_state"]["belief_ratio"]
            
            # 如果信念比例高，干预效果更好
            if belief_ratio > 0.5:
                base_score *= 1.3
            
            # 限制分数范围
            return min(1.0, max(0.0, base_score))
            
        except Exception as e:
            print(f"评分错误: {e}")
            return 0.0
    
    def execute_intervention(self, intervention_type: str, target: str = "general") -> Dict[str, Any]:
        """执行干预行动"""
        intervention = {
            "type": intervention_type,
            "target": target,
            "timestamp": datetime.now(),
            "success": False,
            "impact": {}
        }
        
        # 模拟干预效果
        if intervention_type == "FACT_CHECK":
            # 事实核查：降低虚假信息的传播
            for info in self.information_pieces:
                if info["type"] in [InformationType.FALSE, InformationType.MISLEADING]:
                    info["verification_score"] += 0.3
                    info["virality_factor"] *= 0.8
            
            intervention["success"] = True
            intervention["impact"] = {"verification_improved": True, "virality_reduced": True}
            
        elif intervention_type == "WARNING_LABEL":
            # 警告标签：增加用户怀疑
            for user_id in self.users:
                if self.users[user_id]["belief_state"] == UserBelief.BELIEVER:
                    if random.random() < 0.3:
                        self.users[user_id]["belief_state"] = UserBelief.SKEPTIC
            
            intervention["success"] = True
            intervention["impact"] = {"skepticism_increased": True}
            
        elif intervention_type == "DOWNRANK":
            # 降权：减少信息可见性
            for info in self.information_pieces:
                info["virality_factor"] *= 0.6
            
            intervention["success"] = True
            intervention["impact"] = {"visibility_reduced": True}
            
        elif intervention_type == "REMOVE":
            # 删除：移除虚假信息
            self.information_pieces = [info for info in self.information_pieces 
                                     if info["type"] == InformationType.TRUE]
            
            intervention["success"] = True
            intervention["impact"] = {"misinfo_removed": True}
            
        elif intervention_type == "EDUCATE":
            # 教育：提高用户事实核查习惯
            for user_id in self.users:
                self.users[user_id]["fact_checking_habit"] = min(1.0, 
                    self.users[user_id]["fact_checking_habit"] + 0.1)
            
            intervention["success"] = True
            intervention["impact"] = {"education_improved": True}
            
        elif intervention_type == "PROMOTE_TRUTH":
            # 推广真实信息
            # 创建新的真实信息
            true_info = {
                "id": f"truth_{len(self.information_pieces)}",
                "title": "事实核查：澄清虚假信息",
                "content": "经过专业核查，之前传播的信息存在误导性...",
                "type": InformationType.TRUE,
                "source_user": "platform",
                "virality_factor": 0.5,
                "credibility_impact": 0.3,
                "spread_count": 0,
                "believers": [],
                "skeptics": [],
                "created_at": datetime.now(),
                "is_verified": True,
                "verification_score": 1.0
            }
            self.information_pieces.append(true_info)
            
            intervention["success"] = True
            intervention["impact"] = {"truth_promoted": True}
        
        # 记录干预
        self.interventions.append(intervention)
        
        return intervention


class LLMDecisionMaker:
    """LLM决策器 - 虚假信息干预专家"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.decision_count = 0
        
        # 历史数据管理
        self.decision_history = []
        self.intervention_history = []
        self.spread_history = []
        
        # 注册决策节点
        self.llm_manager.register_node("misinfo_intervention", {
            "role": "虚假信息干预专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
    
    def make_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态做出干预决策"""
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
                "misinfo_intervention",
                prompt,
                temperature=0.7,
                max_new_tokens=256,
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
                    "action": "FACT_CHECK",
                    "target": "general",
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
                "action": "FACT_CHECK",
                "target": "general", 
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
            "misinformation_state": state.get("misinformation_state", {}),
            "user_beliefs": state.get("user_beliefs", {})
        }
        self.decision_history.append(decision_record)
        
        # 保持历史记录在合理范围内
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]
    
    def _construct_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构造决策提示"""
        network_state = state.get("network_state", {})
        misinformation_state = state.get("misinformation_state", {})
        user_beliefs = state.get("user_beliefs", {})
        intervention_stats = state.get("intervention_stats", {})
        
        # 构建网络状态摘要
        network_summary = f"""
Network Status:
- Total Users: {network_state.get('total_users', 0)}
- Total Connections: {network_state.get('total_connections', 0)}
- Network Density: {network_state.get('network_density', 0):.3f}
"""
        
        # 构建虚假信息状态摘要
        misinfo_summary = f"""
Misinformation Status:
- Total Pieces: {misinformation_state.get('total_pieces', 0)}
- Total Spreads: {misinformation_state.get('total_spreads', 0)}
- Believers: {misinformation_state.get('believers_count', 0)}
- Skeptics: {misinformation_state.get('skeptics_count', 0)}
- Belief Ratio: {misinformation_state.get('belief_ratio', 0):.3f}
"""
        
        # 构建用户信念分布摘要
        belief_summary = f"""
User Belief Distribution:
- Believers: {user_beliefs.get('believer', 0)}
- Skeptics: {user_beliefs.get('skeptic', 0)}
- Neutral: {user_beliefs.get('neutral', 0)}
- Disbelievers: {user_beliefs.get('disbeliever', 0)}
"""
        
        # 构建干预历史摘要
        history_summary = ""
        if self.decision_history:
            recent_decisions = self.decision_history[-3:]  # 最近3个决策
            history_summary = "\nRecent Interventions:\n"
            for record in recent_decisions:
                decision = record["decision"]
                history_summary += f"- Step {record['step']}: {decision['action']} - {decision.get('reasoning', '')[:30]}...\n"
        
        # 重构后的简洁提示
        prompt = f"""You are a misinformation intervention expert in a social network simulation.

REQUIRED RESPONSE FORMAT:
ACTION: [FACT_CHECK|WARNING_LABEL|DOWNRANK|REMOVE|EDUCATE|PROMOTE_TRUTH] [GENERAL|HIGH_SPREAD|HIGH_BELIEF]
REASONING: [brief explanation]

Available Actions:
1. FACT_CHECK - Verify and debunk false information
2. WARNING_LABEL - Add warning labels to misleading content
3. DOWNRANK - Reduce visibility of false information
4. REMOVE - Remove false information completely
5. EDUCATE - Improve user media literacy
6. PROMOTE_TRUTH - Promote verified true information

Available Targets:
- GENERAL: General intervention across all misinformation
- HIGH_SPREAD: Target high-spread misinformation specifically
- HIGH_BELIEF: Target information with many believers specifically

{network_summary.strip()}
{misinfo_summary.strip()}
{belief_summary.strip()}
{history_summary.strip()}

Choose the best intervention strategy to combat misinformation. Respond ONLY in the required format above."""
        
        return prompt
    
    def _parse_decision_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM决策响应"""
        response = response.strip()
        
        print(f"🔍 解析响应: {response[:200]}...")
        
        # 尝试解析标准格式
        try:
            # 查找ACTION行 - 支持多种格式
            action_patterns = [
                # 标准格式: ACTION: FACT_CHECK GENERAL
                r'ACTION:\s*([A-Z_]+)\s+([A-Z_]+)',
                # 带方括号格式: ACTION: EDUCATE [general]
                r'ACTION:\s*([A-Z_]+)\s+\[([A-Z_]+)\]',
                # 小写格式: action: fact_check general
                r'action:\s*([A-Z_]+)\s+([A-Z_]+)',
                # 首字母大写格式: Action: Fact_Check General
                r'Action:\s*([A-Z_]+)\s+([A-Z_]+)',
                # 无冒号空格格式: ACTION FACT_CHECK GENERAL
                r'ACTION\s+([A-Z_]+)\s+([A-Z_]+)',
                # 等号格式: ACTION=FACT_CHECK GENERAL
                r'ACTION\s*=\s*([A-Z_]+)\s+([A-Z_]+)',
            ]
            
            action = None
            target = None
            
            for pattern in action_patterns:
                action_match = re.search(pattern, response, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).upper()
                    target = action_match.group(2).upper()
                    print(f"✅ 找到ACTION: {action} {target}")
                    break
            
            if not action or not target:
                print("❌ 未找到完整的ACTION字段")
                return None
            
            # 验证动作是否有效
            valid_actions = [
                "FACT_CHECK", "WARNING_LABEL", "DOWNRANK", 
                "REMOVE", "EDUCATE", "PROMOTE_TRUTH"
            ]
            
            if action not in valid_actions:
                print(f"❌ 无效的ACTION: {action}")
                return None
            
            # 验证目标是否有效
            valid_targets = ["GENERAL", "HIGH_SPREAD", "HIGH_BELIEF"]
            if target not in valid_targets:
                print(f"⚠️  无效的TARGET: {target}，使用GENERAL")
                target = "GENERAL"
            
            # 查找REASONING行
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
            
            print(f"✅ 解析成功: {action} {target} | {reasoning[:30]}...")
            
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
            "FACT_CHECK", "WARNING_LABEL", "DOWNRANK", 
            "REMOVE", "EDUCATE", "PROMOTE_TRUTH"
        ]
        
        for action in valid_actions:
            if action in response_upper:
                return {
                    "action": action,
                    "target": "GENERAL",
                    "reasoning": f"Extracted action '{action}' from response"
                }
        
        # 如果没有找到有效动作，返回None
        return None


def create_rl_misinfo_workflow(llm_manager) -> tuple[SG_Workflow, RLTrainer, LLMDecisionMaker]:
    """创建基于RL的LLM决策虚假信息干预工作流"""
    
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
    workflow = SG_Workflow("rl_misinfo_workflow", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建虚假信息沙盒
    sandbox = MisinformationSandbox(
        initial_users=100,
        max_users=1000,
        initial_misinfo_count=5
    )
    
    # 创建虚假信息环境节点
    def misinfo_env_func(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """虚假信息环境节点函数"""
        # 获取当前状态
        case = sandbox.case_generator()
        current_state = case["state"]
        
        # 使用LLM做出决策
        decision_result = decision_maker.make_decision(current_state)
        decision = decision_result["decision"]
        
        # 执行干预决策
        try:
            # 执行干预
            intervention_result = sandbox.execute_intervention(
                decision["action"], 
                decision.get("target", "general")
            )
            
            # 验证和执行决策
            score = sandbox.verify_score(
                f"{decision['action']} {decision.get('target', 'general')}",
                case
            )
            
            # 计算奖励
            reward = score * 10
            
            # 构建状态特征
            state_features = {
                "total_users": current_state["network_state"]["total_users"],
                "belief_ratio": current_state["misinformation_state"]["belief_ratio"],
                "total_spreads": current_state["misinformation_state"]["total_spreads"],
                "intervention_count": current_state["intervention_stats"]["total_interventions"],
                "decision_type": _encode_intervention_type(decision["action"])
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
                "intervention": intervention_result,
                "score": score,
                "reward": reward,
                "rl_update": update_result,
                "sandbox_id": sandbox.sandbox_id
            }
            
        except Exception as e:
            print(f"虚假信息干预执行错误: {e}")
            return {
                "state": current_state,
                "decision": {"action": "FACT_CHECK", "reasoning": f"执行错误: {e}"},
                "score": 0.0,
                "reward": 0.0,
                "error": str(e)
            }
    
    # 添加虚假信息环境节点
    misinfo_env_node = EnhancedWorkflowNode(
        "misinfo_environment",
        NodeType.SANDBOX,
        sandbox=sandbox,
        condition=NodeCondition(),
        limits=NodeLimits(max_visits=10, resource_cost={"energy": 10, "tokens": 5})
    )
    workflow.add_node(misinfo_env_node)
    
    return workflow, rl_trainer, decision_maker


def _encode_intervention_type(action: str) -> int:
    """编码干预类型"""
    action_map = {
        "FACT_CHECK": 1,
        "WARNING_LABEL": 2,
        "DOWNRANK": 3,
        "REMOVE": 4,
        "EDUCATE": 5,
        "PROMOTE_TRUTH": 6
    }
    return action_map.get(action, 0)


def run_rl_misinfo_demo(steps: int = 5):
    """运行基于RL的LLM决策虚假信息干预演示"""
    
    print("🔥 SandGraph Misinformation Spread Demo")
    print("=" * 60)
    
    # 1. 创建LLM管理器
    print("\n1. Creating LLM Manager")
    llm_manager = create_shared_llm_manager(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        backend="huggingface",
        temperature=0.7,
        max_length=512,
        device="auto",
        torch_dtype="float16"
    )
    
    # 2. 创建工作流和RL训练器
    print("\n2. Creating RL Misinformation Workflow")
    workflow, rl_trainer, decision_maker = create_rl_misinfo_workflow(llm_manager)
    
    # 3. 执行多步虚假信息干预
    print(f"\n3. Executing {steps} Misinformation Intervention Steps")
    
    results = []
    for step in range(steps):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        try:
            # 直接执行虚假信息环境节点
            node = workflow.nodes.get("misinfo_environment")
            if node and node.sandbox:
                # 获取当前状态
                case = node.sandbox.case_generator()
                current_state = case["state"]
                
                # 使用LLM做出决策
                decision_result = decision_maker.make_decision(current_state)
                decision = decision_result["decision"]
                
                # 执行干预决策
                try:
                    # 执行干预
                    intervention_result = node.sandbox.execute_intervention(
                        decision["action"], 
                        decision.get("target", "general")
                    )
                    
                    # 验证和执行决策
                    score = node.sandbox.verify_score(
                        f"{decision['action']} {decision.get('target', 'general')}",
                        case
                    )
                    
                    # 计算奖励
                    reward = score * 10
                    
                    # 构建状态特征
                    state_features = {
                        "total_users": current_state["network_state"]["total_users"],
                        "belief_ratio": current_state["misinformation_state"]["belief_ratio"],
                        "total_spreads": current_state["misinformation_state"]["total_spreads"],
                        "intervention_count": current_state["intervention_stats"]["total_interventions"],
                        "decision_type": _encode_intervention_type(decision["action"])
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
                        "state": current_state,
                        "decision": decision,
                        "llm_response": decision_result["llm_response"],
                        "intervention": intervention_result,
                        "score": score,
                        "reward": reward,
                        "rl_update": update_result,
                        "sandbox_id": node.sandbox.sandbox_id
                    }
                    
                    print(f"LLM Decision: {decision['action']} {decision.get('target', '')}")
                    print(f"Decision Reason: {decision.get('reasoning', '')}")
                    print(f"Intervention Success: {intervention_result.get('success', False)}")
                    print(f"Intervention Score: {score:.3f}")
                    print(f"RL Reward: {reward:.3f}")
                    
                    # 显示当前状态
                    misinfo_state = current_state["misinformation_state"]
                    print(f"Total Misinformation: {misinfo_state['total_pieces']}")
                    print(f"Total Spreads: {misinfo_state['total_spreads']}")
                    print(f"Belief Ratio: {misinfo_state['belief_ratio']:.3f}")
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"❌ Intervention Execution Error: {e}")
                    result = {
                        "state": current_state,
                        "decision": {"action": "FACT_CHECK", "reasoning": f"Execution Error: {e}"},
                        "score": 0.0,
                        "reward": 0.0,
                        "error": str(e)
                    }
                    results.append(result)
            else:
                print("❌ Misinformation Environment Node Not Found or Invalid")
        
        except Exception as e:
            print(f"❌ Step {step + 1} Execution Error: {e}")
    
    # 4. 输出最终结果
    print("\n4. Final Results")
    
    # 计算统计信息
    total_reward = sum(r.get("reward", 0) for r in results)
    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    decision_count = decision_maker.decision_count
    
    print(f"Total Decisions: {decision_count}")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Average Score: {avg_score:.3f}")
    
    # 显示RL训练统计
    rl_stats = rl_trainer.get_training_stats()
    print(f"RL Training Steps: {rl_stats['training_step']}")
    print(f"RL Algorithm: {rl_stats['algorithm']}")
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SandGraph Misinformation Spread Demo")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("--test", action="store_true", help="Run tests instead of demo")
    
    args = parser.parse_args()
    
    if args.test:
        # 运行测试
        print("🔥 SandGraph Misinformation Spread Demo 测试")
        print("=" * 80)
        
        success1 = test_misinformation_sandbox()
        success2 = test_llm_decision_maker()
        success3 = test_parsing_logic()
        
        if success1 and success2 and success3:
            print("\n🎉 所有测试通过!")
        else:
            print("\n💥 部分测试失败!")
    else:
        # 运行演示
        print("🔥 SandGraph Misinformation Spread Demo")
        print("=" * 60)
        print(f"Steps: {args.steps}")
        
        try:
            results = run_rl_misinfo_demo(args.steps)
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()


def test_misinformation_sandbox():
    """测试虚假信息沙盒"""
    print("🔥 测试Misinformation Sandbox")
    print("=" * 60)
    
    # 创建沙盒
    sandbox = MisinformationSandbox(
        initial_users=50,
        max_users=200,
        initial_misinfo_count=3
    )
    
    print(f"✅ 沙盒创建成功: {sandbox.sandbox_id}")
    print(f"用户数量: {len(sandbox.users)}")
    print(f"连接数量: {len(sandbox.connections)}")
    print(f"虚假信息数量: {len(sandbox.information_pieces)}")
    
    # 测试状态生成
    case = sandbox.case_generator()
    state = case["state"]
    
    print(f"\n网络状态:")
    print(f"- 总用户: {state['network_state']['total_users']}")
    print(f"- 总连接: {state['network_state']['total_connections']}")
    print(f"- 网络密度: {state['network_state']['network_density']:.3f}")
    
    print(f"\n虚假信息状态:")
    print(f"- 总信息: {state['misinformation_state']['total_pieces']}")
    print(f"- 总传播: {state['misinformation_state']['total_spreads']}")
    print(f"- 相信者: {state['misinformation_state']['believers_count']}")
    print(f"- 怀疑者: {state['misinformation_state']['skeptics_count']}")
    print(f"- 信念比例: {state['misinformation_state']['belief_ratio']:.3f}")
    
    print(f"\n用户信念分布:")
    for belief, count in state['user_beliefs'].items():
        print(f"- {belief}: {count}")
    
    # 测试干预
    print(f"\n测试干预效果:")
    intervention_result = sandbox.execute_intervention("FACT_CHECK", "general")
    print(f"干预类型: {intervention_result['type']}")
    print(f"干预成功: {intervention_result['success']}")
    print(f"干预影响: {intervention_result['impact']}")
    
    # 测试评分
    score = sandbox.verify_score("FACT_CHECK general", case)
    print(f"干预评分: {score:.3f}")
    
    return True


def test_llm_decision_maker():
    """测试LLM决策器"""
    print("\n🔥 测试LLM Decision Maker")
    print("=" * 60)
    
    # 创建LLM管理器
    print("1. 创建LLM管理器...")
    try:
        llm_manager = create_shared_llm_manager(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            backend="huggingface",
            temperature=0.7,
            max_length=512,
            device="auto",
            torch_dtype="float16"
        )
        print("✅ LLM管理器创建成功")
    except Exception as e:
        print(f"❌ LLM管理器创建失败: {e}")
        return False
    
    # 创建决策器
    print("2. 创建决策器...")
    decision_maker = LLMDecisionMaker(llm_manager)
    
    # 创建测试状态
    print("3. 创建测试状态...")
    test_state = {
        "network_state": {
            "total_users": 100,
            "total_connections": 500,
            "network_density": 0.05
        },
        "misinformation_state": {
            "total_pieces": 5,
            "total_spreads": 25,
            "believers_count": 15,
            "skeptics_count": 10,
            "belief_ratio": 0.6
        },
        "user_beliefs": {
            "believer": 30,
            "skeptic": 20,
            "neutral": 40,
            "disbeliever": 10
        },
        "intervention_stats": {
            "total_interventions": 2,
            "successful_interventions": 1,
            "recent_interventions": 1
        }
    }
    
    # 测试决策生成
    print("4. 测试决策生成...")
    try:
        decision_result = decision_maker.make_decision(test_state)
        decision = decision_result["decision"]
        
        print(f"\n✅ 决策生成成功!")
        print(f"Action: {decision['action']}")
        print(f"Target: {decision['target']}")
        print(f"Reasoning: {decision['reasoning']}")
        print(f"Decision Count: {decision_result['decision_count']}")
        print(f"LLM Response Length: {len(decision_result['llm_response'])} characters")
        
        # 检查是否还是fallback决策
        if "Fallback decision" in decision['reasoning']:
            print("⚠️  警告: 仍然是fallback决策")
            print(f"完整LLM响应: {decision_result['llm_response']}")
            return False
        else:
            print("✅ 成功解析到有效决策!")
            return True
        
    except Exception as e:
        print(f"\n❌ 决策生成失败: {e}")
        return False


def test_parsing_logic():
    """测试解析逻辑"""
    print("\n🔥 测试解析逻辑")
    print("=" * 60)
    
    # 创建决策器（不需要LLM）
    decision_maker = LLMDecisionMaker(None)
    
    # 测试不同的响应格式
    test_cases = [
        "ACTION: FACT_CHECK GENERAL\nREASONING: High belief ratio requires fact checking",
        "ACTION: WARNING_LABEL HIGH_SPREAD\nREASONING: High spread misinformation needs warning",
        "ACTION: REMOVE HIGH_BELIEF\nREASONING: Remove false information with many believers",
        "action: educate general\nreasoning: Improve user literacy",
        "Action: Promote_Truth General\nReasoning: Promote verified information",
        "ACTION: EDUCATE [GENERAL]\nREASONING: Improve user media literacy",
        "ACTION: FACT_CHECK [HIGH_SPREAD]\nREASONING: Target high-spread misinformation",
        "ACTION WARNING_LABEL GENERAL\nREASONING: Add warning labels",
        "ACTION=REMOVE HIGH_BELIEF\nREASONING: Remove false information",
    ]
    
    test_state = {
        "network_state": {"total_users": 100},
        "misinformation_state": {"belief_ratio": 0.6},
        "user_beliefs": {"believer": 30},
        "intervention_stats": {"total_interventions": 2}
    }
    
    success_count = 0
    for i, test_response in enumerate(test_cases):
        print(f"\n测试案例 {i+1}: {test_response}")
        try:
            decision = decision_maker._parse_decision_response(test_response)
            if decision:
                print(f"✅ 解析成功: {decision['action']} {decision['target']}")
                success_count += 1
            else:
                print("❌ 解析失败")
        except Exception as e:
            print(f"❌ 解析异常: {e}")
    
    print(f"\n📊 解析测试结果: {success_count}/{len(test_cases)} 成功")
    return success_count >= len(test_cases) * 0.8  # 80%成功率


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
SandGraph OASIS API集成演示 - 基于真实OASIS API

集成真实的OASIS (Open Agent Social Interaction Simulations) API到SandGraph框架：
1. 使用真实的OASIS API进行大规模社交网络模拟
2. 智能体行为分析和优化
3. 社交网络动态研究
4. 与SandGraph的RL优化框架集成
"""

import sys
import os
import time
import json
import argparse
import asyncio
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
import re

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import (
    SG_Workflow, WorkflowMode, EnhancedWorkflowNode,
    NodeType, NodeCondition, NodeLimits, GameState
)
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm


class OasisAPISandbox:
    """OASIS API沙盒 - 集成真实OASIS API"""
    
    def __init__(self, 
                 profile_path: str = None,
                 database_path: str = "./data/oasis_simulation.db",
                 platform: str = "reddit",
                 available_actions: List[str] = None):
        
        self.sandbox_id = f"oasis_api_{int(time.time())}"
        self.profile_path = profile_path
        self.database_path = database_path
        self.platform = platform
        self.available_actions = available_actions or [
            "like_post", "dislike_post", "create_post", "create_comment",
            "like_comment", "dislike_comment", "search_posts", "search_user",
            "trend", "refresh", "do_nothing", "follow", "mute"
        ]
        
        # OASIS环境状态
        self.env = None
        self.agent_graph = None
        self.current_state = {}
        self.interaction_history = []
        
        # 初始化OASIS环境
        self._initialize_oasis_environment()
    
    def _initialize_oasis_environment(self):
        """初始化OASIS环境"""
        try:
            # 尝试导入OASIS
            import oasis
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType, ModelType
            from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph
            
            print("✅ OASIS API 导入成功")
            
            # 创建默认用户配置文件（如果未提供）
            if not self.profile_path:
                self.profile_path = self._create_default_profile()
            
            # 定义模型
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O_MINI,
            )
            
            # 转换动作类型
            action_types = []
            for action in self.available_actions:
                if hasattr(ActionType, action.upper()):
                    action_types.append(getattr(ActionType, action.upper()))
            
            # 生成智能体图
            self.agent_graph = asyncio.run(generate_reddit_agent_graph(
                profile_path=self.profile_path,
                model=self.model,
                available_actions=action_types,
            ))
            
            # 创建环境
            self.env = oasis.make(
                agent_graph=self.agent_graph,
                platform=oasis.DefaultPlatformType.REDDIT,
                database_path=self.database_path,
            )
            
            print(f"✅ OASIS环境初始化成功: {self.platform}")
            
        except ImportError as e:
            print(f"❌ OASIS API 导入失败: {e}")
            print("请安装OASIS: pip install camel-oasis")
            self.env = None
        except Exception as e:
            print(f"❌ OASIS环境初始化失败: {e}")
            self.env = None
    
    def _create_default_profile(self) -> str:
        """创建默认用户配置文件"""
        import tempfile
        import os
        
        # 创建临时配置文件
        profile_data = {
            "user_id": "default_user",
            "interests": ["technology", "AI", "programming"],
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.7,
                "neuroticism": 0.3
            },
            "social_network": {
                "followers": [],
                "following": [],
                "posts": [],
                "comments": []
            },
            "behavior_patterns": {
                "post_frequency": 0.3,
                "comment_frequency": 0.5,
                "like_frequency": 0.7,
                "share_frequency": 0.2
            }
        }
        
        # 创建临时文件
        temp_dir = "./data"
        os.makedirs(temp_dir, exist_ok=True)
        profile_path = os.path.join(temp_dir, "default_user_profile.json")
        
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"✅ 创建默认配置文件: {profile_path}")
        return profile_path
    
    async def reset_environment(self):
        """重置OASIS环境"""
        if self.env:
            try:
                await self.env.reset()
                print("✅ OASIS环境重置成功")
            except Exception as e:
                print(f"❌ OASIS环境重置失败: {e}")
    
    async def execute_action(self, agent_id: int, action_type: str, action_args: Dict[str, Any] = None):
        """执行OASIS动作"""
        if not self.env or not self.agent_graph:
            return {"success": False, "error": "OASIS环境未初始化"}
        
        try:
            from oasis import ActionType, LLMAction, ManualAction
            
            # 获取智能体
            agent = self.agent_graph.get_agent(agent_id)
            if not agent:
                return {"success": False, "error": f"智能体 {agent_id} 不存在"}
            
            # 创建动作
            if action_args:
                action = ManualAction(
                    action_type=getattr(ActionType, action_type.upper()),
                    action_args=action_args
                )
            else:
                action = LLMAction()
            
            # 执行动作
            actions = {agent: action}
            result = await self.env.step(actions)
            
            return {
                "success": True,
                "result": result,
                "agent_id": agent_id,
                "action_type": action_type
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def case_generator(self) -> Dict[str, Any]:
        """生成当前状态"""
        if not self.env:
            return self._generate_fallback_state()
        
        try:
            # 获取OASIS环境状态
            state = {
                "platform": self.platform,
                "agents_count": len(self.agent_graph.get_agents()) if self.agent_graph else 0,
                "database_path": self.database_path,
                "available_actions": self.available_actions,
                "interaction_history": self.interaction_history[-10:],
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "state": state,
                "metadata": {
                    "sandbox_id": self.sandbox_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ 获取OASIS状态失败: {e}")
            return self._generate_fallback_state()
    
    def _generate_fallback_state(self) -> Dict[str, Any]:
        """生成备用状态（当OASIS不可用时）"""
        return {
            "state": {
                "platform": self.platform,
                "agents_count": 0,
                "database_path": self.database_path,
                "available_actions": self.available_actions,
                "interaction_history": [],
                "oasis_available": False,
                "timestamp": datetime.now().isoformat()
            },
            "metadata": {
                "sandbox_id": self.sandbox_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def verify_score(self, action: str, case: Dict[str, Any]) -> float:
        """验证OASIS动作的效果"""
        try:
            # 解析动作
            action_parts = action.split()
            if len(action_parts) < 2:
                return 0.0
            
            action_type = action_parts[0].lower()
            target = action_parts[1] if len(action_parts) > 1 else "general"
            
            # 基础分数
            base_score = 0.0
            
            # 根据动作类型评分
            if action_type == "create_post":
                base_score = 0.7
            elif action_type == "create_comment":
                base_score = 0.6
            elif action_type == "like_post":
                base_score = 0.4
            elif action_type == "follow":
                base_score = 0.5
            elif action_type == "search_posts":
                base_score = 0.3
            elif action_type == "trend":
                base_score = 0.8
            else:
                base_score = 0.2
            
            # 根据目标调整分数
            if target == "engagement":
                base_score *= 1.2
            elif target == "growth":
                base_score *= 1.1
            elif target == "general":
                base_score *= 1.0
            
            # 考虑OASIS可用性
            current_state = case["state"]
            if not current_state.get("oasis_available", True):
                base_score *= 0.5  # OASIS不可用时降低分数
            
            # 限制分数范围
            return min(1.0, max(0.0, base_score))
            
        except Exception as e:
            print(f"评分错误: {e}")
            return 0.0
    
    async def execute_oasis_action(self, agent_id: int, action_type: str, action_args: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行OASIS动作"""
        result = await self.execute_action(agent_id, action_type, action_args)
        
        # 记录交互历史
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "action_type": action_type,
            "action_args": action_args,
            "success": result.get("success", False)
        })
        
        return result


class LLMOasisDecisionMaker:
    """LLM OASIS决策器"""
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.decision_count = 0
        
        # 历史数据管理
        self.decision_history = []
        self.oasis_history = []
        
        # 注册决策节点
        self.llm_manager.register_node("oasis_decision", {
            "role": "OASIS社交网络行为专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
    
    def make_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态做出OASIS决策"""
        self.decision_count += 1
        
        # 构建决策提示
        prompt = self._construct_decision_prompt(current_state)
        
        print("=" * 80)
        print(f"OASIS Decision {self.decision_count} - Complete Prompt Content:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        try:
            # 生成LLM响应
            response = self.llm_manager.generate_for_node(
                "oasis_decision",
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
                    "agent_id": 0,
                    "action_type": "create_post",
                    "action_args": {"content": "Hello OASIS world!"},
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
            print(f"❌ OASIS decision generation failed: {e}")
            fallback_decision = {
                "agent_id": 0,
                "action_type": "create_post",
                "action_args": {"content": "Hello OASIS world!"},
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
            "oasis_state": state,
        }
        self.decision_history.append(decision_record)
        
        # 保持历史记录在合理范围内
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]
    
    def _construct_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构造决策提示"""
        platform = state.get("platform", "reddit")
        agents_count = state.get("agents_count", 0)
        available_actions = state.get("available_actions", [])
        oasis_available = state.get("oasis_available", True)
        
        # 构建OASIS状态摘要
        oasis_summary = f"""
OASIS Platform Status:
- Platform: {platform}
- Agents Count: {agents_count}
- OASIS Available: {oasis_available}
- Available Actions: {', '.join(available_actions[:5])}...
"""
        
        # 构建决策历史摘要
        history_summary = ""
        if self.decision_history:
            recent_decisions = self.decision_history[-3:]  # 最近3个决策
            history_summary = "\nRecent OASIS Actions:\n"
            for record in recent_decisions:
                decision = record["decision"]
                history_summary += f"- Step {record['step']}: Agent {decision.get('agent_id', '')} {decision.get('action_type', '')} - {decision.get('reasoning', '')[:30]}...\n"
        
        # 重构后的简洁提示
        prompt = f"""You are an OASIS social network behavior expert.

REQUIRED RESPONSE FORMAT:
ACTION: [AGENT_ID] [ACTION_TYPE] [ACTION_ARGS]
REASONING: [brief explanation]

Available Actions: {', '.join(available_actions)}

Available Agents: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

{oasis_summary.strip()}
{history_summary.strip()}

Choose the best OASIS action to maximize social engagement and network growth. Respond ONLY in the required format above."""
        
        return prompt
    
    def _parse_decision_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM决策响应"""
        response = response.strip()
        
        print(f"🔍 解析响应: {response[:200]}...")
        
        # 尝试解析标准格式
        try:
            # 查找ACTION行 - 支持多种格式
            action_patterns = [
                # 标准格式: ACTION: 0 create_post {"content": "Hello"}
                r'ACTION:\s*(\d+)\s+([a-z_]+)\s+(.+)',
                # 小写格式: action: 0 create_post {"content": "Hello"}
                r'action:\s*(\d+)\s+([a-z_]+)\s+(.+)',
                # 首字母大写格式: Action: 0 Create_Post {"content": "Hello"}
                r'Action:\s*(\d+)\s+([A-Z_]+)\s+(.+)',
            ]
            
            agent_id = None
            action_type = None
            action_args = None
            
            for pattern in action_patterns:
                action_match = re.search(pattern, response, re.IGNORECASE)
                if action_match:
                    agent_id = int(action_match.group(1))
                    action_type = action_match.group(2).lower()
                    action_args_str = action_match.group(3).strip()
                    
                    # 尝试解析action_args
                    try:
                        action_args = json.loads(action_args_str)
                    except json.JSONDecodeError:
                        # 如果不是JSON格式，创建简单的参数
                        action_args = {"content": action_args_str}
                    
                    print(f"✅ 找到ACTION: Agent {agent_id} {action_type} {action_args}")
                    break
            
            if agent_id is None or action_type is None:
                print("❌ 未找到完整的ACTION字段")
                return None
            
            # 验证智能体ID
            if agent_id < 0 or agent_id > 9:
                print(f"❌ 无效的AGENT_ID: {agent_id}")
                return None
            
            # 验证动作类型
            valid_actions = [
                "like_post", "dislike_post", "create_post", "create_comment",
                "like_comment", "dislike_comment", "search_posts", "search_user",
                "trend", "refresh", "do_nothing", "follow", "mute"
            ]
            
            if action_type not in valid_actions:
                print(f"❌ 无效的ACTION_TYPE: {action_type}")
                return None
            
            # 查找REASONING行
            reasoning_patterns = [
                r'REASONING:\s*(.+?)(?:\n|$)',  # 标准格式
                r'reasoning:\s*(.+?)(?:\n|$)',  # 小写
                r'Reasoning:\s*(.+?)(?:\n|$)',  # 首字母大写
            ]
            
            reasoning = "No reasoning provided"
            for pattern in reasoning_patterns:
                reasoning_match = re.search(pattern, response, re.IGNORECASE)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    print(f"✅ 找到REASONING: {reasoning[:50]}...")
                    break
            
            print(f"✅ 解析成功: Agent {agent_id} {action_type} | {reasoning[:30]}...")
            
            return {
                "agent_id": agent_id,
                "action_type": action_type,
                "action_args": action_args,
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
            "CREATE_POST", "CREATE_COMMENT", "LIKE_POST", 
            "FOLLOW", "SEARCH_POSTS", "TREND"
        ]
        
        for action in valid_actions:
            if action in response_upper:
                return {
                    "agent_id": 0,
                    "action_type": action.lower(),
                    "action_args": {"content": f"Generated {action.lower()} content"},
                    "reasoning": f"Extracted action '{action}' from response"
                }
        
        # 如果没有找到有效动作，返回None
        return None


def create_rl_oasis_api_workflow(llm_manager) -> tuple[SG_Workflow, RLTrainer, LLMOasisDecisionMaker]:
    """创建基于RL的LLM决策OASIS API工作流"""
    
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
    decision_maker = LLMOasisDecisionMaker(llm_manager)
    
    # 创建工作流
    workflow = SG_Workflow("rl_oasis_api_workflow", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建OASIS API沙盒
    sandbox = OasisAPISandbox(
        profile_path=None,  # 使用默认配置文件
        database_path="./data/oasis_simulation.db",
        platform="reddit"
    )
    
    return workflow, rl_trainer, decision_maker, sandbox


def _encode_oasis_action(action: str) -> int:
    """编码OASIS动作类型"""
    action_map = {
        "create_post": 1,
        "create_comment": 2,
        "like_post": 3,
        "follow": 4,
        "search_posts": 5,
        "trend": 6
    }
    return action_map.get(action, 0)


async def run_rl_oasis_api_demo(steps: int = 5):
    """运行基于RL的LLM决策OASIS API演示"""
    
    print("🏝️ SandGraph OASIS API Integration Demo")
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
    print("\n2. Creating RL OASIS API Workflow")
    workflow, rl_trainer, decision_maker, sandbox = create_rl_oasis_api_workflow(llm_manager)
    
    # 3. 初始化OASIS环境
    print("\n3. Initializing OASIS Environment")
    await sandbox.reset_environment()
    
    # 4. 执行多步OASIS API调用
    print(f"\n4. Executing {steps} OASIS API Steps")
    
    results = []
    for step in range(steps):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        try:
            # 获取当前状态
            case = sandbox.case_generator()
            current_state = case["state"]
            
            # 使用LLM做出决策
            decision_result = decision_maker.make_decision(current_state)
            decision = decision_result["decision"]
            
            # 执行OASIS决策
            try:
                # 执行OASIS动作
                action_result = await sandbox.execute_oasis_action(
                    decision["agent_id"],
                    decision["action_type"],
                    decision.get("action_args", {})
                )
                
                # 验证和执行决策
                score = sandbox.verify_score(
                    f"{decision['action_type']} {decision.get('action_args', {}).get('content', 'general')}",
                    case
                )
                
                # 计算奖励
                reward = score * 10
                
                # 构建状态特征
                state_features = {
                    "agents_count": current_state.get("agents_count", 0),
                    "platform": current_state.get("platform", "reddit"),
                    "oasis_available": current_state.get("oasis_available", True),
                    "decision_type": _encode_oasis_action(decision["action_type"])
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
                    "action_result": action_result,
                    "score": score,
                    "reward": reward,
                    "rl_update": update_result,
                    "sandbox_id": sandbox.sandbox_id
                }
                
                print(f"LLM Decision: Agent {decision['agent_id']} {decision['action_type']}")
                print(f"Decision Reason: {decision.get('reasoning', '')}")
                print(f"Action Success: {action_result.get('success', False)}")
                print(f"OASIS Score: {score:.3f}")
                print(f"RL Reward: {reward:.3f}")
                
                # 显示当前OASIS状态
                print(f"Platform: {current_state.get('platform', 'unknown')}")
                print(f"Agents Count: {current_state.get('agents_count', 0)}")
                print(f"OASIS Available: {current_state.get('oasis_available', True)}")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ OASIS Action Execution Error: {e}")
                result = {
                    "state": current_state,
                    "decision": {"action_type": "create_post", "reasoning": f"Execution Error: {e}"},
                    "score": 0.0,
                    "reward": 0.0,
                    "error": str(e)
                }
                results.append(result)
        
        except Exception as e:
            print(f"❌ Step {step + 1} Execution Error: {e}")
    
    # 5. 输出最终结果
    print("\n5. Final Results")
    
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
    parser = argparse.ArgumentParser(description="SandGraph OASIS API Integration Demo")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    parser.add_argument("--test", action="store_true", help="Run tests instead of demo")
    parser.add_argument("--profile", type=str, help="Path to user profile file")
    parser.add_argument("--platform", type=str, default="reddit", choices=["reddit", "twitter"], help="OASIS platform")
    
    args = parser.parse_args()
    
    if args.test:
        # 运行测试
        print("🏝️ SandGraph OASIS API Integration Demo 测试")
        print("=" * 80)
        
        # 这里可以添加测试函数
        print("✅ 测试功能待实现")
    else:
        # 运行演示
        print("🏝️ SandGraph OASIS API Integration Demo")
        print("=" * 60)
        print(f"Steps: {args.steps}")
        print(f"Platform: {args.platform}")
        if args.profile:
            print(f"Profile: {args.profile}")
        
        try:
            results = asyncio.run(run_rl_oasis_api_demo(args.steps))
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 
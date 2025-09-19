#!/usr/bin/env python3
"""
Enhanced Sandbox-RL OASIS社交网络模拟演示 - 基于原始OASIS demo，集成WanDB和TensorBoard监控

集成OASIS (Open Agent Social Interaction Simulations) 到Sandbox-RL框架：
1. 大规模智能体社交网络模拟
2. 信息传播和群体行为研究
3. 社交网络动态分析
4. 智能体行为优化
5. 实时监控和可视化
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
from sandbox_rl.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from sandbox_rl.core.monitoring import (
    SocialNetworkMonitor, 
    MonitoringConfig, 
    SocialNetworkMetrics, 
    MetricsCollector,
    create_monitor
)

# 导入原始OASIS demo的类
from oasis_social_demo import (
    SocialActionType, UserProfile, SocialPost, OasisSocialSandbox, LLMSocialDecisionMaker
)


class EnhancedOasisSocialSandbox(OasisSocialSandbox):
    """增强版OASIS社交网络沙盒 - 集成监控功能"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sandbox_id = f"enhanced_oasis_social_{int(time.time())}"
        
        # 监控相关
        self.metrics_history = []
        self.monitor = None
        self.start_time = time.time()
    
    def setup_monitoring(self, config: MonitoringConfig):
        """设置监控系统"""
        self.monitor = create_monitor(config)
        
        # 添加告警回调
        self.monitor.add_alert_callback(self._handle_alert)
        
        print("✅ OASIS Social Network monitoring system initialized")
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """处理监控告警"""
        print(f"🚨 OASIS ALERT [{alert['severity'].upper()}]: {alert['message']}")
        
        # 记录告警到文件
        alert_log_path = "./logs/enhanced_oasis_alerts.json"
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)
        
        try:
            alerts = []
            if os.path.exists(alert_log_path):
                with open(alert_log_path, "r") as f:
                    alerts = json.load(f)
            
            alerts.append(alert)
            
            with open(alert_log_path, "w") as f:
                json.dump(alerts, f, indent=2)
        except Exception as e:
            print(f"Failed to log alert: {e}")
    
    def _collect_metrics(self) -> SocialNetworkMetrics:
        """收集OASIS社交网络指标"""
        # 计算统计信息
        total_users = len(self.users)
        total_posts = len(self.posts)
        total_comments = len(self.comments)
        
        # 计算互动统计
        total_likes = sum(post.likes for post in self.posts.values())
        total_shares = sum(post.shares for post in self.posts.values())
        total_follows = sum(len(user.following) for user in self.users.values())
        
        # 计算活跃用户
        active_users = sum(1 for user in self.users.values() 
                          if len(user.posts) > 0 or len(user.comments) > 0)
        
        # 计算网络密度
        total_possible_connections = total_users * (total_users - 1)
        actual_connections = sum(len(user.following) for user in self.users.values())
        network_density = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        # 计算平均关注者数
        avg_followers = sum(len(user.followers) for user in self.users.values()) / total_users if total_users > 0 else 0
        
        # 计算参与度
        engagement_rate = (total_likes + total_comments + total_shares) / (total_posts * 10) if total_posts > 0 else 0
        
        # 计算热门内容
        viral_posts = sum(1 for post in self.posts.values() if post.trending_score > 0.5)
        trending_topics = len(set([post.content.split('#')[1].split()[0] 
                                  for post in self.posts.values() 
                                  if '#' in post.content]))
        
        # 创建指标对象
        metrics = SocialNetworkMetrics(
            total_users=total_users,
            active_users=active_users,
            new_users=int(total_users * 0.1),  # 模拟新用户
            churned_users=int(total_users * 0.02),  # 模拟流失用户
            user_growth_rate=0.08,  # 模拟增长率
            total_posts=total_posts,
            total_likes=total_likes,
            total_comments=total_comments,
            total_shares=total_shares,
            engagement_rate=min(1.0, engagement_rate),
            avg_session_time=random.uniform(15, 45),
            bounce_rate=random.uniform(0.1, 0.3),
            retention_rate=random.uniform(0.6, 0.9),
            viral_posts=viral_posts,
            trending_topics=trending_topics,
            content_quality_score=random.uniform(0.6, 0.9),
            user_satisfaction_score=random.uniform(0.5, 0.95),
            content_diversity_score=random.uniform(0.4, 0.8),
            controversy_level=random.uniform(0.0, 0.4),
            network_density=network_density,
            avg_followers=avg_followers,
            avg_following=total_follows / total_users if total_users > 0 else 0,
            clustering_coefficient=random.uniform(0.2, 0.6),
            network_growth_rate=0.05,
            total_communities=random.randint(3, 10),
            avg_community_size=random.uniform(5, 20),
            community_engagement=random.uniform(0.3, 0.7),
            cross_community_interactions=random.randint(10, 50),
            influencer_count=int(total_users * 0.1),
            avg_influence_score=random.uniform(0.3, 0.8),
            viral_spread_rate=random.uniform(0.1, 0.5),
            information_cascade_depth=random.uniform(1.5, 4.0),
            response_time_avg=random.uniform(0.5, 2.0),
            error_rate=random.uniform(0.01, 0.05),
            system_uptime=time.time() - self.start_time
        )
        
        return metrics


class EnhancedLLMSocialDecisionMaker(LLMSocialDecisionMaker):
    """增强版LLM社交决策器 - 集成监控功能"""
    
    def __init__(self, llm_manager):
        super().__init__(llm_manager)
        # 重新注册节点名称
        self.llm_manager.register_node("enhanced_oasis_decision", {
            "role": "OASIS社交网络策略专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
    
    def make_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态做出决策 - 增强版"""
        self.decision_count += 1
        
        # 构建增强的决策提示
        prompt = self._construct_enhanced_decision_prompt(current_state)
        
        print("=" * 80)
        print(f"Enhanced OASIS Decision {self.decision_count} - Complete Prompt:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        try:
            # 生成LLM响应
            response = self.llm_manager.generate_for_node(
                "enhanced_oasis_decision",
                prompt,
                temperature=0.7,
                max_new_tokens=256
            )
            
            print(f"LLM Response Status: {response.status if hasattr(response, 'status') else 'unknown'}")
            print(f"LLM Complete Response: {response.text}")
            
            # 解析响应
            decision = self._parse_decision_response(response.text)
            
            if decision is None:
                decision = {
                    "action": "CREATE_POST",
                    "user_id": "user_0",
                    "target_id": None,
                    "content": "Hello OASIS world!",
                    "reasoning": "Fallback decision"
                }
            
            # 更新历史
            self._update_history(current_state, decision, response.text)
            
            return {
                "decision": decision,
                "llm_response": response.text,
                "decision_count": self.decision_count
            }
            
        except Exception as e:
            print(f"❌ Decision generation failed: {e}")
            fallback_decision = {
                "action": "CREATE_POST",
                "user_id": "user_0",
                "target_id": None,
                "content": "Hello OASIS world!",
                "reasoning": f"Error: {str(e)}"
            }
            
            self._update_history(current_state, fallback_decision, f"Error: {str(e)}")
            
            return {
                "decision": fallback_decision,
                "llm_response": f"Error: {str(e)}",
                "decision_count": self.decision_count
            }
    
    def _construct_enhanced_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构建增强的决策提示"""
        network_state = state.get("state", {}).get("network_state", {})
        trending_content = state.get("state", {}).get("trending_content", [])
        active_users = state.get("state", {}).get("active_users", [])
        
        prompt = f"""
你是一个OASIS社交网络策略专家。基于当前的网络状态，请做出最优的社交网络管理决策。

当前网络状态：
- 总用户数: {network_state.get('total_users', 0)}
- 总帖子数: {network_state.get('total_posts', 0)}
- 总点赞数: {network_state.get('total_likes', 0)}
- 总分享数: {network_state.get('total_shares', 0)}
- 网络密度: {network_state.get('network_density', 0):.3f}

热门帖子 (前3个):
{chr(10).join([f"- {post['content'][:50]}... (分数: {post['trending_score']:.2f})" for post in trending_content[:3]])}

活跃用户 (前3个):
{chr(10).join([f"- {user['user_id']} (活跃度: {user['activity_score']}, 关注者: {user['followers_count']})" for user in active_users[:3]])}

可用的动作类型：
- CREATE_POST: 创建新帖子
- CREATE_COMMENT: 评论帖子
- LIKE_POST: 点赞帖子
- FOLLOW: 关注用户
- SHARE: 分享帖子
- TREND: 提升帖子热度

可用用户: user_0, user_1, user_2, user_3, user_4, user_5, user_6, user_7, user_8, user_9
可用帖子: post_0, post_1, post_2, post_3, post_4, post_5, post_6, post_7, post_8, post_9

请以以下格式回复：
ACTION: [动作类型] [用户ID] [目标ID] [内容]
REASONING: [决策理由]

例如：
ACTION: CREATE_POST user_3 "Exploring the latest AI developments! #AI #technology"
REASONING: Creating content about trending technology topics can increase engagement.
"""
        
        return prompt
    
    def _parse_decision_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM决策响应 - 增强版"""
        response = response.strip()
        
        print(f"🔍 解析响应: {response[:200]}...")
        
        try:
            # 查找ACTION行 - 支持多种格式
            action_patterns = [
                # 标准格式: ACTION: CREATE_POST user_0 target_id content
                r'ACTION:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+([a-z_0-9_]+)\s+(.+)',
                # 无目标格式: ACTION: CREATE_POST user_0 content
                r'ACTION:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+(.+)',
                # 连字符格式: ACTION: CREATE_POST - user_5 - content
                r'ACTION:\s*([A-Z_]+)\s*[-–]\s*([a-z_0-9_]+)\s*[-–]\s*(.+)',
                # 连字符格式无目标: ACTION: CREATE_POST - user_5 content
                r'ACTION:\s*([A-Z_]+)\s*[-–]\s*([a-z_0-9_]+)\s+(.+)',
                # 方括号格式: ACTION: CREATE_COMMENT [USER_1] [TARGET_POST] content
                r'ACTION:\s*([A-Z_]+)\s+\[([^\]]+)\]\s+\[([^\]]+)\]\s+(.+)',
                # 方括号格式无目标: ACTION: CREATE_POST [USER_1] content
                r'ACTION:\s*([A-Z_]+)\s+\[([^\]]+)\]\s+(.+)',
                # 小写格式: action: create_post user_0 target_id content
                r'action:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+([a-z_0-9_]+)\s+(.+)',
                # 首字母大写格式: Action: Create_Post user_0 target_id content
                r'Action:\s*([A-Z_]+)\s+([a-z_0-9]+)\s+([a-z_0-9_]+)\s+(.+)',
            ]
            
            action = None
            user_id = None
            target_id = None
            content = None
            
            for pattern in action_patterns:
                action_match = re.search(pattern, response, re.IGNORECASE)
                if action_match:
                    action = action_match.group(1).upper()
                    
                    # 处理方括号格式
                    if '[' in action_match.group(2):
                        # 方括号格式
                        user_id_raw = action_match.group(2).strip('[]')
                        if len(action_match.groups()) >= 4:
                            target_id_raw = action_match.group(3).strip('[]')
                            content = action_match.group(4).strip()
                        else:
                            content = action_match.group(3).strip()
                        
                        # 处理占位符
                        user_id = self._resolve_placeholder(user_id_raw)
                        target_id = self._resolve_placeholder(target_id_raw) if 'target_id_raw' in locals() else None
                    else:
                        # 标准格式或连字符格式
                        user_id = action_match.group(2)
                        if len(action_match.groups()) >= 4:
                            target_id = action_match.group(3)
                            content = action_match.group(4).strip()
                        else:
                            content = action_match.group(3).strip()
                    
                    print(f"✅ 找到ACTION: {action} {user_id} {target_id or 'None'} {content[:30]}...")
                    break
            
            if not action or not user_id:
                print("❌ 未找到完整的ACTION字段")
                return None
            
            # 验证动作是否有效
            valid_actions = [
                "CREATE_POST", "CREATE_COMMENT", "LIKE_POST", 
                "FOLLOW", "SHARE", "TREND"
            ]
            
            if action not in valid_actions:
                print(f"❌ 无效的ACTION: {action}")
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
            
            print(f"✅ 解析成功: {action} {user_id} | {reasoning[:30]}...")
            
            return {
                "action": action,
                "user_id": user_id,
                "target_id": target_id,
                "content": content,
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"❌ Decision parsing failed: {e}")
            return None


def run_enhanced_rl_oasis_demo(steps: int = 10, 
                              enable_wandb: bool = True,
                              enable_tensorboard: bool = True,
                              wandb_project: str = "sandgraph-enhanced-oasis"):
    """运行增强版RL OASIS演示 - 基于原始OASIS demo"""
    
    print("🚀 Enhanced OASIS Social Network Demo with Monitoring")
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
    
    # 2. 创建监控配置
    monitor_config = MonitoringConfig(
        enable_wandb=enable_wandb,
        enable_tensorboard=enable_tensorboard,
        wandb_project_name=wandb_project,
        wandb_run_name=f"enhanced_oasis_{int(time.time())}",
        tensorboard_log_dir="./logs/enhanced_oasis",
        log_file_path="./logs/enhanced_oasis_metrics.json",
        metrics_sampling_interval=2.0,
        engagement_rate_threshold=0.15,
        user_growth_threshold=0.08
    )
    
    # 3. 创建沙盒和决策器
    print("\n2. Creating Enhanced OASIS Components")
    sandbox = EnhancedOasisSocialSandbox(
        initial_users=50,
        max_users=1000,
        initial_posts=20
    )
    
    # 设置监控
    sandbox.setup_monitoring(monitor_config)
    
    # 创建决策器
    decision_maker = EnhancedLLMSocialDecisionMaker(llm_manager)
    
    # 创建RL训练器
    rl_config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=0.001,
        batch_size=32,
        gamma=0.99
    )
    rl_trainer = RLTrainer(rl_config, llm_manager)
    
    # 4. 执行多步社交网络模拟
    print(f"\n3. Executing {steps} Enhanced OASIS Social Network Steps")
    
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
            
            # 执行社交决策
            try:
                # 执行社交行动
                action_result = sandbox.execute_social_action(
                    decision["action"],
                    decision["user_id"],
                    decision.get("target_id"),
                    decision.get("content")
                )
                
                # 验证和执行决策
                score = sandbox.verify_score(
                    f"{decision['action']} {decision.get('target_id', 'general')}",
                    case
                )
                
                # 计算奖励
                reward = score * 10
                
                # 构建状态特征
                state_features = {
                    "total_users": current_state["network_state"].get("total_users", 0),
                    "network_density": current_state["network_state"].get("network_density", 0.0),
                    "total_posts": current_state["network_state"].get("total_posts", 0),
                    "total_likes": current_state["network_state"].get("total_likes", 0),
                    "decision_type": _encode_social_action(decision["action"])
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
                
                print(f"LLM Decision: {decision['action']} {decision.get('user_id', '')}")
                print(f"Decision Reason: {decision.get('reasoning', '')}")
                print(f"Action Success: {action_result.get('success', False)}")
                print(f"Social Score: {score:.3f}")
                print(f"RL Reward: {reward:.3f}")
                
                # 显示当前网络状态
                network_state = current_state["network_state"]
                print(f"Total Users: {network_state.get('total_users', 0)}")
                print(f"Total Posts: {network_state.get('total_posts', 0)}")
                print(f"Network Density: {network_state.get('network_density', 0.0):.3f}")
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Social Action Execution Error: {e}")
                result = {
                    "state": current_state,
                    "decision": {"action": "CREATE_POST", "reasoning": f"Execution Error: {e}"},
                    "score": 0.0,
                    "reward": 0.0,
                    "error": str(e)
                }
                results.append(result)
        
        except Exception as e:
            print(f"❌ Step {step + 1} Execution Error: {e}")
    
    # 5. 输出最终结果
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
    
    # 导出监控结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 导出指标
    if sandbox.monitor:
        metrics_file = f"./logs/enhanced_oasis_metrics_{timestamp}.json"
        sandbox.monitor.export_metrics(metrics_file, "json")
        
        print(f"\n📁 Results exported:")
        print(f"   - Metrics: {metrics_file}")
        
        # 打印最终统计
        if sandbox.metrics_history:
            final_metrics = sandbox.metrics_history[-1]
            print(f"\n📊 Final Statistics:")
            print(f"   - Total Users: {final_metrics.total_users}")
            print(f"   - Engagement Rate: {final_metrics.engagement_rate:.3f}")
            print(f"   - Content Quality: {final_metrics.content_quality_score:.3f}")
            print(f"   - Network Density: {final_metrics.network_density:.3f}")
            print(f"   - Total Alerts: {len(sandbox.monitor.alerts) if sandbox.monitor else 0}")
    
    print("\n✅ Enhanced OASIS demo completed!")
    
    return results


def _encode_social_action(action: str) -> int:
    """编码社交行动类型"""
    action_map = {
        "CREATE_POST": 1,
        "CREATE_COMMENT": 2,
        "LIKE_POST": 3,
        "FOLLOW": 4,
        "SHARE": 5,
        "TREND": 6
    }
    return action_map.get(action, 0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Enhanced OASIS Social Network Demo with Monitoring")
    
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of simulation steps")
    parser.add_argument("--enable-wandb", action="store_true", default=True,
                       help="Enable WanDB logging")
    parser.add_argument("--enable-tensorboard", action="store_true", default=True,
                       help="Enable TensorBoard logging")
    parser.add_argument("--wandb-project", type=str, default="sandgraph-enhanced-oasis",
                       help="WanDB project name")
    parser.add_argument("--tensorboard-dir", type=str, default="./logs/enhanced_oasis",
                       help="TensorBoard log directory")
    
    args = parser.parse_args()
    
    # 运行演示
    run_enhanced_rl_oasis_demo(
        steps=args.steps,
        enable_wandb=args.enable_wandb,
        enable_tensorboard=args.enable_tensorboard,
        wandb_project=args.wandb_project
    )


if __name__ == "__main__":
    main() 
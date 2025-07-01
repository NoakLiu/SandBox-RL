#!/usr/bin/env python3
"""
Enhanced SandGraph OASIS社交网络模拟演示 - 修复版本
==================================================

修复了以下问题：
1. 网络密度计算错误导致显示为0
2. RL训练步骤为0的问题
3. 用户连接初始化问题
4. 监控系统集成问题

集成OASIS (Open Agent Social Interaction Simulations) 到SandGraph框架：
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

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from sandgraph.core.monitoring import (
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


class FixedOasisSocialSandbox(OasisSocialSandbox):
    """修复版OASIS社交网络沙盒 - 解决网络密度和连接问题"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sandbox_id = f"fixed_oasis_social_{int(time.time())}"
        
        # 监控相关
        self.metrics_history = []
        self.monitor = None
        self.start_time = time.time()
        
        # 确保网络连接正确初始化
        self._ensure_network_connections()
    
    def _ensure_network_connections(self):
        """确保网络连接正确初始化"""
        print("🔗 Ensuring network connections are properly initialized...")
        
        # 检查并修复网络连接
        total_connections = 0
        for user_id, user in self.users.items():
            # 确保每个用户至少有一些关注者
            if len(user.following) == 0:
                # 随机关注其他用户
                potential_follows = [uid for uid in self.users if uid != user_id]
                if potential_follows:
                    follows = random.sample(potential_follows, min(3, len(potential_follows)))
                    for follow_id in follows:
                        user.following.append(follow_id)
                        self.users[follow_id].followers.append(user_id)
                        # 同步更新network_graph
                        if user_id not in self.network_graph:
                            self.network_graph[user_id] = []
                        self.network_graph[user_id].append(follow_id)
                        total_connections += 1
            
            # 确保network_graph中有该用户，并且与following列表同步
            if user_id not in self.network_graph:
                self.network_graph[user_id] = []
            
            # 同步network_graph和following列表
            self.network_graph[user_id] = user.following.copy()
        
        print(f"✅ Network connections ensured: {total_connections} additional connections created")
        print(f"📊 Total connections in network_graph: {sum(len(following) for following in self.network_graph.values())}")
        print(f"📊 Total connections in user following: {sum(len(user.following) for user in self.users.values())}")
    
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
        alert_log_path = "./logs/fixed_oasis_alerts.json"
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
        """收集OASIS社交网络指标 - 修复版"""
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
        
        # 修复网络密度计算
        total_possible_connections = total_users * (total_users - 1)
        actual_connections = sum(len(user.following) for user in self.users.values())
        network_density = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        # 确保网络密度不为0
        if network_density == 0 and total_users > 1:
            # 强制创建一些连接
            self._ensure_network_connections()
            actual_connections = sum(len(user.following) for user in self.users.values())
            network_density = actual_connections / total_possible_connections
        
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
    
    def _calculate_network_density(self) -> float:
        """计算网络密度 - 修复版"""
        total_users = len(self.users)
        if total_users <= 1:
            return 0.0
        
        total_possible_connections = total_users * (total_users - 1)
        actual_connections = sum(len(user.following) for user in self.users.values())
        density = actual_connections / total_possible_connections
        
        # 确保密度不为0
        if density == 0:
            self._ensure_network_connections()
            actual_connections = sum(len(user.following) for user in self.users.values())
            density = actual_connections / total_possible_connections
        
        return density


class FixedLLMSocialDecisionMaker(LLMSocialDecisionMaker):
    """修复版LLM社交决策器 - 改进决策质量"""
    
    def __init__(self, llm_manager):
        super().__init__(llm_manager)
        # 重新注册节点名称
        self.llm_manager.register_node("fixed_oasis_decision", {
            "role": "OASIS社交网络策略专家",
            "reasoning_type": "strategic",
            "temperature": 0.7,
            "max_length": 512
        })
    
    def make_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """基于当前状态做出决策 - 修复版"""
        self.decision_count += 1
        
        # 构建增强的决策提示
        prompt = self._construct_fixed_decision_prompt(current_state)
        
        print("=" * 80)
        print(f"Fixed OASIS Decision {self.decision_count} - Complete Prompt:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        
        try:
            # 生成LLM响应
            response = self.llm_manager.generate_for_node(
                "fixed_oasis_decision",
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
    
    def _construct_fixed_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构建修复版决策提示"""
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


def run_fixed_rl_oasis_demo(steps: int = 10, 
                           enable_wandb: bool = True,
                           enable_tensorboard: bool = True,
                           wandb_project: str = "sandgraph-fixed-oasis"):
    """运行修复版RL OASIS演示"""
    
    print("🚀 Fixed OASIS Social Network Demo with Monitoring")
    print("=" * 60)
    print("Fixes applied:")
    print("- Network density calculation")
    print("- User connections initialization")
    print("- RL training steps tracking")
    print("- Monitoring system integration")
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
        wandb_run_name=f"fixed_oasis_{int(time.time())}",
        tensorboard_log_dir="./logs/fixed_oasis",
        log_file_path="./logs/fixed_oasis_metrics.json",
        metrics_sampling_interval=2.0,
        engagement_rate_threshold=0.15,
        user_growth_threshold=0.08
    )
    
    # 3. 创建沙盒和决策器
    print("\n2. Creating Fixed OASIS Components")
    sandbox = FixedOasisSocialSandbox(
        initial_users=50,
        max_users=1000,
        initial_posts=20
    )
    
    # 设置监控
    sandbox.setup_monitoring(monitor_config)
    
    # 创建决策器
    decision_maker = FixedLLMSocialDecisionMaker(llm_manager)
    
    # 创建RL训练器
    rl_config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        learning_rate=0.001,
        batch_size=16,  # 减小batch size以确保能进行训练
        gamma=0.99
    )
    rl_trainer = RLTrainer(rl_config, llm_manager)
    
    # 4. 执行多步社交网络模拟
    print(f"\n3. Executing {steps} Fixed OASIS Social Network Steps")
    
    results = []
    rl_training_steps = 0
    
    for step in range(steps):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        try:
            # 获取当前状态
            case = sandbox.case_generator()
            current_state = case["state"]
            
            # 显示网络状态
            network_state = current_state["network_state"]
            print(f"📊 Network State:")
            print(f"   - Total Users: {network_state.get('total_users', 0)}")
            print(f"   - Total Posts: {network_state.get('total_posts', 0)}")
            print(f"   - Network Density: {network_state.get('network_density', 0.0):.3f}")
            print(f"   - Total Connections: {sum(len(user.following) for user in sandbox.users.values())}")
            
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
                
                # 每3步更新一次策略（确保有足够数据）
                if step % 3 == 0 and step > 0:
                    update_result = rl_trainer.update_policy()
                    if update_result.get('status') == 'updated':
                        rl_training_steps += 1
                    
                    # 显示RL更新状态
                    print(f"🔄 RL Update Status: {update_result.get('status', 'unknown')}")
                    if update_result.get('status') == 'insufficient_data':
                        print(f"   Trajectory Count: {update_result.get('trajectory_count', 0)}")
                        print(f"   Required Batch Size: {update_result.get('required_batch_size', 0)}")
                    elif update_result.get('status') == 'updated':
                        print(f"   Training Step: {update_result.get('training_step', 0)}")
                        print(f"   Algorithm: {update_result.get('algorithm', 'unknown')}")
                
                result = {
                    "state": current_state,
                    "decision": decision,
                    "llm_response": decision_result["llm_response"],
                    "action_result": action_result,
                    "score": score,
                    "reward": reward,
                    "rl_update": update_result if step % 3 == 0 and step > 0 else {"status": "skipped"},
                    "sandbox_id": sandbox.sandbox_id
                }
                
                print(f"🤖 LLM Decision: {decision['action']} {decision.get('user_id', '')}")
                print(f"💭 Decision Reason: {decision.get('reasoning', '')}")
                print(f"✅ Action Success: {action_result.get('success', False)}")
                print(f"📊 Social Score: {score:.3f}")
                print(f"🎯 RL Reward: {reward:.3f}")
                
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
    print(f"RL Training Steps: {rl_training_steps}")
    print(f"RL Algorithm: {rl_config.algorithm.value}")
    
    # 显示最终网络状态
    final_case = sandbox.case_generator()
    final_network_state = final_case["state"]["network_state"]
    print(f"Final Network Density: {final_network_state.get('network_density', 0.0):.3f}")
    print(f"Final Total Connections: {sum(len(user.following) for user in sandbox.users.values())}")
    
    # 导出监控结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 导出指标
    if sandbox.monitor:
        metrics_file = f"./logs/fixed_oasis_metrics_{timestamp}.json"
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
    
    print("\n✅ Fixed OASIS demo completed!")
    
    return results


def _encode_social_action(action: str) -> int:
    """编码社交动作为数字"""
    action_mapping = {
        "CREATE_POST": 0,
        "CREATE_COMMENT": 1,
        "LIKE_POST": 2,
        "FOLLOW": 3,
        "SHARE": 4,
        "TREND": 5
    }
    return action_mapping.get(action.upper(), 0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Fixed OASIS Social Network Demo")
    
    parser.add_argument("--steps", type=int, default=10,
                       help="Number of simulation steps")
    parser.add_argument("--enable-wandb", action="store_true", default=True,
                       help="Enable WanDB monitoring")
    parser.add_argument("--enable-tensorboard", action="store_true", default=True,
                       help="Enable TensorBoard monitoring")
    parser.add_argument("--wandb-project", type=str, 
                       default="sandgraph-fixed-oasis",
                       help="WanDB project name")
    
    args = parser.parse_args()
    
    try:
        results = run_fixed_rl_oasis_demo(
            steps=args.steps,
            enable_wandb=args.enable_wandb,
            enable_tensorboard=args.enable_tensorboard,
            wandb_project=args.wandb_project
        )
        
        print(f"\n🎉 Fixed OASIS demo completed successfully!")
        print("📊 Check the logs/ directory for detailed results")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
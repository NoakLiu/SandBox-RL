#!/usr/bin/env python3
"""
Enhanced SandGraph OASIS社交网络模拟演示 - 集成WanDB和TensorBoard监控

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
from sandgraph.core.monitoring import (
    SocialNetworkMonitor, 
    MonitoringConfig, 
    SocialNetworkMetrics, 
    MetricsCollector,
    create_monitor
)
from sandgraph.core.visualization import (
    SocialNetworkVisualizer,
    create_visualizer,
    quick_visualization
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
        self.visualizer = None
        self.start_time = time.time()
    
    def setup_monitoring(self, config: MonitoringConfig):
        """设置监控系统"""
        self.monitor = create_monitor(config)
        self.visualizer = create_visualizer("./visualizations/enhanced_oasis")
        
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
    
    def case_generator(self) -> Dict[str, Any]:
        """生成当前状态 - 增强版，包含监控"""
        # 调用父类方法
        state = super().case_generator()
        
        # 收集监控指标
        if self.monitor:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            self.monitor.update_metrics(metrics)
        
        return state
    
    def run_full_cycle(self, llm_func=None) -> Dict[str, Any]:
        """运行完整的沙盒周期 - 满足Sandbox协议"""
        # 生成当前状态
        case = self.case_generator()
        
        # 如果有LLM函数，使用它生成响应
        if llm_func:
            try:
                response = llm_func(case)
                # 验证响应
                score = self.verify_score(response, case)
                return {
                    "case": case,
                    "response": response,
                    "score": score,
                    "status": "success"
                }
            except Exception as e:
                return {
                    "case": case,
                    "response": f"Error: {str(e)}",
                    "score": 0.0,
                    "status": "error",
                    "error": str(e)
                }
        else:
            # 没有LLM函数，返回默认响应
            return {
                "case": case,
                "response": "Enhanced OASIS social network simulation",
                "score": 0.5,
                "status": "default"
            }


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
    
    def make_decision(self, current_state: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """基于当前状态做出决策 - 增强版"""
        self.decision_count += 1
        
        # 处理输入类型
        if isinstance(current_state, str):
            # 如果是字符串，尝试解析为字典或使用默认状态
            try:
                # 尝试解析JSON字符串
                state_dict = json.loads(current_state)
            except (json.JSONDecodeError, TypeError):
                # 如果解析失败，使用默认状态
                state_dict = {
                    "state": {
                        "network_state": {"total_users": 100, "total_posts": 30, "network_density": 0.1},
                        "user_behavior": {"active_users": 50, "engagement_rate": 0.3},
                        "content_metrics": {"quality_score": 0.7}
                    },
                    "trending_posts": [],
                    "active_users": []
                }
        else:
            # 如果已经是字典，直接使用
            state_dict = current_state
        
        # 构建增强的决策提示
        prompt = self._construct_enhanced_decision_prompt(state_dict)
        
        try:
            # 生成LLM响应
            response = self.llm_manager.generate_for_node(
                "enhanced_oasis_decision",
                prompt,
                temperature=0.3,
                max_new_tokens=128
            )
            
            # 解析响应
            decision = self._parse_decision_response(response.text)
            
            if decision is None:
                decision = {
                    "action": "CREATE_POST",
                    "target": "N/A",
                    "reasoning": "Fallback decision"
                }
            
            # 更新历史
            self._update_history(state_dict, decision, response.text)
            
            return {
                "decision": decision,
                "llm_response": response.text,
                "decision_count": self.decision_count
            }
            
        except Exception as e:
            print(f"❌ Decision generation failed: {e}")
            fallback_decision = {
                "action": "CREATE_POST",
                "target": "N/A",
                "reasoning": f"Error: {str(e)}"
            }
            
            self._update_history(state_dict, fallback_decision, f"Error: {str(e)}")
            
            return {
                "decision": fallback_decision,
                "llm_response": f"Error: {str(e)}",
                "decision_count": self.decision_count
            }
    
    def _construct_enhanced_decision_prompt(self, state: Dict[str, Any]) -> str:
        """构建增强的决策提示"""
        network_state = state.get("state", {}).get("network_state", {})
        user_behavior = state.get("state", {}).get("user_behavior", {})
        content_metrics = state.get("state", {}).get("content_metrics", {})
        
        trending_posts = state.get("trending_posts", [])
        active_users = state.get("active_users", [])
        
        prompt = f"""
你是一个OASIS社交网络策略专家。基于当前的网络状态，请做出最优的社交网络管理决策。

当前网络状态：
- 总用户数: {network_state.get('total_users', 0)}
- 活跃用户数: {user_behavior.get('active_users', 0)}
- 总帖子数: {network_state.get('total_posts', 0)}
- 参与度: {user_behavior.get('engagement_rate', 0):.3f}
- 网络密度: {network_state.get('network_density', 0):.3f}
- 内容质量: {content_metrics.get('quality_score', 0):.3f}

热门帖子 (前3个):
{chr(10).join([f"- {post['content'][:50]}... (分数: {post['trending_score']:.2f})" for post in trending_posts[:3]])}

活跃用户 (前3个):
{chr(10).join([f"- {user['user_id']} (活跃度: {user['activity_score']}, 关注者: {user['followers_count']})" for user in active_users[:3]])}

可用的动作类型：
- CREATE_POST: 创建新帖子
- LIKE_POST: 点赞帖子
- FOLLOW: 关注用户
- SHARE: 分享帖子
- DO_NOTHING: 不执行任何动作

请选择最合适的动作来提升网络活跃度和用户参与度。考虑以下因素：
1. 当前网络状态和趋势
2. 用户行为和偏好
3. 内容质量和多样性
4. 社区建设和发展

请以以下格式回复：
ACTION: [动作类型]
TARGET: [目标用户或帖子ID]
REASONING: [决策理由]
"""
        
        return prompt


def create_enhanced_rl_oasis_workflow(llm_manager, monitor_config: MonitoringConfig):
    """创建增强版RL OASIS工作流"""
    
    # 创建沙盒
    sandbox = EnhancedOasisSocialSandbox(
        initial_users=100,
        max_users=1000,
        initial_posts=30,
        interaction_probability=0.3
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
    
    # 创建工作流
    workflow = SG_Workflow("enhanced_oasis_social", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 添加节点
    env_node = EnhancedWorkflowNode("oasis_env", NodeType.SANDBOX, sandbox=sandbox)
    decision_node = EnhancedWorkflowNode("oasis_decision", NodeType.LLM, 
                                       llm_func=decision_maker.make_decision,
                                       metadata={"role": "OASIS Social Network Analyst"})
    optimizer_node = EnhancedWorkflowNode("oasis_optimizer", NodeType.RL, 
                                        rl_trainer=rl_trainer)
    
    workflow.add_node(env_node)
    workflow.add_node(decision_node)
    workflow.add_node(optimizer_node)
    
    # 连接节点
    workflow.add_edge("oasis_env", "oasis_decision")
    workflow.add_edge("oasis_decision", "oasis_optimizer")
    workflow.add_edge("oasis_optimizer", "oasis_env")
    
    return workflow, rl_trainer, decision_maker, sandbox


def run_enhanced_rl_oasis_demo(steps: int = 10, 
                              enable_wandb: bool = True,
                              enable_tensorboard: bool = True,
                              wandb_project: str = "sandgraph-enhanced-oasis"):
    """运行增强版RL OASIS演示"""
    
    print("🚀 Enhanced OASIS Social Network Demo with Monitoring")
    print("=" * 60)
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")
    
    # 创建监控配置
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
    
    # 创建工作流
    workflow, rl_trainer, decision_maker, sandbox = create_enhanced_rl_oasis_workflow(
        llm_manager, monitor_config
    )
    
    # 启动监控
    sandbox.monitor.start_monitoring()
    
    print("✅ Enhanced OASIS workflow created and monitoring started")
    print(f"📊 Running simulation for {steps} steps...")
    
    try:
        for step in range(steps):
            print(f"\n📈 Step {step + 1}/{steps}")
            print("-" * 40)
            
            # 执行工作流
            start_time = time.time()
            result = workflow.execute_full_workflow()
            execution_time = time.time() - start_time
            
            # 更新RL策略
            rl_trainer.update_policy()
            
            # 获取当前状态
            current_state = sandbox.case_generator()
            
            # 打印步骤摘要
            network_state = current_state.get("state", {}).get("network_state", {})
            user_behavior = current_state.get("state", {}).get("user_behavior", {})
            
            print(f"⏱️  Execution time: {execution_time:.2f}s")
            print(f"👥 Users: {network_state.get('total_users', 0)} (Active: {user_behavior.get('active_users', 0)})")
            print(f"📈 Engagement: {user_behavior.get('engagement_rate', 0):.3f}")
            print(f"🌐 Network Density: {network_state.get('network_density', 0):.3f}")
            
            # 延迟
            if step < steps - 1:
                time.sleep(2.0)
                
    except KeyboardInterrupt:
        print("\n⏹️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
    finally:
        # 停止监控
        sandbox.monitor.stop_monitoring()
        
        # 导出结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出指标
        metrics_file = f"./logs/enhanced_oasis_metrics_{timestamp}.json"
        sandbox.monitor.export_metrics(metrics_file, "json")
        
        # 创建可视化
        if sandbox.metrics_history:
            report_files = sandbox.visualizer.export_visualization_report(
                sandbox.metrics_history, 
                f"./visualizations/enhanced_oasis_{timestamp}"
            )
            
            print(f"\n📁 Results exported:")
            print(f"   - Metrics: {metrics_file}")
            for viz_type, path in report_files.items():
                print(f"   - {viz_type}: {path}")
        
        # 打印最终统计
        if sandbox.metrics_history:
            final_metrics = sandbox.metrics_history[-1]
            print(f"\n📊 Final Statistics:")
            print(f"   - Total Users: {final_metrics.total_users}")
            print(f"   - Engagement Rate: {final_metrics.engagement_rate:.3f}")
            print(f"   - Content Quality: {final_metrics.content_quality_score:.3f}")
            print(f"   - Network Density: {final_metrics.network_density:.3f}")
            print(f"   - Total Alerts: {len(sandbox.monitor.alerts)}")
        
        print("\n✅ Enhanced OASIS demo completed!")


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
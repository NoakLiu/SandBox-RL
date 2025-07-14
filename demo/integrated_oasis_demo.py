#!/usr/bin/env python3
"""
集成Oasis演示 - 在原始Oasis中集成自进化LLM功能
==============================================

这个演示展示了如何将自进化LLM功能集成到原始的Oasis社交网络模拟中，
实现模型在运行过程中的自我优化和进化。
"""

import sys
import os
import time
import json
import logging
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入原始Oasis和自进化Oasis模块
from sandgraph.core.self_evolving_oasis import (
    create_self_evolving_oasis,
    EvolutionStrategy,
    TaskType,
    SelfEvolvingConfig
)


class IntegratedOasisSystem:
    """集成Oasis系统 - 结合原始Oasis和自进化LLM功能"""
    
    def __init__(self, 
                 enable_evolution: bool = True,
                 evolution_strategy: str = "multi_model",
                 enable_lora: bool = True,
                 enable_kv_cache_compression: bool = True):
        
        self.enable_evolution = enable_evolution
        self.evolution_strategy = evolution_strategy
        self.enable_lora = enable_lora
        self.enable_kv_cache_compression = enable_kv_cache_compression
        
        # 初始化自进化LLM系统
        if self.enable_evolution:
            self.evolving_system = create_self_evolving_oasis(
                evolution_strategy=evolution_strategy,
                enable_lora=enable_lora,
                enable_kv_cache_compression=enable_kv_cache_compression,
                model_pool_size=5,
                evolution_interval=3
            )
            logger.info("自进化LLM系统已初始化")
        
        # 模拟原始Oasis的社交网络状态
        self.users = {}
        self.posts = {}
        self.interactions = []
        self.simulation_step = 0
        
        # 初始化网络
        self._initialize_network()
        
        logger.info("集成Oasis系统初始化完成")
    
    def _initialize_network(self):
        """初始化社交网络"""
        logger.info("初始化集成社交网络...")
        
        # 创建用户
        for i in range(50):
            user_id = f"user_{i}"
            self.users[user_id] = {
                "id": user_id,
                "interests": ["tech", "social", "news", "ai"],
                "activity_level": 0.7,
                "followers": [],
                "following": [],
                "beliefs": {"ai_positive": 0.5, "tech_savvy": 0.6},
                "engagement_history": []
            }
        
        # 创建连接
        for user_id in self.users:
            following_count = min(8, len(self.users) - 1)
            potential_follows = [uid for uid in self.users if uid != user_id]
            follows = random.sample(potential_follows, following_count)
            
            for follow_id in follows:
                self.users[user_id]["following"].append(follow_id)
                self.users[follow_id]["followers"].append(user_id)
        
        # 创建初始内容
        self._create_initial_content()
    
    def _create_initial_content(self):
        """创建初始内容"""
        post_templates = [
            "Exciting developments in AI technology! 🤖",
            "The future of social networks looks promising! 📱",
            "Interesting insights about online behavior patterns! 📊",
            "How AI is transforming our digital interactions! 💡",
            "The evolution of social media platforms! 🌐"
        ]
        
        for i in range(25):
            post_id = f"post_{i}"
            author_id = random.choice(list(self.users.keys()))
            content = random.choice(post_templates)
            
            self.posts[post_id] = {
                "id": post_id,
                "author_id": author_id,
                "content": content,
                "likes": random.randint(0, 20),
                "shares": random.randint(0, 5),
                "comments": [],
                "created_at": time.time(),
                "topic": "ai_technology"
            }
    
    def simulate_step(self) -> Dict[str, Any]:
        """执行一个模拟步骤"""
        self.simulation_step += 1
        logger.info(f"执行集成Oasis步骤 {self.simulation_step}")
        
        results = {
            "step": self.simulation_step,
            "network_state": self._get_network_state(),
            "evolution_results": {},
            "task_results": {}
        }
        
        # 如果启用自进化功能
        if self.enable_evolution:
            try:
                # 执行自进化Oasis步骤
                evolution_result = self.evolving_system.simulate_step()
                results["evolution_results"] = evolution_result
                
                # 使用自进化LLM处理特定任务
                task_results = self._process_evolution_tasks()
                results["task_results"] = task_results
                
                # 更新网络状态
                self._update_network_with_evolution(evolution_result)
                
            except Exception as e:
                logger.error(f"自进化步骤执行失败: {e}")
                results["evolution_error"] = str(e)
        
        # 执行原始Oasis逻辑
        self._execute_original_oasis_logic()
        
        return results
    
    def _process_evolution_tasks(self) -> Dict[str, Any]:
        """使用自进化LLM处理任务"""
        tasks = {}
        
        try:
            # 1. 内容生成任务
            content_result = self.evolving_system.evolving_llm.process_task(
                TaskType.CONTENT_GENERATION,
                f"Generate engaging content for step {self.simulation_step} about AI and social networks",
                {"user_count": len(self.users), "post_count": len(self.posts)}
            )
            tasks["content_generation"] = content_result
            
            # 2. 行为分析任务
            behavior_result = self.evolving_system.evolving_llm.process_task(
                TaskType.BEHAVIOR_ANALYSIS,
                "Analyze current user engagement and interaction patterns",
                {"active_users": self._get_active_users(), "interactions": len(self.interactions)}
            )
            tasks["behavior_analysis"] = behavior_result
            
            # 3. 网络优化任务
            network_result = self.evolving_system.evolving_llm.process_task(
                TaskType.NETWORK_OPTIMIZATION,
                "Suggest optimizations for network connectivity and user engagement",
                {"network_density": self._calculate_network_density()}
            )
            tasks["network_optimization"] = network_result
            
            # 4. 趋势预测任务
            trend_result = self.evolving_system.evolving_llm.process_task(
                TaskType.TREND_PREDICTION,
                "Predict upcoming trends in AI and social media",
                {"current_trends": self._get_current_trends()}
            )
            tasks["trend_prediction"] = trend_result
            
            # 5. 用户参与度任务
            engagement_result = self.evolving_system.evolving_llm.process_task(
                TaskType.USER_ENGAGEMENT,
                "Recommend strategies to increase user engagement and activity",
                {"engagement_metrics": self._get_engagement_metrics()}
            )
            tasks["user_engagement"] = engagement_result
            
        except Exception as e:
            logger.error(f"任务处理失败: {e}")
            tasks["error"] = str(e)
        
        return tasks
    
    def _update_network_with_evolution(self, evolution_result: Dict[str, Any]):
        """根据进化结果更新网络"""
        try:
            # 根据进化统计调整网络参数
            evolution_stats = evolution_result.get("evolution_stats", {})
            
            # 如果模型性能良好，增加网络活跃度
            if evolution_stats.get("recent_performance_avg", 0) > 0.7:
                for user_id in self.users:
                    self.users[user_id]["activity_level"] = min(1.0, 
                        self.users[user_id]["activity_level"] + 0.05)
            
            # 根据任务结果调整用户行为
            task_results = evolution_result.get("tasks", {})
            
            # 如果内容生成效果好，增加新帖子
            if "content_generation" in task_results:
                content_result = task_results["content_generation"]
                if "error" not in content_result and content_result.get("performance_score", 0) > 0.6:
                    self._create_new_post()
            
            # 如果网络优化效果好，增加连接
            if "network_optimization" in task_results:
                network_result = task_results["network_optimization"]
                if "error" not in network_result and network_result.get("performance_score", 0) > 0.6:
                    self._create_new_connections()
            
        except Exception as e:
            logger.error(f"网络更新失败: {e}")
    
    def _execute_original_oasis_logic(self):
        """执行原始Oasis逻辑"""
        # 模拟用户互动
        for post_id, post in self.posts.items():
            # 随机增加点赞和分享
            if random.random() < 0.3:
                post["likes"] += random.randint(1, 3)
            
            if random.random() < 0.1:
                post["shares"] += 1
        
        # 记录互动
        self.interactions.append({
            "type": "simulation_step",
            "step": self.simulation_step,
            "timestamp": time.time(),
            "total_users": len(self.users),
            "total_posts": len(self.posts)
        })
    
    def _create_new_post(self):
        """创建新帖子"""
        post_id = f"post_{len(self.posts)}"
        author_id = random.choice(list(self.users.keys()))
        
        # 使用自进化LLM生成内容
        try:
            content_result = self.evolving_system.evolving_llm.process_task(
                TaskType.CONTENT_GENERATION,
                "Generate a short, engaging social media post about AI technology",
                {"context": "social_network"}
            )
            
            if "error" not in content_result:
                content = content_result["response"].text[:100] + "..."  # 限制长度
            else:
                content = "AI technology is evolving rapidly! 🤖"
                
        except Exception:
            content = "AI technology is evolving rapidly! 🤖"
        
        self.posts[post_id] = {
            "id": post_id,
            "author_id": author_id,
            "content": content,
            "likes": 0,
            "shares": 0,
            "comments": [],
            "created_at": time.time(),
            "topic": "ai_technology",
            "generated_by_evolution": True
        }
    
    def _create_new_connections(self):
        """创建新连接"""
        # 随机选择一些用户创建新连接
        for _ in range(5):
            user1 = random.choice(list(self.users.keys()))
            user2 = random.choice(list(self.users.keys()))
            
            if user1 != user2 and user2 not in self.users[user1]["following"]:
                self.users[user1]["following"].append(user2)
                self.users[user2]["followers"].append(user1)
    
    def _get_network_state(self) -> Dict[str, Any]:
        """获取网络状态"""
        return {
            "total_users": len(self.users),
            "total_posts": len(self.posts),
            "total_interactions": len(self.interactions),
            "network_density": self._calculate_network_density(),
            "active_users": self._get_active_users(),
            "simulation_step": self.simulation_step
        }
    
    def _calculate_network_density(self) -> float:
        """计算网络密度"""
        total_connections = sum(len(user["following"]) for user in self.users.values())
        max_possible_connections = len(self.users) * (len(self.users) - 1)
        return total_connections / max_possible_connections if max_possible_connections > 0 else 0.0
    
    def _get_active_users(self) -> int:
        """获取活跃用户数"""
        return len([user for user in self.users.values() if user["activity_level"] > 0.5])
    
    def _get_current_trends(self) -> List[str]:
        """获取当前趋势"""
        return ["AI", "social media", "technology", "digital transformation"]
    
    def _get_engagement_metrics(self) -> Dict[str, Any]:
        """获取参与度指标"""
        total_likes = sum(post["likes"] for post in self.posts.values())
        total_shares = sum(post["shares"] for post in self.posts.values())
        
        return {
            "total_likes": total_likes,
            "total_shares": total_shares,
            "avg_likes_per_post": total_likes / len(self.posts) if self.posts else 0,
            "avg_shares_per_post": total_shares / len(self.posts) if self.posts else 0,
            "active_users": self._get_active_users()
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计"""
        stats = {
            "network": self._get_network_state(),
            "evolution_enabled": self.enable_evolution,
            "evolution_strategy": self.evolution_strategy,
            "lora_enabled": self.enable_lora,
            "kv_cache_compression_enabled": self.enable_kv_cache_compression
        }
        
        if self.enable_evolution:
            try:
                evolution_stats = self.evolving_system.evolving_llm.get_evolution_stats()
                stats["evolution"] = evolution_stats
            except Exception as e:
                stats["evolution_error"] = str(e)
        
        return stats
    
    def save_state(self, path: str) -> bool:
        """保存状态"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # 保存网络状态
            network_path = os.path.join(path, "integrated_network_state.json")
            with open(network_path, 'w') as f:
                json.dump({
                    "users": self.users,
                    "posts": self.posts,
                    "interactions": self.interactions,
                    "simulation_step": self.simulation_step,
                    "config": {
                        "enable_evolution": self.enable_evolution,
                        "evolution_strategy": self.evolution_strategy,
                        "enable_lora": self.enable_lora,
                        "enable_kv_cache_compression": self.enable_kv_cache_compression
                    }
                }, f, indent=2, default=str)
            
            # 保存进化状态
            if self.enable_evolution:
                evolution_path = os.path.join(path, "evolution")
                self.evolving_system.save_state(evolution_path)
            
            logger.info(f"集成状态已保存到: {path}")
            return True
            
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
            return False


def run_integrated_demo(steps: int = 15, save_path: str = "./data/integrated_oasis"):
    """运行集成演示"""
    
    print("🚀 集成Oasis系统演示")
    print("=" * 60)
    print("特性:")
    print("- 原始Oasis社交网络模拟")
    print("- 自进化LLM功能集成")
    print("- LoRA模型参数压缩")
    print("- KV缓存压缩")
    print("- 多模型协同")
    print("- 在线模型适配")
    print("=" * 60)
    
    # 创建集成Oasis系统
    integrated_system = IntegratedOasisSystem(
        enable_evolution=True,
        evolution_strategy="multi_model",
        enable_lora=True,
        enable_kv_cache_compression=True
    )
    
    # 执行模拟步骤
    results = []
    for step in range(steps):
        print(f"\n--- 步骤 {step + 1} ---")
        
        try:
            result = integrated_system.simulate_step()
            results.append(result)
            
            # 显示网络状态
            network_state = result['network_state']
            print(f"网络状态: 用户{network_state['total_users']}, 帖子{network_state['total_posts']}, 密度{network_state['network_density']:.3f}")
            
            # 显示进化结果
            if 'evolution_results' in result and result['evolution_results']:
                evolution_stats = result['evolution_results'].get('evolution_stats', {})
                print(f"进化步骤: {evolution_stats.get('evolution_step', 0)}")
                
                # 显示任务性能
                tasks = result.get('task_results', {})
                for task_name, task_result in tasks.items():
                    if isinstance(task_result, dict) and 'error' not in task_result:
                        print(f"  {task_name}: 性能 {task_result.get('performance_score', 0):.3f}")
            
        except Exception as e:
            print(f"步骤 {step + 1} 执行失败: {e}")
    
    # 保存状态
    if save_path:
        integrated_system.save_state(save_path)
        print(f"\n状态已保存到: {save_path}")
    
    # 显示最终统计
    final_stats = integrated_system.get_comprehensive_stats()
    
    print(f"\n=== 最终统计 ===")
    print(f"网络用户数: {final_stats['network']['total_users']}")
    print(f"网络帖子数: {final_stats['network']['total_posts']}")
    print(f"网络密度: {final_stats['network']['network_density']:.3f}")
    print(f"活跃用户数: {final_stats['network']['active_users']}")
    
    if 'evolution' in final_stats:
        evolution_stats = final_stats['evolution']
        print(f"进化步骤: {evolution_stats.get('evolution_step', 0)}")
        print(f"模型池大小: {evolution_stats.get('model_pool_size', 0)}")
        print(f"LoRA启用: {evolution_stats.get('lora_enabled', False)}")
        print(f"KV缓存压缩: {evolution_stats.get('kv_cache_compression_enabled', False)}")
    
    return results


if __name__ == "__main__":
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="集成Oasis系统演示")
    parser.add_argument("--steps", type=int, default=15, help="模拟步数")
    parser.add_argument("--save-path", type=str, default="./data/integrated_oasis", help="保存路径")
    parser.add_argument("--strategy", type=str, default="multi_model", 
                       choices=["gradient_based", "meta_learning", "adaptive_compression", "multi_model"],
                       help="进化策略")
    parser.add_argument("--no-evolution", action="store_true", help="禁用自进化功能")
    
    args = parser.parse_args()
    
    try:
        results = run_integrated_demo(args.steps, args.save_path)
        print("\n✅ 集成演示完成!")
        
    except Exception as e:
        print(f"\n❌ 集成演示失败: {e}")
        import traceback
        traceback.print_exc() 
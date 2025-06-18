from sandgraph.core.sandbox import Sandbox
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, EnhancedWorkflowNode
from sandgraph.core.workflow import NodeType
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig, RLAlgorithm
from typing import Dict, Any, List, Optional, Tuple
import random
import json
import os
import sys

class SocialNetworkEnvironment(Sandbox):
    """社交网络环境子集"""
    
    def __init__(self, oasis_interface):
        super().__init__("social_network", "社交网络模拟环境")
        self.oasis = oasis_interface
        self.network = self.oasis.get_network_state()
        self.posts = self.oasis.get_recent_posts()
        self.interactions = self.oasis.get_interactions()
        self.llm_weights: Dict[str, float] = {"decision": 1.0, "content": 1.0}  # LLM权重初始化
        
    def case_generator(self) -> Dict[str, Any]:
        """生成社交网络场景"""
        return {
            "network_state": self.network,
            "recent_posts": self.posts,
            "interaction_history": self.interactions[-10:] if self.interactions else [],
            "llm_weights": self.llm_weights
        }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """生成提示词"""
        return f"""基于以下社交网络状态，分析并生成下一步行动：
                网络状态：{json.dumps(case['network_state'], ensure_ascii=False)}
                最近发帖：{json.dumps(case['recent_posts'], ensure_ascii=False)}
                互动历史：{json.dumps(case['interaction_history'], ensure_ascii=False)}
                当前LLM权重：{json.dumps(case['llm_weights'], ensure_ascii=False)}

                请分析当前状态并决定：
                1. 是否需要建立新的社交连接
                2. 是否需要对某些帖子进行互动
                3. 是否需要发布新的内容
                4. 如何优化社交网络结构

                请给出具体的行动建议，格式如下：
                {{
                    "action_type": "new_connection|post_interaction|new_post|network_optimization",
                    "details": {{
                        // 具体行动细节
                    }},
                    "reasoning": "行动理由"
                }}"""
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证响应并计算得分"""
        try:
            action = json.loads(response)
            score = 0.0
            
            # 评估行动合理性
            if action.get("action_type") == "new_connection":
                score += 0.3
            elif action.get("action_type") == "post_interaction":
                score += 0.3
            elif action.get("action_type") == "new_post":
                score += 0.3
            elif action.get("action_type") == "network_optimization":
                score += 0.1
                
            # 评估行动理由的合理性
            if "reasoning" in action and len(action["reasoning"]) > 50:
                score += 0.2
                
            return score + format_score
        except:
            return 0.0
    
    def update_network_state(self, action: Dict[str, Any], llm_weights: Optional[Dict[str, float]] = None) -> None:
        """更新网络状态和LLM权重"""
        if "action_type" in action:
            if action["action_type"] == "new_connection":
                self.network["connections"].append(action["details"])
            elif action["action_type"] == "new_post":
                user_id = action["details"]["user_id"]
                if user_id not in self.posts:
                    self.posts[user_id] = []
                self.posts[user_id].append(action["details"]["content"])
            elif action["action_type"] == "post_interaction":
                self.interactions.append(action["details"])
        
        # 更新LLM权重
        if llm_weights is not None:
            self.llm_weights = llm_weights

def create_social_network_workflow(oasis_interface) -> Tuple[SG_Workflow, RLTrainer]:
    """创建社交网络工作流"""
    
    # 创建LLM管理器 - 使用默认的真实LLM配置
    llm_manager = create_shared_llm_manager(
        temperature=0.7,
        max_length=512,
        device="auto",
        torch_dtype="float16"  # 使用float16以节省显存
    )
    
    # 加载模型
    llm_manager.load_model()
    
    # 创建RL训练器
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
    rl_trainer = RLTrainer(rl_config, llm_manager)
    
    # 注册LLM节点
    llm_manager.register_node("decision_maker", {
        "role": "社交网络分析师",
        "reasoning_type": "strategic"
    })
    llm_manager.register_node("content_generator", {
        "role": "内容创作者",
        "reasoning_type": "creative"
    })
    
    # 创建工作流
    workflow = SG_Workflow("social_network_sim", WorkflowMode.TRADITIONAL, llm_manager)
    
    # 创建环境节点
    env = SocialNetworkEnvironment(oasis_interface)
    
    # 定义LLM函数
    def decision_maker_llm(prompt: str) -> str:
        response = llm_manager.generate_for_node("decision_maker", prompt)
        return response.text
    
    def content_generator_llm(prompt: str) -> str:
        response = llm_manager.generate_for_node("content_generator", prompt)
        return response.text
    
    # 添加节点
    network_env_node = EnhancedWorkflowNode("network_env", NodeType.SANDBOX, sandbox=env)
    decision_maker_node = EnhancedWorkflowNode("decision_maker", NodeType.LLM, llm_func=decision_maker_llm, metadata={"role": "社交网络分析师"})
    content_generator_node = EnhancedWorkflowNode("content_generator", NodeType.LLM, llm_func=content_generator_llm, metadata={"role": "内容创作者"})
    
    workflow.add_node(network_env_node)
    workflow.add_node(decision_maker_node)
    workflow.add_node(content_generator_node)
    
    # 连接节点
    workflow.add_edge("network_env", "decision_maker")
    workflow.add_edge("decision_maker", "content_generator")
    workflow.add_edge("content_generator", "network_env")
    
    return workflow, rl_trainer

def run_social_network_simulation(oasis_interface, steps: int = 10) -> List[Dict[str, Any]]:
    """运行社交网络模拟"""
    
    # 创建工作流和RL训练器
    workflow, rl_trainer = create_social_network_workflow(oasis_interface)
    
    # 执行工作流
    results = []
    for step in range(steps):
        print(f"\n执行第 {step + 1} 步...")
        
        # 获取当前状态
        env_node = workflow.nodes.get("network_env")
        if env_node is None or env_node.sandbox is None:
            print("环境节点不存在或无效")
            continue
            
        # 类型断言
        if not isinstance(env_node.sandbox, SocialNetworkEnvironment):
            print("环境节点类型错误")
            continue
            
        current_state = env_node.sandbox.case_generator()
        
        # 执行工作流
        workflow_result = workflow.execute_full_workflow()
        
        # 更新环境状态
        if "response" in workflow_result:
            try:
                action = json.loads(workflow_result["response"])
                
                # 使用RL更新LLM权重
                state = {
                    "network_size": len(current_state["network_state"]["users"]),
                    "connection_count": len(current_state["network_state"]["connections"]),
                    "post_count": len(current_state["recent_posts"]),
                    "interaction_count": len(current_state["interaction_history"]),
                    "decision_weight": current_state["llm_weights"]["decision"],
                    "content_weight": current_state["llm_weights"]["content"],
                    "action_size": len(action.get("details", {})),
                    "reasoning_size": len(action.get("reasoning", "")),
                    "step": step,
                    "score": workflow_result.get("score", 0.0)
                }
                
                # 打印LLM输入
                print(f"LLM输入: {json.dumps(state, ensure_ascii=False)}")
                
                # 添加经验到RL训练器
                rl_trainer.add_experience(
                    state=state,
                    action=json.dumps(action),
                    reward=workflow_result.get("score", 0.0),
                    done=step == steps - 1
                )
                
                # 更新策略
                update_result = rl_trainer.update_policy()
                
                # 打印LLM输出和动作选择
                print(f"LLM输出: {json.dumps(action, ensure_ascii=False)}")
                print(f"动作选择: {action.get('action_type')} - {action.get('reasoning')}")
                
                # 更新环境状态和LLM权重
                if update_result.get("status") == "updated":
                    new_weights = {
                        "decision": max(0.1, min(2.0, current_state["llm_weights"]["decision"] * (1 + update_result.get("policy_gradient", 0.0)))),
                        "content": max(0.1, min(2.0, current_state["llm_weights"]["content"] * (1 + update_result.get("value_gradient", 0.0))))
                    }
                    env_node.sandbox.update_network_state(action, new_weights)
                    
                    # 打印权重更新信息
                    print(f"LLM权重更新:")
                    print(f"  决策权重: {current_state['llm_weights']['decision']:.2f} -> {new_weights['decision']:.2f}")
                    print(f"  内容权重: {current_state['llm_weights']['content']:.2f} -> {new_weights['content']:.2f}")
                else:
                    print(f"策略更新状态: {update_result.get('status')}")
                    if update_result.get('status') == 'insufficient_data':
                        print(f"数据不足，当前轨迹数: {update_result.get('trajectory_count', 0)}")
                
            except Exception as e:
                print(f"更新状态时出错: {e}")
        
        # 获取更新后的状态
        current_state = env_node.sandbox.case_generator()
        results.append(current_state)
        
        # 打印训练统计
        stats = rl_trainer.get_training_stats()
        print(f"RL训练统计: 步骤 {stats['training_step']}, 算法 {stats['algorithm']}")
    
    return results

if __name__ == "__main__":
    # 创建OASIS接口
    class OASIS:
        def get_network_state(self) -> Dict[str, Any]:
            return {
                "users": [
                    {
                        "id": "user1",
                        "name": "Alice",
                        "profile": {
                            "age": 28,
                            "location": "New York",
                            "occupation": "Software Engineer",
                            "interests": ["tech", "music", "AI", "photography"],
                            "skills": ["Python", "Machine Learning", "Web Development"],
                            "education": ["Computer Science", "Data Science"],
                            "languages": ["English", "Chinese", "Spanish"]
                        },
                        "activity_level": 0.8,
                        "influence_score": 0.75,
                        "join_date": "2023-01-15"
                    },
                    {
                        "id": "user2",
                        "name": "Bob",
                        "profile": {
                            "age": 32,
                            "location": "San Francisco",
                            "occupation": "Data Scientist",
                            "interests": ["sports", "food", "data visualization", "hiking"],
                            "skills": ["R", "Statistics", "Data Analysis"],
                            "education": ["Statistics", "Business Analytics"],
                            "languages": ["English", "French"]
                        },
                        "activity_level": 0.6,
                        "influence_score": 0.65,
                        "join_date": "2023-02-20"
                    },
                    {
                        "id": "user3",
                        "name": "Carol",
                        "profile": {
                            "age": 25,
                            "location": "London",
                            "occupation": "UX Designer",
                            "interests": ["art", "travel", "design", "fashion"],
                            "skills": ["UI Design", "User Research", "Prototyping"],
                            "education": ["Design", "Psychology"],
                            "languages": ["English", "Italian"]
                        },
                        "activity_level": 0.9,
                        "influence_score": 0.85,
                        "join_date": "2023-03-10"
                    },
                    {
                        "id": "user4",
                        "name": "David",
                        "profile": {
                            "age": 35,
                            "location": "Berlin",
                            "occupation": "AI Researcher",
                            "interests": ["AI", "robotics", "philosophy", "chess"],
                            "skills": ["Deep Learning", "Computer Vision", "NLP"],
                            "education": ["Computer Science", "AI"],
                            "languages": ["English", "German", "Russian"]
                        },
                        "activity_level": 0.7,
                        "influence_score": 0.9,
                        "join_date": "2023-01-05"
                    },
                    {
                        "id": "user5",
                        "name": "Eve",
                        "profile": {
                            "age": 29,
                            "location": "Tokyo",
                            "occupation": "Content Creator",
                            "interests": ["gaming", "anime", "digital art", "streaming"],
                            "skills": ["Video Editing", "Content Strategy", "Social Media"],
                            "education": ["Media Studies", "Digital Marketing"],
                            "languages": ["English", "Japanese", "Korean"]
                        },
                        "activity_level": 0.95,
                        "influence_score": 0.8,
                        "join_date": "2023-02-01"
                    }
                ],
                "connections": [
                    {
                        "from": "user1",
                        "to": "user2",
                        "type": "friend",
                        "strength": 0.8,
                        "interaction_frequency": 0.7,
                        "common_interests": ["tech", "AI"],
                        "created_at": "2023-02-15"
                    },
                    {
                        "from": "user2",
                        "to": "user3",
                        "type": "colleague",
                        "strength": 0.6,
                        "interaction_frequency": 0.5,
                        "common_interests": ["design", "data visualization"],
                        "created_at": "2023-03-20"
                    },
                    {
                        "from": "user3",
                        "to": "user4",
                        "type": "mentor",
                        "strength": 0.9,
                        "interaction_frequency": 0.8,
                        "common_interests": ["AI", "art"],
                        "created_at": "2023-03-25"
                    },
                    {
                        "from": "user4",
                        "to": "user5",
                        "type": "collaborator",
                        "strength": 0.7,
                        "interaction_frequency": 0.6,
                        "common_interests": ["AI", "digital art"],
                        "created_at": "2023-04-01"
                    }
                ],
                "communities": [
                    {
                        "id": "tech_community",
                        "name": "Tech Enthusiasts",
                        "members": ["user1", "user2", "user4"],
                        "topics": ["AI", "Programming", "Data Science"],
                        "activity_level": 0.85
                    },
                    {
                        "id": "creative_community",
                        "name": "Creative Minds",
                        "members": ["user3", "user5"],
                        "topics": ["Art", "Design", "Digital Media"],
                        "activity_level": 0.9
                    }
                ],
                "trending_topics": [
                    {
                        "topic": "AI Ethics",
                        "engagement": 0.85,
                        "related_users": ["user1", "user4"],
                        "sentiment": "positive"
                    },
                    {
                        "topic": "Digital Art",
                        "engagement": 0.75,
                        "related_users": ["user3", "user5"],
                        "sentiment": "positive"
                    }
                ]
            }
        
        def get_recent_posts(self) -> Dict[str, List[Dict[str, Any]]]:
            return {
                "user1": [
                    {
                        "id": "post1",
                        "content": "Just finished implementing a new machine learning model for image recognition! #AI #ML",
                        "type": "achievement",
                        "timestamp": "2024-03-15T10:30:00",
                        "engagement": {
                            "likes": 45,
                            "comments": 12,
                            "shares": 8
                        },
                        "tags": ["AI", "ML", "Programming"],
                        "mentions": ["user4"],
                        "sentiment": "positive"
                    },
                    {
                        "id": "post2",
                        "content": "Beautiful sunset in New York today! 📸 #photography #NYC",
                        "type": "photo",
                        "timestamp": "2024-03-14T18:45:00",
                        "engagement": {
                            "likes": 78,
                            "comments": 15,
                            "shares": 5
                        },
                        "tags": ["Photography", "NYC", "Nature"],
                        "mentions": [],
                        "sentiment": "positive"
                    }
                ],
                "user2": [
                    {
                        "id": "post3",
                        "content": "New data visualization project: Analyzing global climate change patterns. Check it out! #DataScience #ClimateChange",
                        "type": "project",
                        "timestamp": "2024-03-15T09:15:00",
                        "engagement": {
                            "likes": 62,
                            "comments": 18,
                            "shares": 25
                        },
                        "tags": ["DataScience", "ClimateChange", "Visualization"],
                        "mentions": ["user1", "user4"],
                        "sentiment": "positive"
                    }
                ],
                "user3": [
                    {
                        "id": "post4",
                        "content": "Just launched my new design portfolio! Would love your feedback. #UXDesign #Portfolio",
                        "type": "portfolio",
                        "timestamp": "2024-03-15T11:20:00",
                        "engagement": {
                            "likes": 95,
                            "comments": 28,
                            "shares": 15
                        },
                        "tags": ["UXDesign", "Portfolio", "Design"],
                        "mentions": ["user5"],
                        "sentiment": "positive"
                    }
                ],
                "user4": [
                    {
                        "id": "post5",
                        "content": "Exciting breakthrough in our AI research! Paper coming soon. #AI #Research",
                        "type": "research",
                        "timestamp": "2024-03-15T08:45:00",
                        "engagement": {
                            "likes": 120,
                            "comments": 35,
                            "shares": 42
                        },
                        "tags": ["AI", "Research", "Science"],
                        "mentions": ["user1", "user2"],
                        "sentiment": "positive"
                    }
                ],
                "user5": [
                    {
                        "id": "post6",
                        "content": "Live streaming my latest digital art creation! Join me! #DigitalArt #Streaming",
                        "type": "stream",
                        "timestamp": "2024-03-15T12:00:00",
                        "engagement": {
                            "likes": 150,
                            "comments": 45,
                            "shares": 30
                        },
                        "tags": ["DigitalArt", "Streaming", "Art"],
                        "mentions": ["user3"],
                        "sentiment": "positive"
                    }
                ]
            }
        
        def get_interactions(self) -> List[Dict[str, Any]]:
            return [
                {
                    "id": "interaction1",
                    "from": "user1",
                    "to": "user4",
                    "type": "comment",
                    "content": "Great work on the AI model! Would love to collaborate on this.",
                    "post_id": "post5",
                    "timestamp": "2024-03-15T09:00:00",
                    "sentiment": "positive",
                    "engagement": {
                        "likes": 8,
                        "replies": 2
                    }
                },
                {
                    "id": "interaction2",
                    "from": "user2",
                    "to": "user1",
                    "type": "like",
                    "content": None,
                    "post_id": "post1",
                    "timestamp": "2024-03-15T10:35:00",
                    "sentiment": "positive",
                    "engagement": {
                        "likes": 0,
                        "replies": 0
                    }
                },
                {
                    "id": "interaction3",
                    "from": "user3",
                    "to": "user5",
                    "type": "share",
                    "content": "Amazing digital art!",
                    "post_id": "post6",
                    "timestamp": "2024-03-15T12:05:00",
                    "sentiment": "positive",
                    "engagement": {
                        "likes": 12,
                        "replies": 3
                    }
                },
                {
                    "id": "interaction4",
                    "from": "user4",
                    "to": "user2",
                    "type": "comment",
                    "content": "Your climate change visualization is impressive! Let's discuss potential collaboration.",
                    "post_id": "post3",
                    "timestamp": "2024-03-15T09:30:00",
                    "sentiment": "positive",
                    "engagement": {
                        "likes": 5,
                        "replies": 1
                    }
                },
                {
                    "id": "interaction5",
                    "from": "user5",
                    "to": "user3",
                    "type": "comment",
                    "content": "Your portfolio looks fantastic! Love your design style.",
                    "post_id": "post4",
                    "timestamp": "2024-03-15T11:25:00",
                    "sentiment": "positive",
                    "engagement": {
                        "likes": 7,
                        "replies": 2
                    }
                }
            ]
    
    # 运行模拟
    oasis = OASIS()
    results = run_social_network_simulation(oasis, steps=5) 
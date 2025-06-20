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
    """Social Network Environment Subset"""
    
    def __init__(self, oasis_interface):
        super().__init__("social_network", "Social Network Simulation Environment")
        self.oasis = oasis_interface
        self.network = self.oasis.get_network_state()
        self.posts = self.oasis.get_recent_posts()
        self.interactions = self.oasis.get_interactions()
        self.llm_weights: Dict[str, float] = {"decision": 1.0, "content": 1.0}  # LLM weights initialization
        
    def case_generator(self) -> Dict[str, Any]:
        """Generate social network scenario"""
        return {
            "network_state": self.network,
            "recent_posts": self.posts,
            "interaction_history": self.interactions[-10:] if self.interactions else [],
            "llm_weights": self.llm_weights
        }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """Generate prompt"""
        return f"""Based on the following social network state, analyze and generate the next action:
                Network State: {json.dumps(case['network_state'], ensure_ascii=False)}
                Recent Posts: {json.dumps(case['recent_posts'], ensure_ascii=False)}
                Interaction History: {json.dumps(case['interaction_history'], ensure_ascii=False)}
                Current LLM Weights: {json.dumps(case['llm_weights'], ensure_ascii=False)}

                Please analyze the current state and decide:
                1. Whether to establish new social connections
                2. Whether to interact with certain posts
                3. Whether to create new content
                4. How to optimize the social network structure

                Please provide specific action suggestions in the following format:
                {{
                    "action_type": "new_connection|post_interaction|new_post|network_optimization",
                    "details": {{
                        // Specific action details
                    }},
                    "reasoning": "Action reasoning"
                }}"""
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """Verify response and calculate score"""
        try:
            action = json.loads(response)
            score = 0.0
            
            # Evaluate action rationality
            if action.get("action_type") == "new_connection":
                score += 0.3
            elif action.get("action_type") == "post_interaction":
                score += 0.3
            elif action.get("action_type") == "new_post":
                score += 0.3
            elif action.get("action_type") == "network_optimization":
                score += 0.1
                
            # Evaluate action reasoning rationality
            if "reasoning" in action and len(action["reasoning"]) > 50:
                score += 0.2
                
            return score + format_score
        except:
            return 0.0
    
    def update_network_state(self, action: Dict[str, Any], llm_weights: Optional[Dict[str, float]] = None) -> None:
        """Update network state and LLM weights"""
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
        
        # Update LLM weights
        if llm_weights is not None:
            self.llm_weights = llm_weights

def create_social_network_workflow(oasis_interface) -> Tuple[SG_Workflow, RLTrainer]:
    """Create social network workflow"""
    
    # Create LLM manager - using Qwen-1.8B model
    llm_manager = create_shared_llm_manager(
        model_name="Qwen/Qwen-1_8B-Chat",  # Use Qwen-1.8B model
        backend="huggingface",
        temperature=0.7,
        max_length=512,
        device="auto",
        torch_dtype="float16"  # Use float16 to save memory
    )
    
    # Load model
    llm_manager.load_model()
    
    # Create RL trainer
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
    
    # Register LLM nodes
    llm_manager.register_node("decision_maker", {
        "role": "Social Network Analyst",
        "reasoning_type": "strategic"
    })
    llm_manager.register_node("content_generator", {
        "role": "Content Creator",
        "reasoning_type": "creative"
    })
    
    # Create workflow
    workflow = SG_Workflow("social_network_sim", WorkflowMode.TRADITIONAL, llm_manager)
    
    # Create environment node
    env = SocialNetworkEnvironment(oasis_interface)
    
    # Define LLM functions
    def decision_maker_llm(prompt: str) -> str:
        print(f"🤖 Decision maker LLM called with prompt: {prompt[:100]}...")
        response = llm_manager.generate_for_node("decision_maker", prompt)
        print(f"🤖 Decision maker LLM response: {response.text[:100]}...")
        return response.text
    
    def content_generator_llm(prompt: str) -> str:
        print(f"🤖 Content generator LLM called with prompt: {prompt[:100]}...")
        response = llm_manager.generate_for_node("content_generator", prompt)
        print(f"🤖 Content generator LLM response: {response.text[:100]}...")
        return response.text
    
    # Add nodes
    network_env_node = EnhancedWorkflowNode("network_env", NodeType.SANDBOX, sandbox=env)
    decision_maker_node = EnhancedWorkflowNode("decision_maker", NodeType.LLM, llm_func=decision_maker_llm, metadata={"role": "Social Network Analyst"})
    content_generator_node = EnhancedWorkflowNode("content_generator", NodeType.LLM, llm_func=content_generator_llm, metadata={"role": "Content Creator"})
    
    workflow.add_node(network_env_node)
    workflow.add_node(decision_maker_node)
    workflow.add_node(content_generator_node)
    
    # Connect nodes
    workflow.add_edge("network_env", "decision_maker")
    workflow.add_edge("decision_maker", "content_generator")
    workflow.add_edge("content_generator", "network_env")
    
    return workflow, rl_trainer

def run_social_network_simulation(oasis_interface, steps: int = 10) -> List[Dict[str, Any]]:
    """Run social network simulation"""
    
    # Create workflow and RL trainer
    workflow, rl_trainer = create_social_network_workflow(oasis_interface)
    
    # Execute workflow
    results = []
    for step in range(steps):
        print(f"\nExecuting step {step + 1}...")
        
        # Get current state
        env_node = workflow.nodes.get("network_env")
        if env_node is None or env_node.sandbox is None:
            print("Environment node does not exist or is invalid")
            continue
            
        # Type assertion
        if not isinstance(env_node.sandbox, SocialNetworkEnvironment):
            print("Environment node type error")
            continue
            
        current_state = env_node.sandbox.case_generator()
        print(f"Current state generated: {len(current_state['network_state']['users'])} users, {len(current_state['network_state']['connections'])} connections")
        
        # 手动执行工作流步骤
        print("Executing workflow manually...")
        
        # 1. 执行环境节点生成case和prompt
        env_result = env_node.execute({"action": "full_cycle"})
        print(f"Environment result keys: {list(env_result.keys())}")
        
        if "prompt" not in env_result:
            print("❌ Environment node did not generate prompt")
            continue
            
        prompt = env_result["prompt"]
        case = env_result["case"]
        print(f"✅ Generated prompt: {prompt[:100]}...")
        
        # 2. 执行决策LLM节点
        decision_node = workflow.nodes.get("decision_maker")
        if decision_node:
            decision_result = decision_node.execute({"prompt": prompt})
            print(f"Decision result keys: {list(decision_result.keys())}")
            
            if "response" in decision_result:
                llm_response = decision_result["response"]
                print(f"✅ Decision LLM response: {llm_response[:100]}...")
                
                # 3. 执行内容生成LLM节点
                content_node = workflow.nodes.get("content_generator")
                if content_node:
                    content_result = content_node.execute({"prompt": f"Based on the decision: {llm_response}, generate content."})
                    print(f"Content result keys: {list(content_result.keys())}")
                    
                    if "response" in content_result:
                        final_response = content_result["response"]
                        print(f"✅ Content LLM response: {final_response[:100]}...")
                        
                        # 处理LLM响应
                        try:
                            # 尝试解析JSON响应
                            if llm_response.strip().startswith('{'):
                                action = json.loads(llm_response)
                            else:
                                # 如果不是JSON，创建默认action
                                action = {
                                    "action_type": "network_optimization",
                                    "details": {"strategy": "default"},
                                    "reasoning": llm_response
                                }
                            
                            print(f"✅ LLM generated action: {action.get('action_type')}")
                            
                            # Use RL to update LLM weights
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
                                "score": 0.5  # 默认分数
                            }
                            
                            # Print LLM input
                            print(f"LLM Input: {json.dumps(state, ensure_ascii=False)}")
                            
                            # Add experience to RL trainer
                            rl_trainer.add_experience(
                                state=state,
                                action=json.dumps(action),
                                reward=0.5,
                                done=step == steps - 1
                            )
                            
                            # Update policy
                            update_result = rl_trainer.update_policy()
                            
                            # Print LLM output and action selection
                            print(f"LLM Output: {json.dumps(action, ensure_ascii=False)}")
                            print(f"Action Selection: {action.get('action_type')} - {action.get('reasoning')}")
                            
                            # Update environment state and LLM weights
                            if update_result.get("status") == "updated":
                                new_weights = {
                                    "decision": max(0.1, min(2.0, current_state["llm_weights"]["decision"] * (1 + update_result.get("policy_gradient", 0.0)))),
                                    "content": max(0.1, min(2.0, current_state["llm_weights"]["content"] * (1 + update_result.get("value_gradient", 0.0))))
                                }
                                env_node.sandbox.update_network_state(action, new_weights)
                                
                                # Print weight update information
                                print(f"LLM Weight Updates:")
                                print(f"  Decision Weight: {current_state['llm_weights']['decision']:.2f} -> {new_weights['decision']:.2f}")
                                print(f"  Content Weight: {current_state['llm_weights']['content']:.2f} -> {new_weights['content']:.2f}")
                            else:
                                print(f"Policy Update Status: {update_result.get('status')}")
                                if update_result.get('status') == 'insufficient_data':
                                    print(f"Insufficient data, current trajectory count: {update_result.get('trajectory_count', 0)}")
                            
                        except Exception as e:
                            print(f"Error processing LLM response: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("❌ Content LLM did not return response")
                else:
                    print("❌ Content generator node not found")
            else:
                print("❌ Decision LLM did not return response")
        else:
            print("❌ Decision maker node not found")
        
        # Get updated state
        current_state = env_node.sandbox.case_generator()
        results.append(current_state)
        
        # Print training statistics
        stats = rl_trainer.get_training_stats()
        print(f"RL Training Stats: Step {stats['training_step']}, Algorithm {stats['algorithm']}")
    
    return results

if __name__ == "__main__":
    # Create OASIS interface
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
    
    # Run simulation
    oasis = OASIS()
    results = run_social_network_simulation(oasis, steps=5) 
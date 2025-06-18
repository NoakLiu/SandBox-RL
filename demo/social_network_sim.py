from sandgraph.core.sandbox import Sandbox
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, EnhancedWorkflowNode
from sandgraph.core.workflow import NodeType
from sandgraph.core.llm_interface import create_shared_llm_manager
from typing import Dict, Any, List
import random
import json

class SocialNetworkEnvironment(Sandbox):
    """社交网络环境子集"""
    
    def __init__(self, oasis_interface):
        super().__init__("social_network", "社交网络模拟环境")
        self.oasis = oasis_interface
        self.network = {}  # 存储社交网络关系
        self.posts = {}    # 存储用户发帖
        self.interactions = []  # 存储互动记录
        
    def case_generator(self) -> Dict[str, Any]:
        """生成社交网络场景"""
        return {
            "network_state": self.network,
            "recent_posts": self.posts,
            "interaction_history": self.interactions[-10:] if self.interactions else []
        }
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """生成提示词"""
        return f"""基于以下社交网络状态，分析并生成下一步行动：
网络状态：{json.dumps(case['network_state'], ensure_ascii=False)}
最近发帖：{json.dumps(case['recent_posts'], ensure_ascii=False)}
互动历史：{json.dumps(case['interaction_history'], ensure_ascii=False)}

请分析当前状态并决定：
1. 是否需要建立新的社交连接
2. 是否需要对某些帖子进行互动
3. 是否需要发布新的内容
4. 如何优化社交网络结构

请给出具体的行动建议。"""
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """验证响应并计算得分"""
        try:
            action = json.loads(response)
            score = 0.0
            
            # 评估行动合理性
            if "new_connection" in action:
                score += 0.3
            if "post_interaction" in action:
                score += 0.3
            if "new_post" in action:
                score += 0.3
            if "network_optimization" in action:
                score += 0.1
                
            return score + format_score
        except:
            return 0.0

def create_social_network_workflow(oasis_interface):
    """创建社交网络工作流"""
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager("gpt-3.5-turbo")
    
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
        return llm_manager.generate_for_node("decision_maker", prompt).text
    
    def content_generator_llm(prompt: str) -> str:
        return llm_manager.generate_for_node("content_generator", prompt).text
    
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
    
    return workflow

def run_social_network_simulation(oasis_interface, steps: int = 10):
    """运行社交网络模拟"""
    
    # 创建工作流
    workflow = create_social_network_workflow(oasis_interface)
    
    # 执行工作流
    results = []
    for step in range(steps):
        print(f"\n执行第 {step + 1} 步...")
        result = workflow.execute_full_workflow()
        results.append(result)
        
        # 打印结果
        print(f"网络状态: {json.dumps(result.get('network_state', {}), ensure_ascii=False)}")
        print(f"新发帖: {json.dumps(result.get('new_posts', {}), ensure_ascii=False)}")
        print(f"互动: {json.dumps(result.get('interactions', []), ensure_ascii=False)}")
    
    return results

if __name__ == "__main__":
    # 示例：创建OASIS接口
    class MockOASIS:
        def get_network_state(self):
            return {"users": ["user1", "user2", "user3"]}
        
        def get_recent_posts(self):
            return {"user1": ["Hello world!"], "user2": ["Nice day!"]}
        
        def get_interactions(self):
            return [{"from": "user1", "to": "user2", "type": "like"}]
    
    # 运行模拟
    oasis = MockOASIS()
    results = run_social_network_simulation(oasis, steps=5) 
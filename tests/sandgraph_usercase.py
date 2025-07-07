# a general definition of SandGraphX usercase

from sandgraph.core.workflow import WorkflowGraph, WorkflowNode, NodeType
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sandbox import Sandbox

# 1. 定义你的 Sandbox
class MySandbox(Sandbox):
    def __init__(self, sandbox_id="my_sandbox", initial_state=None):
        super().__init__(sandbox_id)
        self.state = initial_state or {}
    
    def execute(self, action):
        """执行动作并返回新状态"""
        # 在这里实现你的 sandbox 逻辑
        new_state = self.state.copy()
        # ... 你的实现 ...
        return new_state
    
    def get_state(self):
        """返回当前状态"""
        return self.state
    
    def reset(self):
        """重置到初始状态"""
        self.state = {}
        
    def case_generator(self):
        """生成测试用例"""
        return {"state": self.state}
        
    def prompt_func(self, case):
        """生成提示信息"""
        return f"当前状态: {case['state']}"
        
    def verify_score(self, response, case, format_score=0.0):
        """验证分数"""
        return 1.0 if response else 0.0

# 2. 定义优化目标
def my_goal(state):
    """定义优化目标函数"""
    # 在这里实现你的目标函数
    score = 0
    # ... 你的实现 ...
    return score

# 3. 使用示例
def main():
    # 创建 sandbox
    sandbox = MySandbox(initial_state={"value": 0})
    
    # 创建 LLM 管理器
    llm_manager = create_shared_llm_manager("optimization_llm")
    
    # 创建工作流
    workflow = WorkflowGraph("optimization_workflow")
    
    # 添加输入节点
    input_node = WorkflowNode(
        node_id="input",
        node_type=NodeType.INPUT
    )
    workflow.add_node(input_node)
    
    # 添加执行节点
    executor_node = WorkflowNode(
        node_id="executor",
        node_type=NodeType.SANDBOX,
        sandbox=sandbox
    )
    workflow.add_node(executor_node)
    
    # 添加输出节点
    output_node = WorkflowNode(
        node_id="output",
        node_type=NodeType.OUTPUT
    )
    workflow.add_node(output_node)
    
    # 添加边
    workflow.add_edge("input", "executor")
    workflow.add_edge("executor", "output")
    
    # 运行优化
    result = workflow.execute({"action": "full_cycle"})
    
    print("优化结果:", result)

if __name__ == "__main__":
    main() 
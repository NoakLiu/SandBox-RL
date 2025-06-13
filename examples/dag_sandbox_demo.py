"""
DAG_Manager和SG_Workflow示例

展示如何使用DAG_Manager和SG_Workflow创建包含Sandbox节点的大图
"""

import asyncio
import time
from typing import Any, Dict
from sandgraph.core.advanced_workflow import DAG_Manager, create_dag_manager, ExecutionContext
from sandgraph.core.enhanced_workflow import SG_Workflow, WorkflowMode
from sandgraph.core.sandbox import Sandbox
from sandgraph.sandbox_implementations import Game24Sandbox

class SimpleSandbox(Sandbox):
    """简单的沙盒实现"""
    
    def __init__(self, sandbox_id: str, description: str = ""):
        super().__init__(sandbox_id, description)
    
    def case_generator(self) -> Dict[str, Any]:
        return {"value": 0}
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        return f"处理值: {case['value']}"
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        return 1.0

async def demo_large_sandbox_dag():
    """演示创建包含多个Sandbox节点的大图"""
    print("\n=== 创建大型Sandbox DAG示例 ===")
    
    # 创建Sandbox节点
    def create_sandbox_task(name: str) -> SimpleSandbox:
        return SimpleSandbox(
            sandbox_id=f"sandbox_{name}",
            description=f"这是一个{name}沙盒节点"
        )
    
    # 创建多个Sandbox节点
    sandbox_a = create_sandbox_task("A")
    sandbox_b = create_sandbox_task("B")
    sandbox_c = create_sandbox_task("C")
    sandbox_d = create_sandbox_task("D")
    sandbox_e = create_sandbox_task("E")
    sandbox_f = create_sandbox_task("F")
    
    # 使用DAG_Manager创建工作流
    dag = (create_dag_manager("large_sandbox_dag", "大型Sandbox DAG示例")
           .add_task_node("node_a", "节点A", sandbox_a.run_full_cycle)
           .add_task_node("node_b", "节点B", sandbox_b.run_full_cycle)
           .add_task_node("node_c", "节点C", sandbox_c.run_full_cycle)
           .add_task_node("node_d", "节点D", sandbox_d.run_full_cycle)
           .add_task_node("node_e", "节点E", sandbox_e.run_full_cycle)
           .add_task_node("node_f", "节点F", sandbox_f.run_full_cycle)
           .connect("node_a", "node_b")
           .connect("node_a", "node_c")
           .connect("node_b", "node_d")
           .connect("node_c", "node_d")
           .connect("node_d", "node_e")
           .connect("node_d", "node_f")
           .build())
    
    print("\n📊 DAG拓扑结构:")
    print(dag.visualize_graph())
    
    # 执行DAG
    context = await dag.execute({"initial_value": "start"})
    
    print("\n📈 执行摘要:")
    summary = dag.get_execution_summary()
    for key, value in summary.items():
        if key != "node_status":
            print(f"  {key}: {value}")
    
    print("\n🔍 节点执行状态:")
    for node_id, status in summary["node_status"].items():
        print(f"  {node_id}: {status['status']} (耗时: {status['duration']:.3f}s)")
    
    return dag

async def demo_sg_workflow():
    """演示使用SG_Workflow创建纯Sandbox工作流"""
    print("\n=== SG_Workflow纯Sandbox模式示例 ===")
    
    # 创建Sandbox节点
    def create_sandbox_task(name: str) -> SimpleSandbox:
        return SimpleSandbox(
            sandbox_id=f"sandbox_{name}",
            description=f"这是一个{name}沙盒节点"
        )
    
    # 创建多个Sandbox节点
    sandboxes = {
        "A": create_sandbox_task("A"),
        "B": create_sandbox_task("B"),
        "C": create_sandbox_task("C"),
        "D": create_sandbox_task("D"),
        "E": create_sandbox_task("E"),
        "F": create_sandbox_task("F")
    }
    
    # 创建SG_Workflow
    sg_workflow = SG_Workflow(
        workflow_id="pure_sandbox_workflow",
        mode=WorkflowMode.SANDBOX_ONLY
    )
    
    # 添加节点和连接
    for name, sandbox in sandboxes.items():
        sg_workflow.add_node(name, sandbox)
    
    # 创建复杂的连接关系
    sg_workflow.add_edge("A", "B")
    sg_workflow.add_edge("A", "C")
    sg_workflow.add_edge("B", "D")
    sg_workflow.add_edge("C", "D")
    sg_workflow.add_edge("D", "E")
    sg_workflow.add_edge("D", "F")
    
    print("\n📊 SG_Workflow结构:")
    print(sg_workflow.visualize())
    
    # 执行工作流
    result = await sg_workflow.execute()
    
    print("\n📈 执行结果:")
    print(f"  状态: {result.status}")
    print(f"  执行时间: {result.execution_time:.3f}秒")
    
    return sg_workflow

async def main():
    """主函数"""
    print("🚀 开始演示大型Sandbox工作流")
    
    # 运行DAG_Manager示例
    dag = await demo_large_sandbox_dag()
    
    # 运行SG_Workflow示例
    sg_workflow = await demo_sg_workflow()
    
    print("\n✨ 演示完成！")

if __name__ == "__main__":
    asyncio.run(main()) 
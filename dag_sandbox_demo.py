"""
DAG_Manager和SG_Workflow示例

展示如何使用DAG_Manager和SG_Workflow创建包含Sandbox节点的大图
"""

import asyncio
import time
import json
from typing import Any, Dict
from sandgraph.core.dag_manager import DAG_Manager, create_dag_manager, ExecutionContext
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, EnhancedWorkflowNode, NodeType
from sandgraph.core.sandbox import Sandbox
from sandgraph.core.llm_interface import SharedLLMManager, create_shared_llm_manager
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
    sandboxes = {
        "A": create_sandbox_task("A"),  # 入口节点
        "B": create_sandbox_task("B"),  # 第一层
        "C": create_sandbox_task("C"),  # 第一层
        "D": create_sandbox_task("D"),  # 第二层
        "E": create_sandbox_task("E"),  # 第二层
        "F": create_sandbox_task("F"),  # 第二层
        "G": create_sandbox_task("G"),  # 第三层
        "H": create_sandbox_task("H"),  # 第三层
        "I": create_sandbox_task("I"),  # 第三层
        "J": create_sandbox_task("J"),  # 第三层
        "K": create_sandbox_task("K"),  # 第四层
        "L": create_sandbox_task("L"),  # 第四层
        "M": create_sandbox_task("M"),  # 第四层
        "N": create_sandbox_task("N"),  # 第五层
        "O": create_sandbox_task("O"),  # 第五层
        "P": create_sandbox_task("P"),  # 第五层
        "Q": create_sandbox_task("Q"),  # 第五层
        "R": create_sandbox_task("R"),  # 第六层
        "S": create_sandbox_task("S"),  # 第六层
        "T": create_sandbox_task("T"),  # 第六层
        "U": create_sandbox_task("U"),  # 第七层
        "V": create_sandbox_task("V"),  # 第七层
        "W": create_sandbox_task("W"),  # 第八层
        "X": create_sandbox_task("X"),  # 第八层
        "Y": create_sandbox_task("Y"),  # 第九层
        "Z": create_sandbox_task("Z"),  # 出口节点
    }
    
    # 使用DAG_Manager创建工作流
    dag = create_dag_manager("complex_sandbox_dag", "复杂Sandbox DAG示例")
    
    # 添加所有节点
    for name, sandbox in sandboxes.items():
        dag.add_task_node(f"node_{name}", f"节点{name}", 
                         lambda ctx, data, s=sandbox: s.run_full_cycle())
    
    # 创建复杂的连接关系
    # 第一层连接
    dag.connect("node_A", "node_B")
    dag.connect("node_A", "node_C")
    
    # 第二层连接
    dag.connect("node_B", "node_D")
    dag.connect("node_B", "node_E")
    dag.connect("node_C", "node_E")
    dag.connect("node_C", "node_F")
    
    # 第三层连接
    dag.connect("node_D", "node_G")
    dag.connect("node_D", "node_H")
    dag.connect("node_E", "node_H")
    dag.connect("node_E", "node_I")
    dag.connect("node_F", "node_I")
    dag.connect("node_F", "node_J")
    
    # 第四层连接
    dag.connect("node_G", "node_K")
    dag.connect("node_H", "node_K")
    dag.connect("node_H", "node_L")
    dag.connect("node_I", "node_L")
    dag.connect("node_I", "node_M")
    dag.connect("node_J", "node_M")
    
    # 第五层连接
    dag.connect("node_K", "node_N")
    dag.connect("node_K", "node_O")
    dag.connect("node_L", "node_O")
    dag.connect("node_L", "node_P")
    dag.connect("node_M", "node_P")
    dag.connect("node_M", "node_Q")
    
    # 第六层连接
    dag.connect("node_N", "node_R")
    dag.connect("node_O", "node_R")
    dag.connect("node_O", "node_S")
    dag.connect("node_P", "node_S")
    dag.connect("node_P", "node_T")
    dag.connect("node_Q", "node_T")
    
    # 第七层连接
    dag.connect("node_R", "node_U")
    dag.connect("node_S", "node_U")
    dag.connect("node_S", "node_V")
    dag.connect("node_T", "node_V")
    
    # 第八层连接
    dag.connect("node_U", "node_W")
    dag.connect("node_U", "node_X")
    dag.connect("node_V", "node_X")
    
    # 第九层连接
    dag.connect("node_W", "node_Y")
    dag.connect("node_X", "node_Y")
    
    # 最终出口
    dag.connect("node_Y", "node_Z")
    
    # 构建DAG
    dag = dag.build()
    
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
    
    # 创建LLM管理器
    llm_manager = create_shared_llm_manager(
        model_name="mock_llm",
        backend="mock",
        temperature=0.7,
        max_length=512
    )
    
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
        graph_id="pure_sandbox_workflow",
        mode=WorkflowMode.SANDBOX_ONLY,
        llm_manager=llm_manager
    )
    
    # 添加节点和连接
    for name, sandbox in sandboxes.items():
        node = EnhancedWorkflowNode(
            node_id=name,
            node_type=NodeType.SANDBOX,
            sandbox=sandbox
        )
        sg_workflow.add_node(node)
    
    # 创建复杂的连接关系
    sg_workflow.add_edge("A", "B")
    sg_workflow.add_edge("A", "C")
    sg_workflow.add_edge("B", "D")
    sg_workflow.add_edge("C", "D")
    sg_workflow.add_edge("D", "E")
    sg_workflow.add_edge("D", "F")
    
    print("\n📊 SG_Workflow结构:")
    # 将set类型转换为list以支持JSON序列化
    stats = sg_workflow.get_game_stats()
    serializable_stats = {
        "nodes": [
            {
                "id": node["id"],
                "type": node["type"],
                "status": node["status"],
                "dependencies": list(node["dependencies"]),
                "successors": list(node["successors"])
            }
            for node in stats["nodes"]
        ],
        "edges": stats["edges"]
    }
    print(json.dumps(serializable_stats, indent=2))
    
    # 执行工作流
    result = sg_workflow.execute_full_workflow()
    
    print("\n📈 执行结果:")
    print(f"  状态: {result['status']}")
    print(f"  执行时间: {result['total_time']:.3f}秒")
    
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
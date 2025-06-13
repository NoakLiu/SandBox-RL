#!/usr/bin/env python3
"""
高级DAG工作流系统演示

展示SandGraph高级工作流系统的完整功能：
- 环路检测和拓扑排序
- 复杂控制流（条件、循环、并行）
- 多种停止条件
- 错误处理和恢复策略
- 状态管理和数据流控制
- 执行监控和调试

运行方式：
python advanced_workflow_demo.py
"""

import asyncio
import sys
import time
import random
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, '.')

try:
    from sandgraph.core.dag_manager import (
        DAG_Manager, create_dag_manager, ExecutionContext,
        NodeType, ExecutionStatus, StopConditionType,
        AdvancedWorkflowNode, StopCondition
    )
    from sandgraph import Game24Sandbox
    SANDGRAPH_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保SandGraph已正确安装")
    sys.exit(1)


def print_separator(title: str, width: int = 80):
    """打印分隔线"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


async def demo_basic_dag():
    """演示基础DAG工作流"""
    print_separator("基础DAG工作流演示")
    
    # 创建简单的任务函数
    def task_a(context: ExecutionContext, input_data: Any) -> str:
        print("  执行任务A")
        time.sleep(0.1)
        return "result_A"
    
    def task_b(context: ExecutionContext, input_data: Any) -> str:
        print("  执行任务B，依赖A的结果:", input_data.get('task_a', 'N/A'))
        time.sleep(0.1)
        return "result_B"
    
    def task_c(context: ExecutionContext, input_data: Any) -> str:
        print("  执行任务C，依赖A的结果:", input_data.get('task_a', 'N/A'))
        time.sleep(0.1) 
        return "result_C"
    
    def task_d(context: ExecutionContext, input_data: Any) -> str:
        print("  执行任务D，依赖B和C的结果")
        print(f"    B结果: {input_data.get('task_b', 'N/A')}")
        print(f"    C结果: {input_data.get('task_c', 'N/A')}")
        return "result_D"
    
    # 使用构建器创建工作流
    workflow = (create_dag_manager("basic_dag", "基础DAG演示")
                .add_task_node("task_a", "任务A", task_a)
                .add_task_node("task_b", "任务B", task_b)
                .add_task_node("task_c", "任务C", task_c)
                .add_task_node("task_d", "任务D", task_d)
                .connect("task_a", "task_b")
                .connect("task_a", "task_c")
                .connect("task_b", "task_d")
                .connect("task_c", "task_d")
                .build())
    
    print("📊 工作流拓扑排序:", workflow.topological_sort())
    
    # 执行工作流
    context = await workflow.execute({"initial_value": "start"})
    
    print("\n📈 执行摘要:")
    summary = workflow.get_execution_summary()
    for key, value in summary.items():
        if key != "node_status":
            print(f"  {key}: {value}")
    
    print("\n🔍 节点执行状态:")
    for node_id, status in summary["node_status"].items():
        print(f"  {node_id}: {status['status']} (耗时: {status['duration']:.3f}s)")
    
    return workflow


async def demo_cycle_detection():
    """演示环路检测功能"""
    print_separator("环路检测演示")
    
    def dummy_task(context: ExecutionContext, input_data: Any) -> str:
        return "dummy"
    
    workflow = create_dag_manager("cycle_test", "环路检测测试")
    
    # 添加节点
    workflow.add_task_node("node1", "节点1", dummy_task)
    workflow.add_task_node("node2", "节点2", dummy_task)
    workflow.add_task_node("node3", "节点3", dummy_task)
    
    # 添加正常边
    workflow.connect("node1", "node2")
    workflow.connect("node2", "node3")
    
    print("✅ 成功添加边: node1 -> node2 -> node3")
    
    # 尝试添加形成环路的边
    try:
        workflow.connect("node3", "node1")  # 这会形成环路
        print("❌ 环路检测失败！")
    except ValueError as e:
        print(f"✅ 成功检测到环路: {e}")
    
    # 验证图仍然有效
    topo_order = workflow.build().topological_sort()
    print(f"📊 拓扑排序: {topo_order}")


async def demo_condition_workflow():
    """演示条件分支工作流"""
    print_separator("条件分支工作流演示")
    
    def generate_number(context: ExecutionContext, input_data: Any) -> int:
        number = random.randint(1, 100)
        print(f"  生成随机数: {number}")
        context.global_state["random_number"] = number
        return number
    
    def check_even(context: ExecutionContext, input_data: Any) -> bool:
        number = context.global_state.get("random_number", 0)
        is_even = number % 2 == 0
        print(f"  检查 {number} 是否为偶数: {is_even}")
        return is_even
    
    def process_even(context: ExecutionContext, input_data: Any) -> str:
        print("  处理偶数")
        return "even_processed"
    
    def process_odd(context: ExecutionContext, input_data: Any) -> str:
        print("  处理奇数")
        return "odd_processed"
    
    def final_task(context: ExecutionContext, input_data: Any) -> str:
        print("  执行最终任务")
        return "completed"
    
    # 创建条件工作流
    workflow = (create_dag_manager("condition_flow", "条件分支演示")
                .add_task_node("generate", "生成数字", generate_number)
                .add_condition_node("check_even", "检查偶数", check_even, 
                                  true_branch="process_even", false_branch="process_odd")
                .add_task_node("process_even", "处理偶数", process_even)
                .add_task_node("process_odd", "处理奇数", process_odd)
                .add_task_node("final", "最终任务", final_task)
                .connect("generate", "check_even")
                .connect("check_even", "process_even")  # 条件为真时执行
                .connect("check_even", "process_odd")   # 条件为假时执行
                .connect("process_even", "final")
                .connect("process_odd", "final")
                .build())
    
    # 执行工作流
    context = await workflow.execute()
    
    print(f"\n🎯 条件分支结果:")
    for node_id, node_state in context.node_states.items():
        if node_state:
            print(f"  {node_id}: {node_state}")
    
    return workflow


async def demo_loop_workflow():
    """演示循环工作流"""
    print_separator("循环工作流演示")
    
    def initialize_counter(context: ExecutionContext, input_data: Any) -> int:
        context.global_state["counter"] = 0
        print("  初始化计数器: 0")
        return 0
    
    def increment_counter(context: ExecutionContext, input_data: Any) -> int:
        current = context.global_state.get("counter", 0)
        current += 1
        context.global_state["counter"] = current
        print(f"  计数器递增: {current}")
        return current
    
    def loop_condition(context: ExecutionContext, input_data: Any, iteration: int) -> bool:
        counter = context.global_state.get("counter", 0)
        should_continue = counter < 5
        print(f"  循环条件检查 (迭代 {iteration}): counter={counter}, 继续={should_continue}")
        return should_continue
    
    def finalize_result(context: ExecutionContext, input_data: Any) -> str:
        final_count = context.global_state.get("counter", 0)
        print(f"  循环完成，最终计数: {final_count}")
        return f"loop_completed_{final_count}"
    
    # 创建循环工作流
    workflow = (create_dag_manager("loop_flow", "循环演示")
                .add_task_node("init", "初始化", initialize_counter)
                .add_loop_node("loop_increment", "递增循环", increment_counter, 
                              loop_condition, max_iterations=10)
                .add_task_node("finalize", "完成", finalize_result)
                .connect("init", "loop_increment")
                .connect("loop_increment", "finalize")
                .build())
    
    # 执行工作流
    context = await workflow.execute()
    
    print(f"\n🔄 循环执行结果:")
    loop_node = workflow.nodes["loop_increment"]
    print(f"  循环次数: {len(loop_node.result) if loop_node.result else 0}")
    print(f"  最终状态: {context.global_state}")
    
    return workflow


async def demo_parallel_workflow():
    """演示并行执行工作流"""
    print_separator("并行执行工作流演示")
    
    def prepare_data(context: ExecutionContext, input_data: Any) -> List[int]:
        data = [1, 2, 3, 4, 5]
        print(f"  准备数据: {data}")
        return data
    
    def parallel_task_1(context: ExecutionContext, input_data: Any) -> str:
        print("  并行任务1开始执行")
        time.sleep(0.2)  # 模拟耗时操作
        result = "task1_completed"
        print(f"  并行任务1完成: {result}")
        return result
    
    def parallel_task_2(context: ExecutionContext, input_data: Any) -> str:
        print("  并行任务2开始执行")
        time.sleep(0.15)  # 模拟耗时操作
        result = "task2_completed"
        print(f"  并行任务2完成: {result}")
        return result
    
    def parallel_task_3(context: ExecutionContext, input_data: Any) -> str:
        print("  并行任务3开始执行")
        time.sleep(0.1)  # 模拟耗时操作
        result = "task3_completed"
        print(f"  并行任务3完成: {result}")
        return result
    
    def aggregate_results(context: ExecutionContext, input_data: Any) -> str:
        print("  聚合并行任务结果")
        parallel_results = input_data.get("parallel_tasks", [])
        print(f"  并行任务结果: {parallel_results}")
        return f"aggregated_{len(parallel_results)}_results"
    
    # 创建并行工作流
    workflow = (create_dag_manager("parallel_flow", "并行执行演示")
                .add_task_node("prepare", "准备数据", prepare_data)
                .add_parallel_node("parallel_tasks", "并行任务", 
                                 [parallel_task_1, parallel_task_2, parallel_task_3])
                .add_task_node("aggregate", "聚合结果", aggregate_results)
                .connect("prepare", "parallel_tasks")
                .connect("parallel_tasks", "aggregate")
                .build())
    
    # 执行工作流
    start_time = time.time()
    context = await workflow.execute()
    end_time = time.time()
    
    print(f"\n⚡ 并行执行效果:")
    print(f"  总耗时: {end_time - start_time:.3f}秒")
    print(f"  如果串行执行预计需要: 0.45秒 (0.2+0.15+0.1)")
    
    return workflow


async def demo_stop_conditions():
    """演示停止条件"""
    print_separator("停止条件演示")
    
    def long_running_task(context: ExecutionContext, input_data: Any) -> str:
        iteration = context.current_iteration
        print(f"  长时间任务执行中... (迭代 {iteration})")
        time.sleep(0.1)
        return f"iteration_{iteration}"
    
    # 演示1: 最大迭代次数停止
    print("🔄 演示1: 最大迭代次数停止条件")
    workflow1 = (create_dag_manager("max_iter_test", "最大迭代测试")
                 .add_task_node("task", "长时间任务", long_running_task)
                 .add_stop_condition(StopConditionType.MAX_ITERATIONS, 5)
                 .build())
    
    context1 = await workflow1.execute()
    print(f"  停止原因: {context1.stop_reason}")
    print(f"  执行迭代数: {context1.current_iteration}")
    
    # 演示2: 时间限制停止
    print("\n⏰ 演示2: 时间限制停止条件")
    workflow2 = (create_dag_manager("time_limit_test", "时间限制测试")
                 .add_task_node("task", "长时间任务", long_running_task)
                 .add_stop_condition(StopConditionType.TIME_LIMIT, 0.5)  # 0.5秒
                 .build())
    
    context2 = await workflow2.execute()
    print(f"  停止原因: {context2.stop_reason}")
    print(f"  执行时间: {time.time() - context2.start_time:.3f}秒")
    
    # 演示3: 自定义条件停止
    print("\n🎯 演示3: 自定义条件停止")
    def custom_stop_condition(context: ExecutionContext) -> bool:
        return context.current_iteration >= 3
    
    workflow3 = (create_dag_manager("custom_condition_test", "自定义条件测试")
                 .add_task_node("task", "长时间任务", long_running_task)
                 .add_stop_condition(StopConditionType.CONDITION_MET, custom_stop_condition)
                 .build())
    
    context3 = await workflow3.execute()
    print(f"  停止原因: {context3.stop_reason}")
    print(f"  执行迭代数: {context3.current_iteration}")


async def demo_error_handling():
    """演示错误处理和恢复"""
    print_separator("错误处理和恢复演示")
    
    def stable_task(context: ExecutionContext, input_data: Any) -> str:
        print("  稳定任务执行成功")
        return "stable_result"
    
    def unreliable_task(context: ExecutionContext, input_data: Any) -> str:
        # 模拟不稳定的任务
        if random.random() < 0.7:  # 70%概率失败
            raise Exception("模拟任务失败")
        print("  不稳定任务意外成功")
        return "unreliable_success"
    
    def cleanup_task(context: ExecutionContext, input_data: Any) -> str:
        print("  执行清理任务")
        return "cleanup_done"
    
    # 演示1: 重试机制
    print("🔄 演示1: 重试机制")
    workflow1 = (create_dag_manager("retry_test", "重试测试")
                 .add_task_node("stable", "稳定任务", stable_task)
                 .add_task_node("unreliable", "不稳定任务", unreliable_task, 
                               retry_count=3, retry_delay=0.1)
                 .add_task_node("cleanup", "清理任务", cleanup_task)
                 .connect("stable", "unreliable")
                 .connect("unreliable", "cleanup")
                 .build())
    
    try:
        context1 = await workflow1.execute()
        print("  工作流执行完成")
    except Exception as e:
        print(f"  工作流执行失败: {e}")
    
    # 演示2: 跳过失败节点
    print("\n⏭️ 演示2: 跳过失败节点")
    workflow2 = (create_dag_manager("skip_test", "跳过测试")
                 .add_task_node("stable", "稳定任务", stable_task)
                 .add_task_node("unreliable", "不稳定任务", unreliable_task, 
                               skip_on_failure=True)
                 .add_task_node("cleanup", "清理任务", cleanup_task)
                 .connect("stable", "unreliable")
                 .connect("unreliable", "cleanup")
                 .build())
    
    context2 = await workflow2.execute()
    
    print("\n📊 节点执行状态:")
    for node_id, node in workflow2.nodes.items():
        print(f"  {node_id}: {node.status.value}")


async def demo_sandgraph_integration():
    """演示与SandGraph沙盒的集成"""
    print_separator("SandGraph沙盒集成演示")
    
    def create_game24_task(context: ExecutionContext, input_data: Any) -> Dict[str, Any]:
        sandbox = Game24Sandbox()
        case = sandbox.case_generator()
        print(f"  生成Game24题目: {case}")
        context.global_state["game24_case"] = case
        return case
    
    def solve_game24(context: ExecutionContext, input_data: Any) -> str:
        case = context.global_state.get("game24_case", {})
        # 模拟LLM求解
        nums = case.get("puzzle", [6, 6, 6, 6])
        mock_solution = f"({nums[0]}+{nums[1]})+({nums[2]}+{nums[3]})"
        print(f"  模拟求解: {mock_solution}")
        return mock_solution
    
    def verify_solution(context: ExecutionContext, input_data: Any) -> bool:
        solution = input_data.get("solve_game24", "")
        case = context.global_state.get("game24_case", {})
        
        # 简单验证（实际应该使用沙盒的验证方法）
        is_valid = len(solution) > 0 and "+" in solution
        print(f"  验证结果: {'✅ 有效' if is_valid else '❌ 无效'}")
        return is_valid
    
    def report_result(context: ExecutionContext, input_data: Any) -> str:
        verification = input_data.get("verify", False)
        result = "成功" if verification else "失败"
        print(f"  最终报告: Game24求解{result}")
        return f"game24_{result}"
    
    # 创建集成工作流
    workflow = (create_dag_manager("sandgraph_integration", "SandGraph集成演示")
                .add_task_node("create_task", "创建任务", create_game24_task)
                .add_task_node("solve_game24", "求解Game24", solve_game24)
                .add_task_node("verify", "验证解答", verify_solution)
                .add_task_node("report", "报告结果", report_result)
                .connect("create_task", "solve_game24")
                .connect("solve_game24", "verify")
                .connect("verify", "report")
                .build())
    
    # 添加执行监听器
    def execution_listener(node_id: str, status: ExecutionStatus, context: ExecutionContext):
        print(f"  🔔 监听器: 节点 {node_id} 状态变更为 {status.value}")
    
    workflow.add_execution_listener(execution_listener)
    
    # 执行工作流
    context = await workflow.execute()
    
    print(f"\n📈 工作流执行摘要:")
    summary = workflow.get_execution_summary()
    print(f"  成功节点: {len([n for n in workflow.nodes.values() if n.status == ExecutionStatus.SUCCESS])}")
    print(f"  失败节点: {len([n for n in workflow.nodes.values() if n.status == ExecutionStatus.FAILED])}")
    print(f"  总执行时间: {summary['total_time']:.3f}秒")
    
    return workflow


async def demo_workflow_visualization():
    """演示工作流可视化"""
    print_separator("工作流可视化演示")
    
    # 创建一个复杂的工作流用于可视化
    def dummy_task(context: ExecutionContext, input_data: Any) -> str:
        return "dummy"
    
    workflow = (create_dag_manager("visualization_test", "可视化演示")
                .add_task_node("start", "开始任务", dummy_task)
                .add_condition_node("check", "条件检查", lambda c, d: True)
                .add_parallel_node("parallel", "并行处理", [dummy_task, dummy_task])
                .add_loop_node("loop", "循环任务", dummy_task, lambda c, d, i: i < 2)
                .add_task_node("end", "结束任务", dummy_task)
                .connect("start", "check")
                .connect("check", "parallel")
                .connect("parallel", "loop")
                .connect("loop", "end")
                .build())
    
    # 生成可视化JSON
    visualization = workflow.visualize_graph()
    print("🎨 工作流图结构:")
    print(visualization)
    
    return workflow


async def main():
    """主演示函数"""
    print("🚀 高级DAG工作流系统完整演示")
    print("=" * 80)
    
    demos = [
        ("基础DAG工作流", demo_basic_dag),
        ("环路检测", demo_cycle_detection),
        ("条件分支工作流", demo_condition_workflow),
        ("循环工作流", demo_loop_workflow),
        ("并行执行工作流", demo_parallel_workflow),
        ("停止条件", demo_stop_conditions),
        ("错误处理和恢复", demo_error_handling),
        ("SandGraph沙盒集成", demo_sandgraph_integration),
        ("工作流可视化", demo_workflow_visualization),
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n📍 开始演示: {demo_name}")
            result = await demo_func()
            results[demo_name] = {"status": "✅ 成功", "result": result}
            print(f"✅ {demo_name} 演示完成")
        except Exception as e:
            results[demo_name] = {"status": "❌ 失败", "error": str(e)}
            print(f"❌ {demo_name} 演示失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 演示总结
    print_separator("演示总结")
    
    print("📊 演示结果:")
    success_count = 0
    for demo_name, result in results.items():
        print(f"  {result['status']} {demo_name}")
        if "成功" in result['status']:
            success_count += 1
    
    print(f"\n🎯 成功率: {success_count}/{len(demos)} ({success_count/len(demos)*100:.1f}%)")
    
    print("\n💡 高级DAG工作流系统特性总结:")
    features = [
        "✅ 环路检测和拓扑排序",
        "✅ 条件分支和动态路径选择",
        "✅ 循环控制和迭代执行",
        "✅ 并行任务执行",
        "✅ 多种停止条件（时间、迭代、自定义）",
        "✅ 重试机制和错误恢复",
        "✅ 跳过失败节点策略",
        "✅ 执行监听和状态跟踪",
        "✅ 数据流管理和状态共享",
        "✅ 工作流可视化和调试",
        "✅ 与SandGraph沙盒无缝集成"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n🔧 实用建议:")
    print("  • 使用WorkflowBuilder构建复杂工作流")
    print("  • 设置合适的停止条件避免无限执行")
    print("  • 利用并行执行提高性能")
    print("  • 添加错误处理和重试机制")
    print("  • 使用执行监听器进行调试和监控")
    print("  • 集成到MCP客户端实现智能工作流")
    
    print("\n🎉 高级DAG工作流系统演示完成！")


if __name__ == "__main__":
    asyncio.run(main()) 
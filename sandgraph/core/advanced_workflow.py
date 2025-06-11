"""
高级DAG工作流系统

提供完善的有向无环图工作流执行能力，包括：
- 环路检测和拓扑排序
- 复杂控制流（条件、循环、并行）
- 多种停止条件
- 错误处理和恢复策略
- 状态管理和数据流控制
- 执行监控和调试
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Callable, Union, 
    Awaitable, Protocol, TypeVar, Generic
)
from abc import ABC, abstractmethod
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

T = TypeVar('T')

class NodeType(Enum):
    """节点类型枚举"""
    TASK = "task"                    # 普通任务节点
    CONDITION = "condition"          # 条件分支节点
    MERGE = "merge"                  # 合并节点
    LOOP = "loop"                    # 循环节点
    PARALLEL = "parallel"            # 并行执行节点
    BARRIER = "barrier"              # 同步屏障节点
    SUBPROCESS = "subprocess"        # 子流程节点
    TRANSFORM = "transform"          # 数据转换节点

class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"              # 等待执行
    RUNNING = "running"              # 正在执行
    SUCCESS = "success"              # 执行成功
    FAILED = "failed"                # 执行失败
    SKIPPED = "skipped"              # 被跳过
    CANCELLED = "cancelled"          # 被取消
    TIMEOUT = "timeout"              # 执行超时

class StopConditionType(Enum):
    """停止条件类型"""
    MAX_ITERATIONS = "max_iterations"    # 最大迭代次数
    CONDITION_MET = "condition_met"      # 条件满足
    ERROR_THRESHOLD = "error_threshold"  # 错误阈值
    TIME_LIMIT = "time_limit"           # 时间限制
    MANUAL_STOP = "manual_stop"         # 手动停止
    RESOURCE_LIMIT = "resource_limit"   # 资源限制

@dataclass
class ExecutionContext:
    """执行上下文"""
    workflow_id: str
    execution_id: str
    global_state: Dict[str, Any] = field(default_factory=dict)
    node_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    current_iteration: int = 0
    error_count: int = 0
    is_stopped: bool = False
    stop_reason: Optional[str] = None

@dataclass
class StopCondition:
    """停止条件配置"""
    condition_type: StopConditionType
    value: Any
    check_interval: float = 1.0
    
    def check(self, context: ExecutionContext) -> bool:
        """检查停止条件是否满足"""
        if self.condition_type == StopConditionType.MAX_ITERATIONS:
            return context.current_iteration >= self.value
        elif self.condition_type == StopConditionType.TIME_LIMIT:
            return time.time() - context.start_time >= self.value
        elif self.condition_type == StopConditionType.ERROR_THRESHOLD:
            return context.error_count >= self.value
        elif self.condition_type == StopConditionType.CONDITION_MET:
            if callable(self.value):
                return self.value(context)
            return bool(self.value)
        elif self.condition_type == StopConditionType.MANUAL_STOP:
            return context.is_stopped
        return False

class NodeProtocol(Protocol):
    """节点协议接口"""
    
    def execute(self, context: ExecutionContext, input_data: Any) -> Any:
        """执行节点任务"""
        ...
    
    async def execute_async(self, context: ExecutionContext, input_data: Any) -> Any:
        """异步执行节点任务"""
        ...

@dataclass
class AdvancedWorkflowNode:
    """高级工作流节点"""
    node_id: str
    name: str
    node_type: NodeType
    executor: Union[Callable, NodeProtocol, List[Callable]]  # 支持并行节点的多个执行器
    dependencies: Set[str] = field(default_factory=set)
    successors: Set[str] = field(default_factory=set)
    
    # 执行配置
    timeout: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 1.0
    skip_on_failure: bool = False
    parallel_execution: bool = False
    
    # 条件配置（用于条件节点）
    condition_func: Optional[Callable] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None
    
    # 循环配置（用于循环节点）
    loop_condition: Optional[Callable] = None
    max_loop_iterations: int = 100
    
    # 数据转换配置
    input_transform: Optional[Callable] = None
    output_transform: Optional[Callable] = None
    
    # 执行状态
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    execution_count: int = 0

class AdvancedWorkflowGraph:
    """高级DAG工作流图"""
    
    def __init__(self, graph_id: str, name: str = ""):
        self.graph_id = graph_id
        self.name = name or graph_id
        self.nodes: Dict[str, AdvancedWorkflowNode] = {}
        self.stop_conditions: List[StopCondition] = []
        self.global_timeout: Optional[float] = None
        self.max_parallel_tasks: int = 10
        self.execution_context: Optional[ExecutionContext] = None
        
        # 监控和调试
        self.execution_listeners: List[Callable] = []
        self.debug_mode: bool = False
        
    def add_node(self, node: AdvancedWorkflowNode) -> None:
        """添加节点"""
        if node.node_id in self.nodes:
            raise ValueError(f"节点 {node.node_id} 已存在")
        self.nodes[node.node_id] = node
        
    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加边（依赖关系）"""
        if from_node not in self.nodes:
            raise ValueError(f"源节点 {from_node} 不存在")
        if to_node not in self.nodes:
            raise ValueError(f"目标节点 {to_node} 不存在")
        
        self.nodes[from_node].successors.add(to_node)
        self.nodes[to_node].dependencies.add(from_node)
        
        # 检查是否形成环路
        if self._has_cycle():
            # 回滚操作
            self.nodes[from_node].successors.remove(to_node)
            self.nodes[to_node].dependencies.remove(from_node)
            raise ValueError(f"添加边 {from_node} -> {to_node} 会形成环路")
            
    def _has_cycle(self) -> bool:
        """使用DFS检测环路"""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}
        
        def dfs(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True  # 发现环路
            if colors[node_id] == BLACK:
                return False  # 已处理完成
                
            colors[node_id] = GRAY
            for successor in self.nodes[node_id].successors:
                if dfs(successor):
                    return True
            colors[node_id] = BLACK
            return False
        
        return any(dfs(node_id) for node_id in self.nodes if colors[node_id] == WHITE)
    
    def topological_sort(self) -> List[str]:
        """拓扑排序"""
        if self._has_cycle():
            raise ValueError("图中存在环路，无法进行拓扑排序")
        
        in_degree = {node_id: len(node.dependencies) for node_id, node in self.nodes.items()}
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for successor in self.nodes[node_id].successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        
        if len(result) != len(self.nodes):
            raise ValueError("拓扑排序失败，可能存在环路")
        
        return result
    
    def add_stop_condition(self, condition: StopCondition) -> None:
        """添加停止条件"""
        self.stop_conditions.append(condition)
    
    def _check_stop_conditions(self, context: ExecutionContext) -> bool:
        """检查所有停止条件"""
        for condition in self.stop_conditions:
            if condition.check(context):
                context.is_stopped = True
                context.stop_reason = f"停止条件满足: {condition.condition_type.value}"
                return True
        return False
    
    def _get_ready_nodes(self, context: ExecutionContext) -> List[str]:
        """获取可以执行的节点"""
        ready_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.status != ExecutionStatus.PENDING:
                continue
                
            # 检查所有依赖是否已完成
            dependencies_ready = True
            for dep_id in node.dependencies:
                dep_node = self.nodes[dep_id]
                if dep_node.status not in [ExecutionStatus.SUCCESS, ExecutionStatus.SKIPPED]:
                    dependencies_ready = False
                    break
            
            if dependencies_ready:
                ready_nodes.append(node_id)
        
        return ready_nodes
    
    async def _execute_node(self, node_id: str, context: ExecutionContext) -> Any:
        """执行单个节点"""
        node = self.nodes[node_id]
        node.status = ExecutionStatus.RUNNING
        node.start_time = time.time()
        node.execution_count += 1
        
        try:
            # 获取输入数据
            input_data = self._get_node_input_data(node_id, context)
            
            # 输入数据转换
            if node.input_transform:
                input_data = node.input_transform(input_data)
            
            # 根据节点类型执行
            if node.node_type == NodeType.CONDITION:
                result = await self._execute_condition_node(node, context, input_data)
            elif node.node_type == NodeType.LOOP:
                result = await self._execute_loop_node(node, context, input_data)
            elif node.node_type == NodeType.PARALLEL:
                result = await self._execute_parallel_node(node, context, input_data)
            else:
                result = await self._execute_task_node(node, context, input_data)
            
            # 输出数据转换
            if node.output_transform:
                result = node.output_transform(result)
            
            node.result = result
            node.status = ExecutionStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"节点 {node_id} 执行失败: {e}")
            node.error = e
            node.status = ExecutionStatus.FAILED
            context.error_count += 1
            
            if not node.skip_on_failure:
                raise
            else:
                node.status = ExecutionStatus.SKIPPED
        
        finally:
            node.end_time = time.time()
            
            # 记录执行历史
            context.execution_history.append({
                "node_id": node_id,
                "status": node.status.value,
                "start_time": node.start_time,
                "end_time": node.end_time,
                "duration": node.end_time - node.start_time if node.start_time else 0,
                "execution_count": node.execution_count,
                "error": str(node.error) if node.error else None
            })
            
            # 通知监听器
            for listener in self.execution_listeners:
                try:
                    listener(node_id, node.status, context)
                except Exception as e:
                    logger.warning(f"执行监听器失败: {e}")
        
        return node.result
    
    async def _execute_task_node(self, node: AdvancedWorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行任务节点"""
        executor = node.executor
        
        # 对于并行节点，executor可能是List类型，这里需要单个执行器
        if isinstance(executor, list):
            raise ValueError(f"任务节点 {node.node_id} 的执行器不能是列表类型")
        
        # 重试机制
        for attempt in range(node.retry_count + 1):
            try:
                if hasattr(executor, 'execute_async'):
                    result = await executor.execute_async(context, input_data)
                elif hasattr(executor, 'execute'):
                    result = executor.execute(context, input_data)
                elif callable(executor):
                    result = executor(context, input_data)
                else:
                    raise ValueError(f"无效的执行器类型: {type(executor)}")
                
                return result
                
            except Exception as e:
                if attempt < node.retry_count:
                    logger.warning(f"节点 {node.node_id} 第 {attempt + 1} 次尝试失败，{node.retry_delay}秒后重试: {e}")
                    await asyncio.sleep(node.retry_delay)
                else:
                    raise
    
    async def _execute_condition_node(self, node: AdvancedWorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行条件节点"""
        if not node.condition_func:
            raise ValueError(f"条件节点 {node.node_id} 缺少条件函数")
        
        condition_result = node.condition_func(context, input_data)
        if isinstance(condition_result, bool):
            result = condition_result
        else:
            result = bool(condition_result)
        
        # 根据条件结果设置后续执行路径
        if result and node.true_branch:
            context.node_states[node.node_id] = {"branch": "true", "next_node": node.true_branch}
        elif not result and node.false_branch:
            context.node_states[node.node_id] = {"branch": "false", "next_node": node.false_branch}
        
        return result
    
    async def _execute_loop_node(self, node: AdvancedWorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行循环节点"""
        if not node.loop_condition:
            raise ValueError(f"循环节点 {node.node_id} 缺少循环条件")
        
        results = []
        iteration = 0
        
        while iteration < node.max_loop_iterations:
            # 检查循环条件
            if not node.loop_condition(context, input_data, iteration):
                break
            
            # 执行循环体
            loop_result = await self._execute_task_node(node, context, input_data)
            results.append(loop_result)
            
            iteration += 1
            
            # 检查全局停止条件
            if self._check_stop_conditions(context):
                break
        
        return results
    
    async def _execute_parallel_node(self, node: AdvancedWorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行并行节点"""
        executor = node.executor
        if not isinstance(executor, list):
            raise ValueError(f"并行节点 {node.node_id} 的执行器必须是列表类型")
        
        tasks = []
        for i, single_executor in enumerate(executor):
            task = asyncio.create_task(
                self._execute_single_parallel_task(f"{node.node_id}_parallel_{i}", single_executor, context, input_data)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _execute_single_parallel_task(self, task_id: str, executor: Callable, context: ExecutionContext, input_data: Any) -> Any:
        """执行单个并行任务"""
        if hasattr(executor, 'execute_async'):
            return await executor.execute_async(context, input_data)
        elif hasattr(executor, 'execute'):
            return executor.execute(context, input_data)
        elif callable(executor):
            return executor(context, input_data)
        else:
            raise ValueError(f"无效的并行任务执行器: {type(executor)}")
    
    def _get_node_input_data(self, node_id: str, context: ExecutionContext) -> Any:
        """获取节点的输入数据"""
        node = self.nodes[node_id]
        
        if not node.dependencies:
            # 没有依赖的节点使用全局状态
            return context.global_state
        
        # 收集所有依赖节点的输出
        input_data = {}
        for dep_id in node.dependencies:
            dep_node = self.nodes[dep_id]
            if dep_node.status == ExecutionStatus.SUCCESS:
                input_data[dep_id] = dep_node.result
        
        return input_data
    
    async def execute(self, initial_data: Optional[Dict[str, Any]] = None) -> ExecutionContext:
        """执行工作流"""
        # 初始化执行上下文
        self.execution_context = ExecutionContext(
            workflow_id=self.graph_id,
            execution_id=str(uuid.uuid4()),
            global_state=initial_data or {}
        )
        
        context = self.execution_context
        
        try:
            logger.info(f"开始执行工作流 {self.graph_id}")
            
            # 检查图的有效性
            if self._has_cycle():
                raise ValueError("工作流图包含环路")
            
            # 获取拓扑排序（用于调试和验证）
            topo_order = self.topological_sort()
            logger.debug(f"拓扑排序: {topo_order}")
            
            # 执行主循环
            while True:
                # 检查停止条件
                if self._check_stop_conditions(context):
                    logger.info(f"工作流停止: {context.stop_reason}")
                    break
                
                # 获取可执行的节点
                ready_nodes = self._get_ready_nodes(context)
                
                if not ready_nodes:
                    # 检查是否所有节点都已完成
                    all_completed = all(
                        node.status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
                        for node in self.nodes.values()
                    )
                    
                    if all_completed:
                        logger.info("所有节点执行完成")
                        break
                    else:
                        # 可能存在死锁或等待条件
                        logger.warning("没有可执行的节点，但工作流未完成")
                        await asyncio.sleep(0.1)
                        continue
                
                # 并行执行准备好的节点
                tasks = []
                for node_id in ready_nodes[:self.max_parallel_tasks]:
                    task = asyncio.create_task(self._execute_node(node_id, context))
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                context.current_iteration += 1
                await asyncio.sleep(0.01)  # 短暂休眠，避免CPU占用过高
            
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            context.is_stopped = True
            context.stop_reason = f"执行错误: {str(e)}"
            raise
        
        finally:
            # 计算执行统计
            context.execution_history.append({
                "workflow_summary": {
                    "total_nodes": len(self.nodes),
                    "successful_nodes": sum(1 for node in self.nodes.values() if node.status == ExecutionStatus.SUCCESS),
                    "failed_nodes": sum(1 for node in self.nodes.values() if node.status == ExecutionStatus.FAILED),
                    "skipped_nodes": sum(1 for node in self.nodes.values() if node.status == ExecutionStatus.SKIPPED),
                    "total_time": time.time() - context.start_time,
                    "total_iterations": context.current_iteration,
                    "total_errors": context.error_count
                }
            })
            
            logger.info(f"工作流 {self.graph_id} 执行完成")
        
        return context
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        if not self.execution_context:
            return {"status": "not_executed"}
        
        context = self.execution_context
        
        return {
            "workflow_id": self.graph_id,
            "execution_id": context.execution_id,
            "status": "stopped" if context.is_stopped else "running",
            "stop_reason": context.stop_reason,
            "total_time": time.time() - context.start_time,
            "iterations": context.current_iteration,
            "error_count": context.error_count,
            "node_status": {
                node_id: {
                    "status": node.status.value,
                    "execution_count": node.execution_count,
                    "duration": (node.end_time - node.start_time) if node.start_time and node.end_time else 0
                }
                for node_id, node in self.nodes.items()
            }
        }
    
    def add_execution_listener(self, listener: Callable) -> None:
        """添加执行监听器"""
        self.execution_listeners.append(listener)
    
    def visualize_graph(self) -> str:
        """生成图的可视化表示"""
        graph_info = {
            "nodes": [
                {
                    "id": node_id,
                    "name": node.name,
                    "type": node.node_type.value,
                    "status": node.status.value,
                    "dependencies": list(node.dependencies),
                    "successors": list(node.successors)
                }
                for node_id, node in self.nodes.items()
            ],
            "edges": [
                {"from": node_id, "to": successor}
                for node_id, node in self.nodes.items()
                for successor in node.successors
            ]
        }
        
        return json.dumps(graph_info, indent=2, ensure_ascii=False)


# 工厂函数和辅助类

class WorkflowBuilder:
    """工作流构建器"""
    
    def __init__(self, graph_id: str, name: str = ""):
        self.workflow = AdvancedWorkflowGraph(graph_id, name)
    
    def add_task_node(self, node_id: str, name: str, executor: Callable, **kwargs) -> 'WorkflowBuilder':
        """添加任务节点"""
        node = AdvancedWorkflowNode(
            node_id=node_id,
            name=name,
            node_type=NodeType.TASK,
            executor=executor,
            **kwargs
        )
        self.workflow.add_node(node)
        return self
    
    def add_condition_node(self, node_id: str, name: str, condition_func: Callable, 
                          true_branch: Optional[str] = None, false_branch: Optional[str] = None, **kwargs) -> 'WorkflowBuilder':
        """添加条件节点"""
        node = AdvancedWorkflowNode(
            node_id=node_id,
            name=name,
            node_type=NodeType.CONDITION,
            executor=condition_func,
            condition_func=condition_func,
            true_branch=true_branch,
            false_branch=false_branch,
            **kwargs
        )
        self.workflow.add_node(node)
        return self
    
    def add_loop_node(self, node_id: str, name: str, executor: Callable, 
                     loop_condition: Callable, max_iterations: int = 100, **kwargs) -> 'WorkflowBuilder':
        """添加循环节点"""
        node = AdvancedWorkflowNode(
            node_id=node_id,
            name=name,
            node_type=NodeType.LOOP,
            executor=executor,
            loop_condition=loop_condition,
            max_loop_iterations=max_iterations,
            **kwargs
        )
        self.workflow.add_node(node)
        return self
    
    def add_parallel_node(self, node_id: str, name: str, executors: List[Callable], **kwargs) -> 'WorkflowBuilder':
        """添加并行节点"""
        node = AdvancedWorkflowNode(
            node_id=node_id,
            name=name,
            node_type=NodeType.PARALLEL,
            executor=executors,
            **kwargs
        )
        self.workflow.add_node(node)
        return self
    
    def connect(self, from_node: str, to_node: str) -> 'WorkflowBuilder':
        """连接节点"""
        self.workflow.add_edge(from_node, to_node)
        return self
    
    def add_stop_condition(self, condition_type: StopConditionType, value: Any, **kwargs) -> 'WorkflowBuilder':
        """添加停止条件"""
        condition = StopCondition(condition_type, value, **kwargs)
        self.workflow.add_stop_condition(condition)
        return self
    
    def set_global_timeout(self, timeout: float) -> 'WorkflowBuilder':
        """设置全局超时"""
        self.workflow.global_timeout = timeout
        return self
    
    def set_max_parallel_tasks(self, max_tasks: int) -> 'WorkflowBuilder':
        """设置最大并行任务数"""
        self.workflow.max_parallel_tasks = max_tasks
        return self
    
    def build(self) -> AdvancedWorkflowGraph:
        """构建工作流"""
        return self.workflow


def create_advanced_workflow(graph_id: str, name: str = "") -> WorkflowBuilder:
    """创建高级工作流的工厂函数"""
    return WorkflowBuilder(graph_id, name) 
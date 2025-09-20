#!/usr/bin/env python3
"""
Unified Workflow Engine - 统一工作流引擎
======================================

集成所有工作流和执行相关功能：
1. DAG工作流管理
2. 节点执行和调度
3. 异步任务处理
4. 轨迹数据管理
5. 训练服务器架构
"""

import asyncio
import logging
import time
import uuid
import threading
import queue
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型"""
    INPUT = "input"
    SANDBOX = "sandbox"
    LLM = "llm"
    RL = "rl"
    OUTPUT = "output"
    AGGREGATOR = "aggregator"
    TASK = "task"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StopConditionType(Enum):
    """停止条件类型"""
    MAX_ITERATIONS = "max_iterations"
    CONDITION_MET = "condition_met"
    ERROR_THRESHOLD = "error_threshold"
    TIME_LIMIT = "time_limit"
    MANUAL_STOP = "manual_stop"


@dataclass
class TrajectoryStep:
    """轨迹步骤"""
    state: Dict[str, Any]
    action: Any
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """轨迹"""
    agent_id: str
    episode_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "episode_id": self.episode_id,
            "steps": [asdict(s) for s in self.steps],
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trajectory":
        steps = [TrajectoryStep(**s) for s in d.get("steps", [])]
        return Trajectory(
            agent_id=d["agent_id"],
            episode_id=d["episode_id"],
            steps=steps,
            metadata=d.get("metadata", {})
        )


@dataclass
class Sample:
    """训练样本"""
    sample_id: str
    payload: Dict[str, Any]


@dataclass
class Result:
    """结果"""
    sample_id: str
    agent_id: str
    trajectory: Dict[str, Any]


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
    """停止条件"""
    condition_type: StopConditionType
    value: Any
    check_interval: float = 1.0
    
    def check(self, context: ExecutionContext) -> bool:
        """检查停止条件"""
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
    """节点协议"""
    
    def execute(self, context: ExecutionContext, input_data: Any) -> Any:
        """执行节点"""
        ...
    
    async def execute_async(self, context: ExecutionContext, input_data: Any) -> Any:
        """异步执行节点"""
        ...


@dataclass
class WorkflowNode:
    """工作流节点"""
    node_id: str
    name: str
    node_type: NodeType
    executor: Union[Callable, NodeProtocol, List[Callable]]
    dependencies: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    
    # 执行配置
    timeout: Optional[float] = None
    retry_count: int = 0
    retry_delay: float = 1.0
    skip_on_failure: bool = False
    
    # 条件配置
    condition_func: Optional[Callable] = None
    true_branch: Optional[str] = None
    false_branch: Optional[str] = None
    
    # 循环配置
    loop_condition: Optional[Callable] = None
    max_loop_iterations: int = 100
    
    # 执行状态
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    execution_count: int = 0


class TrainerServer:
    """训练服务器"""
    
    def __init__(self, maxsize: int = 1024):
        self.samples = queue.Queue(maxsize=maxsize)
        self.results = queue.Queue(maxsize=maxsize)
        self.is_running = False
    
    def put_samples(self, batch: List[Sample]):
        """添加样本批次"""
        for sample in batch:
            self.samples.put(sample, timeout=5)
    
    def get_sample(self, timeout: float = 5.0) -> Optional[Sample]:
        """获取样本"""
        try:
            return self.samples.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def put_result(self, result: Result):
        """添加结果"""
        self.results.put(result, timeout=5)
    
    def get_results_batch(self, max_items: int = 64, timeout: float = 0.5) -> List[Result]:
        """获取结果批次"""
        results = []
        for _ in range(max_items):
            try:
                result = self.results.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        return results
    
    def stop(self):
        """停止服务器"""
        self.is_running = False


class AgentAdapter(Protocol):
    """智能体适配器协议"""
    
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """执行动作"""
        ...
    
    def learn(self, trajectory: Trajectory) -> None:
        """学习"""
        ...


class LocalAgentClient:
    """本地智能体客户端"""
    
    def __init__(self, agent_id: str, adapter: AgentAdapter, server: TrainerServer):
        self.agent_id = agent_id
        self.adapter = adapter
        self.server = server
    
    def run_once(self) -> bool:
        """运行一次"""
        sample = self.server.get_sample()
        if sample is None:
            return False
        
        obs = sample.payload
        action_dict = self.adapter.act(obs)
        
        # 创建最小轨迹
        step = TrajectoryStep(state=obs, action=action_dict, reward=0.0, done=True, info={})
        traj = Trajectory(agent_id=self.agent_id, episode_id=sample.sample_id, steps=[step])
        
        self.adapter.learn(traj)
        self.server.put_result(Result(sample_id=sample.sample_id, agent_id=self.agent_id, trajectory=traj.to_dict()))
        
        return True


class WorkflowEngine:
    """工作流引擎"""
    
    def __init__(self, workflow_id: str, name: str = ""):
        self.workflow_id = workflow_id
        self.name = name or workflow_id
        self.nodes = {}
        self.stop_conditions = []
        self.global_timeout = None
        self.max_parallel_tasks = 10
        self.execution_context = None
        
        # 监控
        self.execution_listeners = []
        self.debug_mode = False
    
    def add_node(self, node: WorkflowNode):
        """添加节点"""
        if node.node_id in self.nodes:
            raise ValueError(f"节点 {node.node_id} 已存在")
        self.nodes[node.node_id] = node
    
    def add_edge(self, from_node: str, to_node: str):
        """添加边"""
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("源节点或目标节点不存在")
        
        self.nodes[from_node].successors.append(to_node)
        self.nodes[to_node].dependencies.append(from_node)
        
        # 检查环路
        if self._has_cycle():
            self.nodes[from_node].successors.remove(to_node)
            self.nodes[to_node].dependencies.remove(from_node)
            raise ValueError(f"添加边 {from_node} -> {to_node} 会形成环路")
    
    def _has_cycle(self) -> bool:
        """检测环路"""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}
        
        def dfs(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True
            if colors[node_id] == BLACK:
                return False
            
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
            raise ValueError("图中存在环路")
        
        in_degree = {node_id: len(self.nodes[node_id].dependencies) for node_id in self.nodes}
        queue_nodes = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue_nodes:
            node_id = queue_nodes.popleft()
            result.append(node_id)
            
            for successor in self.nodes[node_id].successors:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue_nodes.append(successor)
        
        if len(result) != len(self.nodes):
            raise ValueError("拓扑排序失败")
        
        return result
    
    def add_stop_condition(self, condition: StopCondition):
        """添加停止条件"""
        self.stop_conditions.append(condition)
    
    def _check_stop_conditions(self, context: ExecutionContext) -> bool:
        """检查停止条件"""
        for condition in self.stop_conditions:
            if condition.check(context):
                context.is_stopped = True
                context.stop_reason = f"停止条件满足: {condition.condition_type.value}"
                return True
        return False
    
    def _get_ready_nodes(self, context: ExecutionContext) -> List[str]:
        """获取可执行节点"""
        ready_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.status != ExecutionStatus.PENDING:
                continue
            
            # 检查依赖
            dependencies_ready = all(
                self.nodes[dep_id].status in [ExecutionStatus.SUCCESS, ExecutionStatus.SKIPPED]
                for dep_id in node.dependencies
            )
            
            if dependencies_ready:
                ready_nodes.append(node_id)
        
        return ready_nodes
    
    async def _execute_node(self, node_id: str, context: ExecutionContext) -> Any:
        """执行节点"""
        node = self.nodes[node_id]
        node.status = ExecutionStatus.RUNNING
        node.start_time = time.time()
        node.execution_count += 1
        
        try:
            # 获取输入数据
            input_data = self._get_node_input_data(node_id, context)
            
            # 根据节点类型执行
            if node.node_type == NodeType.CONDITION:
                result = await self._execute_condition_node(node, context, input_data)
            elif node.node_type == NodeType.LOOP:
                result = await self._execute_loop_node(node, context, input_data)
            elif node.node_type == NodeType.PARALLEL:
                result = await self._execute_parallel_node(node, context, input_data)
            else:
                result = await self._execute_task_node(node, context, input_data)
            
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
    
    async def _execute_task_node(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行任务节点"""
        executor = node.executor
        
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
    
    async def _execute_condition_node(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行条件节点"""
        if not node.condition_func:
            raise ValueError(f"条件节点 {node.node_id} 缺少条件函数")
        
        condition_result = node.condition_func(context, input_data)
        result = bool(condition_result)
        
        # 设置分支路径
        if result and node.true_branch:
            context.node_states[node.node_id] = {"branch": "true", "next_node": node.true_branch}
        elif not result and node.false_branch:
            context.node_states[node.node_id] = {"branch": "false", "next_node": node.false_branch}
        
        return result
    
    async def _execute_loop_node(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行循环节点"""
        if not node.loop_condition:
            raise ValueError(f"循环节点 {node.node_id} 缺少循环条件")
        
        results = []
        iteration = 0
        
        while iteration < node.max_loop_iterations:
            if not node.loop_condition(context, input_data, iteration):
                break
            
            loop_result = await self._execute_task_node(node, context, input_data)
            results.append(loop_result)
            
            iteration += 1
            
            if self._check_stop_conditions(context):
                break
        
        return results
    
    async def _execute_parallel_node(self, node: WorkflowNode, context: ExecutionContext, input_data: Any) -> Any:
        """执行并行节点"""
        executor = node.executor
        if not isinstance(executor, list):
            raise ValueError(f"并行节点 {node.node_id} 的执行器必须是列表")
        
        tasks = []
        for i, single_executor in enumerate(executor):
            task = asyncio.create_task(
                self._execute_single_parallel_task(f"{node.node_id}_parallel_{i}", single_executor, context, input_data)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def _execute_single_parallel_task(self, task_id: str, executor: Callable, 
                                          context: ExecutionContext, input_data: Any) -> Any:
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
        """获取节点输入数据"""
        node = self.nodes[node_id]
        
        if not node.dependencies:
            return context.global_state
        
        # 收集依赖节点输出
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
            workflow_id=self.workflow_id,
            execution_id=str(uuid.uuid4()),
            global_state=initial_data or {}
        )
        
        context = self.execution_context
        
        try:
            logger.info(f"开始执行工作流 {self.workflow_id}")
            
            # 检查图有效性
            if self._has_cycle():
                raise ValueError("工作流图包含环路")
            
            # 执行主循环
            while True:
                # 检查停止条件
                if self._check_stop_conditions(context):
                    logger.info(f"工作流停止: {context.stop_reason}")
                    break
                
                # 获取可执行节点
                ready_nodes = self._get_ready_nodes(context)
                
                if not ready_nodes:
                    # 检查是否完成
                    all_completed = all(
                        node.status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.SKIPPED]
                        for node in self.nodes.values()
                    )
                    
                    if all_completed:
                        logger.info("所有节点执行完成")
                        break
                    else:
                        logger.warning("没有可执行节点，可能存在死锁")
                        await asyncio.sleep(0.1)
                        continue
                
                # 并行执行节点
                tasks = []
                for node_id in ready_nodes[:self.max_parallel_tasks]:
                    task = asyncio.create_task(self._execute_node(node_id, context))
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                context.current_iteration += 1
                await asyncio.sleep(0.01)
            
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
                    "total_time": time.time() - context.start_time,
                    "total_iterations": context.current_iteration,
                    "total_errors": context.error_count
                }
            })
            
            logger.info(f"工作流 {self.workflow_id} 执行完成")
        
        return context
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        if not self.execution_context:
            return {"status": "not_executed"}
        
        context = self.execution_context
        
        return {
            "workflow_id": self.workflow_id,
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
    
    def add_execution_listener(self, listener: Callable):
        """添加执行监听器"""
        self.execution_listeners.append(listener)


class WorkflowBuilder:
    """工作流构建器"""
    
    def __init__(self, workflow_id: str, name: str = ""):
        self.workflow = WorkflowEngine(workflow_id, name)
    
    def add_task_node(self, node_id: str, name: str, executor: Callable, **kwargs) -> 'WorkflowBuilder':
        """添加任务节点"""
        node = WorkflowNode(
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
        node = WorkflowNode(
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
        node = WorkflowNode(
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
        node = WorkflowNode(
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
    
    def build(self) -> WorkflowEngine:
        """构建工作流"""
        return self.workflow


class DAGReplayBuffer:
    """DAG重放缓冲区"""
    
    def __init__(self):
        self.episodes = []
        self.lock = threading.Lock()
    
    def start_episode(self):
        """开始新回合"""
        with self.lock:
            self.episodes.append([])
    
    def add_step(self, step: TrajectoryStep):
        """添加步骤"""
        with self.lock:
            if not self.episodes:
                self.episodes.append([])
            self.episodes[-1].append(step)
    
    def finalize_episode(self) -> List[TrajectoryStep]:
        """完成回合"""
        with self.lock:
            if not self.episodes:
                return []
            return self.episodes[-1]
    
    def get_all_episodes(self) -> List[List[TrajectoryStep]]:
        """获取所有回合"""
        with self.lock:
            return self.episodes.copy()
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.episodes.clear()


class RLEngine:
    """RL引擎"""
    
    def __init__(self, algorithm: str = "ppo", learning_rate: float = 2e-4):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.trainer = None
        self.replay_buffer = DAGReplayBuffer()
    
    def initialize(self, llm_manager):
        """初始化"""
        from .rl_framework import create_ppo_trainer, create_grpo_trainer
        
        if self.algorithm == "ppo":
            self.trainer = create_ppo_trainer(llm_manager, self.learning_rate)
        elif self.algorithm == "grpo":
            self.trainer = create_grpo_trainer(llm_manager, self.learning_rate)
        else:
            self.trainer = create_ppo_trainer(llm_manager, self.learning_rate)
    
    def add_step(self, step: Dict[str, Any], group_id: str = "default"):
        """添加步骤"""
        if self.trainer:
            self.trainer.add_experience(
                step.get("state", {}),
                step.get("action", ""),
                step.get("reward", 0.0),
                step.get("done", False),
                group_id
            )
        
        # 添加到重放缓冲区
        traj_step = TrajectoryStep(
            state=step.get("state", {}),
            action=step.get("action", ""),
            reward=step.get("reward", 0.0),
            done=step.get("done", False),
            info=step.get("info", {})
        )
        self.replay_buffer.add_step(traj_step)
    
    def update(self) -> Dict[str, Any]:
        """更新策略"""
        if self.trainer:
            return self.trainer.update_policy()
        return {"status": "no_trainer"}


# 工厂函数
def create_workflow_engine(workflow_id: str, name: str = "") -> WorkflowBuilder:
    """创建工作流引擎"""
    return WorkflowBuilder(workflow_id, name)


def create_trainer_server(maxsize: int = 1024) -> TrainerServer:
    """创建训练服务器"""
    return TrainerServer(maxsize)


def create_agent_client(agent_id: str, adapter: AgentAdapter, server: TrainerServer) -> LocalAgentClient:
    """创建智能体客户端"""
    return LocalAgentClient(agent_id, adapter, server)


def create_dag_replay_buffer() -> DAGReplayBuffer:
    """创建DAG重放缓冲区"""
    return DAGReplayBuffer()


def create_rl_engine(algorithm: str = "ppo", learning_rate: float = 2e-4) -> RLEngine:
    """创建RL引擎"""
    return RLEngine(algorithm, learning_rate)


def write_trajectories_to_jsonl(trajectories: List[Trajectory], file_path: str):
    """写入轨迹到JSONL文件"""
    with open(file_path, 'a', encoding='utf-8') as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + '\n')


def read_trajectories_from_jsonl(file_path: str) -> List[Trajectory]:
    """从JSONL文件读取轨迹"""
    trajectories = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                trajectories.append(Trajectory.from_dict(data))
    except FileNotFoundError:
        logger.warning(f"轨迹文件不存在: {file_path}")
    except Exception as e:
        logger.error(f"读取轨迹文件失败: {e}")
    
    return trajectories


# 示例工作流创建
def create_simple_rl_workflow(llm_manager) -> WorkflowEngine:
    """创建简单RL工作流"""
    builder = create_workflow_engine("simple_rl", "简单RL工作流")
    
    # 定义任务函数
    def generate_task(context, input_data):
        return {"task": "生成任务", "state": {"step": context.current_iteration}}
    
    def execute_rl(context, input_data):
        rl_engine = create_rl_engine()
        rl_engine.initialize(llm_manager)
        
        step_data = {
            "state": input_data.get("state", {}),
            "action": "test_action",
            "reward": 1.0,
            "done": False
        }
        
        rl_engine.add_step(step_data)
        return rl_engine.update()
    
    def aggregate_results(context, input_data):
        return {"summary": "工作流完成", "results": input_data}
    
    # 构建工作流
    workflow = (builder
                .add_task_node("generate", "生成任务", generate_task)
                .add_task_node("rl_update", "RL更新", execute_rl)
                .add_task_node("aggregate", "聚合结果", aggregate_results)
                .connect("generate", "rl_update")
                .connect("rl_update", "aggregate")
                .add_stop_condition(StopConditionType.MAX_ITERATIONS, 10)
                .build())
    
    return workflow

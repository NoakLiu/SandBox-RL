"""
增强工作流系统

支持两种工作流模式：
1. 传统模式：LLM和Sandbox都可以作为节点
2. 纯沙盒模式：每个节点都是Sandbox，但需要LLM进行推理

包含复杂的游戏规则图，支持条件触发和访问限制
"""

from typing import Any, Dict, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
from collections import defaultdict, deque

from .workflow import WorkflowGraph, WorkflowNode, NodeType
from .llm_interface import SharedLLMManager, LLMResponse
from .rl_algorithms import RLTrainer

logger = logging.getLogger(__name__)


class WorkflowMode(Enum):
    """工作流模式"""
    TRADITIONAL = "traditional"  # 传统模式：LLM和Sandbox节点混合
    SANDBOX_ONLY = "sandbox_only"  # 纯沙盒模式：只有Sandbox节点，LLM用于推理


@dataclass
class NodeCondition:
    """节点触发条件"""
    required_nodes: List[str] = field(default_factory=list)  # 必须完成的前置节点
    required_scores: Dict[str, float] = field(default_factory=dict)  # 前置节点的最低分数要求
    required_resources: Dict[str, int] = field(default_factory=dict)  # 需要的资源
    custom_condition: Optional[Callable[[Dict[str, Any]], bool]] = None  # 自定义条件函数


@dataclass
class NodeLimits:
    """节点访问限制"""
    max_visits: int = -1  # 最大访问次数，-1表示无限制
    cooldown_time: float = 0.0  # 冷却时间（秒）
    resource_cost: Dict[str, int] = field(default_factory=dict)  # 每次访问的资源消耗


@dataclass
class GameState:
    """游戏状态"""
    resources: Dict[str, int] = field(default_factory=dict)  # 资源状态
    node_visits: Dict[str, int] = field(default_factory=dict)  # 节点访问次数
    node_last_visit: Dict[str, float] = field(default_factory=dict)  # 节点最后访问时间
    completed_nodes: Set[str] = field(default_factory=set)  # 已完成的节点
    node_scores: Dict[str, float] = field(default_factory=dict)  # 节点得分
    global_score: float = 0.0  # 全局得分
    game_time: float = 0.0  # 游戏时间


class EnhancedSandbox:
    """增强沙盒，支持LLM推理"""
    
    def __init__(self, sandbox_id: str, base_sandbox, llm_manager: SharedLLMManager):
        self.sandbox_id = sandbox_id
        self.base_sandbox = base_sandbox
        self.llm_manager = llm_manager
        
        # 注册到LLM管理器
        self.llm_manager.register_node(f"sandbox_{sandbox_id}", {
            "role": f"沙盒推理器_{sandbox_id}",
            "reasoning_type": "strategic"
        })
    
    def execute_with_reasoning(self, inputs: Dict[str, Any], game_state: GameState) -> Dict[str, Any]:
        """使用LLM推理执行沙盒任务"""
        
        # 1. 生成任务
        case = self.base_sandbox.case_generator()
        
        # 2. 构建推理提示
        reasoning_prompt = self._build_reasoning_prompt(case, inputs, game_state)
        
        # 3. LLM推理
        llm_response = self.llm_manager.generate_for_node(
            f"sandbox_{self.sandbox_id}", 
            reasoning_prompt,
            temperature=0.7
        )
        
        # 4. 基于推理结果执行动作
        action_result = self._execute_action(llm_response, case, game_state)
        
        # 5. 评估结果
        score = self.base_sandbox.verify_score(action_result["action"], case)
        
        return {
            "sandbox_id": self.sandbox_id,
            "case": case,
            "reasoning": llm_response.reasoning,
            "llm_response": llm_response.text,
            "action": action_result["action"],
            "score": score,
            "confidence": llm_response.confidence,
            "metadata": {
                "reasoning_time": action_result.get("reasoning_time", 0),
                "game_state": game_state.__dict__.copy()
            }
        }
    
    def _build_reasoning_prompt(self, case: Dict[str, Any], inputs: Dict[str, Any], 
                               game_state: GameState) -> str:
        """构建推理提示"""
        prompt = f"""
作为智能沙盒推理器 {self.sandbox_id}，请分析以下情况并制定行动策略：

任务信息：
{json.dumps(case, ensure_ascii=False, indent=2)}

当前输入：
{json.dumps(inputs, ensure_ascii=False, indent=2)}

游戏状态：
- 当前资源: {game_state.resources}
- 全局得分: {game_state.global_score}
- 已完成节点: {list(game_state.completed_nodes)}
- 游戏时间: {game_state.game_time:.2f}秒

请进行深入分析并提出最优行动方案。考虑：
1. 任务的核心要求和约束
2. 当前资源和状态的限制
3. 长期策略和短期收益的平衡
4. 风险评估和备选方案

请给出具体的行动建议。
"""
        return prompt
    
    def _execute_action(self, llm_response: LLMResponse, case: Dict[str, Any], 
                       game_state: GameState) -> Dict[str, Any]:
        """基于LLM推理执行具体行动"""
        start_time = time.time()
        
        # 从LLM响应中提取行动（简化实现）
        action_text = llm_response.text
        
        # 这里可以添加更复杂的行动解析逻辑
        # 目前使用简化的映射
        if "计算" in action_text or "数学" in action_text:
            action = f"数学计算基于推理: {action_text[:50]}..."
        elif "策略" in action_text or "规划" in action_text:
            action = f"策略规划基于推理: {action_text[:50]}..."
        else:
            action = f"通用行动基于推理: {action_text[:50]}..."
        
        reasoning_time = time.time() - start_time
        
        return {
            "action": action,
            "reasoning_time": reasoning_time,
            "confidence": llm_response.confidence
        }


class EnhancedWorkflowNode(WorkflowNode):
    """增强工作流节点"""
    
    def __init__(self, node_id: str, node_type: NodeType, 
                 condition: Optional[NodeCondition] = None,
                 limits: Optional[NodeLimits] = None,
                 **kwargs):
        super().__init__(node_id, node_type, **kwargs)
        self.condition = condition or NodeCondition()
        self.limits = limits or NodeLimits()
        self.enhanced_sandbox: Optional[EnhancedSandbox] = None
    
    def can_execute(self, game_state: GameState, execution_context: Dict[str, Any]) -> Tuple[bool, str]:
        """检查节点是否可以执行"""
        
        # 检查访问次数限制
        if self.limits.max_visits > 0:
            visits = game_state.node_visits.get(self.node_id, 0)
            if visits >= self.limits.max_visits:
                return False, f"节点 {self.node_id} 已达到最大访问次数 {self.limits.max_visits}"
        
        # 检查冷却时间
        if self.limits.cooldown_time > 0:
            last_visit = game_state.node_last_visit.get(self.node_id, 0)
            if time.time() - last_visit < self.limits.cooldown_time:
                remaining = self.limits.cooldown_time - (time.time() - last_visit)
                return False, f"节点 {self.node_id} 冷却中，剩余 {remaining:.1f} 秒"
        
        # 检查资源需求
        for resource, required in self.limits.resource_cost.items():
            available = game_state.resources.get(resource, 0)
            if available < required:
                return False, f"资源不足：需要 {resource} {required}，当前 {available}"
        
        # 检查前置节点
        for required_node in self.condition.required_nodes:
            if required_node not in game_state.completed_nodes:
                return False, f"前置节点 {required_node} 未完成"
        
        # 检查前置节点分数
        for node_id, min_score in self.condition.required_scores.items():
            actual_score = game_state.node_scores.get(node_id, 0.0)
            if actual_score < min_score:
                return False, f"节点 {node_id} 分数不足：需要 {min_score}，当前 {actual_score}"
        
        # 检查自定义条件
        if self.condition.custom_condition:
            try:
                if not self.condition.custom_condition(execution_context):
                    return False, "自定义条件不满足"
            except Exception as e:
                return False, f"自定义条件检查失败: {e}"
        
        return True, "可以执行"
    
    def execute_enhanced(self, inputs: Dict[str, Any], game_state: GameState) -> Dict[str, Any]:
        """增强执行方法"""
        
        # 更新访问统计
        game_state.node_visits[self.node_id] = game_state.node_visits.get(self.node_id, 0) + 1
        game_state.node_last_visit[self.node_id] = time.time()
        
        # 消耗资源
        for resource, cost in self.limits.resource_cost.items():
            game_state.resources[resource] = game_state.resources.get(resource, 0) - cost
        
        # 执行节点逻辑
        if self.enhanced_sandbox:
            # 使用增强沙盒执行
            result = self.enhanced_sandbox.execute_with_reasoning(inputs, game_state)
        else:
            # 使用原始执行方法
            result = self.execute(inputs)
        
        # 更新游戏状态
        if "score" in result:
            game_state.node_scores[self.node_id] = result["score"]
            game_state.global_score += result["score"]
        
        game_state.completed_nodes.add(self.node_id)
        
        return result


class EnhancedWorkflowGraph:
    """增强工作流图"""
    
    def __init__(self, graph_id: str, mode: WorkflowMode, llm_manager: SharedLLMManager):
        self.graph_id = graph_id
        self.mode = mode
        self.llm_manager = llm_manager
        self.nodes: Dict[str, EnhancedWorkflowNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.game_state = GameState()
        self.execution_history: List[Dict[str, Any]] = []
        
        # 初始化默认资源
        self.game_state.resources = {
            "energy": 100,
            "tokens": 50,
            "time": 300,
            "knowledge": 10
        }
    
    def add_node(self, node: EnhancedWorkflowNode) -> None:
        """添加节点"""
        if node.node_id in self.nodes:
            raise ValueError(f"节点 {node.node_id} 已存在")
        
        self.nodes[node.node_id] = node
        
        # 如果是纯沙盒模式，为沙盒节点创建增强沙盒
        if (self.mode == WorkflowMode.SANDBOX_ONLY and 
            node.node_type == NodeType.SANDBOX and 
            node.sandbox is not None):
            
            node.enhanced_sandbox = EnhancedSandbox(
                node.node_id, 
                node.sandbox, 
                self.llm_manager
            )
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加边"""
        if from_node not in self.nodes:
            raise ValueError(f"源节点 {from_node} 不存在")
        if to_node not in self.nodes:
            raise ValueError(f"目标节点 {to_node} 不存在")
        
        self.edges.append((from_node, to_node))
        
        # 更新依赖关系
        if from_node not in self.nodes[to_node].dependencies:
            self.nodes[to_node].dependencies.append(from_node)
    
    def get_executable_nodes(self) -> List[str]:
        """获取当前可执行的节点"""
        executable = []
        
        for node_id, node in self.nodes.items():
            if node_id in self.game_state.completed_nodes:
                continue
            
            can_exec, reason = node.can_execute(self.game_state, {
                "current_time": time.time(),
                "global_score": self.game_state.global_score
            })
            
            if can_exec:
                executable.append(node_id)
        
        return executable
    
    def execute_node(self, node_id: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行指定节点"""
        if node_id not in self.nodes:
            raise ValueError(f"节点 {node_id} 不存在")
        
        if inputs is None:
            inputs = {}
        
        node = self.nodes[node_id]
        
        # 检查是否可以执行
        can_exec, reason = node.can_execute(self.game_state, {
            "current_time": time.time(),
            "global_score": self.game_state.global_score
        })
        
        if not can_exec:
            raise RuntimeError(f"节点 {node_id} 无法执行: {reason}")
        
        # 执行节点
        start_time = time.time()
        try:
            result = node.execute_enhanced(inputs, self.game_state)
            execution_time = time.time() - start_time
            
            # 记录执行历史
            self.execution_history.append({
                "node_id": node_id,
                "execution_time": execution_time,
                "inputs": inputs,
                "result": result,
                "game_state_snapshot": self.game_state.__dict__.copy(),
                "status": "success",
                "timestamp": time.time()
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # 记录失败历史
            self.execution_history.append({
                "node_id": node_id,
                "execution_time": execution_time,
                "inputs": inputs,
                "error": str(e),
                "game_state_snapshot": self.game_state.__dict__.copy(),
                "status": "failed",
                "timestamp": time.time()
            })
            
            raise
    
    def execute_full_workflow(self, max_steps: int = 100) -> Dict[str, Any]:
        """执行完整工作流"""
        self.game_state.game_time = time.time()
        executed_nodes = []
        step_count = 0
        
        while step_count < max_steps:
            executable_nodes = self.get_executable_nodes()
            
            if not executable_nodes:
                break
            
            # 选择下一个执行的节点（简单策略：选择第一个）
            # 实际应用中可以使用更复杂的调度策略
            next_node = executable_nodes[0]
            
            try:
                result = self.execute_node(next_node)
                executed_nodes.append({
                    "node_id": next_node,
                    "result": result,
                    "step": step_count
                })
                
            except Exception as e:
                logger.error(f"节点 {next_node} 执行失败: {e}")
                break
            
            step_count += 1
        
        total_time = time.time() - self.game_state.game_time
        
        return {
            "graph_id": self.graph_id,
            "mode": self.mode.value,
            "executed_nodes": executed_nodes,
            "total_steps": step_count,
            "total_time": total_time,
            "final_score": self.game_state.global_score,
            "final_resources": self.game_state.resources.copy(),
            "completed_nodes_count": len(self.game_state.completed_nodes),
            "execution_history": self.execution_history[-10:]  # 最近10条记录
        }
    
    def get_game_stats(self) -> Dict[str, Any]:
        """获取游戏统计信息"""
        return {
            "game_state": self.game_state.__dict__.copy(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "executable_nodes": self.get_executable_nodes(),
            "execution_history_count": len(self.execution_history)
        }


def create_complex_game_graph(llm_manager: SharedLLMManager) -> EnhancedWorkflowGraph:
    """创建复杂的游戏规则图"""
    
    # 创建纯沙盒模式的工作流图
    graph = EnhancedWorkflowGraph("complex_game", WorkflowMode.SANDBOX_ONLY, llm_manager)
    
    # 导入沙盒实现
    from ..sandbox_implementations import Game24Sandbox, SummarizeSandbox
    
    # === 第一层：入门关卡 ===
    # 新手村 - 无限制访问
    tutorial_node = EnhancedWorkflowNode(
        "tutorial", 
        NodeType.SANDBOX,
        sandbox=Game24Sandbox(),
        condition=NodeCondition(),
        limits=NodeLimits(resource_cost={"energy": 5})
    )
    graph.add_node(tutorial_node)
    
    # 基础训练 - 需要完成新手村
    basic_training_node = EnhancedWorkflowNode(
        "basic_training",
        NodeType.SANDBOX,
        sandbox=SummarizeSandbox(),
        condition=NodeCondition(
            required_nodes=["tutorial"],
            required_scores={"tutorial": 0.3}
        ),
        limits=NodeLimits(
            max_visits=3,
            resource_cost={"energy": 10, "tokens": 5}
        )
    )
    graph.add_node(basic_training_node)
    
    # === 第二层：进阶关卡 ===
    # 数学挑战 - 需要基础训练达到一定分数
    math_challenge_node = EnhancedWorkflowNode(
        "math_challenge",
        NodeType.SANDBOX,
        sandbox=Game24Sandbox(),
        condition=NodeCondition(
            required_nodes=["basic_training"],
            required_scores={"basic_training": 0.5}
        ),
        limits=NodeLimits(
            max_visits=5,
            cooldown_time=30.0,  # 30秒冷却
            resource_cost={"energy": 15, "tokens": 8}
        )
    )
    graph.add_node(math_challenge_node)
    
    # 策略思考 - 并行路径
    strategy_node = EnhancedWorkflowNode(
        "strategy_thinking",
        NodeType.SANDBOX,
        sandbox=SummarizeSandbox(),
        condition=NodeCondition(
            required_nodes=["basic_training"],
            required_scores={"basic_training": 0.4}
        ),
        limits=NodeLimits(
            max_visits=4,
            resource_cost={"energy": 12, "knowledge": 2}
        )
    )
    graph.add_node(strategy_node)
    
    # === 第三层：专家关卡 ===
    # 高级数学 - 需要数学挑战高分
    advanced_math_node = EnhancedWorkflowNode(
        "advanced_math",
        NodeType.SANDBOX,
        sandbox=Game24Sandbox(),
        condition=NodeCondition(
            required_nodes=["math_challenge"],
            required_scores={"math_challenge": 0.7},
            custom_condition=lambda ctx: ctx.get("global_score", 0) > 2.0
        ),
        limits=NodeLimits(
            max_visits=2,
            cooldown_time=60.0,  # 1分钟冷却
            resource_cost={"energy": 25, "tokens": 15, "knowledge": 3}
        )
    )
    graph.add_node(advanced_math_node)
    
    # 综合挑战 - 需要多个前置条件
    comprehensive_node = EnhancedWorkflowNode(
        "comprehensive_challenge",
        NodeType.SANDBOX,
        sandbox=SummarizeSandbox(),
        condition=NodeCondition(
            required_nodes=["math_challenge", "strategy_thinking"],
            required_scores={"math_challenge": 0.6, "strategy_thinking": 0.6}
        ),
        limits=NodeLimits(
            max_visits=3,
            cooldown_time=45.0,
            resource_cost={"energy": 20, "tokens": 12, "knowledge": 4}
        )
    )
    graph.add_node(comprehensive_node)
    
    # === 第四层：大师关卡 ===
    # 终极挑战 - 需要所有前置关卡
    ultimate_node = EnhancedWorkflowNode(
        "ultimate_challenge",
        NodeType.SANDBOX,
        sandbox=Game24Sandbox(),
        condition=NodeCondition(
            required_nodes=["advanced_math", "comprehensive_challenge"],
            required_scores={"advanced_math": 0.8, "comprehensive_challenge": 0.7},
            custom_condition=lambda ctx: ctx.get("global_score", 0) > 5.0
        ),
        limits=NodeLimits(
            max_visits=1,  # 只能挑战一次
            cooldown_time=120.0,  # 2分钟冷却
            resource_cost={"energy": 50, "tokens": 25, "knowledge": 5}
        )
    )
    graph.add_node(ultimate_node)
    
    # 构建边连接
    edges = [
        ("tutorial", "basic_training"),
        ("basic_training", "math_challenge"),
        ("basic_training", "strategy_thinking"),
        ("math_challenge", "advanced_math"),
        ("math_challenge", "comprehensive_challenge"),
        ("strategy_thinking", "comprehensive_challenge"),
        ("advanced_math", "ultimate_challenge"),
        ("comprehensive_challenge", "ultimate_challenge")
    ]
    
    for from_node, to_node in edges:
        graph.add_edge(from_node, to_node)
    
    return graph 
"""
工作流图定义和执行引擎

支持定义复杂的LLM-沙盒交互图并执行
"""

from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
import logging
from dataclasses import dataclass, field
from .sandbox import Sandbox, SandboxProtocol


class NodeType(Enum):
    """节点类型"""
    INPUT = "input"           # 输入节点
    SANDBOX = "sandbox"       # 沙盒节点
    LLM = "llm"              # LLM节点
    OUTPUT = "output"         # 输出节点
    AGGREGATOR = "aggregator" # 聚合节点


@dataclass
class WorkflowNode:
    """工作流节点"""
    node_id: str
    node_type: NodeType
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 节点特定属性
    sandbox: Optional[Sandbox] = None
    llm_func: Optional[Callable] = None
    aggregator_func: Optional[Callable] = None
    
    def __post_init__(self):
        """初始化后验证"""
        if self.node_type == NodeType.SANDBOX and self.sandbox is None:
            raise ValueError(f"Sandbox节点 {self.node_id} 必须指定sandbox")
        if self.node_type == NodeType.LLM and self.llm_func is None:
            raise ValueError(f"LLM节点 {self.node_id} 必须指定llm_func")
        if self.node_type == NodeType.AGGREGATOR and self.aggregator_func is None:
            raise ValueError(f"聚合节点 {self.node_id} 必须指定aggregator_func")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行节点逻辑"""
        if self.node_type == NodeType.INPUT:
            return inputs
        
        elif self.node_type == NodeType.SANDBOX:
            return self._execute_sandbox(inputs)
        
        elif self.node_type == NodeType.LLM:
            return self._execute_llm(inputs)
        
        elif self.node_type == NodeType.AGGREGATOR:
            return self._execute_aggregator(inputs)
        
        elif self.node_type == NodeType.OUTPUT:
            return inputs
        
        else:
            raise ValueError(f"未知节点类型: {self.node_type}")
    
    def _execute_sandbox(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行沙盒节点"""
        if not self.sandbox:
            raise ValueError("沙盒节点缺少sandbox实例")
        
        # 根据输入决定执行哪个沙盒方法
        action = inputs.get("action", "full_cycle")
        
        if action == "case_generator":
            case = self.sandbox.case_generator()
            return {"case": case, "node_id": self.node_id}
        
        elif action == "prompt_func":
            case = inputs.get("case")
            if not case:
                raise ValueError("prompt_func需要case参数")
            prompt = self.sandbox.prompt_func(case)
            return {"prompt": prompt, "case": case, "node_id": self.node_id}
        
        elif action == "verify_score":
            response = inputs.get("response")
            case = inputs.get("case")
            format_score = inputs.get("format_score", 0.0)
            if not response or not case:
                raise ValueError("verify_score需要response和case参数")
            score = self.sandbox.verify_score(response, case, format_score)
            return {"score": score, "case": case, "response": response, "node_id": self.node_id}
        
        elif action == "full_cycle":
            llm_func = inputs.get("llm_func")
            result = self.sandbox.run_full_cycle(llm_func)
            result["node_id"] = self.node_id
            return result
        
        else:
            raise ValueError(f"未知的沙盒动作: {action}")
    
    def _execute_llm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行LLM节点"""
        if not self.llm_func:
            raise ValueError("LLM节点缺少llm_func")
        
        prompt = inputs.get("prompt", "")
        response = self.llm_func(prompt)
        return {"response": response, "prompt": prompt, "node_id": self.node_id}
    
    def _execute_aggregator(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """执行聚合节点"""
        if not self.aggregator_func:
            raise ValueError("聚合节点缺少aggregator_func")
        
        result = self.aggregator_func(inputs)
        result["node_id"] = self.node_id
        return result


class WorkflowGraph:
    """工作流图执行器"""
    
    def __init__(self, graph_id: str = "default"):
        """初始化工作流图
        
        Args:
            graph_id: 图的唯一标识符
        """
        self.graph_id = graph_id
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[tuple[str, str]] = []
        self.input_node: Optional[str] = None
        self.output_nodes: List[str] = []
        self.execution_history: List[Dict[str, Any]] = []
    
    def add_node(self, node: WorkflowNode) -> None:
        """添加节点"""
        if node.node_id in self.nodes:
            raise ValueError(f"节点 {node.node_id} 已存在")
        
        self.nodes[node.node_id] = node
        
        # 自动设置输入输出节点
        if node.node_type == NodeType.INPUT:
            self.input_node = node.node_id
        elif node.node_type == NodeType.OUTPUT:
            self.output_nodes.append(node.node_id)
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加边（依赖关系）"""
        if from_node not in self.nodes:
            raise ValueError(f"源节点 {from_node} 不存在")
        if to_node not in self.nodes:
            raise ValueError(f"目标节点 {to_node} 不存在")
        
        self.edges.append((from_node, to_node))
        
        # 更新目标节点的依赖列表
        if from_node not in self.nodes[to_node].dependencies:
            self.nodes[to_node].dependencies.append(from_node)
    
    def set_input_node(self, node_id: str) -> None:
        """设置输入节点"""
        if node_id not in self.nodes:
            raise ValueError(f"节点 {node_id} 不存在")
        self.input_node = node_id
    
    def set_output_nodes(self, node_ids: List[str]) -> None:
        """设置输出节点"""
        for node_id in node_ids:
            if node_id not in self.nodes:
                raise ValueError(f"节点 {node_id} 不存在")
        self.output_nodes = node_ids
    
    def topological_sort(self) -> List[str]:
        """拓扑排序，返回执行顺序"""
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        # 计算入度
        for from_node, to_node in self.edges:
            in_degree[to_node] += 1
        
        # 使用Kahn算法进行拓扑排序
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # 更新相邻节点的入度
            for from_node, to_node in self.edges:
                if from_node == current:
                    in_degree[to_node] -= 1
                    if in_degree[to_node] == 0:
                        queue.append(to_node)
        
        if len(result) != len(self.nodes):
            raise ValueError("图中存在环路，无法进行拓扑排序")
        
        return result
    
    def execute(self, initial_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行工作流图
        
        Args:
            initial_input: 初始输入数据
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        if not self.nodes:
            raise ValueError("工作流图为空")
        
        # 如果没有显式设置输入节点，使用第一个INPUT类型节点
        if self.input_node is None:
            input_nodes = [node_id for node_id, node in self.nodes.items() 
                          if node.node_type == NodeType.INPUT]
            if input_nodes:
                self.input_node = input_nodes[0]
        
        # 如果没有设置输出节点，使用所有OUTPUT类型节点
        if not self.output_nodes:
            self.output_nodes = [node_id for node_id, node in self.nodes.items() 
                               if node.node_type == NodeType.OUTPUT]
        
        # 获取执行顺序
        execution_order = self.topological_sort()
        
        # 存储节点执行结果
        node_results = {}
        
        # 初始化输入
        if initial_input is None:
            initial_input = {}
        
        # 按拓扑顺序执行节点
        for node_id in execution_order:
            node = self.nodes[node_id]
            
            # 收集该节点的输入
            if node.dependencies:
                # 聚合依赖节点的输出
                node_input = {}
                for dep_id in node.dependencies:
                    if dep_id in node_results:
                        node_input[dep_id] = node_results[dep_id]
                
                # 如果是输入节点，合并初始输入
                if node.node_type == NodeType.INPUT:
                    node_input.update(initial_input)
            else:
                # 根节点使用初始输入
                node_input = initial_input.copy() if initial_input else {}
            
            # 执行节点
            try:
                result = node.execute(node_input)
                node_results[node_id] = result
                
                # 记录执行历史
                self.execution_history.append({
                    "node_id": node_id,
                    "node_type": node.node_type.value,
                    "input": node_input,
                    "output": result,
                    "status": "success"
                })
                
            except Exception as e:
                error_info = {
                    "node_id": node_id,
                    "node_type": node.node_type.value,
                    "input": node_input,
                    "error": str(e),
                    "status": "failed"
                }
                self.execution_history.append(error_info)
                raise RuntimeError(f"节点 {node_id} 执行失败: {e}") from e
        
        # 收集输出节点的结果
        if self.output_nodes:
            output_results = {}
            for output_node_id in self.output_nodes:
                if output_node_id in node_results:
                    output_results[output_node_id] = node_results[output_node_id]
            return output_results
        else:
            # 如果没有指定输出节点，返回所有结果
            return node_results
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history.copy()
    
    def clear_history(self) -> None:
        """清除执行历史"""
        self.execution_history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """将工作流图序列化为字典"""
        return {
            "graph_id": self.graph_id,
            "nodes": {
                node_id: {
                    "node_type": node.node_type.value,
                    "dependencies": node.dependencies,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": self.edges,
            "input_node": self.input_node,
            "output_nodes": self.output_nodes
        } 
#!/usr/bin/env python3
"""
SandGraph Dynamic Graph Visualizer

动态图可视化系统，展示misinformation传播和cooperate/compete关系
支持实时日志读取和动态可视化
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import defaultdict, deque
import queue

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型"""
    AGENT = "agent"
    MISINFO_SOURCE = "misinfo_source"
    FACT_CHECKER = "fact_checker"
    INFLUENCER = "influencer"
    REGULAR_USER = "regular_user"


class EdgeType(Enum):
    """边类型"""
    COOPERATE = "cooperate"
    COMPETE = "compete"
    MISINFO_SPREAD = "misinfo_spread"
    FACT_CHECK = "fact_check"
    INFLUENCE = "influence"
    NEUTRAL = "neutral"


class InteractionType(Enum):
    """交互类型"""
    SHARE = "share"
    LIKE = "like"
    COMMENT = "comment"
    FACT_CHECK = "fact_check"
    DEBUNK = "debunk"
    COOPERATE = "cooperate"
    COMPETE = "compete"


@dataclass
class GraphNode:
    """图节点"""
    id: str
    node_type: NodeType
    position: Tuple[float, float] = (0.0, 0.0)
    belief: float = 0.5  # 对misinformation的相信程度 (0-1)
    influence: float = 1.0  # 影响力
    followers: int = 0
    credibility: float = 0.5  # 可信度
    color: str = "blue"
    size: int = 100
    label: str = ""
    
    def __post_init__(self):
        if not self.label:
            self.label = f"{self.node_type.value}_{self.id}"


@dataclass
class GraphEdge:
    """图边"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    timestamp: float = 0.0
    interaction_type: Optional[InteractionType] = None
    color: str = "black"
    width: float = 1.0


@dataclass
class GraphEvent:
    """图事件"""
    timestamp: float
    event_type: str
    source_id: str
    target_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


class SandGraphVisualizer:
    """SandGraph动态图可视化器"""
    
    def __init__(self, 
                 log_file: str = "sandgraph_visualization.log",
                 update_interval: float = 1.0,
                 max_nodes: int = 50,
                 max_edges: int = 100):
        
        self.log_file = log_file
        self.update_interval = update_interval
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # 图数据结构
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.events: List[GraphEvent] = []
        
        # 可视化状态
        self.is_running = False
        self.current_time = 0.0
        self.animation = None
        self.fig = None
        self.ax = None
        
        # 颜色映射
        self.node_colors = {
            NodeType.AGENT: "#1f77b4",
            NodeType.MISINFO_SOURCE: "#d62728",
            NodeType.FACT_CHECKER: "#2ca02c",
            NodeType.INFLUENCER: "#ff7f0e",
            NodeType.REGULAR_USER: "#9467bd"
        }
        
        self.edge_colors = {
            EdgeType.COOPERATE: "#2ca02c",
            EdgeType.COMPETE: "#d62728",
            EdgeType.MISINFO_SPREAD: "#ff7f0e",
            EdgeType.FACT_CHECK: "#1f77b4",
            EdgeType.INFLUENCE: "#9467bd",
            EdgeType.NEUTRAL: "#7f7f7f"
        }
        
        # 事件队列
        self.event_queue = queue.Queue()
        
        # 初始化日志文件
        self._init_log_file()
        
        logger.info("SandGraph可视化器初始化完成")
    
    def _init_log_file(self):
        """初始化日志文件"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "version": "1.0"
                    },
                    "events": []
                }, f, indent=2)
    
    def add_node(self, 
                 node_id: str, 
                 node_type: NodeType,
                 position: Optional[Tuple[float, float]] = None,
                 **kwargs) -> GraphNode:
        """添加节点"""
        if position is None:
            position = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
        
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            position=position,
            color=self.node_colors.get(node_type, "#1f77b4"),
            **kwargs
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.__dict__)
        
        return node
    
    def add_edge(self, 
                 source: str, 
                 target: str, 
                 edge_type: EdgeType,
                 weight: float = 1.0,
                 **kwargs) -> GraphEdge:
        """添加边"""
        edge = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            timestamp=self.current_time,
            color=self.edge_colors.get(edge_type, "#7f7f7f"),
            **kwargs
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source, target, **edge.__dict__)
        
        return edge
    
    def log_event(self, event: GraphEvent):
        """记录事件到日志"""
        # 添加到内存
        self.events.append(event)
        self.event_queue.put(event)
        
        # 写入文件
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            data["events"].append({
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "source_id": event.source_id,
                "target_id": event.target_id,
                "data": event.data
            })
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"写入日志失败: {e}")
    
    def create_misinfo_scenario(self, num_agents: int = 10):
        """创建misinformation传播场景"""
        # 清空现有图
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # 创建misinformation源
        misinfo_source = self.add_node(
            "misinfo_source_1",
            NodeType.MISINFO_SOURCE,
            belief=1.0,
            influence=2.0,
            followers=1000,
            credibility=0.1,
            size=200
        )
        
        # 创建fact checker
        fact_checker = self.add_node(
            "fact_checker_1",
            NodeType.FACT_CHECKER,
            belief=0.0,
            influence=1.5,
            followers=500,
            credibility=0.9,
            size=150
        )
        
        # 创建influencer
        influencer = self.add_node(
            "influencer_1",
            NodeType.INFLUENCER,
            belief=0.8,
            influence=1.8,
            followers=800,
            credibility=0.3,
            size=180
        )
        
        # 创建普通用户
        for i in range(num_agents - 3):
            user_id = f"user_{i+1}"
            belief = np.random.uniform(0.1, 0.9)
            influence = np.random.uniform(0.5, 1.5)
            
            user = self.add_node(
                user_id,
                NodeType.REGULAR_USER,
                belief=belief,
                influence=influence,
                followers=np.random.randint(10, 100),
                credibility=np.random.uniform(0.2, 0.8),
                size=50 + int(belief * 100)
            )
        
        # 添加初始边
        self.add_edge("misinfo_source_1", "influencer_1", EdgeType.MISINFO_SPREAD, weight=0.8)
        self.add_edge("influencer_1", "user_1", EdgeType.INFLUENCE, weight=0.6)
        self.add_edge("fact_checker_1", "user_2", EdgeType.FACT_CHECK, weight=0.7)
        
        logger.info(f"创建了包含{len(self.nodes)}个节点的misinformation场景")
    
    def simulate_interaction(self, source_id: str, target_id: str, interaction_type: InteractionType):
        """模拟交互"""
        source_node = self.nodes.get(source_id)
        target_node = self.nodes.get(target_id)
        
        if not source_node or not target_node:
            return
        
        # 根据交互类型更新belief
        if interaction_type == InteractionType.SHARE:
            # 分享misinformation
            belief_transfer = source_node.belief * source_node.influence * 0.1
            target_node.belief = min(1.0, target_node.belief + belief_transfer)
            edge_type = EdgeType.MISINFO_SPREAD
            
        elif interaction_type == InteractionType.FACT_CHECK:
            # 事实核查
            belief_correction = (1.0 - source_node.belief) * source_node.credibility * 0.2
            target_node.belief = max(0.0, target_node.belief - belief_correction)
            edge_type = EdgeType.FACT_CHECK
            
        elif interaction_type == InteractionType.COOPERATE:
            # 合作
            edge_type = EdgeType.COOPERATE
            
        elif interaction_type == InteractionType.COMPETE:
            # 竞争
            edge_type = EdgeType.COMPETE
            
        else:
            edge_type = EdgeType.NEUTRAL
        
        # 添加边
        self.add_edge(source_id, target_id, edge_type, weight=1.0)
        
        # 记录事件
        event = GraphEvent(
            timestamp=self.current_time,
            event_type=interaction_type.value,
            source_id=source_id,
            target_id=target_id,
            data={
                "source_belief": source_node.belief,
                "target_belief": target_node.belief,
                "edge_type": edge_type.value
            }
        )
        self.log_event(event)
        
        # 更新节点大小（基于belief）
        target_node.size = 50 + int(target_node.belief * 150)
    
    def _update_visualization(self, frame):
        """更新可视化"""
        if not self.is_running:
            return
        
        # 清空画布
        self.ax.clear()
        
        # 绘制节点
        for node_id, node in self.nodes.items():
            x, y = node.position
            color = node.color
            size = node.size
            
            # 根据belief调整颜色
            if node.node_type == NodeType.REGULAR_USER:
                # 红色表示相信misinformation，绿色表示不相信
                red = node.belief
                green = 1.0 - node.belief
                color = (red, green, 0.0)
            
            self.ax.scatter(x, y, s=size, c=[color], alpha=0.7, edgecolors='black', linewidth=1)
            
            # 添加标签
            if node.influence > 1.2:  # 只显示重要节点的标签
                self.ax.annotate(node.label, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        # 绘制边
        for edge in self.edges[-self.max_edges:]:  # 只显示最近的边
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            
            if source_node and target_node:
                x1, y1 = source_node.position
                x2, y2 = target_node.position
                
                # 根据边类型设置样式
                linestyle = '-'
                alpha = 0.6
                
                if edge.edge_type == EdgeType.COOPERATE:
                    linestyle = '--'
                    alpha = 0.8
                elif edge.edge_type == EdgeType.COMPETE:
                    linestyle = ':'
                    alpha = 0.8
                elif edge.edge_type == EdgeType.MISINFO_SPREAD:
                    alpha = 0.9
                
                self.ax.plot([x1, x2], [y1, y2], color=edge.color, 
                           linewidth=edge.width, alpha=alpha, linestyle=linestyle)
        
        # 设置画布
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 12)
        self.ax.set_title(f'SandGraph Visualization - Time: {self.current_time:.1f}s')
        self.ax.set_aspect('equal')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Misinfo Source'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Fact Checker'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='Influencer'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                      markersize=10, label='Regular User')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
    
    def start_visualization(self):
        """启动可视化"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 创建动画
        self.animation = animation.FuncAnimation(
            self.fig, self._update_visualization, 
            interval=self.update_interval * 1000,  # 转换为毫秒
            blit=False
        )
        
        # 启动模拟线程
        simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        simulation_thread.start()
        
        plt.show()
    
    def stop_visualization(self):
        """停止可视化"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
        plt.close()
    
    def _simulation_loop(self):
        """模拟循环"""
        while self.is_running:
            # 随机选择两个节点进行交互
            if len(self.nodes) >= 2:
                node_ids = list(self.nodes.keys())
                source_id = np.random.choice(node_ids)
                target_id = np.random.choice([n for n in node_ids if n != source_id])
                
                # 随机选择交互类型
                interaction_types = list(InteractionType)
                interaction_type = np.random.choice(interaction_types)
                
                # 执行交互
                self.simulate_interaction(source_id, target_id, interaction_type)
            
            # 更新时间
            self.current_time += 0.1
            time.sleep(self.update_interval)
    
    def load_from_log(self, log_file: Optional[str] = None):
        """从日志文件加载数据"""
        if log_file is None:
            log_file = self.log_file
        
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            # 清空现有数据
            self.graph.clear()
            self.nodes.clear()
            self.edges.clear()
            self.events.clear()
            
            # 重新创建场景
            self.create_misinfo_scenario()
            
            # 重放事件
            for event_data in data.get("events", []):
                event = GraphEvent(
                    timestamp=event_data["timestamp"],
                    event_type=event_data["event_type"],
                    source_id=event_data["source_id"],
                    target_id=event_data.get("target_id"),
                    data=event_data.get("data", {})
                )
                self.events.append(event)
                
                # 重放交互
                if event.target_id:
                    interaction_type = InteractionType(event.event_type)
                    self.simulate_interaction(event.source_id, event.target_id, interaction_type)
            
            logger.info(f"从日志文件加载了{len(self.events)}个事件")
            
        except Exception as e:
            logger.error(f"加载日志文件失败: {e}")
    
    def export_visualization(self, filename: str = "sandgraph_visualization.png"):
        """导出可视化图像"""
        if self.fig:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图像已导出到: {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_events": len(self.events),
            "node_types": defaultdict(int),
            "edge_types": defaultdict(int),
            "average_belief": 0.0,
            "misinfo_spread_count": 0,
            "fact_check_count": 0,
            "cooperation_count": 0,
            "competition_count": 0
        }
        
        # 统计节点类型
        for node in self.nodes.values():
            stats["node_types"][node.node_type.value] += 1
        
        # 统计边类型
        for edge in self.edges:
            stats["edge_types"][edge.edge_type.value] += 1
            
            if edge.edge_type == EdgeType.MISINFO_SPREAD:
                stats["misinfo_spread_count"] += 1
            elif edge.edge_type == EdgeType.FACT_CHECK:
                stats["fact_check_count"] += 1
            elif edge.edge_type == EdgeType.COOPERATE:
                stats["cooperation_count"] += 1
            elif edge.edge_type == EdgeType.COMPETE:
                stats["competition_count"] += 1
        
        # 计算平均belief
        if self.nodes:
            stats["average_belief"] = np.mean([node.belief for node in self.nodes.values()])
        
        return stats


# 工厂函数
def create_sandgraph_visualizer(log_file: str = "sandgraph_visualization.log") -> SandGraphVisualizer:
    """创建SandGraph可视化器"""
    return SandGraphVisualizer(log_file)


def create_misinfo_visualization_demo():
    """创建misinformation传播可视化演示"""
    visualizer = create_sandgraph_visualizer()
    visualizer.create_misinfo_scenario(num_agents=15)
    return visualizer

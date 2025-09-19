#!/usr/bin/env python3
"""
Advanced Sandbox-RL Visualization Demo

高级可视化演示，支持两组对抗传播、动态颜色变化、GIF和PNG保存
"""

import os
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import threading
import imageio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """节点类型"""
    AGENT = "agent"
    MISINFO_SOURCE = "misinfo_source"
    FACT_CHECKER = "fact_checker"
    INFLUENCER = "influencer"
    REGULAR_USER = "regular_user"


class GroupType(Enum):
    """组类型"""
    GROUP_A = "group_a"  # 红色组
    GROUP_B = "group_b"  # 蓝色组
    NEUTRAL = "neutral"  # 中性组


class EdgeType(Enum):
    """边类型"""
    COOPERATE = "cooperate"
    COMPETE = "compete"
    MISINFO_SPREAD = "misinfo_spread"
    FACT_CHECK = "fact_check"
    INFLUENCE = "influence"
    NEUTRAL = "neutral"
    CROSS_GROUP = "cross_group"  # 跨组传播


class InteractionType(Enum):
    """交互类型"""
    SHARE = "share"
    LIKE = "like"
    COMMENT = "comment"
    FACT_CHECK = "fact_check"
    DEBUNK = "debunk"
    COOPERATE = "cooperate"
    COMPETE = "compete"
    CROSS_PROPAGATE = "cross_propagate"  # 跨组传播


@dataclass
class GraphNode:
    """图节点"""
    id: str
    node_type: NodeType
    group: GroupType = GroupType.NEUTRAL
    position: Tuple[float, float] = (0.0, 0.0)
    belief: float = 0.5  # 对misinformation的相信程度 (0-1)
    influence: float = 1.0  # 影响力
    followers: int = 0
    credibility: float = 0.5  # 可信度
    color: str = "blue"
    size: int = 100
    label: str = ""
    group_belief: float = 0.5  # 对所属组的忠诚度
    
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
    group_crossing: bool = False  # 是否跨组


class AdvancedSandbox-RLVisualizer:
    """高级Sandbox-RL可视化器"""
    
    def __init__(self, 
                 update_interval: float = 0.5,
                 max_nodes: int = 50,
                 max_edges: int = 100,
                 save_gif: bool = True,
                 save_png_steps: bool = True,
                 png_interval: int = 10):  # 每10帧保存一次PNG
        
        self.update_interval = update_interval
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.save_gif = save_gif
        self.save_png_steps = save_png_steps
        self.png_interval = png_interval
        
        # 图数据结构
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        
        # 可视化状态
        self.is_running = False
        self.current_time = 0.0
        self.frame_count = 0
        self.animation = None
        self.fig = None
        self.ax = None
        
        # GIF和PNG保存
        self.frames_for_gif = []
        self.png_save_path = "visualization_outputs/timesteps"
        os.makedirs(self.png_save_path, exist_ok=True)
        
        # 组颜色映射
        self.group_colors = {
            GroupType.GROUP_A: "#d62728",  # 红色
            GroupType.GROUP_B: "#1f77b4",  # 蓝色
            GroupType.NEUTRAL: "#7f7f7f"   # 灰色
        }
        
        # 边颜色映射
        self.edge_colors = {
            EdgeType.COOPERATE: "#2ca02c",
            EdgeType.COMPETE: "#d62728",
            EdgeType.MISINFO_SPREAD: "#ff7f0e",
            EdgeType.FACT_CHECK: "#1f77b4",
            EdgeType.INFLUENCE: "#9467bd",
            EdgeType.NEUTRAL: "#7f7f7f",
            EdgeType.CROSS_GROUP: "#e377c2"
        }
        
        logger.info("高级Sandbox-RL可视化器初始化完成")
    
    def add_node(self, 
                 node_id: str, 
                 node_type: NodeType,
                 group: GroupType = GroupType.NEUTRAL,
                 position: Optional[Tuple[float, float]] = None,
                 **kwargs) -> GraphNode:
        """添加节点"""
        if position is None:
            # 根据组分配位置
            if group == GroupType.GROUP_A:
                position = (np.random.uniform(-10, -2), np.random.uniform(-10, 10))
            elif group == GroupType.GROUP_B:
                position = (np.random.uniform(2, 10), np.random.uniform(-10, 10))
            else:
                position = (np.random.uniform(-2, 2), np.random.uniform(-10, 10))
        
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            group=group,
            position=position,
            color=self.group_colors.get(group, "#7f7f7f"),
            **kwargs
        )
        
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, 
                 source: str, 
                 target: str, 
                 edge_type: EdgeType,
                 weight: float = 1.0,
                 **kwargs) -> GraphEdge:
        """添加边"""
        source_node = self.nodes.get(source)
        target_node = self.nodes.get(target)
        
        # 检查是否跨组
        group_crossing = False
        if source_node and target_node:
            group_crossing = source_node.group != target_node.group
        
        edge = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            timestamp=self.current_time,
            color=self.edge_colors.get(edge_type, "#7f7f7f"),
            group_crossing=group_crossing,
            **kwargs
        )
        
        self.edges.append(edge)
        return edge
    
    def create_competing_scenario(self, num_agents_per_group: int = 8):
        """创建两组对抗场景"""
        # 清空现有图
        self.nodes.clear()
        self.edges.clear()
        
        # 创建Group A的misinformation源
        misinfo_source_a = self.add_node(
            "misinfo_source_a",
            NodeType.MISINFO_SOURCE,
            GroupType.GROUP_A,
            belief=1.0,
            influence=2.0,
            followers=1000,
            credibility=0.1,
            size=200,
            group_belief=1.0
        )
        
        # 创建Group B的misinformation源
        misinfo_source_b = self.add_node(
            "misinfo_source_b",
            NodeType.MISINFO_SOURCE,
            GroupType.GROUP_B,
            belief=1.0,
            influence=2.0,
            followers=1000,
            credibility=0.1,
            size=200,
            group_belief=1.0
        )
        
        # 创建Group A的fact checker
        fact_checker_a = self.add_node(
            "fact_checker_a",
            NodeType.FACT_CHECKER,
            GroupType.GROUP_A,
            belief=0.0,
            influence=1.5,
            followers=500,
            credibility=0.9,
            size=150,
            group_belief=0.9
        )
        
        # 创建Group B的fact checker
        fact_checker_b = self.add_node(
            "fact_checker_b",
            NodeType.FACT_CHECKER,
            GroupType.GROUP_B,
            belief=0.0,
            influence=1.5,
            followers=500,
            credibility=0.9,
            size=150,
            group_belief=0.9
        )
        
        # 创建Group A的influencer
        influencer_a = self.add_node(
            "influencer_a",
            NodeType.INFLUENCER,
            GroupType.GROUP_A,
            belief=0.8,
            influence=1.8,
            followers=800,
            credibility=0.3,
            size=180,
            group_belief=0.8
        )
        
        # 创建Group B的influencer
        influencer_b = self.add_node(
            "influencer_b",
            NodeType.INFLUENCER,
            GroupType.GROUP_B,
            belief=0.8,
            influence=1.8,
            followers=800,
            credibility=0.3,
            size=180,
            group_belief=0.8
        )
        
        # 创建Group A的普通用户
        for i in range(num_agents_per_group - 3):
            user_id = f"user_a_{i+1}"
            belief = np.random.uniform(0.6, 0.9)  # Group A倾向于相信misinformation
            influence = np.random.uniform(0.5, 1.5)
            
            user = self.add_node(
                user_id,
                NodeType.REGULAR_USER,
                GroupType.GROUP_A,
                belief=belief,
                influence=influence,
                followers=np.random.randint(10, 100),
                credibility=np.random.uniform(0.2, 0.6),
                size=50 + int(belief * 100),
                group_belief=belief
            )
        
        # 创建Group B的普通用户
        for i in range(num_agents_per_group - 3):
            user_id = f"user_b_{i+1}"
            belief = np.random.uniform(0.1, 0.4)  # Group B倾向于不相信misinformation
            influence = np.random.uniform(0.5, 1.5)
            
            user = self.add_node(
                user_id,
                NodeType.REGULAR_USER,
                GroupType.GROUP_B,
                belief=belief,
                influence=influence,
                followers=np.random.randint(10, 100),
                credibility=np.random.uniform(0.4, 0.8),
                size=50 + int(belief * 100),
                group_belief=belief
            )
        
        # 添加初始边（组内传播）
        self.add_edge("misinfo_source_a", "influencer_a", EdgeType.MISINFO_SPREAD, weight=0.8)
        self.add_edge("influencer_a", "user_a_1", EdgeType.INFLUENCE, weight=0.6)
        self.add_edge("fact_checker_a", "user_a_2", EdgeType.FACT_CHECK, weight=0.7)
        
        self.add_edge("misinfo_source_b", "influencer_b", EdgeType.MISINFO_SPREAD, weight=0.8)
        self.add_edge("influencer_b", "user_b_1", EdgeType.INFLUENCE, weight=0.6)
        self.add_edge("fact_checker_b", "user_b_2", EdgeType.FACT_CHECK, weight=0.7)
        
        logger.info(f"创建了包含{len(self.nodes)}个节点的两组对抗场景")
    
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
            
        elif interaction_type == InteractionType.CROSS_PROPAGATE:
            # 跨组传播
            belief_transfer = source_node.belief * source_node.influence * 0.05  # 跨组传播效果较弱
            target_node.belief = min(1.0, target_node.belief + belief_transfer)
            edge_type = EdgeType.CROSS_GROUP
            
        else:
            edge_type = EdgeType.NEUTRAL
        
        # 添加边
        self.add_edge(source_id, target_id, edge_type, weight=1.0)
        
        # 更新节点大小（基于belief）
        target_node.size = 50 + int(target_node.belief * 150)
        
        # 更新节点颜色（基于belief和组）
        self._update_node_color(target_node)
    
    def _update_node_color(self, node: GraphNode):
        """更新节点颜色"""
        if node.node_type == NodeType.REGULAR_USER:
            # 根据belief和组调整颜色
            if node.group == GroupType.GROUP_A:
                # Group A: 红色系，belief越高越红
                red = 0.8 + node.belief * 0.2
                green = 0.2 - node.belief * 0.2
                blue = 0.2 - node.belief * 0.2
            elif node.group == GroupType.GROUP_B:
                # Group B: 蓝色系，belief越高越蓝
                red = 0.2 - node.belief * 0.2
                green = 0.2 - node.belief * 0.2
                blue = 0.8 + node.belief * 0.2
            else:
                # 中性组: 灰色系
                intensity = 0.5 + node.belief * 0.5
                red = green = blue = intensity
            
            node.color = (red, green, blue)
        else:
            # 其他类型节点保持组颜色
            node.color = self.group_colors.get(node.group, "#7f7f7f")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "group_a_nodes": 0,
            "group_b_nodes": 0,
            "neutral_nodes": 0,
            "group_a_avg_belief": 0.0,
            "group_b_avg_belief": 0.0,
            "group_a_misinfo_count": 0,
            "group_b_misinfo_count": 0,
            "cross_group_count": 0,
            "misinfo_spread_count": 0,
            "fact_check_count": 0,
            "cooperation_count": 0,
            "competition_count": 0
        }
        
        # 按组统计
        group_a_beliefs = []
        group_b_beliefs = []
        
        for node in self.nodes.values():
            if node.group == GroupType.GROUP_A:
                stats["group_a_nodes"] += 1
                group_a_beliefs.append(node.belief)
            elif node.group == GroupType.GROUP_B:
                stats["group_b_nodes"] += 1
                group_b_beliefs.append(node.belief)
            else:
                stats["neutral_nodes"] += 1
        
        # 计算平均belief
        if group_a_beliefs:
            stats["group_a_avg_belief"] = np.mean(group_a_beliefs)
        if group_b_beliefs:
            stats["group_b_avg_belief"] = np.mean(group_b_beliefs)
        
        # 统计边类型
        for edge in self.edges:
            if edge.edge_type == EdgeType.MISINFO_SPREAD:
                stats["misinfo_spread_count"] += 1
                # 统计各组misinformation传播
                source_node = self.nodes.get(edge.source)
                if source_node and source_node.group == GroupType.GROUP_A:
                    stats["group_a_misinfo_count"] += 1
                elif source_node and source_node.group == GroupType.GROUP_B:
                    stats["group_b_misinfo_count"] += 1
            elif edge.edge_type == EdgeType.FACT_CHECK:
                stats["fact_check_count"] += 1
            elif edge.edge_type == EdgeType.COOPERATE:
                stats["cooperation_count"] += 1
            elif edge.edge_type == EdgeType.COMPETE:
                stats["competition_count"] += 1
            elif edge.edge_type == EdgeType.CROSS_GROUP:
                stats["cross_group_count"] += 1
        
        return stats


def main():
    """主演示函数"""
    print("🚀 高级Sandbox-RL两组对抗可视化演示")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = AdvancedSandbox-RLVisualizer(
        update_interval=0.3,
        save_gif=True,
        save_png_steps=True,
        png_interval=5  # 每5帧保存一次PNG
    )
    
    # 创建两组对抗场景
    print("创建两组对抗场景...")
    visualizer.create_competing_scenario(num_agents_per_group=10)
    
    # 显示初始统计
    stats = visualizer.get_statistics()
    print(f"初始状态:")
    print(f"  - 总节点数: {stats['total_nodes']}")
    print(f"  - Group A节点数: {stats['group_a_nodes']}")
    print(f"  - Group B节点数: {stats['group_b_nodes']}")
    print(f"  - Group A平均belief: {stats['group_a_avg_belief']:.3f}")
    print(f"  - Group B平均belief: {stats['group_b_avg_belief']:.3f}")
    
    print(f"\n🎯 启动动态可视化...")
    print("注意: 这将打开一个matplotlib窗口，显示实时动态图")
    print("窗口将显示:")
    print("  - 左侧红色区域: Group A")
    print("  - 右侧蓝色区域: Group B")
    print("  - 中间灰色区域: 中性区域")
    print("  - 节点颜色: 根据belief和组动态变化")
    print("  - 橙色线: Misinformation传播")
    print("  - 蓝色线: 事实核查")
    print("  - 绿色虚线: 合作")
    print("  - 红色点线: 竞争")
    print("  - 紫色点划线: 跨组传播")
    print("  - 自动保存GIF和PNG文件")
    
    # 启动可视化
    try:
        visualizer.start_visualization(duration=30.0)  # 运行30秒
    except KeyboardInterrupt:
        print("\n用户中断，停止可视化")
        visualizer.stop_visualization()
    except Exception as e:
        print(f"\n可视化出错: {e}")
        visualizer.stop_visualization()


if __name__ == "__main__":
    main()

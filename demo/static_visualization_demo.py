#!/usr/bin/env python3
"""
Static Sandbox-RL Visualization Demo

静态可视化演示，生成并保存可视化图像
"""

import os
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
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


class StaticSandbox-RLVisualizer:
    """静态Sandbox-RL可视化器"""
    
    def __init__(self, max_nodes: int = 50, max_edges: int = 100):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        
        # 图数据结构
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        
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
        
        logger.info("静态Sandbox-RL可视化器初始化完成")
    
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
            timestamp=time.time(),
            color=self.edge_colors.get(edge_type, "#7f7f7f"),
            **kwargs
        )
        
        self.edges.append(edge)
        return edge
    
    def create_misinfo_scenario(self, num_agents: int = 10):
        """创建misinformation传播场景"""
        # 清空现有图
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
        
        # 更新节点大小（基于belief）
        target_node.size = 50 + int(target_node.belief * 150)
    
    def create_visualization(self, title: str = "Sandbox-RL Visualization", save_path: Optional[str] = None):
        """创建可视化图像"""
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, 10))
        
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
            
            ax.scatter(x, y, s=size, c=[color], alpha=0.7, edgecolors='black', linewidth=1)
            
            # 添加标签
            if node.influence > 1.2:  # 只显示重要节点的标签
                ax.annotate(node.label, (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, fontweight='bold')
        
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
                linewidth = 1.5
                
                if edge.edge_type == EdgeType.COOPERATE:
                    linestyle = '--'
                    alpha = 0.8
                    linewidth = 2.0
                elif edge.edge_type == EdgeType.COMPETE:
                    linestyle = ':'
                    alpha = 0.8
                    linewidth = 2.0
                elif edge.edge_type == EdgeType.MISINFO_SPREAD:
                    alpha = 0.9
                    linewidth = 2.5
                elif edge.edge_type == EdgeType.FACT_CHECK:
                    alpha = 0.8
                    linewidth = 2.0
                
                ax.plot([x1, x2], [y1, y2], color=edge.color, 
                       linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        
        # 设置画布
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_aspect('equal')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=12, label='Misinfo Source', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=12, label='Fact Checker', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=12, label='Influencer', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                      markersize=12, label='Regular User', markeredgecolor='black')
        ]
        
        # 添加边类型图例
        edge_legend_elements = [
            plt.Line2D([0], [0], color='orange', linewidth=2.5, label='Misinfo Spread'),
            plt.Line2D([0], [0], color='blue', linewidth=2.0, label='Fact Check'),
            plt.Line2D([0], [0], color='green', linewidth=2.0, linestyle='--', label='Cooperation'),
            plt.Line2D([0], [0], color='red', linewidth=2.0, linestyle=':', label='Competition')
        ]
        
        # 合并图例
        all_legend_elements = legend_elements + edge_legend_elements
        ax.legend(handles=all_legend_elements, loc='upper right', fontsize=10, 
                 bbox_to_anchor=(1.15, 1.0))
        
        # 添加统计信息
        stats = self.get_statistics()
        stats_text = f"""
Statistics:
• Total Nodes: {stats['total_nodes']}
• Total Edges: {stats['total_edges']}
• Average Belief: {stats['average_belief']:.3f}
• Misinfo Spread: {stats['misinfo_spread_count']}
• Fact Checks: {stats['fact_check_count']}
• Cooperation: {stats['cooperation_count']}
• Competition: {stats['competition_count']}
        """
        
        ax.text(-11, -10, stats_text, fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图像已保存到: {save_path}")
        
        return fig, ax
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
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


def main():
    """主演示函数"""
    print("🚀 静态Sandbox-RL可视化演示")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = StaticSandbox-RLVisualizer()
    
    # 创建misinformation场景
    print("创建misinformation传播场景...")
    visualizer.create_misinfo_scenario(num_agents=20)
    
    # 显示初始统计
    stats = visualizer.get_statistics()
    print(f"初始状态:")
    print(f"  - 节点数: {stats['total_nodes']}")
    print(f"  - 边数: {stats['total_edges']}")
    print(f"  - 平均belief: {stats['average_belief']:.3f}")
    
    print(f"\n节点类型分布:")
    for node_type, count in stats['node_types'].items():
        print(f"  - {node_type}: {count}")
    
    # 模拟大量交互
    print(f"\n模拟交互过程...")
    node_ids = list(visualizer.nodes.keys())
    
    # 第一轮：misinformation传播
    print("  第一轮: Misinformation传播")
    for i in range(10):
        source_id = np.random.choice(node_ids)
        target_id = np.random.choice([n for n in node_ids if n != source_id])
        visualizer.simulate_interaction(source_id, target_id, InteractionType.SHARE)
    
    # 第二轮：事实核查
    print("  第二轮: 事实核查")
    for i in range(8):
        source_id = np.random.choice(node_ids)
        target_id = np.random.choice([n for n in node_ids if n != source_id])
        visualizer.simulate_interaction(source_id, target_id, InteractionType.FACT_CHECK)
    
    # 第三轮：合作和竞争
    print("  第三轮: 合作和竞争")
    for i in range(12):
        source_id = np.random.choice(node_ids)
        target_id = np.random.choice([n for n in node_ids if n != source_id])
        interaction_type = InteractionType.COOPERATE if i % 2 == 0 else InteractionType.COMPETE
        visualizer.simulate_interaction(source_id, target_id, interaction_type)
    
    # 显示最终统计
    stats = visualizer.get_statistics()
    print(f"\n最终状态:")
    print(f"  - 节点数: {stats['total_nodes']}")
    print(f"  - 边数: {stats['total_edges']}")
    print(f"  - 平均belief: {stats['average_belief']:.3f}")
    print(f"  - Misinformation传播: {stats['misinfo_spread_count']}")
    print(f"  - 事实核查: {stats['fact_check_count']}")
    print(f"  - 合作: {stats['cooperation_count']}")
    print(f"  - 竞争: {stats['competition_count']}")
    
    # 创建可视化图像
    print(f"\n🎨 生成可视化图像...")
    
    # 确保输出目录存在
    os.makedirs("visualization_outputs", exist_ok=True)
    
    # 生成主可视化图像
    fig, ax = visualizer.create_visualization(
        title="Sandbox-RL: Misinformation Propagation & Cooperation/Competition Network",
        save_path="visualization_outputs/sandgraph_network_visualization.png"
    )
    
    # 生成详细统计图
    create_statistics_visualization(visualizer, "visualization_outputs/sandgraph_statistics.png")
    
    print("✅ 可视化演示完成！")
    print("📁 生成的图像文件:")
    print("  - visualization_outputs/sandgraph_network_visualization.png")
    print("  - visualization_outputs/sandgraph_statistics.png")
    
    # 显示图像
    plt.show()


def create_statistics_visualization(visualizer, save_path: str):
    """创建统计可视化"""
    stats = visualizer.get_statistics()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 节点类型分布饼图
    node_types = list(stats['node_types'].keys())
    node_counts = list(stats['node_types'].values())
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    
    ax1.pie(node_counts, labels=node_types, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Node Type Distribution', fontsize=14, fontweight='bold')
    
    # 2. 边类型分布柱状图
    edge_types = list(stats['edge_types'].keys())
    edge_counts = list(stats['edge_types'].values())
    edge_colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
    
    bars = ax2.bar(edge_types, edge_counts, color=edge_colors[:len(edge_types)])
    ax2.set_title('Edge Type Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count')
    
    # 添加数值标签
    for bar, count in zip(bars, edge_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    # 3. Belief分布直方图
    beliefs = [node.belief for node in visualizer.nodes.values()]
    ax3.hist(beliefs, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.set_title('Belief Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Belief Value')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(beliefs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(beliefs):.3f}')
    ax3.legend()
    
    # 4. 交互统计雷达图
    interaction_stats = [
        stats['misinfo_spread_count'],
        stats['fact_check_count'],
        stats['cooperation_count'],
        stats['competition_count']
    ]
    
    categories = ['Misinfo\nSpread', 'Fact\nCheck', 'Cooperation', 'Competition']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    interaction_stats += interaction_stats[:1]  # 闭合图形
    angles += angles[:1]
    
    ax4.plot(angles, interaction_stats, 'o-', linewidth=2, color='orange')
    ax4.fill(angles, interaction_stats, alpha=0.25, color='orange')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('Interaction Statistics', fontsize=14, fontweight='bold')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"统计可视化已保存到: {save_path}")


if __name__ == "__main__":
    main()

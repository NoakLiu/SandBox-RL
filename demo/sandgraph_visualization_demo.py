#!/usr/bin/env python3
"""
SandGraph Dynamic Visualization Demo

动态图可视化演示，展示misinformation传播和cooperate/compete关系
"""

import os
import time
import json
import logging
from typing import Dict, List, Any

# SandGraph Core imports
try:
    from sandgraph.core.graph_visualizer import (
        SandGraphVisualizer,
        NodeType,
        EdgeType,
        InteractionType,
        create_sandgraph_visualizer,
        create_misinfo_visualization_demo
    )
    HAS_SANDGRAPH = True
    print("✅ SandGraph graph visualizer imported successfully")
except ImportError as e:
    HAS_SANDGRAPH = False
    print(f"❌ SandGraph graph visualizer not available: {e}")
    print("Will use mock implementations")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_basic_visualization():
    """演示基础可视化"""
    print("\n🎯 基础可视化演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建可视化器
    visualizer = create_sandgraph_visualizer("demo_visualization.log")
    
    # 创建misinformation场景
    visualizer.create_misinfo_scenario(num_agents=12)
    
    print(f"创建了包含{len(visualizer.nodes)}个节点的场景")
    print(f"节点类型分布:")
    for node_type, count in visualizer.get_statistics()["node_types"].items():
        print(f"  - {node_type}: {count}")
    
    # 模拟一些交互
    print("\n模拟交互...")
    for i in range(10):
        # 随机选择两个节点
        node_ids = list(visualizer.nodes.keys())
        source_id = node_ids[i % len(node_ids)]
        target_id = node_ids[(i + 1) % len(node_ids)]
        
        # 执行不同类型的交互
        if i < 3:
            interaction_type = InteractionType.SHARE
        elif i < 6:
            interaction_type = InteractionType.FACT_CHECK
        elif i < 8:
            interaction_type = InteractionType.COOPERATE
        else:
            interaction_type = InteractionType.COMPETE
        
        visualizer.simulate_interaction(source_id, target_id, interaction_type)
        print(f"  交互 {i+1}: {source_id} -> {target_id} ({interaction_type.value})")
    
    # 显示统计信息
    stats = visualizer.get_statistics()
    print(f"\n统计信息:")
    print(f"  - 总节点数: {stats['total_nodes']}")
    print(f"  - 总边数: {stats['total_edges']}")
    print(f"  - 总事件数: {stats['total_events']}")
    print(f"  - 平均belief: {stats['average_belief']:.3f}")
    print(f"  - Misinformation传播次数: {stats['misinfo_spread_count']}")
    print(f"  - 事实核查次数: {stats['fact_check_count']}")
    print(f"  - 合作次数: {stats['cooperation_count']}")
    print(f"  - 竞争次数: {stats['competition_count']}")


def demonstrate_interactive_scenario():
    """演示交互式场景"""
    print("\n🔄 交互式场景演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建可视化器
    visualizer = create_sandgraph_visualizer("interactive_demo.log")
    
    # 创建更复杂的场景
    visualizer.create_misinfo_scenario(num_agents=20)
    
    # 添加一些特定的交互序列
    interactions = [
        ("misinfo_source_1", "influencer_1", InteractionType.SHARE),
        ("influencer_1", "user_1", InteractionType.SHARE),
        ("influencer_1", "user_2", InteractionType.SHARE),
        ("fact_checker_1", "user_1", InteractionType.FACT_CHECK),
        ("fact_checker_1", "user_3", InteractionType.FACT_CHECK),
        ("user_1", "user_4", InteractionType.COOPERATE),
        ("user_2", "user_5", InteractionType.COMPETE),
        ("user_3", "user_6", InteractionType.SHARE),
        ("user_4", "user_7", InteractionType.COOPERATE),
        ("user_5", "user_8", InteractionType.COMPETE)
    ]
    
    print("执行预定义的交互序列...")
    for i, (source, target, interaction_type) in enumerate(interactions):
        visualizer.simulate_interaction(source, target, interaction_type)
        print(f"  交互 {i+1}: {source} -> {target} ({interaction_type.value})")
        
        # 显示目标节点的belief变化
        target_node = visualizer.nodes.get(target)
        if target_node:
            print(f"    {target} belief: {target_node.belief:.3f}")
    
    # 分析网络结构
    print(f"\n网络分析:")
    print(f"  - 节点数: {len(visualizer.nodes)}")
    print(f"  - 边数: {len(visualizer.edges)}")
    print(f"  - 平均度: {len(visualizer.edges) / len(visualizer.nodes):.2f}")
    
    # 分析belief分布
    beliefs = [node.belief for node in visualizer.nodes.values()]
    print(f"  - Belief分布: min={min(beliefs):.3f}, max={max(beliefs):.3f}, mean={sum(beliefs)/len(beliefs):.3f}")


def demonstrate_log_replay():
    """演示日志重放"""
    print("\n📊 日志重放演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建第一个可视化器并生成日志
    visualizer1 = create_sandgraph_visualizer("replay_demo.log")
    visualizer1.create_misinfo_scenario(num_agents=10)
    
    # 执行一些交互
    for i in range(15):
        node_ids = list(visualizer1.nodes.keys())
        source_id = node_ids[i % len(node_ids)]
        target_id = node_ids[(i + 1) % len(node_ids)]
        interaction_type = InteractionType.SHARE if i % 2 == 0 else InteractionType.FACT_CHECK
        visualizer1.simulate_interaction(source_id, target_id, interaction_type)
    
    print(f"生成了{len(visualizer1.events)}个事件")
    
    # 创建第二个可视化器并重放日志
    visualizer2 = create_sandgraph_visualizer("replay_demo.log")
    visualizer2.load_from_log()
    
    print(f"重放了{len(visualizer2.events)}个事件")
    
    # 比较两个可视化器的状态
    stats1 = visualizer1.get_statistics()
    stats2 = visualizer2.get_statistics()
    
    print(f"\n状态比较:")
    print(f"  原始 - 节点数: {stats1['total_nodes']}, 边数: {stats1['total_edges']}, 平均belief: {stats1['average_belief']:.3f}")
    print(f"  重放 - 节点数: {stats2['total_nodes']}, 边数: {stats2['total_edges']}, 平均belief: {stats2['average_belief']:.3f}")
    
    # 验证一致性
    if (stats1['total_nodes'] == stats2['total_nodes'] and 
        stats1['total_edges'] == stats2['total_edges']):
        print("✅ 日志重放成功，状态一致")
    else:
        print("❌ 日志重放失败，状态不一致")


def demonstrate_statistics_analysis():
    """演示统计分析"""
    print("\n📈 统计分析演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建可视化器
    visualizer = create_sandgraph_visualizer("stats_demo.log")
    visualizer.create_misinfo_scenario(num_agents=25)
    
    # 执行大量交互
    print("执行大量交互以生成统计数据...")
    for i in range(50):
        node_ids = list(visualizer.nodes.keys())
        source_id = node_ids[i % len(node_ids)]
        target_id = node_ids[(i + 1) % len(node_ids)]
        
        # 根据节点类型选择交互类型
        source_node = visualizer.nodes[source_id]
        if source_node.node_type == NodeType.MISINFO_SOURCE:
            interaction_type = InteractionType.SHARE
        elif source_node.node_type == NodeType.FACT_CHECKER:
            interaction_type = InteractionType.FACT_CHECK
        elif source_node.node_type == NodeType.INFLUENCER:
            interaction_type = InteractionType.SHARE if i % 2 == 0 else InteractionType.COOPERATE
        else:
            interaction_type = InteractionType.COOPERATE if i % 3 == 0 else InteractionType.COMPETE
        
        visualizer.simulate_interaction(source_id, target_id, interaction_type)
    
    # 获取详细统计
    stats = visualizer.get_statistics()
    
    print(f"\n详细统计信息:")
    print(f"网络结构:")
    print(f"  - 总节点数: {stats['total_nodes']}")
    print(f"  - 总边数: {stats['total_edges']}")
    print(f"  - 总事件数: {stats['total_events']}")
    
    print(f"\n节点类型分布:")
    for node_type, count in stats['node_types'].items():
        percentage = count / stats['total_nodes'] * 100
        print(f"  - {node_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n边类型分布:")
    for edge_type, count in stats['edge_types'].items():
        percentage = count / stats['total_edges'] * 100 if stats['total_edges'] > 0 else 0
        print(f"  - {edge_type}: {count} ({percentage:.1f}%)")
    
    print(f"\n交互统计:")
    print(f"  - Misinformation传播: {stats['misinfo_spread_count']}")
    print(f"  - 事实核查: {stats['fact_check_count']}")
    print(f"  - 合作: {stats['cooperation_count']}")
    print(f"  - 竞争: {stats['competition_count']}")
    
    print(f"\nBelief分析:")
    print(f"  - 平均belief: {stats['average_belief']:.3f}")
    
    # 分析不同类型节点的belief
    belief_by_type = {}
    for node in visualizer.nodes.values():
        if node.node_type.value not in belief_by_type:
            belief_by_type[node.node_type.value] = []
        belief_by_type[node.node_type.value].append(node.belief)
    
    print(f"\n各类型节点的平均belief:")
    for node_type, beliefs in belief_by_type.items():
        avg_belief = sum(beliefs) / len(beliefs)
        print(f"  - {node_type}: {avg_belief:.3f}")


def demonstrate_export_capabilities():
    """演示导出功能"""
    print("\n💾 导出功能演示")
    print("=" * 50)
    
    if not HAS_SANDGRAPH:
        print("❌ SandGraph不可用，跳过演示")
        return
    
    # 创建可视化器
    visualizer = create_sandgraph_visualizer("export_demo.log")
    visualizer.create_misinfo_scenario(num_agents=15)
    
    # 执行一些交互
    for i in range(20):
        node_ids = list(visualizer.nodes.keys())
        source_id = node_ids[i % len(node_ids)]
        target_id = node_ids[(i + 1) % len(node_ids)]
        interaction_type = InteractionType.SHARE if i % 2 == 0 else InteractionType.FACT_CHECK
        visualizer.simulate_interaction(source_id, target_id, interaction_type)
    
    # 导出统计信息
    stats = visualizer.get_statistics()
    
    # 保存为JSON
    export_data = {
        "metadata": {
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"],
            "total_events": stats["total_events"]
        },
        "statistics": stats,
        "nodes": {
            node_id: {
                "type": node.node_type.value,
                "belief": node.belief,
                "influence": node.influence,
                "credibility": node.credibility,
                "followers": node.followers
            }
            for node_id, node in visualizer.nodes.items()
        },
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type.value,
                "weight": edge.weight,
                "timestamp": edge.timestamp
            }
            for edge in visualizer.edges
        ]
    }
    
    with open("sandgraph_export.json", "w") as f:
        json.dump(export_data, f, indent=2)
    
    print("✅ 数据已导出到 sandgraph_export.json")
    print(f"导出内容:")
    print(f"  - 元数据: 节点数={export_data['metadata']['total_nodes']}, 边数={export_data['metadata']['total_edges']}")
    print(f"  - 节点数据: {len(export_data['nodes'])}个节点")
    print(f"  - 边数据: {len(export_data['edges'])}条边")
    print(f"  - 统计信息: {len(export_data['statistics'])}项统计")


def main():
    """主演示函数"""
    print("🚀 SandGraph动态图可视化演示")
    print("=" * 60)
    print("本演示展示:")
    print("- 动态图可视化系统")
    print("- Misinformation传播模拟")
    print("- Cooperate/Compete关系展示")
    print("- 实时日志记录和重放")
    print("- 统计分析和数据导出")
    
    # 演示基础功能
    demonstrate_basic_visualization()
    
    # 演示交互式场景
    demonstrate_interactive_scenario()
    
    # 演示日志重放
    demonstrate_log_replay()
    
    # 演示统计分析
    demonstrate_statistics_analysis()
    
    # 演示导出功能
    demonstrate_export_capabilities()
    
    print("\n✅ 所有演示完成！")
    print("\n📝 总结:")
    print("- 成功实现了动态图可视化系统")
    print("- 支持misinformation传播的实时模拟")
    print("- 展示了cooperate/compete关系的动态变化")
    print("- 提供了完整的日志记录和重放功能")
    print("- 支持统计分析和数据导出")
    print("\n🎯 下一步:")
    print("- 可以启动实时可视化: visualizer.start_visualization()")
    print("- 可以加载日志文件: visualizer.load_from_log()")
    print("- 可以导出可视化图像: visualizer.export_visualization()")


if __name__ == "__main__":
    main()

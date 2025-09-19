#!/usr/bin/env python3
"""
Test Graph Visualizer

简单测试图可视化器功能
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sandbox_rl.core.graph_visualizer import (
        NodeType,
        EdgeType,
        InteractionType,
        GraphNode,
        GraphEdge,
        GraphEvent,
        Sandbox-RLVisualizer,
        create_sandgraph_visualizer
    )
    print("✅ Successfully imported graph visualizer modules")
    
    # Test node types
    print("\n🎯 Testing Node Types:")
    for node_type in NodeType:
        print(f"  - {node_type.value}")
    
    # Test edge types
    print("\n🔗 Testing Edge Types:")
    for edge_type in EdgeType:
        print(f"  - {edge_type.value}")
    
    # Test interaction types
    print("\n🤝 Testing Interaction Types:")
    for interaction_type in InteractionType:
        print(f"  - {interaction_type.value}")
    
    # Test graph node creation
    print("\n📊 Testing Graph Node Creation:")
    node = GraphNode(
        id="test_node",
        node_type=NodeType.AGENT,
        belief=0.7,
        influence=1.5,
        followers=100,
        credibility=0.8
    )
    print(f"  - ID: {node.id}")
    print(f"  - Type: {node.node_type.value}")
    print(f"  - Belief: {node.belief}")
    print(f"  - Influence: {node.influence}")
    print(f"  - Followers: {node.followers}")
    print(f"  - Credibility: {node.credibility}")
    print(f"  - Label: {node.label}")
    
    # Test graph edge creation
    print("\n🔗 Testing Graph Edge Creation:")
    edge = GraphEdge(
        source="node1",
        target="node2",
        edge_type=EdgeType.COOPERATE,
        weight=0.8,
        timestamp=123.45
    )
    print(f"  - Source: {edge.source}")
    print(f"  - Target: {edge.target}")
    print(f"  - Type: {edge.edge_type.value}")
    print(f"  - Weight: {edge.weight}")
    print(f"  - Timestamp: {edge.timestamp}")
    
    # Test graph event creation
    print("\n📝 Testing Graph Event Creation:")
    event = GraphEvent(
        timestamp=123.45,
        event_type="share",
        source_id="node1",
        target_id="node2",
        data={"belief": 0.7, "influence": 1.2}
    )
    print(f"  - Timestamp: {event.timestamp}")
    print(f"  - Event Type: {event.event_type}")
    print(f"  - Source ID: {event.source_id}")
    print(f"  - Target ID: {event.target_id}")
    print(f"  - Data: {event.data}")
    
    # Test visualizer creation
    print("\n🤖 Testing Visualizer Creation:")
    visualizer = create_sandgraph_visualizer("test_visualization.log")
    print(f"  - Log File: {visualizer.log_file}")
    print(f"  - Update Interval: {visualizer.update_interval}")
    print(f"  - Max Nodes: {visualizer.max_nodes}")
    print(f"  - Max Edges: {visualizer.max_edges}")
    
    # Test scenario creation
    print("\n🎬 Testing Scenario Creation:")
    visualizer.create_misinfo_scenario(num_agents=8)
    print(f"  - Total Nodes: {len(visualizer.nodes)}")
    print(f"  - Total Edges: {len(visualizer.edges)}")
    
    # Test node types in scenario
    print(f"  - Node Types:")
    for node_type, count in visualizer.get_statistics()["node_types"].items():
        print(f"    * {node_type}: {count}")
    
    # Test interaction simulation
    print("\n🔄 Testing Interaction Simulation:")
    node_ids = list(visualizer.nodes.keys())
    if len(node_ids) >= 2:
        source_id = node_ids[0]
        target_id = node_ids[1]
        
        # Test different interaction types
        for interaction_type in [InteractionType.SHARE, InteractionType.FACT_CHECK, InteractionType.COOPERATE]:
            visualizer.simulate_interaction(source_id, target_id, interaction_type)
            print(f"  - {source_id} -> {target_id} ({interaction_type.value})")
        
        # Check belief changes
        target_node = visualizer.nodes.get(target_id)
        if target_node:
            print(f"  - {target_id} belief after interactions: {target_node.belief:.3f}")
    
    # Test statistics
    print("\n📈 Testing Statistics:")
    stats = visualizer.get_statistics()
    print(f"  - Total Nodes: {stats['total_nodes']}")
    print(f"  - Total Edges: {stats['total_edges']}")
    print(f"  - Total Events: {stats['total_events']}")
    print(f"  - Average Belief: {stats['average_belief']:.3f}")
    print(f"  - Misinfo Spread Count: {stats['misinfo_spread_count']}")
    print(f"  - Fact Check Count: {stats['fact_check_count']}")
    print(f"  - Cooperation Count: {stats['cooperation_count']}")
    print(f"  - Competition Count: {stats['competition_count']}")
    
    print("\n✅ All tests passed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the Sandbox-RL root directory")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()

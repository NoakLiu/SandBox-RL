# Examples Guide

This guide provides detailed explanations and usage examples for all SandGraphX demos and features.

## 📋 Table of Contents

1. [Basic Demos](#basic-demos)
2. [Social Network Demos](#social-network-demos)
3. [Advanced Features](#advanced-features)
4. [Monitoring Examples](#monitoring-examples)
5. [Integration Examples](#integration-examples)

## 🎯 Basic Demos

### 1. Trading System Demo

**Purpose**: Demonstrates LLM-based trading decision making with RL optimization.

**Input**: Market data, portfolio state, trading parameters  
**Process**: LLM analyzes market → generates trading decisions → RL optimizes strategy  
**Output**: Trading actions, performance metrics, optimized weights

```bash
# Run trading demo
python demo/trading_demo.py --strategy simulated --steps 5
```

**Key Features**:
- Market data simulation
- LLM-based trading decisions
- RL strategy optimization
- Performance tracking

### 2. Social Network Analysis Demo

**Purpose**: Basic social network analysis with LLM insights and RL optimization.

**Input**: Network topology, user interactions, content data  
**Process**: LLM analyzes patterns → generates insights → RL optimizes recommendations  
**Output**: Network insights, user recommendations, engagement metrics

```bash
# Run social network demo
python demo/social_network_demo.py --steps 10
```

**Key Features**:
- Network topology analysis
- User interaction modeling
- Content recommendation
- Engagement optimization

## 🌐 Social Network Demos

### 3. Enhanced Social Network with Monitoring

**Purpose**: Advanced social network simulation with comprehensive monitoring and visualization.

**Input**: Network data with comprehensive monitoring  
**Process**: LLM analysis → RL optimization → Real-time monitoring → Advanced visualization  
**Output**: Network insights, performance metrics, interactive dashboards, trend analysis

```bash
# Run enhanced social network demo with monitoring
python demo/enhanced_social_network_demo.py \
    --steps 20 \
    --initial-users 100 \
    --enable-wandb \
    --enable-tensorboard \
    --wandb-project "sandgraph-enhanced-social"
```

**Key Features**:
- Real-time monitoring with WanDB
- Advanced visualization
- Performance tracking
- Network dynamics simulation
- Multi-backend logging

### 4. OASIS Social Network Simulation

**Purpose**: OASIS-style social network simulation with user behavior modeling.

**Input**: User profiles, social network topology, content data  
**Process**: LLM analyzes social dynamics → generates user behaviors → RL optimizes engagement strategies  
**Output**: Social interactions, network growth metrics, engagement optimization

```bash
# Run OASIS social network demo
python demo/oasis_social_demo.py --steps 5
```

**Key Features**:
- OASIS-style user modeling
- Social dynamics simulation
- Network growth tracking
- Engagement optimization

### 5. Enhanced OASIS Social Demo

**Purpose**: Enhanced OASIS demo with monitoring and advanced features.

```bash
# Run enhanced OASIS demo
python demo/enhanced_oasis_social_demo.py \
    --steps 15 \
    --initial-users 50 \
    --enable-wandb \
    --wandb-project "sandgraph-oasis-enhanced"
```

**Key Features**:
- Enhanced monitoring
- Advanced user modeling
- Real-time metrics
- Performance optimization

### 6. Misinformation Spread Analysis

**Purpose**: Analyze and combat misinformation spread in social networks.

**Input**: Social network data, user beliefs, information content  
**Process**: LLM analyzes misinformation patterns → generates intervention strategies → RL optimizes intervention effectiveness  
**Output**: Intervention actions, belief change metrics, spread reduction statistics

```bash
# Run misinformation spread demo
python demo/misinformation_spread_demo.py --steps 5
```

**Key Features**:
- Misinformation detection
- Intervention strategies
- Belief impact modeling
- Spread reduction analysis

### 7. Comprehensive Misinformation Analysis

**Purpose**: Multi-agent competition in misinformation spread with integrated optimization.

**Input**: Social network data, multi-agent competition, integrated optimization  
**Process**: SandGraph LLM vs Rules vs Human simulation → Real-time monitoring → Integrated optimization → Performance comparison  
**Output**: Competition results, network dynamics, belief impact analysis, comprehensive metrics

```bash
# Run comprehensive misinformation demo with full integration
python demo/comprehensive_misinformation_demo.py \
    --steps 50 \
    --num-users 1000 \
    --enable-wandb \
    --enable-tensorboard \
    --wandb-project "sandgraph-misinformation-competition"

# Run with custom configuration
python demo/comprehensive_misinformation_demo.py \
    --steps 100 \
    --num-users 2000 \
    --network-density 0.15 \
    --model-name "mistralai/Mistral-7B-Instruct-v0.2" \
    --kv-cache-size 10000 \
    --max-memory-gb 8.0 \
    --rollout-batch-size 32 \
    --posts-per-agent 5 \
    --enable-wandb \
    --wandb-project "sandgraph-advanced-misinformation"
```

**Key Features**:
- **Multi-Agent Competition**: SandGraph LLM competes against rule-based and human-simulated agents
- **Integrated Optimization**: Combines LLM frozen & adaptive update with AReaL KV cache optimization
- **Real-time Monitoring**: WanDB and TensorBoard integration for comprehensive tracking
- **Network Dynamics**: Realistic social network behavior with belief impact modeling
- **Performance Analysis**: Detailed comparison of agent performance in misinformation spread

**Expected Results**:
- SandGraph LLM should achieve higher misinformation spread percentage (>50%)
- Superior belief impact compared to traditional approaches
- Real-time performance tracking and visualization
- Comprehensive analysis of network dynamics and agent behavior

## 🔧 Advanced Features

### 8. LLMs Frozen & Adaptive Update

**Purpose**: Advanced parameter management for large language models.

**Input**: LLM model, training data, performance metrics  
**Process**: Parameter importance analysis → selective freezing → adaptive updates → performance monitoring  
**Output**: Optimized model parameters, performance statistics, update history

```bash
# Run simple demo (no numpy required)
python demo/llm_frozen_adaptive_simple_demo.py

# Run full demo (requires numpy)
python demo/llm_frozen_adaptive_demo.py --demo all

# Run specific demo
python demo/llm_frozen_adaptive_simple_demo.py --demo adaptive
```

**Key Features**:
- Parameter freezing/unfreezing
- Multiple update strategies
- Adaptive learning rates
- Performance monitoring
- Checkpoint management

### 9. AReaL KV Cache Optimization

**Purpose**: Advanced RL training optimizations based on AReaL framework.

**Input**: RL training data, KV cache configuration, rollout parameters  
**Process**: Asynchronous RL training → streaming generation → KV cache management → decoupled PPO updates  
**Output**: Optimized policies, cache performance metrics, training statistics

```bash
# Run all AReaL optimizations
python demo/areal_kv_cache_demo.py --demo all

# Run specific components
python demo/areal_kv_cache_demo.py --demo kv_cache --cache-size 5000
python demo/areal_kv_cache_demo.py --demo rollout --batch-size 8
python demo/areal_kv_cache_demo.py --demo ppo --memory-gb 4.0
python demo/areal_kv_cache_demo.py --demo streaming --enable-streaming
```

**Key Features**:
- Asynchronous RL training
- Streaming generation
- KV cache management
- Decoupled PPO updates
- Memory optimization

### 10. Enhanced RL Cache with Areal Integration

**Purpose**: Enhanced RL training with Areal framework integration.

**Input**: Enhanced RL configuration, Areal framework integration  
**Process**: Advanced caching → parallel processing → performance optimization → comprehensive monitoring  
**Output**: Enhanced training performance, cache statistics, memory efficiency metrics

```bash
# Run enhanced RL cache demo
python demo/enhanced_rl_cache_demo.py \
    --demo all \
    --cache-size 10000 \
    --enable-parallel
```

**Key Features**:
- Advanced caching strategies
- Parallel processing
- Performance optimization
- Comprehensive monitoring
- Memory efficiency

## 📊 Monitoring Examples

### 11. Monitoring System Example

**Purpose**: Comprehensive monitoring and visualization system demonstration.

**Input**: Sample social network metrics  
**Process**: Real-time monitoring → Alert system → Multi-backend logging → Visualization  
**Output**: Comprehensive monitoring reports, interactive dashboards, trend analysis

```bash
# Run monitoring example
python demo/monitoring_example.py
```

**Key Features**:
- Real-time metrics tracking
- Multi-backend support (WanDB, TensorBoard, file logging)
- Alert system
- Advanced visualization
- Trend analysis

## 🔗 Integration Examples

### 12. Async Architecture Demo

**Purpose**: Demonstrates the asynchronous architecture components described in SandGraph_Archi.md.

**Input**: Agent configurations, LLM policies, slot management settings  
**Process**: Async LLM calls → Parallel agent processing → Reward-based resource allocation → Distributed workflow execution  
**Output**: Parallel processing results, resource utilization metrics, async workflow performance

```bash
# Run all async architecture demos
python demo/async_architecture_demo.py --demo all

# Run specific components
python demo/async_architecture_demo.py --demo vllm
python demo/async_architecture_demo.py --demo slot
python demo/async_architecture_demo.py --demo sandbox
python demo/async_architecture_demo.py --demo workflow
python demo/async_architecture_demo.py --demo simulation
python demo/async_architecture_demo.py --demo parallel
```

**Key Features**:
- **Async LLM Client**: Non-blocking VLLM calls with retry mechanisms
- **Reward-Based Slot Management**: Dynamic resource allocation based on agent rewards
- **OASIS Sandbox**: Belief-based agent grouping and management
- **Async Agent Workflow**: Parallel inference and weight updates
- **Distributed Architecture**: Multi-agent parallel processing
- **Intelligent Scheduling**: Priority-based task dispatching

**Expected Results**:
- Parallel processing of multiple agents
- Efficient resource utilization
- Real-time performance monitoring
- Scalable architecture for large-scale simulations

### 13. Complete Integration Example

This example shows how to integrate all SandGraphX features:

```python
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.llm_frozen_adaptive import FrozenAdaptiveManager
from sandgraph.core.areal_kv_cache import create_areal_style_trainer
from sandgraph.core.monitoring import SocialNetworkMonitor, MonitoringConfig
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode

# 1. Initialize LLM Manager
llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")

# 2. Initialize Frozen & Adaptive Manager
frozen_adaptive_manager = FrozenAdaptiveManager()

# 3. Initialize AReaL Trainer
areal_trainer = create_areal_style_trainer(
    kv_cache_size=5000,
    max_memory_gb=4.0,
    rollout_batch_size=16,
    enable_streaming=True
)

# 4. Initialize Monitor
monitor = SocialNetworkMonitor(
    MonitoringConfig(
        enable_wandb=True,
        enable_tensorboard=True,
        wandb_project_name="sandgraph-integration"
    )
)

# 5. Create Workflow
workflow = SG_Workflow("integration_workflow", WorkflowMode.TRADITIONAL, llm_manager)

# 6. Add nodes and execute
workflow.add_node(NodeType.SANDBOX, "environment", {"sandbox": MySandbox()})
workflow.add_node(NodeType.LLM, "decision", {"role": "Decision Maker"})
workflow.add_node(NodeType.RL, "optimizer", {"algorithm": "PPO"})

# 7. Execute with monitoring
monitor.start_monitoring()
result = workflow.execute_full_workflow()
monitor.stop_monitoring()
```

## 🎯 Demo Configuration Options

### Common Parameters

Most demos support these common parameters:

```bash
--steps              # Number of simulation steps
--enable-wandb       # Enable WanDB monitoring
--enable-tensorboard # Enable TensorBoard monitoring
--wandb-project      # WanDB project name
--model-name         # LLM model to use
```

### Advanced Parameters

Some demos support advanced parameters:

```bash
--num-users          # Number of users in network
--network-density    # Network connection density
--kv-cache-size      # AReaL KV cache size
--max-memory-gb      # Maximum memory usage
--rollout-batch-size # AReaL rollout batch size
--posts-per-agent    # Posts per agent per step
```

## 📈 Expected Results

### Performance Benchmarks

- **Social Network Demos**: 80-95% engagement rates
- **Misinformation Analysis**: 50-70% spread reduction
- **AReaL Optimization**: 30-50% training speed improvement
- **LLM Frozen & Adaptive**: 20-40% memory efficiency gain

### Monitoring Metrics

- Real-time performance tracking
- Network dynamics visualization
- Agent behavior analysis
- Resource utilization monitoring

## 🚀 Next Steps

1. **Start Simple**: Begin with basic demos to understand the framework
2. **Add Monitoring**: Enable WanDB/TensorBoard for better insights
3. **Explore Advanced Features**: Try LLM frozen & adaptive and AReaL optimization
4. **Customize**: Modify demos for your specific use case
5. **Scale Up**: Increase network size and complexity

## 🆘 Troubleshooting

### Common Demo Issues

1. **Memory Issues**: Reduce `--num-users` or `--kv-cache-size`
2. **Model Loading**: Check internet connection for model downloads
3. **WanDB Issues**: Run `wandb login` before using WanDB features
4. **Performance**: Use smaller models for faster execution

### Getting Help

- Check demo-specific error messages
- Review the [API Reference](api_reference.md)
- Look at demo source code for implementation details
- Open an issue on GitHub for bugs or feature requests 
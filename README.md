# SandGraphX

<div align="center">
  <img src="assets/logo.png" alt="SandGraphX Logo" width="200"/>
</div>

SandGraphX is an intelligent optimization framework based on Environment Subsets abstraction and Optimization Goals. It coordinates LLM decision-making and RL weight updates through SandBox Workflow Graph to achieve automated optimization of complex tasks.

## 🌟 Core Concepts

### 1. Environment Subsets
- Decompose complex environments into manageable subsets
- Each subset is an independent SandBox
- Support custom state spaces and action spaces
- Provide standardized execution and evaluation interfaces

### 2. Optimization Goals
- Define specific optimization objectives for tasks
- Support single-objective or multi-objective optimization
- Support custom scoring functions
- Provide quantitative assessment of goal achievement

### 3. SandBox Workflow Graph
- Decompose tasks into multiple SandBox nodes
- Organize node relationships through Directed Acyclic Graph (DAG)
- Support parallel and sequential execution
- Implement state transfer and result aggregation between nodes

### 4. Intelligent Decision System
- **RL Weight Updates**: Optimize decision strategies
- **State Management**: Track and update system states
- **Isolated Interaction with LLM and Resources**: SandBox as workflow graph nodes are isolated from LLM (Decision Making), RL (LLM Weight Update), and Computational Resources (GPU, CPU, etc.), with SandGraphX globally managing the latter two.

<div align="center">
  <img src="assets/archi.jpeg" alt="SandGraphX Architecture" width="800"/>
</div>

## 🌟 Core Features

- **Sandbox Environment**: Standardized task environments following InternBootCamp patterns
- **Workflow Graph**: Support for Sandbox DAG Workflow
- **Standardized Communication**: Use official MCP protocol for Sandbox communication with LLM for computation
- **Multiple Usage Scenarios**: From single sandbox (single node) execution to complex multi-stage (multiple node, large DAGs) workflows
- **Dynamic Workflow Engine**: Support for complex DAG (Directed Acyclic Graph) workflows, enabling multi-node collaboration
- **Intelligent State Management**: Each node maintains independent states, supporting dynamic updates and state tracking
- **Resource Management System**: Resource (energy, tokens, time, knowledge) management mechanisms
- **Adaptive Decision Making**: Support for intelligent decisions based on historical information and current states
- **Extensible Architecture**: Easy to add new node types and functional modules
- **🔥 Rich LLM Model Support**: Support for various popular large language models, including:
  - **Default Recommendation**: Mistral-7B
  - **🆕 Advanced Recommendation**: Qwen3-14B
  - **Chinese Models**: Qwen-7B, Yi-6B, ChatGLM3
  - **Code Models**: CodeLLaMA, StarCoder
  - **Lightweight**: Phi-2, Gemma-2B
  - **High Performance**: LLaMA2-13B
  - **Open Source Alternatives**: GPT-2, Falcon
- **📊 Advanced Monitoring & Visualization**: Comprehensive real-time monitoring with WanDB and TensorBoard integration
  - **Real-time Metrics Tracking**: Monitor social network metrics in real-time
  - **Multi-backend Support**: WanDB, TensorBoard, file logging, and console output
  - **Alert System**: Configurable alerts for critical thresholds
  - **Advanced Visualization**: Static dashboards, interactive plots, trend analysis, and correlation heatmaps
  - **Comprehensive Metrics**: User, engagement, content, network, community, influence, and performance metrics
- **🔒 LLMs Frozen & Adaptive Update**: Advanced parameter management for large language models
  - **Parameter Freezing**: Freeze specific layers or parameters to maintain stability
  - **Multiple Update Strategies**: FROZEN, ADAPTIVE, SELECTIVE, INCREMENTAL, GRADUAL strategies
  - **Adaptive Learning Rate**: Automatically adjust learning rates based on performance
  - **Parameter Importance Analysis**: Analyze and rank parameters by importance
  - **Performance Monitoring**: Real-time performance tracking and trend analysis
  - **Checkpoint & Rollback**: Save checkpoints and rollback to optimal states
  - **Thread-Safe Operations**: Multi-threaded parameter management with locks
- **🚀 AReaL Deep Integration**: Advanced framework integration based on [AReaL](https://github.com/inclusionAI/AReaL)
  - **Multi-Level Integration**: BASIC, ADVANCED, and FULL integration levels
  - **Advanced Caching System**: Multiple backends with adaptive policies (LRU, LFU, Priority-based)
  - **Distributed Processing**: Task scheduling and node management
  - **Real-time Metrics Collection**: Performance monitoring and data analysis
  - **Adaptive Resource Management**: Intelligent resource allocation and optimization
  - **Fault Tolerance**: Error handling and recovery mechanisms
  - **High-Performance Data Structures**: Optimized for large-scale operations
  - **Asynchronous RL Training**: Decoupled generation and training for improved efficiency
  - **Streaming Generation**: Real-time generation with reward computation
  - **Interruptible Rollout**: KV cache management with task interruption support
  - **Data Staleness Control**: Rollout controller with configurable staleness thresholds
  - **Decoupled PPO Loss**: Stable training with separated policy and value losses
  - **Memory-Efficient KV Cache**: Adaptive cache policies with compression
  - **Multi-threaded Processing**: Parallel rollout execution with worker pools
- **🎯 Comprehensive Misinformation Analysis**: Advanced social network analysis with multi-agent competition
  - **Multi-Agent Competition**: SandGraph LLM vs Rule-Based vs Human Simulation
  - **Real-time Performance Tracking**: Monitor agent performance and network dynamics
  - **Integrated Optimization**: Combine LLM frozen & adaptive update with AReaL KV cache
  - **WanDB Integration**: Comprehensive monitoring and visualization of competition results
  - **Network Dynamics Simulation**: Realistic social network behavior modeling
  - **Belief Impact Analysis**: Track misinformation impact on user beliefs
- **⚡ Reward-Based Slot Management**: Intelligent resource allocation with reward-driven preemption
  - **Dynamic Priority Scheduling**: Adjust task priorities based on reward values
  - **Smart Preemption**: High-reward tasks can preempt low-reward tasks
  - **Resource-Aware Allocation**: Real-time monitoring of CPU, memory, and GPU usage
  - **Adaptive Frozen Integration**: Deep integration with adaptive frozen LLM management
  - **Performance Optimization**: Dynamic slot allocation based on model performance
  - **Fair Scheduling**: Balance high-value tasks with system fairness
  - **Real-time Monitoring**: Track slot execution states and performance metrics
  - **Resource Limits**: Configurable resource limits and usage thresholds
- **🧬 Self-Evolving Oasis System**: Advanced social network simulation with self-evolving LLM capabilities
  - **LoRA Model Compression**: Reduce model parameters using LoRA technology for efficient multi-model deployment
  - **KV Cache Compression**: Compress attention mechanism key-value cache to improve inference efficiency
  - **Online Model Adaptation**: Dynamically adjust model parameters based on social network dynamics
  - **Self-Evolution Learning**: Models continuously optimize and evolve during runtime
  - **Multi-Model Collaboration**: Different models handle different types of tasks
  - **Evolution Strategies**: Support multiple evolution strategies (multi-model collaboration, adaptive compression, gradient-based, meta-learning)
  - **Task Distribution**: Intelligent task assignment to specialized models
  - **Performance Monitoring**: Real-time monitoring of evolution effects and performance metrics
  - **State Persistence**: Save and load evolution states for continuous learning
  - **Resource Optimization**: Adaptive resource allocation based on model performance
- **🎯 Oasis Task Framework**: Comprehensive task framework for social network research
  - **Information Propagation Tasks**: Content generation, propagation path analysis, influence assessment, spread velocity prediction
  - **Competition Analysis Tasks**: Group behavior analysis, competition strategy optimization, adversarial behavior detection, equilibrium calculation
  - **Misinformation Management Tasks**: Real-time misinformation detection, spread blocking strategies, truth propagation promotion, impact scope assessment
  - **Network Optimization Tasks**: Connection optimization, community detection, stability maintenance, performance monitoring
  - **Scenario-Driven Design**: Focus on key scenarios like misinformation spread and group competition
  - **Intelligent Task Scheduling**: Automatic task sequence selection based on scenario type
  - **Performance Monitoring**: Real-time task performance tracking and evolution triggering
  - **Multi-Model Task Assignment**: Specialized models for different task types
  - **Evolutionary Optimization**: Continuous improvement of task execution strategies
- **🐦 Twitter Misinformation Simulation**: Advanced social network misinformation spread simulation with OASIS integration
  - **OASIS Core Integration**: Direct integration with OASIS agent graph and social agents
  - **SandGraph Core Integration**: Full utilization of SandGraph Core components (LLM, LoRA, RL, Slot Management)
  - **Enhanced Belief Propagation**: Complex belief spread mechanisms with dynamic strength and influence scores
  - **Multi-Mode LLM Decision**: Support for frozen/adaptive/lora modes with asynchronous processing
  - **Real-time Monitoring**: Comprehensive metrics tracking and visualization
  - **Intervention Analysis**: Study intervention strategies and their effectiveness
  - **Network Dynamics**: Realistic social network behavior modeling with polarization analysis

## 📁 File Structure

```
SandGraphX/
├── sandgraph/                    # Core package directory
│   ├── core/                     # Core functional modules
│   │   ├── workflow.py          # Basic workflow implementation
│   │   ├── sg_workflow.py       # SandGraph workflow implementation
│   │   ├── dag_manager.py       # DAG graph management
│   │   ├── llm_interface.py     # LLM interface
│   │   ├── llm_frozen_adaptive.py # LLMs frozen & adaptive update
│   │   ├── reward_based_slot_manager.py # Reward-based slot management
│   │   ├── enhanced_rl_algorithms.py # Enhanced RL algorithms (Areal integration)
│   │   ├── areal_kv_cache.py      # AReaL-style KV cache optimization
│   │   ├── areal_integration.py   # AReaL deep integration framework
│   │   ├── lora_compression.py    # LoRA compression for model parameters and KV cache
│   │   ├── self_evolving_oasis.py # Self-evolving Oasis system with LoRA and multi-model collaboration
│   │   ├── sandbox.py           # Sandbox base class
│   │   ├── rl_framework.py      # Reinforcement learning framework
│   │   ├── rl_algorithms.py     # Reinforcement learning algorithms
│   │   ├── monitoring.py        # Social network monitoring system
│   │   └── visualization.py     # Data visualization module
│   ├── sandbox_implementations.py # Sandbox implementations
│   └── examples.py              # Example code
├── demo/                        # Example code directory
│   ├── trading_demo.py         # Trading system example
│   ├── social_network_demo.py  # Social network analysis demo
│   ├── misinformation_spread_demo.py # Misinformation spread demo
│   ├── comprehensive_misinformation_demo.py # Comprehensive misinformation analysis with multi-agent competition
│   ├── oasis_social_demo.py    # OASIS social network simulation
│   ├── enhanced_social_network_demo.py # Enhanced demo with monitoring
│   ├── enhanced_oasis_social_demo.py   # Enhanced OASIS demo with monitoring
│   ├── enhanced_rl_cache_demo.py       # Enhanced RL cache demo
│   ├── areal_kv_cache_demo.py          # AReaL KV cache optimization demo
│   ├── enhanced_areal_integration_demo.py # Enhanced AReaL integration demo
│   ├── monitoring_example.py   # Monitoring system example
│   ├── llm_frozen_adaptive_demo.py # LLMs frozen & adaptive demo (full)
│   ├── llm_frozen_adaptive_simple_demo.py # LLMs frozen & adaptive demo (simple)
│   ├── reward_based_slot_demo.py # Reward-based slot management demo
│   ├── new_rl_algorithms_demo.py # New RL algorithms (SAC, TD3) demo
│   ├── lora_example.py         # LoRA compression example
│   ├── self_evolving_oasis_demo.py # Self-evolving Oasis system demo
│   ├── integrated_oasis_demo.py # Integrated Oasis with self-evolving LLM demo
│   └── oasis_task_implementation.py # Oasis task implementation with scenario-driven design
├── docs/                        # Documentation
│   ├── monitoring_guide.md     # Comprehensive monitoring guide
│   ├── llm_frozen_adaptive_guide.md # LLMs frozen & adaptive guide
│   ├── reward_based_slot_guide.md # Reward-based slot management guide
│   ├── areal_integration_guide.md # AReaL deep integration guide
│   ├── lora_compression_guide.md # LoRA compression guide
│   ├── self_evolving_oasis_guide.md # Self-evolving Oasis system guide
│   ├── oasis_task_definitions.md # Oasis task definitions with scenario-driven API design
│   ├── examples_guide.md       # Complete examples guide
│   ├── quick_start_guide.md    # Quick start guide
│   ├── api_reference.md        # API reference
│   └── LLM_MODELS.md           # LLM models support guide
├── logs/                        # Log files and monitoring data
├── visualizations/              # Generated visualizations
├── test_lora.py                # LoRA compression test script
├── test_self_evolving_oasis.py # Self-evolving Oasis system test script
├── test_oasis_task_implementation.py # Oasis task implementation test script
└── setup.py                     # Installation configuration
```

## 🏗️ System Architecture

```
┌───────────────────────────────────────────────────────┐
│                      SandGraph Core                   │
├─────────────┬─────────────┬─────────────┬─────────────┤
│  Workflow   │   SandBox   │    LLM      │     RL      │
│   Engine    │  Manager    │  Manager    │  Manager    │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │             │
       ▼             ▼             ▼             ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  DAG Nodes  │ │ Environment │ │  Decision   │ │  Weight     │
│             │ │  Subsets    │ │  Making     │ │  Updates    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    SandGraphX Manager                   │
├─────────────────────────────────────────────────────────┤
│  • User Input: Environment subset definitions and optimization goals │
│  • Workflow: DAG graph construction and execution management         │
│  • Optimization: LLM decision optimization and RL weight updates     │
│  • Resources: Global resource management and SandBox isolation       │
│  • Monitoring: Execution state tracking and performance analysis     │
│  • Extension: Support for custom nodes and optimization strategies   │
│  • LLM Management: Frozen & adaptive parameter management            │
│  • AReaL Integration: Advanced caching, metrics, and optimization    │
│  • Slot Management: Reward-based resource allocation and preemption  │
│  • Oasis Tasks: Scenario-driven social network analysis              │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                Monitoring & Visualization               │
├─────────────────────────────────────────────────────────┤
│  • Real-time Metrics: WanDB, TensorBoard, file logging │
│  • Alert System: Configurable thresholds and callbacks  │
│  • Visualization: Dashboards, trends, correlation maps │
│  • Export: JSON, CSV, images, interactive HTML         │
│  • LLM Monitoring: Parameter importance, update history │
│  • AReaL Metrics: Advanced performance and resource monitoring │
│  • Oasis Metrics: Misinformation detection, competition analysis │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create new conda environment
conda create -n sandgraph python=3.11
conda activate sandgraph

# Clone repository
git clone https://github.com/NoakLiu/SandGraphX.git
cd SandGraphX

# Run installation script
chmod +x quick_install.sh
./quick_install.sh
```

### 2. Basic Usage

```python
from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode

# Create LLM manager (default uses Mistral-7B)
llm_manager = create_shared_llm_manager("mistralai/Mistral-7B-Instruct-v0.2")

# Create workflow
workflow = SG_Workflow("my_workflow", WorkflowMode.TRADITIONAL, llm_manager)

# Add nodes and execute
workflow.add_node(NodeType.SANDBOX, "env", {"sandbox": MyEnvironment()})
workflow.add_node(NodeType.LLM, "decision", {"role": "Decision Maker"})
workflow.add_edge("env", "decision")

result = workflow.execute_full_workflow()
```

### 3. Run Your First Demo

```bash
# Run a simple social network demo
python demo/social_network_demo.py --steps 10

# Run with monitoring enabled
python demo/enhanced_social_network_demo.py --steps 20 --enable-wandb

# Run AReaL integration demo
python demo/enhanced_areal_integration_demo.py --demo basic

# Run reward-based slot management demo
python demo/reward_based_slot_demo.py --demo all

# Run Oasis task implementation demo (scenario-driven)
python demo/oasis_task_implementation.py --scenarios all

# Run Oasis task implementation tests
python test_oasis_task_implementation.py

# Run Twitter Misinformation simulation demo
cd demo/twitter_misinfo
python run_simulation.py

# Run Twitter Misinformation integration tests
python test_integration.py
```

### 4. Oasis Task Framework Examples

```python
# Error Information Spread Detection
from demo.oasis_task_implementation import OasisTaskScheduler, OasisScenarioConfig

# Configure misinformation detection scenario
misinformation_scenario = {
    "content": "虚假新闻内容...",
    "source_profile": {"credibility_score": 0.3},
    "propagation_context": {"spread_velocity": "fast"},
    "fact_check_data": {"verified_sources": []}
}

# Execute scenario
scheduler = OasisTaskScheduler(evolving_llm, config)
result = await scheduler.execute_scenario(
    scenario_type="misinformation_spread",
    scenario_data=misinformation_scenario
)

# Group Competition Analysis
competition_scenario = {
    "group_a": {"size": 5000, "influence": 0.7},
    "group_b": {"size": 3000, "influence": 0.6},
    "competition_history": [],
    "network_state": {"total_users": 100000}
}

result = await scheduler.execute_scenario(
    scenario_type="group_competition",
    scenario_data=competition_scenario
)
```

## 📚 Documentation

- **[Quick Start Guide](docs/quick_start_guide.md)** - Get up and running in minutes
- **[Examples Guide](docs/examples_guide.md)** - Complete examples with detailed explanations
- **[API Reference](docs/api_reference.md)** - Comprehensive API documentation
- **[Monitoring Guide](docs/monitoring_guide.md)** - Advanced monitoring and visualization
- **[LLM Frozen & Adaptive Guide](docs/llm_frozen_adaptive_guide.md)** - LLM parameter management
- **[Reward-Based Slot Management Guide](docs/reward_based_slot_guide.md)** - Intelligent resource allocation
- **[AReaL Integration Guide](docs/areal_integration_guide.md)** - Deep integration with AReaL framework
- **[Oasis Task Definitions](docs/oasis_task_definitions.md)** - Scenario-driven task framework for social network research
- **[Twitter Misinformation Simulation Guide](docs/twitter_misinformation_simulation_guide.md)** - Advanced social network misinformation spread simulation with OASIS integration

## 🔥 LLM Model Support

SandGraph supports various mainstream large language models:

| Model Type | Recommended Model | Parameter Size | Memory Requirements |
|------------|------------------|----------------|-------------------|
| **Default Recommendation** | **Mistral-7B** | 7B | 8-16GB |
| **🆕 Advanced Recommendation** | **Qwen3-14B** | 14B | 16-32GB |
| **Chinese Models** | Qwen-7B, Yi-6B, ChatGLM3 | 6-7B | 8-16GB |
| **Code Models** | CodeLLaMA, StarCoder | 7-15B | 8-16GB |
| **Lightweight** | Phi-2, Gemma-2B | 2-3B | 2-4GB |
| **High Performance** | LLaMA2-13B | 13B | 16-32GB |
| **Open Source Alternatives** | GPT-2, Falcon | 1-7B | 2-16GB |

## 🎯 Key Research Scenarios

### 1. Misinformation Spread Analysis
- **Real-time Detection**: Identify false information as it spreads
- **Propagation Tracking**: Analyze how misinformation moves through networks
- **Impact Assessment**: Measure the effects on user beliefs and behaviors
- **Intervention Strategies**: Design effective countermeasures

### 2. Group Competition Dynamics
- **Competition Analysis**: Study how different groups compete for influence
- **Strategy Optimization**: Help groups develop effective strategies
- **Conflict Resolution**: Identify and mitigate potential conflicts
- **Network Stability**: Maintain healthy competition without polarization

### 3. Information Propagation Research
- **Content Generation**: Create content optimized for specific audiences
- **Spread Prediction**: Forecast how information will propagate
- **Influence Mapping**: Identify key influencers and nodes
- **Network Optimization**: Improve information flow efficiency

### 4. Twitter Misinformation Simulation
- **OASIS Integration**: Direct integration with OASIS agent graph and social agents
- **Enhanced Belief Propagation**: Complex belief spread mechanisms with dynamic strength and influence scores
- **Multi-Mode LLM Decision**: Support for frozen/adaptive/lora modes with asynchronous processing
- **Real-time Monitoring**: Comprehensive metrics tracking and visualization
- **Intervention Analysis**: Study intervention strategies and their effectiveness
- **Network Dynamics**: Realistic social network behavior modeling with polarization analysis
- **SandGraph Core Integration**: Full utilization of SandGraph Core components (LLM, LoRA, RL, Slot Management)

## 📄 License

MIT License

## 🤝 Contact

- Email - dong.liu.dl2367@yale.edu 
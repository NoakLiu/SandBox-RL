# Twitter Misinformation Simulation Guide

## Overview

The Twitter Misinformation Simulation is an advanced social network analysis framework that directly integrates OASIS core components and Sandbox-RL Core functionality. This system provides a comprehensive platform for studying misinformation spread, belief propagation, and intervention strategies in social networks.

## Key Features

### 1. OASIS Core Integration
- **Direct OASIS Agent Graph Usage**: Directly calls `generate_twitter_agent_graph` instead of using mock implementations
- **SocialAgent Support**: Uses OASIS SocialAgent class for agent management
- **AgentGraph Integration**: Leverages OASIS AgentGraph for graph structure management
- **Automatic Fallback**: Automatically uses mock implementation when OASIS is unavailable

### 2. Sandbox-RL Core Deep Integration
- **LLM Management**: Uses `create_shared_llm_manager` and `create_frozen_adaptive_llm`
- **LoRA Compression**: Integrates `create_online_lora_manager` for model compression
- **Reinforcement Learning**: Uses `RLTrainer` and `RLConfig` for PPO training
- **Slot Management**: Uses `RewardBasedSlotManager` for resource allocation
- **Monitoring System**: Integrates `MonitoringConfig` for real-time monitoring

### 3. Enhanced Belief Propagation
- **Dynamic Belief Strength**: Each agent has dynamic belief strength (0-1)
- **Influence Scores**: Each agent has influence scores (0-1)
- **Neighbor Influence**: Complex influence mechanisms based on neighbor beliefs
- **Belief Conversion Probability**: Dynamic calculation of belief change probability
- **Interaction History**: Detailed interaction history recording

### 4. Multi-Mode LLM Decision Making
- **Frozen Mode**: Uses only LLM for decision making
- **Adaptive Mode**: Combines RL for weight updates
- **LoRA Mode**: Supports LoRA weight pluggable fine-tuning
- **Async Support**: Supports asynchronous LLM calls

## File Structure

```
demo/twitter_misinfo/
├── run_simulation.py          # Main execution script (OASIS integrated)
├── workflow.py               # Workflow management (Sandbox-RL Core integrated)
├── sandbox.py               # Sandbox environment (enhanced belief propagation)
├── llm_policy.py            # LLM policy (multi-mode support)
├── reward.py                # Reward functions
├── test_integration.py      # Integration test script
└── oasis_core/              # OASIS core components
    ├── agents_generator.py  # Agent generator
    ├── agent.py            # SocialAgent class
    ├── agent_graph.py      # AgentGraph class
    ├── agent_action.py     # Agent actions
    └── agent_environment.py # Agent environment
```

## Quick Start

### 1. Basic Run
```bash
cd demo/twitter_misinfo
python run_simulation.py
```

### 2. Integration Test
```bash
cd demo/twitter_misinfo
python test_integration.py
```

### 3. Custom Configuration
```python
# Modify parameters in run_simulation.py
simulation = OasisTwitterMisinfoSimulation(
    num_agents=50,           # Number of agents
    profile_path="path/to/profile.csv",  # OASIS profile file
    llm_mode='adaptive',     # LLM mode
    enable_monitoring=True,   # Enable monitoring
    enable_slot_management=True  # Enable slot management
)
```

## Configuration Parameters

### Simulation Parameters
- `num_agents`: Total number of agents (default: 50)
- `max_steps`: Number of simulation steps (default: 20)
- `llm_mode`: LLM mode ('frozen'/'adaptive'/'lora')
- `enable_monitoring`: Enable monitoring (default: True)
- `enable_slot_management`: Enable slot management (default: True)

### Belief Distribution Weights
- TRUMP: 35%
- BIDEN: 35%
- NEUTRAL: 20%
- SWING: 10%

### Sandbox-RL Core Configuration
- `model_name`: LLM model name (default: "qwen-2")
- `backend`: Backend type (default: "vllm")
- `url`: LLM service URL (default: "http://localhost:8001/v1")
- `temperature`: Temperature parameter (default: 0.7)

## Core Components

### 1. Enhanced Sandbox (`sandbox.py`)

```python
class TwitterMisinformationSandbox:
    """Enhanced Twitter misinformation spread sandbox"""
    
    def __init__(self, agent_graph, trump_ratio=0.5, seed=42):
        # Initialize Sandbox-RL Core components
        # Support OASIS agent graph
        
    def _initialize_agent_states(self):
        # Initialize extended agent states
        # Include belief strength, influence scores, etc.
        
    def _calculate_belief_change_probability(self, agent_state, action, neighbor_beliefs):
        # Dynamic calculation of belief change probability
        
    def get_polarization_score(self) -> float:
        # Calculate polarization score
        
    def get_influence_spread(self) -> float:
        # Calculate influence spread
```

### 2. Multi-Mode LLM Policy (`llm_policy.py`)

```python
class LLMPolicy:
    """LLM policy supporting multiple modes"""
    
    def __init__(self, mode='frozen', reward_fn=None, ...):
        # Support frozen/adaptive/lora three modes
        
    def _generate_enhanced_prompt(self, agent_id: int, prompt: str, state: Dict[str, Any]) -> str:
        # Generate enhanced prompt with more context
        
    async def decide_async(self, prompts: Dict[int, str], state: Optional[Dict[str, Any]] = None):
        # Async decision support
        
    def _calculate_decision_confidence(self, agent_id: int, prompt: str, state: Dict[str, Any]) -> float:
        # Calculate decision confidence
```

### 3. Enhanced Workflow (`workflow.py`)

```python
class TwitterMisinfoWorkflow:
    """Sandbox-RL Core integrated workflow"""
    
    def __init__(self, agent_graph, reward_fn=trump_dominance_reward, 
                 llm_mode='frozen', enable_monitoring=True, enable_slot_management=True):
        # Initialize Sandbox-RL Core components
        
    def _calculate_belief_polarization(self, beliefs):
        # Calculate belief polarization degree
        
    def _calculate_influence_spread(self, beliefs, actions):
        # Calculate influence spread
        
    def get_simulation_metrics(self) -> List[SimulationMetrics]:
        # Get simulation metrics
        
    def export_metrics(self, filename: str = "simulation_metrics.json"):
        # Export simulation metrics
```

## Belief Types and States

### BeliefType Enum
- **TRUMP**: Trump supporters
- **BIDEN**: Biden supporters
- **NEUTRAL**: Neutral users
- **SWING**: Swing voters

### AgentState Data Class
```python
@dataclass
class AgentState:
    agent_id: int
    belief_type: BeliefType
    belief_strength: float      # 0-1, belief strength
    influence_score: float      # 0-1, influence score
    neighbors: List[int]        # Neighbor list
    posts_history: List[Dict]   # Post history
    interactions_history: List[Dict]  # Interaction history
    last_activity: float        # Last activity time
```

### SimulationMetrics Data Class
```python
@dataclass
class SimulationMetrics:
    step: int                   # Simulation step
    trump_count: int           # Trump supporter count
    biden_count: int           # Biden supporter count
    neutral_count: int         # Neutral user count
    swing_count: int           # Swing voter count
    reward: float              # Reward value
    slot_reward: float         # Slot reward value
    belief_polarization: float # Belief polarization degree
    influence_spread: float    # Influence spread
```

## Output Results

### 1. Console Output Example
```
=== OASIS Twitter Misinformation Simulation ===
Successfully imported OASIS core components
Using OASIS to generate 30 agents...
Initializing Sandbox-RL Core components...
Starting 10-step simulation...
Agent Graph Info: {'total_agents': 30, 'oasis_available': True, 'graph_type': 'OASIS Twitter Agent Graph'}

=== Step 1/10 ===
Processing subgraph: subgraph_trump (10 agents)
[LLM][subgraph_trump] Decision: spread_misinfo - Spread Trump victory information
Processing subgraph: subgraph_biden (9 agents)
[LLM][subgraph_biden] Decision: counter_misinfo - Counter Trump supporters' false information

--- Step 1 Summary ---
  subgraph_trump: TRUMP (10 people, strength:0.75, influence:0.68)
  subgraph_biden: BIDEN (9 people, strength:0.72, influence:0.65)
```

### 2. File Outputs
- `twitter_misinfo_simulation_results.png`: Visualization results
- `simulation_metrics.json`: Detailed simulation metrics
- `oasis_misinfo_simulation_results.json`: OASIS version results

### 3. Final Statistics Example
```json
{
  "total_agents": 30,
  "belief_distribution": {
    "TRUMP": 10,
    "BIDEN": 9,
    "NEUTRAL": 6,
    "SWING": 5
  },
  "final_statistics": {
    "total_steps": 10,
    "final_trump_count": 10,
    "final_biden_count": 9,
    "final_belief_polarization": 0.033,
    "final_influence_spread": 0.65,
    "total_reward": 8.5,
    "total_slot_reward": 3.2
  }
}
```

## Technical Features

### 1. OASIS Deep Integration
- **Direct OASIS Component Usage**: No longer relies on mock implementations
- **Agent Graph Management**: Leverages OASIS graph structure management
- **SocialAgent Support**: Uses OASIS agent classes
- **Automatic Fallback**: Automatically degrades when OASIS is unavailable

### 2. Sandbox-RL Core Comprehensive Integration
- **LLM Management**: Uses `create_shared_llm_manager` and `create_frozen_adaptive_llm`
- **LoRA Compression**: Integrates `create_online_lora_manager`
- **Reinforcement Learning**: Uses `RLTrainer` and `RLConfig`
- **Slot Management**: Uses `RewardBasedSlotManager`
- **Monitoring System**: Integrates `MonitoringConfig`

### 3. Enhanced Belief Propagation Mechanism
- **Dynamic Belief Strength**: Dynamically adjusted based on interactions
- **Influence Spread**: Calculates influence spread in networks
- **Complex Neighbor Influence**: Complex influence mechanisms based on neighbor beliefs
- **Belief Conversion Probability**: Dynamic calculation of belief change probability

### 4. Async Support
- **Async LLM Calls**: Supports async decision making
- **Parallel Processing**: Supports parallel agent decision making
- **Event Loop Management**: Automatic event loop management

## Extension Suggestions

### 1. Add New Belief Types
```python
class BeliefType(Enum):
    TRUMP = "TRUMP"
    BIDEN = "BIDEN"
    NEUTRAL = "NEUTRAL"
    SWING = "SWING"
    # Add new belief types
    INDEPENDENT = "INDEPENDENT"
    LIBERTARIAN = "LIBERTARIAN"
```

### 2. Implement Cross-Subgraph Interaction
```python
def execute_cross_subgraph_action(self, source_subgraph: str, target_subgraph: str, action: str):
    """Execute cross-subgraph action"""
    # Implement subgraph interaction logic
```

### 3. Add Intervention Mechanisms
```python
def apply_intervention(self, intervention_type: str, target_subgraphs: List[str]):
    """Apply intervention measures"""
    # Implement fact-checking, warning labels, etc.
```

## Troubleshooting

### 1. OASIS Import Errors
```
Warning: OASIS core components not available: [error message]
Using mock implementation
```
The system automatically uses mock implementation without affecting basic functionality.

### 2. Sandbox-RL Core Import Errors
```
Warning: Sandbox-RL Core components not available: [error message]
Using basic implementation
```
The system uses basic implementation while preserving core functionality.

### 3. LLM Connection Errors
If vLLM service is not started, the system automatically uses fallback responses:
```
Error generating LLM response: [error message]
Using fallback response
```

### 4. Visualization Errors
If matplotlib is not available:
```
matplotlib not available, skipping visualization
```

## Performance Optimization

### 1. Async Processing
- Supports async LLM calls
- Parallel processing of multiple agent decisions
- Automatic event loop management

### 2. Caching Mechanisms
- LLM response caching
- Subgraph state caching
- Computation result caching

### 3. Memory Management
- Timely cleanup of historical data
- Optimized data structures
- Memory usage control

## Testing

### Run Integration Tests
```bash
cd demo/twitter_misinfo
python test_integration.py
```

Tests check:
- Basic workflow functionality
- OASIS integration status
- Sandbox-RL Core integration status
- Component initialization status

## Summary

This implementation provides a complete Twitter misinformation spread simulation framework that achieves powerful simulation capabilities through deep integration of OASIS core components and Sandbox-RL Core functionality:

1. **OASIS Deep Integration**: Direct use of OASIS agent graph and social agents
2. **Sandbox-RL Core Comprehensive Integration**: Full utilization of all Sandbox-RL Core functionality
3. **Enhanced Belief Propagation Mechanism**: Support for complex belief propagation and influence mechanisms
4. **Multi-Mode LLM Decision Making**: Support for frozen/adaptive/lora three modes
5. **Complete Monitoring System**: Real-time tracking and visualization support
6. **Async Support**: Support for async processing and parallel decision making

This framework provides powerful tools for studying misinformation spread, group behavior analysis, intervention strategy evaluation, and social network dynamics.

## Related Documentation

- **[Twitter Misinformation README](demo/twitter_misinfo/README.md)** - Detailed implementation guide
- **[Twitter Misinformation CHANGELOG](demo/twitter_misinfo/CHANGELOG.md)** - Change log and technical details
- **[Oasis Scripts README](demo/oasis_scripts/README_twitter_misinfo.md)** - OASIS integration guide 
# OASIS + SandGraph Multi-Agent Adversarial Simulation: Technical Report

## Overview

OASIS + SandGraph is a multi-agent simulation platform for social network, information propagation, and adversarial intervention research. It integrates Large Language Models (LLMs), Reinforcement Learning (RL), LoRA weight adaptation, and advanced workflow (DAG/SG_Workflow) to support realistic social network structures, opinion spread, intervention mechanisms, and various optimization goals.

---

## 1. Four Core Components

### 1.1 Environment Subset / Sandbox
**Goal:**
- Abstract subgraphs, communities, or topics in social networks as sandbox environments.
- Maintain node (user/agent) states, neighbor relations, information flow, and beliefs.
- Provide standardized `step`/`execute` interfaces for multi-agent concurrent decision-making and state evolution.

**Implementation:**
- Use Twitter/Reddit agent graph as the base; nodes are users, edges are follow/interact relations.
- Each agent maintains a `group` (e.g., "TRUMP"/"BIDEN"), belief, and history.
- Sandbox class (e.g., `TwitterMisinformationSandbox`) implements `get_prompts()`, `step(actions)`, `get_state()` for decoupling with LLM/RL/intervention modules.
- Supports multiple information types (true, false, misleading, unverified) and intervention types (fact-check, downrank, educate, etc).

**Example:**
```python
class TwitterMisinformationSandbox(Sandbox):
    def __init__(self, agent_graph, trump_ratio=0.5, seed=42): ...
    def get_prompts(self): ...
    def step(self, actions): ...
    def get_state(self): ...
```

---

### 1.2 Optimization Goal
**Goal:**
- Support various optimization goals: maximize a group's share, minimize polarization, slot reward preemption, maximize intervention effect, etc.
- Reward functions are configurable for RL/LLM policy optimization.

**Implementation:**
- Implement multiple reward functions in `reward.py`, e.g., `polarization_reward`, `trump_dominance_reward`, `slot_reward`.
- Reward functions are called in workflow/RLTrainer to drive LLM/RL policy learning.

**Example:**
```python
def polarization_reward(state, action, next_state): ...
def trump_dominance_reward(state, action, next_state): ...
def slot_reward(state, action, next_state): ...
```

---

### 1.3 Workflow Graph / SG_Workflow
**Goal:**
- Use DAG/SG_Workflow to organize multi-stage, multi-agent, multi-strategy complex simulation flows.
- Support node types (sandbox, LLM, RL, intervention, aggregator), conditional branches, parallelism, loops, etc.
- Easy to plug in new nodes and extend features.

**Implementation:**
- Use `SG_Workflow` or custom `TwitterMisinfoWorkflow` to organize the main simulation flow.
- Each node can be a sandbox, LLM decision, RL optimization, intervention, etc., connected by directed edges.
- Supports slot reward, RL weight update, LLM frozen/adaptive/lora switching.

**Example:**
```python
class TwitterMisinfoWorkflow:
    def __init__(self, agent_graph, reward_fn, llm_mode): ...
    def run(self, max_steps=30): ...
```

---

### 1.4 Intelligent Decision System (LLM/RL/LoRA)
**Goal:**
- Support LLM frozen (pretrained only), adaptive (RL fine-tuning), lora (LoRA plug-in fine-tuning) modes.
- Support RLTrainer, slot reward preemption, online weight update, LoRA compression.
- Flexible switching and combination of agent decision strategies.

**Implementation:**
- Use SandGraph core's `llm_frozen_adaptive.py` and `lora_compression.py` for weight management.
- `LLMPolicy` encapsulates all three modes, automatically calling the corresponding LLM/weight update interface.
- RLTrainer can be integrated with LLM/LoRA for RL+LLM joint optimization.

**Example:**
```python
from sandgraph.core.llm_frozen_adaptive import create_frozen_adaptive_llm, UpdateStrategy
from sandgraph.core.lora_compression import create_online_lora_manager
class LLMPolicy:
    def __init__(self, mode, ...):
        if mode == 'frozen':
            self.llm = create_frozen_adaptive_llm(base_llm, strategy=UpdateStrategy.FROZEN)
        elif mode == 'adaptive':
            self.llm = create_frozen_adaptive_llm(base_llm, strategy=UpdateStrategy.ADAPTIVE)
        elif mode == 'lora':
            self.lora_manager = create_online_lora_manager(...)
            self.lora_manager.register_model(...)
            self.lora_manager.load_model_with_lora(...)
            self.llm = base_llm
    def decide(self, prompts, state=None): ...
```

---

## 2. Running Instructions

1. **Prepare Environment**
   - Install OASIS, SandGraph, and dependencies (see README).
   - Start vLLM or Huggingface LLM service.
   - Prepare agent graph data (e.g., user_data.json).

2. **Run Simulation Script**
   - For misinformation spread demo:
     ```bash
     python demo/twitter_misinfo/run_simulation.py
     ```
   - Or in OASIS framework:
     ```bash
     python demo/oasis_scripts/twitter_simulation.py
     ```

3. **Switch LLM Mode**
   - Set `llm_mode='frozen'|'adaptive'|'lora'` in workflow/LLMPolicy initialization.
   - In lora mode, specify lora_path for online weight adaptation.

4. **Visualization and Analysis**
   - Each round outputs Trump/Biden group counts, slot reward, RL reward curves.
   - Supports WanDB, TensorBoard, matplotlib, etc.

---

## 3. Misinformation Group Competition Implementation

1. **Group Initialization**
   - After agent graph generation, randomly assign half as Trump supporters, half as Biden supporters (`agent.group = "TRUMP"/"BIDEN"`).

2. **LLM Decision**
   - Each agent's prompt includes its own group and neighbor groups; LLM decides which message to post/forward/oppose.

3. **Propagation Rule**
   - If agent's decision differs from its group and most neighbors are the other group, it may be persuaded to switch group.
   - Supports slot reward (mainstream opinion change), polarization reward, etc.

4. **Intervention Mechanism**
   - Insert fact-check, downrank, educate, etc. nodes to dynamically adjust information flow and belief evolution.

5. **RL/LoRA Joint Optimization**
   - RLTrainer can use slot reward, polarization reward, etc. to optimize LLM/LoRA weights for adaptive adversarial strategies.

6. **Result Analysis**
   - Track Trump/Biden group counts, mainstream opinion changes, polarization, intervention effects, etc.

---

## 4. Summary

- OASIS + SandGraph enables multi-agent, complex network, LLM+RL+LoRA joint optimization for misinformation adversarial simulation.
- All four core components (sandbox, optimization goal, workflow, intelligent decision) are fully engineered and extensible.
- Supports multiple running modes, reward mechanisms, intervention strategies, and visualization, suitable for research, prototyping, and strategy testing.

---

For detailed code, experiment scripts, or further explanation of any part, please contact the development team or refer to the project README. 
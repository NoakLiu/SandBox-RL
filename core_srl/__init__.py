#!/usr/bin/env python3
"""
Core SRL - Multi-Model Reinforcement Learning Core
==================================================

Advanced multi-model RL framework for training multiple LLMs with cooperative-competitive dynamics.
Emphasizes weight updates, parameter sharing, and distributed training with modern models.

Multi-Model Training Focus:
- Simultaneous training of 4-8 modern LLMs (Qwen3-14B, GPT-4o, Claude-3.5)
- Cooperative-competitive RL with dynamic weight updates
- Real-time parameter synchronization and adaptation
- Distributed LoRA management across GPU clusters

System Architecture:
- VERL Integration: Efficient RL training with vLLM backend
- AReaL Framework: Advanced caching and resource optimization  
- KVCache-Centric: Optimized memory management for multi-model scenarios
- Core Cooperative-Competitive RL: Novel algorithm for multi-agent learning

Key Features:
- LLMs weight updates during training
- Multi-model parameter sharing strategies
- Adaptive LoRA rank adjustment
- Real-time performance monitoring
- Checkpoint management and recovery
"""

# LLM Manager - Modern LLM interfaces
from .llm_manager import (
    LLMBackend,
    LLMConfig,
    LLMResponse,
    BaseLLM,
    HuggingFaceLLM,
    OpenAILLM,
    SharedLLMManager,
    UpdateStrategy,
    ParameterImportance,
    AdaptiveLearningRate,
    create_llm_config,
    create_llm,
    create_shared_llm_manager,
    create_qwen_manager,
    get_available_models
)

# RL Framework - Reinforcement learning algorithms
from .rl_framework import (
    RLAlgorithm,
    CooperationType,
    CompetenceType,
    CooperationFactor,
    CompetenceFactor,
    RLConfig,
    TrajectoryStep,
    ExperienceBuffer,
    RLAlgorithmBase,
    PPOAlgorithm,
    GRPOAlgorithm,
    OnPolicyRLAgent,
    MultiAgentOnPolicyRL,
    RLTrainer,
    CoopCompeteEnv,
    SimplePG,
    OurMethodPolicy,
    run_benchmark,
    create_ppo_trainer,
    create_grpo_trainer,
    create_multi_agent_system
)

# LoRA System - Parameter adaptation and management
from .lora_system import (
    LoRAUpdateStrategy,
    LoRALoadingStatus,
    CompressionType,
    LoRAConfig,
    LoRAVersion,
    LoRAAdapter,
    LoRALayer,
    LoRAManager,
    LoRAHotSwapManager,
    LoRAPublisher,
    LoRARLStrategy,
    DistributedLoRAScheduler,
    create_lora_config,
    create_lora_manager,
    create_hotswap_manager,
    create_distributed_lora_scheduler,
    create_lora_rl_strategy,
    get_lora_presets,
    create_8gpu_lora_configs
)

# Scheduler - Resource management and task orchestration
from .scheduler import (
    ModelRole,
    InteractionType,
    TaskPriority,
    SlotState,
    ModelProfile,
    TaskDefinition,
    SlotInfo,
    InteractionResult,
    ResourceManager,
    SlotManager,
    VLLMClient,
    CapabilityAnalyzer,
    InteractionOrchestrator,
    UnifiedScheduler,
    create_unified_scheduler,
    create_cooperative_scheduler,
    create_competitive_scheduler,
    create_task_definition
)

# Cache Optimizer - KV cache and optimization systems
from .cache_optimizer import (
    CachePolicy,
    RolloutStatus,
    KVCacheConfig,
    RolloutConfig,
    CachedKVState,
    RolloutTask,
    KVCacheManager,
    RolloutController,
    ArealIntegrationManager,
    VERLTrainer,
    DecoupledPPOTrainer,
    AReaLVERLBridge,
    create_kv_cache_manager,
    create_areal_integration,
    create_verl_trainer,
    create_areal_verl_bridge,
    create_decoupled_ppo_trainer,
    run_integrated_training_demo
)

# Monitoring - Metrics collection and visualization
from .monitoring import (
    NodeType,
    EdgeType,
    InteractionType as GraphInteractionType,
    SocialNetworkMetrics,
    GraphNode,
    GraphEdge,
    GraphEvent,
    MonitoringConfig,
    MetricsCollector,
    GraphVisualizer,
    PerformanceMonitor,
    UnifiedMonitor,
    create_monitoring_config,
    create_unified_monitor,
    create_graph_visualizer,
    create_performance_monitor,
    create_social_network_metrics,
    quick_monitoring_demo
)

# Workflow Engine - DAG execution and training pipelines
from .workflow_engine import (
    NodeType as WorkflowNodeType,
    ExecutionStatus,
    StopConditionType,
    TrajectoryStep as WorkflowTrajectoryStep,
    Trajectory,
    Sample,
    Result,
    ExecutionContext,
    StopCondition,
    WorkflowNode,
    TrainerServer,
    AgentAdapter,
    LocalAgentClient,
    WorkflowEngine,
    WorkflowBuilder,
    DAGReplayBuffer,
    RLEngine,
    create_workflow_engine,
    create_trainer_server,
    create_agent_client,
    create_dag_replay_buffer,
    create_rl_engine,
    write_trajectories_to_jsonl,
    read_trajectories_from_jsonl,
    create_simple_rl_workflow
)

# Environments - Multi-model training environments
from .environments import (
    EnvironmentType,
    Action,
    EnvironmentConfig,
    MazeConfig,
    BaseEnvironment,
    CooperativeCompetitiveEnv,
    MultiAgentStagedEnv,
    TeamBattleEnv,
    MazeEnv,
    SocialInteractionEnv,
    BanditEnv,
    EnvironmentFactory,
    MultiModelTrainingEnvironment,
    SandboxProtocol,
    Sandbox,
    create_multi_model_coop_compete_env,
    create_multi_model_team_battle,
    create_multi_model_staged_env,
    create_maze_training_env,
    create_social_training_env
)

# Multi-Model Trainer - Core training system
from .multimodel_trainer import (
    TrainingMode,
    WeightUpdateStrategy,
    MultiModelConfig,
    ModelState,
    TrainingCheckpoint,
    MultiModelTrainer,
    create_multimodel_trainer,
    create_cooperative_multimodel_trainer,
    create_competitive_multimodel_trainer,
    quick_start_multimodel_training,
    list_available_checkpoints,
    load_checkpoint_metadata
)

__version__ = "2.0.0"
__author__ = "SRL Team"

# Main exports for multi-model training
__all__ = [
    # CORE MULTI-MODEL TRAINING
    "MultiModelTrainer",
    "create_multimodel_trainer", 
    "create_cooperative_multimodel_trainer",
    "create_competitive_multimodel_trainer",
    "quick_start_multimodel_training",
    
    # SYSTEM COMPONENTS
    "SharedLLMManager",
    "RLTrainer", 
    "LoRAManager",
    "UnifiedScheduler",
    "MultiModelTrainingEnvironment",
    "UnifiedMonitor",
    "WorkflowEngine",
    
    # CONFIGURATION
    "MultiModelConfig",
    "LLMConfig",
    "RLConfig", 
    "LoRAConfig",
    "EnvironmentConfig",
    "TrainingMode",
    "WeightUpdateStrategy",
    
    # MODERN LLM MANAGERS
    "create_qwen3_manager",
    "create_qwen_coder_manager", 
    "create_qwen_math_manager",
    "create_openai_manager",
    "create_claude_manager",
    "create_llama3_manager",
    
    # TRAINING ENVIRONMENTS
    "create_multi_model_coop_compete_env",
    "create_multi_model_team_battle",
    "create_multi_model_staged_env",
    
    # CHECKPOINT MANAGEMENT
    "TrainingCheckpoint",
    "list_available_checkpoints",
    "load_checkpoint_metadata",
    
    # BENCHMARKING & EVALUATION
    "run_benchmark",
    "CoopCompeteEnv",
    "SimplePG", 
    "OurMethodPolicy",
    
    # CORE ENUMS
    "LLMBackend",
    "RLAlgorithm",
    "EnvironmentType",
    "ModelRole"
]

def get_version_info() -> Dict[str, str]:
    """Get version and component information"""
    return {
        "version": __version__,
        "components": {
            "llm_manager": "Modern LLM interfaces with adaptive updates",
            "rl_framework": "Multi-agent RL with cooperation/competition",
            "lora_system": "Efficient parameter adaptation",
            "scheduler": "Resource management and orchestration", 
            "cache_optimizer": "KV cache and VERL/AReaL integration",
            "monitoring": "Real-time metrics and visualization",
            "workflow_engine": "DAG execution pipelines",
            "environments": "Multi-model training scenarios"
        },
        "focus": "Multi-model reinforcement learning training"
    }


def quick_start_multi_model_training(num_models: int = 4, model_type: str = "qwen3") -> Dict[str, Any]:
    """Quick start setup for multi-model training"""
    
    # Create modern LLM manager
    if model_type == "qwen3":
        llm_manager = create_qwen_manager("Qwen/Qwen2.5-14B-Instruct")
    else:
        llm_manager = create_shared_llm_manager()
    
    # Create multi-agent RL system
    rl_system = create_multi_agent_system(num_models, enable_cooperation=True)
    
    # Create training environment
    env = create_multi_model_coop_compete_env(num_models)
    
    # Create scheduler
    scheduler = create_unified_scheduler(num_gpus=num_models)
    
    # Register models with scheduler
    for i in range(num_models):
        scheduler.register_model(f"model_{i}", i, ModelRole.GENERALIST)
    
    # Create monitoring
    monitor = create_unified_monitor()
    
    return {
        "llm_manager": llm_manager,
        "rl_system": rl_system,
        "environment": env,
        "scheduler": scheduler,
        "monitor": monitor,
        "setup": "multi_model_training_ready"
    }


# Print initialization message
logger.info(f"Core SRL v{__version__} initialized - Multi-model RL training ready")

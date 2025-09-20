#!/usr/bin/env python3
"""
RL Scheduler Example
===================

Example showing advanced scheduling and resource management for multi-model
reinforcement learning training with cooperative-competitive dynamics.
"""

import asyncio
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from core_srl import MultiModelTrainer, MultiModelConfig, TrainingMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRole(Enum):
    """Model roles in the training system"""
    GENERALIST = "generalist"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    COMPETITOR = "competitor"
    COLLABORATOR = "collaborator"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


class ResourceType(Enum):
    """Types of resources to manage"""
    GPU_MEMORY = "gpu_memory"
    CPU_CORES = "cpu_cores"
    NETWORK_BANDWIDTH = "network_bandwidth"
    STORAGE_IO = "storage_io"


@dataclass
class ModelProfile:
    """Profile for a model in the training system"""
    model_id: str
    gpu_id: int
    port: int
    role: ModelRole
    capabilities: Dict[str, float]
    performance_history: List[float] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    is_healthy: bool = True
    last_health_check: float = 0.0
    cooperation_score: float = 0.5
    competition_score: float = 0.5


@dataclass
class TaskDefinition:
    """Definition of a training task"""
    task_id: str
    task_type: str
    complexity: float
    required_capabilities: List[str]
    priority: TaskPriority = TaskPriority.MEDIUM
    collaboration_required: bool = False
    competition_allowed: bool = True
    estimated_duration: float = 1.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Resource allocation for a model"""
    model_id: str
    gpu_memory: float
    cpu_cores: float
    network_bandwidth: float
    storage_io: float
    allocation_time: float = field(default_factory=time.time)


class RLScheduler:
    """Advanced RL Scheduler for multi-model training"""
    
    def __init__(self, max_models: int = 8, total_gpu_memory: float = 32.0):
        self.max_models = max_models
        self.total_gpu_memory = total_gpu_memory
        
        # Model management
        self.model_profiles: Dict[str, ModelProfile] = {}
        self.active_tasks: Dict[str, TaskDefinition] = {}
        self.completed_tasks: Dict[str, TaskDefinition] = {}
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        
        # Scheduling queues
        self.task_queues = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.MEDIUM: deque(),
            TaskPriority.LOW: deque(),
            TaskPriority.BACKGROUND: deque()
        }
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.cooperation_history = defaultdict(list)
        self.competition_results = defaultdict(list)
        
        # Synchronization
        self.lock = threading.Lock()
        self.scheduler_running = False
    
    def register_model(self, model_profile: ModelProfile):
        """Register a model with the scheduler"""
        with self.lock:
            self.model_profiles[model_profile.model_id] = model_profile
            
            # Initialize resource allocation
            base_memory = self.total_gpu_memory / self.max_models
            self.resource_allocations[model_profile.model_id] = ResourceAllocation(
                model_id=model_profile.model_id,
                gpu_memory=base_memory,
                cpu_cores=2.0,  # Base allocation
                network_bandwidth=100.0,  # Mbps
                storage_io=50.0  # MB/s
            )
            
            logger.info(f"Registered model {model_profile.model_id} with role {model_profile.role.value}")
    
    def submit_task(self, task: TaskDefinition) -> str:
        """Submit a task to the scheduler"""
        with self.lock:
            # Add to appropriate priority queue
            self.task_queues[task.priority].append(task)
            
            logger.info(f"Submitted task {task.task_id} with priority {task.priority.value}")
            return task.task_id
    
    def assign_task_to_model(self, task: TaskDefinition) -> Optional[str]:
        """Assign a task to the best available model"""
        best_model_id = None
        best_score = -1.0
        
        for model_id, profile in self.model_profiles.items():
            if not profile.is_healthy:
                continue
            
            # Calculate suitability score
            score = self._calculate_task_model_score(task, profile)
            
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        if best_model_id:
            self.active_tasks[task.task_id] = task
            logger.info(f"Assigned task {task.task_id} to model {best_model_id} (score: {best_score:.3f})")
        
        return best_model_id
    
    def _calculate_task_model_score(self, task: TaskDefinition, profile: ModelProfile) -> float:
        """Calculate how suitable a model is for a task"""
        score = 0.0
        
        # Check capability match
        for capability in task.required_capabilities:
            if capability in profile.capabilities:
                score += profile.capabilities[capability]
            else:
                score -= 0.5  # Penalty for missing capability
        
        # Performance history bonus
        if profile.performance_history:
            avg_performance = sum(profile.performance_history[-10:]) / len(profile.performance_history[-10:])
            score += avg_performance * 0.3
        
        # Role-based scoring
        if task.collaboration_required and profile.role in [ModelRole.COLLABORATOR, ModelRole.COORDINATOR]:
            score += 0.5
        elif task.competition_allowed and profile.role == ModelRole.COMPETITOR:
            score += 0.3
        
        # Resource availability
        allocation = self.resource_allocations.get(profile.model_id)
        if allocation:
            resource_score = min(
                allocation.gpu_memory / task.resource_requirements.get('gpu_memory', 1.0),
                allocation.cpu_cores / task.resource_requirements.get('cpu_cores', 1.0)
            )
            score += resource_score * 0.2
        
        return max(0.0, score)
    
    async def run_scheduler(self):
        """Main scheduler loop"""
        self.scheduler_running = True
        logger.info("Starting RL scheduler...")
        
        while self.scheduler_running:
            try:
                # Process tasks from priority queues
                await self._process_task_queues()
                
                # Update model health and performance
                await self._update_model_health()
                
                # Rebalance resources if needed
                await self._rebalance_resources()
                
                # Update cooperation/competition dynamics
                await self._update_dynamics()
                
                # Sleep between scheduling cycles
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5.0)
        
        logger.info("RL scheduler stopped")
    
    async def _process_task_queues(self):
        """Process tasks from priority queues"""
        with self.lock:
            # Process queues in priority order
            for priority in TaskPriority:
                queue = self.task_queues[priority]
                
                while queue and len(self.active_tasks) < self.max_models:
                    task = queue.popleft()
                    
                    # Try to assign task
                    assigned_model = self.assign_task_to_model(task)
                    if assigned_model:
                        # Simulate task execution
                        asyncio.create_task(self._execute_task(task, assigned_model))
                    else:
                        # Put back in queue if no model available
                        queue.appendleft(task)
                        break
    
    async def _execute_task(self, task: TaskDefinition, model_id: str):
        """Execute a task on a specific model"""
        start_time = time.time()
        
        try:
            # Simulate task execution time
            execution_time = task.estimated_duration + (hash(task.task_id) % 100) / 100.0
            await asyncio.sleep(execution_time)
            
            # Update model performance
            performance_score = 0.5 + (hash(f"{task.task_id}_{model_id}") % 100) / 200.0
            
            with self.lock:
                profile = self.model_profiles[model_id]
                profile.performance_history.append(performance_score)
                
                # Keep only recent history
                if len(profile.performance_history) > 100:
                    profile.performance_history = profile.performance_history[-100:]
                
                # Update cooperation/competition scores
                if task.collaboration_required:
                    profile.cooperation_score = 0.9 * profile.cooperation_score + 0.1 * performance_score
                if task.competition_allowed:
                    profile.competition_score = 0.9 * profile.competition_score + 0.1 * performance_score
                
                # Move task to completed
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
                
                # Record metrics
                self.performance_metrics[model_id].append({
                    'task_id': task.task_id,
                    'performance': performance_score,
                    'execution_time': execution_time,
                    'timestamp': time.time()
                })
            
            logger.info(f"Completed task {task.task_id} on {model_id}: score={performance_score:.3f}")
            
        except Exception as e:
            logger.error(f"Task execution failed for {task.task_id}: {e}")
            
            with self.lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    async def _update_model_health(self):
        """Update model health status"""
        current_time = time.time()
        
        with self.lock:
            for model_id, profile in self.model_profiles.items():
                # Simple health check based on recent performance
                if profile.performance_history:
                    recent_performance = profile.performance_history[-5:]
                    avg_performance = sum(recent_performance) / len(recent_performance)
                    
                    # Mark unhealthy if performance is consistently low
                    profile.is_healthy = avg_performance > 0.2
                    
                    if not profile.is_healthy:
                        logger.warning(f"Model {model_id} marked as unhealthy (avg_perf={avg_performance:.3f})")
                
                profile.last_health_check = current_time
    
    async def _rebalance_resources(self):
        """Rebalance resources based on model performance"""
        with self.lock:
            # Calculate performance-based resource weights
            total_performance = 0.0
            performance_scores = {}
            
            for model_id, profile in self.model_profiles.items():
                if profile.performance_history and profile.is_healthy:
                    avg_performance = sum(profile.performance_history[-10:]) / len(profile.performance_history[-10:])
                    performance_scores[model_id] = avg_performance
                    total_performance += avg_performance
                else:
                    performance_scores[model_id] = 0.1  # Minimum allocation
                    total_performance += 0.1
            
            # Reallocate GPU memory based on performance
            if total_performance > 0:
                for model_id, allocation in self.resource_allocations.items():
                    performance_ratio = performance_scores[model_id] / total_performance
                    new_memory = self.total_gpu_memory * performance_ratio
                    
                    # Smooth allocation changes
                    allocation.gpu_memory = 0.9 * allocation.gpu_memory + 0.1 * new_memory
                    
                    logger.debug(f"Rebalanced {model_id}: GPU memory = {allocation.gpu_memory:.2f}GB")
    
    async def _update_dynamics(self):
        """Update cooperation/competition dynamics"""
        with self.lock:
            # Calculate global cooperation and competition levels
            total_cooperation = sum(p.cooperation_score for p in self.model_profiles.values())
            total_competition = sum(p.competition_score for p in self.model_profiles.values())
            
            avg_cooperation = total_cooperation / len(self.model_profiles)
            avg_competition = total_competition / len(self.model_profiles)
            
            # Record dynamics history
            self.cooperation_history['global'].append({
                'timestamp': time.time(),
                'avg_cooperation': avg_cooperation,
                'avg_competition': avg_competition
            })
            
            # Adjust task generation based on dynamics
            if avg_cooperation > 0.7:
                # Generate more collaborative tasks
                await self._generate_collaborative_tasks()
            elif avg_competition > 0.7:
                # Generate more competitive tasks
                await self._generate_competitive_tasks()
    
    async def _generate_collaborative_tasks(self):
        """Generate collaborative tasks to encourage cooperation"""
        task = TaskDefinition(
            task_id=f"collab_task_{int(time.time())}",
            task_type="collaborative_training",
            complexity=0.6,
            required_capabilities=["cooperation", "knowledge_sharing"],
            priority=TaskPriority.MEDIUM,
            collaboration_required=True,
            competition_allowed=False,
            estimated_duration=2.0,
            resource_requirements={"gpu_memory": 2.0, "cpu_cores": 1.0}
        )
        
        self.submit_task(task)
    
    async def _generate_competitive_tasks(self):
        """Generate competitive tasks to encourage competition"""
        task = TaskDefinition(
            task_id=f"comp_task_{int(time.time())}",
            task_type="competitive_training",
            complexity=0.8,
            required_capabilities=["optimization", "performance"],
            priority=TaskPriority.HIGH,
            collaboration_required=False,
            competition_allowed=True,
            estimated_duration=1.5,
            resource_requirements={"gpu_memory": 3.0, "cpu_cores": 2.0}
        )
        
        self.submit_task(task)
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.scheduler_running = False
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduler statistics"""
        with self.lock:
            return {
                "registered_models": len(self.model_profiles),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "queue_sizes": {
                    priority.value: len(queue) 
                    for priority, queue in self.task_queues.items()
                },
                "model_health": {
                    model_id: profile.is_healthy 
                    for model_id, profile in self.model_profiles.items()
                },
                "resource_allocations": {
                    model_id: {
                        "gpu_memory": alloc.gpu_memory,
                        "cpu_cores": alloc.cpu_cores
                    }
                    for model_id, alloc in self.resource_allocations.items()
                },
                "performance_summary": {
                    model_id: {
                        "avg_performance": sum(profile.performance_history[-10:]) / len(profile.performance_history[-10:]) if profile.performance_history else 0.0,
                        "cooperation_score": profile.cooperation_score,
                        "competition_score": profile.competition_score
                    }
                    for model_id, profile in self.model_profiles.items()
                }
            }


async def basic_scheduler_example():
    """Basic RL scheduler example"""
    print("Basic RL Scheduler Example")
    print("=" * 30)
    
    # Create scheduler
    scheduler = RLScheduler(max_models=4, total_gpu_memory=16.0)
    
    # Register models with different roles
    models = [
        ModelProfile(
            model_id="model_0",
            gpu_id=0,
            port=8001,
            role=ModelRole.GENERALIST,
            capabilities={"general": 0.8, "cooperation": 0.7}
        ),
        ModelProfile(
            model_id="model_1",
            gpu_id=1,
            port=8002,
            role=ModelRole.COLLABORATOR,
            capabilities={"cooperation": 0.9, "knowledge_sharing": 0.8}
        ),
        ModelProfile(
            model_id="model_2",
            gpu_id=2,
            port=8003,
            role=ModelRole.COMPETITOR,
            capabilities={"optimization": 0.9, "performance": 0.8}
        ),
        ModelProfile(
            model_id="model_3",
            gpu_id=3,
            port=8004,
            role=ModelRole.SPECIALIST,
            capabilities={"specialized": 0.95, "accuracy": 0.9}
        )
    ]
    
    # Register models
    for model in models:
        scheduler.register_model(model)
    
    # Submit various tasks
    tasks = [
        TaskDefinition(
            task_id="task_1",
            task_type="general_training",
            complexity=0.5,
            required_capabilities=["general"],
            priority=TaskPriority.HIGH
        ),
        TaskDefinition(
            task_id="task_2",
            task_type="collaborative_task",
            complexity=0.7,
            required_capabilities=["cooperation"],
            priority=TaskPriority.MEDIUM,
            collaboration_required=True
        ),
        TaskDefinition(
            task_id="task_3",
            task_type="competitive_task",
            complexity=0.8,
            required_capabilities=["optimization"],
            priority=TaskPriority.HIGH,
            competition_allowed=True
        )
    ]
    
    for task in tasks:
        scheduler.submit_task(task)
    
    # Run scheduler for a short time
    scheduler_task = asyncio.create_task(scheduler.run_scheduler())
    
    # Let it run for 10 seconds
    await asyncio.sleep(10)
    
    # Stop scheduler
    scheduler.stop_scheduler()
    await scheduler_task
    
    # Show final statistics
    stats = scheduler.get_scheduler_stats()
    print("\nScheduler Statistics:")
    print(f"  Completed tasks: {stats['completed_tasks']}")
    print(f"  Model health: {stats['model_health']}")
    print("  Performance summary:")
    for model_id, perf in stats['performance_summary'].items():
        print(f"    {model_id}: avg_perf={perf['avg_performance']:.3f}, "
              f"coop={perf['cooperation_score']:.3f}, comp={perf['competition_score']:.3f}")


async def integrated_scheduler_training():
    """Example integrating scheduler with multi-model training"""
    print("\nIntegrated Scheduler + Multi-Model Training")
    print("=" * 45)
    
    # Create scheduler
    scheduler = RLScheduler(max_models=3, total_gpu_memory=12.0)
    
    # Register models
    for i in range(3):
        model = ModelProfile(
            model_id=f"model_{i}",
            gpu_id=i,
            port=8001 + i,
            role=ModelRole.GENERALIST if i == 0 else ModelRole.COLLABORATOR if i == 1 else ModelRole.COMPETITOR,
            capabilities={"general": 0.8, "cooperation": 0.6 + i * 0.1, "competition": 0.4 + i * 0.2}
        )
        scheduler.register_model(model)
    
    # Start scheduler
    scheduler_task = asyncio.create_task(scheduler.run_scheduler())
    
    # Create multi-model trainer
    config = MultiModelConfig(
        num_models=3,
        training_mode=TrainingMode.MIXED,
        max_episodes=50,
        cooperation_strength=0.6,
        competition_intensity=0.4
    )
    
    trainer = MultiModelTrainer(config)
    
    try:
        # Simulate training episodes with scheduler integration
        for episode in range(20):
            # Submit training task to scheduler
            task = TaskDefinition(
                task_id=f"training_ep_{episode}",
                task_type="multi_model_training",
                complexity=0.6,
                required_capabilities=["general", "cooperation"],
                priority=TaskPriority.HIGH,
                collaboration_required=episode % 3 == 0,  # Some episodes require collaboration
                competition_allowed=episode % 2 == 0,     # Some episodes allow competition
                estimated_duration=1.0
            )
            
            scheduler.submit_task(task)
            
            # Small delay between episodes
            await asyncio.sleep(0.5)
        
        # Let scheduler process tasks
        await asyncio.sleep(15)
        
        # Get final statistics
        stats = scheduler.get_scheduler_stats()
        
        print("Integrated Training Results:")
        print(f"  Episodes scheduled: 20")
        print(f"  Tasks completed: {stats['completed_tasks']}")
        print("  Final model performance:")
        
        for model_id, perf in stats['performance_summary'].items():
            print(f"    {model_id}: {perf['avg_performance']:.3f} avg performance")
        
        # Show resource allocation
        print("  Resource allocations:")
        for model_id, resources in stats['resource_allocations'].items():
            print(f"    {model_id}: {resources['gpu_memory']:.2f}GB GPU memory")
    
    finally:
        scheduler.stop_scheduler()
        await scheduler_task
        await trainer.shutdown()


async def main():
    """Main example function"""
    print("RL Scheduler Examples")
    print("=" * 40)
    
    # Run basic scheduler example
    await basic_scheduler_example()
    
    # Run integrated training example
    await integrated_scheduler_training()
    
    print("\nRL scheduler examples completed!")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Async Architecture Example
==========================

Example showing asynchronous architecture for multi-model training
with concurrent task processing and workflow management.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks in the system"""
    MODEL_TRAINING = "model_training"
    DATA_PROCESSING = "data_processing"
    EVALUATION = "evaluation"
    CHECKPOINT_SAVE = "checkpoint_save"
    MONITORING = "monitoring"


@dataclass
class AsyncTask:
    """Asynchronous task definition"""
    task_id: str
    task_type: TaskType
    model_id: str
    payload: Dict[str, Any]
    priority: int = 1
    max_retries: int = 3
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0


class AsyncTaskQueue:
    """Asynchronous task queue with priority handling"""
    
    def __init__(self, max_concurrent_tasks: int = 4):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.pending_tasks: deque = deque()
        self.running_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: Dict[str, AsyncTask] = {}
        self.failed_tasks: Dict[str, AsyncTask] = {}
        self.task_handlers: Dict[TaskType, Callable] = {}
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
    
    def register_handler(self, task_type: TaskType, handler: Callable):
        """Register a handler for a specific task type"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for {task_type.value}")
    
    async def submit_task(self, task: AsyncTask) -> str:
        """Submit a task to the queue"""
        # Insert task based on priority (higher priority first)
        inserted = False
        for i, existing_task in enumerate(self.pending_tasks):
            if task.priority > existing_task.priority:
                self.pending_tasks.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self.pending_tasks.append(task)
        
        logger.info(f"Submitted task {task.task_id} (type: {task.task_type.value}, priority: {task.priority})")
        return task.task_id
    
    async def start_workers(self):
        """Start worker tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_tasks = []
        
        for i in range(self.max_concurrent_tasks):
            worker_task = asyncio.create_task(self._worker(f"worker_{i}"))
            self.worker_tasks.append(worker_task)
        
        logger.info(f"Started {self.max_concurrent_tasks} worker tasks")
    
    async def stop_workers(self):
        """Stop all worker tasks"""
        self.is_running = False
        
        # Cancel all worker tasks
        for worker_task in self.worker_tasks:
            worker_task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Stopped all worker tasks")
    
    async def _worker(self, worker_id: str):
        """Worker task that processes tasks from the queue"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get next task
                task = await self._get_next_task()
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process task
                await self._process_task(task, worker_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[AsyncTask]:
        """Get the next task from the queue"""
        if not self.pending_tasks:
            return None
        
        task = self.pending_tasks.popleft()
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self.running_tasks[task.task_id] = task
        
        return task
    
    async def _process_task(self, task: AsyncTask, worker_id: str):
        """Process a single task"""
        logger.info(f"Worker {worker_id} processing task {task.task_id}")
        
        try:
            # Get handler for task type
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler registered for task type {task.task_type.value}")
            
            # Execute task with timeout
            result = await asyncio.wait_for(
                handler(task.payload),
                timeout=task.timeout
            )
            
            # Mark task as completed
            task.status = TaskStatus.SUCCESS
            task.result = result
            task.completed_at = time.time()
            
            # Move to completed tasks
            del self.running_tasks[task.task_id]
            self.completed_tasks[task.task_id] = task
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            await self._handle_task_failure(task, f"Task timed out after {task.timeout}s")
        except Exception as e:
            await self._handle_task_failure(task, str(e))
    
    async def _handle_task_failure(self, task: AsyncTask, error_message: str):
        """Handle task failure with retry logic"""
        task.retry_count += 1
        task.error = error_message
        
        if task.retry_count < task.max_retries:
            # Retry task
            task.status = TaskStatus.PENDING
            task.started_at = None
            await self.submit_task(task)
            logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
        else:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            
            # Move to failed tasks
            del self.running_tasks[task.task_id]
            self.failed_tasks[task.task_id] = task
            
            logger.error(f"Task {task.task_id} failed permanently: {error_message}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks": len(self.pending_tasks) + len(self.running_tasks) + 
                          len(self.completed_tasks) + len(self.failed_tasks)
        }


class AsyncModelTrainer:
    """Asynchronous multi-model trainer"""
    
    def __init__(self, num_models: int = 4):
        self.num_models = num_models
        self.task_queue = AsyncTaskQueue(max_concurrent_tasks=num_models)
        self.model_states = {}
        self.training_metrics = defaultdict(list)
        
        # Initialize model states
        for i in range(num_models):
            model_id = f"model_{i}"
            self.model_states[model_id] = {
                "episodes_completed": 0,
                "total_reward": 0.0,
                "last_update": time.time()
            }
        
        # Register task handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register handlers for different task types"""
        self.task_queue.register_handler(TaskType.MODEL_TRAINING, self._handle_training_task)
        self.task_queue.register_handler(TaskType.EVALUATION, self._handle_evaluation_task)
        self.task_queue.register_handler(TaskType.CHECKPOINT_SAVE, self._handle_checkpoint_task)
        self.task_queue.register_handler(TaskType.MONITORING, self._handle_monitoring_task)
    
    async def _handle_training_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model training task"""
        model_id = payload["model_id"]
        episode_data = payload["episode_data"]
        
        # Simulate training work
        training_time = 0.5 + (hash(model_id) % 10) * 0.1  # Vary training time
        await asyncio.sleep(training_time)
        
        # Simulate reward calculation
        reward = 0.5 + (hash(f"{model_id}_{episode_data}") % 100) / 200.0
        
        # Update model state
        self.model_states[model_id]["episodes_completed"] += 1
        self.model_states[model_id]["total_reward"] += reward
        self.model_states[model_id]["last_update"] = time.time()
        
        return {
            "model_id": model_id,
            "reward": reward,
            "training_time": training_time,
            "episode": self.model_states[model_id]["episodes_completed"]
        }
    
    async def _handle_evaluation_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model evaluation task"""
        model_id = payload["model_id"]
        
        # Simulate evaluation
        await asyncio.sleep(0.3)
        
        state = self.model_states[model_id]
        avg_reward = state["total_reward"] / max(1, state["episodes_completed"])
        
        return {
            "model_id": model_id,
            "avg_reward": avg_reward,
            "episodes_completed": state["episodes_completed"],
            "evaluation_time": time.time()
        }
    
    async def _handle_checkpoint_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle checkpoint saving task"""
        checkpoint_data = payload["checkpoint_data"]
        
        # Simulate checkpoint saving
        await asyncio.sleep(1.0)
        
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        return {
            "checkpoint_id": checkpoint_id,
            "saved_at": time.time(),
            "data_size": len(str(checkpoint_data))
        }
    
    async def _handle_monitoring_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle monitoring task"""
        # Collect metrics from all models
        metrics = {}
        
        for model_id, state in self.model_states.items():
            avg_reward = state["total_reward"] / max(1, state["episodes_completed"])
            metrics[model_id] = {
                "avg_reward": avg_reward,
                "episodes": state["episodes_completed"],
                "last_update": state["last_update"]
            }
        
        # Store metrics
        self.training_metrics["monitoring_snapshots"].append({
            "timestamp": time.time(),
            "metrics": metrics
        })
        
        return metrics
    
    async def start_training(self, max_episodes: int = 100):
        """Start asynchronous training"""
        logger.info(f"Starting async training with {self.num_models} models for {max_episodes} episodes")
        
        # Start task queue workers
        await self.task_queue.start_workers()
        
        try:
            # Submit training tasks for all models
            training_tasks = []
            
            for episode in range(max_episodes):
                for model_id in self.model_states.keys():
                    task = AsyncTask(
                        task_id=f"train_{model_id}_ep_{episode}",
                        task_type=TaskType.MODEL_TRAINING,
                        model_id=model_id,
                        payload={
                            "model_id": model_id,
                            "episode_data": f"episode_{episode}"
                        },
                        priority=2  # Training has high priority
                    )
                    
                    await self.task_queue.submit_task(task)
                
                # Submit evaluation tasks every 10 episodes
                if episode % 10 == 0:
                    for model_id in self.model_states.keys():
                        eval_task = AsyncTask(
                            task_id=f"eval_{model_id}_ep_{episode}",
                            task_type=TaskType.EVALUATION,
                            model_id=model_id,
                            payload={"model_id": model_id},
                            priority=1  # Lower priority than training
                        )
                        
                        await self.task_queue.submit_task(eval_task)
                
                # Submit monitoring task every 20 episodes
                if episode % 20 == 0:
                    monitor_task = AsyncTask(
                        task_id=f"monitor_ep_{episode}",
                        task_type=TaskType.MONITORING,
                        model_id="all",
                        payload={},
                        priority=1
                    )
                    
                    await self.task_queue.submit_task(monitor_task)
            
            # Submit final checkpoint task
            checkpoint_task = AsyncTask(
                task_id="final_checkpoint",
                task_type=TaskType.CHECKPOINT_SAVE,
                model_id="all",
                payload={"checkpoint_data": self.model_states},
                priority=3  # Highest priority
            )
            
            await self.task_queue.submit_task(checkpoint_task)
            
            # Wait for all tasks to complete
            await self._wait_for_completion()
            
        finally:
            # Stop workers
            await self.task_queue.stop_workers()
    
    async def _wait_for_completion(self):
        """Wait for all tasks to complete"""
        logger.info("Waiting for all tasks to complete...")
        
        while True:
            stats = self.task_queue.get_stats()
            
            if stats["pending_tasks"] == 0 and stats["running_tasks"] == 0:
                break
            
            # Log progress every 5 seconds
            logger.info(f"Progress: {stats['completed_tasks']} completed, "
                       f"{stats['running_tasks']} running, "
                       f"{stats['pending_tasks']} pending")
            
            await asyncio.sleep(5.0)
        
        logger.info("All tasks completed!")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        stats = self.task_queue.get_stats()
        
        model_summaries = {}
        for model_id, state in self.model_states.items():
            avg_reward = state["total_reward"] / max(1, state["episodes_completed"])
            model_summaries[model_id] = {
                "episodes_completed": state["episodes_completed"],
                "total_reward": state["total_reward"],
                "avg_reward": avg_reward
            }
        
        return {
            "task_stats": stats,
            "model_summaries": model_summaries,
            "monitoring_snapshots": len(self.training_metrics["monitoring_snapshots"])
        }


async def basic_async_example():
    """Basic asynchronous task processing example"""
    print("Basic Async Task Processing Example")
    print("=" * 40)
    
    # Create task queue
    task_queue = AsyncTaskQueue(max_concurrent_tasks=2)
    
    # Register simple handler
    async def simple_handler(payload):
        await asyncio.sleep(payload.get("duration", 1.0))
        return {"processed": payload, "timestamp": time.time()}
    
    task_queue.register_handler(TaskType.DATA_PROCESSING, simple_handler)
    
    # Start workers
    await task_queue.start_workers()
    
    try:
        # Submit some tasks
        tasks = []
        for i in range(5):
            task = AsyncTask(
                task_id=f"task_{i}",
                task_type=TaskType.DATA_PROCESSING,
                model_id="test",
                payload={"data": f"item_{i}", "duration": 0.5 + i * 0.2},
                priority=i % 3  # Vary priorities
            )
            
            await task_queue.submit_task(task)
            tasks.append(task)
        
        # Wait for completion
        while True:
            stats = task_queue.get_stats()
            if stats["pending_tasks"] == 0 and stats["running_tasks"] == 0:
                break
            
            print(f"Status: {stats['completed_tasks']} completed, "
                  f"{stats['running_tasks']} running, "
                  f"{stats['pending_tasks']} pending")
            
            await asyncio.sleep(1.0)
        
        # Show results
        final_stats = task_queue.get_stats()
        print(f"\nFinal Results:")
        print(f"  Completed: {final_stats['completed_tasks']}")
        print(f"  Failed: {final_stats['failed_tasks']}")
        
    finally:
        await task_queue.stop_workers()


async def multi_model_async_training():
    """Multi-model asynchronous training example"""
    print("\nMulti-Model Async Training Example")
    print("=" * 40)
    
    # Create async trainer
    trainer = AsyncModelTrainer(num_models=3)
    
    # Start training
    await trainer.start_training(max_episodes=20)
    
    # Get summary
    summary = trainer.get_training_summary()
    
    print("Training Summary:")
    print(f"  Task Statistics: {summary['task_stats']}")
    print("  Model Performance:")
    
    for model_id, stats in summary['model_summaries'].items():
        print(f"    {model_id}: {stats['episodes_completed']} episodes, "
              f"{stats['avg_reward']:.3f} avg reward")
    
    print(f"  Monitoring Snapshots: {summary['monitoring_snapshots']}")


async def main():
    """Main example function"""
    print("Async Architecture Examples")
    print("=" * 50)
    
    # Run basic async example
    await basic_async_example()
    
    # Run multi-model training example
    await multi_model_async_training()
    
    print("\nAsync architecture examples completed!")


if __name__ == "__main__":
    asyncio.run(main())

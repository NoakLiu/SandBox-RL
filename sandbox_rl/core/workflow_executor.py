"""
Paper-aligned alias: Workflow Graph Executor.

This module exposes a thin wrapper over the existing DAG manager, giving
names that match the paper terminology without altering behavior.
"""

from .dag_manager import DAG_Manager as WorkflowGraphExecutor

__all__ = ["WorkflowGraphExecutor"]



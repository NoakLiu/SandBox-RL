def test_paper_aligned_exports():
    import sandgraph.core as core

    # Workflow executor alias
    assert hasattr(core, "WorkflowGraphExecutor")

    # DAG replay buffer
    assert hasattr(core, "DAGReplayBuffer")
    assert hasattr(core, "DAGTraceStep")

    # RL engine wrapper
    assert hasattr(core, "RLEngine")



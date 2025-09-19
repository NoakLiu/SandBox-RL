#!/usr/bin/env python3
"""
Minimal local trainer server scaffolding with in-process queues.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Sample:
    sample_id: str
    payload: Dict[str, Any]


@dataclass
class Result:
    sample_id: str
    agent_id: str
    trajectory: Dict[str, Any]  # serialized Trajectory


class TrainerServer:
    def __init__(self, maxsize: int = 1024):
        self.samples = queue.Queue(maxsize=maxsize)
        self.results = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()

    def put_samples(self, batch: List[Sample]) -> None:
        for s in batch:
            self.samples.put(s, timeout=5)

    def get_sample(self, timeout: float = 5.0) -> Optional[Sample]:
        try:
            return self.samples.get(timeout=timeout)
        except queue.Empty:
            return None

    def put_result(self, result: Result) -> None:
        self.results.put(result, timeout=5)

    def get_results_batch(self, max_items: int = 64, timeout: float = 0.5) -> List[Result]:
        out: List[Result] = []
        end = time.time() + timeout
        while time.time() < end and len(out) < max_items:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                time.sleep(0.01)
        return out

    def stop(self) -> None:
        self._stop.set()



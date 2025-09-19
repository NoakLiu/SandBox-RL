#!/usr/bin/env python3
"""
Simple gridworld maze environment for Sandbox-RL applications.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set
import random


Action = int  # 0=up,1=right,2=down,3=left


@dataclass
class MazeConfig:
    width: int = 7
    height: int = 7
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (6, 6)
    walls: Set[Tuple[int, int]] = None  # type: ignore
    max_steps: int = 64


class MazeEnv:
    def __init__(self, cfg: MazeConfig):
        self.cfg = cfg
        if self.cfg.walls is None:
            self.cfg.walls = set()
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.pos = (int(self.cfg.start[0]), int(self.cfg.start[1]))
        self.steps = 0
        return self.pos

    def step(self, a: Action) -> Tuple[Tuple[int, int], float, bool]:
        x, y = self.pos
        nx, ny = x, y
        if a == 0:  # up
            ny = max(0, y - 1)
        elif a == 1:  # right
            nx = min(self.cfg.width - 1, x + 1)
        elif a == 2:  # down
            ny = min(self.cfg.height - 1, y + 1)
        elif a == 3:  # left
            nx = max(0, x - 1)
        if (nx, ny) in self.cfg.walls:
            nx, ny = x, y  # blocked
        self.pos = (nx, ny)
        self.steps += 1
        done = self.pos == self.cfg.goal or self.steps >= self.cfg.max_steps
        reward = 1.0 if self.pos == self.cfg.goal else -0.01
        return self.pos, reward, done

    def render_ascii(self) -> str:
        grid = []
        for j in range(self.cfg.height):
            row = []
            for i in range(self.cfg.width):
                if (i, j) == self.pos:
                    row.append('A')
                elif (i, j) == self.cfg.goal:
                    row.append('G')
                elif (i, j) in self.cfg.walls:
                    row.append('#')
                else:
                    row.append('.')
            grid.append(''.join(row))
        return '\n'.join(grid)



#!/usr/bin/env python3
"""
Unified Environments - Multi-Model Training Environments
========================================================

Provides environments for multi-model RL training:
1. Cooperative-competitive environments
2. Multi-agent staged environments  
3. Team battle environments
4. Maze environments for spatial reasoning
5. Concordia-style social environments
"""

import logging
import time
import random
import math
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class EnvironmentType(Enum):
    """Environment types for multi-model training"""
    COOPERATIVE_COMPETITIVE = "cooperative_competitive"
    MULTI_AGENT_STAGED = "multi_agent_staged"
    TEAM_BATTLE = "team_battle"
    MAZE_NAVIGATION = "maze_navigation"
    SOCIAL_INTERACTION = "social_interaction"


class Action(Enum):
    """Basic actions for maze environment"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class EnvironmentConfig:
    """Base environment configuration"""
    env_type: EnvironmentType
    num_agents: int = 8
    max_steps: int = 1000
    base_scale: float = 1.0
    noise_std: float = 0.05
    difficulty: float = 0.3
    cooperation_weight: float = 0.25
    competition_weight: float = 0.25


@dataclass
class MazeConfig:
    """Maze environment configuration"""
    width: int = 7
    height: int = 7
    start: Tuple[int, int] = (0, 0)
    goal: Tuple[int, int] = (6, 6)
    walls: Set[Tuple[int, int]] = field(default_factory=set)
    max_steps: int = 64


class BaseEnvironment(ABC):
    """Base class for all training environments"""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.step_count = 0
        self.episode_count = 0
        self.agents = {}
        self.state = {}
        
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        pass
    
    @abstractmethod
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get observation for specific agent"""
        pass


class CooperativeCompetitiveEnv(BaseEnvironment):
    """Environment for cooperative-competitive multi-model training"""
    
    def __init__(self, cooperation_level: float, difficulty: float, config: EnvironmentConfig):
        super().__init__(config)
        self.cooperation_level = max(0.0, min(1.0, cooperation_level))
        self.difficulty = max(0.0, min(1.0, difficulty))
        
    def reset(self) -> Dict[str, Any]:
        """Reset to initial state"""
        self.step_count = 0
        self.episode_count += 1
        
        # Initialize agent states
        self.state = {
            "episode": self.episode_count,
            "cooperation_level": self.cooperation_level,
            "difficulty": self.difficulty
        }
        
        return self.state
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """Execute environment step with multi-agent actions"""
        self.step_count += 1
        
        # Calculate rewards based on cooperation/competition dynamics
        rewards = {}
        agent_ids = list(actions.keys())
        
        for agent_id, action in actions.items():
            # Base reward scaled by difficulty
            base_reward = self.config.base_scale * max(0.1, 1.0 - self.difficulty)
            
            # Cooperation bonus/penalty
            if len(agent_ids) >= 2:
                other_actions = [actions[aid] for aid in agent_ids if aid != agent_id]
                cooperation_ratio = sum(other_actions) / len(other_actions) if other_actions else 0.5
                
                if action == 1:  # Cooperate
                    cooperation_bonus = self.config.cooperation_weight * self.cooperation_level * cooperation_ratio
                else:  # Compete
                    competition_bonus = self.config.competition_weight * (1.0 - self.cooperation_level) * (1.0 - cooperation_ratio)
                    cooperation_bonus = competition_bonus
            else:
                cooperation_bonus = 0.0
            
            # Add noise for realistic training
            noise = random.gauss(0, self.config.noise_std) if self.config.noise_std > 0 else 0.0
            
            total_reward = max(0.0, base_reward + cooperation_bonus + noise)
            rewards[agent_id] = total_reward
        
        # Check if episode is done
        done = self.step_count >= self.config.max_steps
        
        # Update state
        self.state.update({
            "step": self.step_count,
            "actions": actions,
            "cooperation_ratio": sum(actions.values()) / len(actions) if actions else 0.0
        })
        
        info = {
            "cooperation_level": self.cooperation_level,
            "difficulty": self.difficulty,
            "step_count": self.step_count
        }
        
        return self.state, rewards, done, info
    
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get observation for agent"""
        return {
            "agent_id": agent_id,
            "step": self.step_count,
            "cooperation_level": self.cooperation_level,
            "difficulty": self.difficulty,
            "episode": self.episode_count
        }


class MultiAgentStagedEnv(BaseEnvironment):
    """Multi-agent environment with staged reward dynamics"""
    
    def __init__(self, config: EnvironmentConfig, warmup_episodes: int = 20):
        super().__init__(config)
        self.warmup_episodes = warmup_episodes
        self.adversary_fraction = 0.25
        
        # Pre-assign adversaries
        num_adversaries = max(1, int(config.num_agents * self.adversary_fraction))
        self.adversaries = set(range(num_adversaries))
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment"""
        self.step_count = 0
        self.episode_count += 1
        
        self.state = {
            "episode": self.episode_count,
            "num_agents": self.config.num_agents,
            "warmup_phase": self.episode_count < self.warmup_episodes
        }
        
        return self.state
    
    def step(self, actions: List[int]) -> Tuple[Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """Execute step with staged reward dynamics"""
        self.step_count += 1
        
        # Progressive base reward to enable learning signal
        base0 = max(0.1, 1.0 - self.config.difficulty)
        base = base0 * (1.0 + 0.3 * min(1.0, max(0.0, (self.episode_count - self.warmup_episodes) / max(1, self.warmup_episodes))))
        
        num_agents = len(actions)
        rewards = [base] * num_agents
        
        if self.episode_count < self.warmup_episodes:
            # Warmup phase: identical rewards for fair start
            return self.state, rewards, False, {"phase": "warmup"}
        
        # Divergence phase: cooperative externality
        cooperation_ratio = sum(actions) / max(1, num_agents)
        
        for i, action in enumerate(actions):
            payoff = self.config.cooperation_weight * (cooperation_ratio - 0.5)
            
            # Adversaries gain when others cooperate but they compete
            if i in self.adversaries:
                if action == 0:  # Compete
                    payoff += 0.1 * cooperation_ratio
                else:
                    payoff -= 0.05 * cooperation_ratio
            
            # Mismatch penalty
            if (action == 1 and cooperation_ratio < 0.3) or (action == 0 and cooperation_ratio > 0.7):
                payoff -= 0.05
            
            # Add small noise for clearer learning signal
            rewards[i] = max(0.0, base + payoff + random.gauss(0, 0.005))
        
        done = self.step_count >= self.config.max_steps
        
        self.state.update({
            "step": self.step_count,
            "cooperation_ratio": cooperation_ratio,
            "phase": "divergence"
        })
        
        info = {
            "adversaries": list(self.adversaries),
            "cooperation_ratio": cooperation_ratio,
            "phase": "divergence"
        }
        
        return self.state, rewards, done, info
    
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get observation for agent"""
        return {
            "agent_id": agent_id,
            "step": self.step_count,
            "episode": self.episode_count,
            "warmup_phase": self.episode_count < self.warmup_episodes,
            "is_adversary": int(agent_id.split('_')[-1]) in self.adversaries if '_' in str(agent_id) else False
        }


class TeamBattleEnv(BaseEnvironment):
    """4v4 team battle environment for competitive multi-model training"""
    
    def __init__(self, config: EnvironmentConfig, warmup_episodes: int = 10):
        super().__init__(config)
        self.warmup_episodes = warmup_episodes
        
        if config.num_agents != 8:
            logger.warning(f"TeamBattleEnv expects 8 agents, got {config.num_agents}")
            config.num_agents = 8
    
    def reset(self) -> Dict[str, Any]:
        """Reset team battle environment"""
        self.step_count = 0
        self.episode_count += 1
        
        self.state = {
            "episode": self.episode_count,
            "team_a": list(range(4)),  # Agents 0-3
            "team_b": list(range(4, 8)),  # Agents 4-7
            "warmup_phase": self.episode_count < self.warmup_episodes
        }
        
        return self.state
    
    def step(self, actions: List[int]) -> Tuple[Dict[str, Any], List[float], bool, Dict[str, Any]]:
        """Execute team battle step"""
        if len(actions) != 8:
            raise ValueError("TeamBattleEnv expects exactly 8 agents")
        
        self.step_count += 1
        base = max(0.1, 1.0 - self.config.difficulty)
        
        if self.episode_count < self.warmup_episodes:
            # Warmup: equal rewards
            return self.state, [base] * 8, False, {"phase": "warmup"}
        
        # Team dynamics
        team_a_actions = actions[:4]
        team_b_actions = actions[4:]
        coop_a = sum(team_a_actions) / 4.0
        coop_b = sum(team_b_actions) / 4.0
        
        rewards = []
        for i in range(8):
            team = 0 if i < 4 else 1
            action = actions[i]
            own_coop = coop_a if team == 0 else coop_b
            opp_coop = coop_b if team == 0 else coop_a
            
            # Team benefit increases with own cooperation
            payoff = self.config.cooperation_weight * (own_coop - 0.5)
            
            if action == 0:  # Compete: exploit opponent cooperation
                payoff += 0.15 * opp_coop
            else:  # Cooperate: penalty for low team cooperation
                payoff -= 0.05 * (1 - own_coop)
            
            rewards.append(max(0.0, base + payoff + random.gauss(0, 0.005)))
        
        done = self.step_count >= self.config.max_steps
        
        self.state.update({
            "step": self.step_count,
            "team_a_coop": coop_a,
            "team_b_coop": coop_b,
            "phase": "battle"
        })
        
        info = {
            "team_cooperation": {"team_a": coop_a, "team_b": coop_b},
            "phase": "battle"
        }
        
        return self.state, rewards, done, info
    
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get observation for team member"""
        agent_idx = int(agent_id.split('_')[-1]) if '_' in str(agent_id) else 0
        team = "A" if agent_idx < 4 else "B"
        
        return {
            "agent_id": agent_id,
            "team": team,
            "step": self.step_count,
            "episode": self.episode_count,
            "warmup_phase": self.episode_count < self.warmup_episodes
        }


class MazeEnv(BaseEnvironment):
    """Maze navigation environment for spatial reasoning"""
    
    def __init__(self, config: MazeConfig):
        super().__init__(EnvironmentConfig(env_type=EnvironmentType.MAZE_NAVIGATION))
        self.maze_config = config
        self.current_pos = config.start
        self.goal_pos = config.goal
        self.walls = config.walls or set()
        
    def reset(self) -> Tuple[int, int]:
        """Reset maze to start position"""
        self.current_pos = self.maze_config.start
        self.step_count = 0
        return self.current_pos
    
    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action in maze"""
        x, y = self.current_pos
        
        # Calculate new position based on action
        if action == Action.UP:
            new_pos = (x, y - 1)
        elif action == Action.DOWN:
            new_pos = (x, y + 1)
        elif action == Action.LEFT:
            new_pos = (x - 1, y)
        elif action == Action.RIGHT:
            new_pos = (x + 1, y)
        else:
            new_pos = self.current_pos
        
        # Check bounds and walls
        new_x, new_y = new_pos
        if (0 <= new_x < self.maze_config.width and 
            0 <= new_y < self.maze_config.height and 
            new_pos not in self.walls):
            self.current_pos = new_pos
        
        self.step_count += 1
        
        # Calculate reward
        if self.current_pos == self.goal_pos:
            reward = 10.0  # Goal reached
            done = True
        elif self.step_count >= self.maze_config.max_steps:
            reward = -1.0  # Timeout penalty
            done = True
        else:
            # Distance-based reward
            goal_x, goal_y = self.goal_pos
            curr_x, curr_y = self.current_pos
            distance = abs(goal_x - curr_x) + abs(goal_y - curr_y)  # Manhattan distance
            reward = -0.1 - 0.01 * distance  # Small step penalty + distance penalty
        
        return self.current_pos, reward, done
    
    def get_observation(self, agent_id: str = "agent") -> Dict[str, Any]:
        """Get current observation"""
        return {
            "position": self.current_pos,
            "goal": self.goal_pos,
            "step": self.step_count,
            "walls_nearby": self._get_nearby_walls()
        }
    
    def _get_nearby_walls(self) -> List[Tuple[int, int]]:
        """Get walls near current position"""
        x, y = self.current_pos
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (x + dx, y + dy) in self.walls:
                    nearby.append((x + dx, y + dy))
        return nearby
    
    def render_ascii(self) -> str:
        """Render maze as ASCII"""
        lines = []
        for y in range(self.maze_config.height):
            line = ""
            for x in range(self.maze_config.width):
                if (x, y) == self.current_pos:
                    line += "A"  # Agent
                elif (x, y) == self.goal_pos:
                    line += "G"  # Goal
                elif (x, y) in self.walls:
                    line += "#"  # Wall
                else:
                    line += "."  # Empty
            lines.append(line)
        return "\n".join(lines)


class SocialInteractionEnv(BaseEnvironment):
    """Social interaction environment for multi-model training"""
    
    def __init__(self, config: EnvironmentConfig, scenario: str = "trading"):
        super().__init__(config)
        self.scenario = scenario
        self.agents_state = {}
        self.social_network = defaultdict(list)
        self.interaction_history = []
        
    def reset(self) -> Dict[str, Any]:
        """Reset social environment"""
        self.step_count = 0
        self.episode_count += 1
        
        # Initialize agent states based on scenario
        self.agents_state = {}
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            if self.scenario == "trading":
                self.agents_state[agent_id] = {
                    "resources": 100,
                    "reputation": 0.5,
                    "trading_history": []
                }
            elif self.scenario == "negotiation":
                self.agents_state[agent_id] = {
                    "position": random.uniform(0.2, 0.8),
                    "flexibility": random.uniform(0.3, 0.7),
                    "negotiation_history": []
                }
        
        self.state = {
            "scenario": self.scenario,
            "episode": self.episode_count,
            "agents": self.agents_state.copy()
        }
        
        return self.state
    
    def step(self, actions: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """Execute social interaction step"""
        self.step_count += 1
        rewards = {}
        
        # Process interactions
        for agent_id, action in actions.items():
            reward = self._calculate_social_reward(agent_id, action)
            rewards[agent_id] = reward
            
            # Update agent state
            if agent_id in self.agents_state:
                self._update_agent_state(agent_id, action, reward)
        
        # Record interaction
        self.interaction_history.append({
            "step": self.step_count,
            "actions": actions.copy(),
            "rewards": rewards.copy()
        })
        
        done = self.step_count >= self.config.max_steps
        
        self.state.update({
            "step": self.step_count,
            "agents": self.agents_state.copy()
        })
        
        info = {
            "scenario": self.scenario,
            "total_interactions": len(self.interaction_history),
            "network_density": len(self.social_network) / max(1, self.config.num_agents)
        }
        
        return self.state, rewards, done, info
    
    def _calculate_social_reward(self, agent_id: str, action: str) -> float:
        """Calculate reward based on social action"""
        base_reward = 1.0
        
        if self.scenario == "trading":
            if "cooperate" in action.lower() or "trade" in action.lower():
                return base_reward + 1.5
            elif "compete" in action.lower():
                return base_reward + 0.5
        elif self.scenario == "negotiation":
            if "compromise" in action.lower() or "agree" in action.lower():
                return base_reward + 2.0
            elif "insist" in action.lower():
                return base_reward + 0.3
        
        return base_reward
    
    def _update_agent_state(self, agent_id: str, action: str, reward: float):
        """Update agent's social state"""
        if agent_id not in self.agents_state:
            return
        
        agent_state = self.agents_state[agent_id]
        
        if self.scenario == "trading":
            agent_state["trading_history"].append({
                "action": action,
                "reward": reward,
                "step": self.step_count
            })
            # Update reputation based on cooperation
            if "cooperate" in action.lower():
                agent_state["reputation"] = min(1.0, agent_state["reputation"] + 0.1)
            elif "defect" in action.lower():
                agent_state["reputation"] = max(0.0, agent_state["reputation"] - 0.05)
        
        elif self.scenario == "negotiation":
            agent_state["negotiation_history"].append({
                "action": action,
                "reward": reward,
                "step": self.step_count
            })
    
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get social observation for agent"""
        base_obs = {
            "agent_id": agent_id,
            "step": self.step_count,
            "scenario": self.scenario
        }
        
        if agent_id in self.agents_state:
            base_obs.update(self.agents_state[agent_id])
        
        # Add network information
        base_obs["connections"] = self.social_network.get(agent_id, [])
        base_obs["network_size"] = len(self.social_network)
        
        return base_obs


class BanditEnv:
    """2-armed bandit environment for classic RL"""
    
    def __init__(self, arm0_prob: float = 0.5, arm1_prob_start: float = 0.5, 
                 arm1_prob_after: float = 0.7, switch_episode: int = 50):
        self.arm0_prob = arm0_prob
        self.arm1_prob_start = arm1_prob_start
        self.arm1_prob_after = arm1_prob_after
        self.switch_episode = switch_episode
        self.episode_count = 0
    
    def step(self, action: int) -> float:
        """Execute bandit action"""
        # Determine current arm probabilities
        arm1_prob = self.arm1_prob_start if self.episode_count < self.switch_episode else self.arm1_prob_after
        
        # Calculate reward
        if action == 1:  # Choose arm 1
            reward = 1.0 if random.random() < arm1_prob else 0.0
        else:  # Choose arm 0
            reward = 1.0 if random.random() < self.arm0_prob else 0.0
        
        self.episode_count += 1
        return reward
    
    def get_observation(self) -> Dict[str, Any]:
        """Get bandit observation"""
        return {
            "episode": self.episode_count,
            "switch_point": self.switch_episode,
            "phase": "before_switch" if self.episode_count < self.switch_episode else "after_switch"
        }


class EnvironmentFactory:
    """Factory for creating training environments"""
    
    @staticmethod
    def create_cooperative_competitive(cooperation_level: float = 0.6, difficulty: float = 0.3,
                                     num_agents: int = 2) -> CooperativeCompetitiveEnv:
        """Create cooperative-competitive environment"""
        config = EnvironmentConfig(
            env_type=EnvironmentType.COOPERATIVE_COMPETITIVE,
            num_agents=num_agents,
            difficulty=difficulty,
            cooperation_weight=0.25,
            competition_weight=0.25
        )
        return CooperativeCompetitiveEnv(cooperation_level, difficulty, config)
    
    @staticmethod
    def create_multi_agent_staged(num_agents: int = 8, difficulty: float = 0.3,
                                warmup_episodes: int = 20) -> MultiAgentStagedEnv:
        """Create multi-agent staged environment"""
        config = EnvironmentConfig(
            env_type=EnvironmentType.MULTI_AGENT_STAGED,
            num_agents=num_agents,
            difficulty=difficulty
        )
        return MultiAgentStagedEnv(config, warmup_episodes)
    
    @staticmethod
    def create_team_battle(difficulty: float = 0.3, warmup_episodes: int = 10) -> TeamBattleEnv:
        """Create team battle environment"""
        config = EnvironmentConfig(
            env_type=EnvironmentType.TEAM_BATTLE,
            num_agents=8,
            difficulty=difficulty
        )
        return TeamBattleEnv(config, warmup_episodes)
    
    @staticmethod
    def create_maze(width: int = 7, height: int = 7, 
                   start: Tuple[int, int] = (0, 0),
                   goal: Tuple[int, int] = (6, 6),
                   walls: Optional[Set[Tuple[int, int]]] = None) -> MazeEnv:
        """Create maze environment"""
        config = MazeConfig(
            width=width,
            height=height,
            start=start,
            goal=goal,
            walls=walls or set()
        )
        return MazeEnv(config)
    
    @staticmethod
    def create_social_interaction(scenario: str = "trading", num_agents: int = 4) -> SocialInteractionEnv:
        """Create social interaction environment"""
        config = EnvironmentConfig(
            env_type=EnvironmentType.SOCIAL_INTERACTION,
            num_agents=num_agents
        )
        return SocialInteractionEnv(config, scenario)
    
    @staticmethod
    def create_bandit(arm0_prob: float = 0.5, arm1_prob_start: float = 0.5,
                     arm1_prob_after: float = 0.7, switch_episode: int = 50) -> BanditEnv:
        """Create bandit environment"""
        return BanditEnv(arm0_prob, arm1_prob_start, arm1_prob_after, switch_episode)


class MultiModelTrainingEnvironment:
    """Unified environment for multi-model training scenarios"""
    
    def __init__(self, env_type: EnvironmentType, **kwargs):
        self.env_type = env_type
        self.environments = {}
        self.active_env = None
        
        # Create environment based on type
        if env_type == EnvironmentType.COOPERATIVE_COMPETITIVE:
            self.active_env = EnvironmentFactory.create_cooperative_competitive(**kwargs)
        elif env_type == EnvironmentType.MULTI_AGENT_STAGED:
            self.active_env = EnvironmentFactory.create_multi_agent_staged(**kwargs)
        elif env_type == EnvironmentType.TEAM_BATTLE:
            self.active_env = EnvironmentFactory.create_team_battle(**kwargs)
        elif env_type == EnvironmentType.MAZE_NAVIGATION:
            self.active_env = EnvironmentFactory.create_maze(**kwargs)
        elif env_type == EnvironmentType.SOCIAL_INTERACTION:
            self.active_env = EnvironmentFactory.create_social_interaction(**kwargs)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        logger.info(f"Created multi-model training environment: {env_type.value}")
    
    def reset(self) -> Any:
        """Reset active environment"""
        return self.active_env.reset()
    
    def step(self, actions: Any) -> Tuple[Any, Any, bool, Dict[str, Any]]:
        """Execute step in active environment"""
        return self.active_env.step(actions)
    
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get observation for agent"""
        return self.active_env.get_observation(agent_id)
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information"""
        return {
            "env_type": self.env_type.value,
            "step_count": getattr(self.active_env, 'step_count', 0),
            "episode_count": getattr(self.active_env, 'episode_count', 0),
            "config": getattr(self.active_env, 'config', {}).__dict__ if hasattr(getattr(self.active_env, 'config', {}), '__dict__') else {}
        }


# Sandbox Protocol for compatibility
class SandboxProtocol:
    """Protocol for sandbox compatibility"""
    
    def case_generator(self) -> Dict[str, Any]:
        """Generate case for training"""
        return {"task": "multi_model_training", "difficulty": "medium"}
    
    def prompt_func(self, case: Dict[str, Any]) -> str:
        """Generate prompt from case"""
        return f"Execute task: {case.get('task', 'unknown')}"
    
    def verify_score(self, response: str, case: Dict[str, Any], format_score: float = 0.0) -> float:
        """Verify and score response"""
        # Simple scoring based on response length and relevance
        base_score = min(1.0, len(response.split()) / 50.0)
        
        # Bonus for task-relevant keywords
        task = case.get('task', '')
        if task in response.lower():
            base_score += 0.2
        
        return min(1.0, base_score + format_score)


class Sandbox(SandboxProtocol):
    """Base sandbox implementation"""
    
    def __init__(self, sandbox_id: str, description: str = ""):
        self.sandbox_id = sandbox_id
        self.description = description
        self.execution_count = 0
    
    def run_full_cycle(self, llm_response_generator: Optional[Callable] = None) -> Dict[str, Any]:
        """Run complete sandbox cycle"""
        self.execution_count += 1
        
        # Generate case
        case = self.case_generator()
        
        # Create prompt
        prompt = self.prompt_func(case)
        
        # Get LLM response
        if llm_response_generator:
            response = llm_response_generator(prompt)
        else:
            response = f"Default response for: {prompt[:50]}..."
        
        # Verify and score
        score = self.verify_score(response, case)
        
        return {
            "case": case,
            "prompt": prompt,
            "response": response,
            "score": score,
            "execution_count": self.execution_count,
            "sandbox_id": self.sandbox_id
        }


# Factory functions for common multi-model training setups
def create_multi_model_coop_compete_env(num_models: int = 8, cooperation_level: float = 0.6) -> MultiModelTrainingEnvironment:
    """Create cooperative-competitive environment for multiple models"""
    return MultiModelTrainingEnvironment(
        EnvironmentType.COOPERATIVE_COMPETITIVE,
        cooperation_level=cooperation_level,
        difficulty=0.3,
        num_agents=num_models
    )


def create_multi_model_team_battle() -> MultiModelTrainingEnvironment:
    """Create 4v4 team battle environment"""
    return MultiModelTrainingEnvironment(
        EnvironmentType.TEAM_BATTLE,
        difficulty=0.3,
        warmup_episodes=10
    )


def create_multi_model_staged_env(num_models: int = 8) -> MultiModelTrainingEnvironment:
    """Create staged environment with warmup and divergence phases"""
    return MultiModelTrainingEnvironment(
        EnvironmentType.MULTI_AGENT_STAGED,
        num_agents=num_models,
        difficulty=0.3,
        warmup_episodes=20
    )


def create_maze_training_env(complexity: str = "medium") -> MultiModelTrainingEnvironment:
    """Create maze environment for spatial reasoning training"""
    if complexity == "simple":
        width, height = 5, 5
        walls = {(2, 1), (2, 2), (2, 3)}
    elif complexity == "complex":
        width, height = 10, 10
        walls = {(3, i) for i in range(1, 8)} | {(6, i) for i in range(2, 9)}
    else:  # medium
        width, height = 7, 7
        walls = {(2, 2), (2, 3), (3, 2), (4, 4), (4, 5)}
    
    return MultiModelTrainingEnvironment(
        EnvironmentType.MAZE_NAVIGATION,
        width=width,
        height=height,
        walls=walls
    )


def create_social_training_env(scenario: str = "trading", num_models: int = 6) -> MultiModelTrainingEnvironment:
    """Create social interaction environment for multi-model training"""
    return MultiModelTrainingEnvironment(
        EnvironmentType.SOCIAL_INTERACTION,
        scenario=scenario,
        num_agents=num_models
    )

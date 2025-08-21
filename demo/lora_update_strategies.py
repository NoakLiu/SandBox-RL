#!/usr/bin/env python3
"""
LoRA Update Strategies for Single vLLM + 8 LoRA Adapters

This module implements two key update strategies:
1. Team-Based Update: Reward Partial Sharing (分化) - LoRAs share partial rewards to encourage specialization
2. Adversarial Update: Compete with negative reward (卷王) - LoRAs compete for resources with negative rewards for losers

Architecture: Single vLLM server with 8 GPUs serving 8 LoRA adapters via model=adapter_name
"""

import asyncio
import time
import random
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import aiohttp
import requests

logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    """LoRA update strategy types"""
    INDEPENDENT = "independent"      # Each LoRA updates independently
    TEAM_BASED = "team_based"       # Reward partial sharing for differentiation
    ADVERSARIAL = "adversarial"     # Compete with negative rewards (卷王)


@dataclass
class LoRAAdapter:
    """Represents a LoRA adapter in the system"""
    adapter_id: int
    adapter_name: str
    current_reward: float = 0.5
    total_reward: float = 0.0
    update_count: int = 0
    specialization_score: float = 0.0
    competition_score: float = 0.0
    team_id: Optional[int] = None
    performance_history: List[float] = field(default_factory=list)
    last_update_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.adapter_name = f"lora{self.adapter_id}"


@dataclass
class TeamConfig:
    """Team configuration for team-based updates"""
    team_id: int
    member_ids: List[int]
    sharing_ratio: float = 0.3  # How much reward is shared within team
    specialization_bonus: float = 0.1  # Bonus for specialization
    cooperation_bonus: float = 0.05  # Bonus for team cooperation


@dataclass
class CompetitionConfig:
    """Competition configuration for adversarial updates"""
    winner_bonus: float = 0.2
    loser_penalty: float = -0.1
    resource_contention_factor: float = 0.3
    volume_king_threshold: float = 0.7  # Threshold for 卷王 phenomenon


class SingleVLLMClient:
    """Client for single vLLM server with 8 LoRA adapters"""
    
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.session = None
        self.adapters = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        """Initialize 8 LoRA adapters"""
        for i in range(1, 9):
            self.adapters[i] = LoRAAdapter(adapter_id=i)
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def generate_with_adapter(self, prompt: str, adapter_id: int, 
                                  temperature: float = 0.7, max_tokens: int = 100) -> str:
        """Generate text using specific LoRA adapter"""
        if adapter_id not in self.adapters:
            raise ValueError(f"Invalid adapter ID: {adapter_id}")
        
        adapter = self.adapters[adapter_id]
        
        try:
            session = await self._get_session()
            url = f"{self.server_url}/v1/chat/completions"
            
            payload = {
                "model": adapter.adapter_name,  # Use adapter name as model
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return result
                else:
                    logger.warning(f"Generation failed for adapter {adapter_id}: {response.status}")
                    return f"[Mock] Adapter {adapter_id} response"
        
        except Exception as e:
            logger.error(f"Error generating with adapter {adapter_id}: {e}")
            return f"[Mock] Adapter {adapter_id} response"
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.server_url}/health", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()


class TeamBasedUpdater:
    """Team-Based Update Strategy: Reward Partial Sharing for differentiation"""
    
    def __init__(self, vllm_client: SingleVLLMClient, teams: List[TeamConfig]):
        self.vllm_client = vllm_client
        self.teams = {team.team_id: team for team in teams}
        self.update_history = []
        
        # Assign team IDs to adapters
        for team in teams:
            for adapter_id in team.member_ids:
                if adapter_id in self.vllm_client.adapters:
                    self.vllm_client.adapters[adapter_id].team_id = team.team_id
    
    async def update_team_rewards(self, task_results: Dict[int, float]) -> Dict[int, float]:
        """
        Update rewards using team-based strategy
        
        Args:
            task_results: Dict[adapter_id, raw_reward]
            
        Returns:
            Dict[adapter_id, final_reward] with partial sharing applied
        """
        final_rewards = {}
        
        for team_id, team_config in self.teams.items():
            team_members = team_config.member_ids
            team_results = {aid: task_results.get(aid, 0.0) for aid in team_members}
            
            if not team_results:
                continue
            
            # Calculate team average reward
            team_avg_reward = sum(team_results.values()) / len(team_results)
            
            # Calculate specialization bonus
            reward_variance = np.var(list(team_results.values()))
            specialization_bonus = min(team_config.specialization_bonus, reward_variance)
            
            # Apply partial sharing and bonuses
            for adapter_id, raw_reward in team_results.items():
                adapter = self.vllm_client.adapters[adapter_id]
                
                # Individual reward (70% of raw reward)
                individual_reward = raw_reward * (1 - team_config.sharing_ratio)
                
                # Shared reward (30% of team average)
                shared_reward = team_avg_reward * team_config.sharing_ratio
                
                # Cooperation bonus (if team performs well)
                cooperation_bonus = team_config.cooperation_bonus if team_avg_reward > 0.6 else 0.0
                
                # Final reward
                final_reward = individual_reward + shared_reward + specialization_bonus + cooperation_bonus
                
                # Update adapter stats
                adapter.current_reward = final_reward
                adapter.total_reward += final_reward
                adapter.update_count += 1
                adapter.specialization_score = reward_variance
                adapter.performance_history.append(final_reward)
                adapter.last_update_time = time.time()
                
                final_rewards[adapter_id] = final_reward
                
                logger.info(f"Team {team_id} - Adapter {adapter_id}: "
                          f"raw={raw_reward:.3f}, final={final_reward:.3f}, "
                          f"specialization={specialization_bonus:.3f}")
        
        # Record update
        self.update_history.append({
            'timestamp': time.time(),
            'strategy': 'team_based',
            'team_results': {team_id: {
                'avg_reward': sum(task_results.get(aid, 0.0) for aid in team.member_ids) / len(team.member_ids),
                'specialization': np.var([task_results.get(aid, 0.0) for aid in team.member_ids])
            } for team_id, team in self.teams.items()},
            'final_rewards': final_rewards
        })
        
        return final_rewards
    
    def get_team_statistics(self) -> Dict[str, Any]:
        """Get team-based update statistics"""
        if not self.update_history:
            return {}
        
        latest = self.update_history[-1]
        
        return {
            'strategy': 'team_based',
            'teams': {
                team_id: {
                    'member_count': len(team.member_ids),
                    'avg_reward': latest['team_results'][team_id]['avg_reward'],
                    'specialization': latest['team_results'][team_id]['specialization'],
                    'members': team.member_ids
                }
                for team_id, team in self.teams.items()
            },
            'total_updates': len(self.update_history),
            'differentiation_level': sum(
                latest['team_results'][team_id]['specialization'] 
                for team_id in self.teams.keys()
            ) / len(self.teams)
        }


class AdversarialUpdater:
    """Adversarial Update Strategy: Compete with negative rewards (卷王)"""
    
    def __init__(self, vllm_client: SingleVLLMClient, config: CompetitionConfig):
        self.vllm_client = vllm_client
        self.config = config
        self.update_history = []
        self.competition_history = []
    
    async def update_competitive_rewards(self, task_results: Dict[int, float]) -> Dict[int, float]:
        """
        Update rewards using adversarial strategy
        
        Args:
            task_results: Dict[adapter_id, raw_reward]
            
        Returns:
            Dict[adapter_id, final_reward] with competition penalties/bonuses
        """
        if not task_results:
            return {}
        
        # Rank adapters by performance
        ranked_adapters = sorted(task_results.items(), key=lambda x: x[1], reverse=True)
        
        # Determine winners and losers
        num_adapters = len(ranked_adapters)
        winner_count = max(1, num_adapters // 3)  # Top 1/3 are winners
        loser_count = max(1, num_adapters // 3)   # Bottom 1/3 are losers
        
        winners = [aid for aid, _ in ranked_adapters[:winner_count]]
        losers = [aid for aid, _ in ranked_adapters[-loser_count:]]
        neutrals = [aid for aid, _ in ranked_adapters[winner_count:-loser_count]]
        
        final_rewards = {}
        competition_intensity = 0.0
        
        # Apply competitive rewards
        for adapter_id, raw_reward in task_results.items():
            adapter = self.vllm_client.adapters[adapter_id]
            
            if adapter_id in winners:
                # Winner bonus
                final_reward = raw_reward + self.config.winner_bonus
                competition_bonus = self.config.winner_bonus
                adapter.competition_score += 1
                
            elif adapter_id in losers:
                # Loser penalty
                final_reward = raw_reward + self.config.loser_penalty
                competition_bonus = self.config.loser_penalty
                adapter.competition_score -= 1
                
            else:
                # Neutral (no bonus/penalty)
                final_reward = raw_reward
                competition_bonus = 0.0
            
            # Resource contention penalty (if many adapters competing)
            if len(task_results) > 4:
                contention_penalty = -self.config.resource_contention_factor * (len(task_results) - 4) / 4
                final_reward += contention_penalty
                competition_bonus += contention_penalty
            
            # Ensure reward doesn't go below minimum
            final_reward = max(0.1, final_reward)
            
            # Update adapter stats
            adapter.current_reward = final_reward
            adapter.total_reward += final_reward
            adapter.update_count += 1
            adapter.performance_history.append(final_reward)
            adapter.last_update_time = time.time()
            
            final_rewards[adapter_id] = final_reward
            competition_intensity += abs(competition_bonus)
            
            logger.info(f"Competition - Adapter {adapter_id}: "
                      f"raw={raw_reward:.3f}, final={final_reward:.3f}, "
                      f"bonus={competition_bonus:.3f}, "
                      f"status={'winner' if adapter_id in winners else 'loser' if adapter_id in losers else 'neutral'}")
        
        # Check for 卷王 phenomenon
        volume_king_detected = competition_intensity > self.config.volume_king_threshold
        
        # Record competition
        competition_record = {
            'timestamp': time.time(),
            'winners': winners,
            'losers': losers,
            'neutrals': neutrals,
            'competition_intensity': competition_intensity,
            'volume_king_detected': volume_king_detected,
            'raw_rewards': task_results,
            'final_rewards': final_rewards
        }
        
        self.competition_history.append(competition_record)
        self.update_history.append(competition_record)
        
        if volume_king_detected:
            logger.warning(f"卷王现象检测到! 竞争强度: {competition_intensity:.3f}")
        
        return final_rewards
    
    def get_competition_statistics(self) -> Dict[str, Any]:
        """Get adversarial competition statistics"""
        if not self.competition_history:
            return {}
        
        latest = self.competition_history[-1]
        
        # Calculate competition metrics
        total_competitions = len(self.competition_history)
        volume_king_count = sum(1 for record in self.competition_history if record['volume_king_detected'])
        
        # Adapter competition scores
        adapter_scores = {}
        for adapter_id, adapter in self.vllm_client.adapters.items():
            adapter_scores[adapter_id] = {
                'competition_score': adapter.competition_score,
                'total_reward': adapter.total_reward,
                'update_count': adapter.update_count,
                'avg_reward': adapter.total_reward / max(adapter.update_count, 1)
            }
        
        return {
            'strategy': 'adversarial',
            'total_competitions': total_competitions,
            'volume_king_count': volume_king_count,
            'volume_king_rate': volume_king_count / max(total_competitions, 1),
            'latest_competition': {
                'winners': latest['winners'],
                'losers': latest['losers'],
                'competition_intensity': latest['competition_intensity'],
                'volume_king_detected': latest['volume_king_detected']
            },
            'adapter_competition_scores': adapter_scores,
            'competition_intensity_trend': [
                record['competition_intensity'] for record in self.competition_history[-10:]
            ]
        }


class LoRAUpdateOrchestrator:
    """Main orchestrator for LoRA update strategies"""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8001",
                 strategy: UpdateStrategy = UpdateStrategy.TEAM_BASED):
        
        self.vllm_client = SingleVLLMClient(server_url)
        self.strategy = strategy
        
        # Initialize teams for team-based strategy
        teams = [
            TeamConfig(team_id=1, member_ids=[1, 2, 3, 4], sharing_ratio=0.3),
            TeamConfig(team_id=2, member_ids=[5, 6, 7, 8], sharing_ratio=0.3)
        ]
        
        # Initialize competition config for adversarial strategy
        competition_config = CompetitionConfig(
            winner_bonus=0.2,
            loser_penalty=-0.1,
            resource_contention_factor=0.3,
            volume_king_threshold=0.7
        )
        
        # Initialize updaters
        self.team_updater = TeamBasedUpdater(self.vllm_client, teams)
        self.adversarial_updater = AdversarialUpdater(self.vllm_client, competition_config)
        
        # Task templates for different strategies
        self.task_templates = {
            UpdateStrategy.TEAM_BASED: [
                "Analyze the given text and provide a specialized response focusing on technical details.",
                "Generate a creative solution for the problem with innovative approaches.",
                "Evaluate the situation and provide strategic recommendations.",
                "Create a comprehensive analysis with multiple perspectives."
            ],
            UpdateStrategy.ADVERSARIAL: [
                "Compete with other models to provide the best solution for this problem.",
                "Demonstrate superior performance compared to other adapters.",
                "Show why your approach is better than alternatives.",
                "Prove your expertise in this competitive environment."
            ]
        }
    
    async def run_update_cycle(self, num_rounds: int = 10) -> Dict[str, Any]:
        """Run a complete update cycle"""
        logger.info(f"Starting {self.strategy.value} update cycle with {num_rounds} rounds")
        
        results = {
            'strategy': self.strategy.value,
            'rounds': [],
            'final_statistics': {}
        }
        
        for round_num in range(num_rounds):
            logger.info(f"Round {round_num + 1}/{num_rounds}")
            
            # Generate tasks for all adapters
            task_results = await self._generate_round_tasks(round_num)
            
            # Apply update strategy
            if self.strategy == UpdateStrategy.TEAM_BASED:
                final_rewards = await self.team_updater.update_team_rewards(task_results)
                round_stats = self.team_updater.get_team_statistics()
                
            elif self.strategy == UpdateStrategy.ADVERSARIAL:
                final_rewards = await self.adversarial_updater.update_competitive_rewards(task_results)
                round_stats = self.adversarial_updater.get_competition_statistics()
                
            else:  # INDEPENDENT
                final_rewards = task_results
                round_stats = {'strategy': 'independent'}
            
            # Record round results
            round_result = {
                'round': round_num + 1,
                'raw_rewards': task_results,
                'final_rewards': final_rewards,
                'statistics': round_stats
            }
            
            results['rounds'].append(round_result)
            
            # Log progress
            avg_reward = sum(final_rewards.values()) / len(final_rewards)
            logger.info(f"Round {round_num + 1} complete - Avg reward: {avg_reward:.3f}")
            
            # Small delay between rounds
            await asyncio.sleep(1)
        
        # Final statistics
        if self.strategy == UpdateStrategy.TEAM_BASED:
            results['final_statistics'] = self.team_updater.get_team_statistics()
        elif self.strategy == UpdateStrategy.ADVERSARIAL:
            results['final_statistics'] = self.adversarial_updater.get_competition_statistics()
        
        logger.info(f"Update cycle complete for {self.strategy.value}")
        return results
    
    async def _generate_round_tasks(self, round_num: int) -> Dict[int, float]:
        """Generate tasks for all adapters and collect rewards"""
        task_results = {}
        
        # Select task template based on strategy
        templates = self.task_templates.get(self.strategy, ["Generate a response."])
        base_prompt = templates[round_num % len(templates)]
        
        # Generate tasks for all adapters
        tasks = []
        for adapter_id in range(1, 9):
            # Add some variation to tasks
            variation = f" (Adapter {adapter_id} specific variation)"
            prompt = base_prompt + variation
            
            task = asyncio.create_task(
                self._execute_adapter_task(adapter_id, prompt)
            )
            tasks.append((adapter_id, task))
        
        # Wait for all tasks to complete
        for adapter_id, task in tasks:
            try:
                reward = await task
                task_results[adapter_id] = reward
            except Exception as e:
                logger.error(f"Task failed for adapter {adapter_id}: {e}")
                task_results[adapter_id] = 0.1  # Minimum reward
        
        return task_results
    
    async def _execute_adapter_task(self, adapter_id: int, prompt: str) -> float:
        """Execute a task for a specific adapter and return reward"""
        start_time = time.time()
        
        try:
            # Generate response
            response = await self.vllm_client.generate_with_adapter(prompt, adapter_id)
            
            # Calculate reward based on response quality and speed
            response_time = time.time() - start_time
            
            # Quality score (simple heuristic)
            quality_score = min(1.0, len(response) / 50.0)  # Longer responses get higher scores
            
            # Speed score (faster is better, but not too fast)
            speed_score = max(0.0, 1.0 - response_time / 10.0)
            
            # Combined reward
            reward = (quality_score * 0.7 + speed_score * 0.3) * 0.8 + 0.2  # Scale to 0.2-1.0
            
            # Add some randomness for realistic variation
            reward += random.uniform(-0.05, 0.05)
            reward = max(0.1, min(1.0, reward))
            
            return reward
            
        except Exception as e:
            logger.error(f"Error executing task for adapter {adapter_id}: {e}")
            return 0.1  # Minimum reward
    
    async def close(self):
        """Clean up resources"""
        await self.vllm_client.close()


# Factory functions
def create_team_based_orchestrator(server_url: str = "http://localhost:8001") -> LoRAUpdateOrchestrator:
    """Create orchestrator with team-based strategy"""
    return LoRAUpdateOrchestrator(server_url, UpdateStrategy.TEAM_BASED)


def create_adversarial_orchestrator(server_url: str = "http://localhost:8001") -> LoRAUpdateOrchestrator:
    """Create orchestrator with adversarial strategy"""
    return LoRAUpdateOrchestrator(server_url, UpdateStrategy.ADVERSARIAL)


async def main():
    """Example usage"""
    print("🚀 LoRA Update Strategies Demo")
    print("=" * 50)
    
    # Test team-based strategy
    print("\n📊 Testing Team-Based Update (Reward Partial Sharing)")
    team_orchestrator = create_team_based_orchestrator()
    team_results = await team_orchestrator.run_update_cycle(num_rounds=5)
    
    print(f"Team-based results: {json.dumps(team_results['final_statistics'], indent=2)}")
    await team_orchestrator.close()
    
    # Test adversarial strategy
    print("\n⚔️ Testing Adversarial Update (Competition with Negative Rewards)")
    adv_orchestrator = create_adversarial_orchestrator()
    adv_results = await adv_orchestrator.run_update_cycle(num_rounds=5)
    
    print(f"Adversarial results: {json.dumps(adv_results['final_statistics'], indent=2)}")
    await adv_orchestrator.close()
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())



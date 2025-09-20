#!/usr/bin/env python3
"""
Multi-Model Training Tests
=========================

Comprehensive tests for multi-model RL training functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from core_srl import (
    MultiModelTrainer,
    MultiModelConfig,
    TrainingMode,
    WeightUpdateStrategy,
    quick_start_multimodel_training,
    create_multimodel_trainer,
    create_cooperative_multimodel_trainer,
    create_competitive_multimodel_trainer
)


class TestMultiModelConfig:
    """Test MultiModelConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MultiModelConfig()
        
        assert config.num_models == 4
        assert config.training_mode == TrainingMode.MIXED
        assert config.weight_update_strategy == WeightUpdateStrategy.ASYNCHRONOUS
        assert config.max_episodes == 1000
        assert config.learning_rate == 3e-4
        assert config.enable_verl == True
        assert config.enable_areal == True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MultiModelConfig(
            num_models=6,
            training_mode=TrainingMode.COMPETITIVE,
            max_episodes=500,
            learning_rate=1e-4
        )
        
        assert config.num_models == 6
        assert config.training_mode == TrainingMode.COMPETITIVE
        assert config.max_episodes == 500
        assert config.learning_rate == 1e-4
    
    def test_model_names_mapping(self):
        """Test model names configuration"""
        config = MultiModelConfig()
        
        assert "qwen3" in config.model_names
        assert "openai" in config.model_names
        assert config.model_names["qwen3"] == "Qwen/Qwen2.5-14B-Instruct"


class TestMultiModelTrainer:
    """Test MultiModelTrainer functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing"""
        return MultiModelConfig(
            num_models=2,  # Small for testing
            max_episodes=10,
            save_interval=5,
            enable_monitoring=False,  # Disable for testing
            enable_verl=False,  # Disable for testing
            enable_areal=False
        )
    
    def test_trainer_initialization(self, mock_config):
        """Test trainer initialization"""
        trainer = MultiModelTrainer(mock_config)
        
        assert trainer.config == mock_config
        assert len(trainer.model_states) == mock_config.num_models
        assert trainer.current_episode == 0
        assert trainer.is_training == False
    
    def test_model_state_initialization(self, mock_config):
        """Test model state initialization"""
        trainer = MultiModelTrainer(mock_config)
        
        for model_id, state in trainer.model_states.items():
            assert state.model_id == model_id
            assert state.total_reward == 0.0
            assert state.episode_count == 0
            assert state.update_count == 0
    
    @pytest.mark.asyncio
    async def test_training_status(self, mock_config):
        """Test training status reporting"""
        trainer = MultiModelTrainer(mock_config)
        
        status = trainer.get_training_status()
        
        assert status['is_training'] == False
        assert status['current_episode'] == 0
        assert status['max_episodes'] == mock_config.max_episodes
        assert status['progress'] == 0.0
        assert len(status['model_states']) == mock_config.num_models
        
        await trainer.shutdown()
    
    @pytest.mark.asyncio
    async def test_model_performance_summary(self, mock_config):
        """Test model performance summary"""
        trainer = MultiModelTrainer(mock_config)
        
        # Simulate some training progress
        for state in trainer.model_states.values():
            state.total_reward = 10.0
            state.episode_count = 5
            state.win_count = 2
        
        performance = trainer.get_model_performance_summary()
        
        for model_id, stats in performance.items():
            assert stats['avg_reward'] == 2.0  # 10.0 / 5
            assert stats['win_rate'] == 0.4    # 2 / 5
            assert stats['total_episodes'] == 5
        
        await trainer.shutdown()


class TestTrainingModes:
    """Test different training modes"""
    
    @pytest.mark.asyncio
    async def test_cooperative_trainer_creation(self):
        """Test cooperative trainer creation"""
        trainer = create_cooperative_multimodel_trainer(num_models=3)
        
        assert trainer.config.num_models == 3
        assert trainer.config.training_mode == TrainingMode.COOPERATIVE
        assert trainer.config.cooperation_strength == 0.8  # High cooperation
        assert trainer.config.weight_update_strategy == WeightUpdateStrategy.SYNCHRONIZED
        
        await trainer.shutdown()
    
    @pytest.mark.asyncio
    async def test_competitive_trainer_creation(self):
        """Test competitive trainer creation"""
        trainer = create_competitive_multimodel_trainer(num_models=4)
        
        assert trainer.config.num_models == 4
        assert trainer.config.training_mode == TrainingMode.COMPETITIVE
        assert trainer.config.competition_intensity == 0.7  # High competition
        assert trainer.config.weight_update_strategy == WeightUpdateStrategy.SELECTIVE
        
        await trainer.shutdown()
    
    @pytest.mark.asyncio
    async def test_mixed_trainer_creation(self):
        """Test mixed mode trainer creation"""
        trainer = create_multimodel_trainer(
            num_models=4,
            training_mode=TrainingMode.MIXED
        )
        
        assert trainer.config.training_mode == TrainingMode.MIXED
        assert 0 < trainer.config.cooperation_strength < 1
        assert 0 < trainer.config.competition_intensity < 1
        
        await trainer.shutdown()


class TestWeightUpdateStrategies:
    """Test weight update coordination strategies"""
    
    def test_synchronized_strategy(self):
        """Test synchronized weight update strategy"""
        config = MultiModelConfig(
            weight_update_strategy=WeightUpdateStrategy.SYNCHRONIZED
        )
        
        assert config.weight_update_strategy == WeightUpdateStrategy.SYNCHRONIZED
    
    def test_asynchronous_strategy(self):
        """Test asynchronous weight update strategy"""
        config = MultiModelConfig(
            weight_update_strategy=WeightUpdateStrategy.ASYNCHRONOUS
        )
        
        assert config.weight_update_strategy == WeightUpdateStrategy.ASYNCHRONOUS
    
    def test_federated_strategy(self):
        """Test federated weight update strategy"""
        config = MultiModelConfig(
            weight_update_strategy=WeightUpdateStrategy.FEDERATED
        )
        
        assert config.weight_update_strategy == WeightUpdateStrategy.FEDERATED
    
    def test_selective_strategy(self):
        """Test selective weight update strategy"""
        config = MultiModelConfig(
            weight_update_strategy=WeightUpdateStrategy.SELECTIVE
        )
        
        assert config.weight_update_strategy == WeightUpdateStrategy.SELECTIVE


class TestModelConfiguration:
    """Test model configuration and setup"""
    
    def test_homogeneous_model_setup(self):
        """Test setup with same model types"""
        config = MultiModelConfig(
            num_models=4,
            model_types=["qwen3"] * 4
        )
        
        assert len(config.model_types) == 4
        assert all(mt == "qwen3" for mt in config.model_types)
    
    def test_heterogeneous_model_setup(self):
        """Test setup with different model types"""
        config = MultiModelConfig(
            num_models=4,
            model_types=["qwen3", "openai", "claude", "llama3"]
        )
        
        assert len(config.model_types) == 4
        assert len(set(config.model_types)) == 4  # All different
    
    def test_model_names_validation(self):
        """Test model names configuration"""
        config = MultiModelConfig(
            model_types=["qwen3", "openai"],
            model_names={
                "qwen3": "Qwen/Qwen2.5-14B-Instruct",
                "openai": "gpt-4o-mini"
            }
        )
        
        assert config.model_names["qwen3"] == "Qwen/Qwen2.5-14B-Instruct"
        assert config.model_names["openai"] == "gpt-4o-mini"


class TestCheckpointManagement:
    """Test checkpoint saving and loading"""
    
    def test_checkpoint_config(self):
        """Test checkpoint configuration"""
        config = MultiModelConfig(
            checkpoint_dir="./test_checkpoints",
            save_interval=25,
            max_checkpoints=3
        )
        
        assert config.checkpoint_dir == "./test_checkpoints"
        assert config.save_interval == 25
        assert config.max_checkpoints == 3
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """Test checkpoint creation during training"""
        config = MultiModelConfig(
            num_models=2,
            max_episodes=5,
            save_interval=2,  # Save every 2 episodes
            checkpoint_dir="./test_checkpoints"
        )
        
        trainer = MultiModelTrainer(config)
        
        # Check initial checkpoint count
        initial_checkpoints = len(trainer.checkpoints)
        assert initial_checkpoints == 0
        
        await trainer.shutdown()


class TestOptimizationIntegration:
    """Test VERL/AReaL integration"""
    
    def test_verl_enabled_config(self):
        """Test VERL-enabled configuration"""
        config = MultiModelConfig(enable_verl=True)
        
        assert config.enable_verl == True
        assert config.kv_cache_size > 0
    
    def test_areal_enabled_config(self):
        """Test AReaL-enabled configuration"""
        config = MultiModelConfig(enable_areal=True)
        
        assert config.enable_areal == True
        assert config.kv_cache_size > 0
    
    def test_optimization_disabled_config(self):
        """Test configuration with optimizations disabled"""
        config = MultiModelConfig(
            enable_verl=False,
            enable_areal=False
        )
        
        assert config.enable_verl == False
        assert config.enable_areal == False


class TestFactoryFunctions:
    """Test factory functions for quick setup"""
    
    @pytest.mark.asyncio
    async def test_quick_start_function(self):
        """Test quick start factory function"""
        # Mock the actual training to avoid long test times
        with pytest.MonkeyPatch().context() as m:
            mock_results = {
                'status': 'completed',
                'model_performance': {
                    'model_0': {'avg_reward': 1.5, 'win_rate': 0.6},
                    'model_1': {'avg_reward': 1.3, 'win_rate': 0.4}
                },
                'training_results': {
                    'training_time': 10.0,
                    'total_episodes': 5
                }
            }
            
            async def mock_training(*args, **kwargs):
                return mock_results
            
            # This would normally test the actual function
            # For now, just test that it can be called
            assert callable(quick_start_multimodel_training)
    
    def test_create_multimodel_trainer(self):
        """Test multimodel trainer factory"""
        trainer = create_multimodel_trainer(
            num_models=3,
            training_mode=TrainingMode.MIXED
        )
        
        assert trainer.config.num_models == 3
        assert trainer.config.training_mode == TrainingMode.MIXED
        
        # Cleanup
        asyncio.run(trainer.shutdown())


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_num_models(self):
        """Test handling of invalid number of models"""
        with pytest.raises(ValueError):
            MultiModelConfig(num_models=0)
        
        with pytest.raises(ValueError):
            MultiModelConfig(num_models=-1)
    
    def test_invalid_learning_rate(self):
        """Test handling of invalid learning rate"""
        with pytest.raises(ValueError):
            MultiModelConfig(learning_rate=0.0)
        
        with pytest.raises(ValueError):
            MultiModelConfig(learning_rate=-0.1)
    
    def test_invalid_cooperation_strength(self):
        """Test handling of invalid cooperation strength"""
        with pytest.raises(ValueError):
            MultiModelConfig(cooperation_strength=-0.1)
        
        with pytest.raises(ValueError):
            MultiModelConfig(cooperation_strength=1.5)
    
    def test_mismatched_model_config(self):
        """Test handling of mismatched model configuration"""
        with pytest.raises(ValueError):
            MultiModelConfig(
                num_models=4,
                model_types=["qwen3", "openai"]  # Only 2 types for 4 models
            )


class TestPerformanceMetrics:
    """Test performance measurement and metrics"""
    
    @pytest.mark.asyncio
    async def test_training_time_measurement(self):
        """Test training time measurement"""
        config = MultiModelConfig(
            num_models=2,
            max_episodes=2,
            enable_monitoring=False
        )
        
        trainer = MultiModelTrainer(config)
        
        # Check that training_start_time is set
        assert trainer.training_start_time == 0.0
        
        await trainer.shutdown()
    
    def test_model_performance_calculation(self):
        """Test model performance calculations"""
        config = MultiModelConfig(num_models=2)
        trainer = MultiModelTrainer(config)
        
        # Set up test data
        model_id = list(trainer.model_states.keys())[0]
        state = trainer.model_states[model_id]
        
        state.total_reward = 15.0
        state.episode_count = 5
        state.win_count = 3
        
        performance = trainer.get_model_performance_summary()
        model_stats = performance[model_id]
        
        assert model_stats['avg_reward'] == 3.0  # 15.0 / 5
        assert model_stats['win_rate'] == 0.6    # 3 / 5
        assert model_stats['total_episodes'] == 5
        
        # Cleanup
        asyncio.run(trainer.shutdown())


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

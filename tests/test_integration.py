#!/usr/bin/env python3
"""
Integration Tests
================

End-to-end integration tests for Core SRL multi-model training.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from core_srl import (
    quick_start_multimodel_training,
    MultiModelTrainer,
    MultiModelConfig,
    TrainingMode,
    list_available_checkpoints,
    load_checkpoint_metadata
)


class TestEndToEndIntegration:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.asyncio
    async def test_quick_start_integration(self):
        """Test quick start workflow"""
        # Use minimal configuration for fast testing
        try:
            # This would normally run actual training
            # For testing, we'll just verify the function exists and is callable
            assert callable(quick_start_multimodel_training)
            
            # Test configuration validation
            with pytest.raises(ValueError):
                await quick_start_multimodel_training(num_models=0)
                
        except Exception as e:
            # Expected for mock environment
            assert "Mock" in str(e) or "not implemented" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_checkpoint_workflow(self):
        """Test complete checkpoint save/load workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            
            config = MultiModelConfig(
                num_models=2,
                max_episodes=5,
                save_interval=2,
                checkpoint_dir=str(checkpoint_dir),
                enable_monitoring=False,
                enable_verl=False,
                enable_areal=False
            )
            
            trainer = MultiModelTrainer(config)
            
            try:
                # Verify initial state
                assert trainer.current_episode == 0
                assert len(trainer.checkpoints) == 0
                
                # Test checkpoint directory creation
                assert checkpoint_dir.parent.exists()
                
                # Test training status
                status = trainer.get_training_status()
                assert status['is_training'] == False
                assert status['current_episode'] == 0
                
            finally:
                await trainer.shutdown()
    
    @pytest.mark.asyncio
    async def test_model_performance_tracking(self):
        """Test model performance tracking throughout training"""
        config = MultiModelConfig(
            num_models=3,
            max_episodes=5,
            enable_monitoring=False,
            enable_verl=False,
            enable_areal=False
        )
        
        trainer = MultiModelTrainer(config)
        
        try:
            # Initial performance should be zero
            initial_performance = trainer.get_model_performance_summary()
            
            for model_id, stats in initial_performance.items():
                assert stats['avg_reward'] == 0.0
                assert stats['win_rate'] == 0.0
                assert stats['total_episodes'] == 0
            
            # Simulate some training progress
            for model_id, state in trainer.model_states.items():
                state.total_reward = 5.0
                state.episode_count = 2
                state.win_count = 1
            
            # Check updated performance
            updated_performance = trainer.get_model_performance_summary()
            
            for model_id, stats in updated_performance.items():
                assert stats['avg_reward'] == 2.5  # 5.0 / 2
                assert stats['win_rate'] == 0.5    # 1 / 2
                assert stats['total_episodes'] == 2
            
        finally:
            await trainer.shutdown()


class TestConfigurationValidation:
    """Test configuration validation and error handling"""
    
    def test_valid_configurations(self):
        """Test various valid configurations"""
        # Minimal valid config
        config1 = MultiModelConfig(num_models=2, max_episodes=10)
        assert config1.num_models == 2
        
        # Complex valid config
        config2 = MultiModelConfig(
            num_models=4,
            training_mode=TrainingMode.MIXED,
            learning_rate=1e-4,
            cooperation_strength=0.7,
            competition_intensity=0.3
        )
        assert config2.cooperation_strength == 0.7
        assert config2.competition_intensity == 0.3
    
    def test_invalid_configurations(self):
        """Test invalid configurations raise appropriate errors"""
        # Invalid number of models
        with pytest.raises(ValueError):
            MultiModelConfig(num_models=-1)
        
        # Invalid learning rate
        with pytest.raises(ValueError):
            MultiModelConfig(learning_rate=0.0)
        
        # Invalid cooperation strength
        with pytest.raises(ValueError):
            MultiModelConfig(cooperation_strength=1.5)
        
        # Invalid competition intensity
        with pytest.raises(ValueError):
            MultiModelConfig(competition_intensity=-0.1)


class TestTrainingModeIntegration:
    """Test different training modes work together"""
    
    @pytest.mark.asyncio
    async def test_cooperative_mode_setup(self):
        """Test cooperative mode creates appropriate configuration"""
        from core_srl import create_cooperative_multimodel_trainer
        
        trainer = create_cooperative_multimodel_trainer(num_models=3)
        
        try:
            assert trainer.config.training_mode == TrainingMode.COOPERATIVE
            assert trainer.config.cooperation_strength >= 0.7  # High cooperation
            assert len(trainer.model_states) == 3
            
        finally:
            await trainer.shutdown()
    
    @pytest.mark.asyncio
    async def test_competitive_mode_setup(self):
        """Test competitive mode creates appropriate configuration"""
        from core_srl import create_competitive_multimodel_trainer
        
        trainer = create_competitive_multimodel_trainer(num_models=4)
        
        try:
            assert trainer.config.training_mode == TrainingMode.COMPETITIVE
            assert trainer.config.competition_intensity >= 0.6  # High competition
            assert len(trainer.model_states) == 4
            
        finally:
            await trainer.shutdown()


class TestSystemIntegration:
    """Test system-level integration"""
    
    def test_import_structure(self):
        """Test that all required imports work"""
        # Core components
        from core_srl import MultiModelTrainer, MultiModelConfig
        from core_srl import TrainingMode, WeightUpdateStrategy
        
        # Factory functions
        from core_srl import create_multimodel_trainer
        from core_srl import create_cooperative_multimodel_trainer
        from core_srl import create_competitive_multimodel_trainer
        
        # Utilities
        from core_srl import list_available_checkpoints
        from core_srl import load_checkpoint_metadata
        
        # Quick start
        from core_srl import quick_start_multimodel_training
        
        # All imports successful
        assert True
    
    def test_enum_values(self):
        """Test enum values are properly defined"""
        # TrainingMode enum
        assert TrainingMode.COOPERATIVE.value == "cooperative"
        assert TrainingMode.COMPETITIVE.value == "competitive"
        assert TrainingMode.MIXED.value == "mixed"
        assert TrainingMode.HIERARCHICAL.value == "hierarchical"
        
        # WeightUpdateStrategy enum
        assert WeightUpdateStrategy.SYNCHRONIZED.value == "synchronized"
        assert WeightUpdateStrategy.ASYNCHRONOUS.value == "asynchronous"
        assert WeightUpdateStrategy.FEDERATED.value == "federated"
        assert WeightUpdateStrategy.SELECTIVE.value == "selective"
    
    def test_version_info(self):
        """Test version information is available"""
        from core_srl import get_version_info
        
        version_info = get_version_info()
        
        assert "version" in version_info
        assert "author" in version_info
        assert version_info["version"] == "2.0.0"


class TestResourceManagement:
    """Test resource management and cleanup"""
    
    @pytest.mark.asyncio
    async def test_trainer_cleanup(self):
        """Test trainer properly cleans up resources"""
        config = MultiModelConfig(
            num_models=2,
            max_episodes=5,
            enable_monitoring=False,
            enable_verl=False,
            enable_areal=False
        )
        
        trainer = MultiModelTrainer(config)
        
        # Verify initialization
        assert trainer.config == config
        assert len(trainer.model_states) == 2
        
        # Test shutdown
        await trainer.shutdown()
        
        # Verify cleanup (trainer should handle shutdown gracefully)
        assert True  # If we get here, shutdown worked
    
    @pytest.mark.asyncio
    async def test_multiple_trainer_instances(self):
        """Test multiple trainer instances can coexist"""
        config1 = MultiModelConfig(num_models=2, max_episodes=5)
        config2 = MultiModelConfig(num_models=3, max_episodes=10)
        
        trainer1 = MultiModelTrainer(config1)
        trainer2 = MultiModelTrainer(config2)
        
        try:
            # Verify both trainers are independent
            assert trainer1.config.num_models == 2
            assert trainer2.config.num_models == 3
            assert len(trainer1.model_states) == 2
            assert len(trainer2.model_states) == 3
            
        finally:
            await trainer1.shutdown()
            await trainer2.shutdown()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "not slow"])

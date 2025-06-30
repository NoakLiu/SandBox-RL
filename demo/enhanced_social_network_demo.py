"""
Enhanced Social Network Demo with Monitoring Integration
=======================================================

This demo showcases the comprehensive monitoring capabilities for social network analysis
with real-time metrics tracking using WanDB and TensorBoard.
"""

import sys
import os
import time
import json
import random
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandgraph.core.llm_interface import create_shared_llm_manager
from sandgraph.core.sg_workflow import SG_Workflow, WorkflowMode, EnhancedWorkflowNode
from sandgraph.core.workflow import NodeType
from sandgraph.core.rl_algorithms import RLTrainer, RLConfig
from sandgraph.core.monitoring import (
    SocialNetworkMonitor, 
    MonitoringConfig, 
    SocialNetworkMetrics, 
    MetricsCollector,
    create_monitor
)
from sandgraph.sandbox_implementations import SocialNetworkSandbox


class EnhancedSocialNetworkDemo:
    """Enhanced social network demo with comprehensive monitoring"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.monitor = None
        self.workflow = None
        self.rl_trainer = None
        self.sandbox = None
        
        # Initialize components
        self._setup_monitoring()
        self._setup_workflow()
        
    def _setup_monitoring(self):
        """Setup monitoring system"""
        monitor_config = MonitoringConfig(
            enable_wandb=self.config.get("enable_wandb", True),
            enable_tensorboard=self.config.get("enable_tensorboard", True),
            wandb_project_name=self.config.get("wandb_project", "sandgraph-enhanced-social"),
            wandb_run_name=self.config.get("wandb_run_name", f"social_demo_{int(time.time())}"),
            tensorboard_log_dir=self.config.get("tensorboard_log_dir", "./logs/enhanced_social"),
            log_file_path=self.config.get("log_file_path", "./logs/enhanced_social_metrics.json"),
            metrics_sampling_interval=self.config.get("sampling_interval", 2.0),
            engagement_rate_threshold=self.config.get("engagement_threshold", 0.15),
            user_growth_threshold=self.config.get("growth_threshold", 0.08)
        )
        
        self.monitor = create_monitor(monitor_config)
        
        # Add alert callback
        self.monitor.add_alert_callback(self._handle_alert)
        
        print("✅ Monitoring system initialized")
        
    def _setup_workflow(self):
        """Setup the social network workflow"""
        # Create LLM manager
        model_name = self.config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.2")
        llm_manager = create_shared_llm_manager(model_name)
        
        # Create workflow
        self.workflow = SG_Workflow("enhanced_social_network", WorkflowMode.TRADITIONAL, llm_manager)
        
        # Create sandbox
        initial_users = self.config.get("initial_users", 50)
        self.sandbox = SocialNetworkSandbox(initial_users=initial_users)
        
        # Create RL trainer
        rl_config = RLConfig(
            algorithm="PPO",
            learning_rate=0.001,
            batch_size=32,
            gamma=0.99
        )
        self.rl_trainer = RLTrainer(rl_config, llm_manager)
        
        # Add nodes to workflow
        env_node = EnhancedWorkflowNode("social_env", NodeType.SANDBOX, sandbox=self.sandbox)
        decision_node = EnhancedWorkflowNode("decision_maker", NodeType.LLM, 
                                           llm_func=self._decision_maker_llm,
                                           metadata={"role": "Social Network Analyst"})
        optimizer_node = EnhancedWorkflowNode("optimizer", NodeType.RL, 
                                            rl_trainer=self.rl_trainer)
        
        self.workflow.add_node(env_node)
        self.workflow.add_node(decision_node)
        self.workflow.add_node(optimizer_node)
        
        # Connect nodes
        self.workflow.add_edge("social_env", "decision_maker")
        self.workflow.add_edge("decision_maker", "optimizer")
        self.workflow.add_edge("optimizer", "social_env")
        
        print("✅ Workflow system initialized")
        
    def _decision_maker_llm(self, prompt: str) -> str:
        """Enhanced decision maker LLM function"""
        print(f"🤖 Decision maker LLM called with prompt: {prompt[:100]}...")
        
        # Get current state from sandbox
        current_state = self.sandbox.case_generator()
        
        # Create comprehensive decision prompt
        enhanced_prompt = self._create_enhanced_prompt(current_state, prompt)
        
        # Generate decision using LLM manager
        response = self.workflow.llm_manager.generate_for_node("decision_maker", enhanced_prompt)
        
        print(f"🤖 Decision maker response: {response.text[:100]}...")
        return response.text
    
    def _create_enhanced_prompt(self, state: Dict[str, Any], base_prompt: str) -> str:
        """Create enhanced prompt with monitoring context"""
        
        # Extract current metrics
        network_state = state.get("state", {}).get("network_state", {})
        user_behavior = state.get("state", {}).get("user_behavior", {})
        content_metrics = state.get("state", {}).get("content_metrics", {})
        network_dynamics = state.get("state", {}).get("network_dynamics", {})
        
        # Get monitoring summary if available
        monitoring_context = ""
        if self.monitor:
            summary = self.monitor.get_metrics_summary()
            if summary:
                trends = summary.get("trends", {})
                alerts = summary.get("alerts", [])
                
                monitoring_context = f"""
Monitoring Context:
- Recent Engagement Trend: {trends.get('engagement_rate', [])[-3:] if trends.get('engagement_rate') else 'N/A'}
- Recent Growth Trend: {trends.get('user_growth_rate', [])[-3:] if trends.get('user_growth_rate') else 'N/A'}
- Recent Quality Trend: {trends.get('content_quality', [])[-3:] if trends.get('content_quality') else 'N/A'}
- Active Alerts: {len(alerts)} alerts in the last 10 updates
"""
        
        enhanced_prompt = f"""
{base_prompt}

Current Network State:
- Total Users: {network_state.get('total_users', 0)}
- Active Users: {user_behavior.get('active_users', 0)}
- Engagement Rate: {user_behavior.get('engagement_rate', 0):.3f}
- Content Quality: {content_metrics.get('quality_score', 0):.3f}
- Network Density: {network_state.get('network_density', 0):.3f}

Network Dynamics:
- Mood: {network_dynamics.get('mood', 0):.2f}
- Competition Level: {network_dynamics.get('competition_level', 0):.2f}
- Innovation Rate: {network_dynamics.get('innovation_rate', 0):.2f}
- Crisis Level: {network_dynamics.get('crisis_level', 0):.2f}

{monitoring_context}

Based on this comprehensive view, provide strategic recommendations to improve:
1. User engagement and retention
2. Content quality and diversity
3. Network growth and community building
4. Overall platform health

Focus on actionable insights that can be implemented immediately.
"""
        
        return enhanced_prompt
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle monitoring alerts"""
        print(f"🚨 ALERT [{alert['severity'].upper()}]: {alert['message']}")
        
        # Log alert to file
        alert_log_path = "./logs/alerts.json"
        os.makedirs(os.path.dirname(alert_log_path), exist_ok=True)
        
        try:
            alerts = []
            if os.path.exists(alert_log_path):
                with open(alert_log_path, "r") as f:
                    alerts = json.load(f)
            
            alerts.append(alert)
            
            with open(alert_log_path, "w") as f:
                json.dump(alerts, f, indent=2)
        except Exception as e:
            print(f"Failed to log alert: {e}")
    
    def _collect_metrics(self) -> SocialNetworkMetrics:
        """Collect comprehensive metrics from current state"""
        if not self.sandbox:
            return SocialNetworkMetrics()
        
        # Get current state
        state = self.sandbox.case_generator()
        state_data = state.get("state", {})
        
        # Extract data
        network_state = state_data.get("network_state", {})
        user_behavior = state_data.get("user_behavior", {})
        content_metrics = state_data.get("content_metrics", {})
        network_dynamics = state_data.get("network_dynamics", {})
        
        # Create metrics using collector
        metrics = MetricsCollector.create_social_network_metrics(
            network_state=network_state,
            user_behavior=user_behavior,
            content_metrics=content_metrics,
            additional_data={
                "network_growth_rate": network_dynamics.get("growth_rate", 0),
                "avg_influence_score": network_dynamics.get("avg_influence", 0),
                "viral_spread_rate": network_dynamics.get("viral_rate", 0),
                "total_communities": network_state.get("communities", 0),
                "avg_community_size": network_state.get("avg_community_size", 0),
                "community_engagement": network_state.get("community_engagement", 0),
                "cross_community_interactions": network_state.get("cross_interactions", 0),
                "influencer_count": network_state.get("influencer_count", 0),
                "information_cascade_depth": network_dynamics.get("cascade_depth", 0),
                "response_time_avg": 1.5,  # Simulated
                "error_rate": 0.02,  # Simulated
                "system_uptime": time.time() - self.monitor.start_time if self.monitor else 0
            }
        )
        
        return metrics
    
    def run_simulation(self, steps: int = 10, delay: float = 2.0):
        """Run the enhanced social network simulation"""
        print(f"🚀 Starting Enhanced Social Network Simulation ({steps} steps)")
        print("=" * 60)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        try:
            for step in range(steps):
                print(f"\n📊 Step {step + 1}/{steps}")
                print("-" * 40)
                
                # Execute workflow step
                start_time = time.time()
                result = self.workflow.execute_full_workflow()
                execution_time = time.time() - start_time
                
                # Collect and update metrics
                metrics = self._collect_metrics()
                self.monitor.update_metrics(metrics)
                
                # Update RL weights
                if self.rl_trainer:
                    self.rl_trainer.update_policy()
                
                # Print step summary
                self._print_step_summary(step + 1, metrics, execution_time, result)
                
                # Delay between steps
                if step < steps - 1:  # Don't delay after last step
                    time.sleep(delay)
                    
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during simulation: {e}")
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Export final results
            self._export_results()
            
            print("\n✅ Simulation completed!")
    
    def _print_step_summary(self, step: int, metrics: SocialNetworkMetrics, 
                           execution_time: float, result: Any):
        """Print summary for current step"""
        print(f"⏱️  Execution time: {execution_time:.2f}s")
        print(f"👥 Users: {metrics.total_users} (Active: {metrics.active_users})")
        print(f"📈 Engagement: {metrics.engagement_rate:.3f}")
        print(f"🎯 Quality: {metrics.content_quality_score:.3f}")
        print(f"🌐 Network Density: {metrics.network_density:.3f}")
        print(f"📊 Viral Posts: {metrics.viral_posts}")
        print(f"🔥 Trending Topics: {metrics.trending_topics}")
        
        # Print alerts if any
        if self.monitor.alerts:
            recent_alerts = self.monitor.alerts[-3:]  # Last 3 alerts
            for alert in recent_alerts:
                print(f"🚨 {alert['message']}")
    
    def _export_results(self):
        """Export simulation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export metrics
        metrics_file = f"./logs/enhanced_social_metrics_{timestamp}.json"
        self.monitor.export_metrics(metrics_file, "json")
        
        # Export summary report
        summary_file = f"./logs/enhanced_social_summary_{timestamp}.json"
        summary = self.monitor.get_metrics_summary()
        
        with open(summary_file, "w") as f:
            json.dump({
                "simulation_config": self.config,
                "final_summary": summary,
                "total_steps": len(self.monitor.metrics_history),
                "total_alerts": len(self.monitor.alerts),
                "simulation_duration": time.time() - self.monitor.start_time
            }, f, indent=2)
        
        print(f"📁 Results exported to:")
        print(f"   - Metrics: {metrics_file}")
        print(f"   - Summary: {summary_file}")
        
        # Print final statistics
        if self.monitor.metrics_history:
            final_metrics = self.monitor.metrics_history[-1]
            print(f"\n📊 Final Statistics:")
            print(f"   - Total Users: {final_metrics.total_users}")
            print(f"   - Engagement Rate: {final_metrics.engagement_rate:.3f}")
            print(f"   - Content Quality: {final_metrics.content_quality_score:.3f}")
            print(f"   - Network Density: {final_metrics.network_density:.3f}")
            print(f"   - Total Alerts: {len(self.monitor.alerts)}")


def main():
    """Main function to run the enhanced social network demo"""
    parser = argparse.ArgumentParser(description="Enhanced Social Network Demo with Monitoring")
    
    parser.add_argument("--steps", type=int, default=10, 
                       help="Number of simulation steps")
    parser.add_argument("--delay", type=float, default=2.0,
                       help="Delay between steps in seconds")
    parser.add_argument("--initial-users", type=int, default=50,
                       help="Initial number of users")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="LLM model to use")
    parser.add_argument("--enable-wandb", action="store_true", default=True,
                       help="Enable WanDB logging")
    parser.add_argument("--enable-tensorboard", action="store_true", default=True,
                       help="Enable TensorBoard logging")
    parser.add_argument("--wandb-project", type=str, default="sandgraph-enhanced-social",
                       help="WanDB project name")
    parser.add_argument("--tensorboard-dir", type=str, default="./logs/enhanced_social",
                       help="TensorBoard log directory")
    parser.add_argument("--sampling-interval", type=float, default=2.0,
                       help="Metrics sampling interval")
    parser.add_argument("--engagement-threshold", type=float, default=0.15,
                       help="Engagement rate alert threshold")
    parser.add_argument("--growth-threshold", type=float, default=0.08,
                       help="User growth rate alert threshold")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "steps": args.steps,
        "delay": args.delay,
        "initial_users": args.initial_users,
        "model_name": args.model,
        "enable_wandb": args.enable_wandb,
        "enable_tensorboard": args.enable_tensorboard,
        "wandb_project": args.wandb_project,
        "tensorboard_log_dir": args.tensorboard_dir,
        "sampling_interval": args.sampling_interval,
        "engagement_threshold": args.engagement_threshold,
        "growth_threshold": args.growth_threshold
    }
    
    # Create and run demo
    demo = EnhancedSocialNetworkDemo(config)
    demo.run_simulation(steps=args.steps, delay=args.delay)


if __name__ == "__main__":
    main() 
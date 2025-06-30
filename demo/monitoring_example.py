"""
Social Network Monitoring Example
================================

This example demonstrates how to use the comprehensive monitoring and visualization
system for social network analysis with WanDB and TensorBoard integration.
"""

import sys
import os
import time
import random
from typing import Dict, Any

# Add the parent directory to the path to import sandgraph modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandgraph.core.monitoring import (
    SocialNetworkMonitor, 
    MonitoringConfig, 
    SocialNetworkMetrics, 
    MetricsCollector,
    create_monitor
)
from sandgraph.core.visualization import (
    SocialNetworkVisualizer,
    create_visualizer,
    quick_visualization
)


def generate_sample_metrics(step: int) -> SocialNetworkMetrics:
    """Generate sample metrics for demonstration"""
    
    # Simulate realistic social network metrics
    base_users = 100
    growth_factor = 1 + (step * 0.05)  # 5% growth per step
    total_users = int(base_users * growth_factor)
    active_users = int(total_users * random.uniform(0.6, 0.9))
    
    # Engagement metrics
    engagement_rate = random.uniform(0.1, 0.3)
    total_posts = int(total_users * random.uniform(0.5, 2.0))
    total_likes = int(total_posts * random.uniform(5, 20))
    total_comments = int(total_posts * random.uniform(1, 5))
    total_shares = int(total_posts * random.uniform(0.5, 3))
    
    # Content metrics
    viral_posts = int(total_posts * random.uniform(0.01, 0.1))
    trending_topics = random.randint(1, 10)
    content_quality = random.uniform(0.4, 0.9)
    user_satisfaction = random.uniform(0.5, 0.95)
    content_diversity = random.uniform(0.3, 0.8)
    controversy_level = random.uniform(0.0, 0.4)
    
    # Network metrics
    network_density = random.uniform(0.1, 0.5)
    avg_followers = random.uniform(10, 50)
    avg_following = random.uniform(15, 40)
    clustering_coefficient = random.uniform(0.2, 0.6)
    network_growth_rate = random.uniform(0.02, 0.1)
    
    # Community metrics
    total_communities = random.randint(3, 15)
    avg_community_size = random.uniform(5, 25)
    community_engagement = random.uniform(0.3, 0.8)
    cross_community_interactions = random.randint(10, 100)
    
    # Influence metrics
    influencer_count = int(total_users * random.uniform(0.05, 0.15))
    avg_influence_score = random.uniform(0.3, 0.8)
    viral_spread_rate = random.uniform(0.1, 0.5)
    information_cascade_depth = random.uniform(1.5, 4.0)
    
    # Performance metrics
    response_time_avg = random.uniform(0.5, 3.0)
    error_rate = random.uniform(0.01, 0.05)
    system_uptime = step * 60  # Simulate 1 minute per step
    
    # User segments
    user_segments = {
        "influencers": int(total_users * 0.1),
        "creators": int(total_users * 0.2),
        "consumers": int(total_users * 0.6),
        "lurkers": int(total_users * 0.1)
    }
    
    # Activity patterns
    activity_patterns = {
        "morning": random.uniform(0.2, 0.4),
        "afternoon": random.uniform(0.3, 0.5),
        "evening": random.uniform(0.4, 0.6),
        "night": random.uniform(0.1, 0.3)
    }
    
    # Content preferences
    content_preferences = {
        "tech": random.uniform(0.2, 0.4),
        "entertainment": random.uniform(0.3, 0.5),
        "news": random.uniform(0.1, 0.3),
        "lifestyle": random.uniform(0.2, 0.4)
    }
    
    return SocialNetworkMetrics(
        total_users=total_users,
        active_users=active_users,
        new_users=int(total_users * 0.1),
        churned_users=int(total_users * 0.02),
        user_growth_rate=network_growth_rate,
        total_posts=total_posts,
        total_likes=total_likes,
        total_comments=total_comments,
        total_shares=total_shares,
        engagement_rate=engagement_rate,
        avg_session_time=random.uniform(15, 45),
        bounce_rate=random.uniform(0.1, 0.3),
        retention_rate=random.uniform(0.6, 0.9),
        viral_posts=viral_posts,
        trending_topics=trending_topics,
        content_quality_score=content_quality,
        user_satisfaction_score=user_satisfaction,
        content_diversity_score=content_diversity,
        controversy_level=controversy_level,
        network_density=network_density,
        avg_followers=avg_followers,
        avg_following=avg_following,
        clustering_coefficient=clustering_coefficient,
        network_growth_rate=network_growth_rate,
        total_communities=total_communities,
        avg_community_size=avg_community_size,
        community_engagement=community_engagement,
        cross_community_interactions=cross_community_interactions,
        influencer_count=influencer_count,
        avg_influence_score=avg_influence_score,
        viral_spread_rate=viral_spread_rate,
        information_cascade_depth=information_cascade_depth,
        user_segments=user_segments,
        activity_patterns=activity_patterns,
        content_preferences=content_preferences,
        peak_activity_hours=[9, 12, 18, 21],
        daily_active_users=[active_users] * 7,
        weekly_growth_trend=[network_growth_rate] * 4,
        response_time_avg=response_time_avg,
        error_rate=error_rate,
        system_uptime=system_uptime
    )


def run_monitoring_example():
    """Run a complete monitoring example"""
    
    print("🚀 Social Network Monitoring Example")
    print("=" * 50)
    
    # 1. Setup monitoring configuration
    config = MonitoringConfig(
        enable_wandb=False,  # Set to True if you have WanDB configured
        enable_tensorboard=True,
        enable_console_logging=True,
        enable_file_logging=True,
        wandb_project_name="sandgraph-monitoring-example",
        tensorboard_log_dir="./logs/monitoring_example",
        log_file_path="./logs/monitoring_example_metrics.json",
        metrics_sampling_interval=1.0,
        engagement_rate_threshold=0.15,
        user_growth_threshold=0.08,
        error_rate_threshold=0.05,
        response_time_threshold=2.0
    )
    
    # 2. Create monitor
    monitor = create_monitor(config)
    
    # 3. Add alert callback
    def alert_handler(alert):
        print(f"🚨 ALERT: {alert['message']}")
    
    monitor.add_alert_callback(alert_handler)
    
    # 4. Start monitoring
    monitor.start_monitoring()
    
    print("✅ Monitoring started")
    print("📊 Generating sample metrics...")
    
    # 5. Generate and update metrics
    metrics_history = []
    for step in range(20):  # 20 simulation steps
        print(f"Step {step + 1}/20")
        
        # Generate sample metrics
        metrics = generate_sample_metrics(step)
        metrics_history.append(metrics)
        
        # Update monitor
        monitor.update_metrics(metrics)
        
        # Small delay to simulate real-time data
        time.sleep(0.5)
    
    # 6. Stop monitoring
    monitor.stop_monitoring()
    
    print("✅ Monitoring completed")
    
    # 7. Export results
    print("📁 Exporting results...")
    monitor.export_metrics("./logs/monitoring_example_export.json", "json")
    
    # 8. Create visualizations
    print("📊 Creating visualizations...")
    visualizer = create_visualizer("./visualizations/monitoring_example")
    
    # Create different types of visualizations
    dashboard_path = visualizer.create_dashboard(metrics_history)
    interactive_path = visualizer.create_interactive_dashboard(metrics_history)
    trend_path = visualizer.create_trend_analysis(metrics_history)
    heatmap_path = visualizer.create_heatmap(metrics_history)
    
    # Export comprehensive report
    report_files = visualizer.export_visualization_report(metrics_history)
    
    print("✅ Visualizations created:")
    for viz_type, path in report_files.items():
        print(f"   - {viz_type}: {path}")
    
    # 9. Print summary
    summary = monitor.get_metrics_summary()
    print(f"\n📊 Final Summary:")
    print(f"   - Total data points: {len(metrics_history)}")
    print(f"   - Total alerts: {len(monitor.alerts)}")
    print(f"   - Final users: {metrics_history[-1].total_users}")
    print(f"   - Final engagement: {metrics_history[-1].engagement_rate:.3f}")
    print(f"   - Final quality: {metrics_history[-1].content_quality_score:.3f}")
    
    print("\n🎉 Example completed successfully!")
    print("📖 Check the generated files in ./logs/ and ./visualizations/ directories")


def demonstrate_metrics_collector():
    """Demonstrate the MetricsCollector functionality"""
    
    print("\n🔧 MetricsCollector Demonstration")
    print("=" * 40)
    
    # Sample network state
    network_state = {
        "users": [
            {"id": "user1", "activity_level": 0.8, "followers": ["user2", "user3"], "following": ["user2"]},
            {"id": "user2", "activity_level": 0.6, "followers": ["user1"], "following": ["user1", "user3"]},
            {"id": "user3", "activity_level": 0.9, "followers": ["user1"], "following": ["user2"]}
        ],
        "connections": [
            {"from": "user1", "to": "user2"},
            {"from": "user2", "to": "user3"},
            {"from": "user1", "to": "user3"}
        ]
    }
    
    # Sample user behavior
    user_behavior = {
        "total_users": 3,
        "active_users": 2,
        "posts_created": 15,
        "likes_given": 45,
        "comments_made": 12,
        "shares_made": 8,
        "avg_session_time": 25.5,
        "bounce_rate": 0.15,
        "retention_rate": 0.85
    }
    
    # Sample content metrics
    content_metrics = {
        "viral_posts": 2,
        "trending_topics": 5,
        "quality_score": 0.75,
        "satisfaction_score": 0.82,
        "diversity_score": 0.68,
        "controversy_level": 0.12
    }
    
    # Create comprehensive metrics
    metrics = MetricsCollector.create_social_network_metrics(
        network_state=network_state,
        user_behavior=user_behavior,
        content_metrics=content_metrics,
        additional_data={
            "network_growth_rate": 0.1,
            "avg_influence_score": 0.6,
            "viral_spread_rate": 0.3
        }
    )
    
    print("📊 Generated Metrics:")
    print(f"   - Total Users: {metrics.total_users}")
    print(f"   - Active Users: {metrics.active_users}")
    print(f"   - Engagement Rate: {metrics.engagement_rate:.3f}")
    print(f"   - Network Density: {metrics.network_density:.3f}")
    print(f"   - Content Quality: {metrics.content_quality_score:.3f}")
    print(f"   - Viral Posts: {metrics.viral_posts}")
    print(f"   - Avg Followers: {metrics.avg_followers:.1f}")


def demonstrate_quick_visualization():
    """Demonstrate quick visualization functionality"""
    
    print("\n⚡ Quick Visualization Demonstration")
    print("=" * 40)
    
    # Generate some sample metrics
    metrics_history = []
    for step in range(10):
        metrics = generate_sample_metrics(step)
        metrics_history.append(metrics)
    
    # Use quick visualization
    output_dir = "./visualizations/quick_demo"
    report_files = quick_visualization(metrics_history, output_dir)
    
    print("📊 Quick visualization completed:")
    for viz_type, path in report_files.items():
        print(f"   - {viz_type}: {path}")


if __name__ == "__main__":
    # Run the main example
    run_monitoring_example()
    
    # Demonstrate additional features
    demonstrate_metrics_collector()
    demonstrate_quick_visualization()
    
    print("\n🎯 All demonstrations completed!")
    print("📖 Check the generated files to see the monitoring and visualization results.") 
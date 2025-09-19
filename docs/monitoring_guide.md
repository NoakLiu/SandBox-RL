# Social Network Monitoring & Visualization Guide

## Overview

The Sandbox-RLX social network monitoring system provides comprehensive real-time tracking, analysis, and visualization capabilities for social network metrics. It integrates with popular monitoring tools like WanDB and TensorBoard for enhanced data analysis and visualization.

## Features

### 🔍 Real-time Monitoring
- **Live Metrics Tracking**: Monitor social network metrics in real-time
- **Alert System**: Configurable alerts for critical thresholds
- **Multi-backend Support**: WanDB, TensorBoard, file logging, and console output
- **Thread-safe Operations**: Asynchronous metrics collection and processing

### 📊 Comprehensive Metrics
- **User Metrics**: Growth, activity, engagement, retention
- **Content Metrics**: Quality, diversity, viral spread, trending topics
- **Network Metrics**: Density, clustering, influence, community structure
- **Performance Metrics**: Response times, error rates, system health
- **Behavioral Metrics**: User segments, activity patterns, preferences

### 📈 Advanced Visualization
- **Static Dashboards**: Comprehensive multi-panel charts using Matplotlib
- **Interactive Dashboards**: Dynamic plots using Plotly
- **Trend Analysis**: Time-series analysis and pattern recognition
- **Correlation Heatmaps**: Metric relationship analysis
- **Export Capabilities**: JSON, CSV, and image formats

## Quick Start

### 1. Basic Monitoring Setup

```python
from sandbox_rl.core.monitoring import create_monitor, MonitoringConfig

# Create monitoring configuration
config = MonitoringConfig(
    enable_wandb=True,
    enable_tensorboard=True,
    wandb_project_name="my-social-network",
    tensorboard_log_dir="./logs/tensorboard"
)

# Create monitor
monitor = create_monitor(config)

# Start monitoring
monitor.start_monitoring()
```

### 2. Collecting Metrics

```python
from sandbox_rl.core.monitoring import SocialNetworkMetrics, MetricsCollector

# Method 1: Create metrics manually
metrics = SocialNetworkMetrics(
    total_users=1000,
    active_users=750,
    engagement_rate=0.25,
    content_quality_score=0.8
)

# Method 2: Use MetricsCollector for automatic calculation
metrics = MetricsCollector.create_social_network_metrics(
    network_state=network_data,
    user_behavior=behavior_data,
    content_metrics=content_data
)

# Update monitor
monitor.update_metrics(metrics)
```

### 3. Creating Visualizations

```python
from sandbox_rl.core.visualization import create_visualizer

# Create visualizer
visualizer = create_visualizer("./visualizations")

# Generate different types of visualizations
dashboard_path = visualizer.create_dashboard(metrics_history)
interactive_path = visualizer.create_interactive_dashboard(metrics_history)
trend_path = visualizer.create_trend_analysis(metrics_history)
heatmap_path = visualizer.create_heatmap(metrics_history)

# Export comprehensive report
report_files = visualizer.export_visualization_report(metrics_history)
```

## Detailed Usage

### Monitoring Configuration

The `MonitoringConfig` class allows fine-grained control over the monitoring system:

```python
config = MonitoringConfig(
    # General settings
    enable_wandb=True,
    enable_tensorboard=True,
    enable_console_logging=True,
    enable_file_logging=True,
    
    # WanDB settings
    wandb_project_name="sandgraph-social-network",
    wandb_entity="your-username",
    wandb_run_name="experiment-001",
    
    # TensorBoard settings
    tensorboard_log_dir="./logs/tensorboard",
    
    # File logging settings
    log_file_path="./logs/metrics.json",
    
    # Sampling settings
    metrics_sampling_interval=1.0,  # seconds
    history_window_size=1000,       # data points to keep in memory
    
    # Alert thresholds
    engagement_rate_threshold=0.15,
    user_growth_threshold=0.08,
    error_rate_threshold=0.05,
    response_time_threshold=2.0
)
```

### Metrics Collection

The `MetricsCollector` provides helper methods for calculating metrics from raw data:

```python
# Calculate network-level metrics
network_metrics = MetricsCollector.calculate_network_metrics(network_state)

# Calculate engagement metrics
engagement_metrics = MetricsCollector.calculate_engagement_metrics(user_behavior)

# Calculate content metrics
content_metrics = MetricsCollector.calculate_content_metrics(content_data)

# Create comprehensive metrics object
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
```

### Alert System

Configure custom alert handlers for real-time notifications:

```python
def custom_alert_handler(alert):
    """Custom alert handler"""
    print(f"🚨 {alert['severity'].upper()}: {alert['message']}")
    
    # Send email, Slack notification, etc.
    if alert['severity'] == 'error':
        send_emergency_notification(alert)

# Add alert callback
monitor.add_alert_callback(custom_alert_handler)
```

### Advanced Visualization

Create custom visualizations with the `SocialNetworkVisualizer`:

```python
visualizer = SocialNetworkVisualizer("./output")

# Create specific chart types
dashboard_path = visualizer.create_dashboard(metrics_history)
interactive_path = visualizer.create_interactive_dashboard(metrics_history)
trend_path = visualizer.create_trend_analysis(metrics_history)
heatmap_path = visualizer.create_heatmap(metrics_history)

# Export comprehensive report
report_files = visualizer.export_visualization_report(metrics_history)
```

## Integration with Sandbox-RLX Workflow

### Enhanced Social Network Demo

Use the enhanced demo for comprehensive monitoring:

```bash
# Run enhanced social network demo with monitoring
python demo/enhanced_social_network_demo.py \
    --steps 20 \
    --initial-users 100 \
    --enable-wandb \
    --enable-tensorboard \
    --wandb-project "sandgraph-enhanced-social" \
    --tensorboard-dir "./logs/enhanced_social"
```

### Custom Integration

Integrate monitoring into your own workflows:

```python
from sandbox_rl.core.monitoring import create_monitor, SocialNetworkMetrics
from sandbox_rl.core.visualization import create_visualizer

class MySocialNetworkWorkflow:
    def __init__(self):
        # Setup monitoring
        self.monitor = create_monitor()
        self.visualizer = create_visualizer()
        self.metrics_history = []
        
    def run_simulation(self, steps):
        self.monitor.start_monitoring()
        
        for step in range(steps):
            # Execute workflow step
            result = self.execute_workflow_step()
            
            # Collect metrics
            metrics = self.collect_metrics(result)
            self.metrics_history.append(metrics)
            
            # Update monitor
            self.monitor.update_metrics(metrics)
        
        # Stop monitoring and create visualizations
        self.monitor.stop_monitoring()
        self.visualizer.export_visualization_report(self.metrics_history)
```

## WanDB Integration

### Setup WanDB

1. Install WanDB:
```bash
pip install wandb
```

2. Login to WanDB:
```bash
wandb login
```

3. Configure monitoring:
```python
config = MonitoringConfig(
    enable_wandb=True,
    wandb_project_name="sandgraph-social-network",
    wandb_entity="your-username"
)
```

### WanDB Dashboard

The monitoring system automatically logs metrics to WanDB with organized categories:

- **Users**: Total users, active users, growth rates
- **Engagement**: Posts, likes, comments, shares, engagement rates
- **Content**: Quality scores, viral posts, trending topics
- **Network**: Density, clustering, influence metrics
- **Performance**: Response times, error rates, system health

## TensorBoard Integration

### Setup TensorBoard

1. Install TensorBoard:
```bash
pip install tensorboard
```

2. Configure monitoring:
```python
config = MonitoringConfig(
    enable_tensorboard=True,
    tensorboard_log_dir="./logs/tensorboard"
)
```

### Viewing TensorBoard

1. Start TensorBoard:
```bash
tensorboard --logdir=./logs/tensorboard
```

2. Open browser at `http://localhost:6006`

### Available Metrics in TensorBoard

- **Scalars**: All numerical metrics over time
- **Histograms**: Distribution of user activity and engagement
- **Custom**: Network structure and community metrics

## Metrics Reference

### User Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `total_users` | Total number of users | ≥ 0 |
| `active_users` | Number of active users | 0 - total_users |
| `new_users` | New users in current period | ≥ 0 |
| `churned_users` | Users who left | ≥ 0 |
| `user_growth_rate` | User growth rate | 0.0 - 1.0 |

### Engagement Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `engagement_rate` | Overall engagement rate | 0.0 - 1.0 |
| `total_posts` | Total posts created | ≥ 0 |
| `total_likes` | Total likes given | ≥ 0 |
| `total_comments` | Total comments made | ≥ 0 |
| `total_shares` | Total shares made | ≥ 0 |
| `avg_session_time` | Average session duration (minutes) | ≥ 0 |
| `bounce_rate` | User bounce rate | 0.0 - 1.0 |
| `retention_rate` | User retention rate | 0.0 - 1.0 |

### Content Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `content_quality_score` | Overall content quality | 0.0 - 1.0 |
| `user_satisfaction_score` | User satisfaction | 0.0 - 1.0 |
| `content_diversity_score` | Content diversity | 0.0 - 1.0 |
| `controversy_level` | Content controversy | 0.0 - 1.0 |
| `viral_posts` | Number of viral posts | ≥ 0 |
| `trending_topics` | Number of trending topics | ≥ 0 |

### Network Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `network_density` | Network connection density | 0.0 - 1.0 |
| `avg_followers` | Average followers per user | ≥ 0 |
| `avg_following` | Average following per user | ≥ 0 |
| `clustering_coefficient` | Network clustering | 0.0 - 1.0 |
| `network_growth_rate` | Network growth rate | 0.0 - 1.0 |

### Community Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `total_communities` | Number of communities | ≥ 0 |
| `avg_community_size` | Average community size | ≥ 0 |
| `community_engagement` | Community engagement rate | 0.0 - 1.0 |
| `cross_community_interactions` | Cross-community interactions | ≥ 0 |

### Influence Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `influencer_count` | Number of influencers | ≥ 0 |
| `avg_influence_score` | Average influence score | 0.0 - 1.0 |
| `viral_spread_rate` | Viral content spread rate | 0.0 - 1.0 |
| `information_cascade_depth` | Information cascade depth | ≥ 0 |

### Performance Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `response_time_avg` | Average response time (seconds) | ≥ 0 |
| `error_rate` | System error rate | 0.0 - 1.0 |
| `system_uptime` | System uptime (seconds) | ≥ 0 |

## Best Practices

### 1. Configuration Management

```python
# Use environment variables for sensitive settings
import os

config = MonitoringConfig(
    enable_wandb=os.getenv("ENABLE_WANDB", "false").lower() == "true",
    wandb_entity=os.getenv("WANDB_ENTITY"),
    wandb_project_name=os.getenv("WANDB_PROJECT", "sandgraph-social")
)
```

### 2. Error Handling

```python
try:
    monitor.update_metrics(metrics)
except Exception as e:
    logger.error(f"Failed to update metrics: {e}")
    # Fallback to file logging only
```

### 3. Resource Management

```python
# Always stop monitoring when done
try:
    monitor.start_monitoring()
    # ... your code ...
finally:
    monitor.stop_monitoring()
```

### 4. Data Export

```python
# Export data in multiple formats
monitor.export_metrics("./logs/metrics.json", "json")
monitor.export_metrics("./logs/metrics.csv", "csv")

# Get summary for quick analysis
summary = monitor.get_metrics_summary()
```

### 5. Visualization Optimization

```python
# For large datasets, sample data for visualization
if len(metrics_history) > 1000:
    sampled_history = metrics_history[::10]  # Sample every 10th point
else:
    sampled_history = metrics_history

visualizer.create_dashboard(sampled_history)
```

## Troubleshooting

### Common Issues

1. **WanDB Connection Failed**
   - Check internet connection
   - Verify WanDB credentials: `wandb login`
   - Check project permissions

2. **TensorBoard Not Starting**
   - Verify TensorBoard installation: `pip install tensorboard`
   - Check log directory permissions
   - Ensure port 6006 is available

3. **Memory Issues**
   - Reduce `history_window_size` in configuration
   - Clear old metrics periodically
   - Use data sampling for large datasets

4. **Visualization Errors**
   - Install required libraries: `pip install matplotlib plotly seaborn`
   - Check data format and ranges
   - Verify output directory permissions

### Performance Optimization

1. **Reduce Sampling Frequency**
   ```python
   config.metrics_sampling_interval = 5.0  # Sample every 5 seconds
   ```

2. **Limit History Size**
   ```python
   config.history_window_size = 500  # Keep last 500 data points
   ```

3. **Disable Unused Backends**
   ```python
   config.enable_wandb = False  # Disable if not needed
   config.enable_tensorboard = False
   ```

## Examples

### Complete Example

See `demo/monitoring_example.py` for a complete working example.

### Quick Start Example

```python
from sandbox_rl.core.monitoring import create_monitor, SocialNetworkMetrics
from sandbox_rl.core.visualization import quick_visualization

# Setup monitoring
monitor = create_monitor()
monitor.start_monitoring()

# Generate sample metrics
metrics_history = []
for step in range(10):
    metrics = SocialNetworkMetrics(
        total_users=100 + step * 10,
        active_users=80 + step * 8,
        engagement_rate=0.2 + step * 0.01
    )
    metrics_history.append(metrics)
    monitor.update_metrics(metrics)

# Stop monitoring
monitor.stop_monitoring()

# Create visualizations
quick_visualization(metrics_history, "./visualizations")
```

## API Reference

### MonitoringConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_wandb` | bool | True | Enable WanDB logging |
| `enable_tensorboard` | bool | True | Enable TensorBoard logging |
| `enable_console_logging` | bool | True | Enable console output |
| `enable_file_logging` | bool | True | Enable file logging |
| `wandb_project_name` | str | "sandgraph-social-network" | WanDB project name |
| `wandb_entity` | Optional[str] | None | WanDB entity/username |
| `wandb_run_name` | Optional[str] | None | WanDB run name |
| `tensorboard_log_dir` | str | "./logs/tensorboard" | TensorBoard log directory |
| `log_file_path` | str | "./logs/social_network_metrics.json" | Metrics log file |
| `metrics_sampling_interval` | float | 1.0 | Sampling interval in seconds |
| `history_window_size` | int | 1000 | Number of data points to keep |
| `engagement_rate_threshold` | float | 0.1 | Engagement alert threshold |
| `user_growth_threshold` | float | 0.05 | Growth alert threshold |
| `error_rate_threshold` | float | 0.05 | Error rate alert threshold |
| `response_time_threshold` | float | 2.0 | Response time alert threshold |

### SocialNetworkMonitor

| Method | Description |
|--------|-------------|
| `start_monitoring()` | Start the monitoring system |
| `stop_monitoring()` | Stop the monitoring system |
| `update_metrics(metrics)` | Update current metrics |
| `add_alert_callback(callback)` | Add alert handler |
| `get_metrics_summary()` | Get metrics summary |
| `export_metrics(filepath, format)` | Export metrics to file |

### SocialNetworkVisualizer

| Method | Description |
|--------|-------------|
| `create_dashboard(metrics_history)` | Create static dashboard |
| `create_interactive_dashboard(metrics_history)` | Create interactive dashboard |
| `create_trend_analysis(metrics_history)` | Create trend analysis |
| `create_heatmap(metrics_history)` | Create correlation heatmap |
| `export_visualization_report(metrics_history)` | Export comprehensive report |

## Contributing

To extend the monitoring system:

1. Add new metrics to `SocialNetworkMetrics`
2. Implement calculation logic in `MetricsCollector`
3. Add visualization methods to `SocialNetworkVisualizer`
4. Update documentation and examples

## License

This monitoring system is part of Sandbox-RLX and follows the same MIT license. 
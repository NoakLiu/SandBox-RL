"""
Social Network Monitoring and Metrics System
============================================

This module provides comprehensive monitoring capabilities for social network analysis
with integration for WanDB and TensorBoard for real-time visualization and tracking.
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import threading
import queue

# Optional imports for external monitoring tools
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

logger = logging.getLogger(__name__)


@dataclass
class SocialNetworkMetrics:
    """Comprehensive social network metrics data structure"""
    
    # User Metrics
    total_users: int = 0
    active_users: int = 0
    new_users: int = 0
    churned_users: int = 0
    user_growth_rate: float = 0.0
    
    # Engagement Metrics
    total_posts: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_shares: int = 0
    engagement_rate: float = 0.0
    avg_session_time: float = 0.0
    bounce_rate: float = 0.0
    retention_rate: float = 0.0
    
    # Content Metrics
    viral_posts: int = 0
    trending_topics: int = 0
    content_quality_score: float = 0.0
    user_satisfaction_score: float = 0.0
    content_diversity_score: float = 0.0
    controversy_level: float = 0.0
    
    # Network Metrics
    network_density: float = 0.0
    avg_followers: float = 0.0
    avg_following: float = 0.0
    clustering_coefficient: float = 0.0
    network_growth_rate: float = 0.0
    
    # Community Metrics
    total_communities: int = 0
    avg_community_size: float = 0.0
    community_engagement: float = 0.0
    cross_community_interactions: int = 0
    
    # Influence Metrics
    influencer_count: int = 0
    avg_influence_score: float = 0.0
    viral_spread_rate: float = 0.0
    information_cascade_depth: float = 0.0
    
    # Behavioral Metrics
    user_segments: Dict[str, int] = None
    activity_patterns: Dict[str, float] = None
    content_preferences: Dict[str, float] = None
    
    # Temporal Metrics
    peak_activity_hours: List[int] = None
    daily_active_users: List[int] = None
    weekly_growth_trend: List[float] = None
    
    # Performance Metrics
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    system_uptime: float = 0.0
    
    def __post_init__(self):
        if self.user_segments is None:
            self.user_segments = {}
        if self.activity_patterns is None:
            self.activity_patterns = {}
        if self.content_preferences is None:
            self.content_preferences = {}
        if self.peak_activity_hours is None:
            self.peak_activity_hours = []
        if self.daily_active_users is None:
            self.daily_active_users = []
        if self.weekly_growth_trend is None:
            self.weekly_growth_trend = []


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    
    # General settings
    enable_wandb: bool = True
    enable_tensorboard: bool = True
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    
    # WanDB settings
    wandb_project_name: str = "sandgraph-social-network"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # TensorBoard settings
    tensorboard_log_dir: str = "./logs/tensorboard"
    
    # File logging settings
    log_file_path: str = "./logs/social_network_metrics.json"
    
    # Sampling settings
    metrics_sampling_interval: float = 1.0  # seconds
    history_window_size: int = 1000  # number of data points to keep in memory
    
    # Alert thresholds
    engagement_rate_threshold: float = 0.1
    user_growth_threshold: float = 0.05
    error_rate_threshold: float = 0.05
    response_time_threshold: float = 2.0


class SocialNetworkMonitor:
    """Comprehensive social network monitoring system"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.history_window_size)
        self.current_metrics = SocialNetworkMetrics()
        self.start_time = time.time()
        self.last_update = time.time()
        
        # Initialize monitoring backends
        self.wandb_run = None
        self.tensorboard_writer = None
        self.setup_monitoring_backends()
        
        # Threading for async metrics collection
        self.metrics_queue = queue.Queue()
        self.monitoring_thread = None
        self.is_running = False
        
        # Alert system
        self.alerts = []
        self.alert_callbacks = []
        
        logger.info("Social Network Monitor initialized")
    
    def setup_monitoring_backends(self):
        """Setup WanDB and TensorBoard backends"""
        
        # Setup WanDB
        if self.config.enable_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project_name,
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_run_name or f"social_network_{int(time.time())}",
                    config=asdict(self.config)
                )
                logger.info("WanDB monitoring enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize WanDB: {e}")
                self.wandb_run = None
        
        # Setup TensorBoard
        if self.config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(self.config.tensorboard_log_dir)
                logger.info("TensorBoard monitoring enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.tensorboard_writer = None
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Social Network Monitor started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        # Close monitoring backends
        if self.wandb_run:
            self.wandb_run.finish()
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        logger.info("Social Network Monitor stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Process metrics from queue
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    self._process_metrics(metrics)
                
                # Sleep for sampling interval
                time.sleep(self.config.metrics_sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def update_metrics(self, metrics: SocialNetworkMetrics):
        """Update current metrics (thread-safe)"""
        self.metrics_queue.put(metrics)
    
    def _process_metrics(self, metrics: SocialNetworkMetrics):
        """Process and store metrics"""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)
        
        # Log to different backends
        self._log_to_wandb(metrics)
        self._log_to_tensorboard(metrics)
        self._log_to_file(metrics)
        self._log_to_console(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        self.last_update = time.time()
    
    def _log_to_wandb(self, metrics: SocialNetworkMetrics):
        """Log metrics to WanDB"""
        if not self.wandb_run:
            return
        
        try:
            wandb.log({
                # User metrics
                "users/total": metrics.total_users,
                "users/active": metrics.active_users,
                "users/new": metrics.new_users,
                "users/churned": metrics.churned_users,
                "users/growth_rate": metrics.user_growth_rate,
                
                # Engagement metrics
                "engagement/posts": metrics.total_posts,
                "engagement/likes": metrics.total_likes,
                "engagement/comments": metrics.total_comments,
                "engagement/shares": metrics.total_shares,
                "engagement/rate": metrics.engagement_rate,
                "engagement/session_time": metrics.avg_session_time,
                "engagement/bounce_rate": metrics.bounce_rate,
                "engagement/retention_rate": metrics.retention_rate,
                
                # Content metrics
                "content/viral_posts": metrics.viral_posts,
                "content/trending_topics": metrics.trending_topics,
                "content/quality_score": metrics.content_quality_score,
                "content/satisfaction_score": metrics.user_satisfaction_score,
                "content/diversity_score": metrics.content_diversity_score,
                "content/controversy_level": metrics.controversy_level,
                
                # Network metrics
                "network/density": metrics.network_density,
                "network/avg_followers": metrics.avg_followers,
                "network/avg_following": metrics.avg_following,
                "network/clustering_coefficient": metrics.clustering_coefficient,
                "network/growth_rate": metrics.network_growth_rate,
                
                # Community metrics
                "communities/total": metrics.total_communities,
                "communities/avg_size": metrics.avg_community_size,
                "communities/engagement": metrics.community_engagement,
                "communities/cross_interactions": metrics.cross_community_interactions,
                
                # Influence metrics
                "influence/influencer_count": metrics.influencer_count,
                "influence/avg_score": metrics.avg_influence_score,
                "influence/viral_spread_rate": metrics.viral_spread_rate,
                "influence/cascade_depth": metrics.information_cascade_depth,
                
                # Performance metrics
                "performance/response_time": metrics.response_time_avg,
                "performance/error_rate": metrics.error_rate,
                "performance/uptime": metrics.system_uptime,
                
                # User segments
                **{f"segments/{segment}": count for segment, count in metrics.user_segments.items()},
                
                # Activity patterns
                **{f"activity/{pattern}": value for pattern, value in metrics.activity_patterns.items()},
                
                # Content preferences
                **{f"preferences/{pref}": value for pref, value in metrics.content_preferences.items()}
            })
        except Exception as e:
            logger.error(f"Failed to log to WanDB: {e}")
    
    def _log_to_tensorboard(self, metrics: SocialNetworkMetrics):
        """Log metrics to TensorBoard"""
        if not self.tensorboard_writer:
            return
        
        try:
            step = len(self.metrics_history)
            
            # User metrics
            self.tensorboard_writer.add_scalar("Users/Total", metrics.total_users, step)
            self.tensorboard_writer.add_scalar("Users/Active", metrics.active_users, step)
            self.tensorboard_writer.add_scalar("Users/Growth_Rate", metrics.user_growth_rate, step)
            
            # Engagement metrics
            self.tensorboard_writer.add_scalar("Engagement/Rate", metrics.engagement_rate, step)
            self.tensorboard_writer.add_scalar("Engagement/Session_Time", metrics.avg_session_time, step)
            self.tensorboard_writer.add_scalar("Engagement/Retention_Rate", metrics.retention_rate, step)
            
            # Content metrics
            self.tensorboard_writer.add_scalar("Content/Quality_Score", metrics.content_quality_score, step)
            self.tensorboard_writer.add_scalar("Content/Satisfaction_Score", metrics.user_satisfaction_score, step)
            self.tensorboard_writer.add_scalar("Content/Diversity_Score", metrics.content_diversity_score, step)
            
            # Network metrics
            self.tensorboard_writer.add_scalar("Network/Density", metrics.network_density, step)
            self.tensorboard_writer.add_scalar("Network/Clustering_Coefficient", metrics.clustering_coefficient, step)
            self.tensorboard_writer.add_scalar("Network/Growth_Rate", metrics.network_growth_rate, step)
            
            # Influence metrics
            self.tensorboard_writer.add_scalar("Influence/Avg_Score", metrics.avg_influence_score, step)
            self.tensorboard_writer.add_scalar("Influence/Viral_Spread_Rate", metrics.viral_spread_rate, step)
            
            # Performance metrics
            self.tensorboard_writer.add_scalar("Performance/Response_Time", metrics.response_time_avg, step)
            self.tensorboard_writer.add_scalar("Performance/Error_Rate", metrics.error_rate, step)
            
            # Histograms for distributions
            if metrics.daily_active_users:
                self.tensorboard_writer.add_histogram("Users/Daily_Active", 
                                                    np.array(metrics.daily_active_users), step)
            
            if metrics.weekly_growth_trend:
                self.tensorboard_writer.add_histogram("Users/Weekly_Growth", 
                                                    np.array(metrics.weekly_growth_trend), step)
            
            # Flush to ensure data is written
            self.tensorboard_writer.flush()
            
        except Exception as e:
            logger.error(f"Failed to log to TensorBoard: {e}")
    
    def _log_to_file(self, metrics: SocialNetworkMetrics):
        """Log metrics to file"""
        if not self.config.enable_file_logging:
            return
        
        try:
            os.makedirs(os.path.dirname(self.config.log_file_path), exist_ok=True)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": asdict(metrics)
            }
            
            with open(self.config.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to log to file: {e}")
    
    def _log_to_console(self, metrics: SocialNetworkMetrics):
        """Log metrics to console"""
        if not self.config.enable_console_logging:
            return
        
        logger.info(f"Social Network Metrics - "
                   f"Users: {metrics.total_users} (Active: {metrics.active_users}), "
                   f"Engagement: {metrics.engagement_rate:.3f}, "
                   f"Quality: {metrics.content_quality_score:.3f}")
    
    def _check_alerts(self, metrics: SocialNetworkMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # Engagement rate alert
        if metrics.engagement_rate < self.config.engagement_rate_threshold:
            alerts.append({
                "type": "low_engagement",
                "message": f"Engagement rate ({metrics.engagement_rate:.3f}) below threshold ({self.config.engagement_rate_threshold})",
                "severity": "warning",
                "timestamp": datetime.now().isoformat()
            })
        
        # User growth alert
        if metrics.user_growth_rate < self.config.user_growth_threshold:
            alerts.append({
                "type": "low_growth",
                "message": f"User growth rate ({metrics.user_growth_rate:.3f}) below threshold ({self.config.user_growth_threshold})",
                "severity": "warning",
                "timestamp": datetime.now().isoformat()
            })
        
        # Error rate alert
        if metrics.error_rate > self.config.error_rate_threshold:
            alerts.append({
                "type": "high_error_rate",
                "message": f"Error rate ({metrics.error_rate:.3f}) above threshold ({self.config.error_rate_threshold})",
                "severity": "error",
                "timestamp": datetime.now().isoformat()
            })
        
        # Response time alert
        if metrics.response_time_avg > self.config.response_time_threshold:
            alerts.append({
                "type": "high_response_time",
                "message": f"Average response time ({metrics.response_time_avg:.2f}s) above threshold ({self.config.response_time_threshold}s)",
                "severity": "warning",
                "timestamp": datetime.now().isoformat()
            })
        
        # Add new alerts
        self.alerts.extend(alerts)
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add a callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points
        
        summary = {
            "current": asdict(self.current_metrics),
            "trends": {
                "engagement_rate": [m.engagement_rate for m in recent_metrics],
                "user_growth_rate": [m.user_growth_rate for m in recent_metrics],
                "content_quality": [m.content_quality_score for m in recent_metrics],
                "network_density": [m.network_density for m in recent_metrics]
            },
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "uptime": time.time() - self.start_time,
            "last_update": self.last_update
        }
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file"""
        try:
            if format.lower() == "json":
                with open(filepath, "w") as f:
                    json.dump({
                        "config": asdict(self.config),
                        "metrics_history": [asdict(m) for m in self.metrics_history],
                        "alerts": self.alerts,
                        "summary": self.get_metrics_summary()
                    }, f, indent=2)
            elif format.lower() == "csv":
                import pandas as pd
                df = pd.DataFrame([asdict(m) for m in self.metrics_history])
                df.to_csv(filepath, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")


class MetricsCollector:
    """Helper class for collecting metrics from social network data"""
    
    @staticmethod
    def calculate_network_metrics(network_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate network-level metrics from network state"""
        metrics = {}
        
        users = network_state.get("users", [])
        connections = network_state.get("connections", [])
        
        if not users:
            return metrics
        
        # Basic user metrics
        metrics["total_users"] = len(users)
        metrics["active_users"] = sum(1 for user in users if user.get("activity_level", 0) > 0.5)
        
        # Network density
        total_possible_connections = len(users) * (len(users) - 1)
        if total_possible_connections > 0:
            metrics["network_density"] = len(connections) / total_possible_connections
        
        # Average followers/following
        total_followers = sum(len(user.get("followers", [])) for user in users)
        total_following = sum(len(user.get("following", [])) for user in users)
        metrics["avg_followers"] = total_followers / len(users) if users else 0
        metrics["avg_following"] = total_following / len(users) if users else 0
        
        # Clustering coefficient (simplified)
        if connections:
            metrics["clustering_coefficient"] = min(1.0, len(connections) / (len(users) * 2))
        
        return metrics
    
    @staticmethod
    def calculate_engagement_metrics(user_behavior: Dict[str, Any]) -> Dict[str, float]:
        """Calculate engagement metrics from user behavior data"""
        metrics = {}
        
        total_users = user_behavior.get("total_users", 1)
        active_users = user_behavior.get("active_users", 0)
        
        # Engagement rate
        metrics["engagement_rate"] = active_users / total_users if total_users > 0 else 0
        
        # Content engagement
        metrics["total_posts"] = user_behavior.get("posts_created", 0)
        metrics["total_likes"] = user_behavior.get("likes_given", 0)
        metrics["total_comments"] = user_behavior.get("comments_made", 0)
        metrics["total_shares"] = user_behavior.get("shares_made", 0)
        
        # Session metrics
        metrics["avg_session_time"] = user_behavior.get("avg_session_time", 0)
        metrics["bounce_rate"] = user_behavior.get("bounce_rate", 0)
        metrics["retention_rate"] = user_behavior.get("retention_rate", 0)
        
        return metrics
    
    @staticmethod
    def calculate_content_metrics(content_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate content-related metrics"""
        metrics = {}
        
        # Viral content
        metrics["viral_posts"] = content_data.get("viral_posts", 0)
        metrics["trending_topics"] = content_data.get("trending_topics", 0)
        
        # Quality metrics
        metrics["content_quality_score"] = content_data.get("quality_score", 0)
        metrics["user_satisfaction_score"] = content_data.get("satisfaction_score", 0)
        metrics["content_diversity_score"] = content_data.get("diversity_score", 0)
        metrics["controversy_level"] = content_data.get("controversy_level", 0)
        
        return metrics
    
    @staticmethod
    def create_social_network_metrics(
        network_state: Dict[str, Any],
        user_behavior: Dict[str, Any],
        content_metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> SocialNetworkMetrics:
        """Create comprehensive social network metrics from various data sources"""
        
        # Calculate different metric categories
        network_metrics = MetricsCollector.calculate_network_metrics(network_state)
        engagement_metrics = MetricsCollector.calculate_engagement_metrics(user_behavior)
        content_metrics_calc = MetricsCollector.calculate_content_metrics(content_metrics)
        
        # Create metrics object
        metrics = SocialNetworkMetrics(
            # Network metrics
            total_users=network_metrics.get("total_users", 0),
            active_users=network_metrics.get("active_users", 0),
            network_density=network_metrics.get("network_density", 0),
            avg_followers=network_metrics.get("avg_followers", 0),
            avg_following=network_metrics.get("avg_following", 0),
            clustering_coefficient=network_metrics.get("clustering_coefficient", 0),
            
            # Engagement metrics
            engagement_rate=engagement_metrics.get("engagement_rate", 0),
            total_posts=engagement_metrics.get("total_posts", 0),
            total_likes=engagement_metrics.get("total_likes", 0),
            total_comments=engagement_metrics.get("total_comments", 0),
            total_shares=engagement_metrics.get("total_shares", 0),
            avg_session_time=engagement_metrics.get("avg_session_time", 0),
            bounce_rate=engagement_metrics.get("bounce_rate", 0),
            retention_rate=engagement_metrics.get("retention_rate", 0),
            
            # Content metrics
            viral_posts=content_metrics_calc.get("viral_posts", 0),
            trending_topics=content_metrics_calc.get("trending_topics", 0),
            content_quality_score=content_metrics_calc.get("content_quality_score", 0),
            user_satisfaction_score=content_metrics_calc.get("user_satisfaction_score", 0),
            content_diversity_score=content_metrics_calc.get("content_diversity_score", 0),
            controversy_level=content_metrics_calc.get("controversy_level", 0)
        )
        
        # Add additional data if provided
        if additional_data:
            for key, value in additional_data.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
        
        return metrics


# Convenience functions for easy integration
def create_monitor(config: Optional[MonitoringConfig] = None) -> SocialNetworkMonitor:
    """Create a social network monitor with default or custom configuration"""
    if config is None:
        config = MonitoringConfig()
    
    return SocialNetworkMonitor(config)


def log_metrics_to_wandb(metrics: Dict[str, Any], project_name: str = "sandgraph-social-network"):
    """Quick function to log metrics to WanDB"""
    if not WANDB_AVAILABLE:
        logger.warning("WanDB not available")
        return
    
    try:
        wandb.log(metrics)
    except Exception as e:
        logger.error(f"Failed to log to WanDB: {e}")


def log_metrics_to_tensorboard(metrics: Dict[str, Any], log_dir: str = "./logs/tensorboard"):
    """Quick function to log metrics to TensorBoard"""
    if not TENSORBOARD_AVAILABLE:
        logger.warning("TensorBoard not available")
        return
    
    try:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        step = int(time.time())  # Use timestamp as step
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, step)
        
        writer.flush()
        writer.close()
    except Exception as e:
        logger.error(f"Failed to log to TensorBoard: {e}") 
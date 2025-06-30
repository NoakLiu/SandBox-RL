"""
Social Network Visualization Module
==================================

This module provides comprehensive visualization capabilities for social network
monitoring data, including real-time charts, dashboards, and interactive plots.
"""

import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Optional imports for visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

from .monitoring import SocialNetworkMetrics

logger = logging.getLogger(__name__)


class SocialNetworkVisualizer:
    """Comprehensive social network data visualizer"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for matplotlib
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('seaborn-v0_8')
            if SEABORN_AVAILABLE:
                sns.set_palette("husl")
    
    def create_dashboard(self, metrics_history: List[SocialNetworkMetrics], 
                        save_path: Optional[str] = None) -> str:
        """Create a comprehensive dashboard with multiple charts"""
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for dashboard creation")
            return ""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define subplot layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. User Growth and Engagement (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_user_metrics(metrics_history, ax1)
        
        # 2. Network Metrics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_network_metrics(metrics_history, ax2)
        
        # 3. Content Performance (second row)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_content_metrics(metrics_history, ax3)
        
        # 4. Engagement Breakdown (third row left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_engagement_breakdown(metrics_history, ax4)
        
        # 5. Community Metrics (third row middle)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_community_metrics(metrics_history, ax5)
        
        # 6. Influence Metrics (third row right)
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_influence_metrics(metrics_history, ax6)
        
        # 7. Performance Metrics (bottom row)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_performance_metrics(metrics_history, ax7)
        
        # Add title
        fig.suptitle('Social Network Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")
        else:
            save_path = os.path.join(self.output_dir, f"dashboard_{int(time.time())}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def _plot_user_metrics(self, metrics_history: List[SocialNetworkMetrics], ax):
        """Plot user growth and engagement metrics"""
        if not metrics_history:
            return
        
        steps = range(len(metrics_history))
        total_users = [m.total_users for m in metrics_history]
        active_users = [m.active_users for m in metrics_history]
        engagement_rates = [m.engagement_rate for m in metrics_history]
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        # Plot user counts
        line1 = ax.plot(steps, total_users, 'b-', linewidth=2, label='Total Users')
        line2 = ax.plot(steps, active_users, 'g-', linewidth=2, label='Active Users')
        
        # Plot engagement rate
        line3 = ax2.plot(steps, engagement_rates, 'r--', linewidth=2, label='Engagement Rate')
        
        # Customize axes
        ax.set_xlabel('Simulation Steps')
        ax.set_ylabel('Number of Users', color='b')
        ax2.set_ylabel('Engagement Rate', color='r')
        ax.set_title('User Growth and Engagement')
        
        # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def _plot_network_metrics(self, metrics_history: List[SocialNetworkMetrics], ax):
        """Plot network structure metrics"""
        if not metrics_history:
            return
        
        steps = range(len(metrics_history))
        density = [m.network_density for m in metrics_history]
        clustering = [m.clustering_coefficient for m in metrics_history]
        avg_followers = [m.avg_followers for m in metrics_history]
        
        # Create stacked area plot
        ax.fill_between(steps, 0, density, alpha=0.6, label='Network Density')
        ax.fill_between(steps, density, [d + c for d, c in zip(density, clustering)], 
                       alpha=0.6, label='Clustering Coefficient')
        ax.fill_between(steps, [d + c for d, c in zip(density, clustering)], 
                       [d + c + f/10 for d, c, f in zip(density, clustering, avg_followers)], 
                       alpha=0.6, label='Avg Followers (scaled)')
        
        ax.set_xlabel('Simulation Steps')
        ax.set_ylabel('Metric Values')
        ax.set_title('Network Structure Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_content_metrics(self, metrics_history: List[SocialNetworkMetrics], ax):
        """Plot content performance metrics"""
        if not metrics_history:
            return
        
        steps = range(len(metrics_history))
        quality = [m.content_quality_score for m in metrics_history]
        satisfaction = [m.user_satisfaction_score for m in metrics_history]
        diversity = [m.content_diversity_score for m in metrics_history]
        controversy = [m.controversy_level for m in metrics_history]
        
        # Plot multiple lines
        ax.plot(steps, quality, 'b-', linewidth=2, label='Content Quality')
        ax.plot(steps, satisfaction, 'g-', linewidth=2, label='User Satisfaction')
        ax.plot(steps, diversity, 'r-', linewidth=2, label='Content Diversity')
        ax.plot(steps, controversy, 'orange', linewidth=2, label='Controversy Level')
        
        ax.set_xlabel('Simulation Steps')
        ax.set_ylabel('Score')
        ax.set_title('Content Performance Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_engagement_breakdown(self, metrics_history: List[SocialNetworkMetrics], ax):
        """Plot engagement breakdown as pie chart"""
        if not metrics_history:
            return
        
        # Use the latest metrics
        latest = metrics_history[-1]
        
        engagement_data = [
            latest.total_posts,
            latest.total_likes,
            latest.total_comments,
            latest.total_shares
        ]
        
        labels = ['Posts', 'Likes', 'Comments', 'Shares']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(engagement_data, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title('Engagement Breakdown')
    
    def _plot_community_metrics(self, metrics_history: List[SocialNetworkMetrics], ax):
        """Plot community-related metrics"""
        if not metrics_history:
            return
        
        steps = range(len(metrics_history))
        communities = [m.total_communities for m in metrics_history]
        avg_size = [m.avg_community_size for m in metrics_history]
        engagement = [m.community_engagement for m in metrics_history]
        
        # Create bar chart for latest values
        categories = ['Communities', 'Avg Size', 'Engagement']
        values = [communities[-1], avg_size[-1], engagement[-1]]
        
        bars = ax.bar(categories, values, color=['#ff9999', '#66b3ff', '#99ff99'])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
        
        ax.set_title('Community Metrics')
        ax.set_ylabel('Value')
    
    def _plot_influence_metrics(self, metrics_history: List[SocialNetworkMetrics], ax):
        """Plot influence and viral metrics"""
        if not metrics_history:
            return
        
        steps = range(len(metrics_history))
        influencers = [m.influencer_count for m in metrics_history]
        avg_influence = [m.avg_influence_score for m in metrics_history]
        viral_rate = [m.viral_spread_rate for m in metrics_history]
        
        # Create scatter plot
        scatter = ax.scatter(avg_influence, viral_rate, c=influencers, 
                           s=[i*10 for i in influencers], alpha=0.6, cmap='viridis')
        
        ax.set_xlabel('Average Influence Score')
        ax.set_ylabel('Viral Spread Rate')
        ax.set_title('Influence vs Viral Spread')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Influencer Count')
    
    def _plot_performance_metrics(self, metrics_history: List[SocialNetworkMetrics], ax):
        """Plot system performance metrics"""
        if not metrics_history:
            return
        
        steps = range(len(metrics_history))
        response_time = [m.response_time_avg for m in metrics_history]
        error_rate = [m.error_rate for m in metrics_history]
        uptime = [m.system_uptime for m in metrics_history]
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        # Plot response time and error rate
        line1 = ax.plot(steps, response_time, 'b-', linewidth=2, label='Response Time (s)')
        line2 = ax.plot(steps, error_rate, 'r-', linewidth=2, label='Error Rate')
        
        # Plot uptime
        line3 = ax2.plot(steps, uptime, 'g--', linewidth=2, label='System Uptime (s)')
        
        # Customize axes
        ax.set_xlabel('Simulation Steps')
        ax.set_ylabel('Time/Rate', color='b')
        ax2.set_ylabel('Uptime (s)', color='g')
        ax.set_title('System Performance Metrics')
        
        # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    def create_interactive_dashboard(self, metrics_history: List[SocialNetworkMetrics], 
                                   save_path: Optional[str] = None) -> str:
        """Create an interactive dashboard using Plotly"""
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for interactive dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('User Growth & Engagement', 'Network Metrics', 
                          'Content Performance', 'Community Overview',
                          'Influence Analysis', 'System Performance'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"colspan": 2}, None],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        if not metrics_history:
            return ""
        
        steps = list(range(len(metrics_history)))
        
        # 1. User Growth & Engagement
        fig.add_trace(
            go.Scatter(x=steps, y=[m.total_users for m in metrics_history], 
                      name='Total Users', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=[m.active_users for m in metrics_history], 
                      name='Active Users', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=[m.engagement_rate for m in metrics_history], 
                      name='Engagement Rate', line=dict(color='red', dash='dash'),
                      yaxis='y2'),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Network Metrics
        fig.add_trace(
            go.Scatter(x=steps, y=[m.network_density for m in metrics_history], 
                      name='Network Density', fill='tonexty'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=steps, y=[m.clustering_coefficient for m in metrics_history], 
                      name='Clustering Coefficient', fill='tonexty'),
            row=1, col=2
        )
        
        # 3. Content Performance
        fig.add_trace(
            go.Scatter(x=steps, y=[m.content_quality_score for m in metrics_history], 
                      name='Quality', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=[m.user_satisfaction_score for m in metrics_history], 
                      name='Satisfaction', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=[m.content_diversity_score for m in metrics_history], 
                      name='Diversity', line=dict(color='red')),
            row=2, col=1
        )
        
        # 4. Community Overview (bar chart)
        latest = metrics_history[-1]
        fig.add_trace(
            go.Bar(x=['Communities', 'Avg Size', 'Engagement'], 
                  y=[latest.total_communities, latest.avg_community_size, latest.community_engagement],
                  name='Community Metrics'),
            row=3, col=1
        )
        
        # 5. Influence Analysis (scatter plot)
        avg_influence = [m.avg_influence_score for m in metrics_history]
        viral_rate = [m.viral_spread_rate for m in metrics_history]
        influencer_count = [m.influencer_count for m in metrics_history]
        
        fig.add_trace(
            go.Scatter(x=avg_influence, y=viral_rate, mode='markers',
                      marker=dict(size=[i*2 for i in influencer_count], 
                                color=influencer_count, colorscale='viridis',
                                showscale=True, colorbar=dict(title="Influencers")),
                      name='Influence vs Viral'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Social Network Analytics Dashboard",
            height=800,
            showlegend=True
        )
        
        # Save or return
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        else:
            save_path = os.path.join(self.output_dir, f"interactive_dashboard_{int(time.time())}.html")
            fig.write_html(save_path)
        
        return save_path
    
    def create_trend_analysis(self, metrics_history: List[SocialNetworkMetrics], 
                            save_path: Optional[str] = None) -> str:
        """Create trend analysis charts"""
        
        if not MATPLOTLIB_AVAILABLE or not metrics_history:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Trend Analysis', fontsize=16, fontweight='bold')
        
        steps = range(len(metrics_history))
        
        # 1. Growth Trends
        ax1 = axes[0, 0]
        growth_rates = [m.user_growth_rate for m in metrics_history]
        network_growth = [m.network_growth_rate for m in metrics_history]
        
        ax1.plot(steps, growth_rates, 'b-', label='User Growth Rate')
        ax1.plot(steps, network_growth, 'g-', label='Network Growth Rate')
        ax1.set_title('Growth Trends')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Growth Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Engagement Trends
        ax2 = axes[0, 1]
        engagement = [m.engagement_rate for m in metrics_history]
        retention = [m.retention_rate for m in metrics_history]
        bounce = [m.bounce_rate for m in metrics_history]
        
        ax2.plot(steps, engagement, 'b-', label='Engagement Rate')
        ax2.plot(steps, retention, 'g-', label='Retention Rate')
        ax2.plot(steps, bounce, 'r-', label='Bounce Rate')
        ax2.set_title('Engagement Trends')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Content Trends
        ax3 = axes[1, 0]
        viral = [m.viral_posts for m in metrics_history]
        trending = [m.trending_topics for m in metrics_history]
        
        ax3.plot(steps, viral, 'b-', label='Viral Posts')
        ax3.plot(steps, trending, 'g-', label='Trending Topics')
        ax3.set_title('Content Trends')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Trends
        ax4 = axes[1, 1]
        response_time = [m.response_time_avg for m in metrics_history]
        error_rate = [m.error_rate for m in metrics_history]
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(steps, response_time, 'b-', label='Response Time')
        line2 = ax4_twin.plot(steps, error_rate, 'r-', label='Error Rate')
        
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Response Time (s)', color='b')
        ax4_twin.set_ylabel('Error Rate', color='r')
        ax4.set_title('Performance Trends')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, f"trend_analysis_{int(time.time())}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def create_heatmap(self, metrics_history: List[SocialNetworkMetrics], 
                      save_path: Optional[str] = None) -> str:
        """Create correlation heatmap of metrics"""
        
        if not SEABORN_AVAILABLE or not metrics_history:
            return ""
        
        # Extract key metrics for correlation analysis
        metrics_data = []
        for m in metrics_history:
            metrics_data.append({
                'total_users': m.total_users,
                'active_users': m.active_users,
                'engagement_rate': m.engagement_rate,
                'content_quality': m.content_quality_score,
                'network_density': m.network_density,
                'avg_influence': m.avg_influence_score,
                'viral_rate': m.viral_spread_rate,
                'response_time': m.response_time_avg,
                'error_rate': m.error_rate
            })
        
        # Create correlation matrix
        import pandas as pd
        df = pd.DataFrame(metrics_data)
        corr_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Metrics Correlation Heatmap')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, f"correlation_heatmap_{int(time.time())}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        return save_path
    
    def export_visualization_report(self, metrics_history: List[SocialNetworkMetrics], 
                                  output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export comprehensive visualization report"""
        
        if output_dir is None:
            output_dir = self.output_dir
        
        timestamp = int(time.time())
        report_files = {}
        
        try:
            # Create dashboard
            dashboard_path = os.path.join(output_dir, f"dashboard_{timestamp}.png")
            report_files['dashboard'] = self.create_dashboard(metrics_history, dashboard_path)
            
            # Create interactive dashboard
            interactive_path = os.path.join(output_dir, f"interactive_dashboard_{timestamp}.html")
            report_files['interactive'] = self.create_interactive_dashboard(metrics_history, interactive_path)
            
            # Create trend analysis
            trend_path = os.path.join(output_dir, f"trend_analysis_{timestamp}.png")
            report_files['trends'] = self.create_trend_analysis(metrics_history, trend_path)
            
            # Create correlation heatmap
            heatmap_path = os.path.join(output_dir, f"correlation_heatmap_{timestamp}.png")
            report_files['heatmap'] = self.create_heatmap(metrics_history, heatmap_path)
            
            # Create summary report
            summary_path = os.path.join(output_dir, f"visualization_summary_{timestamp}.json")
            self._create_summary_report(metrics_history, summary_path)
            report_files['summary'] = summary_path
            
            logger.info(f"Visualization report exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualization report: {e}")
        
        return report_files
    
    def _create_summary_report(self, metrics_history: List[SocialNetworkMetrics], 
                             save_path: str):
        """Create summary report with key statistics"""
        
        if not metrics_history:
            return
        
        latest = metrics_history[-1]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_data_points": len(metrics_history),
            "latest_metrics": {
                "total_users": latest.total_users,
                "active_users": latest.active_users,
                "engagement_rate": latest.engagement_rate,
                "content_quality_score": latest.content_quality_score,
                "network_density": latest.network_density,
                "viral_posts": latest.viral_posts,
                "trending_topics": latest.trending_topics,
                "avg_influence_score": latest.avg_influence_score,
                "response_time_avg": latest.response_time_avg,
                "error_rate": latest.error_rate
            },
            "trends": {
                "user_growth": [m.user_growth_rate for m in metrics_history],
                "engagement_trend": [m.engagement_rate for m in metrics_history],
                "quality_trend": [m.content_quality_score for m in metrics_history],
                "network_growth": [m.network_growth_rate for m in metrics_history]
            },
            "statistics": {
                "avg_engagement": np.mean([m.engagement_rate for m in metrics_history]),
                "max_engagement": np.max([m.engagement_rate for m in metrics_history]),
                "min_engagement": np.min([m.engagement_rate for m in metrics_history]),
                "avg_quality": np.mean([m.content_quality_score for m in metrics_history]),
                "total_viral_posts": sum([m.viral_posts for m in metrics_history]),
                "total_trending_topics": sum([m.trending_topics for m in metrics_history])
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)


# Convenience functions
def create_visualizer(output_dir: str = "./visualizations") -> SocialNetworkVisualizer:
    """Create a social network visualizer"""
    return SocialNetworkVisualizer(output_dir)


def quick_visualization(metrics_history: List[SocialNetworkMetrics], 
                       output_dir: str = "./visualizations") -> Dict[str, str]:
    """Quick visualization of metrics history"""
    visualizer = create_visualizer(output_dir)
    return visualizer.export_visualization_report(metrics_history, output_dir) 
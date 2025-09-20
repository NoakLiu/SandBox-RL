#!/usr/bin/env python3
"""
Unified Monitoring & Visualization - 统一监控可视化
=================================================

集成所有Monitoring and visualization功能：
1. Social network metrics监控
2. 动态图可视化
3. Performance metrics收集
4. 实时数据展示
5. 报告生成和导出
"""

import logging
import time
import threading
import queue
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class NodeType(Enum):
    """Node type"""
    AGENT = "agent"
    MISINFO_SOURCE = "misinfo_source"
    FACT_CHECKER = "fact_checker"
    INFLUENCER = "influencer"
    REGULAR_USER = "regular_user"


class EdgeType(Enum):
    """边类型"""
    COOPERATE = "cooperate"
    COMPETE = "compete"
    MISINFO_SPREAD = "misinfo_spread"
    FACT_CHECK = "fact_check"
    INFLUENCE = "influence"
    NEUTRAL = "neutral"


class InteractionType(Enum):
    """Interaction type"""
    SHARE = "share"
    LIKE = "like"
    COMMENT = "comment"
    FACT_CHECK = "fact_check"
    COOPERATE = "cooperate"
    COMPETE = "compete"


@dataclass
class SocialNetworkMetrics:
    """Social network metrics"""
    # 用户指标
    total_users: int = 0
    active_users: int = 0
    new_users: int = 0
    user_growth_rate: float = 0.0
    
    # 参与度指标
    total_posts: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_shares: int = 0
    engagement_rate: float = 0.0
    avg_session_time: float = 0.0
    
    # 内容指标
    viral_posts: int = 0
    trending_topics: int = 0
    content_quality_score: float = 0.0
    controversy_level: float = 0.0
    
    # 网络指标
    network_density: float = 0.0
    avg_followers: float = 0.0
    clustering_coefficient: float = 0.0
    
    # 影响力指标
    influencer_count: int = 0
    avg_influence_score: float = 0.0
    viral_spread_rate: float = 0.0
    
    # Performance metrics
    response_time_avg: float = 0.0
    error_rate: float = 0.0
    system_uptime: float = 0.0
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphNode:
    """图节点"""
    id: str
    node_type: NodeType
    position: Tuple[float, float] = (0.0, 0.0)
    belief: float = 0.5
    influence: float = 1.0
    followers: int = 0
    credibility: float = 0.5
    color: str = "blue"
    size: int = 100
    label: str = ""
    
    def __post_init__(self):
        if not self.label:
            self.label = f"{self.node_type.value}_{self.id}"


@dataclass
class GraphEdge:
    """图边"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    timestamp: float = 0.0
    interaction_type: Optional[InteractionType] = None
    color: str = "black"
    width: float = 1.0


@dataclass
class GraphEvent:
    """图事件"""
    timestamp: float
    event_type: str
    source_id: str
    target_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    log_file_path: str = "./logs/metrics.json"
    metrics_sampling_interval: float = 1.0
    history_window_size: int = 1000
    
    # 警报阈值
    engagement_rate_threshold: float = 0.1
    error_rate_threshold: float = 0.05
    response_time_threshold: float = 2.0


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.history_window_size)
        self.metrics_queue = queue.Queue()
        self.is_running = False
        self.lock = threading.Lock()
        
        # 警报回调
        self.alert_callbacks = []
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        logger.info("指标监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        logger.info("指标监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 处理队列中的指标
                while not self.metrics_queue.empty():
                    try:
                        metrics = self.metrics_queue.get_nowait()
                        self._process_metrics(metrics)
                    except queue.Empty:
                        break
                
                time.sleep(self.config.metrics_sampling_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
    
    def update_metrics(self, metrics: SocialNetworkMetrics):
        """更新指标"""
        self.metrics_queue.put(metrics)
    
    def _process_metrics(self, metrics: SocialNetworkMetrics):
        """处理指标"""
        with self.lock:
            self.metrics_history.append(metrics)
        
        # 记录到文件
        if self.config.enable_file_logging:
            self._log_to_file(metrics)
        
        # 控制台输出
        if self.config.enable_console_logging:
            self._log_to_console(metrics)
        
        # 检查警报
        self._check_alerts(metrics)
    
    def _log_to_file(self, metrics: SocialNetworkMetrics):
        """记录到文件"""
        try:
            os.makedirs(os.path.dirname(self.config.log_file_path), exist_ok=True)
            
            metrics_dict = {
                "timestamp": metrics.timestamp,
                "total_users": metrics.total_users,
                "active_users": metrics.active_users,
                "engagement_rate": metrics.engagement_rate,
                "network_density": metrics.network_density,
                "avg_influence_score": metrics.avg_influence_score,
                "response_time_avg": metrics.response_time_avg,
                "error_rate": metrics.error_rate
            }
            
            with open(self.config.log_file_path, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
                
        except Exception as e:
            logger.error(f"文件日志记录失败: {e}")
    
    def _log_to_console(self, metrics: SocialNetworkMetrics):
        """控制台输出"""
        logger.info(f"指标更新 - 用户: {metrics.total_users}, 活跃: {metrics.active_users}, "
                   f"参与度: {metrics.engagement_rate:.3f}, 响应时间: {metrics.response_time_avg:.3f}s")
    
    def _check_alerts(self, metrics: SocialNetworkMetrics):
        """检查警报"""
        alerts = []
        
        if metrics.engagement_rate < self.config.engagement_rate_threshold:
            alerts.append({
                "type": "low_engagement",
                "message": f"参与度过低: {metrics.engagement_rate:.3f}",
                "severity": "warning"
            })
        
        if metrics.error_rate > self.config.error_rate_threshold:
            alerts.append({
                "type": "high_error_rate",
                "message": f"错误率过高: {metrics.error_rate:.3f}",
                "severity": "critical"
            })
        
        if metrics.response_time_avg > self.config.response_time_threshold:
            alerts.append({
                "type": "slow_response",
                "message": f"响应时间过长: {metrics.response_time_avg:.3f}s",
                "severity": "warning"
            })
        
        # 触发警报回调
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"警报回调失败: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """添加警报回调"""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = list(self.metrics_history)[-10:]
            
            return {
                "total_samples": len(self.metrics_history),
                "recent_avg_users": sum(m.total_users for m in recent_metrics) / len(recent_metrics),
                "recent_avg_engagement": sum(m.engagement_rate for m in recent_metrics) / len(recent_metrics),
                "recent_avg_response_time": sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics),
                "latest_metrics": recent_metrics[-1] if recent_metrics else None
            }


class GraphVisualizer:
    """图可视化器"""
    
    def __init__(self, log_file: str = "graph_visualization.log", update_interval: float = 1.0):
        self.log_file = log_file
        self.update_interval = update_interval
        self.max_nodes = 50
        self.max_edges = 100
        
        # 图数据
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = []
        self.events = []
        
        # 可视化状态
        self.is_running = False
        self.current_time = 0.0
        self.animation = None
        self.fig = None
        self.ax = None
        
        # 颜色映射
        self.node_colors = {
            NodeType.AGENT: "#1f77b4",
            NodeType.MISINFO_SOURCE: "#d62728",
            NodeType.FACT_CHECKER: "#2ca02c",
            NodeType.INFLUENCER: "#ff7f0e",
            NodeType.REGULAR_USER: "#9467bd"
        }
        
        self.edge_colors = {
            EdgeType.COOPERATE: "#2ca02c",
            EdgeType.COMPETE: "#d62728",
            EdgeType.MISINFO_SPREAD: "#ff7f0e",
            EdgeType.FACT_CHECK: "#1f77b4",
            EdgeType.INFLUENCE: "#9467bd",
            EdgeType.NEUTRAL: "#7f7f7f"
        }
        
        # 初始化日志
        self._init_log_file()
        
        logger.info("图可视化器初始化完成")
    
    def _init_log_file(self):
        """初始化日志文件"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump({
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "version": "1.0"
                    },
                    "events": []
                }, f, indent=2)
    
    def add_node(self, node_id: str, node_type: NodeType, 
                position: Optional[Tuple[float, float]] = None, **kwargs) -> GraphNode:
        """添加节点"""
        if position is None and NUMPY_AVAILABLE:
            position = (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
        elif position is None:
            position = (0.0, 0.0)
        
        node = GraphNode(
            id=node_id,
            node_type=node_type,
            position=position,
            color=self.node_colors.get(node_type, "#1f77b4"),
            **kwargs
        )
        
        self.nodes[node_id] = node
        if NETWORKX_AVAILABLE:
            self.graph.add_node(node_id, **node.__dict__)
        
        return node
    
    def add_edge(self, source: str, target: str, edge_type: EdgeType, 
                weight: float = 1.0, **kwargs) -> GraphEdge:
        """添加边"""
        edge = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
            timestamp=self.current_time,
            color=self.edge_colors.get(edge_type, "#7f7f7f"),
            **kwargs
        )
        
        self.edges.append(edge)
        if NETWORKX_AVAILABLE:
            self.graph.add_edge(source, target, **edge.__dict__)
        
        return edge
    
    def log_event(self, event: GraphEvent):
        """记录事件"""
        self.events.append(event)
        
        # 写入文件
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            data["events"].append({
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "source_id": event.source_id,
                "target_id": event.target_id,
                "data": event.data
            })
            
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"事件日志记录失败: {e}")
    
    def create_misinfo_scenario(self, num_agents: int = 10):
        """创建misinformation传播场景"""
        # 清空现有图
        if NETWORKX_AVAILABLE:
            self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # 创建节点
        self.add_node("misinfo_source_1", NodeType.MISINFO_SOURCE, 
                     belief=1.0, influence=2.0, followers=1000, size=200)
        
        self.add_node("fact_checker_1", NodeType.FACT_CHECKER,
                     belief=0.0, influence=1.5, followers=500, size=150)
        
        self.add_node("influencer_1", NodeType.INFLUENCER,
                     belief=0.8, influence=1.8, followers=800, size=180)
        
        # 创建普通用户
        for i in range(num_agents - 3):
            user_id = f"user_{i+1}"
            if NUMPY_AVAILABLE:
                belief = np.random.uniform(0.1, 0.9)
                influence = np.random.uniform(0.5, 1.5)
                followers = np.random.randint(10, 100)
            else:
                belief = 0.5
                influence = 1.0
                followers = 50
            
            self.add_node(user_id, NodeType.REGULAR_USER,
                         belief=belief, influence=influence, followers=followers,
                         size=50 + int(belief * 100))
        
        # 添加初始边
        self.add_edge("misinfo_source_1", "influencer_1", EdgeType.MISINFO_SPREAD, weight=0.8)
        self.add_edge("influencer_1", "user_1", EdgeType.INFLUENCE, weight=0.6)
        self.add_edge("fact_checker_1", "user_2", EdgeType.FACT_CHECK, weight=0.7)
        
        logger.info(f"创建了包含{len(self.nodes)}个节点的misinformation场景")
    
    def simulate_interaction(self, source_id: str, target_id: str, interaction_type: InteractionType):
        """模拟交互"""
        source_node = self.nodes.get(source_id)
        target_node = self.nodes.get(target_id)
        
        if not source_node or not target_node:
            return
        
        # 根据Interaction type更新belief
        if interaction_type == InteractionType.SHARE:
            belief_transfer = source_node.belief * source_node.influence * 0.1
            target_node.belief = min(1.0, target_node.belief + belief_transfer)
            edge_type = EdgeType.MISINFO_SPREAD
            
        elif interaction_type == InteractionType.FACT_CHECK:
            belief_correction = (1.0 - source_node.belief) * source_node.credibility * 0.2
            target_node.belief = max(0.0, target_node.belief - belief_correction)
            edge_type = EdgeType.FACT_CHECK
            
        elif interaction_type == InteractionType.COOPERATE:
            edge_type = EdgeType.COOPERATE
            
        elif interaction_type == InteractionType.COMPETE:
            edge_type = EdgeType.COMPETE
            
        else:
            edge_type = EdgeType.NEUTRAL
        
        # 添加边
        self.add_edge(source_id, target_id, edge_type, weight=1.0)
        
        # 记录事件
        event = GraphEvent(
            timestamp=self.current_time,
            event_type=interaction_type.value,
            source_id=source_id,
            target_id=target_id,
            data={
                "source_belief": source_node.belief,
                "target_belief": target_node.belief,
                "edge_type": edge_type.value
            }
        )
        self.log_event(event)
        
        # 更新节点大小
        target_node.size = 50 + int(target_node.belief * 150)
    
    def start_visualization(self):
        """启动可视化"""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib不可用，无法启动可视化")
            return
        
        if self.is_running:
            return
        
        self.is_running = True
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 创建动画
        self.animation = animation.FuncAnimation(
            self.fig, self._update_visualization,
            interval=self.update_interval * 1000,
            blit=False
        )
        
        # 启动模拟线程
        simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        simulation_thread.start()
        
        plt.show()
    
    def _update_visualization(self, frame):
        """更新可视化"""
        if not self.is_running or not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        
        # 绘制节点
        for node_id, node in self.nodes.items():
            x, y = node.position
            color = node.color
            size = node.size
            
            # 根据belief调整颜色
            if node.node_type == NodeType.REGULAR_USER and NUMPY_AVAILABLE:
                red = node.belief
                green = 1.0 - node.belief
                color = (red, green, 0.0)
            
            self.ax.scatter(x, y, s=size, c=[color], alpha=0.7, edgecolors='black', linewidth=1)
            
            # 添加标签
            if node.influence > 1.2:
                self.ax.annotate(node.label, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        # 绘制边
        for edge in self.edges[-self.max_edges:]:
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)
            
            if source_node and target_node:
                x1, y1 = source_node.position
                x2, y2 = target_node.position
                
                linestyle = '-'
                alpha = 0.6
                
                if edge.edge_type == EdgeType.COOPERATE:
                    linestyle = '--'
                    alpha = 0.8
                elif edge.edge_type == EdgeType.COMPETE:
                    linestyle = ':'
                    alpha = 0.8
                
                self.ax.plot([x1, x2], [y1, y2], color=edge.color, 
                           linewidth=edge.width, alpha=alpha, linestyle=linestyle)
        
        # 设置画布
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 12)
        self.ax.set_title(f'SRL Graph Visualization - Time: {self.current_time:.1f}s')
        self.ax.set_aspect('equal')
    
    def _simulation_loop(self):
        """模拟循环"""
        while self.is_running:
            # 随机交互
            if len(self.nodes) >= 2:
                node_ids = list(self.nodes.keys())
                if NUMPY_AVAILABLE:
                    source_id = np.random.choice(node_ids)
                    target_id = np.random.choice([n for n in node_ids if n != source_id])
                    interaction_type = np.random.choice(list(InteractionType))
                else:
                    import random
                    source_id = random.choice(node_ids)
                    target_id = random.choice([n for n in node_ids if n != source_id])
                    interaction_type = random.choice(list(InteractionType))
                
                self.simulate_interaction(source_id, target_id, interaction_type)
            
            self.current_time += 0.1
            time.sleep(self.update_interval)
    
    def stop_visualization(self):
        """停止可视化"""
        self.is_running = False
        if self.animation and MATPLOTLIB_AVAILABLE:
            self.animation.event_source.stop()
            plt.close()
    
    def export_visualization(self, filename: str = "graph_visualization.png"):
        """导出可视化"""
        if self.fig and MATPLOTLIB_AVAILABLE:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图像已导出: {filename}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取Statistics"""
        stats = {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_events": len(self.events),
            "node_types": defaultdict(int),
            "edge_types": defaultdict(int),
            "average_belief": 0.0
        }
        
        # 统计Node type
        for node in self.nodes.values():
            stats["node_types"][node.node_type.value] += 1
        
        # 统计边类型
        for edge in self.edges:
            stats["edge_types"][edge.edge_type.value] += 1
        
        # 计算平均belief
        if self.nodes:
            if NUMPY_AVAILABLE:
                stats["average_belief"] = np.mean([node.belief for node in self.nodes.values()])
            else:
                beliefs = [node.belief for node in self.nodes.values()]
                stats["average_belief"] = sum(beliefs) / len(beliefs)
        
        return stats


class PerformanceMonitor:
    """Performance monitoring器"""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.performance_history = deque(maxlen=1000)
        self.is_monitoring = False
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """启动Performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    metrics = self._collect_performance_metrics()
                    with self.lock:
                        self.performance_history.append(metrics)
                    
                    time.sleep(self.sampling_interval)
                except Exception as e:
                    logger.error(f"Performance monitoring错误: {e}")
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
        logger.info("Performance monitoring已启动")
    
    def stop_monitoring(self):
        """停止Performance monitoring"""
        self.is_monitoring = False
        logger.info("Performance monitoring已停止")
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """收集Performance metrics"""
        try:
            import psutil
            
            # 系统指标
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU指标（如果可用）
            gpu_metrics = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_metrics[f"gpu_{i}"] = {
                        "utilization": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    }
            except ImportError:
                pass
            
            return {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "disk_percent": disk.percent,
                "gpu_metrics": gpu_metrics
            }
            
        except ImportError:
            return {
                "timestamp": time.time(),
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        with self.lock:
            if not self.performance_history:
                return {}
            
            recent_metrics = list(self.performance_history)[-10:]
            
            if NUMPY_AVAILABLE:
                cpu_values = [m["cpu_percent"] for m in recent_metrics]
                memory_values = [m["memory_percent"] for m in recent_metrics]
                
                return {
                    "avg_cpu_percent": np.mean(cpu_values),
                    "max_cpu_percent": np.max(cpu_values),
                    "avg_memory_percent": np.mean(memory_values),
                    "max_memory_percent": np.max(memory_values),
                    "sample_count": len(self.performance_history)
                }
            else:
                cpu_values = [m["cpu_percent"] for m in recent_metrics]
                memory_values = [m["memory_percent"] for m in recent_metrics]
                
                return {
                    "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
                    "max_cpu_percent": max(cpu_values),
                    "avg_memory_percent": sum(memory_values) / len(memory_values),
                    "max_memory_percent": max(memory_values),
                    "sample_count": len(self.performance_history)
                }


class UnifiedMonitor:
    """Unified monitor"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        
        # 初始化组件
        self.metrics_collector = MetricsCollector(self.config)
        self.graph_visualizer = GraphVisualizer()
        self.performance_monitor = PerformanceMonitor()
        
        # 状态管理
        self.is_running = False
        
        logger.info("Unified monitor初始化完成")
    
    def start(self):
        """启动监控"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动各组件
        self.metrics_collector.start_monitoring()
        self.performance_monitor.start_monitoring()
        
        logger.info("Unified monitor已启动")
    
    def stop(self):
        """停止监控"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止各组件
        self.metrics_collector.stop_monitoring()
        self.performance_monitor.stop_monitoring()
        self.graph_visualizer.stop_visualization()
        
        logger.info("Unified monitor已停止")
    
    def update_metrics(self, metrics: SocialNetworkMetrics):
        """更新指标"""
        self.metrics_collector.update_metrics(metrics)
    
    def create_visualization_scenario(self, num_agents: int = 10):
        """创建Visualization scenario"""
        self.graph_visualizer.create_misinfo_scenario(num_agents)
    
    def start_graph_visualization(self):
        """启动图可视化"""
        self.graph_visualizer.start_visualization()
    
    def simulate_graph_interaction(self, source_id: str, target_id: str, interaction_type: InteractionType):
        """模拟图交互"""
        self.graph_visualizer.simulate_interaction(source_id, target_id, interaction_type)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计"""
        return {
            "metrics_summary": self.metrics_collector.get_metrics_summary(),
            "graph_statistics": self.graph_visualizer.get_statistics(),
            "performance_summary": self.performance_monitor.get_performance_summary(),
            "monitoring_status": {
                "is_running": self.is_running,
                "components_active": {
                    "metrics_collector": self.metrics_collector.is_running,
                    "performance_monitor": self.performance_monitor.is_monitoring,
                    "graph_visualizer": self.graph_visualizer.is_running
                }
            }
        }
    
    def export_report(self, filepath: str):
        """导出报告"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "config": {
                    "log_file_path": self.config.log_file_path,
                    "sampling_interval": self.config.metrics_sampling_interval,
                    "history_window_size": self.config.history_window_size
                },
                "statistics": self.get_comprehensive_stats()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"监控报告已导出: {filepath}")
            
        except Exception as e:
            logger.error(f"导出报告失败: {e}")


# 工厂函数
def create_monitoring_config(log_file_path: str = "./logs/metrics.json",
                           sampling_interval: float = 1.0) -> MonitoringConfig:
    """创建Monitoring configuration"""
    return MonitoringConfig(
        log_file_path=log_file_path,
        metrics_sampling_interval=sampling_interval
    )


def create_unified_monitor(config: Optional[MonitoringConfig] = None) -> UnifiedMonitor:
    """创建Unified monitor"""
    return UnifiedMonitor(config)


def create_graph_visualizer(log_file: str = "graph_visualization.log") -> GraphVisualizer:
    """创建图可视化器"""
    return GraphVisualizer(log_file)


def create_performance_monitor(sampling_interval: float = 1.0) -> PerformanceMonitor:
    """创建Performance monitoring器"""
    return PerformanceMonitor(sampling_interval)


def create_social_network_metrics(total_users: int = 0, active_users: int = 0, 
                                 engagement_rate: float = 0.0, **kwargs) -> SocialNetworkMetrics:
    """创建Social network metrics"""
    return SocialNetworkMetrics(
        total_users=total_users,
        active_users=active_users,
        engagement_rate=engagement_rate,
        **kwargs
    )


def quick_monitoring_demo(num_agents: int = 10, duration: int = 30):
    """快速监控演示"""
    monitor = create_unified_monitor()
    
    try:
        # 启动监控
        monitor.start()
        
        # 创建Visualization scenario
        monitor.create_visualization_scenario(num_agents)
        
        # 模拟一些指标更新
        for i in range(duration):
            metrics = create_social_network_metrics(
                total_users=100 + i,
                active_users=50 + i // 2,
                engagement_rate=0.5 + (i % 10) * 0.05,
                response_time_avg=1.0 + (i % 5) * 0.2
            )
            monitor.update_metrics(metrics)
            
            # 模拟图交互
            if i % 5 == 0:
                monitor.simulate_graph_interaction("user_1", "user_2", InteractionType.COOPERATE)
            elif i % 7 == 0:
                monitor.simulate_graph_interaction("misinfo_source_1", "user_3", InteractionType.SHARE)
            
            time.sleep(1)
        
        # 获取最终统计
        final_stats = monitor.get_comprehensive_stats()
        logger.info(f"监控演示完成，最终统计: {final_stats}")
        
        return final_stats
        
    finally:
        monitor.stop()

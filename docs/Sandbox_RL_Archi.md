# Sandbox-RL Architecture Guide

## KVCache-Centric Optimization System

Sandbox-RL introduces a revolutionary KVCache-centric optimization system that maximizes cache reuse and throughput while maintaining strict memory constraints across distributed environments.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              KVCache-Centric Optimization System                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │   Conductor     │    │   Scheduler     │    │   Scheduler     │            │
│  │                 │    │   (Prefill)     │    │  (Decoding)     │            │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘            │
│            │                      │                      │                    │
│            ▼                      ▼                      ▼                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │ Prefill Instance│    │ Prefill Pool    │    │Decoding Instance│            │
│  │                 │    │                 │    │                 │            │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘            │
│            │                      │                      │                    │
│            ▼                      ▼                      ▼                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │ Paged KVCache   │    │ Distributed     │    │ Paged KVCache   │            │
│  │ GPU/VRAM        │    │ KVCache Pool    │    │ GPU/VRAM        │            │
│  │ Local           │    │ CPU/DRAM/SSD    │    │ Local           │            │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘            │
│            │                      │                      │                    │
│            └──────────────────────┼──────────────────────┘                    │
│                                   │                                           │
│                                   ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        RDMA Inter-node Transfer                        │  │
│  │                    Distributed KVCache Pool                           │  │
│  │                         CPU/DRAM/SSD                                  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                           Optimization Goals                           │  │
│  │                                                                         │  │
│  │  Prefill Stage:                    Decoding Stage:                      │  │
│  │  • Max Cache Reuse                • Max Throughput                     │  │
│  │  • s.t. TTFT SLO                  • s.t. TBT SLO                       │  │
│  │  • Minimum MFU                    • KVCache < VRAM                     │  │
│  │  • KVCache < DRAM                                                      │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Conductor
The central orchestrator that manages the entire KVCache optimization pipeline:
- **Task Coordination**: Manages prefill and decoding instances
- **Resource Allocation**: Distributes KVCache across GPU/VRAM and CPU/DRAM/SSD
- **Load Balancing**: Ensures optimal resource utilization across nodes

### 2. Cache-Aware Prefill Scheduler
Intelligent scheduling for prefill operations:
- **Chunked Prefill**: Breaks large prompts into manageable chunks
- **Prefill Pool Management**: Maintains a pool of prefill instances
- **Cache Reuse Optimization**: Maximizes cache reuse across similar prompts

### 3. Cache-Aware Decoding Scheduler
Optimized scheduling for decoding operations:
- **Load Balancing**: Distributes decoding tasks across available instances
- **Memory Management**: Ensures KVCache stays within VRAM limits
- **Throughput Optimization**: Maximizes decoding throughput while maintaining TBT SLO

### 4. Paged KVCache System
Efficient memory management across different storage tiers:
- **GPU/VRAM**: High-speed local cache for active computations
- **CPU/DRAM**: Medium-speed cache for frequently accessed data
- **SSD**: Persistent storage for large cache pools

### 5. Distributed KVCache Pool
Multi-tier distributed cache management:
- **Inter-node Transfer**: RDMA-based high-speed data transfer
- **Cache Synchronization**: Ensures consistency across distributed nodes
- **Fault Tolerance**: Handles node failures gracefully

## Optimization Goals

### Prefill Stage Optimization
- **Primary Goal**: Maximize Cache Reuse
- **Constraints**:
  - Time to First Token (TTFT) SLO compliance
  - Minimum Model FLOPS Utilization (MFU)
  - KVCache memory usage < DRAM capacity

### Decoding Stage Optimization
- **Primary Goal**: Maximize Throughput
- **Constraints**:
  - Time Between Tokens (TBT) SLO compliance
  - KVCache memory usage < VRAM capacity

## Implementation Details

### KVCache Management
```python
class KVCacheManager:
    def __init__(self, gpu_memory_limit: int, cpu_memory_limit: int):
        self.gpu_cache = PagedKVCache("gpu", gpu_memory_limit)
        self.cpu_cache = DistributedKVCache("cpu", cpu_memory_limit)
        self.rdma_transfer = RDMATransfer()
    
    async def get_cache(self, key: str, tier: str = "auto"):
        """Retrieve KVCache from appropriate tier"""
        if tier == "auto":
            tier = self._select_optimal_tier(key)
        
        if tier == "gpu":
            return await self.gpu_cache.get(key)
        else:
            return await self.cpu_cache.get(key)
    
    async def prefetch_to_gpu(self, key: str):
        """Prefetch cache from CPU to GPU"""
        cpu_data = await self.cpu_cache.get(key)
        await self.gpu_cache.put(key, cpu_data)
```

### Cache-Aware Scheduling
```python
class CacheAwareScheduler:
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.prefill_pool = PrefillPool()
        self.decoding_pool = DecodingPool()
    
    async def schedule_prefill(self, prompt: str, priority: float):
        """Schedule prefill with cache awareness"""
        # Check cache hit rate
        cache_key = self._generate_cache_key(prompt)
        cache_hit = await self.kv_cache_manager.check_cache(cache_key)
        
        if cache_hit:
            # Use cached computation
            return await self._use_cached_prefill(cache_key)
        else:
            # Schedule new prefill
            return await self._schedule_new_prefill(prompt, priority)
    
    async def schedule_decoding(self, context: str, max_tokens: int):
        """Schedule decoding with memory constraints"""
        # Estimate memory usage
        memory_usage = self._estimate_memory_usage(context, max_tokens)
        
        if memory_usage > self.vram_limit:
            # Use distributed decoding
            return await self._schedule_distributed_decoding(context, max_tokens)
        else:
            # Use local decoding
            return await self._schedule_local_decoding(context, max_tokens)
```

### RDMA Inter-node Transfer
```python
class RDMATransfer:
    def __init__(self, node_configs: List[NodeConfig]):
        self.nodes = {config.node_id: config for config in node_configs}
        self.connections = {}
    
    async def transfer_kvcache(self, source_node: str, target_node: str, 
                             cache_data: bytes):
        """High-speed KVCache transfer between nodes"""
        connection = await self._get_connection(source_node, target_node)
        await connection.send(cache_data)
    
    async def _get_connection(self, source: str, target: str):
        """Get or create RDMA connection"""
        conn_key = f"{source}->{target}"
        if conn_key not in self.connections:
            self.connections[conn_key] = await self._create_rdma_connection(
                self.nodes[source], self.nodes[target]
            )
        return self.connections[conn_key]
```

## Performance Metrics

### Cache Efficiency Metrics
- **Cache Hit Rate**: Percentage of requests served from cache
- **Cache Reuse Ratio**: Ratio of reused cache entries to total entries
- **Memory Utilization**: Percentage of available memory used

### Throughput Metrics
- **Tokens per Second**: Decoding throughput
- **Prefill Latency**: Time to complete prefill operations
- **End-to-End Latency**: Total time from request to response

### Resource Utilization
- **GPU Utilization**: Percentage of GPU compute used
- **Memory Bandwidth**: Memory transfer rates
- **Network Bandwidth**: Inter-node transfer rates

## Integration with Sandbox-RL

The KVCache-centric optimization system integrates seamlessly with Sandbox-RL's core components:

### LLM Interface Integration
```python
class OptimizedLLMInterface:
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.scheduler = CacheAwareScheduler(kv_cache_manager)
    
    async def generate(self, prompt: str, **kwargs):
        """Generate text with KVCache optimization"""
        # Schedule prefill with cache awareness
        prefill_result = await self.scheduler.schedule_prefill(prompt, 1.0)
        
        # Schedule decoding with memory constraints
        decoding_result = await self.scheduler.schedule_decoding(
            prefill_result.context, kwargs.get('max_tokens', 100)
        )
        
        return decoding_result.text
```

### RL Framework Integration
```python
class KVCacheAwareRLFramework:
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.rl_engine = RLEngine()
    
    async def train_step(self, batch: List[Experience]):
        """Training step with KVCache optimization"""
        # Prefetch frequently used caches
        await self._prefetch_training_caches(batch)
        
        # Execute training with optimized cache access
        return await self.rl_engine.train(batch)
```

## Future Enhancements

### Advanced Cache Strategies
- **Predictive Caching**: ML-based cache prediction
- **Adaptive Cache Sizing**: Dynamic cache size adjustment
- **Multi-tier Cache**: Additional cache tiers (NVMe, etc.)

### Distributed Optimizations
- **Federated Caching**: Cross-cluster cache sharing
- **Edge Computing**: Cache distribution to edge nodes
- **Hybrid Cloud**: Cloud and on-premise cache integration

### Performance Monitoring
- **Real-time Metrics**: Live performance monitoring
- **Predictive Analytics**: Performance trend analysis
- **Automated Optimization**: Self-tuning cache parameters

## Conclusion

The KVCache-centric optimization system represents a significant advancement in distributed LLM serving, providing:

- **Maximum Cache Reuse**: Optimized cache utilization across all tiers
- **High Throughput**: Efficient decoding with strict SLO compliance
- **Resource Efficiency**: Optimal use of GPU, CPU, and network resources
- **Scalability**: Seamless scaling across multiple nodes and clusters
- **Fault Tolerance**: Robust handling of node failures and network issues

This architecture enables Sandbox-RL to achieve unprecedented performance in multi-agent reinforcement learning scenarios while maintaining strict resource constraints and service level objectives.

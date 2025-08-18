# 8GPU分布式vLLM部署故障排除指南

## 问题诊断

### 1. 编译缓存问题
**症状**: 
```
RuntimeError: vLLM failed to compile the model. The most likely reason for this is that a previous compilation failed, leading to a corrupted compilation artifact.
```

**解决方案**:
```bash
# 清理编译缓存
rm -rf ~/.cache/vllm/torch_compile_cache
rm -rf ~/.cache/torch/compiled_cache

# 使用禁用编译的启动脚本
./demo/launch_8gpu_no_compile.sh
```

### 2. GPU内存不足
**症状**:
```
ValueError: Free memory on device (63.99/79.35 GiB) on startup is less than desired GPU memory utilization (0.9, 71.41 GiB)
```

**解决方案**:
- 降低GPU内存利用率: `--gpu-memory-utilization 0.4`
- 减少最大序列长度: `--max-model-len 8192`
- 减少并发序列数: `--max-num-seqs 128`

### 3. 连接失败
**症状**:
```
Cannot connect to host localhost:8001 ssl:default [Multiple exceptions: [Errno 111] Connect call failed]
```

**解决方案**:
- 检查vLLM实例是否正在运行
- 验证端口是否被占用
- 检查防火墙设置

## 启动脚本说明

### 1. 保守配置 (推荐)
```bash
./demo/launch_8gpu_conservative.sh
```
- GPU内存利用率: 40%
- 最大序列长度: 8192
- 最大并发序列: 128
- 适合内存受限环境

### 2. 优化配置
```bash
./demo/launch_8gpu_optimized.sh
```
- GPU内存利用率: 60%
- 最大序列长度: 16384
- 最大并发序列: 256
- 需要更多GPU内存

### 3. 禁用编译模式
```bash
./demo/launch_8gpu_no_compile.sh
```
- 禁用torch.compile
- 避免编译缓存问题
- 性能可能略有下降

### 4. 清理和测试
```bash
./demo/clean_and_launch.sh
```
- 清理所有缓存
- 测试单个实例
- 验证基本功能

## 健康检查

### 检查所有实例状态
```bash
for i in {8001..8008}; do
  echo -n "端口 $i: "
  if curl -s "http://localhost:$i/health" > /dev/null 2>&1; then
    echo "✅ 健康"
  else
    echo "❌ 未响应"
  fi
done
```

### 测试API调用
```bash
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-2",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

## 性能优化建议

### 1. 内存优化
- 根据GPU内存大小调整利用率
- 减少最大序列长度
- 优化并发序列数

### 2. 网络优化
- 使用本地连接避免网络延迟
- 配置适当的超时时间
- 启用连接池

### 3. 模型优化
- 使用量化模型减少内存占用
- 启用模型缓存
- 优化批处理大小

## 常见问题

### Q: 为什么8个实例无法同时启动？
A: 可能是GPU内存不足，尝试使用保守配置或减少实例数量。

### Q: 编译失败怎么办？
A: 使用 `--disable-compilation` 参数或清理编译缓存。

### Q: 连接超时怎么处理？
A: 增加启动等待时间，检查网络配置，验证端口可用性。

### Q: 如何监控GPU使用情况？
A: 使用 `nvidia-smi` 命令实时监控GPU状态。

## 日志分析

### 关键日志位置
- 实例日志: `vllm_gpu0.log` 到 `vllm_gpu7.log`
- 系统日志: `/var/log/syslog`
- vLLM日志: 控制台输出

### 常见错误模式
1. **内存错误**: 检查GPU内存使用情况
2. **编译错误**: 清理缓存或禁用编译
3. **网络错误**: 检查端口和防火墙设置
4. **模型加载错误**: 验证模型路径和格式

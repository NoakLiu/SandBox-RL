# 多模型训练系统服务器运行指南

## 概述

本指南介绍如何在服务器环境中运行多模型训练系统，包括内存优化、存储配置和检查点管理。

## 系统要求

### 硬件要求
- **内存**: 建议32GB以上
- **CPU**: 16核心以上
- **GPU**: 支持CUDA的GPU（可选）
- **存储**: 使用本地存储或CPFS，避免NAS

### 软件要求
- Python 3.8+
- PyTorch (可选，用于GPU支持)
- psutil (用于系统监控)

## 快速开始

### 1. 安装依赖

```bash
# 安装基础依赖
pip install psutil

# 安装GPU支持（可选）
pip install torch torchvision torchaudio
```

### 2. 运行训练

```bash
# 使用服务器运行脚本
python demo/run_training_server.py
```

### 3. 查看结果

训练完成后，结果将保存在以下目录：
- `./training_outputs/` - 训练输出
- `./checkpoints/` - 检查点文件
- `./logs/` - 日志文件

## 配置说明

### 内存配置

脚本会自动设置以下环境变量来优化内存使用：

```bash
# 32GB内存限制
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32768

# 多线程配置
OMP_NUM_THREADS=16
MKL_NUM_THREADS=16
NUMEXPR_NUM_THREADS=16
```

### 存储配置

- **临时文件**: 使用 `./temp/` 目录
- **检查点**: 保存到 `./checkpoints/` 目录
- **日志**: 保存到 `./logs/` 目录
- **输出**: 保存到 `./training_outputs/` 目录

### GPU配置

如果系统有GPU，脚本会自动检测并配置：

```bash
# 使用所有可用GPU
CUDA_VISIBLE_DEVICES=0,1,2,3
```

## 运行模式

### 1. 协同训练模式
- 模型之间相互合作
- 共享资源和信息
- 适合复杂任务

### 2. 竞争训练模式
- 模型之间相互竞争
- 独立优化性能
- 适合性能对比

### 3. 组队博弈模式
- 模型分组对抗
- 团队协作与竞争
- 适合策略研究

### 4. 混合训练模式
- 结合多种训练方式
- 动态调整策略
- 适合复杂场景

## 监控和日志

### 实时监控
脚本会实时监控以下指标：
- 内存使用情况
- CPU使用率
- 磁盘使用情况
- GPU使用情况（如果可用）

### 日志文件
- `./logs/training_runner.log` - 运行器日志
- `./logs/training.log` - 训练过程日志

### 检查点
每次训练完成后会自动保存检查点，包含：
- 运行配置
- 性能指标
- 训练结果
- 日志文件

## 故障排除

### 内存不足
如果遇到内存不足问题：

1. 减少内存限制：
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:16384  # 16GB
```

2. 减少模型数量：
修改 `demo/multi_model_single_env_simple.py` 中的 `max_models` 参数

### GPU问题
如果GPU不可用：

1. 检查CUDA安装：
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

2. 禁用GPU：
```bash
export CUDA_VISIBLE_DEVICES=""
```

### 存储问题
如果存储空间不足：

1. 清理临时文件：
```bash
rm -rf ./temp/*
```

2. 清理旧检查点：
```bash
rm -rf ./checkpoints/checkpoint_*
```

## 性能优化建议

### 1. 内存优化
- 使用梯度累积减少内存使用
- 启用混合精度训练
- 定期清理缓存

### 2. 存储优化
- 使用本地SSD存储
- 避免频繁的小文件写入
- 定期压缩检查点

### 3. 计算优化
- 使用多进程训练
- 启用数据并行
- 优化数据加载

## 示例运行

### 基本运行
```bash
cd /path/to/SandGraph
python demo/run_training_server.py
```

### 后台运行
```bash
nohup python demo/run_training_server.py > training.log 2>&1 &
```

### 指定输出目录
```bash
export OUTPUT_DIR=/path/to/output
python demo/run_training_server.py
```

## 输出文件说明

### 训练结果
- `multi_model_training_simple_results.json` - 训练结果数据
- `run_report.json` - 运行报告

### 检查点
- `config.json` - 运行配置
- `performance_metrics.json` - 性能指标
- 训练结果和日志文件

### 可视化
运行可视化脚本生成图表：
```bash
python demo/multi_model_visualization.py
```

生成的图表包括：
- 折线图 - 训练趋势
- 柱状图 - 性能对比
- 雷达图 - 能力分析
- 3D热力图 - 性能分布
- 散点图 - 关系分析
- 综合仪表板

## 注意事项

1. **存储位置**: 确保使用本地存储或CPFS，避免NAS
2. **内存管理**: 监控内存使用，避免OOM
3. **检查点**: 定期保存检查点，防止数据丢失
4. **日志**: 保留完整日志用于问题诊断
5. **清理**: 定期清理临时文件和旧检查点

## 联系支持

如果遇到问题，请检查：
1. 系统资源是否充足
2. 依赖是否正确安装
3. 日志文件中的错误信息
4. 检查点文件是否完整

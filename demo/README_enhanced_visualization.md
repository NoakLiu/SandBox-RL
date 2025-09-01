# 增强版SandGraph两组对抗可视化

## 🎯 功能特性

### 1. 两组对抗传播
- **Group A (红色组)**: 倾向于相信misinformation
- **Group B (蓝色组)**: 倾向于不相信misinformation  
- **中性区域**: 中间灰色区域，用于跨组交互

### 2. 动态颜色变化 ✅
- **Group A用户**: 红色系，belief越高越红
- **Group B用户**: 蓝色系，belief越高越蓝
- **节点大小**: 根据belief值动态调整 (60-260像素)
- **颜色范围**: 0.1-1.0 RGB值，确保明显变化

### 3. 无坐标轴显示 ✅
- 完全移除x轴和y轴
- 移除所有边框线
- 纯净的可视化界面

### 4. 边类型可视化
- **橙色实线**: Misinformation传播
- **蓝色实线**: 事实核查
- **绿色虚线**: 合作
- **红色点线**: 竞争
- **紫色点划线**: 跨组传播

### 5. 实时统计信息
- 各组节点数和平均belief
- Misinformation传播次数统计
- 跨组交互数量监控
- 实时时间显示

## 📁 生成的文件

### 主要文件
- `sandgraph_competition_final.png` - 最终状态图像 (1.2MB)
- `sandgraph_animation.gif` - 完整GIF动画 (2.0MB)
- `sandgraph_animation_optimized.gif` - 优化版GIF (0.7MB)

### 时间步序列
- `timesteps/timestep_XXXX.png` - 20个时间步图像
- 每5帧保存一次，共20个文件

### 其他可视化
- `sandgraph_network_visualization.png` - 网络可视化
- `sandgraph_statistics.png` - 统计分析图

## 🚀 使用方法

### 1. 运行可视化演示
```bash
python demo/simple_visualization_demo.py
```

### 2. 查看结果
```bash
python demo/show_visualization_results.py
```

### 3. 生成GIF动画
```bash
python demo/create_animation_gif.py
```

### 4. 查看文件
```bash
# 查看最终图像
open visualization_outputs/sandgraph_competition_final.png

# 查看GIF动画
open visualization_outputs/sandgraph_animation.gif

# 查看时间步序列
ls visualization_outputs/timesteps/
```

## 🎨 技术实现

### 核心类
- `EnhancedSandGraphVisualizer` - 增强版可视化器
- `GroupType` - 组类型枚举 (GROUP_A, GROUP_B, NEUTRAL)
- `InteractionType` - 交互类型枚举 (包含CROSS_PROPAGATE)

### 动态颜色算法
```python
# Group A: 红色系
red = 0.6 + node.belief * 0.4
green = 0.1 - node.belief * 0.1
blue = 0.1 - node.belief * 0.1

# Group B: 蓝色系  
red = 0.1 - node.belief * 0.1
green = 0.1 - node.belief * 0.1
blue = 0.6 + node.belief * 0.4
```

### 交互效果增强
- **Misinformation传播**: 0.15倍强度 (原0.1)
- **事实核查**: 0.25倍强度 (原0.2)
- **跨组传播**: 0.08倍强度 (原0.05)
- **合作/竞争**: ±0.05 belief调整

### 节点大小计算
```python
node.size = 60 + int(node.belief * 200)  # 60-260像素范围
```

## 📊 可视化效果

### 初始状态
- Group A平均belief: ~0.7 (高)
- Group B平均belief: ~0.35 (低)
- 总节点数: 20 (每组10个)

### 动态变化
- 节点颜色实时变化
- 节点大小动态调整
- 边连接实时更新
- 统计信息实时刷新

### 分组背景
- 左侧淡红色区域: Group A
- 右侧淡蓝色区域: Group B
- 中间淡灰色区域: 中性区域

## 🔧 自定义选项

### 可视化参数
```python
visualizer = EnhancedSandGraphVisualizer(
    update_interval=0.3,      # 更新间隔
    save_gif=True,           # 保存GIF
    save_png_steps=True,     # 保存PNG
    png_interval=5           # PNG保存间隔
)
```

### 场景参数
```python
visualizer.create_competing_scenario(
    num_agents_per_group=10  # 每组节点数
)
```

## 🎬 动画特性

### GIF动画信息
- **帧数**: 20帧
- **帧率**: 1.5 FPS
- **总时长**: 13.3秒
- **图像尺寸**: 1349x1219像素

### 优化版本
- **帧数**: 20帧 (前20帧)
- **帧率**: 2.0 FPS
- **文件大小**: 0.7MB (vs 2.0MB)
- **图像尺寸**: 674x609像素 (下采样)

## ✅ 完成的功能

1. ✅ 两组对抗传播
2. ✅ 动态颜色变化
3. ✅ 无坐标轴显示
4. ✅ GIF动画保存
5. ✅ PNG时间步保存
6. ✅ 实时统计信息
7. ✅ 边类型可视化
8. ✅ 分组背景显示
9. ✅ 节点大小动态调整
10. ✅ 跨组交互模拟

## 🎉 总结

这个增强版可视化系统成功实现了：

- **两组对抗传播**的复杂网络动态
- **动态颜色变化**的视觉反馈
- **无坐标轴**的纯净界面
- **GIF和PNG**的完整保存功能
- **实时统计**的信息展示

所有功能都已按照要求实现并测试通过！

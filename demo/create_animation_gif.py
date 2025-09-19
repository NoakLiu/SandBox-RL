#!/usr/bin/env python3
"""
Create Animation GIF

从保存的时间步PNG图像创建GIF动画
"""

import os
import glob
from pathlib import Path
import imageio
import numpy as np

def create_gif_from_timesteps():
    """从时间步图像创建GIF动画"""
    print("🎬 创建Sandbox-RL动态可视化GIF动画")
    print("=" * 50)
    
    # 检查时间步图像目录
    timesteps_dir = Path("visualization_outputs/timesteps")
    
    if not timesteps_dir.exists():
        print("❌ 时间步图像目录不存在")
        return
    
    # 获取所有时间步PNG文件
    png_files = sorted(timesteps_dir.glob("timestep_*.png"))
    
    if not png_files:
        print("❌ 没有找到时间步图像文件")
        return
    
    print(f"📁 找到 {len(png_files)} 个时间步图像")
    
    # 读取图像
    print("📖 读取图像文件...")
    images = []
    for png_file in png_files:
        try:
            image = imageio.imread(png_file)
            images.append(image)
            print(f"  ✅ {png_file.name}")
        except Exception as e:
            print(f"  ❌ {png_file.name}: {e}")
    
    if not images:
        print("❌ 没有成功读取任何图像")
        return
    
    # 创建GIF
    gif_path = "visualization_outputs/sandgraph_animation.gif"
    print(f"\n🎬 创建GIF动画: {gif_path}")
    
    try:
        # 使用较慢的帧率以便观察变化
        imageio.mimsave(gif_path, images, fps=1.5, duration=0.67)
        
        # 获取文件大小
        file_size = Path(gif_path).stat().st_size / (1024 * 1024)
        print(f"✅ GIF创建成功! 文件大小: {file_size:.1f}MB")
        print(f"📊 动画信息:")
        print(f"  - 帧数: {len(images)}")
        print(f"  - 帧率: 1.5 FPS")
        print(f"  - 总时长: {len(images) / 1.5:.1f} 秒")
        print(f"  - 图像尺寸: {images[0].shape[1]}x{images[0].shape[0]}")
        
    except Exception as e:
        print(f"❌ 创建GIF失败: {e}")
        return
    
    print(f"\n🎯 GIF动画特性:")
    print("  • 无坐标轴显示")
    print("  • 动态颜色变化")
    print("  • 两组对抗传播")
    print("  • 实时统计信息")
    print("  • 边类型可视化")
    
    print(f"\n📱 查看GIF:")
    print(f"  open {gif_path}")
    print(f"  或在浏览器中打开: file://{os.path.abspath(gif_path)}")

def create_optimized_gif():
    """创建优化版本的GIF（更小文件大小）"""
    print("\n🔧 创建优化版本GIF...")
    
    timesteps_dir = Path("visualization_outputs/timesteps")
    png_files = sorted(timesteps_dir.glob("timestep_*.png"))
    
    if not png_files:
        return
    
    # 读取图像并调整大小
    images = []
    for png_file in png_files[:20]:  # 只取前20帧以减小文件大小
        try:
            image = imageio.imread(png_file)
            # 调整图像大小
            height, width = image.shape[:2]
            new_width = width // 2
            new_height = height // 2
            # 简单的下采样
            image_small = image[::2, ::2, :]
            images.append(image_small)
        except Exception as e:
            print(f"  ❌ 处理 {png_file.name}: {e}")
    
    if images:
        optimized_gif_path = "visualization_outputs/sandgraph_animation_optimized.gif"
        try:
            imageio.mimsave(optimized_gif_path, images, fps=2.0, duration=0.5)
            file_size = Path(optimized_gif_path).stat().st_size / (1024 * 1024)
            print(f"✅ 优化GIF创建成功! 文件大小: {file_size:.1f}MB")
        except Exception as e:
            print(f"❌ 创建优化GIF失败: {e}")

def main():
    """主函数"""
    print("🚀 Sandbox-RL动态可视化GIF生成器")
    print("=" * 60)
    
    # 检查依赖
    try:
        import imageio
        print("✅ imageio库可用")
    except ImportError:
        print("❌ 需要安装imageio库: pip install imageio")
        return
    
    # 创建标准GIF
    create_gif_from_timesteps()
    
    # 创建优化版本
    create_optimized_gif()
    
    print(f"\n🎉 GIF生成完成!")
    print("所有GIF文件已保存到 visualization_outputs/ 目录")

if __name__ == "__main__":
    main()

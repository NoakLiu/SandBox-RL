#!/usr/bin/env python3
"""
Show Visualization Results

展示可视化结果和说明
"""

import os
import glob
from pathlib import Path

def main():
    """展示可视化结果"""
    print("🎨 Sandbox-RL两组对抗可视化结果展示")
    print("=" * 60)
    
    # 检查生成的文件
    visualization_dir = Path("visualization_outputs")
    
    if not visualization_dir.exists():
        print("❌ 可视化输出目录不存在，请先运行可视化演示")
        return
    
    print("📁 生成的文件:")
    print("-" * 40)
    
    # 检查主要图像文件
    main_files = [
        "sandgraph_competition_final.png",
        "sandgraph_network_visualization.png", 
        "sandgraph_statistics.png"
    ]
    
    for file_name in main_files:
        file_path = visualization_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_name} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {file_name} (未找到)")
    
    # 检查时间步图像
    timesteps_dir = visualization_dir / "timesteps"
    if timesteps_dir.exists():
        png_files = list(timesteps_dir.glob("timestep_*.png"))
        if png_files:
            print(f"✅ timesteps/ 目录包含 {len(png_files)} 个时间步图像")
            print(f"   时间范围: {png_files[0].stem} 到 {png_files[-1].stem}")
        else:
            print("❌ timesteps/ 目录为空")
    else:
        print("❌ timesteps/ 目录不存在")
    
    print("\n📊 可视化特性说明:")
    print("-" * 40)
    print("🎯 两组对抗传播:")
    print("  • Group A (红色组): 倾向于相信misinformation")
    print("  • Group B (蓝色组): 倾向于不相信misinformation")
    print("  • 中性区域: 中间灰色区域")
    
    print("\n🎨 动态颜色变化:")
    print("  • Group A用户: 红色系，belief越高越红")
    print("  • Group B用户: 蓝色系，belief越高越蓝")
    print("  • 节点大小: 根据belief值动态调整")
    
    print("\n🔗 边类型和样式:")
    print("  • 橙色实线: Misinformation传播")
    print("  • 蓝色实线: 事实核查")
    print("  • 绿色虚线: 合作")
    print("  • 红色点线: 竞争")
    print("  • 紫色点划线: 跨组传播")
    
    print("\n📈 统计信息:")
    print("  • 实时显示各组节点数和平均belief")
    print("  • 跟踪misinformation传播次数")
    print("  • 监控跨组交互数量")
    
    print("\n💾 保存的文件:")
    print("  • 最终状态图像: sandgraph_competition_final.png")
    print("  • 时间步序列: timesteps/timestep_XXXX.png")
    print("  • 网络可视化: sandgraph_network_visualization.png")
    print("  • 统计分析: sandgraph_statistics.png")
    
    print("\n🎬 如何使用:")
    print("-" * 40)
    print("1. 查看最终状态:")
    print("   open visualization_outputs/sandgraph_competition_final.png")
    
    print("\n2. 查看时间步序列:")
    print("   ls visualization_outputs/timesteps/")
    
    print("\n3. 创建GIF动画 (可选):")
    print("   # 使用ImageMagick或其他工具")
    print("   convert timesteps/timestep_*.png sandgraph_animation.gif")
    
    print("\n4. 重新运行可视化:")
    print("   python demo/simple_visualization_demo.py")
    
    print("\n✅ 可视化演示完成！")
    print("所有图像文件已保存到 visualization_outputs/ 目录")


if __name__ == "__main__":
    main()

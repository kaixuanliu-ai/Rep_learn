#!/usr/bin/env python3
"""
训练监控脚本 - 用于监控长时间的H100训练
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def monitor_results(result_path, check_interval=30):
    """监控训练结果"""
    print(f"开始监控训练结果: {result_path}")
    print(f"检查间隔: {check_interval}秒")
    print("按 Ctrl+C 停止监控")
    
    try:
        iteration = 0
        while True:
            print(f"\n=== 检查 #{iteration+1} ===")
            
            # 检查权重文件
            weight_files = list(Path(result_path).glob("W_*.pth"))
            if weight_files:
                print(f"✅ 发现 {len(weight_files)} 个权重文件")
                latest_weight = max(weight_files, key=os.path.getctime)
                weight_time = time.ctime(os.path.getctime(latest_weight))
                print(f"   最新权重: {latest_weight.name} ({weight_time})")
            else:
                print("❌ 未发现权重文件")
            
            # 检查phi文件
            phi_files = list(Path(result_path).glob("phi_*.pth"))
            if phi_files:
                print(f"✅ 发现 {len(phi_files)} 个特征文件")
            else:
                print("❌ 未发现特征文件")
            
            # 检查counts文件
            counts_file = Path(result_path) / "counts.npy"
            if counts_file.exists():
                try:
                    counts = np.load(counts_file)
                    print(f"✅ counts形状: {counts.shape}")
                    print(f"   状态访问统计: 状态0={counts[-1,0]}, 状态1={counts[-1,1]}")
                except Exception as e:
                    print(f"❌ 读取counts失败: {e}")
            else:
                print("❌ 未发现counts文件")
            
            # 检查loss图像
            loss_files = list(Path(result_path).glob("*_loss.pdf"))
            if loss_files:
                print(f"✅ 发现 {len(loss_files)} 个损失图像")
            
            print(f"   结果目录大小: {get_dir_size(result_path):.2f} MB")
            
            time.sleep(check_interval)
            iteration += 1
            
    except KeyboardInterrupt:
        print(f"\n监控停止。共进行了 {iteration+1} 次检查。")

def get_dir_size(path):
    """计算目录大小（MB）"""
    total = 0
    try:
        for entry in Path(path).rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        return 0
    return total / (1024 * 1024)

def compare_results(cmdp_path, original_path):
    """比较CMDP和原版的训练结果"""
    print("\n=== 训练结果对比 ===")
    
    paths = {"CMDP": cmdp_path, "Original": original_path}
    
    for name, path in paths.items():
        print(f"\n{name} 版本:")
        if not Path(path).exists():
            print("  ❌ 结果目录不存在")
            continue
            
        # 权重文件数量
        weight_files = list(Path(path).glob("W_*.pth"))
        print(f"  权重文件: {len(weight_files)}/100")
        
        # counts分析
        counts_file = Path(path) / "counts.npy"
        if counts_file.exists():
            try:
                counts = np.load(counts_file)
                total_visits = counts.sum()
                print(f"  总访问次数: {total_visits}")
                print(f"  最后状态访问: {counts[-1,:2]}")
            except:
                print("  ❌ counts读取失败")
        else:
            print("  ❌ 无counts文件")
        
        # 目录大小
        size = get_dir_size(path)
        print(f"  目录大小: {size:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--path', type=str, help='要监控的结果路径')
    parser.add_argument('--compare', action='store_true', help='比较CMDP和原版结果')
    parser.add_argument('--interval', type=int, default=60, help='检查间隔（秒）')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results("h100_cmdp_results", "h100_original_results")
    elif args.path:
        monitor_results(args.path, args.interval)
    else:
        print("用法示例:")
        print("  python monitor_training.py --path h100_cmdp_results")
        print("  python monitor_training.py --compare")
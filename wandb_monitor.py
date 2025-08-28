#!/usr/bin/env python3
"""
WandB实时监控脚本
"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wandb

class WandBMonitor:
    def __init__(self, entity, projects, refresh_interval=60):
        self.entity = entity
        self.projects = projects if isinstance(projects, list) else [projects]
        self.refresh_interval = refresh_interval
        self.api = wandb.Api()
        
    def get_runs(self, project, filters=None):
        """获取项目的运行"""
        try:
            runs = self.api.runs(f"{self.entity}/{project}", filters=filters)
            return list(runs)
        except Exception as e:
            print(f"❌ 获取运行失败 {project}: {e}")
            return []
    
    def get_latest_run(self, project):
        """获取最新的运行"""
        runs = self.get_runs(project)
        if runs:
            return runs[0]  # wandb.Api().runs() 默认按时间排序
        return None
    
    def get_run_metrics(self, run, metrics=None):
        """获取运行的指标数据"""
        if metrics is None:
            metrics = ['eval', 'rep_learn_time', 'lsvi_time', 'reached']
        
        data = {}
        try:
            history = run.scan_history()
            for row in history:
                for metric in metrics:
                    if metric in row:
                        if metric not in data:
                            data[metric] = []
                        data[metric].append(row[metric])
        except Exception as e:
            print(f"❌ 获取指标失败 {run.name}: {e}")
            
        return data
    
    def print_run_status(self, run):
        """打印运行状态"""
        print(f"\n📊 运行状态: {run.name}")
        print(f"   状态: {run.state}")
        print(f"   开始时间: {run.created_at}")
        print(f"   持续时间: {datetime.now() - run.created_at}")
        print(f"   URL: {run.url}")
        
        # 获取最新指标
        summary = run.summary
        key_metrics = ['eval', 'reached', 'rep_learn_time', 'lsvi_time', 'episode']
        for metric in key_metrics:
            if metric in summary:
                print(f"   {metric}: {summary[metric]}")
    
    def compare_runs(self):
        """比较不同项目的运行"""
        print(f"\n🔄 比较项目运行状态")
        print("=" * 60)
        
        for project in self.projects:
            print(f"\n📁 项目: {project}")
            latest_run = self.get_latest_run(project)
            
            if latest_run:
                self.print_run_status(latest_run)
                
                # 获取关键指标
                data = self.get_run_metrics(latest_run)
                if 'eval' in data and data['eval']:
                    eval_scores = data['eval']
                    current_score = eval_scores[-1] if eval_scores else 0
                    max_score = max(eval_scores) if eval_scores else 0
                    print(f"   评估进度: 当前 {current_score:.3f}, 最高 {max_score:.3f}")
                    
                    if len(eval_scores) >= 2:
                        trend = eval_scores[-1] - eval_scores[-2]
                        trend_symbol = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                        print(f"   趋势: {trend_symbol} {trend:+.3f}")
            else:
                print("   ❌ 没有找到运行")
    
    def monitor_convergence(self, target_eval=0.8, patience=10):
        """监控收敛情况"""
        print(f"\n🎯 收敛监控 (目标: eval >= {target_eval})")
        
        for project in self.projects:
            print(f"\n检查项目: {project}")
            latest_run = self.get_latest_run(project)
            
            if not latest_run:
                print("   ❌ 无活跃运行")
                continue
                
            data = self.get_run_metrics(latest_run)
            
            if 'eval' not in data or not data['eval']:
                print("   ❌ 无评估数据")
                continue
                
            eval_scores = data['eval']
            current_score = eval_scores[-1]
            
            # 检查是否达到目标
            if current_score >= target_eval:
                print(f"   ✅ 已收敛！当前评分: {current_score:.3f}")
                continue
            
            # 检查收敛趋势
            if len(eval_scores) >= patience:
                recent_scores = eval_scores[-patience:]
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                if trend > 0.001:
                    print(f"   📈 正在收敛，趋势: +{trend:.4f}")
                elif abs(trend) <= 0.001:
                    print(f"   ⚠️ 收敛停滞，可能需要调参")
                else:
                    print(f"   📉 性能下降，趋势: {trend:.4f}")
            
            print(f"   📊 当前: {current_score:.3f}, 目标: {target_eval}, 差距: {target_eval - current_score:.3f}")
    
    def plot_comparison(self, save_path="wandb_comparison.png"):
        """绘制对比图表"""
        plt.figure(figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, project in enumerate(self.projects):
            latest_run = self.get_latest_run(project)
            if not latest_run:
                continue
                
            data = self.get_run_metrics(latest_run)
            
            # 绘制eval曲线
            plt.subplot(2, 2, 1)
            if 'eval' in data and data['eval']:
                eval_data = data['eval']
                plt.plot(eval_data, label=f"{project} (latest: {eval_data[-1]:.3f})", 
                        color=colors[i % len(colors)])
            plt.title('Evaluation Return')
            plt.xlabel('Episodes')
            plt.ylabel('Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制reached曲线
            plt.subplot(2, 2, 2)
            if 'reached' in data and data['reached']:
                plt.plot(data['reached'], label=project, color=colors[i % len(colors)])
            plt.title('Reached Time Steps')
            plt.xlabel('Episodes')
            plt.ylabel('Max Reached Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 绘制训练时间
            plt.subplot(2, 2, 3)
            if 'rep_learn_time' in data and data['rep_learn_time']:
                plt.plot(data['rep_learn_time'], label=f"{project} Rep", 
                        color=colors[i % len(colors)], linestyle='--')
            if 'lsvi_time' in data and data['lsvi_time']:
                plt.plot(data['lsvi_time'], label=f"{project} LSVI", 
                        color=colors[i % len(colors)], linestyle=':')
        
        plt.subplot(2, 2, 3)
        plt.title('Training Time per Update')
        plt.xlabel('Episodes')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 对比图已保存: {save_path}")
        
    def run_monitoring(self):
        """运行持续监控"""
        print(f"🚀 开始WandB监控")
        print(f"   实体: {self.entity}")
        print(f"   项目: {', '.join(self.projects)}")
        print(f"   刷新间隔: {self.refresh_interval}秒")
        print(f"   按 Ctrl+C 停止监控")
        
        try:
            iteration = 0
            while True:
                print(f"\n{'='*20} 监控检查 #{iteration+1} {'='*20}")
                print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 比较运行状态
                self.compare_runs()
                
                # 监控收敛
                self.monitor_convergence()
                
                # 每隔几次生成对比图
                if iteration % 5 == 0:
                    self.plot_comparison()
                
                print(f"\n⏰ 等待 {self.refresh_interval} 秒...")
                time.sleep(self.refresh_interval)
                iteration += 1
                
        except KeyboardInterrupt:
            print(f"\n🛑 监控停止，共进行了 {iteration+1} 次检查")
            print("📊 生成最终对比图...")
            self.plot_comparison("final_wandb_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='WandB训练监控工具')
    parser.add_argument('--entity', type=str, required=True, help='WandB实体名称')
    parser.add_argument('--projects', type=str, nargs='+', 
                       default=['briee_cmdp_h100', 'briee_original_h100'],
                       help='要监控的项目列表')
    parser.add_argument('--interval', type=int, default=120, help='监控间隔（秒）')
    parser.add_argument('--once', action='store_true', help='只运行一次检查')
    
    args = parser.parse_args()
    
    monitor = WandBMonitor(args.entity, args.projects, args.interval)
    
    if args.once:
        monitor.compare_runs()
        monitor.monitor_convergence()
        monitor.plot_comparison()
    else:
        monitor.run_monitoring()

if __name__ == "__main__":
    main()
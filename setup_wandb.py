#!/usr/bin/env python3
"""
WandB设置和配置助手
"""
import os
import subprocess
import sys
import wandb
from pathlib import Path

def setup_wandb():
    """设置WandB配置"""
    print("🔧 WandB 配置助手")
    print("=" * 50)
    
    # 1. 检查wandb安装
    try:
        import wandb
        print("✅ WandB已安装")
    except ImportError:
        print("❌ WandB未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb
        print("✅ WandB安装完成")
    
    # 2. 检查登录状态
    try:
        api = wandb.Api()
        user = api.viewer
        print(f"✅ 已登录WandB，用户: {user.username}")
        print(f"   实体: {user.username}")
        return user.username
    except wandb.errors.AuthenticationError:
        print("❌ 未登录WandB")
        
        # 尝试自动登录
        api_key = input("请输入您的WandB API Key (或按Enter跳过): ").strip()
        if api_key:
            try:
                wandb.login(key=api_key)
                api = wandb.Api()
                user = api.viewer
                print(f"✅ 登录成功，用户: {user.username}")
                return user.username
            except Exception as e:
                print(f"❌ 登录失败: {e}")
        
        print("请手动运行: wandb login")
        return None

def test_wandb_logging():
    """测试WandB日志记录"""
    print("\n🧪 测试WandB日志记录...")
    
    try:
        # 创建测试项目
        with wandb.init(
            project="briee_test",
            job_type="setup_test",
            name="wandb_setup_test",
            tags=["test", "setup"]
        ) as run:
            
            # 记录一些测试数据
            for i in range(10):
                run.log({
                    "test_metric": i * 0.1,
                    "test_loss": 1.0 / (i + 1),
                    "step": i
                })
            
            print(f"✅ 测试日志记录成功")
            print(f"   运行URL: {run.url}")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def create_wandb_config():
    """创建WandB配置文件"""
    print("\n📝 创建WandB配置...")
    
    config_content = """
# WandB 配置文件
# 将此文件加载到您的环境中：source wandb_config.sh

# CMDP项目配置
export WANDB_PROJECT_CMDP="briee_cmdp_h100"
export WANDB_PROJECT_ORIGINAL="briee_original_h100"

# 实体配置（替换为您的用户名）
export WANDB_ENTITY="your_username"

# 其他配置
export WANDB_CONSOLE="off"  # 减少控制台输出
export WANDB_SILENT="true"  # 静默模式
# export WANDB_MODE="offline"  # 取消注释以启用离线模式

echo "WandB配置已加载："
echo "  CMDP项目: $WANDB_PROJECT_CMDP"
echo "  原版项目: $WANDB_PROJECT_ORIGINAL"
echo "  实体: $WANDB_ENTITY"
"""
    
    config_file = Path("wandb_config.sh")
    config_file.write_text(config_content)
    print(f"✅ 配置文件已创建: {config_file}")
    print("   使用方法: source wandb_config.sh")

def show_monitoring_guide():
    """显示WandB监控指南"""
    print(f"""
📊 WandB监控指南
{'='*50}

🎯 项目访问:
   CMDP版本: https://wandb.ai/[your-username]/briee_cmdp_h100
   原版比较: https://wandb.ai/[your-username]/briee_original_h100

📈 关键指标监控:
   • eval: 评估回报（主要收敛指标）
   • rep_learn_time: 表示学习时间
   • lsvi_time: 策略学习时间
   • reached: 最远到达的时间步
   • state 0/1: 状态访问统计
   • cmdp_enabled: CMDP是否启用
   • cmdp_b: CMDP约束参数

🔍 监控要点:
   1. eval指标应该逐步上升至0.8-1.0
   2. reached应该逐步增加至100
   3. state访问应该相对均衡
   4. 训练时间应该稳定

⚙️ 自定义面板建议:
   • 创建对比面板同时显示CMDP和原版
   • 设置eval指标的报警阈值
   • 监控训练效率指标（时间）

🚨 异常检测:
   • eval长时间不提升
   • reached停止增长
   • 训练时间异常增长
   • 状态访问极度不平衡
""")

def main():
    """主函数"""
    print("🚀 开始WandB设置...")
    
    # 1. 设置WandB
    username = setup_wandb()
    
    # 2. 测试日志记录
    if username:
        test_success = test_wandb_logging()
        if test_success:
            print("✅ WandB设置完成！")
        else:
            print("⚠️ WandB设置有问题，请检查网络连接")
    else:
        print("⚠️ 请先完成WandB登录")
    
    # 3. 创建配置文件
    create_wandb_config()
    
    # 4. 显示监控指南
    show_monitoring_guide()
    
    # 5. 提供下一步建议
    print(f"""
🎯 下一步操作:
{'='*30}
1. 编辑 wandb_config.sh 文件，设置正确的用户名
2. 加载配置: source wandb_config.sh
3. 运行训练: bash run_h100_cmdp.sh
4. 在浏览器中监控: https://wandb.ai

💡 提示: 
   • 可以同时运行CMDP和原版进行对比
   • 使用WandB的Compare功能对比不同运行
   • 设置邮件/Slack通知获取训练状态更新
""")

if __name__ == "__main__":
    main()
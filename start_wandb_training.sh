#!/bin/bash

# 一键启动WandB监控训练脚本
# 用法: bash start_wandb_training.sh [wandb_entity]

WANDB_ENTITY=${1:-"your_username"}

echo "🚀 启动带WandB监控的H100训练"
echo "=================================================="
echo "WandB实体: $WANDB_ENTITY"
echo ""

# 1. 设置WandB配置
echo "1️⃣ 设置WandB配置..."
if [ "$WANDB_ENTITY" = "your_username" ]; then
    echo "⚠️ 请提供您的WandB用户名："
    echo "   用法: bash start_wandb_training.sh your_wandb_username"
    exit 1
fi

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
fi

# 2. 验证WandB登录
echo ""
echo "2️⃣ 验证WandB登录..."
python -c "import wandb; api = wandb.Api(); print(f'✅ 已登录，用户: {api.viewer.username}')" 2>/dev/null || {
    echo "❌ WandB未登录，请运行："
    echo "   wandb login"
    echo "   或运行: python setup_wandb.py"
    exit 1
}

# 3. 启动训练（后台运行）
echo ""
echo "3️⃣ 启动训练..."
echo "🟦 启动CMDP版本训练（后台）..."
nohup bash run_h100_cmdp.sh h100_cmdp_results 10 $WANDB_ENTITY > cmdp_training.log 2>&1 &
CMDP_PID=$!
echo "   CMDP训练PID: $CMDP_PID"

echo "🟨 等待5秒后启动原版训练..."
sleep 5

echo "🟩 启动原版BRIEE训练（后台）..."
nohup bash run_h100_original.sh h100_original_results 10 $WANDB_ENTITY > original_training.log 2>&1 &
ORIGINAL_PID=$!
echo "   原版训练PID: $ORIGINAL_PID"

# 4. 保存进程信息
echo ""
echo "4️⃣ 保存训练信息..."
cat > training_info.txt << EOF
训练启动信息
启动时间: $(date)
WandB实体: $WANDB_ENTITY

进程信息:
CMDP训练PID: $CMDP_PID
原版训练PID: $ORIGINAL_PID

日志文件:
CMDP日志: cmdp_training.log
原版日志: original_training.log

WandB项目:
CMDP: https://wandb.ai/$WANDB_ENTITY/briee_cmdp_h100
原版: https://wandb.ai/$WANDB_ENTITY/briee_original_h100

停止训练:
kill $CMDP_PID $ORIGINAL_PID
EOF

echo "✅ 训练信息已保存到 training_info.txt"

# 5. 启动WandB监控（可选）
echo ""
echo "5️⃣ 启动WandB监控..."
read -p "是否启动实时WandB监控？(y/N): " start_monitor

if [ "$start_monitor" = "y" ] || [ "$start_monitor" = "Y" ]; then
    echo "🖥️ 启动WandB监控..."
    python wandb_monitor.py --entity $WANDB_ENTITY --interval 120
else
    echo "⏭️ 跳过监控，您可以稍后运行："
    echo "   python wandb_monitor.py --entity $WANDB_ENTITY"
fi

echo ""
echo "🎉 训练已启动！"
echo ""
echo "📊 监控方式："
echo "1. WandB网页: https://wandb.ai/$WANDB_ENTITY/"
echo "2. 本地监控: python wandb_monitor.py --entity $WANDB_ENTITY"
echo "3. 日志文件: tail -f cmdp_training.log"
echo "4. 进程状态: ps aux | grep python"
echo ""
echo "🛑 停止训练:"
echo "   kill $CMDP_PID $ORIGINAL_PID"
echo "   或运行: pkill -f 'python main.py'"
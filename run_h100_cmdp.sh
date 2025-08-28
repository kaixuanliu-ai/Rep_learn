#!/bin/bash

# Horizon=100 CMDP-BRIEE 优化配置
# 针对长时间步的优化参数设置

SAVE_PATH=${1:-"h100_cmdp_results"}
NUM_THREADS=${2:-10}
WANDB_ENTITY=${3:-$(whoami)}

echo "运行 Horizon=100 CMDP-BRIEE（修复版本）"
echo "优化配置用于长时间步训练"
echo "参数设置:"
echo "  Horizon: 100"
echo "  线程数: $NUM_THREADS"
echo "  保存路径: $SAVE_PATH"
echo "  WandB项目: briee_cmdp_h100"
echo "  WandB实体: $WANDB_ENTITY"
echo "  预计训练时间: 2-4小时"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "已激活虚拟环境"
fi

# WandB配置
export WANDB_PROJECT="briee_cmdp_h100"
export WANDB_ENTITY=$WANDB_ENTITY
# export WANDB_MODE=offline  # 注释掉离线模式，启用在线监控

echo "WandB配置:"
echo "  项目: $WANDB_PROJECT"
echo "  实体: $WANDB_ENTITY"

# 运行 H100 CMDP-BRIEE
python main.py \
    --horizon 100 \
    --num_threads $NUM_THREADS \
    --temp_path $SAVE_PATH \
    --enable_cmdp True \
    --cmdp_b 0.1 \
    --exp_name "cmdp_briee_h100_fixed" \
    --num_envs 50 \
    --num_episodes 10000000 \
    --batch_size 512 \
    --update_frequency 3 \
    --rep_num_update 30 \
    --rep_num_feature_update 64 \
    --rep_num_adv_update 64 \
    --hidden_dim 256 \
    --recent_size 10000 \
    --lsvi_recent_size 10000 \
    --discriminator_lr 1e-2 \
    --feature_lr 1e-2 \
    --linear_lr 1e-2

echo "Horizon=100 CMDP-BRIEE 训练完成！"
echo "结果保存在: $SAVE_PATH"
#!/bin/bash

# CMDP-BRIEE 运行脚本
# 使用 CMDP 约束的 BRIEE 算法

# 设置默认参数
HORIZON=${1:-20}
NUM_THREADS=${2:-4}
SAVE_PATH=${3:-"cmdp_results"}

echo "运行 CMDP-BRIEE 算法（修复版本）"
echo "说明: CMDP约束仅在评估阶段生效，训练期间保持原BRIEE逻辑"
echo "参数设置:"
echo "  Horizon: $HORIZON"
echo "  线程数: $NUM_THREADS"
echo "  保存路径: $SAVE_PATH"

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "已激活虚拟环境"
fi

# 运行 CMDP-BRIEE（修复版本）
python main.py \
    --horizon $HORIZON \
    --num_threads $NUM_THREADS \
    --temp_path $SAVE_PATH \
    --enable_cmdp True \
    --cmdp_b 0.1 \
    --exp_name "cmdp_briee_fixed_h${HORIZON}_b0.1" \
    --num_envs 30 \
    --num_episodes 5000000 \
    --batch_size 256 \
    --update_frequency 3 \
    --rep_num_update 20 \
    --rep_num_feature_update 32 \
    --rep_num_adv_update 32

echo "CMDP-BRIEE（修复版本）训练完成！"
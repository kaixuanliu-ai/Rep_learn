#!/bin/bash

# 原始 BRIEE 运行脚本（用于对比）
# 不使用 CMDP 约束的原始 BRIEE 算法

# 设置默认参数
HORIZON=${1:-20}
NUM_THREADS=${2:-4}
SAVE_PATH=${3:-"original_results"}

echo "运行原始 BRIEE 算法"
echo "参数设置:"
echo "  Horizon: $HORIZON"
echo "  线程数: $NUM_THREADS"
echo "  保存路径: $SAVE_PATH"

# 激活虚拟环境（如果存在）
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "已激活虚拟环境"
fi

# 运行原始 BRIEE
python main.py \
    --horizon $HORIZON \
    --num_threads $NUM_THREADS \
    --temp_path $SAVE_PATH \
    --enable_cmdp False \
    --exp_name "original_briee_h${HORIZON}" \
    --num_envs 20 \
    --num_episodes 1000000 \
    --batch_size 256 \
    --update_frequency 5 \
    --rep_num_update 15 \
    --rep_num_feature_update 32 \
    --rep_num_adv_update 32

echo "原始 BRIEE 训练完成！"
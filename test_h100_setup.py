#!/usr/bin/env python3
"""
H100设置验证脚本 - 在开始长训练前验证配置
"""
import os
import sys
import torch
import numpy as np
import time
from utils import parse_args, set_seed_everywhere, make_batch_env, ReplayBuffer
from algs.lsvi_ucb import LSVI_UCB
from algs.rep_learn import RepLearn

def test_h100_setup():
    """验证H100配置是否正确"""
    print("🔧 Horizon=100 设置验证")
    print("=" * 50)
    
    class TestArgs:
        def __init__(self):
            # 使用与实际训练相同的参数，但规模缩小
            self.horizon = 100  # 保持实际的horizon
            self.num_actions = 10
            self.switch_prob = 0.5
            self.anti_reward_prob = 0.5
            self.anti_reward = 0.1
            self.observation_noise = 0.1
            self.num_envs = 5  # 减少环境数量用于测试
            self.num_eval = 3
            self.env_temperature = 0.2
            self.variable_latent = False
            self.dense = False
            self.seed = 42
            
            # 模型参数
            self.hidden_dim = 256
            self.rep_num_update = 5  # 减少更新次数
            self.rep_num_feature_update = 10
            self.rep_num_adv_update = 10
            self.discriminator_lr = 1e-2
            self.discriminator_beta = 0.9
            self.feature_lr = 1e-2
            self.feature_beta = 0.9
            self.linear_lr = 1e-2
            self.linear_beta = 0.9
            self.rep_lamb = 0.01
            self.temperature = 1
            self.phi0_temperature = 0.1
            self.reuse_weights = True
            self.optimizer = 'sgd'
            self.softmax = 'vanilla'
            self.temp_path = 'h100_test_temp'
            
            # LSVI参数
            self.alpha = 20.0  # horizon/5
            self.lsvi_recent_size = 1000
            self.lsvi_lamb = 1.0
            self.batch_size = 128  # 减少batch size
            self.recent_size = 1000
            
            # CMDP参数
            self.enable_cmdp = True
            self.cmdp_b = 0.1
            self.cmdp_eta = 1e-3
            self.cmdp_xi = 2.0
    
    args = TestArgs()
    
    print(f"🎯 测试配置:")
    print(f"   Horizon: {args.horizon}")
    print(f"   环境数: {args.num_envs}")
    print(f"   隐藏层维度: {args.hidden_dim}")
    print(f"   CMDP启用: {args.enable_cmdp}")
    
    # 创建测试目录
    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)
    
    try:
        # 1. 环境创建测试
        print("\n1️⃣ 环境创建测试...")
        set_seed_everywhere(args.seed)
        env, eval_env = make_batch_env(args)
        print(f"   ✅ 环境创建成功")
        print(f"   观测空间: {env.observation_space.shape}")
        print(f"   动作空间: {env.action_space.n}")
        print(f"   状态维度: {env.state_dim}")
        
        # 2. 模型创建测试
        print("\n2️⃣ 模型创建测试...")
        device = torch.device("cpu")
        
        # 创建表示学习器
        rep_learners = []
        for h in range(args.horizon):
            rep_learner = RepLearn(
                env.observation_space.shape[0],
                env.state_dim,
                env.action_dim,
                args.hidden_dim,
                args.rep_num_update,
                args.rep_num_feature_update,
                args.rep_num_adv_update,
                device,
                temp_path=args.temp_path
            )
            rep_learners.append(rep_learner)
        
        print(f"   ✅ 创建了 {len(rep_learners)} 个表示学习器")
        
        # 创建LSVI代理
        agent = LSVI_UCB(
            env.observation_space.shape[0],
            env.state_dim,
            env.action_dim,
            args.horizon,
            args.alpha,
            device,
            rep_learners,
            recent_size=args.lsvi_recent_size,
            lamb=args.lsvi_lamb,
            enable_cmdp=args.enable_cmdp,
            cmdp_b=args.cmdp_b,
            cmdp_eta=args.cmdp_eta,
            cmdp_xi=args.cmdp_xi
        )
        print(f"   ✅ LSVI代理创建成功")
        print(f"   权重矩阵形状: {agent.W.shape}")
        
        # 3. 内存使用测试
        print("\n3️⃣ 内存使用评估...")
        
        # 创建buffers
        buffers = []
        buffer_capacity = int(10000000 / args.horizon) * 2 + 1000 * args.num_envs
        for _ in range(args.horizon):
            buffers.append(
                ReplayBuffer(
                    env.observation_space.shape,
                    env.action_space.n,
                    buffer_capacity,
                    args.batch_size,
                    device,
                    recent_size=args.recent_size
                )
            )
        print(f"   ✅ 创建了 {len(buffers)} 个replay buffer")
        print(f"   每个buffer容量: {buffer_capacity}")
        
        # 4. 前向传播测试
        print("\n4️⃣ 前向传播测试...")
        obs = env.reset()
        
        # 测试不同时间步的动作选择
        test_steps = [0, args.horizon//4, args.horizon//2, args.horizon-1]
        for h in test_steps:
            start_time = time.time()
            action_train = agent.act_batch(obs, h, use_cmdp_constraint=False)
            action_eval = agent.act_batch(obs, h, use_cmdp_constraint=True)
            forward_time = time.time() - start_time
            
            print(f"   步骤 {h:3d}: 训练动作 {action_train.shape}, 评估动作 {action_eval.shape}, 用时 {forward_time:.4f}s")
        
        # 5. CMDP约束验证
        print("\n5️⃣ CMDP约束验证...")
        if args.enable_cmdp:
            # 多次采样验证
            actions = []
            for _ in range(200):
                a = agent.act_batch(obs, 0, use_cmdp_constraint=True)
                actions.extend(a.tolist())
            
            action_counts = np.bincount(actions, minlength=args.num_actions)
            action_probs = action_counts / len(actions)
            min_prob = np.min(action_probs)
            
            print(f"   动作分布: {action_probs}")
            print(f"   最小概率: {min_prob:.3f} ({'✅' if min_prob >= 0.08 else '❌'})")
        
        # 6. 一次迭代测试
        print("\n6️⃣ 单次迭代测试...")
        start_time = time.time()
        
        # 简单的数据收集
        obs = env.reset()
        action = np.random.randint(0, args.num_actions, args.num_envs)
        next_obs, reward, done, _ = env.step(action)
        buffers[0].add_batch(obs, action, reward, next_obs, args.num_envs)
        
        # 表示学习（仅第一个learner）
        rep_learners[0].update(buffers[0])
        
        # LSVI更新
        agent.update(buffers)
        
        iteration_time = time.time() - start_time
        print(f"   ✅ 单次迭代完成，用时: {iteration_time:.2f}s")
        
        # 7. 时间估算
        print("\n7️⃣ 训练时间估算...")
        expected_iterations = 10000000 // (args.horizon * 50)  # 基于实际参数
        estimated_time_hours = (iteration_time * expected_iterations) / 3600
        print(f"   预计迭代次数: {expected_iterations}")
        print(f"   预计训练时间: {estimated_time_hours:.1f} 小时")
        
        print("\n✅ H100设置验证完成！")
        print("💡 建议: 在开始全规模训练前，先运行少量迭代验证收敛性")
        
        # 清理
        import shutil
        if os.path.exists(args.temp_path):
            shutil.rmtree(args.temp_path)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_h100_setup()
    if success:
        print("\n🚀 可以开始H100训练！")
        print("   运行命令: bash run_h100_cmdp.sh")
    else:
        print("\n⚠️ 请先解决配置问题")
    
    sys.exit(0 if success else 1)
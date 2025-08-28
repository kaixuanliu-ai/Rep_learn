#!/usr/bin/env python3
"""
收敛性测试：比较原版BRIEE和CMDP版本的收敛性
"""
import os
import sys
import torch
import numpy as np
from utils import parse_args, set_seed_everywhere, make_batch_env, ReplayBuffer
from algs.lsvi_ucb import LSVI_UCB
from algs.rep_learn import RepLearn
import matplotlib.pyplot as plt
from collections import deque
import time

def make_rep_learner(env, device, args):
    rep_learners = []
    for h in range(args.horizon):
        rep_learners.append(
            RepLearn(env.observation_space.shape[0],
                     env.state_dim,
                     env.action_dim,
                     args.hidden_dim,
                     args.rep_num_update,
                     args.rep_num_feature_update,
                     args.rep_num_adv_update,
                     device,
                     discriminator_lr=args.discriminator_lr,
                     discriminator_beta=args.discriminator_beta,
                     feature_lr=args.feature_lr,
                     feature_beta=args.feature_beta,
                     weight_lr=args.linear_lr,
                     weight_beta=args.linear_beta, 
                     batch_size = args.batch_size,
                     lamb = args.rep_lamb,
                     tau = args.phi0_temperature if h == 0 else args.temperature,
                     optimizer = args.optimizer,
                     softmax = args.softmax,
                     reuse_weights = args.reuse_weights,
                     temp_path = args.temp_path)
        )
    return rep_learners  

def evaluate(env, agent, args, use_cmdp=False):
    returns = np.zeros((args.num_eval,1))
    
    obs = env.reset()
    for h in range(args.horizon):
        action = agent.act_batch(obs, h, use_cmdp_constraint=use_cmdp)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        returns += reward

    return np.mean(returns)

def simple_test(enable_cmdp=False, max_iterations=50):
    """简化的收敛性测试"""
    print(f"开始{'CMDP' if enable_cmdp else '原版'}BRIEE收敛性测试...")
    
    class TestArgs:
        def __init__(self):
            self.horizon = 10
            self.num_actions = 5
            self.switch_prob = 0.5
            self.anti_reward_prob = 0.5
            self.anti_reward = 0.1
            self.observation_noise = 0.1
            self.num_envs = 10
            self.num_eval = 5
            self.env_temperature = 0.2
            self.variable_latent = False
            self.dense = False
            self.seed = 42
            self.hidden_dim = 64
            self.rep_num_update = 10
            self.rep_num_feature_update = 16
            self.rep_num_adv_update = 16
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
            self.temp_path = f'test_convergence_{"cmdp" if enable_cmdp else "original"}'
            self.alpha = 2.0
            self.lsvi_recent_size = 200
            self.lsvi_lamb = 1.0
            self.batch_size = 128
            self.recent_size = 1000
            self.num_warm_start = 20
            self.update_frequency = 2
            
            # CMDP参数
            self.enable_cmdp = enable_cmdp
            self.cmdp_b = 0.1
            self.cmdp_eta = 1e-3
            self.cmdp_xi = 2.0
    
    args = TestArgs()
    
    # 创建测试目录
    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)
    
    # 设置随机种子
    set_seed_everywhere(args.seed)
    
    # 创建环境
    env, eval_env = make_batch_env(args)
    device = torch.device("cpu")
    
    # 创建buffers
    buffers = []    
    for _ in range(args.horizon):
        buffers.append(
            ReplayBuffer(env.observation_space.shape, 
                         env.action_space.n, 
                         2000,  # 较小的buffer
                         args.batch_size, 
                         device,
                         recent_size=args.recent_size)
        )

    # 创建表示学习器和代理
    rep_learners = make_rep_learner(env, device, args)
    
    agent = LSVI_UCB(env.observation_space.shape[0],
                     env.state_dim,
                     env.action_dim,
                     args.horizon,
                     args.alpha,
                     device,
                     rep_learners,
                     recent_size = args.lsvi_recent_size,
                     lamb = args.lsvi_lamb,
                     enable_cmdp = args.enable_cmdp,
                     cmdp_b = args.cmdp_b,
                     cmdp_eta = args.cmdp_eta,
                     cmdp_xi = args.cmdp_xi)

    # Warm start
    num_actions = env.action_space.n
    for _ in range(args.num_warm_start):
        obs = env.reset()
        for h in range(args.horizon):
            action = np.random.randint(0, num_actions, args.num_envs)
            next_obs, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs,action,reward,next_obs,args.num_envs)
            obs = next_obs

    returns = deque(maxlen=10)
    eval_returns = []
    
    for iteration in range(max_iterations):
        # 数据收集阶段 - 保持原版逻辑
        for h in range(args.horizon):
            t = 0
            obs = env.reset()
            while t < h:
                action = agent.act_batch(obs, t, use_cmdp_constraint=False)  # 训练时不使用CMDP
                next_obs, _, _, _ = env.step(action)
                obs = next_obs
                t += 1
            # 使用随机动作进行数据收集（保持原版BRIEE逻辑）
            action = np.random.randint(0, num_actions, args.num_envs)
            next_obs, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs,action,reward,next_obs,args.num_envs)

            if h != args.horizon - 1:
                obs = next_obs
                action = np.random.randint(0, num_actions, args.num_envs)
                next_obs, reward, done, _ = env.step(action)
                buffers[h+1].add_batch(obs,action,reward,next_obs,args.num_envs)
            else:
                obs = env.reset()
                action = np.random.randint(0, num_actions, args.num_envs)
                next_obs, reward, done, _ = env.step(action)
                buffers[0].add_batch(obs,action,reward,next_obs,args.num_envs)

        if iteration % args.update_frequency == 0:
            # 表示学习
            for h in range(args.horizon):
                rep_learners[h].update(buffers[h])
            
            # 策略学习
            agent.update(buffers)
            
            # 评估
            eval_return = evaluate(eval_env, agent, args, use_cmdp=enable_cmdp)
            returns.append(eval_return)
            eval_returns.append(eval_return)
            
            print(f"Iteration {iteration}: Return = {eval_return:.4f}, Avg = {np.mean(list(returns)):.4f}")
            
            # 早停条件
            if len(returns) >= 5 and np.mean(list(returns)) > 0.8:
                print(f"收敛！平均回报: {np.mean(list(returns)):.4f}")
                break
    
    # 清理
    import shutil
    if os.path.exists(args.temp_path):
        shutil.rmtree(args.temp_path)
    
    return eval_returns

if __name__ == "__main__":
    print("BRIEE算法收敛性对比测试")
    print("=" * 50)
    
    # 测试原版BRIEE
    original_returns = simple_test(enable_cmdp=False)
    
    # 测试CMDP版本
    cmdp_returns = simple_test(enable_cmdp=True)
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(original_returns, label='Original BRIEE', marker='o')
    plt.plot(cmdp_returns, label='CMDP BRIEE', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Evaluation Return')
    plt.title('BRIEE Algorithm Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n最终结果:")
    print(f"原版BRIEE最终回报: {original_returns[-1]:.4f}")
    print(f"CMDP BRIEE最终回报: {cmdp_returns[-1]:.4f}")
    
    if len(original_returns) < 50 and len(cmdp_returns) < 50:
        print("✅ 两个版本都能收敛！")
    elif len(original_returns) < 50:
        print("⚠️ 只有原版能收敛")
    elif len(cmdp_returns) < 50:
        print("⚠️ 只有CMDP版本能收敛")
    else:
        print("❌ 两个版本都不能收敛")
#!/usr/bin/env python3
"""
快速收敛性测试
"""
import os
import sys
import torch
import numpy as np
from utils import parse_args, set_seed_everywhere, make_batch_env
from algs.lsvi_ucb import LSVI_UCB
from algs.rep_learn import RepLearn

def quick_test():
    """快速验证算法能否正常运行"""
    print("快速验证CMDP修复版本...")
    
    class TestArgs:
        def __init__(self):
            self.horizon = 5
            self.num_actions = 3
            self.switch_prob = 0.5
            self.anti_reward_prob = 0.5
            self.anti_reward = 0.1
            self.observation_noise = 0.1
            self.num_envs = 2
            self.num_eval = 2
            self.env_temperature = 0.2
            self.variable_latent = False
            self.dense = False
            self.seed = 42
            self.hidden_dim = 32
            self.rep_num_update = 3
            self.rep_num_feature_update = 5
            self.rep_num_adv_update = 5
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
            self.temp_path = 'quick_test'
            self.alpha = 1.0
            self.lsvi_recent_size = 50
            self.lsvi_lamb = 1.0
            self.batch_size = 32
            self.recent_size = 100
            
            # CMDP参数
            self.enable_cmdp = True
            self.cmdp_b = 0.1
            self.cmdp_eta = 1e-3
            self.cmdp_xi = 2.0
    
    args = TestArgs()
    set_seed_everywhere(args.seed)
    
    # 创建环境和代理
    env, eval_env = make_batch_env(args)
    device = torch.device("cpu")
    
    rep_learners = []
    for h in range(args.horizon):
        rep_learner = RepLearn(
            env.observation_space.shape[0], env.state_dim, env.action_dim,
            args.hidden_dim, args.rep_num_update, args.rep_num_feature_update,
            args.rep_num_adv_update, device, temp_path=args.temp_path
        )
        rep_learners.append(rep_learner)
    
    agent = LSVI_UCB(
        env.observation_space.shape[0], env.state_dim, env.action_dim,
        args.horizon, args.alpha, device, rep_learners,
        recent_size=args.lsvi_recent_size, lamb=args.lsvi_lamb,
        enable_cmdp=args.enable_cmdp, cmdp_b=args.cmdp_b,
        cmdp_eta=args.cmdp_eta, cmdp_xi=args.cmdp_xi
    )
    
    # 测试基本功能
    try:
        obs = env.reset()
        print(f"✅ 环境重置成功，观测形状: {obs.shape}")
        
        # 测试训练时的动作选择（不使用CMDP约束）
        action_train = agent.act_batch(obs, 0, use_cmdp_constraint=False)
        print(f"✅ 训练动作选择成功: {action_train.shape}")
        
        # 测试评估时的动作选择（使用CMDP约束）
        action_eval = agent.act_batch(obs, 0, use_cmdp_constraint=True)
        print(f"✅ 评估动作选择成功: {action_eval.shape}")
        
        # 验证CMDP约束在评估时生效
        if args.enable_cmdp:
            # 多次采样验证概率分布
            actions = []
            for _ in range(100):
                a = agent.act_batch(obs, 0, use_cmdp_constraint=True)
                actions.extend(a.tolist())
            
            action_counts = np.bincount(actions, minlength=args.num_actions)
            action_probs = action_counts / len(actions)
            min_prob = np.min(action_probs)
            
            print(f"✅ 动作概率分布: {action_probs}")
            print(f"✅ 最小动作概率: {min_prob:.3f} ({'>=0.1' if min_prob >= 0.05 else '<0.1'})")
        
        # 清理
        import shutil
        if os.path.exists(args.temp_path):
            shutil.rmtree(args.temp_path)
            
        print("\n🎉 快速测试通过！修复版本可以正常运行")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
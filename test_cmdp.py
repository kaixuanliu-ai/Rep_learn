#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from utils import parse_args, set_seed_everywhere, make_batch_env
from algs.lsvi_ucb import LSVI_UCB
from algs.rep_learn import RepLearn

def test_cmdp_functionality():
    """测试CMDP功能的基本功能"""
    print("开始CMDP功能测试...")
    
    # 创建最小化的测试参数
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
            self.rep_num_update = 5
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
            self.temp_path = 'test_temp'
            self.alpha = 2.0
            self.lsvi_recent_size = 100
            self.lsvi_lamb = 1.0
            
            # CMDP参数
            self.enable_cmdp = True
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
    try:
        env, eval_env = make_batch_env(args)
        print("✓ 环境创建成功")
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        return False
    
    device = torch.device("cpu")
    
    # 创建表示学习器（简化版）
    rep_learners = []
    for h in range(args.horizon):
        try:
            rep_learner = RepLearn(
                env.observation_space.shape[0],
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
                batch_size=32,
                lamb=args.rep_lamb,
                tau=args.phi0_temperature if h == 0 else args.temperature,
                optimizer=args.optimizer,
                softmax=args.softmax,
                reuse_weights=args.reuse_weights,
                temp_path=args.temp_path
            )
            rep_learners.append(rep_learner)
        except Exception as e:
            print(f"✗ 表示学习器创建失败 (h={h}): {e}")
            return False
    
    print("✓ 表示学习器创建成功")
    
    # 创建CMDP代理
    try:
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
        print("✓ CMDP代理创建成功")
    except Exception as e:
        print(f"✗ CMDP代理创建失败: {e}")
        return False
    
    # 测试混合策略
    try:
        obs = env.reset()
        action_probs = agent.get_mixed_policy_probs(torch.FloatTensor(obs).to(device), 0)
        print(f"✓ 混合策略概率计算成功: 形状={action_probs.shape}")
        
        # 验证概率和为1
        prob_sums = torch.sum(action_probs, dim=1)
        if torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6):
            print("✓ 混合策略概率和验证通过")
        else:
            print(f"✗ 混合策略概率和验证失败: {prob_sums}")
            return False
            
        # 验证最小探索概率
        min_probs = torch.min(action_probs, dim=1)[0]
        if torch.all(min_probs >= args.cmdp_b - 1e-6):
            print(f"✓ 最小探索概率约束验证通过: min_prob={torch.min(min_probs).item():.4f} >= {args.cmdp_b}")
        else:
            print(f"✗ 最小探索概率约束验证失败: min_prob={torch.min(min_probs).item():.4f} < {args.cmdp_b}")
            return False
            
    except Exception as e:
        print(f"✗ 混合策略测试失败: {e}")
        return False
    
    # 测试动作选择
    try:
        actions = agent.act_batch(obs, 0)
        print(f"✓ 批量动作选择成功: 形状={actions.shape}")
        
        single_action = agent.act(obs[0], 0)
        print(f"✓ 单个动作选择成功: action={single_action}")
    except Exception as e:
        print(f"✗ 动作选择测试失败: {e}")
        return False
    
    # 测试拉格朗日乘子更新
    try:
        initial_Y_k = agent.Y_k
        new_Y_k = agent.update_lagrange_multiplier(obs[0])
        print(f"✓ 拉格朗日乘子更新成功: {initial_Y_k} -> {new_Y_k}")
    except Exception as e:
        print(f"✗ 拉格朗日乘子更新失败: {e}")
        return False
    
    # 清理测试目录
    try:
        import shutil
        if os.path.exists(args.temp_path):
            shutil.rmtree(args.temp_path)
        print("✓ 清理测试目录成功")
    except Exception as e:
        print(f"⚠ 清理测试目录失败: {e}")
    
    print("\n🎉 CMDP功能测试全部通过！")
    return True

def test_original_briee_compatibility():
    """测试原始BRIEE算法兼容性"""
    print("\n开始原始BRIEE兼容性测试...")
    
    class TestArgs:
        def __init__(self):
            self.horizon = 3
            self.num_actions = 2
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
            self.temp_path = 'test_temp2'
            self.alpha = 1.0
            self.lsvi_recent_size = 50
            self.lsvi_lamb = 1.0
            
            # 禁用CMDP
            self.enable_cmdp = False
            self.cmdp_b = 0.1
            self.cmdp_eta = 1e-3
            self.cmdp_xi = 2.0
    
    args = TestArgs()
    
    try:
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
            enable_cmdp=args.enable_cmdp
        )
        
        # 测试原始行为
        obs = env.reset()
        actions = agent.act_batch(obs, 0)
        print("✓ 原始BRIEE模式正常工作")
        
        # 清理
        import shutil
        if os.path.exists(args.temp_path):
            shutil.rmtree(args.temp_path)
        
        return True
        
    except Exception as e:
        print(f"✗ 原始BRIEE兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    success1 = test_cmdp_functionality()
    success2 = test_original_briee_compatibility()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！CMDP扩展实施成功！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，请检查代码。")
        sys.exit(1)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class LSVI_UCB(object): 

    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        horizon,
        alpha,
        device,
        rep_learners,
        lamb = 1,
        recent_size=0,
        enable_cmdp=False,
        cmdp_b=0.1,
        cmdp_eta=1e-3,
        cmdp_xi=None,
    ):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.feature_dim = state_dim * action_dim

        self.device = device

        self.rep_learners = rep_learners

        self.lamb = lamb
        self.alpha = alpha

        self.recent_size = recent_size

        self.W = torch.rand((self.horizon, self.feature_dim)).to(self.device)
        self.Sigma_invs = torch.zeros((self.horizon, self.feature_dim, self.feature_dim)).to(self.device)

        self.Q_max = torch.tensor(float(self.horizon)).to(self.device)
        
        # CMDP parameters
        self.enable_cmdp = enable_cmdp
        self.cmdp_b = cmdp_b
        self.cmdp_eta = cmdp_eta
        self.cmdp_xi = cmdp_xi if cmdp_xi is not None else 2 * horizon / np.sqrt(1000)  # default upper bound
        
        # CMDP state variables
        self.Y_k = 0.0  # Lagrange multiplier
        self.W_r = torch.rand((self.horizon, self.feature_dim)).to(self.device)  # reward Q-function weights
        self.W_g = torch.zeros((self.horizon, self.feature_dim)).to(self.device)  # cost Q-function weights
        self.episode_count = 0

    def Q_values(self, obs, h):
        Qs = torch.zeros((len(obs),self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            actions = torch.zeros((len(obs),self.action_dim)).to(self.device)
            actions[:,a] = 1
            with torch.no_grad():
                feature = self.rep_learners[h].phi(obs,actions)
            Q_est = torch.matmul(feature, self.W[h]).squeeze() 
            ucb = torch.sqrt(torch.sum(torch.matmul(feature, self.Sigma_invs[h]) * feature, dim=1))
            
            Qs[:,a] = torch.minimum(Q_est + self.alpha * ucb, self.Q_max)

        return Qs

    def get_mixed_policy_probs(self, obs, h):
        """
        计算混合策略的概率分布
        pi_mix(a|s) = (1 - b) * pi_current(a|s) + b * U(A)
        """
        if not self.enable_cmdp:
            # 如果不启用CMDP，返回原始策略（贪婪策略）
            Qs = self.Q_values(obs, h)
            action_probs = torch.zeros_like(Qs)
            best_actions = torch.argmax(Qs, dim=1)
            action_probs[torch.arange(len(obs)), best_actions] = 1.0
            return action_probs
        
        # 获取原始策略
        Qs = self.Q_values(obs, h)
        
        # 使用更大的温度参数来增加随机性，便于混合
        pi_current = F.softmax(Qs / 1.0, dim=1)  
        
        # 均匀策略
        uniform_policy = torch.ones_like(pi_current) / self.action_dim
        
        # 混合策略：(1-b) * pi_current + b * uniform
        mixed_policy = (1 - self.cmdp_b) * pi_current + self.cmdp_b * uniform_policy
        
        # 确保最低概率约束：如果任何动作概率低于b，则增加其概率
        min_probs = torch.min(mixed_policy, dim=1, keepdim=True)[0]
        if torch.any(min_probs < self.cmdp_b - 1e-6):
            # 重新计算以确保约束满足
            # 方法：给每个动作至少分配 b 的概率，剩余概率按原比例分配
            remaining_prob = 1.0 - self.action_dim * self.cmdp_b
            if remaining_prob < 0:
                remaining_prob = 0
            # 归一化原策略并分配剩余概率
            normalized_original = pi_current * remaining_prob
            mixed_policy = self.cmdp_b + normalized_original
        
        return mixed_policy

    def compute_cost(self, obs, actions, h):
        """
        计算约束违反成本
        这里简化为：如果动作概率低于阈值则有成本
        """
        batch_size = obs.shape[0]
        
        # 获取当前状态下各动作的概率（基于当前Q函数）
        if hasattr(self, 'W_r') and self.W_r is not None:
            # 如果已有奖励权重，使用它们
            current_probs = self.get_mixed_policy_probs(obs, h)
        else:
            # 否则使用均匀分布作为初始估计
            current_probs = torch.ones(batch_size, self.action_dim).to(self.device) / self.action_dim
        
        # 提取对应动作的概率
        action_indices = torch.argmax(actions, dim=1)
        selected_probs = current_probs[torch.arange(batch_size), action_indices]
        
        # 成本：如果概率低于b则成本为(b - prob)，否则为0
        cost = torch.clamp(self.cmdp_b - selected_probs, min=0.0).unsqueeze(-1)
        
        return cost

    def Q_values_reward(self, obs, h):
        """使用奖励权重计算Q值"""
        Qs = torch.zeros((len(obs), self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            actions = torch.zeros((len(obs), self.action_dim)).to(self.device)
            actions[:, a] = 1
            with torch.no_grad():
                feature = self.rep_learners[h].phi(obs, actions)
            Q_est = torch.matmul(feature, self.W_r[h]).squeeze()
            ucb = torch.sqrt(torch.sum(torch.matmul(feature, self.Sigma_invs[h]) * feature, dim=1))
            
            Qs[:, a] = torch.minimum(Q_est + self.alpha * ucb, self.Q_max)

        return Qs

    def Q_values_cost(self, obs, h):
        """使用成本权重计算Q值"""
        Qs = torch.zeros((len(obs), self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            actions = torch.zeros((len(obs), self.action_dim)).to(self.device)
            actions[:, a] = 1
            with torch.no_grad():
                feature = self.rep_learners[h].phi(obs, actions)
            Q_est = torch.matmul(feature, self.W_g[h]).squeeze()
            # 对成本Q函数，不添加UCB项，因为我们想要保守估计
            Qs[:, a] = Q_est

        return Qs

    def get_initial_cost_value(self, initial_obs):
        """
        计算初始状态的成本价值函数 V_g_1(x_1)
        用于拉格朗日乘子更新
        """
        if not self.enable_cmdp:
            return 0.0
            
        with torch.no_grad():
            obs = torch.FloatTensor(initial_obs).to(self.device).unsqueeze(0)
            Q_g_values = self.Q_values_cost(obs, 0)  # h=0 为初始时间步
            V_g = torch.max(Q_g_values, dim=1)[0]
            
        return V_g.item()

    def update_lagrange_multiplier(self, initial_obs):
        """
        更新拉格朗日乘子
        Y_{k+1} = max(min(Y_k + eta * (b - V_g_1(x_1)), xi), 0)
        """
        if not self.enable_cmdp:
            return
            
        self.episode_count += 1
        
        # 计算 V_g_1(x_1)
        V_g_1 = self.get_initial_cost_value(initial_obs)
        
        # 更新拉格朗日乘子
        constraint_violation = self.cmdp_b - V_g_1
        self.Y_k = max(min(self.Y_k + self.cmdp_eta * constraint_violation, self.cmdp_xi), 0.0)
        
        return self.Y_k

    def act_batch(self, obs, h, use_cmdp_constraint=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            
            if self.enable_cmdp and use_cmdp_constraint:
                # 仅在需要时使用CMDP约束（评估期间）
                action_probs = self.get_mixed_policy_probs(obs, h)
                # 使用多项式采样
                action = torch.multinomial(action_probs, 1).squeeze(-1)
            else:
                # 原始贪婪策略（训练期间的策略学习）
                Qs = self.Q_values(obs, h)
                action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()


    def act(self, obs, h, use_cmdp_constraint=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            
            if self.enable_cmdp and use_cmdp_constraint:
                # 仅在需要时使用CMDP约束
                action_probs = self.get_mixed_policy_probs(obs, h)
                action = torch.multinomial(action_probs, 1).squeeze(-1)
            else:
                # 原始贪婪策略
                Qs = self.Q_values(obs, h)
                action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()

    def update(self, buffers):
        assert len(buffers) == self.horizon

        for h in range(self.horizon)[::-1]:
            if self.recent_size > 0:
                obses, actions, rewards, next_obses = buffers[h].get_full(device=self.device, recent_size=self.recent_size)
            else:
                obses, actions, rewards, next_obses = buffers[h].get_full(device=self.device)
            
            with torch.no_grad():
                feature = self.rep_learners[h].phi(obses,actions)
            Sigma = torch.matmul(feature.T, feature) + self.lamb * torch.eye(self.feature_dim).to(self.device)
            self.Sigma_invs[h] = torch.inverse(Sigma)

            # 保持原始BRIEE的更新逻辑不变
            if h == self.horizon - 1:
                target_Q = rewards
            else:
                Q_prime = torch.max(self.Q_values(next_obses, h+1),dim=1)[0].unsqueeze(-1)
                target_Q = rewards + Q_prime

            feature_target = torch.sum(feature * target_Q, dim=0)
            self.W[h] = torch.matmul(self.Sigma_invs[h], feature_target)            

    def save_weight(self, path):
        for h in range(self.horizon):
            torch.save(self.W[h],"{}/W_{}.pth".format(path,str(h)))
            torch.save(self.Sigma_invs[h], "{}/Sigma_{}.pth".format(path,str(h)))

    def load_weight(self, path):
        for h in range(self.horizon):
            self.W[h] = torch.load("{}/W_{}.pth".format(path,str(h))).to(self.device)
            self.Sigma_invs[h] = torch.load("{}/Sigma_{}.pth".format(path,str(h))).to(self.device)










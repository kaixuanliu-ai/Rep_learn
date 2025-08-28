# 🚀 WandB监控H100训练完整指南

## 🎯 概述

本指南将帮助您使用Weights & Biases (WandB) 监控horizon=100级别的BRIEE算法训练，包括：
- ✅ CMDP版本与原版BRIEE的对比
- ✅ 实时训练指标监控
- ✅ 收敛性分析
- ✅ 自动化训练管理

## 📋 快速开始

### 方法1：一键启动（推荐）
```bash
# 替换your_wandb_username为您的WandB用户名
bash start_wandb_training.sh your_wandb_username
```

### 方法2：手动配置
```bash
# 1. 设置WandB
python setup_wandb.py

# 2. 启动训练
bash run_h100_cmdp.sh h100_cmdp_results 10 your_wandb_username
bash run_h100_original.sh h100_original_results 10 your_wandb_username

# 3. 启动监控
python wandb_monitor.py --entity your_wandb_username
```

## 🔧 详细配置步骤

### 第一步：WandB账号设置

1. **注册WandB账号**
   ```bash
   # 访问 https://wandb.ai/ 注册账号
   # 获取API Key: https://wandb.ai/authorize
   ```

2. **登录WandB**
   ```bash
   source venv/bin/activate
   wandb login
   # 或者
   python setup_wandb.py  # 交互式设置
   ```

3. **验证登录**
   ```bash
   python -c "import wandb; print(wandb.Api().viewer.username)"
   ```

### 第二步：项目配置

训练将创建两个WandB项目：
- `briee_cmdp_h100`: CMDP版本训练
- `briee_original_h100`: 原版BRIEE训练

### 第三步：启动训练

```bash
# 方法1：一键启动（包含后台运行）
bash start_wandb_training.sh your_wandb_username

# 方法2：分别启动
bash run_h100_cmdp.sh h100_cmdp_results 10 your_wandb_username
bash run_h100_original.sh h100_original_results 10 your_wandb_username
```

## 📊 监控功能

### 1. Web界面监控

访问您的WandB项目：
- CMDP版本: `https://wandb.ai/your_username/briee_cmdp_h100`
- 原版对比: `https://wandb.ai/your_username/briee_original_h100`

**关键指标**：
- `eval`: 评估回报（主要收敛指标，目标：0.8-1.0）
- `reached`: 最远到达的时间步（目标：100）
- `rep_learn_time`: 表示学习时间
- `lsvi_time`: 策略学习时间
- `state 0/1`: 状态访问统计

### 2. 实时命令行监控

```bash
# 基础监控
python wandb_monitor.py --entity your_wandb_username

# 自定义监控
python wandb_monitor.py \
    --entity your_wandb_username \
    --projects briee_cmdp_h100 briee_original_h100 \
    --interval 60

# 单次检查
python wandb_monitor.py --entity your_wandb_username --once
```

### 3. 本地日志监控

```bash
# 查看实时日志
tail -f cmdp_training.log
tail -f original_training.log

# 检查进程状态
ps aux | grep "python main.py"

# 查看资源使用
htop
```

## 📈 监控要点

### 收敛指标
- ✅ **评估回报 (eval)**: 应从0逐步上升至0.8-1.0
- ✅ **到达步数 (reached)**: 应从0逐步增长至100
- ✅ **状态访问**: state 0和state 1应相对均衡
- ✅ **训练稳定性**: 时间指标应保持相对稳定

### 异常检测
- ❌ **eval长时间不提升**: 可能需要调整学习率
- ❌ **reached停止增长**: 可能陷入局部最优
- ❌ **训练时间异常增长**: 可能存在内存泄漏
- ❌ **状态访问极度不平衡**: 可能探索不充分

### CMDP特有指标
- `cmdp_enabled`: 确认CMDP约束已启用
- `cmdp_b`: 最低探索概率参数（应为0.1）

## 🎛️ WandB面板配置

### 创建对比面板
1. 在WandB中点击"Create Panel"
2. 选择"Line Plot"
3. 添加两个项目的数据
4. 设置X轴为时间或episode
5. 设置Y轴为关键指标（eval, reached等）

### 设置报警
1. 进入项目设置
2. 添加Alert规则：
   - `eval < 0.5` 且运行时间 > 2小时
   - `reached`停止增长超过30分钟
3. 配置通知方式（邮件/Slack）

## 🛠️ 管理命令

### 训练控制
```bash
# 查看训练进程
cat training_info.txt

# 停止训练
kill $(cat training_info.txt | grep "PID:" | awk '{print $3}')
# 或强制停止所有
pkill -f "python main.py"

# 重启训练
bash start_wandb_training.sh your_wandb_username
```

### 结果分析
```bash
# 生成对比图表
python wandb_monitor.py --entity your_wandb_username --once

# 查看保存的检查点
ls -la h100_*_results/

# 分析训练日志
grep "eval" cmdp_training.log | tail -10
```

## 📊 预期结果

### 训练参数
- **Horizon**: 100
- **Episodes**: 10,000,000  
- **预计时间**: 2-4小时
- **内存使用**: 2-4GB
- **存储需求**: 500MB-1GB

### 收敛期望
- **CMDP版本**: 应该与原版有相似的收敛速度
- **最终性能**: eval应达到0.8-1.0
- **约束满足**: 评估时每个动作概率≥0.1
- **训练稳定**: 无明显的崩溃或异常

## 🔧 故障排除

### 常见问题

1. **WandB登录失败**
   ```bash
   wandb login --relogin
   # 或设置API key环境变量
   export WANDB_API_KEY=your_api_key
   ```

2. **训练进程意外停止**
   ```bash
   # 检查日志
   tail -50 cmdp_training.log
   # 检查内存使用
   free -h
   ```

3. **WandB同步问题**
   ```bash
   # 检查网络连接
   wandb status
   # 手动同步离线数据
   wandb sync wandb/offline-run-*
   ```

4. **性能监控异常**
   ```bash
   # 检查磁盘空间
   df -h
   # 检查CPU/内存使用
   htop
   ```

## 💡 最佳实践

1. **预先测试**: 运行`test_h100_setup.py`验证配置
2. **并行对比**: 同时运行CMDP和原版进行对比
3. **定期检查**: 每1-2小时检查一次训练状态
4. **保存中间结果**: 训练会自动保存，但建议定期备份
5. **资源监控**: 注意CPU、内存和磁盘使用情况

## 📞 支持

如果遇到问题：
1. 检查本指南的故障排除部分
2. 查看WandB官方文档: https://docs.wandb.ai/
3. 检查训练日志文件寻找错误信息

---

🎉 **现在您可以开始带WandB监控的H100级别训练了！**

```bash
bash start_wandb_training.sh your_wandb_username
```
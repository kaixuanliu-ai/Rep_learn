# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of BRIEE (Block-MDP Efficient Representation Learning), an ICML 2022 paper on model-free representation learning in reinforcement learning. The codebase focuses on representation learning and representational transfer in block MDPs.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
pip install virtualenv
virtualenv /path/to/venv --python=python3
. /path/to/venv/bin/activate

# Install dependencies
pip install -r requirement.txt
```

### Running Experiments
```bash
# Standard run
bash run.sh [horizon] [num_threads] [save_path]

# Simplex feature variant
bash run_simplex.sh [num_threads] [save_path]

# Dense reward variant  
bash run_dense.sh [num_threads] [save_path]

# Direct execution
python main.py --horizon 100 --num_threads 10 --temp_path results
```

## Architecture

### Core Components
- `main.py`: Entry point with experiment orchestration, multiprocessing for representation learning, and evaluation loops
- `utils.py`: Command-line argument parsing with extensive hyperparameter configuration
- `algs/`: Algorithm implementations
  - `base_learner.py`: Base classes for Feature networks, Discriminators, and core learning components
  - `rep_learn.py`: RepLearn algorithm for representation learning with adversarial training
  - `lsvi_ucb.py`: LSVI-UCB algorithm for policy learning using learned representations
- `envs/Lock_batch.py`: Batch environment implementation for the Lock domain (Block MDP)

### Key Design Patterns
- **Hierarchical Learning**: Separate representation learners for each horizon step
- **Multiprocessing**: Parallel representation learning across horizon steps using `multiprocessing.Process`
- **Batch Processing**: Vectorized environment interactions for efficiency (`num_envs` parameter)
- **Replay Buffers**: Separate buffers for each horizon step with recent experience prioritization

### Configuration System
All hyperparameters are centralized in `utils.py` with categories:
- Environment settings (horizon, switch_prob, observation_noise)
- Representation learning (hidden_dim, learning rates, update frequencies)  
- LSVI-UCB parameters (alpha, lambda regularization)
- Training logistics (batch_size, num_episodes, num_threads)

### Experiment Tracking
Uses Weights & Biases (wandb) for logging. Set `WANDB_MODE='offline'` in main.py to disable online logging.

## Important Notes
- PyTorch is used but GPU is not required (CPU device hardcoded)
- Random seeds are set across numpy, torch, and environment for reproducibility
- Temporary files are saved to configurable `temp_path` directory
- The codebase assumes the Lock batch environment structure for Block MDP experiments
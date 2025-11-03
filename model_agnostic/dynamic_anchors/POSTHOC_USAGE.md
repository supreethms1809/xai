# Post-hoc Dynamic Anchors - Production Usage Guide

## Overview

This script trains dynamic anchors in a **post-hoc** fashion:
1. **First**: Train classifier fully to completion with early stopping
2. **Then**: Train RL policy on frozen classifier using PPO
3. **Finally**: Freeze policy and evaluate with greedy rollouts

## Quick Start

### Basic Usage (uses dataset-specific defaults)

```bash
# Breast Cancer dataset (small, fast)
python post_hoc_dynamicAnchor.py --dataset breast_cancer

# Synthetic dataset (medium complexity)
python post_hoc_dynamicAnchor.py --dataset synthetic

# Covertype dataset (large, complex)
python post_hoc_dynamicAnchor.py --dataset covtype
```

### Production-Rready Commands

#### Breast Cancer (Recommended for quick testing)
```bash
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --classifier_epochs 100 \
    --num_greedy_rollouts 20 \
    --seed 42
```
**Defaults**: 25 RL episodes, 40 steps, no perturbations, bootstrap mode

#### Synthetic Dataset
```bash
python post_hoc_dynamicAnchor.py \
    --dataset synthetic \
    --classifier_epochs 100 \
    --num_greedy_rollouts 20 \
    --seed 42
```
**Defaults**: 30 RL episodes, 50 steps, uniform perturbations, 2048 samples

#### Covertype Dataset (Most Complex)
```bash
python post_hoc_dynamicAnchor.py \
    --dataset covtype \
    --classifier_epochs 100 \
    --num_greedy_rollouts 20 \
    --seed 42
```
**Defaults**: 60 RL episodes, 90 steps, uniform perturbations, 8192 samples

## Dataset-Specific Defaults

| Parameter | Breast Cancer | Synthetic | Covertype |
|-----------|--------------|-----------|-----------|
| **RL Episodes** | 25 | 30 | 60 |
| **RL Steps/Episode** | 40 | 50 | 90 |
| **Perturbations** | ❌ False | ✅ True | ✅ True |
| **Mode** | bootstrap | uniform | uniform |
| **N Perturb** | 1024 | 2048 | 8192 |
| **Entropy Coef** | 0.02 | 0.02 | 0.015 |
| **Precision Target** | 0.95 | 0.95 | 0.95 |
| **Coverage Target** | 0.05 | 0.04 | 0.02 |

## Advanced Options

### Override RL Parameters
```bash
# Use custom RL episodes (overrides dataset default)
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --rl_episodes 50 \
    --rl_steps 60

# Custom entropy coefficient
python post_hoc_dynamicAnchor.py \
    --dataset synthetic \
    --rl_entropy_coef 0.05
```

### Toggle Perturbations
```bash
# Enable perturbations (overrides dataset default)
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --use_perturbation

# Disable perturbations
python post_hoc_dynamicAnchor.py \
    --dataset synthetic \
    --no-perturbation
```

### Custom Perturbation Settings
```bash
# Bootstrap mode with custom samples
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --use_perturbation \
    --perturbation_mode bootstrap \
    --n_perturb 2048

# Uniform mode
python post_hoc_dynamicAnchor.py \
    --dataset synthetic \
    --use_perturbation \
    --perturbation_mode uniform \
    --n_perturb 4096
```

### Classifier Training
```bash
# Custom early stopping patience
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --classifier_epochs 200 \
    --classifier_patience 20 \
    --classifier_lr 0.0005

# Large batch size for faster training
python post_hoc_dynamicAnchor.py \
    --dataset covtype \
    --classifier_batch_size 512
```

### RL Training Parameters
```bash
# PPO settings
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --rl_ppo_epochs 8 \
    --rl_clip_epsilon 0.1 \
    --rl_batch_size 256

# RL learning rate
python post_hoc_dynamicAnchor.py \
    --dataset synthetic \
    --rl_lr 1e-4
```

### Evaluation Settings
```bash
# More greedy rollouts for better statistics
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --num_greedy_rollouts 100

# Show all features in rules
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --max_features_in_rule 0

# Show 3 most important features
python post_hoc_dynamicAnchor.py \
    --dataset synthetic \
    --max_features_in_rule 3
```

### Visualization
```bash
# Enable plots (default: disabled)
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --show_plots
```

### Device Selection
```bash
# Force CPU
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --device cpu

# Force CUDA
python post_hoc_dynamicAnchor.py \
    --dataset covtype \
    --device cuda

# Auto-detect (default)
python post_hoc_dynamicAnchor.py \
    --dataset synthetic \
    --device auto
```

## Output Files

Each run saves 4 files with consistent naming:

```
classifier_posthoc_{dataset}_{seed}_{classifier_epochs}_{rl_episodes}_{rl_steps}.pth
policy_posthoc_{dataset}_{seed}_{classifier_epochs}_{rl_episodes}_{rl_steps}.pth
value_fn_posthoc_{dataset}_{seed}_{classifier_epochs}_{rl_episodes}_{rl_steps}.pth
results_posthoc_{dataset}_{seed}_{classifier_epochs}_{rl_episodes}_{rl_steps}.json
```

Example:
```
classifier_posthoc_breast_cancer_42_100_25_40.pth
policy_posthoc_breast_cancer_42_100_25_40.pth
value_fn_posthoc_breast_cancer_42_100_25_40.pth
results_posthoc_breast_cancer_42_100_25_40.json
```

## Comparison with Joint Training

| Aspect | Post-hoc (`post_hoc_dynamicAnchor.py`) | Joint (`dyn_anchor_PPO.py`) |
|--------|--------------------------------------|----------------------------|
| **Classifier Training** | Full training first with early stopping | Interleaved with RL episodes |
| **RL Training** | After classifier is frozen | Concurrent with classifier updates |
| **Use Case** | When classifier is already trained | End-to-end joint optimization |
| **Evaluation** | Freeze both models | Greedy rollouts |

## All Available Options

```bash
python post_hoc_dynamicAnchor.py --help
```

Key parameters:
- `--dataset`: Dataset choice (breast_cancer, synthetic, covtype)
- `--seed`: Random seed for reproducibility
- `--device`: Device (auto, cuda, mps, cpu)
- `--classifier_epochs`: Max epochs for classifier training
- `--classifier_patience`: Early stopping patience
- `--rl_episodes`: RL episodes (None = dataset default)
- `--rl_steps`: RL steps per episode (None = dataset default)
- `--rl_ppo_epochs`: PPO update epochs
- `--rl_entropy_coef`: Entropy coefficient (None = dataset default)
- `--use_perturbation` / `--no-perturbation`: Toggle perturbations
- `--perturbation_mode`: bootstrap or uniform
- `--n_perturb`: Number of perturbation samples
- `--num_greedy_rollouts`: Number of evaluation rollouts
- `--max_features_in_rule`: Max features to show (0 = all)
- `--show_plots`: Enable visualization

## Tips for Production

1. **Start with breast_cancer** for quick validation
2. **Use dataset defaults** unless you have specific requirements
3. **Increase num_greedy_rollouts** (50-100) for reliable statistics
4. **Set seed** for reproducibility across experiments
5. **Disable plots** in headless environments
6. **Use CUDA** for faster training on large datasets
7. **Monitor classifier early stopping** - adjust patience if needed

## Example Production Pipeline

```bash
#!/bin/bash
# Run multiple seeds for robustness
for seed in 42 123 456 789 999; do
    python post_hoc_dynamicAnchor.py \
        --dataset breast_cancer \
        --seed $seed \
        --num_greedy_rollouts 50 \
        --classifier_epochs 100
done
```

# Post-hoc Dynamic Anchors - Quick Reference

## 🚀 Quick Production Commands

### Basic Usage (Recommended)
```bash
# Breast Cancer (fast, ~2-5 min)
python post_hoc_dynamicAnchor.py --dataset breast_cancer

# Synthetic (medium, ~5-10 min)
python post_hoc_dynamicAnchor.py --dataset synthetic

# Covertype (slow, ~30-60 min)
python post_hoc_dynamicAnchor.py --dataset covtype
```

### With Standard Options
```bash
python post_hoc_dynamicAnchor.py \
    --dataset breast_cancer \
    --seed 42 \
    --num_greedy_rollouts 50
```

## 📊 Dataset Defaults Summary

| Dataset | RL Episodes | RL Steps | Perturbations | Runtime |
|---------|-------------|----------|---------------|---------|
| **breast_cancer** | 25 | 40 | ❌ No | ~3 min |
| **synthetic** | 30 | 50 | ✅ Uniform | ~8 min |
| **covtype** | 60 | 90 | ✅ Uniform | ~45 min |

## 🎯 Most Common Overrides

```bash
# More episodes for better convergence
--rl_episodes 100

# More evaluation rollouts
--num_greedy_rollouts 100

# Show more features in rules
--max_features_in_rule 10

# Custom seed
--seed 42
```

## 📁 Output Files

```
classifier_posthoc_{dataset}_{seed}_{clf_epochs}_{rl_eps}_{rl_steps}.pth
policy_posthoc_{dataset}_{seed}_{clf_epochs}_{rl_eps}_{rl_steps}.pth
value_fn_posthoc_{dataset}_{seed}_{clf_epochs}_{rl_eps}_{rl_steps}.pth
results_posthoc_{dataset}_{seed}_{clf_epochs}_{rl_eps}_{rl_steps}.json
```

## 🔍 Get All Options

```bash
python post_hoc_dynamicAnchor.py --help
```

## 📖 Full Documentation

See `POSTHOC_USAGE.md` for complete documentation.

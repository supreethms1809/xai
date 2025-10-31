# Analysis: Why Dynamic Anchors Show Better Metrics

## Key Differences Between Static and Dynamic Anchors

### 1. **Fundamental Approach Difference**
- **Static Anchors**: Per-instance explanations
  - Finds a different anchor for EACH test instance (20 anchors per class)
  - Each anchor is optimized specifically for that instance
  - Coverage is calculated per instance (local coverage)
  
- **Dynamic Anchors**: Per-class explanations
  - Finds ONE anchor per class (1 anchor per class)
  - Anchor must work for ALL instances of that class
  - Coverage is calculated globally (fraction of all training instances in box)

### 2. **Coverage Calculation**
- **Static**: 
  - Coverage = per-instance coverage (how many samples from the perturbation distribution satisfy the anchor)
  - Each instance gets its own optimized anchor → tighter, instance-specific → lower average coverage
  - Uses test instances as seeds for explanation
  
- **Dynamic**: 
  - Coverage = fraction of training instances that fall within the box
  - One box must cover many instances → naturally broader → higher coverage
  - Uses training data (`X_train`) for coverage calculation (line 1257 in dyn_anchor.py)

### 3. **Optimization Objective**
- **Static**: 
  - Bandit algorithm optimizes per instance
  - Goal: Find minimal anchor that satisfies precision threshold for THIS specific instance
  - May sacrifice coverage for precision on that instance
  
- **Dynamic**: 
  - RL optimizes globally across all training instances
  - Goal: Maximize reward = α·precision_gain + β·coverage_gain
  - Explicitly optimizes for coverage alongside precision (α=0.7, β=0.6)

### 4. **Data Used for Metrics**
- **Static**: Uses test instances (20 instances from test set)
- **Dynamic**: Uses training data for coverage calculation (`X_train` in AnchorEnv)

### 5. **Search Space**
- **Static**: 
  - Discrete feature conditions (from discretization bins)
  - Per-instance optimization allows very specific rules
  
- **Dynamic**: 
  - Continuous or discrete bounds (can be more flexible)
  - Global optimization may find broader, more generalizable rules

## Why Dynamic Shows Better Coverage

1. **Per-Class vs Per-Instance**: Dynamic finds one anchor per class that must work for all instances, leading to broader anchors with higher coverage.

2. **Explicit Coverage Optimization**: Dynamic's reward function explicitly optimizes for coverage (β=0.6 weight on coverage_gain).

3. **Training Data**: Dynamic computes coverage on training data, which may have different distribution than test instances.

4. **Global Optimization**: RL learns a policy that optimizes globally, while static optimizes locally per instance.

## Why Dynamic Shows Similar/Higher Precision

1. **RL Training**: The policy is trained over many episodes to maintain precision while expanding coverage.

2. **Co-training**: The classifier and policy are co-trained, allowing dynamic to adapt anchors as classifier improves.

3. **Greedy Rollout**: After training, greedy evaluation selects actions that maintain precision.

4. **Fallback Mechanism**: If greedy produces low-quality anchors, it falls back to best training episode.

## Potential Issues in Comparison

1. **Different Data**: Dynamic uses training data, static uses test instances → not apples-to-apples
2. **Per-Instance vs Per-Class**: Fundamentally different explanation types
3. **Averaging**: Static averages 20 per-instance anchors; Dynamic may use single greedy rollout (check `num_greedy_rollouts`)

## Recommendations

1. **Use Same Data**: Make dynamic compute coverage on test data to match static
2. **Ensure Averaging**: Verify dynamic is using 20 rollouts and averaging them
3. **Understand Goal**: If you want per-instance explanations, static is better. If you want per-class rules, dynamic is more appropriate.
4. **Separate Metrics**: Report both per-instance (static-style) and per-class (dynamic-style) metrics separately


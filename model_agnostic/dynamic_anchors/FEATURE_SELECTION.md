# Why Static Shows 2 Features vs Dynamic Shows More

## Static Anchors Feature Selection

**Algorithm**: Multi-armed bandit (beam search variant)
- **Goal**: Find the **minimal** set of features that satisfy precision threshold
- **Process**: 
  1. Starts with an empty anchor
  2. Tries adding each feature condition one at a time
  3. Evaluates precision after each addition
  4. Stops as soon as precision ≥ threshold (e.g., 0.95)
  5. Selects the **shortest** anchor that meets precision
  
**Result**: Typically 2-3 features because it stops as soon as precision is met

**Example**: 
- Add "worst perimeter > 0.55" → precision = 0.92 (not enough)
- Add "worst area > 0.30" → precision = 0.96 (meets threshold!)
- **Stop** → Final anchor: 2 features

---

## Dynamic Anchors Feature Selection

**Algorithm**: Reinforcement Learning (Policy Gradient)
- **Goal**: Maximize reward = α·precision_gain + β·coverage_gain
- **Process**:
  1. Policy can adjust bounds on **all features simultaneously**
  2. Each step can tighten/expand any feature's bounds
  3. Policy learns to tighten multiple features together
  4. No explicit "minimal feature" constraint
  
**Feature Selection**: Shows ALL features that were tightened by ≥5% from initial width

**Result**: Often 5+ features because:
1. RL optimizes globally - can tighten many features at once
2. Reward encourages both precision AND coverage
3. Multiple features tightened = better precision+coverage balance
4. No mechanism to prefer fewer features (unlike static's greedy selection)

**Example**:
- Tightens "mean perimeter" by 10% → helps precision
- Tightens "mean concave points" by 8% → helps precision  
- Tightens "perimeter error" by 12% → helps precision
- Tightens "concave points error" by 7% → helps precision
- Tightens "worst fractal dimension" by 9% → helps precision
- All 5 features shown (all tightened > 5%)

---

## Key Differences

| Aspect | Static | Dynamic |
|--------|--------|---------|
| **Selection Strategy** | Greedy, minimal | RL, optimized for reward |
| **Feature Constraint** | Stops when precision met | No constraint |
| **Objective** | Minimal features satisfying precision | Max precision + coverage |
| **Typical # Features** | 2-3 | 5+ |

---

## Why Dynamic Shows More

1. **No minimality constraint**: Static stops early; dynamic keeps optimizing
2. **Global optimization**: RL learns to use multiple features together
3. **Coverage objective**: β=0.6 weight on coverage means tightening many features can help balance precision+coverage
4. **Different search**: Static is greedy (one feature at a time); Dynamic explores feature combinations

---

## Solution: Limit to Top 5 Features

I've updated the code to limit dynamic anchors to show only the **top 5 tightened features** (sorted by tightness), matching the approach used in training logs. This makes the output more comparable to static anchors while still showing the most important features.

This way:
- Static: Shows 2-3 features (minimal set from bandit algorithm)
- Dynamic: Shows top 5 tightened features (most constrained features from RL)

This is more fair and interpretable!


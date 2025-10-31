# Dynamic Anchors: Continuous vs Discrete Mode

## Without Discretization (Continuous Mode)

### Environment: `AnchorEnv`

**Data Representations:**
- `X_unit`: Normalized to [0, 1] per feature
  - `X_unit = (X_std - X_min) / X_range` where X_min/X_range come from training data
- `X_std`: Standardized (z-score normalized) - actual values used by classifier
- `lower`/`upper`: Continuous bounds in normalized [0, 1] space

**Initialization:**
- `lower = [0.0, 0.0, ..., 0.0]` (all features start at 0)
- `upper = [1.0, 1.0, ..., 1.0]` (all features start at 1)
- Or if `x_star_unit` provided: centered window around that point with `initial_window` (default 0.1)

**Coverage Calculation:**
1. `_mask_in_box()`: Checks if each row in `X_unit` falls within bounds
   - `(X_unit[:, j] >= lower[j]) & (X_unit[:, j] <= upper[j])` for each feature j
   - Result: boolean mask indicating which instances are in the box
2. Coverage = fraction of instances in box = `mask.mean()`
3. Uses continuous comparisons - no binning!

**Precision Calculation:**
1. Gets indices where `mask == True` (covered instances)
2. Uses `X_std[covered]` - the standardized continuous values
3. Passes to classifier for prediction
4. Precision = fraction of covered instances predicted as target class

**Actions:**
- Step sizes: `step_fracs = (0.005, 0.01, 0.02)` (default)
- Actions move bounds by **proportional steps** relative to current width:
  - `rel_step = step_frac * current_width`
  - Example: If current width is 0.5 and step_frac=0.01, moves by 0.005
- Can make very fine-grained adjustments
- Minimum width: `min_width = 0.05` (default) - box must be at least 5% of feature range

**Rule Formatting:**
- Shows continuous intervals: `feature ∈ [0.123, 0.456]`
- Or inequality format: `feature > 0.123 and feature <= 0.456`

---

## With Discretization (Discrete Mode)

### Environment: `DiscreteAnchorEnv`

**Data Representations:**
- `X_bins`: Integer bin indices per feature (e.g., 0, 1, 2, ...)
- `X_std`: Standardized continuous values (same as continuous mode)
- `lower`/`upper`: Integer bin indices (e.g., lower=0, upper=5 means bins 0-5)
- `bin_edges`: Thresholds that define bin boundaries
- `bin_reps`: Representative continuous values for each bin (used for classifier evaluation)

**Initialization:**
- `lower = [0, 0, ..., 0]` (all features start at bin 0)
- `upper = [max_bin_0, max_bin_1, ..., max_bin_n]` (all features start at max bin)
- Each feature has different max bin count

**Coverage Calculation:**
1. `_mask_in_box()`: Checks if each row in `X_bins` falls within bin bounds
   - `(X_bins[:, j] >= lower[j]) & (X_bins[:, j] <= upper[j])` for each feature j
   - Compares **integer bin indices** (not continuous values!)
   - Result: boolean mask
2. Coverage = fraction of instances in box = `mask.mean()`
3. **Uses discrete bin comparisons** - this is different from continuous mode!

**Precision Calculation:**
1. Gets indices where `mask == True` (covered instances)
2. Uses `X_std[covered]` - the standardized continuous values (same as continuous mode)
3. Passes to classifier for prediction
4. Precision = fraction of covered instances predicted as target class
5. **Uses continuous values for classifier** (same as continuous mode)

**Actions:**
- Step sizes: `step_bins = (1, 1, 1)` - always moves by integer number of bins
- Actions move bounds by **integer bin steps**:
  - Example: If in bin 3, action moves to bin 2 or bin 4 (can't do bin 3.5)
- Minimum width: `max(2 bins, 10% of bin range)` per feature
- More coarse-grained adjustments than continuous mode

**Rule Formatting:**
- Converts bin bounds to continuous thresholds using `bin_edges`
- Shows inequalities: `feature > 0.55` or `feature <= 0.30`
- Matches static anchor format

---

## Key Differences

| Aspect | Continuous Mode | Discrete Mode |
|--------|----------------|---------------|
| **Environment** | `AnchorEnv` | `DiscreteAnchorEnv` |
| **Bounds Type** | Continuous [0, 1] | Integer bin indices |
| **Coverage Calculation** | Compares `X_unit` (normalized [0,1]) | Compares `X_bins` (integer indices) |
| **Precision Calculation** | Uses `X_std[covered]` (continuous) | Uses `X_std[covered]` (continuous) |
| **Action Steps** | Proportional fractions (0.005, 0.01, 0.02) | Integer bins (1, 1, 1) |
| **Granularity** | Very fine (can do 0.001 increments) | Coarse (can only do whole bins) |
| **Min Width** | 5% of feature range | 2 bins or 10% of bin range |
| **Rule Format** | Interval notation or inequalities | Inequalities (matches static) |

---

## Why Coverage Differs

The coverage calculation is **fundamentally different**:

- **Continuous**: Checks if normalized values fall in continuous bounds
  - Example: Feature value 0.5 falls in [0.4, 0.6] → True
  
- **Discrete**: Checks if bin indices fall in bin bounds
  - Example: Feature value maps to bin 3, bounds are bins [2, 4] → True
  - But if feature maps to bin 5, bounds [2, 4] → False (even if continuous value might be close!)

This can lead to different coverage results because:
1. Discrete mode has quantization error (continuous → bin mapping)
2. Bin boundaries may not align perfectly with the desired coverage
3. A feature value near a bin boundary might fall in/out of coverage differently in discrete vs continuous

---

## Summary

**Without Discretization:**
- Everything is continuous
- Fine-grained control (can make tiny adjustments)
- Coverage and actions work in continuous [0, 1] space
- More flexible but larger action space

**With Discretization:**
- Coverage uses discrete bins (integer comparisons)
- Precision uses continuous values (same as without discretization)
- Actions work in integer bin steps
- More constrained but matches static anchor approach
- Rule format matches static anchors (inequalities)


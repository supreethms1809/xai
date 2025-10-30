import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return logits; apply softmax only when probabilities are needed
        return self.net(x)


class AnchorEnv:
    """
    Dynamic anchors environment over a hyper-rectangle (bounding box) in feature space.

    - State: concatenation of [lower_bounds, upper_bounds] in normalized feature space (range [0, 1])
             plus current precision, coverage.
    - Actions: choose (feature_idx, direction, magnitude)
        * direction in {shrink_lower, expand_lower, shrink_upper, expand_upper}
        * magnitude in {small, medium, large} -> applied as fraction of feature range
    - Reward: precision_gain * alpha + coverage_gain * beta - overlap_penalty - invalid_penalty
              computed w.r.t. the classifier predictions.
    """

    def __init__(
        self,
        X_unit: np.ndarray,
        X_std: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        classifier: SimpleClassifier,
        device: torch.device,
        target_class: int = 1,
        step_fracs=(0.005, 0.01, 0.02),
        min_width: float = 0.05,
        alpha: float = 0.7,
        beta: float = 0.6,
        gamma: float = 0.1,
        precision_target: float = 0.8,
        coverage_target: float = 0.02,
        precision_blend_lambda: float = 0.5,
        drift_penalty_weight: float = 0.05,
        use_perturbation: bool = False,
        perturbation_mode: str = "bootstrap",  # "bootstrap" or "uniform"
        n_perturb: int = 1024,
        X_min: np.ndarray | None = None,
        X_range: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        min_coverage_floor: float = 0.005,
        js_penalty_weight: float = 0.05,
        x_star_unit: np.ndarray | None = None,
        initial_window: float = 0.1,
    ):
        self.X_unit = X_unit  # normalized to [0,1]
        self.X_std = X_std    # standardized (the scale the classifier was trained on)
        self.y = y.astype(int)
        self.feature_names = feature_names
        self.n_features = X_unit.shape[1]
        self.classifier = classifier
        self.device = device
        self.target_class = int(target_class)
        self.step_fracs = step_fracs
        self.min_width = min_width
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Actions enumerated: (feature, direction, magnitude_idx)
        self.directions = ("shrink_lower", "expand_lower", "shrink_upper", "expand_upper")
        self.n_actions = self.n_features * len(self.directions) * len(self.step_fracs)

        # Box state
        self.lower = np.zeros(self.n_features, dtype=np.float32)
        self.upper = np.ones(self.n_features, dtype=np.float32)
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        # Targets / weights
        self.precision_target = precision_target
        self.coverage_target = coverage_target
        self.precision_blend_lambda = precision_blend_lambda
        self.drift_penalty_weight = drift_penalty_weight

        # History for visualization
        self.box_history = []

        self.use_perturbation = bool(use_perturbation)
        self.perturbation_mode = str(perturbation_mode)
        self.n_perturb = int(n_perturb)
        self.X_min = X_min
        self.X_range = X_range
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.min_coverage_floor = float(min_coverage_floor)
        self.js_penalty_weight = float(js_penalty_weight)
        self.x_star_unit = x_star_unit.astype(np.float32) if x_star_unit is not None else None
        self.initial_window = float(initial_window)

    def _mask_in_box(self) -> np.ndarray:
        conds = []
        for j in range(self.n_features):
            conds.append((self.X_unit[:, j] >= self.lower[j]) & (self.X_unit[:, j] <= self.upper[j]))
        mask = np.logical_and.reduce(conds) if conds else np.ones(self.X_unit.shape[0], dtype=bool)
        return mask

    def _unit_to_std(self, X_unit_samples: np.ndarray) -> np.ndarray:
        if self.X_min is None or self.X_range is None:
            raise ValueError("X_min/X_range must be set for uniform perturbation sampling.")
        return (X_unit_samples * self.X_range) + self.X_min

    def _current_metrics(self) -> tuple:
        mask = self._mask_in_box()
        covered = np.where(mask)[0]
        coverage = float(mask.mean())
        if covered.size == 0:
            return 0.0, coverage, {"hard_precision": 0.0, "avg_prob": 0.0, "n_points": 0, "sampler": "none"}

        # Select inputs either from empirical subset or via perturbation sampler
        if not self.use_perturbation:
            X_eval = self.X_std[covered]
            y_eval = self.y[covered]
            n_points = int(X_eval.shape[0])
            sampler_note = "empirical"
        else:
            n_samp = min(self.n_perturb, max(1, covered.size))
            if self.perturbation_mode == "bootstrap":
                # Resample existing covered rows with replacement (keeps true labels)
                idx = self.rng.choice(covered, size=n_samp, replace=True)
                X_eval = self.X_std[idx]
                y_eval = self.y[idx]
                n_points = int(n_samp)
                sampler_note = "bootstrap"
            elif self.perturbation_mode == "uniform":
                # Sample uniformly within the current box in unit space; then invert to std space
                U = np.zeros((n_samp, self.n_features), dtype=np.float32)
                for j in range(self.n_features):
                    low, up = float(self.lower[j]), float(self.upper[j])
                    U[:, j] = self.rng.uniform(low=low, high=up, size=n_samp).astype(np.float32)
                X_eval = self._unit_to_std(U)
                y_eval = None  # unknown under synthetic sampling
                n_points = int(n_samp)
                sampler_note = "uniform"
            else:
                raise ValueError(f"Unknown perturbation_mode '{self.perturbation_mode}'. Use 'bootstrap' or 'uniform'.")

        with torch.no_grad():
            inputs = torch.from_numpy(X_eval).float().to(self.device)
            logits = self.classifier(inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        preds = probs.argmax(axis=1)
        positive_idx = (preds == self.target_class)
        if y_eval is None:
            # Model-consistency precision under synthetic sampling: fraction predicted as target_class
            hard_precision = float(positive_idx.mean())
        else:
            if positive_idx.sum() == 0:
                hard_precision = 0.0
            else:
                hard_precision = float((y_eval[positive_idx] == self.target_class).mean())

        avg_prob = float(probs[:, self.target_class].mean())
        precision_proxy = (
            self.precision_blend_lambda * hard_precision + (1.0 - self.precision_blend_lambda) * avg_prob
        )
        return precision_proxy, coverage, {
            "hard_precision": hard_precision,
            "avg_prob": avg_prob,
            "n_points": int(n_points),
            "sampler": sampler_note,
        }

    def reset(self):
        if self.x_star_unit is None:
            self.lower[:] = 0.0
            self.upper[:] = 1.0
        else:
            w = self.initial_window
            self.lower = np.clip(self.x_star_unit - w, 0.0, 1.0)
            self.upper = np.clip(self.x_star_unit + w, 0.0, 1.0)
        self.prev_lower = self.lower.copy()
        self.prev_upper = self.upper.copy()
        self.box_history = [(self.lower.copy(), self.upper.copy())]
        precision, coverage, _ = self._current_metrics()
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        return state

    def _apply_action(self, action: int):
        f = action // (len(self.directions) * len(self.step_fracs))
        rem = action % (len(self.directions) * len(self.step_fracs))
        d = rem // len(self.step_fracs)
        m = rem % len(self.step_fracs)

        direction = self.directions[d]
        step = float(self.step_fracs[m])
        # Make step relative to current width for scale-invariant moves
        cur_width = max(1e-6, self.upper[f] - self.lower[f])
        rel_step = step * cur_width

        if direction == "shrink_lower":
            self.lower[f] = min(self.lower[f] + rel_step, self.upper[f] - self.min_width)
        elif direction == "expand_lower":
            self.lower[f] = max(self.lower[f] - rel_step, 0.0)
        elif direction == "shrink_upper":
            self.upper[f] = max(self.upper[f] - rel_step, self.lower[f] + self.min_width)
        elif direction == "expand_upper":
            self.upper[f] = min(self.upper[f] + rel_step, 1.0)

        # Ensure valid
        if self.upper[f] - self.lower[f] < self.min_width:
            mid = 0.5 * (self.upper[f] + self.lower[f])
            self.lower[f] = max(0.0, mid - self.min_width / 2.0)
            self.upper[f] = min(1.0, mid + self.min_width / 2.0)

    def step(self, action: int):
        prev_precision, prev_coverage, _ = self._current_metrics()
        prev_lower = self.lower.copy()
        prev_upper = self.upper.copy()
        # Pre-compute previous box volume
        prev_widths = np.maximum(prev_upper - prev_lower, 1e-9)
        prev_vol = float(np.prod(prev_widths))
        self._apply_action(action)
        precision, coverage, details = self._current_metrics()

        # Enforce a minimum coverage floor by reverting overly aggressive actions
        coverage_clipped = False
        if coverage < self.min_coverage_floor:
            # revert bounds
            self.lower = prev_lower
            self.upper = prev_upper
            # recompute with reverted bounds
            precision, coverage, details = self._current_metrics()
            coverage_clipped = True

        precision_gain = precision - prev_precision
        coverage_gain = coverage - prev_coverage

        # Penalize too small boxes
        widths = self.upper - self.lower
        overlap_penalty = self.gamma * float((widths < (2 * self.min_width)).mean())

        # Penalize large drift to promote stability
        drift = float(np.linalg.norm(self.upper - prev_upper) + np.linalg.norm(self.lower - prev_lower))
        drift_penalty = self.drift_penalty_weight * drift

        # JS-like penalty based on volume overlap (distributional shift proxy)
        # Compute intersection and union volumes of axis-aligned boxes in unit space
        inter_lower = np.maximum(self.lower, prev_lower)
        inter_upper = np.minimum(self.upper, prev_upper)
        inter_widths = np.maximum(inter_upper - inter_lower, 0.0)
        inter_vol = float(np.prod(np.maximum(inter_widths, 0.0)))
        curr_widths = np.maximum(self.upper - self.lower, 1e-9)
        curr_vol = float(np.prod(curr_widths))
        eps = 1e-12
        if inter_vol <= eps:
            js_proxy = 1.0  # maximal mismatch
        else:
            # Symmetric KL proxy over uniform distributions on the two boxes
            dkl_prev_to_mix = np.log((prev_vol + curr_vol) / (2.0 * inter_vol + eps) + eps)
            dkl_curr_to_mix = np.log((prev_vol + curr_vol) / (2.0 * inter_vol + eps) + eps)
            # Since both directions equal for uniform + overlap based on intersection with mixture,
            # JS proxy simplifies to this shared log term; keep bounded in [0,1] via mapping
            js_proxy = 1.0 - float(inter_vol / (0.5 * (prev_vol + curr_vol) + eps))
            js_proxy = float(np.clip(js_proxy, 0.0, 1.0))
        js_penalty = self.js_penalty_weight * js_proxy

        reward = self.alpha * precision_gain + self.beta * coverage_gain - overlap_penalty - drift_penalty - js_penalty

        self.box_history.append((self.lower.copy(), self.upper.copy()))
        self.prev_lower = prev_lower
        self.prev_upper = prev_upper
        state = np.concatenate([self.lower, self.upper, np.array([precision, coverage], dtype=np.float32)])
        done = bool(precision >= self.precision_target and coverage >= self.coverage_target)
        info = {"precision": precision, "coverage": coverage, "drift": drift, "js_penalty": js_penalty, "coverage_clipped": coverage_clipped, **details}
        return state, reward, done, info


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits

class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)

# --- Device selection helper ---
DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")

def select_device(device_preference: str = "auto") -> torch.device:
    device_preference = (device_preference or "auto").lower()
    if device_preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_preference == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device_preference == "cpu":
        return torch.device("cpu")
    # auto: prefer CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_dynamic_anchors(
    episodes: int = 30,
    steps_per_episode: int = 60,
    classifier_epochs_per_round: int = 3,
    seed: int = 42,
    target_classes=None,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    reg_lambda_inside_anchor: float = 0.0,
    dataset: str = "covtype",
    device_preference: str = "auto",
    use_perturbation: bool = True,
    perturbation_mode: str = "uniform",
    n_perturb: int = 4096,
    debug: bool = True,
    local_instance_index: int = -1,
    initial_window: float = 0.1,
):
    """
    Train dynamic anchors using RL and classifier co-training.

    Args:
        episodes: number of RL episodes
        steps_per_episode: RL steps per episode
        classifier_epochs_per_round: classifier epochs per RL episode
        seed: random seed
        target_classes: tuple of target class labels
        entropy_coef: entropy regularization coefficient
        value_coef: value loss coefficient
        reg_lambda_inside_anchor: regularization inside anchor
        dataset: which dataset to use; one of 'breast_cancer', 'synthetic', or 'covtype'
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Dataset loading: supports 'breast_cancer', 'synthetic', or 'covtype'
    if dataset == "breast_cancer":
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = list(ds.feature_names)
    elif dataset == "synthetic":
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=12, n_informative=8, n_classes=2, random_state=seed)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    elif dataset == "covtype":
        X, y = fetch_covtype(return_X_y=True, as_frame=False)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'breast_cancer', 'synthetic', or 'covtype'.")

    # Normalize class labels to 0..C-1 if needed
    unique_classes = np.unique(y)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_to_idx[c] for c in y], dtype=int)
    num_classes = int(len(unique_classes))
    if target_classes is None:
        target_classes = tuple(range(num_classes))

    # Split before scaling to avoid leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # Build unit-space stats from train only; apply to both
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit_train = (X_train - X_min) / X_range
    X_unit_test = (X_test - X_min) / X_range

    device = select_device(device_preference)
    print(f"[device] using {device}")
    classifier = SimpleClassifier(X_train.shape[1], num_classes).to(device)
    clf_opt = optim.Adam(classifier.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    # Environment built on full normalized data but will reflect classifier behavior
    rng_local = np.random.default_rng(seed)
    # Local per-instance mode: restrict to class of x* and center initial box near x*
    x_star_unit = None
    if local_instance_index is not None and local_instance_index >= 0:
        idx = int(local_instance_index)
        idx = max(0, min(idx, X_unit_test.shape[0] - 1))
        x_star_unit = X_unit_test[idx]
        y_star = int(y_test[idx])
        target_classes = (y_star,)
        if debug:
            print(f"[local] anchoring on test idx={idx}, class={y_star}")

    envs = {
        c: AnchorEnv(
            X_unit_train, X_train, y_train, feature_names, classifier, device,
            target_class=c,
            X_min=X_min, X_range=X_range,
            use_perturbation=use_perturbation,
            perturbation_mode=perturbation_mode,
            n_perturb=n_perturb,
            rng=rng_local,
            x_star_unit=x_star_unit,
            initial_window=initial_window,
        ) for c in target_classes
    }

    # Assume homogeneous state/action dims across envs
    any_env = next(iter(envs.values()))
    state_dim = 2 * any_env.n_features + 2
    action_dim = any_env.n_actions
    policy = PolicyNet(state_dim, action_dim).to(device)
    value_fn = ValueNet(state_dim).to(device)
    policy_opt = optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=3e-4)

    def train_classifier_one_round(batch_size: int = 256):
        classifier.train()
        dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        last_loss = None
        last_train_acc = None
        for e in range(1, classifier_epochs_per_round + 1):
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_count = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                clf_opt.zero_grad()
                logits = classifier(xb)
                loss = ce(logits, yb)
                # Optional regularization for consistency inside high-precision boxes
                if reg_lambda_inside_anchor > 0.0:
                    with torch.no_grad():
                        combined_mask = np.zeros(X.shape[0], dtype=bool)
                        for env in envs.values():
                            prec, cov, det = env._current_metrics()
                            if det["hard_precision"] >= env.precision_target and det["n_points"] > 0:
                                combined_mask |= env._mask_in_box()
                        idx = np.where(combined_mask)[0]
                    if idx.size > 0:
                        in_box_inputs = torch.from_numpy(X[idx]).float().to(device)
                        with torch.no_grad():
                            p_detach = classifier(in_box_inputs).detach()
                        reg = p_detach.var()
                        loss = loss + reg_lambda_inside_anchor * reg
                loss.backward()
                clf_opt.step()

                with torch.no_grad():
                    preds = logits.argmax(dim=1)
                    correct = (preds == yb).sum().item()
                    epoch_correct += correct
                    epoch_count += yb.size(0)
                    epoch_loss_sum += loss.item() * yb.size(0)

            last_loss = epoch_loss_sum / max(1, epoch_count)
            last_train_acc = epoch_correct / max(1, epoch_count)
            print(f"[clf] epoch {e}/{classifier_epochs_per_round} | loss={last_loss:.4f} | train_acc={last_train_acc:.3f} | samples={epoch_count}")
        return last_loss, last_train_acc

    def evaluate_classifier():
        classifier.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(X_test).float().to(device)
            logits = classifier(inputs)
            preds = logits.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        return float(acc)

    episode_rewards = []
    test_acc_history = []
    box_history_per_episode = []
    drift_history_per_episode = []
    prec_cov_history_per_episode = []
    # Per-class precision/coverage logging
    per_class_prec_cov = {c: [] for c in target_classes}
    # Per-class final box per episode
    per_class_box_history = {c: [] for c in target_classes}

    for ep in range(episodes):
        # 1) Train classifier a bit each round
        last_loss, last_train_acc = train_classifier_one_round()
        acc = evaluate_classifier()
        test_acc_history.append(acc)

        # 2) RL loop adjusting anchor boxes per class using current classifier
        episode_drifts = []
        episode_prec_cov = []
        total_return = 0.0
        for cls, env in envs.items():
            state = env.reset()
            # Capture true pre-step metrics for histories, but print zeros for reset log
            tp0, tc0, td0 = env._current_metrics()
            episode_prec_cov.append((tp0, tc0, td0.get("hard_precision", 0.0)))
            if debug:
                lw = (env.upper - env.lower)
                tightened = np.where(lw < 0.999)[0]
                # Show tightest 3 features (smallest widths)
                topk_narrow = np.argsort(lw)[:3]
                narrow_bounds = ", ".join([f"{feature_names[i]}:[{env.lower[i]:.2f},{env.upper[i]:.2f}]" for i in topk_narrow])
                print(f"[env cls={cls}] reset | prec=0.000 hard_prec=0.000 cov=0.000 n=0 | width_mean={lw.mean():.3f} width_min={lw.min():.3f} | tightened={len(tightened)} | narrow {narrow_bounds}")
            log_probs = []
            values = []
            rewards = []
            entropies = []
            gamma = 0.99
            info = {}  # ensure defined if no steps
            for t in range(steps_per_episode):
                classifier.eval()
                s = torch.from_numpy(state).float().to(device)
                logits = policy(s)
                probs_pi = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs_pi)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = value_fn(s)

                next_state, reward, done, info = env.step(int(action.item()))
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
                entropies.append(dist.entropy())
                episode_drifts.append(info.get("drift", 0.0))
                episode_prec_cov.append((info.get("precision", 0.0), info.get("coverage", 0.0), info.get("hard_precision", 0.0)))
                state = next_state
                if done:
                    break

            # Per-class precision/coverage logging (track last info for this class)
            last_info_for_cls = info if 'info' in locals() else {}
            if debug:
                lw = (env.upper - env.lower)
                tightened = np.where(lw < 0.999)[0]
                topk_narrow = np.argsort(lw)[:3]
                narrow_bounds = ", ".join([f"{feature_names[i]}:[{env.lower[i]:.2f},{env.upper[i]:.2f}]" for i in topk_narrow])
                print(f"[env cls={cls}] end   | prec={last_info_for_cls.get('precision', 0.0):.3f} hard_prec={last_info_for_cls.get('hard_precision', 0.0):.3f} cov={last_info_for_cls.get('coverage', 0.0):.3f} n={last_info_for_cls.get('n_points', 0)} | width_mean={lw.mean():.3f} width_min={lw.min():.3f} | tightened={len(tightened)} | narrow {narrow_bounds}")
                # Human-readable rule summary with confidence
                # Select up to 5 tightened features to show
                tightened_sorted = np.argsort(lw[tightened]) if tightened.size > 0 else np.array([])
                to_show_idx = (tightened[tightened_sorted[:5]] if tightened.size > 0 else np.array([], dtype=int))
                if to_show_idx.size == 0:
                    cond_str = "any values (no tightened features)"
                else:
                    cond_parts = [f"{feature_names[i]} ∈ [{env.lower[i]:.2f}, {env.upper[i]:.2f}]" for i in to_show_idx]
                    cond_str = " and ".join(cond_parts)
                print(
                    f"[rule cls={cls}] IF {cond_str} THEN class={cls} | "
                    f"soft={last_info_for_cls.get('avg_prob', 0.0):.3f}, hard={last_info_for_cls.get('hard_precision', 0.0):.3f}, "
                    f"blended={last_info_for_cls.get('precision', 0.0):.3f}, coverage={last_info_for_cls.get('coverage', 0.0):.3f}, sampler={last_info_for_cls.get('sampler', 'empirical')}"
                )
            per_class_prec_cov[cls].append({
                'precision': last_info_for_cls.get('precision', 0.0),
                'hard_precision': last_info_for_cls.get('hard_precision', 0.0),
                'coverage': last_info_for_cls.get('coverage', 0.0),
            })
            per_class_box_history[cls].append((env.lower.copy(), env.upper.copy()))

            # Advantage with value baseline and entropy bonus
            policy_opt.zero_grad()
            R = torch.zeros(1, dtype=torch.float32, device=device)
            policy_loss = torch.zeros(1, dtype=torch.float32, device=device)
            value_loss = torch.zeros(1, dtype=torch.float32, device=device)
            for t in reversed(range(len(rewards))):
                R = rewards[t] + gamma * R
                advantage = R - values[t]
                policy_loss = policy_loss - log_probs[t] * advantage.detach()
                value_loss = value_loss + advantage.pow(2) * 0.5
            entropy_term = -entropy_coef * torch.stack(entropies).mean() if entropies else 0.0
            loss = policy_loss + value_coef * value_loss + entropy_term
            loss.backward()
            policy_opt.step()

            total_return += float(torch.stack(rewards).sum().item()) if rewards else 0.0

        # Save box history for visualization (use last env as representative) and logs
        last_env = next(reversed(envs.values()))
        box_history_per_episode.append(last_env.box_history.copy())
        drift_history_per_episode.append(episode_drifts)
        prec_cov_history_per_episode.append(episode_prec_cov)

        episode_rewards.append(total_return)
        last_p, last_c, last_hp = episode_prec_cov[-1] if episode_prec_cov else (0.0, 0.0, 0.0)
        print(f"Episode {ep+1}/{episodes} | return={total_return:.3f} | last_clf_loss={last_loss:.4f} | train_acc={last_train_acc:.3f} | test_acc={acc:.3f} | last_precision={last_p:.3f} | last_cov={last_c:.3f} | last_hard_precision={last_hp:.3f}")

    # Visualization: show evolution of two most varying features
    feat_var = X_unit_train.var(axis=0)
    top2 = np.argsort(-feat_var)[:2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(episode_rewards)
    axes[0].set_title("Episode returns")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")

    axes[1].plot(test_acc_history)
    axes[1].set_title("Classifier test accuracy")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

    # Plot box bounds over episodes for top-2 features
    lower_series_f0 = [h[0][top2[0]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]
    upper_series_f0 = [h[1][top2[0]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]
    lower_series_f1 = [h[0][top2[1]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]
    upper_series_f1 = [h[1][top2[1]] for ep_hist in box_history_per_episode for h in [ep_hist[-1]]]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lower_series_f0, label=f"{feature_names[top2[0]]} lower")
    plt.plot(upper_series_f0, label=f"{feature_names[top2[0]]} upper")
    plt.title("Anchor bounds over episodes (feature 1)")
    plt.xlabel("Episode")
    plt.ylabel("Normalized bound")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lower_series_f1, label=f"{feature_names[top2[1]]} lower")
    plt.plot(upper_series_f1, label=f"{feature_names[top2[1]]} upper")
    plt.title("Anchor bounds over episodes (feature 2)")
    plt.xlabel("Episode")
    plt.ylabel("Normalized bound")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot drift per episode and precision-coverage trajectory
    avg_drift = [np.mean(d) if len(d) > 0 else 0.0 for d in drift_history_per_episode]
    avg_prec = [np.mean([pc[0] for pc in ep]) if len(ep) > 0 else 0.0 for ep in prec_cov_history_per_episode]
    avg_cov = [np.mean([pc[1] for pc in ep]) if len(ep) > 0 else 0.0 for ep in prec_cov_history_per_episode]
    avg_hard_prec = [np.mean([pc[2] for pc in ep]) if len(ep) > 0 else 0.0 for ep in prec_cov_history_per_episode]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(avg_drift)
    axes[0].set_title("Average drift per episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Drift")

    axes[1].plot(avg_prec, label="blended precision")
    axes[1].plot(avg_hard_prec, label="hard precision")
    axes[1].set_title("Precision over episodes")
    axes[1].legend()

    axes[2].plot(avg_cov)
    axes[2].set_title("Coverage over episodes")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Coverage")
    plt.tight_layout()
    plt.show()

    # Per-class precision/coverage over episodes
    if per_class_prec_cov and len(per_class_prec_cov) > 0:
        episodes_idx = np.arange(1, episodes + 1)
        # Ensure equal-length series per class by padding with zeros if some episodes missing
        def series_for(cls, key):
            vals = [d.get(key, 0.0) for d in per_class_prec_cov.get(cls, [])]
            if len(vals) < episodes:
                vals = vals + [0.0] * (episodes - len(vals))
            return np.array(vals[:episodes])

        plt.figure(figsize=(14, 5))
        # Hard precision per class
        plt.subplot(1, 2, 1)
        for cls in target_classes:
            hp = series_for(cls, 'hard_precision')
            plt.plot(episodes_idx, hp, label=f'class {cls}')
        plt.title('Per-class hard precision over episodes')
        plt.xlabel('Episode')
        plt.ylabel('Hard precision')
        plt.legend(ncol=2, fontsize=8)

        # Coverage per class
        plt.subplot(1, 2, 2)
        for cls in target_classes:
            cov = series_for(cls, 'coverage')
            plt.plot(episodes_idx, cov, label=f'class {cls}')
        plt.title('Per-class coverage over episodes')
        plt.xlabel('Episode')
        plt.ylabel('Coverage')
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

    # Final confusion matrix on test set
    classifier.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).float().to(device)
        probs_final = classifier(inputs).cpu().numpy()
    final_preds = probs_final.argmax(axis=1)
    cm = confusion_matrix(y_test, final_preds, labels=list(range(num_classes)))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (test set)')
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.show()

    return {
        "episode_returns": episode_rewards,
        "test_accuracy": test_acc_history,
        "box_history": box_history_per_episode,
        "drift_history": drift_history_per_episode,
        "precision_coverage_history": prec_cov_history_per_episode,
        "top2_features": [feature_names[i] for i in top2],
        "per_class_precision_coverage_history": per_class_prec_cov,
        "per_class_box_history": per_class_box_history,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run dynamic anchor training.")
    parser.add_argument("--dataset", type=str, default="covtype", choices=["breast_cancer", "synthetic", "covtype"], help="Dataset to use")
    parser.add_argument("--episodes", type=int, default=30, help="Number of RL episodes")
    parser.add_argument("--steps", type=int, default=40, help="Steps per episode")
    parser.add_argument("--classifier_epochs", type=int, default=3, help="Classifier epochs per RL episode")
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="Regularization inside anchor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=list(DEVICE_CHOICES), help="Device: auto|cuda|mps|cpu")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_perturbation", dest="use_perturbation", action="store_true", help="Enable perturbation-based sampling inside boxes")
    group.add_argument("--no-perturbation", dest="use_perturbation", action="store_false", help="Disable perturbation-based sampling inside boxes")
    parser.set_defaults(use_perturbation=True)
    parser.add_argument("--perturbation_mode", type=str, default="uniform", choices=["bootstrap", "uniform"], help="Sampler when perturbations are enabled")
    parser.add_argument("--local_instance_index", type=int, default=-1, help="If >=0, run local per-instance anchor for test instance index")
    parser.add_argument("--initial_window", type=float, default=0.1, help="Initial half-width around x* in unit space for local anchors")
    parser.add_argument("--n_perturb", type=int, default=4096, help="Number of synthetic/bootstrapped samples per box evaluation")

    args = parser.parse_args()

    train_dynamic_anchors(
        dataset=args.dataset,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        classifier_epochs_per_round=args.classifier_epochs,
        reg_lambda_inside_anchor=args.reg_lambda,
        seed=args.seed,
        device_preference=args.device,
        use_perturbation=args.use_perturbation,
        perturbation_mode=args.perturbation_mode,
        n_perturb=args.n_perturb,
        local_instance_index=args.local_instance_index,
        initial_window=args.initial_window,
    )



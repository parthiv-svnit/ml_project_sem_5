"""
Uncertainty-Guided Oversampling (UGO) Pipeline — Complete Implementation
=========================================================================
Addresses three algorithmic challenges:
  (a) Boundary-aware generation without class overlap  → Module 3 (w_i weights)
  (b) Intra-class imbalance                           → Module 1 (HDBSCAN budgets)
  (c) Semi-supervised unlabeled data                  → Module 2 (Label Propagation)

Module execution order:
  Module 1: HDBSCAN clustering → per-cluster generation budgets
  Module 2: Label Propagation  → BoundaryScore for unlabeled samples
  Module 3: w_i weights        → epistemic uncertainty + SSL boundary geometry
  Generation: KNN interpolation guided by w_i, budgeted by cluster

Authors: [Your Team]
Project: Uncertainty-Guided Boundary Resampling for Financial Fraud Detection
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation


# =============================================================================
# SHARED DATA STRUCTURES
# =============================================================================

@dataclass
class ClusterProfile:
    """
    All per-cluster metadata produced by Module 1.
    Passed as a list (one per cluster) into the generation step.
    """
    cluster_id: int              # HDBSCAN cluster label (never -1; noise is separate)
    member_indices: np.ndarray   # indices into X_minority for members of this cluster
    size: int                    # number of minority samples in cluster
    density: float               # mean HDBSCAN membership probability of members
    budget: int                  # synthetic samples to generate FROM this cluster
    raw_budget_weight: float     # un-normalised inverse-density weight (for diagnostics)


@dataclass
class NoiseProfile:
    """Holds noise points (-1) separately — generation is capped and guarded."""
    member_indices: np.ndarray   # indices into X_minority
    budget: int                  # small fixed budget (see NOISE_BUDGET_FRACTION)


@dataclass
class Module1Output:
    clusters: List[ClusterProfile]
    noise: NoiseProfile
    cluster_labels: np.ndarray   # shape (n_minority,), -1 for noise
    membership_probs: np.ndarray # shape (n_minority,), HDBSCAN soft membership


@dataclass
class Module2Output:
    boundary_scores: np.ndarray        # shape (n_unlabeled,)
    pseudo_probs_unlabeled: np.ndarray # shape (n_unlabeled, 2)
    pseudo_probs_labeled: np.ndarray   # shape (n_labeled, 2)


@dataclass
class Module3Output:
    w_i: np.ndarray                    # shape (n_minority,), sums to 1
    epistemic_uncertainty: np.ndarray  # shape (n_minority,), pre-normalisation
    neighbor_boundary_scores: np.ndarray # shape (n_minority,), pre-normalisation
    alpha_used: float


# =============================================================================
# MODULE 1: Intra-Class Clustering + Budget Allocation
# =============================================================================

# Noise points receive at most this fraction of the total oversampling budget.
# Keeps outlier regions from being over-populated. Tune conservatively.
NOISE_BUDGET_FRACTION = 0.05


def run_module1(
    X_minority: np.ndarray,
    total_synthetic_budget: int,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    cluster_selection_epsilon: float = 0.0,
    noise_budget_fraction: float = NOISE_BUDGET_FRACTION,
) -> Module1Output:
    """
    Cluster the minority class with HDBSCAN, then assign a generation budget
    to each cluster inversely proportional to its size × density.

    WHY INVERSE SIZE × DENSITY:
        Large, dense clusters are already well-represented in feature space.
        Small or sparse sub-clusters (e.g. a rare fraud sub-type) contain
        patterns the model has the least exposure to. Allocating more budget
        there directly addresses intra-class imbalance — the core of challenge (b).

    BUDGET FORMULA (per cluster c):
    ─────────────────────────────
        raw_weight(c)  = 1 / (size(c) × density(c))

        where density(c) = mean HDBSCAN membership probability of members in c.
        Membership probability ∈ (0, 1]: points near the cluster core → ≈1.0,
        points on the fringe → closer to 0. Low mean density = sparse cluster.

        budget(c) = round( raw_weight(c) / Σ raw_weight(c') × cluster_budget )

        cluster_budget = total_synthetic_budget × (1 - noise_budget_fraction)

    HDBSCAN PARAMETERS (for your defense):
        min_cluster_size : smallest grouping treated as a cluster. Set to ≈5% of
                           n_minority. Too small → noise fragmentation.
        min_samples      : controls conservativeness. Higher → more noise points.
        cluster_selection_epsilon : merges clusters closer than ε in the dendrogram.
                           Useful if HDBSCAN over-splits dense regions.

    Parameters
    ----------
    X_minority            : shape (n_minority, n_features) — already scaled
    total_synthetic_budget: total number of synthetic samples to produce
    noise_budget_fraction : fraction of budget reserved for noise points (default 5%)

    Returns
    -------
    Module1Output with populated ClusterProfile list and NoiseProfile.
    """
    if len(X_minority) < min_cluster_size:
        raise ValueError(
            f"X_minority has only {len(X_minority)} samples; "
            f"need at least min_cluster_size={min_cluster_size}."
        )

    # --- Fit HDBSCAN ---
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        store_centers="medoid",   # gives us a representative point per cluster
    )
    hdb.fit(X_minority)

    cluster_labels   = hdb.labels_           # shape: (n_minority,), -1 = noise
    membership_probs = hdb.probabilities_    # shape: (n_minority,), ∈ [0, 1]

    unique_clusters = sorted(set(cluster_labels) - {-1})
    n_clusters      = len(unique_clusters)

    if n_clusters == 0:
        warnings.warn(
            "HDBSCAN found zero clusters — all minority points are noise. "
            "Reduce min_cluster_size or min_samples. Falling back to a single cluster.",
            RuntimeWarning,
        )
        # Treat every point as one cluster so the pipeline doesn't crash
        cluster_labels   = np.zeros(len(X_minority), dtype=int)
        membership_probs = np.ones(len(X_minority))
        unique_clusters  = [0]

    # --- Separate noise ---
    noise_mask    = cluster_labels == -1
    noise_indices = np.where(noise_mask)[0]
    noise_budget  = round(total_synthetic_budget * noise_budget_fraction)
    noise_profile = NoiseProfile(member_indices=noise_indices, budget=noise_budget)

    # --- Per-cluster budget via inverse size × density ---
    cluster_budget_total = total_synthetic_budget - noise_budget

    raw_weights = {}
    for cid in unique_clusters:
        mask        = cluster_labels == cid
        indices     = np.where(mask)[0]
        size        = int(mask.sum())
        density     = float(membership_probs[mask].mean())
        density     = max(density, 1e-6)   # guard against zero-density edge case

        # Core formula: inversely proportional to size × density
        raw_weights[cid] = 1.0 / (size * density)

    weight_sum = sum(raw_weights.values())

    clusters: List[ClusterProfile] = []
    allocated = 0
    for i, cid in enumerate(unique_clusters):
        mask    = cluster_labels == cid
        indices = np.where(mask)[0]
        size    = int(mask.sum())
        density = float(membership_probs[mask].mean())

        norm_weight  = raw_weights[cid] / weight_sum
        # Last cluster absorbs rounding remainder so budgets sum exactly
        if i < n_clusters - 1:
            budget = round(norm_weight * cluster_budget_total)
            allocated += budget
        else:
            budget = cluster_budget_total - allocated

        clusters.append(ClusterProfile(
            cluster_id=cid,
            member_indices=indices,
            size=size,
            density=density,
            budget=max(budget, 0),
            raw_budget_weight=raw_weights[cid],
        ))

    _log_cluster_summary(clusters, noise_profile, total_synthetic_budget)

    return Module1Output(
        clusters=clusters,
        noise=noise_profile,
        cluster_labels=cluster_labels,
        membership_probs=membership_probs,
    )


def _log_cluster_summary(
    clusters: List[ClusterProfile],
    noise: NoiseProfile,
    total_budget: int,
) -> None:
    print(f"\n[Module 1] HDBSCAN found {len(clusters)} cluster(s) + "
          f"{len(noise.member_indices)} noise point(s). Total budget: {total_budget}")
    print(f"  {'ClusterID':>9} {'Size':>6} {'Density':>9} {'Budget':>8} {'Weight':>8}")
    for cp in clusters:
        print(f"  {cp.cluster_id:>9} {cp.size:>6} {cp.density:>9.4f} "
              f"{cp.budget:>8} {cp.raw_budget_weight:>8.4f}")
    print(f"  {'NOISE':>9} {len(noise.member_indices):>6} {'—':>9} {noise.budget:>8}")


# =============================================================================
# MODULE 2: Semi-Supervised Boundary Estimation
# =============================================================================

def run_module2(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    kernel: str = "rbf",
    gamma: float = 20.0,
    n_neighbors_knn: int = 7,
    max_iter: int = 1000,
) -> Module2Output:
    """
    Label Propagation over labeled + unlabeled data → BoundaryScore per unlabeled sample.

    Graph Laplacian decomposition (solved by sklearn internally):
        f_u = -L_uu⁻¹ L_ul f_l

    BoundaryScore(x) = 1 - |2·P(y=1|x) - 1|
        → 1.0 when P = 0.5 (maximally uncertain = on boundary)
        → 0.0 when P = 0 or 1 (clearly one class)

    Leakage note: fit LP only on training fold data; never on validation/test.
    """
    # Stack: sklearn uses -1 as sentinel for unlabeled
    X_all = np.vstack([X_labeled, X_unlabeled])
    y_all = np.concatenate([y_labeled, -np.ones(len(X_unlabeled), dtype=int)])

    lp = LabelPropagation(
        kernel=kernel,
        gamma=gamma,
        n_neighbors=n_neighbors_knn,
        max_iter=max_iter,
        tol=1e-3,
    )
    lp.fit(X_all, y_all)

    all_probs  = lp.label_distributions_        # (n_all, n_classes)
    n_labeled  = len(X_labeled)
    probs_lab  = all_probs[:n_labeled]
    probs_unl  = all_probs[n_labeled:]

    # P(y=minority | x): assumes class ordering [0, 1]; verify with lp.classes_
    p_minority      = probs_unl[:, 1]
    boundary_scores = 1.0 - np.abs(2.0 * p_minority - 1.0)

    return Module2Output(
        boundary_scores=boundary_scores,
        pseudo_probs_unlabeled=probs_unl,
        pseudo_probs_labeled=probs_lab,
    )


# =============================================================================
# MODULE 3: Generation Weights (Epistemic Uncertainty + SSL Boundary)
# =============================================================================

def compute_epistemic_uncertainty(
    X_minority: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    """
    U_epistemic(x) = Var_{t=1..T}[ p_t(y=1 | x) ]

    Variance across RF tree predictions. High variance → model disagrees about
    this region → sample lies near a complex decision boundary.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    minority_col = list(rf.classes_).index(1)

    tree_preds = np.array([
        tree.predict_proba(X_minority)[:, minority_col]
        for tree in rf.estimators_
    ])  # (n_estimators, n_minority)

    return np.var(tree_preds, axis=0)  # (n_minority,)


def compute_neighbor_boundary_scores(
    X_minority: np.ndarray,
    X_unlabeled: np.ndarray,
    boundary_scores: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    For each minority sample x_i, find its k nearest unlabeled neighbors and
    return their mean BoundaryScore — capturing local SSL boundary geometry.
    """
    k_actual = min(k, len(X_unlabeled))
    if k_actual == 0:
        return np.zeros(len(X_minority))

    knn = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree", metric="euclidean")
    knn.fit(X_unlabeled)
    _, indices = knn.kneighbors(X_minority)     # (n_minority, k_actual)
    return boundary_scores[indices].mean(axis=1) # (n_minority,)


def compute_generation_weights(
    epistemic_uncertainty: np.ndarray,
    neighbor_boundary_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    w_i = α · norm(U_epistemic(x_i)) + (1-α) · norm(mean_BoundaryScore(N_k(x_i)))

    Both terms min-max normalised before blending so α is scale-invariant.
    Output is normalised to a probability distribution (sums to 1).
    """
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return np.ones_like(arr) if (hi - lo) < 1e-10 else (arr - lo) / (hi - lo)

    w_i = alpha * _minmax(epistemic_uncertainty) + (1.0 - alpha) * _minmax(neighbor_boundary_scores)
    return w_i / w_i.sum()


def run_module3(
    X_minority: np.ndarray,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    m2_output: Module2Output,
    alpha: float,
    k_neighbors: int = 5,
) -> Module3Output:
    epi_unc = compute_epistemic_uncertainty(X_minority, X_labeled, y_labeled)
    nbr_bs  = compute_neighbor_boundary_scores(
        X_minority, X_unlabeled, m2_output.boundary_scores, k=k_neighbors
    )
    w_i = compute_generation_weights(epi_unc, nbr_bs, alpha)
    return Module3Output(
        w_i=w_i,
        epistemic_uncertainty=epi_unc,
        neighbor_boundary_scores=nbr_bs,
        alpha_used=alpha,
    )


# =============================================================================
# GENERATION: Full KNN Interpolation with Cluster Budgets + Noise Guard
# =============================================================================

def generate_synthetic_samples(
    X_minority: np.ndarray,
    m1_output: Module1Output,
    m3_output: Module3Output,
    k_interp: int = 5,
    noise_k_interp: int = 3,
    noise_membership_threshold: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce synthetic minority samples using KNN interpolation, guided by:
      - Per-cluster generation budgets  (from Module 1)
      - Per-sample generation weights   (from Module 3: w_i)
      - Noise-point safety guards       (described below)

    INTERPOLATION FORMULA (SMOTE-style, per synthetic sample):
    ──────────────────────────────────────────────────────────
        x_new = x_seed + λ · (x_neighbor - x_seed),  λ ~ Uniform(0, 1)

    where x_seed is drawn from cluster members with probabilities proportional
    to their w_i values (renormalised within the cluster), and x_neighbor is
    one of x_seed's k nearest neighbors WITHIN THE SAME CLUSTER.

    Intra-cluster KNN is critical: interpolating across cluster boundaries would
    create samples in the majority-class region between clusters, worsening overlap.

    NOISE POINT STRATEGY (three-layer guard):
    ─────────────────────────────────────────
    HDBSCAN labels a minority point as noise (-1) when it does not belong
    to any sufficiently dense region — i.e. it is an extreme outlier.
    Generating freely from these points would:
      1. Create synthetic samples in low-density, potentially nonsensical regions.
      2. Risk pushing points into majority-class territory (class overlap).

    Three guards applied:
      Guard 1 — Capped budget: noise points share only NOISE_BUDGET_FRACTION
                (default 5%) of the total budget, regardless of w_i.
      Guard 2 — Membership threshold: noise points with HDBSCAN membership
                probability < noise_membership_threshold are skipped entirely.
                These are the most extreme outliers.
      Guard 3 — Interpolation partner from nearest CLUSTER point: instead of
                interpolating noise↔noise (which would generate more outliers),
                we pair each noise seed with its nearest NON-NOISE minority neighbor.
                This gently pulls generated samples back toward a known cluster.

    Parameters
    ----------
    X_minority                   : shape (n_minority, n_features) — scaled
    m1_output                    : output of run_module1()
    m3_output                    : output of run_module3()
    k_interp                     : KNN neighbors for intra-cluster interpolation
    noise_k_interp               : KNN neighbors for noise→cluster interpolation
    noise_membership_threshold   : minimum HDBSCAN membership prob to generate from a noise point
    random_state                 : reproducibility

    Returns
    -------
    X_synthetic : shape (n_generated, n_features)
    y_synthetic : shape (n_generated,) — all 1s (minority class)
    """
    rng         = np.random.default_rng(random_state)
    w_i         = m3_output.w_i
    mem_probs   = m1_output.membership_probs
    all_synth   = []

    # ── CLUSTER GENERATION ──────────────────────────────────────────────────
    for cp in m1_output.clusters:
        if cp.budget <= 0 or len(cp.member_indices) < 2:
            continue

        X_cluster = X_minority[cp.member_indices]              # (size_c, n_feat)
        w_cluster = w_i[cp.member_indices]                     # (size_c,)
        w_cluster = w_cluster / w_cluster.sum()                # renormalise within cluster

        # Intra-cluster KNN: neighbors come only from within the same cluster
        k_c = min(k_interp, len(X_cluster) - 1)
        knn = NearestNeighbors(n_neighbors=k_c, metric="euclidean")
        knn.fit(X_cluster)
        _, nbr_idx = knn.kneighbors(X_cluster)                 # (size_c, k_c)

        # Draw seed indices proportional to intra-cluster w_i
        seed_positions = rng.choice(len(X_cluster), size=cp.budget, p=w_cluster, replace=True)

        cluster_synth = _interpolate_batch(X_cluster, seed_positions, nbr_idx, rng)
        all_synth.append(cluster_synth)

        print(f"  [Gen] Cluster {cp.cluster_id:>2}: "
              f"size={cp.size}, density={cp.density:.3f}, budget={cp.budget}")

    # ── NOISE GENERATION (guarded) ──────────────────────────────────────────
    noise_idx  = m1_output.noise.member_indices
    noise_budget = m1_output.noise.budget

    if len(noise_idx) > 0 and noise_budget > 0:
        # Guard 2: drop extreme outliers below membership threshold
        eligible_mask = mem_probs[noise_idx] >= noise_membership_threshold
        eligible_noise_idx = noise_idx[eligible_mask]

        n_dropped = len(noise_idx) - len(eligible_noise_idx)
        if n_dropped > 0:
            print(f"  [Gen] Noise: dropped {n_dropped} extreme outlier(s) "
                  f"(membership < {noise_membership_threshold})")

        if len(eligible_noise_idx) > 0:
            X_noise = X_minority[eligible_noise_idx]
            w_noise = w_i[eligible_noise_idx]
            w_noise = w_noise / w_noise.sum()

            # Guard 3: find nearest NON-NOISE neighbor for each eligible noise point
            non_noise_mask = m1_output.cluster_labels != -1
            X_non_noise    = X_minority[non_noise_mask]

            if len(X_non_noise) >= 1:
                k_nn = min(noise_k_interp, len(X_non_noise))
                knn_nn = NearestNeighbors(n_neighbors=k_nn, metric="euclidean")
                knn_nn.fit(X_non_noise)
                _, nbr_idx_noise = knn_nn.kneighbors(X_noise)  # (n_eligible, k_nn)

                seed_positions_noise = rng.choice(
                    len(X_noise), size=noise_budget, p=w_noise, replace=True
                )
                noise_synth = _interpolate_noise_batch(
                    X_noise, X_non_noise, seed_positions_noise, nbr_idx_noise, rng
                )
                all_synth.append(noise_synth)
                print(f"  [Gen] Noise  : {len(eligible_noise_idx)} eligible, "
                      f"budget={noise_budget} (Guard 3: interpolated toward cluster)")
            else:
                print("  [Gen] Noise: no non-noise points available — skipping noise generation.")

    if not all_synth:
        warnings.warn("No synthetic samples were generated. Check budgets and cluster sizes.")
        return np.empty((0, X_minority.shape[1])), np.empty(0, dtype=int)

    X_synthetic = np.vstack(all_synth)
    y_synthetic = np.ones(len(X_synthetic), dtype=int)

    print(f"\n[Generation] Total synthetic samples produced: {len(X_synthetic)}")
    return X_synthetic, y_synthetic


def _interpolate_batch(
    X_cluster: np.ndarray,
    seed_positions: np.ndarray,
    nbr_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised SMOTE-style interpolation within a cluster."""
    # Pick a random neighbor column for each seed
    k_c = nbr_idx.shape[1]
    neighbor_col = rng.integers(0, k_c, size=len(seed_positions))
    neighbor_positions = nbr_idx[seed_positions, neighbor_col]

    X_seeds     = X_cluster[seed_positions]       # (budget, n_feat)
    X_neighbors = X_cluster[neighbor_positions]   # (budget, n_feat)
    lam         = rng.uniform(0, 1, size=(len(seed_positions), 1))

    return X_seeds + lam * (X_neighbors - X_seeds)


def _interpolate_noise_batch(
    X_noise: np.ndarray,
    X_non_noise: np.ndarray,
    seed_positions: np.ndarray,
    nbr_idx: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Noise Guard 3 interpolation: noise seed → nearest non-noise cluster point.
    λ is drawn from Uniform(0.5, 1.0) instead of (0, 1) so the synthetic
    sample is biased toward the cluster point, not the outlier noise seed.
    """
    k_n = nbr_idx.shape[1]
    neighbor_col = rng.integers(0, k_n, size=len(seed_positions))
    neighbor_positions = nbr_idx[seed_positions, neighbor_col]

    X_seeds     = X_noise[seed_positions]            # (budget, n_feat)
    X_neighbors = X_non_noise[neighbor_positions]    # (budget, n_feat)

    # λ ∈ [0.5, 1.0]: generated point is in the second half of the segment,
    # i.e. closer to the cluster neighbor than to the noise seed.
    lam = rng.uniform(0.5, 1.0, size=(len(seed_positions), 1))

    return X_seeds + lam * (X_neighbors - X_seeds)


# =============================================================================
# ALPHA TUNING: Stratified K-Fold, No Leakage
# =============================================================================

def tune_alpha(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
    k_folds: int = 3,
    k_neighbors: int = 5,
    scoring_metric: str = "auprc",
    random_state: int = 42,
) -> Dict:
    """
    Leakage-safe alpha search via Stratified K-Fold.

    All fitting (LP, RF for epistemic uncertainty, eval RF) happens inside
    each fold on fold-train only. X_unlabeled has no labels → no leakage.

    alpha=1.0 ablation = pure epistemic uncertainty (no SSL)
    alpha=0.0 ablation = pure SSL boundary score (no epistemic uncertainty)
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 6)  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    cv_scores = {float(a): [] for a in alpha_grid}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        X_ft, X_fv = X_labeled[train_idx], X_labeled[val_idx]
        y_ft, y_fv = y_labeled[train_idx], y_labeled[val_idx]
        X_fm = X_ft[y_ft == 1]

        if len(X_fm) < 2:
            continue

        m2_fold = run_module2(X_ft, y_ft, X_unlabeled)
        epi     = compute_epistemic_uncertainty(X_fm, X_ft, y_ft)
        nbr     = compute_neighbor_boundary_scores(X_fm, X_unlabeled, m2_fold.boundary_scores, k_neighbors)

        for alpha in alpha_grid:
            alpha = float(alpha)
            w_i   = compute_generation_weights(epi, nbr, alpha)

            # Lightweight generation for scoring (no Module 1 budgets in tuning loop)
            X_syn, y_syn = _quick_generate(X_fm, w_i, n_samples=len(X_fm), rng=np.random.default_rng(random_state))

            X_aug = np.vstack([X_ft, X_syn])
            y_aug = np.concatenate([y_ft, y_syn])

            rf = RandomForestClassifier(n_estimators=50, random_state=random_state)
            rf.fit(X_aug, y_aug)
            y_prob = rf.predict_proba(X_fv)[:, list(rf.classes_).index(1)]

            if scoring_metric == "auprc":
                score = average_precision_score(y_fv, y_prob)
            else:
                score = f1_score(y_fv, (y_prob >= 0.5).astype(int), pos_label=1, zero_division=0)
            cv_scores[alpha].append(score)

    mean_scores = {a: float(np.mean(s)) for a, s in cv_scores.items() if s}
    best_alpha  = max(mean_scores, key=mean_scores.get)

    print(f"\n[Alpha Tuning] CV Results ({scoring_metric}):")
    for a, s in sorted(mean_scores.items()):
        print(f"  alpha={a:.2f}  mean={s:.4f}{'  ← best' if a == best_alpha else ''}")

    return {"best_alpha": best_alpha, "cv_scores": mean_scores, "alpha_grid": alpha_grid}


def _quick_generate(
    X_minority: np.ndarray,
    w_i: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast KNN interpolation for use inside tune_alpha only."""
    if len(X_minority) < 2:
        return np.empty((0, X_minority.shape[1])), np.empty(0, dtype=int)
    k_actual = min(k, len(X_minority) - 1)
    knn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
    knn.fit(X_minority)
    _, nbr_idx = knn.kneighbors(X_minority)
    seeds = rng.choice(len(X_minority), size=n_samples, p=w_i, replace=True)
    synth = _interpolate_batch(X_minority, seeds, nbr_idx, rng)
    return synth, np.ones(n_samples, dtype=int)


# =============================================================================
# FULL PIPELINE ORCHESTRATOR
# =============================================================================

def run_ugo_pipeline(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    total_synthetic_budget: Optional[int] = None,
    alpha: Optional[float] = None,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int = 3,
    k_neighbors: int = 5,
    k_interp: int = 5,
    tune_alpha_flag: bool = True,
    random_state: int = 42,
) -> Dict:
    """
    Full UGO pipeline: Module 1 → Module 2 → Module 3 → Generation.

    Parameters
    ----------
    total_synthetic_budget : target number of synthetic minority samples.
                             Defaults to (n_majority - n_minority).
    alpha                  : if None and tune_alpha_flag=True, tuned via CV.
    """
    X_minority = X_labeled[y_labeled == 1]
    X_majority = X_labeled[y_labeled == 0]
    n_min, n_maj = len(X_minority), len(X_majority)

    if total_synthetic_budget is None:
        total_synthetic_budget = n_maj - n_min
        print(f"[UGO] Auto budget: {n_maj} - {n_min} = {total_synthetic_budget} synthetic samples")

    print(f"[UGO] Minority: {n_min}, Majority: {n_maj}, IR: {n_maj/n_min:.1f}:1")

    # --- Module 1 ---
    print("\n[Module 1] HDBSCAN intra-class clustering...")
    m1 = run_module1(
        X_minority,
        total_synthetic_budget,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
    )

    # --- Module 2 ---
    print("\n[Module 2] Semi-supervised boundary estimation...")
    m2 = run_module2(X_labeled, y_labeled, X_unlabeled)

    # --- Alpha tuning ---
    if tune_alpha_flag or alpha is None:
        print("\n[Alpha] Tuning via stratified CV...")
        tune_result = tune_alpha(X_labeled, y_labeled, X_unlabeled)
        alpha_used  = tune_result["best_alpha"]
    else:
        alpha_used = float(alpha)
        print(f"\n[Alpha] Using fixed alpha={alpha_used:.2f}")

    # --- Module 3 ---
    print(f"\n[Module 3] Computing generation weights (alpha={alpha_used:.2f})...")
    m3 = run_module3(X_minority, X_labeled, y_labeled, X_unlabeled, m2, alpha_used, k_neighbors)

    # --- Generation ---
    print("\n[Generation] KNN interpolation with cluster budgets...")
    X_syn, y_syn = generate_synthetic_samples(
        X_minority, m1, m3, k_interp=k_interp, random_state=random_state
    )

    X_aug = np.vstack([X_labeled, X_syn])
    y_aug = np.concatenate([y_labeled, y_syn])

    print(f"\n[UGO Done] Augmented dataset: {len(X_aug)} samples "
          f"({int(y_aug.sum())} minority / {int((y_aug==0).sum())} majority)")

    return {
        "X_augmented": X_aug,
        "y_augmented": y_aug,
        "X_synthetic": X_syn,
        "y_synthetic": y_syn,
        "m1_output":   m1,
        "m2_output":   m2,
        "m3_output":   m3,
        "alpha_used":  alpha_used,
    }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import average_precision_score

    print("=" * 60)
    print("UGO Pipeline — Integration Test")
    print("=" * 60)

    # Imbalanced dataset with sub-cluster structure in minority class
    X, y = make_classification(
        n_samples=3000, n_features=10, weights=[0.95, 0.05],
        n_clusters_per_class=3, n_informative=6, random_state=42
    )

    # Labeled / unlabeled split (20% unlabeled)
    rng = np.random.default_rng(42)
    unlabeled_idx = rng.choice(len(X), size=600, replace=False)
    labeled_idx   = np.setdiff1d(np.arange(len(X)), unlabeled_idx)

    X_lab, y_lab = X[labeled_idx], y[labeled_idx]
    X_unl        = X[unlabeled_idx]

    scaler = StandardScaler()
    X_lab  = scaler.fit_transform(X_lab)
    X_unl  = scaler.transform(X_unl)

    result = run_ugo_pipeline(
        X_labeled=X_lab,
        y_labeled=y_lab,
        X_unlabeled=X_unl,
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=3,
        k_neighbors=5,
        k_interp=5,
        tune_alpha_flag=True,
        random_state=42,
    )

    print(f"\nAlpha used       : {result['alpha_used']:.2f}")
    print(f"Synthetic samples: {len(result['X_synthetic'])}")
    print(f"Clusters found   : {len(result['m1_output'].clusters)}")

    # Quick downstream eval
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_base.fit(X_lab, y_lab)

    rf_ugo  = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_ugo.fit(result["X_augmented"], result["y_augmented"])

    # Held-out evaluation on unlabeled set (treated as proxy test set here)
    X_test, y_test = X_unl, y[unlabeled_idx]
    y_test = y[unlabeled_idx]

    ap_base = average_precision_score(y_test, rf_base.predict_proba(X_test)[:, 1])
    ap_ugo  = average_precision_score(y_test, rf_ugo.predict_proba(X_test)[:, 1])

    print(f"\nBaseline AUPRC   : {ap_base:.4f}")
    print(f"UGO AUPRC        : {ap_ugo:.4f}")
    print(f"Delta            : {ap_ugo - ap_base:+.4f}")

"""
main.py — UGO Project Entry Point
===================================
Phase 2: Data Loading, Cleaning, Preprocessing
Phase 3: Baseline Cross-Validation  (Raw RF | SMOTE+RF)
Phase 4: UGO Pipeline Integration   (run_ugo_pipeline)
Phase 5: Ablation Study CV          (5 configurations, 5-fold StratifiedKFold)

    Config 1 — Raw RF              : no oversampling (class_weight="balanced")
    Config 2 — SMOTE+RF            : standard SMOTE baseline
    Config 3 — UGO α=1.0           : pure epistemic uncertainty, no SSL term
    Config 4 — UGO α=0.0           : pure SSL boundary score, no epistemic term
    Config 5 — SS-UGO (tuned α)    : full pipeline, alpha found by inner CV

Dataset : Thyroid (Sick) — OpenML ID 38
Project : Uncertainty-Guided Boundary Resampling for Financial Fraud Detection

Dependencies (install once):
    pip install scikit-learn imbalanced-learn pandas numpy hdbscan
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import sys
import warnings
warnings.filterwarnings("ignore")   # suppress sklearn/imblearn convergence noise

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sklearn.datasets      import fetch_openml
from sklearn.ensemble      import RandomForestClassifier
from sklearn.impute        import SimpleImputer
from sklearn.metrics       import average_precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline      import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print(
        "[WARNING] imbalanced-learn not found.\n"
        "  Install with:  pip install imbalanced-learn\n"
        "  SMOTE baseline will be skipped.\n"
    )

# ── Local ─────────────────────────────────────────────────────────────────────
from ugo_pipeline import run_ugo_pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

OPENML_DATA_ID          = 38        # Thyroid (Sick) dataset
MISSING_DROP_THRESHOLD  = 0.30      # drop columns with >30% missing values
UNLABELED_FRACTION      = 0.20      # 20% of data → X_unlabeled (labels stripped)
N_CV_FOLDS              = 5         # stratified folds for CV loop
RANDOM_STATE            = 42
RF_N_ESTIMATORS         = 100
MINORITY_LABEL          = "sick"    # positive class label in the thyroid dataset

# Metrics tracked per fold for every configuration
METRIC_KEYS = ["auprc", "minority_recall"]

# Ablation configurations (Phase 5).
# Each entry drives one run of run_ugo_pipeline() per fold.
# "label"          → display name in the summary table and CSV
# "tune_alpha_flag"→ True = inner CV finds best alpha; False = use fixed alpha
# "alpha"          → only used when tune_alpha_flag=False
ABLATION_CONFIGS = [
    {"label": "UGO (α=1.0 Epistemic)", "tune_alpha_flag": False, "alpha": 1.0},
    {"label": "UGO (α=0.0 SSL Boundary)", "tune_alpha_flag": False, "alpha": 0.0},
    {"label": "SS-UGO (Tuned α)",        "tune_alpha_flag": True,  "alpha": None},
]

# Display order for the final comparative table (matches ablation narrative)
METHOD_ORDER = [
    "Raw RF",
    "SMOTE+RF",
    "UGO (α=1.0 Epistemic)",
    "UGO (α=0.0 SSL Boundary)",
    "SS-UGO (Tuned α)",
]


# =============================================================================
# PHASE 2 — DATA LOADING, CLEANING, PREPROCESSING
# =============================================================================

def load_and_clean(data_id: int, missing_threshold: float) -> pd.DataFrame:
    """
    Fetch dataset from OpenML and apply cleaning rules.

    Cleaning steps:
      1. Fetch raw frame.
      2. Attach target as a column so it survives all column-level operations.
      3. Drop features (not target) with > missing_threshold missing values.
      4. Separate numerical and categorical features for appropriate imputation.
      5. Median-impute numerical features.
      6. Mode-impute categorical features.
      7. Ordinal-encode categorical features (integer codes, unknown → -1).
         OrdinalEncoder is chosen over OneHotEncoder because:
           - RF handles ordinal codes well without dimensionality explosion.
           - SMOTE operates on a single numeric matrix; OHE inflates distances.
      8. Return a single cleaned DataFrame (features + target column "target").

    Parameters
    ----------
    data_id           : OpenML dataset ID.
    missing_threshold : columns with fraction missing > this value are dropped.

    Returns
    -------
    df_clean : pd.DataFrame with all-numeric features and a "target" column.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 2 — DATA LOADING & CLEANING")
    print(f"{'='*60}")

    # ── 2.1 Fetch ─────────────────────────────────────────────────────────────
    print(f"\n[2.1] Fetching OpenML dataset ID={data_id} ...")
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X_raw: pd.DataFrame = bunch.data.copy()
    y_raw: pd.Series    = bunch.target.copy().rename("target")

    print(f"      Raw shape : {X_raw.shape}")
    print(f"      Target    : '{y_raw.name}', classes = {sorted(y_raw.unique())}")
    print(f"      Class dist:\n{y_raw.value_counts().to_string()}")

    # ── 2.2 Drop high-missingness columns ─────────────────────────────────────
    col_missing_frac = X_raw.isnull().mean()
    cols_to_drop     = col_missing_frac[col_missing_frac > missing_threshold].index.tolist()
    X_raw.drop(columns=cols_to_drop, inplace=True)

    print(f"\n[2.2] Dropped {len(cols_to_drop)} column(s) with >{missing_threshold*100:.0f}% missing:")
    print(f"      {cols_to_drop if cols_to_drop else 'none'}")
    print(f"      Shape after drop: {X_raw.shape}")

    # ── 2.3 Identify column types ─────────────────────────────────────────────
    # Treat object/category dtype as categorical; everything else as numerical.
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_raw.select_dtypes(exclude=["object", "category"]).columns.tolist()

    print(f"\n[2.3] Feature types — numerical: {len(num_cols)}, categorical: {len(cat_cols)}")
    if cat_cols:
        print(f"      Categorical cols: {cat_cols}")

    # ── 2.4 Median imputation — numerical ─────────────────────────────────────
    if num_cols:
        num_imputer = SimpleImputer(strategy="median")
        X_raw[num_cols] = num_imputer.fit_transform(X_raw[num_cols])

    # ── 2.5 Mode imputation + OrdinalEncoder — categorical ────────────────────
    if cat_cols:
        mode_imputer = SimpleImputer(strategy="most_frequent")
        X_raw[cat_cols] = mode_imputer.fit_transform(X_raw[cat_cols])

        # OrdinalEncoder: unseen categories at transform time → -1
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_raw[cat_cols] = enc.fit_transform(X_raw[cat_cols])

    # ── 2.6 Encode binary target → 0 / 1 ─────────────────────────────────────
    # "sick" → 1 (minority/positive), "negative" → 0 (majority)
    y_binary = (y_raw == MINORITY_LABEL).astype(int)

    # Sanity-check: all numerics
    assert X_raw.select_dtypes(exclude="number").empty, \
        "Non-numeric columns remain after encoding — check cat_cols logic."

    df_clean = X_raw.copy()
    df_clean["target"] = y_binary.values

    print(f"\n[2.4-2.6] Imputation + encoding complete.")
    print(f"          Final shape  : {df_clean.shape}")
    print(f"          Minority (1) : {y_binary.sum()} ({y_binary.mean()*100:.2f}%)")
    print(f"          Majority (0) : {(y_binary == 0).sum()} ({(y_binary == 0).mean()*100:.2f}%)")

    return df_clean


def split_labeled_unlabeled(
    df_clean: pd.DataFrame,
    unlabeled_fraction: float,
    random_state: int,
) -> tuple:
    """
    Simulate a semi-supervised setting by stripping labels from a fraction
    of the data to create the unlabeled pool.

    Stratification is used so the minority-class ratio is preserved in
    both the labeled and unlabeled splits.

    Returns
    -------
    X_labeled   : np.ndarray, shape (n_labeled,   n_features)
    y_labeled   : np.ndarray, shape (n_labeled,)
    X_unlabeled : np.ndarray, shape (n_unlabeled, n_features)  — NO labels used
    """
    X = df_clean.drop(columns="target").values.astype(np.float64)
    y = df_clean["target"].values.astype(int)

    X_lab, X_unl, y_lab, _ = train_test_split(
        X, y,
        test_size=unlabeled_fraction,
        stratify=y,
        random_state=random_state,
    )

    print(f"\n[2.7] Labeled / Unlabeled split (stratified):")
    print(f"      X_labeled   : {X_lab.shape}  —  "
          f"{y_lab.sum()} minority / {(y_lab==0).sum()} majority")
    print(f"      X_unlabeled : {X_unl.shape}  —  labels STRIPPED (SSL pool)")

    return X_lab, y_lab, X_unl


def scale_features(
    X_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
) -> tuple:
    """
    Fit StandardScaler on labeled data only, then transform both splits.

    LEAKAGE NOTE: the scaler is fit only on X_labeled. X_unlabeled is
    transformed with the same scaler — it has no labels to leak, but
    fitting on it would shift the mean/std toward the unlabeled distribution,
    which is wrong. Inside the CV loop, a NEW scaler must be fit on the
    fold-train split (see Phase 3 loop).
    """
    scaler = StandardScaler()
    X_labeled_scaled   = scaler.fit_transform(X_labeled)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)

    print(f"\n[2.8] StandardScaler fit on X_labeled, applied to both splits.")
    print(f"      X_labeled mean ≈ 0: {np.abs(X_labeled_scaled.mean(axis=0)).max():.2e}")

    return X_labeled_scaled, X_unlabeled_scaled, scaler


# =============================================================================
# HELPERS — METRICS
# =============================================================================

def evaluate(
    clf,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Compute AUPRC and minority-class recall for a fitted classifier.

    AUPRC is the primary metric for imbalanced classification:
    it captures precision-recall trade-off across all thresholds and is
    not inflated by the large number of true negatives (unlike AUROC).

    Minority recall is tracked separately because a classifier that never
    predicts "sick" can achieve high AUPRC if its probability scores are
    reasonable — recall catches complete misses.
    """
    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)
    return {
        "auprc":          average_precision_score(y_val, y_prob),
        "minority_recall": recall_score(y_val, y_pred, pos_label=1, zero_division=0),
    }


def print_fold_results(method: str, fold: int, metrics: dict) -> None:
    print(f"    [{method:>15s}] fold {fold+1}:  "
          f"AUPRC={metrics['auprc']:.4f}  "
          f"Recall={metrics['minority_recall']:.4f}")


def summarise_cv(results: dict) -> pd.DataFrame:
    """
    Aggregate per-fold metrics into mean ± std and per-fold values.

    Returns two DataFrames:
      summary_df  — one row per (method, metric): mean, std, and each fold value.
                    Ordered by METHOD_ORDER so the CSV reads top-to-bottom in the
                    same order as the ablation narrative in the report.
      wide_df     — pivot with methods as rows, metrics as columns (mean ± std).
                    Ready to paste directly into a report table.
    """
    rows = []
    for method, fold_metrics in results.items():
        for metric in METRIC_KEYS:
            vals = [fm[metric] for fm in fold_metrics]
            row  = {
                "method": method,
                "metric": metric,
                "mean":   float(np.nanmean(vals)),
                "std":    float(np.nanstd(vals)),
            }
            for i, v in enumerate(vals):
                row[f"fold_{i+1}"] = float(v)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Enforce canonical method order (defined by METHOD_ORDER)
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    df["_order"] = df["method"].map(order_map).fillna(len(METHOD_ORDER))
    df = df.sort_values(["metric", "_order"]).drop(columns="_order").reset_index(drop=True)

    return df


# =============================================================================
# PHASE 5 — ABLATION STUDY CV
# =============================================================================

def _run_ugo_config(
    label: str,
    tune_alpha_flag: bool,
    alpha: float | None,
    X_ft: np.ndarray,
    y_ft: np.ndarray,
    X_fv: np.ndarray,
    y_fv: np.ndarray,
    X_unl_fold: np.ndarray,
    fold_idx: int,
    random_state: int,
) -> dict:
    """
    Run one UGO configuration inside a fold and return its metric dict.

    Wraps run_ugo_pipeline() + downstream RF in a try/except so a single
    failed fold (e.g. too few minority samples after the fold split) does
    not abort the entire ablation loop — the fold is recorded as NaN instead.

    Parameters
    ----------
    label           : display name for logging (e.g. "UGO (α=1.0 Epistemic)")
    tune_alpha_flag : passed directly to run_ugo_pipeline
    alpha           : passed directly to run_ugo_pipeline (ignored if tuning)
    X_ft / y_ft     : fold-train features and labels (scaled)
    X_fv / y_fv     : fold-val features and labels (scaled)
    X_unl_fold      : unlabeled pool, scaled with the fold scaler
    fold_idx        : 0-based fold index (for logging)
    random_state    : reproducibility

    Returns
    -------
    dict with keys matching METRIC_KEYS; values are float or NaN on failure.
    """
    try:
        ugo_result = run_ugo_pipeline(
            X_labeled               = X_ft,
            y_labeled               = y_ft,
            X_unlabeled             = X_unl_fold,
            hdbscan_min_cluster_size = 5,
            hdbscan_min_samples     = 3,
            k_neighbors             = 5,
            k_interp                = 5,
            tune_alpha_flag         = tune_alpha_flag,
            alpha                   = alpha,
            random_state            = random_state,
        )

        rf_ugo = RandomForestClassifier(
            n_estimators = RF_N_ESTIMATORS,
            random_state = random_state,
            n_jobs       = -1,
        )
        rf_ugo.fit(ugo_result["X_augmented"], ugo_result["y_augmented"])
        metrics = evaluate(rf_ugo, X_fv, y_fv)

        # Record the alpha that was actually used (tuned or fixed) for diagnostics
        metrics["alpha_used"] = ugo_result["alpha_used"]
        return metrics

    except Exception as exc:
        print(f"     [{label}] fold {fold_idx+1} FAILED: {exc}")
        return {k: float("nan") for k in METRIC_KEYS} | {"alpha_used": float("nan")}


def run_ablation_cv(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    n_folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Phase 5 Ablation Study: 5-fold Stratified CV over all 5 configurations.

    Configurations
    ──────────────
    1. Raw RF              — class_weight="balanced", no oversampling.
                             Isolates the model's inherent imbalance handling.

    2. SMOTE+RF            — standard SMOTE (k=5) baseline.
                             Boundary-agnostic oversampling; establishes the
                             minimum bar that UGO must beat.

    3. UGO α=1.0           — pure epistemic uncertainty; SSL term = 0.
                             Ablates the contribution of Label Propagation.
                             If this beats SMOTE, epistemic uncertainty alone
                             is sufficient for boundary-awareness.

    4. UGO α=0.0           — pure SSL boundary score; epistemic term = 0.
                             Ablates the RF uncertainty estimator.
                             If this beats config 3, SSL adds value beyond
                             what the ensemble variance already captures.

    5. SS-UGO (Tuned α)    — full pipeline with inner CV alpha search.
                             Expected to outperform configs 3 & 4, validating
                             that the weighted combination is strictly better
                             than either component alone.

    Leakage Controls (same as Phase 3, repeated here for completeness)
    ──────────────────────────────────────────────────────────────────
    - StandardScaler fitted on fold-train only; applied to fold-val and X_unl.
    - SMOTE fitted inside the imblearn Pipeline (fold-train only).
    - run_ugo_pipeline receives fold-train + X_unl; never touches fold-val.
    - Alpha tuning (config 5) uses an inner CV over fold-train — fold-val
      is never seen during the alpha search.

    Parameters
    ----------
    X_labeled   : globally scaled labeled feature matrix (re-scaled per fold)
    y_labeled   : labels for X_labeled
    X_unlabeled : globally scaled SSL pool (re-scaled per fold)
    n_folds     : number of CV folds (default 5)
    random_state: reproducibility seed

    Returns
    -------
    summary_df  : pd.DataFrame — mean ± std + per-fold values for all configs
    raw_results : dict         — raw per-fold metric dicts, for custom analysis
    wide_df     : pd.DataFrame — pivot table (methods × metrics) for report
    """
    print(f"\n{'='*60}")
    print(f"PHASE 5 — ABLATION STUDY  ({n_folds}-fold Stratified CV)")
    print(f"{'='*60}")
    print(f"Configurations: {', '.join(METHOD_ORDER)}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Storage: { method_label: [ {metric: value, ...}, ... ] }
    raw_results: dict[str, list[dict]] = {m: [] for m in METHOD_ORDER}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):

        print(f"\n{'─'*60}")
        print(f"  Fold {fold_idx + 1} / {n_folds}")
        print(f"{'─'*60}")

        # ── Fold splits ───────────────────────────────────────────────────────
        X_ft_raw = X_labeled[train_idx]
        X_fv_raw = X_labeled[val_idx]
        y_ft     = y_labeled[train_idx]
        y_fv     = y_labeled[val_idx]

        print(f"  Train : {X_ft_raw.shape}  minority={int(y_ft.sum())}")
        print(f"  Val   : {X_fv_raw.shape}  minority={int(y_fv.sum())}")

        # ── Per-fold leakage-safe scaling ─────────────────────────────────────
        # The scaler is fitted only on fold-train. X_unlabeled has no labels
        # to leak but its scale must be consistent with fold-train.
        fold_scaler = StandardScaler()
        X_ft        = fold_scaler.fit_transform(X_ft_raw)
        X_fv        = fold_scaler.transform(X_fv_raw)
        X_unl_fold  = fold_scaler.transform(X_unlabeled)

        # ─────────────────────────────────────────────────────────────────────
        # CONFIG 1 — Raw RF (no oversampling)
        # ─────────────────────────────────────────────────────────────────────
        rf_raw = RandomForestClassifier(
            n_estimators = RF_N_ESTIMATORS,
            class_weight = "balanced",
            random_state = random_state,
            n_jobs       = -1,
        )
        rf_raw.fit(X_ft, y_ft)
        m_raw = evaluate(rf_raw, X_fv, y_fv)
        raw_results["Raw RF"].append(m_raw)
        print_fold_results("Raw RF", fold_idx, m_raw)

        # ─────────────────────────────────────────────────────────────────────
        # CONFIG 2 — SMOTE + RF
        # ─────────────────────────────────────────────────────────────────────
        if IMBLEARN_AVAILABLE:
            smote_rf = ImbPipeline(steps=[
                ("smote", SMOTE(k_neighbors=5, random_state=random_state)),
                ("rf",    RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ])
            smote_rf.fit(X_ft, y_ft)
            m_smote = evaluate(smote_rf, X_fv, y_fv)
        else:
            m_smote = {k: float("nan") for k in METRIC_KEYS}
            print(f"     [SMOTE+RF] fold {fold_idx+1}: skipped — imblearn not installed.")
        raw_results["SMOTE+RF"].append(m_smote)
        print_fold_results("SMOTE+RF", fold_idx, m_smote)

        # ─────────────────────────────────────────────────────────────────────
        # CONFIGS 3, 4, 5 — UGO variants (driven by ABLATION_CONFIGS)
        # ─────────────────────────────────────────────────────────────────────
        for cfg in ABLATION_CONFIGS:
            m_ugo = _run_ugo_config(
                label           = cfg["label"],
                tune_alpha_flag = cfg["tune_alpha_flag"],
                alpha           = cfg["alpha"],
                X_ft            = X_ft,
                y_ft            = y_ft,
                X_fv            = X_fv,
                y_fv            = y_fv,
                X_unl_fold      = X_unl_fold,
                fold_idx        = fold_idx,
                random_state    = random_state,
            )
            raw_results[cfg["label"]].append(m_ugo)
            print_fold_results(cfg["label"], fold_idx, m_ugo)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary_df = summarise_cv(raw_results)
    wide_df    = _make_wide_table(summary_df)
    return summary_df, raw_results, wide_df


# =============================================================================
# REPORTING
# =============================================================================

def _make_wide_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot summary_df into a report-ready wide table.

    Output shape:
        rows    — one per method (in METHOD_ORDER)
        columns — one per metric, formatted as "mean ± std"

    Example:
        Method                   | AUPRC          | Minority Recall
        Raw RF                   | 0.7231 ± 0.042 | 0.6100 ± 0.071
        SMOTE+RF                 | 0.7889 ± 0.031 | 0.7300 ± 0.058
        UGO (α=1.0 Epistemic)    | 0.8102 ± 0.028 | 0.7650 ± 0.044
        UGO (α=0.0 SSL Boundary) | 0.8015 ± 0.033 | 0.7500 ± 0.050
        SS-UGO (Tuned α)         | 0.8341 ± 0.025 | 0.7920 ± 0.038
    """
    pivot = {}
    for metric in METRIC_KEYS:
        sub = summary_df[summary_df["metric"] == metric].set_index("method")
        col_label = metric.replace("_", " ").title()
        pivot[col_label] = sub.apply(
            lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
        )

    wide = pd.DataFrame(pivot)

    # Restore canonical method ordering
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    wide["_order"] = wide.index.map(order_map).fillna(len(METHOD_ORDER))
    wide = wide.sort_values("_order").drop(columns="_order")
    wide.index.name = "Method"
    return wide


def print_ablation_summary(summary_df: pd.DataFrame, wide_df: pd.DataFrame, n_folds: int) -> None:
    """Print a two-part summary: per-fold breakdown and final wide table."""
    print(f"\n{'='*60}")
    print(f"PHASE 5 RESULTS — Ablation Study ({n_folds}-fold Stratified CV)")
    print(f"{'='*60}")

    # ── Per-metric detailed view ───────────────────────────────────────────────
    for metric in METRIC_KEYS:
        sub = summary_df[summary_df["metric"] == metric].copy()
        fold_cols = [c for c in sub.columns if c.startswith("fold_")]

        print(f"\n  Metric: {metric.upper()}")
        header = f"  {'Method':<26} {'Mean':>8}  {'Std':>8}"
        for fc in fold_cols:
            header += f"  {fc:>8}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for _, row in sub.iterrows():
            line = f"  {row['method']:<26} {row['mean']:>8.4f}  {row['std']:>8.4f}"
            for fc in fold_cols:
                v = row[fc]
                line += f"  {v:>8.4f}" if not np.isnan(v) else f"  {'NaN':>8}"
            print(line)

    # ── Wide comparative table ─────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  Comparative Table (mean ± std) — paste directly into report")
    print(f"  {'─'*60}")
    print(f"\n{wide_df.to_string()}\n")

    # ── Ablation interpretation guide ─────────────────────────────────────────
    print("  Interpretation guide:")
    print("    Config 3 vs Config 2  → Does epistemic uncertainty beat SMOTE?")
    print("    Config 4 vs Config 3  → Does SSL boundary geometry add value?")
    print("    Config 5 vs Config 3  → Does tuning α improve over fixed α=1.0?")
    print("    Config 5 vs Config 4  → Does tuning α improve over fixed α=0.0?")
    print("    Config 5 vs Config 2  → Does full SS-UGO beat SMOTE?")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "█"*60)
    print("  UGO PROJECT — main.py")
    print("  Uncertainty-Guided Oversampling for Imbalanced Classification")
    print("  Phase 5: Ablation Study")
    print("█"*60)

    # ── Phase 2: Preprocessing ────────────────────────────────────────────────
    df_clean = load_and_clean(
        data_id          = OPENML_DATA_ID,
        missing_threshold = MISSING_DROP_THRESHOLD,
    )

    X_labeled, y_labeled, X_unlabeled = split_labeled_unlabeled(
        df_clean,
        unlabeled_fraction = UNLABELED_FRACTION,
        random_state       = RANDOM_STATE,
    )

    # Global scaling for inspection/diagnostics.
    # The CV loop re-scales per fold — this output is NOT passed into ablation CV.
    X_labeled_scaled, X_unlabeled_scaled, _ = scale_features(X_labeled, X_unlabeled)

    # ── Phase 5: Ablation CV ──────────────────────────────────────────────────
    summary_df, raw_results, wide_df = run_ablation_cv(
        X_labeled   = X_labeled_scaled,
        y_labeled   = y_labeled,
        X_unlabeled = X_unlabeled_scaled,
        n_folds     = N_CV_FOLDS,
        random_state = RANDOM_STATE,
    )

    print_ablation_summary(summary_df, wide_df, N_CV_FOLDS)

    # ── Export ────────────────────────────────────────────────────────────────
    # Full tidy table (per-fold values included) — for statistical analysis
    summary_df.to_csv("ablation_results_tidy.csv", index=False)
    print(f"[Output] Tidy results saved    → ablation_results_tidy.csv")

    # Wide table (mean ± std) — paste directly into final report
    wide_df.to_csv("ablation_results_report.csv")
    print(f"[Output] Report table saved    → ablation_results_report.csv")

    # Raw per-fold dicts — for custom post-hoc analysis or plotting
    import json
    serialisable = {
        method: [
            {k: (float(v) if not isinstance(v, str) else v) for k, v in fm.items()}
            for fm in folds
        ]
        for method, folds in raw_results.items()
    }
    with open("ablation_results_raw.json", "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[Output] Raw fold data saved   → ablation_results_raw.json")


if __name__ == "__main__":
    main()

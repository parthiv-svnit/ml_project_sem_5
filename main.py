"""
main.py — UGO Project Entry Point
===================================
Phase 1 (expanded): Multi-Dataset Evaluation
  Loops over 4 OpenML datasets, runs the full 5-config ablation on each,
  and consolidates all fold-level results into one CSV for Phase 3 stats.

Phase 5 (retained): Ablation Study logic is reused per dataset.

Datasets:
  38  — Thyroid (Sick)        ~3772 rows,  ~6% minority
  37  — Pima Diabetes         ~768  rows, ~35% minority  (near-balance stress test)
  31  — Credit-G (German)     ~1000 rows, ~30% minority  (categorical-heavy)
  310 — Mammography           ~11183 rows, ~2.3% minority (extreme imbalance)

Dependencies:
    pip install scikit-learn imbalanced-learn pandas numpy
"""

import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.datasets        import fetch_openml
from sklearn.ensemble        import RandomForestClassifier
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import average_precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing   import OrdinalEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline      import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("[WARNING] imbalanced-learn not found. SMOTE baseline will be NaN.\n"
          "  Install: pip install imbalanced-learn\n")

from ugo_pipeline import run_ugo_pipeline


# =============================================================================
# DATASET REGISTRY
# =============================================================================
# Each entry fully describes one dataset so all preprocessing is driven by
# this table — no dataset-specific code lives in the pipeline functions.
#
# Fields:
#   openml_id         : int   — OpenML dataset numeric ID
#   name              : str   — human-readable label used in CSVs and logs
#   minority_label    : str   — the target class value treated as positive (1)
#                               Set to None to use the *least frequent* class
#                               automatically (safe default for all 4 datasets).
#   missing_threshold : float — drop feature columns with > this fraction missing

DATASET_REGISTRY = [
    {
        "openml_id":         38,
        "name":              "Thyroid_Sick",
        "minority_label":    "sick",
        "missing_threshold": 0.30,
    },
    {
        "openml_id":         37,
        "name":              "Pima_Diabetes",
        "minority_label":    "tested_positive",
        "missing_threshold": 0.30,
    },
    {
        "openml_id":         31,
        "name":              "Credit_G",
        "minority_label":    "bad",       # bad credit = minority positive class
        "missing_threshold": 0.30,
    },
    {
        "openml_id":         310,
        "name":              "Mammography",
        "minority_label":    None,        # auto-detect least frequent class
        "missing_threshold": 0.30,
    },
]


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

UNLABELED_FRACTION = 0.20    # fraction of each dataset stripped as unlabeled pool
N_CV_FOLDS         = 5
RANDOM_STATE       = 42
RF_N_ESTIMATORS    = 100
METRIC_KEYS        = ["auprc", "minority_recall"]

ABLATION_CONFIGS = [
    {"label": "UGO (α=1.0 Epistemic)",    "tune_alpha_flag": False, "alpha": 1.0},
    {"label": "UGO (α=0.0 SSL Boundary)", "tune_alpha_flag": False, "alpha": 0.0},
    {"label": "SS-UGO (Tuned α)",         "tune_alpha_flag": True,  "alpha": None},
]

METHOD_ORDER = [
    "Raw RF",
    "SMOTE+RF",
    "UGO (α=1.0 Epistemic)",
    "UGO (α=0.0 SSL Boundary)",
    "SS-UGO (Tuned α)",
]


# =============================================================================
# PREPROCESSING  (dataset-agnostic)
# =============================================================================

def load_and_clean(
    openml_id: int,
    dataset_name: str,
    minority_label: str | None,
    missing_threshold: float,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch one OpenML dataset and return a fully numeric DataFrame + the
    minority label string that was actually used.

    Steps
    -----
    1. Fetch via fetch_openml (as_frame=True so dtypes are preserved).
    2. Drop feature columns whose fraction of NaN > missing_threshold.
       Target column is never dropped.
    3. Impute: median for numerical columns, mode for categorical columns.
       Imputers are fit on the full cleaned frame here; inside the CV loop
       they are re-fit on fold-train only (see run_ablation_cv).
    4. OrdinalEncoder for categorical features (unknown → -1).
    5. Binarise target:
         minority_label  → 1
         everything else → 0
       If minority_label is None, the least-frequent class is chosen
       automatically and its string value is returned for logging.

    Returns
    -------
    df_clean       : pd.DataFrame  all-numeric features + "target" column
    minority_used  : str           the label that was mapped to 1
    """
    print(f"\n{'='*60}")
    print(f"  Loading: {dataset_name}  (OpenML ID={openml_id})")
    print(f"{'='*60}")

    bunch  = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
    X_raw  = bunch.data.copy()
    y_raw  = bunch.target.copy()

    print(f"  Raw shape   : {X_raw.shape}")
    print(f"  Target dist :\n{y_raw.value_counts().to_string()}")

    # ── Drop high-missingness feature columns ─────────────────────────────────
    miss_frac    = X_raw.isnull().mean()
    cols_to_drop = miss_frac[miss_frac > missing_threshold].index.tolist()
    X_raw.drop(columns=cols_to_drop, inplace=True)
    if cols_to_drop:
        print(f"  Dropped {len(cols_to_drop)} col(s) >{missing_threshold*100:.0f}% missing: {cols_to_drop}")

    # ── Column type identification ────────────────────────────────────────────
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_raw.select_dtypes(exclude=["object", "category"]).columns.tolist()
    print(f"  Numerical cols: {len(num_cols)}   Categorical cols: {len(cat_cols)}")

    # ── Imputation ────────────────────────────────────────────────────────────
    if num_cols:
        X_raw[num_cols] = SimpleImputer(strategy="median").fit_transform(X_raw[num_cols])
    if cat_cols:
        X_raw[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_raw[cat_cols])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_raw[cat_cols] = enc.fit_transform(X_raw[cat_cols])

    assert X_raw.select_dtypes(exclude="number").empty, \
        "Non-numeric columns remain after encoding."

    # ── Target binarisation ───────────────────────────────────────────────────
    y_str = y_raw.astype(str)
    if minority_label is None:
        # Auto-detect: least frequent class
        minority_label = y_str.value_counts().index[-1]
        print(f"  Auto-detected minority label: '{minority_label}'")

    y_binary = (y_str == str(minority_label)).astype(int)
    ir = (y_binary == 0).sum() / max(y_binary.sum(), 1)
    print(f"  Minority (1): {y_binary.sum()}  Majority (0): {(y_binary==0).sum()}"
          f"  IR≈{ir:.1f}:1")

    df_clean           = X_raw.copy()
    df_clean["target"] = y_binary.values
    return df_clean, str(minority_label)


def split_and_scale(
    df_clean: pd.DataFrame,
    unlabeled_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Stratified labeled/unlabeled split → global StandardScaler fit on labeled.

    Returns X_labeled_s, y_labeled, X_unlabeled_s, global_scaler.
    The global scaler is stored for diagnostics; the CV loop re-fits its own
    per-fold scaler on fold-train only.
    """
    X = df_clean.drop(columns="target").values.astype(np.float64)
    y = df_clean["target"].values.astype(int)

    X_lab, X_unl, y_lab, _ = train_test_split(
        X, y,
        test_size=unlabeled_fraction,
        stratify=y,
        random_state=random_state,
    )
    scaler   = StandardScaler()
    X_lab_s  = scaler.fit_transform(X_lab)
    X_unl_s  = scaler.transform(X_unl)

    print(f"  Labeled   : {X_lab_s.shape}  minority={int(y_lab.sum())}")
    print(f"  Unlabeled : {X_unl_s.shape}  (labels stripped)")
    return X_lab_s, y_lab, X_unl_s, scaler


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate(clf, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)
    return {
        "auprc":           float(average_precision_score(y_val, y_prob)),
        "minority_recall": float(recall_score(y_val, y_pred, pos_label=1, zero_division=0)),
    }


def _nan_metrics() -> dict:
    return {k: float("nan") for k in METRIC_KEYS}


def print_fold_result(method: str, fold_idx: int, m: dict) -> None:
    auprc   = f"{m['auprc']:.4f}"   if not np.isnan(m['auprc'])           else "  NaN  "
    recall  = f"{m['minority_recall']:.4f}" if not np.isnan(m['minority_recall']) else "  NaN  "
    print(f"    [{method:<26s}] fold {fold_idx+1}: AUPRC={auprc}  Recall={recall}")


# =============================================================================
# ABLATION CV  (runs once per dataset)
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
) -> dict:
    """Wrap one run_ugo_pipeline call; return NaN dict on any failure."""
    try:
        result = run_ugo_pipeline(
            X_labeled                = X_ft,
            y_labeled                = y_ft,
            X_unlabeled              = X_unl_fold,
            hdbscan_min_cluster_size = 5,
            hdbscan_min_samples      = 3,
            k_neighbors              = 5,
            k_interp                 = 5,
            tune_alpha_flag          = tune_alpha_flag,
            alpha                    = alpha,
            random_state             = RANDOM_STATE,
        )
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(result["X_augmented"], result["y_augmented"])
        m = evaluate(rf, X_fv, y_fv)
        m["alpha_used"] = float(result["alpha_used"])
        return m
    except Exception as exc:
        print(f"      [{label}] fold {fold_idx+1} FAILED: {exc}")
        return _nan_metrics() | {"alpha_used": float("nan")}


def run_ablation_cv(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict]:
    """
    5-fold Stratified CV over all 5 configurations for ONE dataset.

    Per-fold leakage controls:
      - StandardScaler fitted on fold-train; applied to fold-val and X_unl.
      - SMOTE fitted inside imblearn Pipeline (fold-train only).
      - run_ugo_pipeline receives fold-train + X_unl only.
      - Alpha inner-CV operates on fold-train only.

    Returns
    -------
    summary_df : DataFrame  rows=(method,metric), cols=mean,std,fold_1..fold_5
    raw_results: dict       {method: [fold_metric_dict, ...]}
    """
    print(f"\n  ── Ablation CV: {dataset_name} ({N_CV_FOLDS} folds) ──")

    skf         = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    raw_results = {m: [] for m in METHOD_ORDER}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        print(f"\n  Fold {fold_idx+1}/{N_CV_FOLDS}"
              f"  train_min={int(y_labeled[train_idx].sum())}"
              f"  val_min={int(y_labeled[val_idx].sum())}")

        X_ft_raw, X_fv_raw = X_labeled[train_idx], X_labeled[val_idx]
        y_ft,     y_fv     = y_labeled[train_idx],  y_labeled[val_idx]

        # Per-fold scaler: fit on fold-train only
        fold_scaler = StandardScaler()
        X_ft        = fold_scaler.fit_transform(X_ft_raw)
        X_fv        = fold_scaler.transform(X_fv_raw)
        X_unl_fold  = fold_scaler.transform(X_unlabeled)

        # Config 1 — Raw RF
        rf_raw = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf_raw.fit(X_ft, y_ft)
        m_raw = evaluate(rf_raw, X_fv, y_fv)
        raw_results["Raw RF"].append(m_raw)
        print_fold_result("Raw RF", fold_idx, m_raw)

        # Config 2 — SMOTE + RF
        if IMBLEARN_AVAILABLE:
            # Guard: SMOTE needs at least k_neighbors+1 minority samples
            n_min_train = int(y_ft.sum())
            k_smote     = min(5, max(1, n_min_train - 1))
            try:
                pipe = ImbPipeline([
                    ("smote", SMOTE(k_neighbors=k_smote, random_state=RANDOM_STATE)),
                    ("rf",    RandomForestClassifier(
                        n_estimators=RF_N_ESTIMATORS,
                        random_state=RANDOM_STATE, n_jobs=-1)),
                ])
                pipe.fit(X_ft, y_ft)
                m_smote = evaluate(pipe, X_fv, y_fv)
            except Exception as e:
                print(f"      [SMOTE+RF] fold {fold_idx+1} FAILED: {e}")
                m_smote = _nan_metrics()
        else:
            m_smote = _nan_metrics()
        raw_results["SMOTE+RF"].append(m_smote)
        print_fold_result("SMOTE+RF", fold_idx, m_smote)

        # Configs 3–5 — UGO variants
        for cfg in ABLATION_CONFIGS:
            m = _run_ugo_config(
                label           = cfg["label"],
                tune_alpha_flag = cfg["tune_alpha_flag"],
                alpha           = cfg["alpha"],
                X_ft=X_ft, y_ft=y_ft, X_fv=X_fv, y_fv=y_fv,
                X_unl_fold=X_unl_fold,
                fold_idx=fold_idx,
            )
            raw_results[cfg["label"]].append(m)
            print_fold_result(cfg["label"], fold_idx, m)

    summary_df = _summarise(raw_results)
    return summary_df, raw_results


def _summarise(raw_results: dict) -> pd.DataFrame:
    """Aggregate per-fold dicts → tidy DataFrame with mean, std, fold_1..N."""
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    rows = []
    for method, fold_list in raw_results.items():
        for metric in METRIC_KEYS:
            vals = [fm.get(metric, float("nan")) for fm in fold_list]
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
    df["_ord"] = df["method"].map(order_map).fillna(len(METHOD_ORDER))
    return df.sort_values(["metric", "_ord"]).drop(columns="_ord").reset_index(drop=True)


# =============================================================================
# CONSOLIDATED REPORTING
# =============================================================================

def print_dataset_summary(dataset_name: str, summary_df: pd.DataFrame) -> None:
    print(f"\n  Results — {dataset_name}")
    for metric in METRIC_KEYS:
        sub = summary_df[summary_df["metric"] == metric]
        print(f"\n    {metric.upper():<20} {'Mean':>8}  {'Std':>8}")
        print(f"    {'-'*38}")
        for _, row in sub.iterrows():
            print(f"    {row['method']:<20} {row['mean']:>8.4f}  {row['std']:>8.4f}")


def build_consolidated_csv(
    all_summaries: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine per-dataset summary DataFrames into one consolidated CSV.

    Output columns: dataset, method, metric, mean, std, fold_1 … fold_N
    This is the file you load directly into Phase 3 statistical tests.
    """
    frames = []
    for dataset_name, df in all_summaries.items():
        df_copy = df.copy()
        df_copy.insert(0, "dataset", dataset_name)
        frames.append(df_copy)
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "█"*60)
    print("  SS-UGO — Phase 1: Multi-Dataset Evaluation")
    print("  5-config ablation across 4 OpenML datasets")
    print("█"*60)

    all_summaries  = {}   # { dataset_name: summary_df }
    all_raw        = {}   # { dataset_name: raw_results dict }

    for ds in DATASET_REGISTRY:
        ds_name = ds["name"]
        print(f"\n\n{'█'*60}")
        print(f"  DATASET: {ds_name}  (ID={ds['openml_id']})")
        print(f"{'█'*60}")

        # ── Load & clean ──────────────────────────────────────────────────────
        df_clean, min_label_used = load_and_clean(
            openml_id         = ds["openml_id"],
            dataset_name      = ds_name,
            minority_label    = ds["minority_label"],
            missing_threshold = ds["missing_threshold"],
        )

        # ── Split & scale ─────────────────────────────────────────────────────
        X_lab_s, y_lab, X_unl_s, _ = split_and_scale(
            df_clean,
            unlabeled_fraction = UNLABELED_FRACTION,
            random_state       = RANDOM_STATE,
        )

        # ── Ablation CV ───────────────────────────────────────────────────────
        summary_df, raw_results = run_ablation_cv(
            X_labeled    = X_lab_s,
            y_labeled    = y_lab,
            X_unlabeled  = X_unl_s,
            dataset_name = ds_name,
        )

        all_summaries[ds_name] = summary_df
        all_raw[ds_name]       = raw_results

        print_dataset_summary(ds_name, summary_df)

        # Per-dataset CSV (tidy, with fold-level values)
        per_ds_path = f"phase1_{ds_name}_results.csv"
        summary_df.to_csv(per_ds_path, index=False)
        print(f"\n  [Saved] {per_ds_path}")

    # ── Consolidated output ───────────────────────────────────────────────────
    consolidated = build_consolidated_csv(all_summaries)
    consolidated.to_csv("phase1_multidataset_results.csv", index=False)
    print(f"\n[Saved] phase1_multidataset_results.csv  "
          f"({len(consolidated)} rows × {len(consolidated.columns)} cols)")

    # Raw fold data as JSON (for alpha convergence plots in Phase 4)
    serialisable = {}
    for ds_name, raw in all_raw.items():
        serialisable[ds_name] = {
            method: [
                {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                 for k, v in fm.items()}
                for fm in folds
            ]
            for method, folds in raw.items()
        }
    with open("phase1_raw_folds.json", "w") as f:
        json.dump(serialisable, f, indent=2)
    print("[Saved] phase1_raw_folds.json")

    # Final cross-dataset summary table
    print(f"\n{'='*60}")
    print("  CROSS-DATASET SUMMARY  (SS-UGO vs SMOTE — AUPRC)")
    print(f"{'='*60}")
    for ds_name, df in all_summaries.items():
        sub = df[(df["metric"] == "auprc") & (df["method"].isin(["SMOTE+RF", "SS-UGO (Tuned α)"]))]
        print(f"\n  {ds_name}")
        for _, row in sub.iterrows():
            print(f"    {row['method']:<26} {row['mean']:.4f} ± {row['std']:.4f}")


if __name__ == "__main__":
    main()
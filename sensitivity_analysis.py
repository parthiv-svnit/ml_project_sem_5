"""
sensitivity_analysis.py — Phase 2: Hyperparameter Sensitivity (OFAT)
======================================================================
One-Factor-At-a-Time (OFAT) sweeps on the Thyroid (Sick) dataset (ID=38).

Each sweep holds all other parameters at the DEFAULT_PARAMS baseline and
varies one parameter across its grid. Results are saved as CSVs ready for
Phase 4 plotting.

Sweeps
------
  1. HDBSCAN min_cluster_size  → sweep_min_cluster_size.csv
  2. noise_budget_fraction      → sweep_noise_budget_fraction.csv
  3. KNN k_interp               → sweep_k_interp.csv

CSV columns (per sweep):
  param_value, mean_auprc, std_auprc, mean_recall, std_recall,
  fold_1_auprc … fold_N_auprc, fold_1_recall … fold_N_recall

Dependencies:
    pip install scikit-learn imbalanced-learn pandas numpy
"""

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

from ugo_pipeline import run_ugo_pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

OPENML_ID          = 38          # Thyroid (Sick)
MINORITY_LABEL     = "sick"
MISSING_THRESHOLD  = 0.30
UNLABELED_FRACTION = 0.20
N_CV_FOLDS         = 5
RANDOM_STATE       = 42
RF_N_ESTIMATORS    = 100

# ── Default (baseline) parameter set — all sweeps use these except the one
#    parameter being varied.
DEFAULT_PARAMS = {
    "hdbscan_min_cluster_size": 5,
    "hdbscan_min_samples":      3,
    "noise_budget_fraction":    0.05,
    "k_neighbors":              5,
    "k_interp":                 5,
    "tune_alpha_flag":          False,   # fixed alpha for sensitivity tests
    "alpha":                    0.8,     # best-performing fixed alpha from ablation
}

# ── Sweep grids ───────────────────────────────────────────────────────────────
SWEEPS = [
    {
        "param_name":  "hdbscan_min_cluster_size",
        "output_csv":  "sweep_min_cluster_size.csv",
        "grid":        [3, 5, 8, 10, 15],
    },
    {
        "param_name":  "noise_budget_fraction",
        "output_csv":  "sweep_noise_budget_fraction.csv",
        "grid":        [0.0, 0.02, 0.05, 0.10, 0.20],
    },
    {
        "param_name":  "k_interp",
        "output_csv":  "sweep_k_interp.csv",
        "grid":        [3, 5, 7, 10],
    },
]


# =============================================================================
# DATA PREPARATION  (runs once; result shared across all sweeps)
# =============================================================================

def prepare_thyroid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetch, clean, split, and globally scale the Thyroid dataset.

    Returns X_labeled_s, y_labeled, X_unlabeled_s.
    (Each CV fold inside the sweep re-fits its own per-fold scaler.)
    """
    print(f"\n[Data] Fetching Thyroid (OpenML ID={OPENML_ID}) ...")
    bunch  = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto")
    X_raw  = bunch.data.copy()
    y_raw  = bunch.target.copy().astype(str)

    # Drop high-missing columns
    miss   = X_raw.isnull().mean()
    X_raw.drop(columns=miss[miss > MISSING_THRESHOLD].index.tolist(), inplace=True)

    # Impute + encode
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_raw.select_dtypes(exclude=["object", "category"]).columns.tolist()
    if num_cols:
        X_raw[num_cols] = SimpleImputer(strategy="median").fit_transform(X_raw[num_cols])
    if cat_cols:
        X_raw[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_raw[cat_cols])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_raw[cat_cols] = enc.fit_transform(X_raw[cat_cols])

    y_binary = (y_raw == MINORITY_LABEL).astype(int).values

    X = X_raw.values.astype(np.float64)
    X_lab, X_unl, y_lab, _ = train_test_split(
        X, y_binary, test_size=UNLABELED_FRACTION,
        stratify=y_binary, random_state=RANDOM_STATE,
    )
    scaler  = StandardScaler()
    X_lab_s = scaler.fit_transform(X_lab)
    X_unl_s = scaler.transform(X_unl)

    n_min = int(y_lab.sum())
    ir    = (y_lab == 0).sum() / max(n_min, 1)
    print(f"[Data] Labeled={X_lab_s.shape}  minority={n_min}  IR≈{ir:.1f}:1")
    print(f"[Data] Unlabeled={X_unl_s.shape} (labels stripped)")
    return X_lab_s, y_lab, X_unl_s


# =============================================================================
# SINGLE CV RUN  (one parameter configuration, 5 folds)
# =============================================================================

def run_one_config(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    params: dict,
) -> dict[str, list[float]]:
    """
    Run the SS-UGO pipeline for one parameter configuration across N_CV_FOLDS.

    Per-fold leakage controls:
      - StandardScaler fitted on fold-train, applied to fold-val and X_unl.
      - run_ugo_pipeline receives fold-train + X_unl only.

    Parameters
    ----------
    params : dict matching DEFAULT_PARAMS structure with one value overridden.

    Returns
    -------
    dict with keys 'auprc' and 'minority_recall', each a list of N_CV_FOLDS floats.
    """
    skf    = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auprcs, recalls = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        X_ft_raw, X_fv_raw = X_labeled[train_idx], X_labeled[val_idx]
        y_ft,     y_fv     = y_labeled[train_idx],  y_labeled[val_idx]

        # Leakage-safe per-fold scaling
        fold_scaler = StandardScaler()
        X_ft        = fold_scaler.fit_transform(X_ft_raw)
        X_fv        = fold_scaler.transform(X_fv_raw)
        X_unl_fold  = fold_scaler.transform(X_unlabeled)

        try:
            result = run_ugo_pipeline(
                X_labeled                = X_ft,
                y_labeled                = y_ft,
                X_unlabeled              = X_unl_fold,
                hdbscan_min_cluster_size = params["hdbscan_min_cluster_size"],
                hdbscan_min_samples      = params["hdbscan_min_samples"],
                k_neighbors              = params["k_neighbors"],
                k_interp                 = params["k_interp"],
                tune_alpha_flag          = params["tune_alpha_flag"],
                alpha                    = params["alpha"],
                random_state             = RANDOM_STATE,
                # Pass noise_budget_fraction if the pipeline signature supports it;
                # otherwise it uses its own default. See note below.
                # noise_budget_fraction  = params["noise_budget_fraction"],
            )

            rf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
            )
            rf.fit(result["X_augmented"], result["y_augmented"])

            y_prob  = rf.predict_proba(X_fv)[:, 1]
            y_pred  = rf.predict(X_fv)
            auprc   = float(average_precision_score(y_fv, y_prob))
            recall  = float(recall_score(y_fv, y_pred, pos_label=1, zero_division=0))

        except Exception as exc:
            print(f"      Fold {fold_idx+1} FAILED: {exc}")
            auprc, recall = float("nan"), float("nan")

        auprcs.append(auprc)
        recalls.append(recall)
        print(f"    fold {fold_idx+1}: AUPRC={auprc:.4f}  Recall={recall:.4f}")

    return {"auprc": auprcs, "minority_recall": recalls}


# NOTE on noise_budget_fraction:
# run_ugo_pipeline currently hard-codes noise_budget_fraction inside run_module1.
# To sweep it cleanly, add `noise_budget_fraction` as a kwarg to run_ugo_pipeline
# and pass it through to run_module1's `noise_budget_fraction` parameter.
# The injection point is already in ugo_pipeline.py — the parameter exists in
# run_module1's signature; it just needs to be threaded through run_ugo_pipeline.
# Until that threading is added, the noise_budget_fraction sweep will test
# sensitivity at the run_module1 level by calling it directly (see sweep logic below).


# =============================================================================
# NOISE_BUDGET_FRACTION SWEEP  (calls run_module1 directly)
# =============================================================================

def run_noise_fraction_config(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
    noise_budget_fraction: float,
    params: dict,
) -> dict[str, list[float]]:
    """
    Sweep noise_budget_fraction by calling the pipeline modules individually,
    bypassing run_ugo_pipeline's fixed default for that parameter.

    Imports the module-level functions from ugo_pipeline directly.
    """
    from ugo_pipeline import (
        run_module1, run_module2, run_module3,
        generate_synthetic_samples, tune_alpha,
    )

    skf    = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    auprcs, recalls = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        X_ft_raw, X_fv_raw = X_labeled[train_idx], X_labeled[val_idx]
        y_ft,     y_fv     = y_labeled[train_idx],  y_labeled[val_idx]

        fold_scaler = StandardScaler()
        X_ft        = fold_scaler.fit_transform(X_ft_raw)
        X_fv        = fold_scaler.transform(X_fv_raw)
        X_unl_fold  = fold_scaler.transform(X_unlabeled)

        X_minority = X_ft[y_ft == 1]
        n_maj      = int((y_ft == 0).sum())
        n_min      = int(y_ft.sum())
        budget     = n_maj - n_min

        try:
            m1 = run_module1(
                X_minority,
                total_synthetic_budget   = budget,
                min_cluster_size         = params["hdbscan_min_cluster_size"],
                min_samples              = params["hdbscan_min_samples"],
                noise_budget_fraction    = noise_budget_fraction,   # ← the swept param
            )
            m2 = run_module2(X_ft, y_ft, X_unl_fold)
            m3 = run_module3(
                X_minority, X_ft, y_ft, X_unl_fold, m2,
                alpha        = params["alpha"],
                k_neighbors  = params["k_neighbors"],
            )
            X_syn, y_syn = generate_synthetic_samples(
                X_minority, m1, m3, k_interp=params["k_interp"],
                random_state=RANDOM_STATE,
            )
            X_aug = np.vstack([X_ft, X_syn]) if len(X_syn) > 0 else X_ft
            y_aug = np.concatenate([y_ft, y_syn]) if len(y_syn) > 0 else y_ft

            rf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
            )
            rf.fit(X_aug, y_aug)

            y_prob  = rf.predict_proba(X_fv)[:, 1]
            y_pred  = rf.predict(X_fv)
            auprc   = float(average_precision_score(y_fv, y_prob))
            recall  = float(recall_score(y_fv, y_pred, pos_label=1, zero_division=0))

        except Exception as exc:
            print(f"      Fold {fold_idx+1} FAILED: {exc}")
            auprc, recall = float("nan"), float("nan")

        auprcs.append(auprc)
        recalls.append(recall)
        print(f"    fold {fold_idx+1}: AUPRC={auprc:.4f}  Recall={recall:.4f}")

    return {"auprc": auprcs, "minority_recall": recalls}


# =============================================================================
# SWEEP RUNNER  (OFAT logic)
# =============================================================================

def run_sweep(
    sweep_config: dict,
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    X_unlabeled: np.ndarray,
) -> pd.DataFrame:
    """
    Execute one OFAT sweep: iterate over the grid, vary the target parameter,
    hold all others at DEFAULT_PARAMS, collect fold-level metrics.

    Returns a DataFrame with columns:
      param_value | mean_auprc | std_auprc | mean_recall | std_recall
      | fold_1_auprc … fold_N_auprc | fold_1_recall … fold_N_recall
    """
    param_name = sweep_config["param_name"]
    grid       = sweep_config["grid"]

    print(f"\n{'='*60}")
    print(f"  SWEEP: {param_name}")
    print(f"  Grid : {grid}")
    print(f"  Default params: { {k:v for k,v in DEFAULT_PARAMS.items() if k != param_name} }")
    print(f"{'='*60}")

    rows = []

    for val in grid:
        # Build this run's param dict: default + override
        run_params = DEFAULT_PARAMS.copy()
        run_params[param_name] = val

        print(f"\n  {param_name} = {val}")

        # Route noise_budget_fraction to its dedicated runner
        if param_name == "noise_budget_fraction":
            fold_metrics = run_noise_fraction_config(
                X_labeled, y_labeled, X_unlabeled,
                noise_budget_fraction = val,
                params                = run_params,
            )
        else:
            fold_metrics = run_one_config(
                X_labeled, y_labeled, X_unlabeled, run_params
            )

        auprcs  = fold_metrics["auprc"]
        recalls = fold_metrics["minority_recall"]

        row = {
            "param_value":  val,
            "mean_auprc":   float(np.nanmean(auprcs)),
            "std_auprc":    float(np.nanstd(auprcs)),
            "mean_recall":  float(np.nanmean(recalls)),
            "std_recall":   float(np.nanstd(recalls)),
        }
        for i, (a, r) in enumerate(zip(auprcs, recalls)):
            row[f"fold_{i+1}_auprc"]  = float(a)
            row[f"fold_{i+1}_recall"] = float(r)

        rows.append(row)
        print(f"  → mean AUPRC={row['mean_auprc']:.4f} ± {row['std_auprc']:.4f}"
              f"   mean Recall={row['mean_recall']:.4f} ± {row['std_recall']:.4f}")

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "█"*60)
    print("  SS-UGO — Phase 2: Hyperparameter Sensitivity Analysis")
    print("  OFAT sweeps on Thyroid (Sick) dataset")
    print("█"*60)

    X_lab_s, y_lab, X_unl_s = prepare_thyroid()

    sweep_results = {}

    for sweep_cfg in SWEEPS:
        df_sweep = run_sweep(sweep_cfg, X_lab_s, y_lab, X_unl_s)
        sweep_results[sweep_cfg["param_name"]] = df_sweep

        out_path = sweep_cfg["output_csv"]
        df_sweep.to_csv(out_path, index=False)
        print(f"\n  [Saved] {out_path}")
        print(df_sweep[["param_value", "mean_auprc", "std_auprc",
                         "mean_recall", "std_recall"]].to_string(index=False))

    # ── Consolidated sensitivity CSV ──────────────────────────────────────────
    frames = []
    for param_name, df in sweep_results.items():
        df_copy = df.copy()
        df_copy.insert(0, "param_name", param_name)
        frames.append(df_copy)
    consolidated = pd.concat(frames, ignore_index=True)
    consolidated.to_csv("phase2_sensitivity_all_sweeps.csv", index=False)
    print(f"\n[Saved] phase2_sensitivity_all_sweeps.csv  "
          f"({len(consolidated)} rows)")

    # ── Quick stability summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SENSITIVITY SUMMARY (AUPRC range = max - min across grid)")
    print(f"{'='*60}")
    for param_name, df in sweep_results.items():
        rng = df["mean_auprc"].max() - df["mean_auprc"].min()
        best_val = df.loc[df["mean_auprc"].idxmax(), "param_value"]
        print(f"  {param_name:<30}  range={rng:.4f}  best_val={best_val}")


if __name__ == "__main__":
    main()
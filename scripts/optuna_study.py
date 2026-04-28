"""
Optuna hyperparameter search for simulation visit_parameters.

Objective: find visit_params that produce simulated data whose
statistical properties best match the real dataset.

Metric (minimised):
    A weighted composite of Wasserstein-1 distances and scalar
    absolute differences
    computed on key summary statistics derived per-patient from both
    real and simulated data.

Usage:
    python optuna_study.py
"""

import warnings
import numpy as np
import pandas as pd
import optuna
from scipy.stats import wasserstein_distance
from IPython.utils import io

optuna.logging.set_verbosity(optuna.logging.WARNING)

from leaspy.io.data import Data
from leaspy.models import JointModel
from leaspy.datasets import load_dataset

# =============================================================================
# 1.  Load real data and fit model
# =============================================================================
FEATURES = ["Y0", "Y1", "Y2", "Y3"]
EVENT_COL = "EVENT_BOOL"    
EVENT_TIME_COL = "EVENT_TIME"
ID_COL = "ID"
TIME_COL = "TIME"

data = Data.from_dataframe(load_dataset("simulated_data_for_joint"), "joint")
real_df = data.to_dataframe()

model = JointModel(name="test_model", nb_events=1)
model.fit(data, "mcmc_saem", seed=1312, n_iter=10000, progress_bar=False)

# =============================================================================
# 2.  Summary statistics computed from a dataframe
# =============================================================================

def per_patient_stats(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Returns arrays of per-patient statistics used as 1-D distributions
    for Wasserstein comparison between real and simulated data.
    Scalar statistics are compared with absolute differences.
    """
    stats = {}

    # ── Survival ──────────────────────────────────────────────────────────────
    event_df = df.drop_duplicates(subset=ID_COL)
    stats["event_time"] = event_df[EVENT_TIME_COL].values.astype(float)
    stats["censoring_rate"] = np.array([event_df[EVENT_COL].mean()])   # scalar

    # ── Visit structure ───────────────────────────────────────────────────────
    grp = df.groupby(ID_COL)[TIME_COL]
    stats["n_visits"]   = grp.count().values.astype(float)
    stats["follow_up"]  = (grp.max() - grp.min()).values.astype(float)
    stats["visit_gap"]  = (
        df.groupby(ID_COL)[TIME_COL]
        .apply(lambda x: np.diff(np.sort(x)).mean() if len(x) > 1 else np.nan)
        .dropna()
        .values
    )

    # ── Longitudinal outcomes ─────────────────────────────────────────────────
    for feat in FEATURES:
        if feat not in df.columns:
            continue
        grp_feat = df.groupby(ID_COL)[feat]
        stats[f"{feat}_baseline"] = grp_feat.first().values.astype(float)
        stats[f"{feat}_last"]     = grp_feat.last().values.astype(float)
        stats[f"{feat}_mean"]     = grp_feat.mean().values.astype(float)

        # Linear slope per patient
        def slope(sub, feat=feat):  
            sub = sub.dropna()
            if len(sub) < 2:
                return np.nan
            t = df.loc[sub.index, TIME_COL].values
            return np.polyfit(t, sub.values, 1)[0]

        slopes = (
            df.groupby(ID_COL)
            .apply(lambda g: slope(g[feat]), include_groups=False)
            .dropna()
            .values
        )
        stats[f"{feat}_slope"] = slopes

    return stats


# Pre-compute real summary statistics once
real_stats = per_patient_stats(real_df)

# =============================================================================
# 3.  Composite loss
# =============================================================================

# Weights per statistic group (tune if needed)
WEIGHTS = {
    "event_time":    3.0,   # survival is the primary target
    "follow_up":     1.5,  
    "n_visits":      1.0,  
    "visit_gap":     1.0,   
    # per-feature contributions are added dynamically below
}
FEAT_WEIGHTS = {
    "baseline": 1.5,
    "last":     1.5,
    "mean":     1.0,
    "slope":    2.0,
}

def composite_loss(real_s: dict, sim_s: dict) -> float:
    """
    Weighted average of Wasserstein-1 distances (continuous distributions)
    and absolute difference for scalar statistics.
    """
    total, weight_sum = 0.0, 0.0

    def _add(key, w):
        nonlocal total, weight_sum
        r, s = real_s.get(key), sim_s.get(key)
        if r is None or s is None:
            return
        r = np.asarray(r, dtype=float)
        s = np.asarray(s, dtype=float)
        r = r[np.isfinite(r)]
        s = s[np.isfinite(s)]
        if len(r) == 0 or len(s) == 0:
            return
        # Normalise by the std of the real distribution so that weights reflect
        # true relative importance rather than being dominated by units/scale.
        scale = float(np.std(r)) if r.size > 1 else 1.0
        if scale == 0.0:
            scale = 1.0
        if r.size == 1:                       # scalar statistic
            total += w * abs(float(r) - float(s.mean())) / scale
        else:
            total += w * wasserstein_distance(r, s) / scale
        weight_sum += w

    _add("event_time", WEIGHTS["event_time"])
    _add("follow_up",  WEIGHTS["follow_up"])
    _add("n_visits",   WEIGHTS["n_visits"])
    _add("visit_gap",  WEIGHTS["visit_gap"])

    # censoring rate: simple absolute difference (already in [0,1], no scaling needed)
    r_cr = real_s.get("censoring_rate")
    s_cr = sim_s.get("censoring_rate")
    if r_cr is not None and s_cr is not None:
        total += 2.0 * abs(float(r_cr[0]) - float(s_cr[0]))
        weight_sum += 2.0

    for feat in FEATURES:
        for suffix, w in FEAT_WEIGHTS.items():
            _add(f"{feat}_{suffix}", w)

    return total / weight_sum if weight_sum > 0 else float("inf")


# =============================================================================
# 4.  Simulation helper
# =============================================================================

def run_simulations(visit_params: dict, n_sims: int = 20) -> list[pd.DataFrame]:
    """Run n_sims simulations and return list of DataFrames."""
    dfs = []
    with io.capture_output():
        for _ in range(n_sims):
            df_s = model.simulate(
                algorithm="joint_simulate",
                features=FEATURES,
                visit_parameters=visit_params,
            )
            dfs.append(df_s.data.to_dataframe())
    return dfs


def average_stats(list_of_dfs: list[pd.DataFrame]) -> dict[str, np.ndarray]:
    """
    Pool all simulated patients across replications into a single
    summary statistic array (mimics a large Monte-Carlo sample).
    """
    all_stats = [per_patient_stats(df) for df in list_of_dfs]
    pooled = {}
    for key in all_stats[0]:
        arrays = [s[key] for s in all_stats if key in s]
        pooled[key] = np.concatenate(arrays)
    return pooled


# =============================================================================
# 5.  Optuna objective
# =============================================================================

N_SIMS_PER_TRIAL = 100

def objective(trial: optuna.Trial) -> float:
    visit_params = {
        "patient_number": 100,
        "visit_type":     "random",

        "first_visit_mean": trial.suggest_float(
            "first_visit_mean", -10.0, 10.0
        ),
        "first_visit_std": trial.suggest_float(
            "first_visit_std", 0.1, 12.0, log=True
        ),

        "time_follow_up_mean": trial.suggest_float(
            "time_follow_up_mean", 1.0, 15.0
        ),
        "time_follow_up_std": trial.suggest_float(
            "time_follow_up_std", 0.1, 5.0, log=True
        ),

        "distance_visit_mean": trial.suggest_float(
            "distance_visit_mean", 1/12, 2.0, log=True   # ~1 month to 2 years
        ),
        "distance_visit_std": trial.suggest_float(
            "distance_visit_std", 1/48, 1.0, log=True
        ),

        "min_spacing_between_visits": trial.suggest_float(
            "min_spacing_between_visits", 0.05, 1.0, log=True
        ),
    }

    # Enforce: min_spacing ≤ distance_visit_mean (sanity constraint)
    if visit_params["min_spacing_between_visits"] > visit_params["distance_visit_mean"]:
        raise optuna.exceptions.TrialPruned()

    try:
        sim_dfs = run_simulations(visit_params, n_sims=N_SIMS_PER_TRIAL)
    except Exception as e:
        warnings.warn(f"Simulation failed: {e}")
        raise optuna.exceptions.TrialPruned()

    sim_stats = average_stats(sim_dfs)
    loss = composite_loss(real_stats, sim_stats)

    return loss


# =============================================================================
# 6.  Run the study
# =============================================================================

if __name__ == "__main__":
    import os
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    N_TRIALS = 500   

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="sim_visit_params",
    )

    # Seed with your known-good baseline so TPE has a warm start
    study.enqueue_trial({
        "first_visit_mean":            0.0,
        "first_visit_std":             5.7,
        "time_follow_up_mean":         6.4,
        "time_follow_up_std":          1.2,
        "distance_visit_mean":         0.7,
        "distance_visit_std":          0.3,
        "min_spacing_between_visits":  0.3,
    })

    # Also seed the defaults so TPE spans the full prior
    study.enqueue_trial({
        "first_visit_mean":            0.0,
        "first_visit_std":             0.4,
        "time_follow_up_mean":         6.0,
        "time_follow_up_std":          1.2,
        "distance_visit_mean":         2/12,
        "distance_visit_std":          0.75/12,
        "min_spacing_between_visits":  1.0,
    })

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print("\n" + "="*60)
    print(f"Best trial #{best.number}  —  loss = {best.value:.6f}")
    print("="*60)
    best_params = {
        "patient_number":              50,
        "visit_type":                  "random",
        **best.params,
    }
    print("\nBest visit_params:")
    for k, v in best_params.items():
        print(f"  {k:35s}: {v}")

    results_df = study.trials_dataframe()
    csv_path = os.path.join(OUTPUT_DIR, "optuna_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll trials saved to {csv_path}")

    try:
        import optuna.visualization as vis
        importances_path = os.path.join(OUTPUT_DIR, "optuna_param_importances.html")
        history_path = os.path.join(OUTPUT_DIR, "optuna_optimization_history.html")
        fig = vis.plot_param_importances(study)
        fig.write_html(importances_path)
        fig2 = vis.plot_optimization_history(study)
        fig2.write_html(history_path)
        print(f"Plots saved to {importances_path} and {history_path}")
    except Exception:
        pass
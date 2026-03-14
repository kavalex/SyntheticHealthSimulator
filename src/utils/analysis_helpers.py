"""
Shared helper functions for analysis notebooks.
Provides standardized setup, analysis routines, and universal statistics saving.
"""

import atexit
import logging
from pathlib import Path

atexit.register(logging.shutdown)
from typing import Optional, Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.paths import get_version, get_data_path
from src.utils.tables import save_table
from src.utils.plots import save_figure
from src.coefficients import DEFAULT_COEFFICIENTS
from src.logging_config import get_logger, setup_logging
from src.config import DEFAULT_CONFIG

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except ImportError:
    IterativeImputer = None
    import warnings
    warnings.warn("IterativeImputer not available. MICE imputation will fallback to median.")

logger = get_logger(__name__)

BIOMARKER_RANGES = DEFAULT_CONFIG.final_biomarker_ranges
EXPECTED_NOISE = {
    "hdl_mgdl": DEFAULT_CONFIG.measurement_noise["hdl"],
    "total_cholesterol_mgdl": DEFAULT_CONFIG.measurement_noise["tc"],
    "sbp_mmhg": DEFAULT_CONFIG.measurement_noise["sbp"],
    "hba1c_percent": DEFAULT_CONFIG.measurement_noise["hba1c"],
    "weight_kg": DEFAULT_CONFIG.measurement_noise["weight"],
}


def setup_analysis(notebook_name: str, version: Optional[str] = None) -> Dict[str, Any]:
    if version is None: version = get_version()
    data_path = get_data_path(version)
    base_output = Path(__file__).parent.parent.parent / "analysis" / notebook_name
    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "tables").mkdir(exist_ok=True)
    (base_output / "figures").mkdir(exist_ok=True)
    log_file = base_output / f"{notebook_name}.log"
    setup_logging(log_file=str(log_file))
    analysis_logger = get_logger("synthetic_health")
    analysis_logger.info(f"Notebook: {notebook_name}")
    return {"version": version, "data_path": data_path, "output_dir": base_output, "logger": analysis_logger}


def save_analysis_table(df: pd.DataFrame, name: str, output_dir: Path, **kwargs) -> str:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    return save_table(df, f"{name}.csv", output_dir=str(tables_dir), **kwargs)


def save_analysis_figure(fig: plt.Figure, name: str, output_dir: Path, **kwargs) -> str:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    return save_figure(fig, name, output_dir=figures_dir, **kwargs)


def save_descriptive_statistics(df: pd.DataFrame, value_cols: List[str], output_dir: Path, filename: str = "statistics",
                                group_col: Optional[str] = None,
                                percentiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> pd.DataFrame:
    if not value_cols: return pd.DataFrame()

    # Filter non-numeric columns automatically
    valid_cols = [c for c in value_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_cols: return pd.DataFrame()

    records = []
    for col in valid_cols:
        if group_col:
            grouped = df.groupby(group_col)[col]
            for name, group in grouped:
                data = group.dropna()
                if data.empty: continue
                stats_rec = {"Group": name, "Feature": col, "N": len(data), "Mean": data.mean(), "Std": data.std()}
                for p in percentiles: stats_rec[f"P{int(p * 100)}"] = data.quantile(p)
                records.append(stats_rec)
        else:
            data = df[col].dropna()
            if data.empty: continue
            stats_rec = {"Feature": col, "N": len(data), "Mean": data.mean(), "Std": data.std()}
            for p in percentiles: stats_rec[f"P{int(p * 100)}"] = data.quantile(p)
            records.append(stats_rec)
    stats_df = pd.DataFrame(records)
    if not stats_df.empty:
        cols_to_round = stats_df.select_dtypes(include=[np.number]).columns
        stats_df[cols_to_round] = stats_df[cols_to_round].round(3)
        save_analysis_table(stats_df, filename, output_dir)
    return stats_df


# --- Existing helpers (imported from previous version) ---

def estimate_ou_parameters(long_df: pd.DataFrame, factors: List[str], id_col: str = "person_id",
                           time_col: str = "year") -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    results = []
    for factor in factors:
        df = long_df.sort_values([id_col, time_col]).copy()
        df["next"] = df.groupby(id_col)[factor].shift(-1)
        df = df.dropna(subset=[factor, "next"])
        if len(df) == 0: continue
        X = df[factor].values.reshape(-1, 1)
        y = df["next"].values
        model = LinearRegression()
        model.fit(X, y)
        b = model.coef_[0]
        theta_est = 1 - b
        results.append({"factor": factor, "theta_estimated": theta_est, "beta": b})
    return pd.DataFrame(results)


def detect_stress_events(long_df: pd.DataFrame, stress_col: str = "stress_level", sleep_col: str = "sleep_hours",
                         alcohol_col: str = "alcohol_g_per_week", id_col: str = "person_id") -> pd.DataFrame:
    df = long_df.sort_values([id_col, "year"]).copy()
    df["stress_event"] = (df.groupby(id_col)[stress_col].diff() > 2.5) | (
            df.groupby(id_col)[sleep_col].diff() < -0.5) | (df.groupby(id_col)[alcohol_col].pct_change() > 0.5)
    return df


def cluster_lifestyle_trajectories(wide_df: pd.DataFrame, features: List[str], n_clusters: int = 3,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    feature_vectors = []
    patient_ids = []
    for _, row in wide_df.iterrows():
        vec = []
        for feat in features:
            cols = [c for c in wide_df.columns if c.startswith(feat)]
            if not cols: continue
            vals = row[cols].values
            vec.extend([np.mean(vals), np.std(vals), np.polyfit(range(len(vals)), vals, 1)[0] if len(vals) > 1 else 0])
        feature_vectors.append(vec)
        patient_ids.append(row["person_id"])
    X = np.array(feature_vectors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return pd.Series(labels, index=patient_ids, name="cluster"), None


def engineer_features(X_df: pd.DataFrame) -> pd.DataFrame:
    X = X_df.copy()
    if all(c in X.columns for c in ["final_bmi", "final_sbp_mmhg", "final_hba1c_percent"]): X[
        "metabolic_syndrome_score"] = (
            (X["final_bmi"] > 30).astype(float) + (X["final_sbp_mmhg"] > 130).astype(float) + (
            X["final_hba1c_percent"] > 5.7).astype(float))
    if all(c in X.columns for c in ["avg_strength_met_minutes", "avg_total_met_minutes"]): X["strength_ratio"] = X[
                                                                                                                     "avg_strength_met_minutes"] / (
                                                                                                                         X[
                                                                                                                             "avg_total_met_minutes"] + 1)
    if "avg_alcohol_g_per_week" in X.columns: X["high_alcohol_risk"] = (X["avg_alcohol_g_per_week"] > 140).astype(float)
    if "avg_sleep_hours" in X.columns: X["sleep_deficit"] = np.maximum(0, 6 - X["avg_sleep_hours"])
    if "age_end" in X.columns: X["age_over_45"] = (X["age_end"] > 45).astype(float)
    if all(c in X.columns for c in ["final_hdl_mgdl", "final_total_cholesterol_mgdl"]): X["non_hdl_cholesterol"] = X[
                                                                                                                       "final_total_cholesterol_mgdl"] - \
                                                                                                                   X[
                                                                                                                       "final_hdl_mgdl"]
    return X


def compare_with_reference(df: pd.DataFrame, reference_dict: Optional[Dict] = None) -> pd.DataFrame:
    ref = DEFAULT_COEFFICIENTS.ref if reference_dict is None else reference_dict
    results = []
    for marker, ref_info in ref.__dict__.items():
        if not hasattr(ref_info, "get"): continue
        if marker not in df.columns: continue
        syn_vals = df[marker].dropna()
        if "age_50_59" in ref_info:
            ref_mean, ref_std = ref_info["age_50_59"]["mean"], ref_info["age_50_59"]["std"]
        elif "male" in ref_info:
            ref_mean, ref_std = (ref_info["male"]["mean"] + ref_info["female"]["mean"]) / 2, (
                    ref_info["male"]["std"] + ref_info["female"]["std"]) / 2
        else:
            continue
        t_stat, p_val = stats.ttest_1samp(syn_vals, ref_mean)
        results.append(
            {"Feature": marker, "Synthetic Mean": syn_vals.mean(), "Reference Mean": ref_mean, "T-test p-value": p_val})
    return pd.DataFrame(results)


def run_baseline_models(X: pd.DataFrame, y: pd.Series, model_names: Optional[List[str]] = None, test_size: float = 0.2,
                        random_state: int = 42) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    if model_names is None: model_names = ["lr", "rf"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    results = []
    if "lr" in model_names:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        lr.fit(X_train_scaled, y_train)
        results.append(
            {"Model": "Logistic Regression", "AUC-ROC": roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])})
    if "rf" in model_names:
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight="balanced")
        rf.fit(X_train, y_train)
        results.append({"Model": "Random Forest", "AUC-ROC": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])})
    return pd.DataFrame(results)


def estimate_measurement_noise(long_df: pd.DataFrame, markers: List[str], id_col: str = "person_id") -> pd.DataFrame:
    results = []
    for marker in markers:
        if marker not in long_df.columns: continue
        diff_df = long_df.groupby(id_col)[marker].diff().dropna()
        if len(diff_df) < 2: continue
        noise_std = diff_df.std() / np.sqrt(2)
        results.append(
            {"Marker": marker, "Estimated Noise σ": noise_std, "Expected Noise σ": EXPECTED_NOISE.get(marker, np.nan)})
    return pd.DataFrame(results)


def prepare_prediction_data_pivot(long_df: pd.DataFrame, marker: str, target_year: int = 15, history_len: int = 5,
                                  id_col: str = "person_id", time_col: str = "year") -> Tuple[np.ndarray, np.ndarray]:
    pivot = long_df.pivot(index=id_col, columns=time_col, values=marker)
    hist_years = list(range(target_year - history_len, target_year))
    missing = [y for y in hist_years if y not in pivot.columns]
    if missing: return None, None
    X = pivot[hist_years].values
    if target_year not in pivot.columns: return None, None
    y = pivot[target_year].values
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def calculate_calibration_metrics(df: pd.DataFrame, risk_col: str, outcome_col: str, n_bins: int = 10) -> Dict[
    str, float]:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    if risk_col not in df.columns or outcome_col not in df.columns: return {}
    y_true, y_pred = df[outcome_col], df[risk_col]
    brier = brier_score_loss(y_true, y_pred)
    eo_ratio = y_pred.sum() / y_true.sum() if y_true.sum() > 0 else np.nan
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    if len(prob_true) >= 2:
        slope, intercept, _, _, _ = stats.linregress(prob_pred, prob_true)
    else:
        slope, intercept = np.nan, np.nan
    return {"Brier Score": brier, "E/O Ratio": eo_ratio, "Calibration Slope": slope, "Calibration Intercept": intercept}


def analyze_missing_pattern(df: pd.DataFrame, age_col: str = "age_start") -> pd.DataFrame:
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_cols = missing_pct[missing_pct > 0]
    if age_col not in df.columns or missing_cols.empty: return pd.DataFrame()
    mar_results = []
    for col in missing_cols.index[:10]:
        contingency = pd.crosstab(df[age_col] < 45, df[col].isna())
        if contingency.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(contingency)
        else:
            chi2, p = np.nan, np.nan
        mar_results.append({"Column": col, "Missing %": missing_pct[col], "Chi-square": chi2, "p-value": p})
    return pd.DataFrame(mar_results)


def impute_data(X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "mice", random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    if method == "median":
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median")
    elif method == "mice":
        if IterativeImputer is None:
            raise ImportError("IterativeImputer is not available. Please install scikit-learn >= 0.24 and ensure enable_iterative_imputer is imported.")
        imp = IterativeImputer(max_iter=10, random_state=random_state, sample_posterior=True)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    return imp.fit_transform(X_train), imp.transform(X_test)


def run_model_comparison(X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series,
                         models: Optional[Dict] = None) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    if models is None: models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)}
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append({"Model": name, "AUC-ROC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])})
    return pd.DataFrame(results)

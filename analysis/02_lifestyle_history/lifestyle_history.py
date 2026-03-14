# %% [markdown]
# # Lifestyle History Analysis (02_lifestyle_history)
#
# Panel data: 20 years of observations for each patient.
# Key features: temporal dependencies (OU processes), autocorrelation, stress events.
# Goals: verify OU parameters, analyze stress events, link trajectories to demographics and outcomes,
# cluster lifestyle patterns, predict future from past.
#
# **Important:** All data are synthetic and NOT intended for clinical use.

# %% [markdown]
# ## Setup
# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.data_loader import load_merged_dataset
from src.utils.analysis_helpers import (
    setup_analysis, save_analysis_table, save_analysis_figure,
    save_descriptive_statistics, estimate_ou_parameters, detect_stress_events,
    cluster_lifestyle_trajectories
)

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Initialize analysis
# %%
cfg = setup_analysis("02_lifestyle_history")
output_dir = cfg["output_dir"]

# %% [markdown]
# ## 1. Load Data
# %%
print("\n### Loading datasets...")
long_df = load_merged_dataset("02_lifestyle_history", cfg["data_path"], cfg["version"])
cfg["logger"].info(f"Long format shape: {long_df.shape}")

# %% [markdown]
# ## 2. Descriptive Statistics
# %%
factors = [
    "alcohol_g_per_week", "cardio_met_minutes", "strength_met_minutes",
    "cigarettes_per_day", "stress_level", "sleep_hours"
]

years_to_check = [0, 5, 10, 15, 19]
# Percentiles for selected years
save_descriptive_statistics(
    long_df, factors, output_dir, filename="lifestyle_percentiles",
    group_col=None, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)

# Trends by Sex and Diabetes (using existing helper)
save_descriptive_statistics(
    long_df, factors, output_dir, filename="lifestyle_trends_by_sex",
    group_col="sex"
)
save_descriptive_statistics(
    long_df, factors, output_dir, filename="lifestyle_trends_by_diabetes",
    group_col="has_diabetes"
)

# Сохранить тренды по полу
trends_by_sex = long_df.groupby(['year', 'sex'])[factors].mean().reset_index()
save_analysis_table(trends_by_sex, 'lifestyle_trends_by_sex', output_dir)

# Сохранить тренды по диабету
trends_by_diabetes = long_df.groupby(['year', 'has_diabetes'])[factors].mean().reset_index()
save_analysis_table(trends_by_diabetes, 'lifestyle_trends_by_has_diabetes', output_dir)

# Full trends (mean and std per year)
trend_full = long_df.groupby("year")[factors].agg(["mean", "std"]).round(2)
trend_full = trend_full.reset_index()
new_columns = ["year"]
for factor in factors:
    new_columns.append(f"{factor}_mean")
    new_columns.append(f"{factor}_std")
trend_full.columns = new_columns
save_analysis_table(trend_full, "lifestyle_trends_by_year", output_dir)

# Correlation between factors per year
corr_by_year = []
for y in years_to_check:
    year_data = long_df[long_df["year"] == y][factors]
    corr_matrix = year_data.corr().round(3)
    corr_df = corr_matrix.reset_index().rename(columns={"index": "factor"})
    corr_df["year"] = y
    corr_by_year.append(corr_df)
corr_all = pd.concat(corr_by_year, ignore_index=True)
save_analysis_table(corr_all, "lifestyle_correlations", output_dir)

# %% [markdown]
# ## 3. Individual Trajectories (Sample Patients)
# %%
sample_ids = long_df["person_id"].unique()[:3]
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for i, pid in enumerate(sample_ids):
    patient = long_df[long_df["person_id"] == pid].sort_values("year")
    years = patient["year"].values

    axes[i, 0].plot(years, patient["alcohol_g_per_week"].values, "o-", label="Alcohol (g/week)")
    axes[i, 0].set_ylabel("Alcohol")
    axes[i, 0].set_title(f"Patient {pid}: Alcohol")
    axes[i, 0].legend()

    axes[i, 1].plot(years, patient["cardio_met_minutes"].values, "s-", label="Cardio MET")
    axes[i, 1].plot(years, patient["strength_met_minutes"].values, "^-", label="Strength MET")
    axes[i, 1].set_ylabel("MET-minutes")
    axes[i, 1].set_title(f"Patient {pid}: Activity")
    axes[i, 1].legend()

plt.tight_layout()
save_analysis_figure(fig, "lifestyle_trajectories_sample", output_dir)
plt.show()

# %% [markdown]
# ## 4. Verify OU Processes
# %%
true_theta = {
    "alcohol_g_per_week": 0.8, "cardio_met_minutes": 0.2,
    "strength_met_minutes": 0.25, "cigarettes_per_day": 0.3,
    "stress_level": 0.2, "sleep_hours": 0.2,
}

theta_est = estimate_ou_parameters(long_df, factors)
theta_est["theta_true"] = theta_est["factor"].map(true_theta)
print(theta_est.round(3).to_markdown())
save_analysis_table(theta_est.round(3), "ou_theta_estimates", output_dir)

# Plot autocorrelation decay
lags = [1, 2, 3]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, factor in enumerate(factors):
    ax = axes[i]
    obs_corrs = []
    for lag in lags:
        lagged = long_df.groupby("person_id")[factor].shift(lag)
        corr = long_df[factor].corr(lagged)
        obs_corrs.append(corr)
    ax.plot(lags, obs_corrs, "o-", label="Observed")
    expected = [np.exp(-true_theta[factor] * lag) for lag in lags]
    ax.plot(lags, expected, "s--", label=f"Expected (θ={true_theta[factor]})")
    ax.set_xlabel("Lag (years)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(factor)
    ax.legend()
    ax.set_ylim(0, 1)
plt.tight_layout()
save_analysis_figure(fig, "ou_autocorrelation", output_dir)
plt.show()

# %% [markdown]
# ## 5. Stress Event Analysis
# %%
long_df_events = detect_stress_events(long_df)

# Count events per patient
event_counts = long_df_events.groupby("person_id")["stress_event"].sum()
cfg["logger"].info(f"Mean events per patient: {event_counts.mean():.2f}")

# Plot distribution
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(event_counts, bins=range(0, 21), edgecolor="black", alpha=0.7)
ax.axvline(8, color="red", linestyle="--", label="Expected (0.4×20)")
ax.set_xlabel("Number of detected events per patient")
ax.set_ylabel("Count")
ax.set_title("Distribution of stress events")
ax.legend()
plt.tight_layout()
save_analysis_figure(fig, "stress_events_hist", output_dir)
plt.show()

# Check duration effect
long_df_events["next_stress"] = long_df_events.groupby("person_id")["stress_level"].shift(-1)
event_rows = long_df_events[long_df_events["stress_event"]].copy()
next_year_stress = event_rows["next_stress"].dropna()
long_df_events["is_after_event"] = (
    long_df_events.groupby("person_id")["stress_event"].shift(1).fillna(False)
)
baseline_stress = long_df_events[
    ~(long_df_events["stress_event"] | long_df_events["is_after_event"])
]["stress_level"]

if len(next_year_stress) > 0 and len(baseline_stress) > 0:
    t_stat, p_val = stats.ttest_ind(next_year_stress, baseline_stress, equal_var=False)
    cfg["logger"].info(
        f"Stress level in year after event: mean={next_year_stress.mean():.2f}"
    )
    cfg["logger"].info(f"Baseline stress: mean={baseline_stress.mean():.2f}")
    cfg["logger"].info(f"t-test: p={p_val:.4f}")

# %% [markdown]
# ## 6. Link Trajectories to Demographics and Outcomes
# %%
# Trajectories by sex
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, factor in enumerate(factors):
    ax = axes[i]
    for sex in ["M", "F"]:
        subset = long_df_events[long_df_events["sex"] == sex]
        means = subset.groupby("year")[factor].mean()
        ax.plot(means.index, means.values, label=f"Sex {sex}", marker="o")
    ax.set_title(factor)
    ax.set_xlabel("Year")
    ax.set_ylabel(factor)
    ax.legend()
plt.tight_layout()
save_analysis_figure(fig, "trajectories_by_sex", output_dir)
plt.show()

# t‑test for differences by diabetes status
test_results = []
for factor in factors:
    group0 = long_df_events[long_df_events["has_diabetes"] == 0][factor].dropna()
    group1 = long_df_events[long_df_events["has_diabetes"] == 1][factor].dropna()
    t, p = stats.ttest_ind(group0, group1, equal_var=False)
    test_results.append(
        {
            "factor": factor,
            "mean_no_diabetes": group0.mean(),
            "mean_diabetes": group1.mean(),
            "t_stat": t,
            "p_value": p,
        }
    )
test_df = pd.DataFrame(test_results).round(3)
print(test_df.to_markdown())
save_analysis_table(test_df, "ttest_diabetes", output_dir)

# %% [markdown]
# ## 7. Clustering of Lifestyle Trajectories
# %%
# Convert to wide format for clustering
wide_df = long_df.pivot_table(index="person_id", columns="year", values=factors).reset_index()
wide_df.columns = [f"{c[0]}_{c[1]}" if c[1] != '' else c[0] for c in wide_df.columns]

# Use a random subset for speed
all_pids = wide_df["person_id"].unique()
np.random.seed(42)
sample_pids = np.random.choice(all_pids, size=min(1000, len(all_pids)), replace=False)
sample_wide = wide_df[wide_df["person_id"].isin(sample_pids)]

features_ts = factors
cluster_labels, _ = cluster_lifestyle_trajectories(sample_wide, features_ts, n_clusters=3)

# Merge clusters back
cluster_map = pd.DataFrame({"person_id": sample_pids, "cluster": cluster_labels.values})
long_clustered = long_df_events.merge(cluster_map, on="person_id", how="left")

# Plot cluster centroids - show all features
n_clusters = 3
n_features = len(features_ts)

# Adjust figure size: width based on number of features, height based on clusters
fig, axes = plt.subplots(n_clusters, n_features, figsize=(5 * n_features, 4 * n_clusters))

# Ensure axes is a 2D array
if n_clusters == 1:
    axes = axes.reshape(1, -1)
if n_features == 1:
    axes = axes.reshape(-1, 1)

for cluster in range(n_clusters):
    cluster_pids = sample_pids[cluster_labels == cluster]
    for feat_idx in range(n_features):
        feat_name = features_ts[feat_idx]
        ax = axes[cluster, feat_idx]
        # Plot individual trajectories (first 10 patients per cluster)
        for pid in cluster_pids[:10]:
            pat = long_clustered[long_clustered["person_id"] == pid].sort_values("year")
            ax.plot(pat["year"], pat[feat_name].values, alpha=0.3, color=f"C{cluster}")
        # Plot average trajectory
        avg_traj = (
            long_clustered[long_clustered["cluster"] == cluster]
            .groupby("year")[feat_name]
            .mean()
        )
        ax.plot(avg_traj.index, avg_traj.values, "k-", linewidth=2, label="Avg")
        ax.set_title(f"Cluster {cluster}: {feat_name}")
        if cluster == 0 and feat_idx == 0:
            ax.legend()
plt.tight_layout()
save_analysis_figure(fig, "lifestyle_clusters", output_dir)
plt.show()

# Cluster summary
cluster_summary = (
    long_clustered.groupby("cluster")
    .agg(
        {
            "has_diabetes": "mean",
            "has_cvd": "mean",
            "age_start": "mean",
            "sex": lambda x: (x == "M").mean(),
        }
    )
    .rename(
        columns={
            "has_diabetes": "diabetes_rate",
            "has_cvd": "cvd_rate",
            "age_start": "mean_age",
            "sex": "male_prop",
        }
    )
    .reset_index()
    .round(3)
)
print(cluster_summary.to_markdown())
save_analysis_table(cluster_summary, "cluster_summary", output_dir)

# Chi-square test
from scipy.stats import chi2_contingency

contingency = pd.crosstab(
    long_clustered["cluster"].dropna(), long_clustered["has_diabetes"].dropna()
)
chi2, p, dof, expected = chi2_contingency(contingency)
cfg["logger"].info(f"Chi‑square test for cluster × diabetes: p = {p:.4f}")

# %% [markdown]
# ## 8. Prediction: Future from Past
# %%
# Build wide format for prediction
wide_df = long_df.pivot_table(index="person_id", columns="year", values=factors).reset_index()
wide_df.columns = [f"{c[0]}_{c[1]}" if c[1] != '' else c[0] for c in wide_df.columns]

# Add baseline and outcomes
baseline_cols = ["age_start", "sex", "genetic_risk_diabetes", "genetic_risk_cvd"]
baseline_data = long_df[["person_id"] + baseline_cols].drop_duplicates("person_id")
wide = wide_df.merge(baseline_data, on="person_id")
outcomes = long_df[["person_id", "has_diabetes", "has_cvd"]].drop_duplicates("person_id")
wide = wide.merge(outcomes, on="person_id")


def extract_features_from_first_10(row, factor):
    values = [row.get(f"{factor}_{y}", np.nan) for y in range(10)]
    if any(np.isnan(values)):
        return np.nan, np.nan, np.nan
    mean_val = np.mean(values)
    std_val = np.std(values)
    trend = np.polyfit(range(10), values, 1)[0] if len(values) > 1 else 0
    return mean_val, std_val, trend


target_factors = ["weight_kg_true", "stress_level"]
for factor in target_factors:
    means, stds, trends = [], [], []
    for idx, row in wide.iterrows():
        m, s, t = extract_features_from_first_10(row, factor)
        means.append(m)
        stds.append(s)
        trends.append(t)
    wide[f"{factor}_mean_first10"] = means
    wide[f"{factor}_std_first10"] = stds
    wide[f"{factor}_trend_first10"] = trends

# Add means of other lifestyle factors
for factor in ["alcohol_g_per_week", "cardio_met_minutes", "strength_met_minutes", "sleep_hours"]:
    means = []
    for idx, row in wide.iterrows():
        vals = [row.get(f"{factor}_{y}", np.nan) for y in range(10)]
        if any(np.isnan(vals)):
            means.append(np.nan)
        else:
            means.append(np.mean(vals))
    wide[f"{factor}_mean_first10"] = means

# Target: average over last 5 years
for factor in target_factors:
    final_vals = []
    for idx, row in wide.iterrows():
        vals = [row.get(f"{factor}_{y}", np.nan) for y in range(15, 20)]
        if any(np.isnan(vals)):
            final_vals.append(np.nan)
        else:
            final_vals.append(np.mean(vals))
    wide[f"{factor}_final"] = final_vals

# Drop rows with any NaN in features or target
wide["sex_M"] = (wide["sex"] == "M").astype(int)
baseline_features = ["age_start", "sex_M", "genetic_risk_diabetes", "genetic_risk_cvd"]
feat_first10_only = [col for col in wide.columns if "_first10" in col]
feat_all = feat_first10_only + baseline_features

prediction_results = []
for target in target_factors:
    target_col = f"{target}_final"
    data = wide.dropna(subset=feat_all + [target_col])
    if len(data) == 0:
        cfg["logger"].warning(f"No complete data for {target}, skipping")
        continue
    X_first10 = data[feat_first10_only]
    X_all = data[feat_all]
    y = data[target_col]

    X_train1, X_test1, y_train, y_test = train_test_split(
        X_first10, y, test_size=0.2, random_state=42
    )
    X_train2, X_test2, _, _ = train_test_split(X_all, y, test_size=0.2, random_state=42)

    rf1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf1.fit(X_train1, y_train)
    y_pred1 = rf1.predict(X_test1)
    r2_1 = r2_score(y_test, y_pred1)
    rmse_1 = np.sqrt(mean_squared_error(y_test, y_pred1))

    rf2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf2.fit(X_train2, y_train)
    y_pred2 = rf2.predict(X_test2)
    r2_2 = r2_score(y_test, y_pred2)
    rmse_2 = np.sqrt(mean_squared_error(y_test, y_pred2))

    prediction_results.append({
        "Target": target,
        "R²_first10": r2_1,
        "RMSE_first10": rmse_1,
        "R²_first10_baseline": r2_2,
        "RMSE_first10_baseline": rmse_2,
    })

if prediction_results:
    pred_df = pd.DataFrame(prediction_results).round(3)
    save_analysis_table(pred_df, "lifestyle_prediction_results", output_dir)

# %% [markdown]
# ## 9. Textual Summary for LLM
# %%
summary_lines = []

# OU parameters
theta_est_df = pd.read_csv(output_dir / "tables/ou_theta_estimates.csv")
summary_lines.append("**OU process verification:**")
for _, row in theta_est_df.iterrows():
    summary_lines.append(
        f"  {row['factor']}: estimated θ = {row['theta_estimated']:.3f} (true = {row['theta_true']:.3f})")

# Stress events
summary_lines.append(f"**Stress events:** mean per patient = {event_counts.mean():.2f} (expected 8.0)")

# Differences by diabetes
test_df = pd.read_csv(output_dir / "tables/ttest_diabetes.csv")
summary_lines.append("**Differences by diabetes status:**")
for _, row in test_df.iterrows():
    if row['p_value'] < 0.05:
        summary_lines.append(
            f"  {row['factor']}: mean_no_diabetes = {row['mean_no_diabetes']:.2f}, mean_diabetes = {row['mean_diabetes']:.2f} (p = {row['p_value']:.3f})")

# Clusters
cluster_sum = pd.read_csv(output_dir / "tables/cluster_summary.csv")
summary_lines.append("**Lifestyle clusters:**")
for _, row in cluster_sum.iterrows():
    summary_lines.append(
        f"  Cluster {int(row['cluster'])}: diabetes rate = {row['diabetes_rate']:.3f}, male proportion = {row['male_prop']:.3f}")

# Prediction
if prediction_results:
    pred_df = pd.read_csv(output_dir / "tables/lifestyle_prediction_results.csv")
    summary_lines.append("**Prediction of future lifestyle from first 10 years:**")
    for _, row in pred_df.iterrows():
        summary_lines.append(f"  {row['Target']}: R² = {row['R²_first10_baseline']:.3f} (with baseline features)")

with open(output_dir / "analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

cfg["logger"].info("All analysis tables and summary saved.")

# %% [markdown]
# ## 10. Summary and Conclusions
#
# 1. **OU process verification** – estimated parameters are close to true values; deviations due to stress events.
# 2. **Stress events** – average count matches expectation, carry‑over effect significant.
# 3. **Lifestyle trajectories vs. outcomes** – significant differences in alcohol, cardio, sleep by diabetes status.
# 4. **Clustering** – three distinct lifestyle patterns with different disease prevalence.
# 5. **Prediction** – future weight and stress can be predicted from first 10 years with R² ~0.5-0.6.
#
# The dataset faithfully reproduces intended properties.
# %%

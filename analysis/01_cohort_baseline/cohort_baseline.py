# %% [markdown]
# # Baseline Cohort Analysis (01_cohort_baseline)
#
# This notebook explores the static characteristics of patients.
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
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load_baseline, load_risks
from src.utils.analysis_helpers import (
    setup_analysis,
    save_analysis_table,
    save_analysis_figure,
    save_descriptive_statistics,
    run_baseline_models,
)
from src.utils.validators import DatasetValidator

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Initialize analysis
# %%
cfg = setup_analysis("01_cohort_baseline")
output_dir = cfg["output_dir"]

# %% [markdown]
# ## 1. Load Data
# %%
print("\n### Loading datasets...")
baseline = load_baseline(cfg["data_path"], cfg["version"])
risks = load_risks(cfg["data_path"], cfg["version"])

# Merge by person_id
df = baseline.merge(
    risks[
        [
            "person_id",
            "has_diabetes",
            "has_cvd",
            "has_stroke",
            "diabetes_risk_10year",
            "cvd_risk_10year",
            "stroke_risk_10year",
        ]
    ],
    on="person_id",
    how="left",
)

cfg["logger"].info(f"Loaded {len(df)} patients, {df.shape[1]} features")

# %% [markdown]
# ## 2. General Information and Summary Statistics
# %%
# Columns for validation and overview (includes sex for grouping/checks)
main_cols = [
    "age_start", "sex", "height_cm", "weight_start_kg", "bmi_start",
    "initial_hdl_mgdl", "initial_total_cholesterol_mgdl",
    "initial_sbp_mmhg", "initial_hba1c_percent",
    "genetic_risk_diabetes", "genetic_risk_cvd", "genetic_risk_stroke",
]

# Numerical columns specifically for statistics calculation (NO string columns)
numeric_cols = [
    "age_start", "height_cm", "weight_start_kg", "bmi_start",
    "initial_hdl_mgdl", "initial_total_cholesterol_mgdl",
    "initial_sbp_mmhg", "initial_hba1c_percent",
    "genetic_risk_diabetes", "genetic_risk_cvd", "genetic_risk_stroke"
]

# Calculate and save stats using the universal function
print("\n### Descriptive Statistics")
save_descriptive_statistics(df, numeric_cols, output_dir, filename="baseline_summary")

# By Sex
save_descriptive_statistics(df, numeric_cols, output_dir, filename="baseline_summary_by_sex", group_col="sex")

# %%
# Sex Distribution (visualization)
sex_counts = df["sex"].value_counts(normalize=True)
sex_table = pd.DataFrame(
    {
        "Sex": sex_counts.index,
        "Proportion": sex_counts.values,
        "Count": df["sex"].value_counts().values,
    }
)
print(sex_table.to_markdown())
save_analysis_table(sex_table, "sex_distribution", output_dir)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sex_counts.plot(kind="bar", ax=ax[0], color=["skyblue", "pink"])
ax[0].set_title("Sex ratio")
ax[0].set_ylabel("Proportion")
ax[0].set_xticklabels(["Male", "Female"], rotation=0)

# Age distribution
ax[1].hist(df["age_start"], bins=30, edgecolor="black", alpha=0.7)
ax[1].axvline(df["age_start"].mean(), color="red", linestyle="--", label=f"Mean = {df['age_start'].mean():.1f}")
ax[1].set_title("Age distribution")
ax[1].set_xlabel("Age at baseline")
ax[1].legend()

plt.tight_layout()
save_analysis_figure(fig, "age_sex_dist", output_dir)
plt.show()

# %% [markdown]
# ## 3. Validation against Generator Specifications
# %%
validator = DatasetValidator(cfg["data_path"])
validator.load_data()
validator._validate_cohort()

val_results = pd.DataFrame([{
    "Check": r.name,
    "Expected": r.details.get("expected", "N/A") if r.details else "N/A",
    "Actual": r.message,
    "Status": "OK" if r.passed else "WARN"
} for r in validator.results if r.name.startswith("Cohort")])

print(val_results.to_markdown())
save_analysis_table(val_results, "validation_check", output_dir)

# %% [markdown]
# ## 4. Correlation Analysis
# %%
corr_cols = [
    "age_start", "bmi_start", "initial_hdl_mgdl", "initial_total_cholesterol_mgdl",
    "initial_sbp_mmhg", "initial_hba1c_percent",
    "genetic_risk_cvd", "genetic_risk_diabetes",
]

corr_matrix = df[corr_cols].corr()

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation matrix of baseline features")
plt.tight_layout()
save_analysis_figure(fig, "correlation_heatmap", output_dir)
plt.show()

save_analysis_table(corr_matrix, "correlation_matrix_full", output_dir)

# Key correlations (specific pairs of interest)
key_pairs = [
    ("bmi_start", "initial_sbp_mmhg"),
    ("bmi_start", "initial_hdl_mgdl"),
    ("age_start", "initial_sbp_mmhg"),
    ("initial_hdl_mgdl", "initial_total_cholesterol_mgdl"),
    ("genetic_risk_cvd", "initial_total_cholesterol_mgdl"),
    ("genetic_risk_diabetes", "initial_hba1c_percent"),
]
key_data = []
for a, b in key_pairs:
    key_data.append(
        {"Feature 1": a, "Feature 2": b, "Correlation": corr_matrix.loc[a, b]}
    )
key_corr_df = pd.DataFrame(key_data).round(3)
print(key_corr_df.to_markdown())
save_analysis_table(key_corr_df, "key_correlations", output_dir)

# %% [markdown]
# ## 5. Modeling: Disease Prediction from Baseline Data
# %%
# Features: demographics + genetics + initial biomarkers
features = [
    "age_start",
    "bmi_start",
    "initial_hdl_mgdl",
    "initial_total_cholesterol_mgdl",
    "initial_sbp_mmhg",
    "initial_hba1c_percent",
    "genetic_risk_diabetes",
    "genetic_risk_cvd",
    "genetic_risk_stroke",
]
# Add sex as categorical
df_encoded = pd.get_dummies(df, columns=["sex"], drop_first=True)
feature_names = features + ["sex_M"]

X = df_encoded[feature_names].copy()

targets = {"diabetes": "has_diabetes", "cvd": "has_cvd", "stroke": "has_stroke"}

all_results = []
for target_name, target_col in targets.items():
    y = df[target_col]
    res = run_baseline_models(X, y)
    res.insert(0, "Disease", target_name)
    all_results.append(res)

results_df = pd.concat(all_results, ignore_index=True)
print(results_df.to_markdown())
save_analysis_table(results_df, "prediction_results", output_dir)

# %% [markdown]
# ### ROC Curve for Diabetes (example)
# %%
# Train models specifically for ROC curve visualization
y = df["has_diabetes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression (scaled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(lr, X_test_scaled, y_test, name="Logistic Regression", ax=ax)
RocCurveDisplay.from_estimator(rf, X_test, y_test, name="Random Forest", ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_title("ROC curves for diabetes prediction")
ax.legend()
save_analysis_figure(fig, "roc_diabetes", output_dir)
plt.show()

# %% [markdown]
# ## 6. Feature Importance (Diabetes Example)
# %%
from sklearn.ensemble import RandomForestClassifier

rf_diabetes = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_diabetes.fit(X, df["has_diabetes"])

importances = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_diabetes.feature_importances_
}).sort_values("Importance", ascending=False)

print(importances.to_markdown())
save_analysis_table(importances, "feature_importance_diabetes", output_dir)

# Visualization
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importances["Feature"], importances["Importance"], color="steelblue")
ax.set_xlabel("Importance")
ax.set_title("Feature importance (Random Forest, diabetes)")
ax.invert_yaxis()
plt.tight_layout()
save_analysis_figure(fig, "feature_importance", output_dir)
plt.show()

# %% [markdown]
# ## 7. Conclusions
#
# 1. **Data match generator specifications** – checks passed via Validator.
# 2. **Correlations** between baseline features are weak or moderate.
# 3. **Predictive power of baseline features** is low (AUC ~0.5-0.6).
#
# **Conclusion:** The synthetic dataset is well balanced; static features alone provide little predictive advantage.

# %% [markdown]
# ## 8. Textual Summary for Reports
# %%
summary_lines = []

# 1. Cohort overview
age_mean = df["age_start"].mean()
age_std = df["age_start"].std()
male_pct = (df["sex"] == "M").mean() * 100
summary_lines.append(
    f"**Cohort overview:** n = {len(df)}; age = {age_mean:.1f} ± {age_std:.1f}; male = {male_pct:.1f}%.")

# 2. Validation status
val_df = pd.read_csv(output_dir / "tables/validation_check.csv")
all_ok = (val_df["Status"] == "OK").all()
summary_lines.append(f"**Validation:** {'All checks passed' if all_ok else 'Some checks failed'}.")

# 3. Key correlations
key_corr_df = pd.read_csv(output_dir / "tables/key_correlations.csv")
summary_lines.append("**Key correlations:**")
for _, row in key_corr_df.iterrows():
    summary_lines.append(f"  {row['Feature 1']} vs {row['Feature 2']}: r = {row['Correlation']:.3f}")

# 4. Disease prediction results
pred_df = pd.read_csv(output_dir / "tables/prediction_results.csv")
summary_lines.append("**Disease prediction from baseline features:**")
for _, row in pred_df.iterrows():
    summary_lines.append(f"  {row['Disease']}: {row['Model']} AUC = {row['AUC-ROC']:.3f}")

# 5. Top 3 features for diabetes
imp_df = pd.read_csv(output_dir / "tables/feature_importance_diabetes.csv")
top3 = imp_df.head(3)
summary_lines.append("**Top 3 features for diabetes prediction:**")
for _, row in top3.iterrows():
    summary_lines.append(f"  {row['Feature']}: {row['Importance']:.3f}")

with open(output_dir / "analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

cfg["logger"].info("Analysis summary saved to analysis_summary.txt")

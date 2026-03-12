# %% [markdown]
# # Health Risks Analysis (04_health_risks)
# 
# Target variables for ML: 6 risks (probabilities) and 6 binary outcomes.
# Key feature: risks are calibrated (SCORE2, FINDRISC), classes are balanced.
# Tasks: calibration validation, risk prediction, factor importance analysis.
# 
# **Important:** All data are synthetic and NOT intended for clinical use.
# %% [markdown]
# ## Setup
# %%
import sys
import warnings

# Add project root to path
sys.path.append('../..')

# Import utility modules
from utils import (
    get_version, get_data_path, get_output_dir,
    load_baseline, load_risks, load_aggregated,
    save_table, save_figure,
    print_markdown_table
)

# Import analysis libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, classification_report, average_precision_score,
    brier_score_loss, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# %% [markdown]
# ## Load configuration
# %%
VERSION = get_version()
DATA_PATH = get_data_path(VERSION)
OUTPUT_DIR = get_output_dir()
FIGURE_FORMAT = "svg"

print(f"Version: {VERSION}")
print(f"Data path: {DATA_PATH}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Log file: logs/04_health_risks_*.txt")
# %% [markdown]
# ## 1. Load Data
# %%
print("\n### Loading datasets...")
risks = load_risks(DATA_PATH, VERSION)
print(f"Risks loaded: {risks.shape}")

baseline = load_baseline(DATA_PATH, VERSION)
print(f"Baseline cohort: {baseline.shape}")

agg = load_aggregated(DATA_PATH, VERSION)
print(f"Aggregated dataset: {agg.shape}")

# Save sex map before merging to prevent loss during merge operations
print("\n### Saving sex mapping before merge...")
sex_map = risks[['person_id', 'sex']].set_index('person_id')
print(f"Sex map created: {len(sex_map)} entries")

# Merge all dataframes
# Merge risks with baseline (exclude sex from baseline to avoid conflict)
df = risks.merge(
    baseline[['person_id', 'age_start', 'height_cm', 'weight_start_kg']],
    on='person_id',
    how='left'
)

# Prepare aggregated columns
agg_cols = ['person_id', 'final_bmi', 'final_hdl_mgdl', 'final_sbp_mmhg',
            'final_hba1c_percent', 'avg_alcohol_g_per_week', 'avg_total_met_minutes',
            'age_end']

# Add cumulative_smoking if available
if 'cumulative_smoking_end' in agg.columns:
    agg_cols.append('cumulative_smoking_end')
elif 'cumulative_smoking_19' in agg.columns:
    agg_cols.append('cumulative_smoking_19')

df = df.merge(agg[agg_cols], on='person_id', how='left')

# Restore sex from saved map
if 'sex' not in df.columns:
    print("⚠️ 'sex' column missing after merge, restoring from saved map...")
    df = df.merge(sex_map.reset_index(), on='person_id', how='left')
    print(f"  Sex restored: {df['sex'].notna().sum()} non-null values")
else:
    print("  'sex' column preserved during merge")

# Rename cumulative_smoking_19 if needed
if 'cumulative_smoking_19' in df.columns and 'cumulative_smoking_end' not in df.columns:
    df = df.rename(columns={'cumulative_smoking_19': 'cumulative_smoking_end'})

print(f"\nMerged dataset: {df.shape}")
print(f"Patients: {df['person_id'].nunique()}")

# Debug: verify all key columns exist
print("\n### Key columns verification:")
key_cols = ['person_id', 'sex', 'age_start', 'cvd_risk_10year', 'has_cvd',
            'final_bmi', 'final_sbp_mmhg', 'cumulative_smoking_end']
for col in key_cols:
    status = "✓" if col in df.columns else "✗"
    print(f"  {status} {col}: {'present' if col in df.columns else 'MISSING'}")

# %% [markdown]
# ## 2. General Information and Summary Statistics
# 
# %%
# Risk columns and outcome columns
risk_cols = ['cvd_risk_10year', 'diabetes_risk_10year', 'stroke_risk_10year',
             'nafld_risk_10year', 'colorectal_cancer_risk_10year', 'cirrhosis_risk_10year']
outcome_cols = ['has_cvd', 'has_diabetes', 'has_stroke',
                'has_nafld', 'has_colorectal_cancer', 'has_cirrhosis']
disease_names = ['CVD', 'Diabetes', 'Stroke', 'NAFLD', 'Cancer', 'Cirrhosis']

# Descriptive statistics for risks
risk_stats = []
for risk, outcome, name in zip(risk_cols, outcome_cols, disease_names):
    if risk in df.columns and outcome in df.columns:
        risk_stats.append({
            'Disease': name,
            'Risk Column': risk,
            'Outcome Column': outcome,
            'Mean Risk': df[risk].mean(),
            'Std Risk': df[risk].std(),
            'Min Risk': df[risk].min(),
            'Max Risk': df[risk].max(),
            'Prevalence': df[outcome].mean(),
            'Risk-Prevalence Diff': abs(df[risk].mean() - df[outcome].mean())
        })

risk_stats_df = pd.DataFrame(risk_stats)

print_markdown_table(risk_stats_df.round(4), title="Risk and prevalence summary")
save_table(risk_stats_df.round(4), 'risk_summary_stats.csv', output_dir=str(OUTPUT_DIR))

# Health score statistics
if 'health_score' in df.columns:
    hs_stats = {
        'Mean': df['health_score'].mean(),
        'Std': df['health_score'].std(),
        'Min': df['health_score'].min(),
        'Max': df['health_score'].max(),
        'Median': df['health_score'].median(),
        'Q25': df['health_score'].quantile(0.25),
        'Q75': df['health_score'].quantile(0.75)
    }

    print_markdown_table(pd.Series(hs_stats).to_frame(), title="Health Score Statistics")
    save_table(pd.Series(hs_stats).to_frame(), 'health_score_stats.csv', output_dir=str(OUTPUT_DIR))

# Estimated event age
if 'estimated_event_age' in df.columns:
    ea_stats = {
        'Mean': df['estimated_event_age'].mean(),
        'Std': df['estimated_event_age'].std(),
        'Min': df['estimated_event_age'].min(),
        'Max': df['estimated_event_age'].max()
    }

    print_markdown_table(pd.Series(ea_stats).to_frame(), title="Estimated Event Age Statistics")
    save_table(pd.Series(ea_stats).to_frame(), 'estimated_event_age_stats.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 3. Mortality Statistics
# %%
print("\n### Mortality Statistics (existing)")
print(f"Estimated event age: mean={df['estimated_event_age'].mean():.1f} years")
print_markdown_table(df['primary_death_cause'].value_counts().to_frame(), title="Primary Death Cause Distribution")
save_table(df['primary_death_cause'].value_counts().to_frame(), 'primary_death_cause_distribution.csv', output_dir=str(OUTPUT_DIR))
# %%
# Visualization (optional)
if 'primary_death_cause' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    df['primary_death_cause'].value_counts().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Primary Death Cause Distribution')
    ax.set_xlabel('Cause')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure(fig, 'primary_death_cause_distribution', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
    plt.show()
# %% [markdown]
# ## 4. Validation of Risk Formulas (Against Generator Specification)
# 
# %%
validation_results = []

# Check 1: Risk ranges (clipping)
risk_ranges = {
    'cvd_risk_10year': (0, 0.5),
    'diabetes_risk_10year': (0, 0.5),
    'stroke_risk_10year': (0, 0.4),
    'nafld_risk_10year': (0, 0.6),
    'colorectal_cancer_risk_10year': (0, 0.3),
    'cirrhosis_risk_10year': (0, 0.2)
}

for col, (low, high) in risk_ranges.items():
    if col in df.columns:
        in_range = ((df[col] >= low) & (df[col] <= high)).mean()
        validation_results.append({
            'Check': f'{col} range',
            'Expected': f'100% in [{low}, {high}]',
            'Actual': f'{in_range:.1%} in range',
            'Status': 'OK' if in_range > 0.99 else 'WARN'
        })

# Check 2: Risk-Prevalence alignment (mean risk ≈ prevalence)
for risk, outcome, name in zip(risk_cols, outcome_cols, disease_names):
    if risk in df.columns and outcome in df.columns:
        diff = abs(df[risk].mean() - df[outcome].mean())
        validation_results.append({
            'Check': f'{name} risk-prevalence',
            'Expected': 'Diff < 0.05',
            'Actual': f'Diff = {diff:.3f}',
            'Status': 'OK' if diff < 0.05 else 'WARN'
        })

# Check 3: Health score range
if 'health_score' in df.columns:
    hs_in_range = ((df['health_score'] >= 0) & (df['health_score'] <= 100)).mean()
    validation_results.append({
        'Check': 'health_score range',
        'Expected': '100% in [0, 100]',
        'Actual': f'{hs_in_range:.1%} in range',
        'Status': 'OK' if hs_in_range > 0.99 else 'WARN'
    })

# Check 4: Genetic risk correlation (from baseline)
genetic_cols = ['genetic_risk_cvd', 'genetic_risk_diabetes', 'genetic_risk_stroke']
risk_targets = ['cvd_risk_10year', 'diabetes_risk_10year', 'stroke_risk_10year']
for gcol, rcol in zip(genetic_cols, risk_targets):
    if gcol in df.columns and rcol in df.columns:
        corr = df[gcol].corr(df[rcol])
        validation_results.append({
            'Check': f'{gcol} → {rcol}',
            'Expected': 'Positive correlation > 0.1',
            'Actual': f'Correlation = {corr:.3f}',
            'Status': 'OK' if corr > 0.1 else 'WARN'
        })

val_df = pd.DataFrame(validation_results)
print("\n### Risk formula validation")
print_markdown_table(val_df, title="Risk formula validation")
save_table(val_df, 'risk_validation_check.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 5. Risk Distributions by Demographics
# 
# %%
# By sex (with check)
print("\n### Risk differences by sex (t-test)")
if 'sex' in df.columns:
    sex_test_results = []
    for risk, name in zip(risk_cols, disease_names):
        if risk in df.columns:
            m_vals = df[df['sex'] == 'M'][risk].dropna()
            f_vals = df[df['sex'] == 'F'][risk].dropna()
            if len(m_vals) > 0 and len(f_vals) > 0:
                t_stat, p_val = stats.ttest_ind(m_vals, f_vals, equal_var=False)
                sex_test_results.append({
                    'Disease': name,
                    'Mean Male': m_vals.mean(),
                    'Mean Female': f_vals.mean(),
                    't-statistic': t_stat,
                    'p-value': p_val,
                    'Significant': 'Yes' if p_val < 0.05 else 'No'
                })

    sex_test_df = pd.DataFrame(sex_test_results)
    print_markdown_table(sex_test_df.round(4), title="Risk differences by sex (t-test)")
    save_table(sex_test_df.round(4), 'risk_by_sex_ttest.csv', output_dir=str(OUTPUT_DIR))

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (risk, name) in enumerate(zip(risk_cols, disease_names)):
        ax = axes[i]
        df.boxplot(column=risk, by='sex', ax=ax)
        ax.set_title(f'{name} risk by sex')
        ax.set_xlabel('Sex')
        ax.set_ylabel('Risk')
        plt.suptitle('')
    plt.tight_layout()
    save_figure(fig, 'risk_by_sex_boxplot', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
    plt.show()
else:
    print("⚠️ Skipping sex analysis: 'sex' column not available")
    sex_test_df = pd.DataFrame()
    save_table(sex_test_df, 'risk_by_sex_ttest.csv', output_dir=str(OUTPUT_DIR))

# By age quintiles
df['age_quintile'] = pd.qcut(df['age_start'], q=5,
                             labels=['Q1 (youngest)', 'Q2', 'Q3', 'Q4', 'Q5 (oldest)'],
                             duplicates='drop')

print("\n### Risk differences by age quintiles (ANOVA)")
age_test_results = []
for risk, name in zip(risk_cols, disease_names):
    if risk in df.columns:
        groups = [df[df['age_quintile'] == q][risk].dropna()
                  for q in df['age_quintile'].unique()]
        if all(len(g) > 0 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            age_test_results.append({
                'Disease': name,
                'F-statistic': f_stat,
                'p-value': p_val,
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })

age_test_df = pd.DataFrame(age_test_results)
print_markdown_table(age_test_df.round(4), title="Risk differences by age quintiles (ANOVA)")
save_table(age_test_df.round(4), 'risk_by_age_anova.csv', output_dir=str(OUTPUT_DIR))
# %% [markdown]
# ## 6. Calibration Validation (Quantitative Metrics)
# 
# %%
print("\n### Calibration metrics for all risks")
calibration_results = []

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (risk, outcome, name) in enumerate(zip(risk_cols, outcome_cols, disease_names)):
    if risk in df.columns and outcome in df.columns:
        # Calibration curve
        prob_true, prob_pred = calibration_curve(df[outcome], df[risk], n_bins=10)

        # Brier score
        brier = brier_score_loss(df[outcome], df[risk])

        # E/O ratio
        expected = df[risk].sum()
        observed = df[outcome].sum()
        eo_ratio = expected / observed if observed > 0 else np.nan

        # Calibration intercept & slope (simplified)
        calib_data = pd.DataFrame({'pred': df[risk], 'actual': df[outcome]})
        calib_data['bin'] = pd.qcut(calib_data['pred'], q=5, duplicates='drop')
        bin_means = calib_data.groupby('bin').agg({'pred': 'mean', 'actual': 'mean'})
        if len(bin_means) >= 2:
            slope, intercept, r_val, p_val, std_err = stats.linregress(
                bin_means['pred'], bin_means['actual']
            )
        else:
            slope, intercept = np.nan, np.nan

        calibration_results.append({
            'Disease': name,
            'Brier Score': brier,
            'E/O Ratio': eo_ratio,
            'Calibration Slope': slope,
            'Calibration Intercept': intercept,
            'Mean Predicted Risk': df[risk].mean(),
            'Observed Prevalence': df[outcome].mean()
        })

        # Plot
        ax = axes[i]
        ax.plot(prob_pred, prob_true, 's-', label='Model', markersize=8)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax.set_xlabel('Mean predicted risk')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(f'{name}\nBrier={brier:.3f}, E/O={eo_ratio:.2f}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, 'risk_calibration_all', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()

calib_df = pd.DataFrame(calibration_results)
print_markdown_table(calib_df.round(4), title="Calibration metrics for all risks")
save_table(calib_df.round(4), 'calibration_metrics.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 7. Correlation Analysis
# 
# %%
# Correlation matrix: risks + features
corr_cols = risk_cols + ['age_start', 'final_bmi', 'final_sbp_mmhg', 'final_hdl_mgdl',
                         'final_hba1c_percent', 'avg_alcohol_g_per_week',
                         'avg_total_met_minutes', 'cumulative_smoking_end', 'health_score']
corr_cols = [c for c in corr_cols if c in df.columns]

corr_matrix = df[corr_cols].corr()

# Heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Correlation matrix: risks and features')
plt.tight_layout()
save_figure(fig, 'risk_correlation_heatmap', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()

save_table(corr_matrix, 'risk_correlation_matrix.csv', output_dir=str(OUTPUT_DIR))

# Key correlations table
key_corrs = []
for risk in risk_cols:
    for feat in ['age_start', 'final_bmi', 'final_sbp_mmhg', 'cumulative_smoking_end']:
        if risk in df.columns and feat in df.columns:
            corr = df[risk].corr(df[feat])
            key_corrs.append({
                'Risk': risk,
                'Feature': feat,
                'Correlation': corr
            })

key_corr_df = pd.DataFrame(key_corrs)
print("\n### Key risk-feature correlations")
print_markdown_table(key_corr_df.round(3), title="Key risk-feature correlations")
save_table(key_corr_df.round(3), 'key_risk_correlations.csv', output_dir=str(OUTPUT_DIR))
# %% [markdown]
# ## 8. Prediction: CVD (Detailed Analysis)
# 
# %%
# Prepare features
feature_cols = ['age_start', 'final_bmi', 'final_hdl_mgdl', 'final_sbp_mmhg',
                'final_hba1c_percent', 'avg_alcohol_g_per_week',
                'avg_total_met_minutes']

# Add cumulative_smoking_end if available
if 'cumulative_smoking_end' in df.columns:
    feature_cols.append('cumulative_smoking_end')

# Drop missing
df_clean = df[feature_cols + ['has_cvd', 'cvd_risk_10year']].dropna()
X = df_clean[feature_cols]
y_binary = df_clean['has_cvd']
y_risk = df_clean['cvd_risk_10year']

print(f"\n### CVD Prediction Dataset")
print(f"Clean data: {len(df_clean)} patients")
print(f"Features: {feature_cols}")

# Train/test split
X_train, X_test, y_train, y_test, y_risk_train, y_risk_test = train_test_split(
    X, y_binary, y_risk, test_size=0.2, random_state=42, stratify=y_binary
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ### Model Comparison
# 
# %%
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}

results = {}
for name, model in models.items():
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    results[name] = {
        'AUC-ROC': auc, 'Avg Precision': avg_precision,
        'Brier': brier, 'model': model
    }
    print(f"\n{name}:")
    print(f"  AUC-ROC: {auc:.3f}")
    print(f"  Avg Precision: {avg_precision:.3f}")
    print(f"  Brier Score: {brier:.3f}")
# %%
# Summary table
model_summary = []
for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
    if name in results:
        model_summary.append({
            'Model': name,
            'AUC-ROC': results[name]['AUC-ROC'],
            'Avg Precision': results[name]['Avg Precision'],
            'Brier Score': results[name]['Brier']
        })

model_summary_df = pd.DataFrame(model_summary)

print_markdown_table(model_summary_df.round(3), title="Model comparison summary")
save_table(model_summary_df.round(3), 'cvd_model_comparison.csv', output_dir=str(OUTPUT_DIR))
# %%
# ROC curves
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(8, 6))
for name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:
    if name in results:
        model = results[name]['model']
        if 'Logistic' in name:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, y_pred_proba, name=name, alpha=0.8, ax=ax)

ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
ax.set_title('ROC curves for CVD prediction')
ax.legend()
plt.tight_layout()
save_figure(fig, 'cvd_roc_curves', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()
# %% [markdown]
# ### Feature Importance (All Models)
# 
# %%
# Random Forest
rf_model = results['Random Forest']['model']
rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
# %%
# Gradient Boosting
gb_model = results['Gradient Boosting']['model']
gb_importance = pd.Series(gb_model.feature_importances_, index=feature_cols).sort_values()
# %%
# Logistic Regression coefficients
lr_model = results['Logistic Regression']['model']
lr_coef = pd.Series(lr_model.coef_[0], index=feature_cols).sort_values()
# %%
# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
rf_importance.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Random Forest Importance')
gb_importance.plot(kind='barh', ax=axes[1], color='green')
axes[1].set_title('Gradient Boosting Importance')
lr_coef.plot(kind='barh', ax=axes[2], color='coral')
axes[2].set_title('Logistic Regression Coefficients')
plt.tight_layout()
save_figure(fig, 'cvd_feature_importance_all', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()
# %%
# Save to CSV
save_table(pd.DataFrame({
    'RF Importance': rf_importance,
    'GB Importance': gb_importance,
    'LR Coef': lr_coef
}), 'cvd_feature_importance.csv', output_dir=str(OUTPUT_DIR))
# %% [markdown]
# ## 9. Prediction: All 6 Diseases (Summary)
# 
# %%
# AUC-ROC for all diseases (Random Forest)
all_disease_results = []

for risk, outcome, name in zip(risk_cols, outcome_cols, disease_names):
    if outcome in df.columns:
        # Features (same for all)
        feat_cols = [c for c in feature_cols if c in df.columns]
        data = df[feat_cols + [outcome]].dropna()
        if len(data) < 100:
            continue

        X_d = data[feat_cols]
        y_d = data[outcome]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
        )

        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, y_pred)
        avg_prec = average_precision_score(y_te, y_pred)
        prevalence = y_d.mean()

        all_disease_results.append({
            'Disease': name,
            'Prevalence': f'{prevalence:.1%}',
            'AUC-ROC': round(auc, 3),
            'Avg Precision': round(avg_prec, 3),
            'N samples': len(data)
        })

all_disease_df = pd.DataFrame(all_disease_results)
print_markdown_table(all_disease_df, title="AUC-ROC for all diseases (Random Forest)")
save_table(all_disease_df, 'all_diseases_auc.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 10. Comparison with Original Risks (SCORE2-like)
# %%
# How well do ML predictions match original risk scores?
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (name, result) in enumerate(list(results.items())[:3]):
    model = result['model']
    if 'Logistic' in name:
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    ax = axes[i]
    ax.scatter(y_risk_test, y_pred_proba, alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect match')
    ax.set_xlabel('Original risk score')
    ax.set_ylabel(f'{name} predicted probability')
    corr = np.corrcoef(y_risk_test, y_pred_proba)[0, 1]
    ax.set_title(f'{name}\nCorrelation: {corr:.3f}')
    ax.legend()

plt.tight_layout()
save_figure(fig, 'model_vs_original_risk', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()
# %% [markdown]
# ## 11. Multiclass Prediction: Any Disease
# %%
# Create multiclass target
def get_primary_condition(row):
    conditions = ['has_cvd', 'has_diabetes', 'has_stroke',
                  'has_nafld', 'has_colorectal_cancer', 'has_cirrhosis']
    for i, cond in enumerate(conditions, 1):
        if row.get(cond, 0) == 1:
            return i
    return 0
# %%
df['primary_condition'] = df.apply(get_primary_condition, axis=1)
print("\n### Primary condition distribution")
print_markdown_table(df['primary_condition'].value_counts().sort_index().to_frame(), title="Primary condition distribution")
save_table(df['primary_condition'].value_counts().sort_index().to_frame(), 'primary_condition_dist.csv', output_dir=str(OUTPUT_DIR))
# %%
# Multiclass model
X_multi = df_clean[feature_cols]
y_multi = df['primary_condition'].loc[X_multi.index]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)
# %%
gb_multi = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_multi.fit(X_train_m, y_train_m)
y_pred_m = gb_multi.predict(X_test_m)
# %%
print("\n### Multiclass classification report")
report = classification_report(
    y_test_m, y_pred_m,
    target_names=['None', 'CVD', 'Diabetes', 'Stroke', 'NAFLD', 'Cancer', 'Cirrhosis'],
    output_dict=True
)
# %%
print(classification_report(
    y_test_m, y_pred_m,
    target_names=['None', 'CVD', 'Diabetes', 'Stroke', 'NAFLD', 'Cancer', 'Cirrhosis']
))
# %%
# Save report
save_table(pd.DataFrame(report).T, 'multiclass_classification_report.csv', output_dir=str(OUTPUT_DIR))
# %%
# Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test_m, y_pred_m)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['None', 'CVD', 'Diabetes', 'Stroke', 'NAFLD', 'Cancer', 'Cirrhosis'],
            yticklabels=['None', 'CVD', 'Diabetes', 'Stroke', 'NAFLD', 'Cancer', 'Cirrhosis'])
ax.set_title('Confusion matrix: multiclass disease prediction')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
save_figure(fig, 'multiclass_confusion_matrix', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()
# %% [markdown]
# ## 12. Cross-Dataset Consistency Check
# %%
# Check person_id consistency between 04 and 05
ids_04 = set(risks['person_id'].unique())
ids_05 = set(agg['person_id'].unique())
common_ids = ids_04.intersection(ids_05)

consistency_check = {
    'IDs in 04_health_risks': len(ids_04),
    'IDs in 05_aggregated': len(ids_05),
    'Common IDs': len(common_ids),
    'Consistency': 'OK' if len(common_ids) == len(ids_04) == len(ids_05) else 'WARN'
}

print_markdown_table(pd.Series(consistency_check).to_frame(), title="Cross-dataset consistency")
save_table(pd.Series(consistency_check).to_frame(), 'cross_dataset_consistency.csv', output_dir=str(OUTPUT_DIR))
# %% [markdown]
# ## 13. Summary and Conclusions
# 
# After comprehensive validation of the health risks module (04_health_risks) and comparison with baseline models (notebook 1), we obtain the following refined insights:
# 
# 1. **Risk formula validation**
#    - **All 6 risk ranges** pass hard clipping verification (100% within specified bounds).
#    - **Risk-prevalence alignment** is excellent: maximum difference < 0.005.
#    - **Health Score** distribution reflects appropriate weighting of the 6 disease risks.
#    - **Genetic risk correlations** are positive and significant (r > 0.1).
# 
# 2. **Calibration quality**
#    - **Brier scores** range from 0.126 (Diabetes) to 0.211 (Cancer).
#    - **E/O ratios** are remarkably close to 1.0: range 0.991–1.022.
#    - **Calibration slopes** near 1.0 for CVD, Diabetes, Stroke, NAFLD.
# 
# 3. **Demographic stratification**
#    - **Sex differences** are statistically significant for CVD, Stroke, and NAFLD.
#    - **Age effects** (ANOVA p < 0.001) are strong for CVD, Stroke, Diabetes, and Cancer.
# 
# 4. **Predictive modeling (CVD focus)**
#    - **Best model**: Logistic Regression achieves AUC-ROC = 0.651.
#    - **Top features**: final_sbp_mmhg, final_hdl_mgdl, avg_total_met_minutes.
#    - **Comparison with baseline (notebook 1)**: AUC improved from 0.549 to 0.651 (+0.102).
# 
# 5. **Multiclass disease prediction**
#    - **7-class classification** achieves Macro F1 = 0.16, Accuracy = 0.27.
#    - **Best recall**: CVD (0.61), None (0.44).
# 
# 6. **Cross-dataset consistency**
#    - **Person ID alignment**: 100% match between 04_health_risks and 05_aggregated_dataset.
# 
# ## Output Files
# 
# All generated tables and figures have been saved to:
# - `output/04_health_risks/` — CSV tables and PNG figures
# - `logs/04_health_risks_*.txt` — Complete execution log
# 
# Files can be passed to an LLM for further discussion.
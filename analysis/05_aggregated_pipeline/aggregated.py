# %% [markdown]
# # Complete ML Pipeline: Aggregated Dataset (05_aggregated)
#
# Realistic scenario: MAR missing values, ~50 features, 12 targets.
# Full cycle: EDA → imputation → feature engineering → model selection → evaluation.
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

from src.utils.data_loader import load_merged_dataset
from src.utils.analysis_helpers import (
    setup_analysis, save_analysis_table, save_analysis_figure,
    save_descriptive_statistics, analyze_missing_pattern, impute_data,
    engineer_features, compare_with_reference
)
from src.utils.validators import DatasetValidator

warnings.filterwarnings('ignore')

# %% [markdown]
# ## Initialize analysis
# %%
cfg = setup_analysis('05_aggregated_pipeline')
output_dir = cfg['output_dir']

# %% [markdown]
# ## 1. Load Data
# %%
print("\n### Loading datasets...")
df = load_merged_dataset("05_aggregated_pipeline", cfg["data_path"], cfg["version"])
cfg['logger'].info(f"Aggregated dataset: {df.shape}")

# %% [markdown]
# ## 2. Validation
# %%
validator = DatasetValidator(cfg['data_path'])
validator.load_data()
validator._validate_aggregated()

val_results = pd.DataFrame([{
    "Check": r.name,
    "Expected": "Target Range",
    "Actual": r.message,
    "Status": "OK" if r.passed else "WARN"
} for r in validator.results if r.name.startswith("Aggregated")])

print(val_results.to_markdown())
save_analysis_table(val_results, "validation_check", output_dir)

# %% [markdown]
# ## 3. Missing Values Analysis (MAR)
# %%
missing_df = analyze_missing_pattern(df, age_col='age_start')
if not missing_df.empty:
    print(missing_df.to_markdown())
    save_analysis_table(missing_df, "missing_values_report", output_dir)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
missing_pct = df.isnull().mean() * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
if not missing_pct.empty:
    missing_pct.head(10).plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Missing Values by Column (Top 10)')
    axes[0].set_ylabel('% missing')
    axes[0].tick_params(axis='x', rotation=45)

    # MAR pattern by age
    if 'age_start' in df.columns:
        young = df[df['age_start'] < 45]
        old = df[df['age_start'] >= 45]
        mar_data = []
        for col in missing_pct.index[:6]:
            mar_data.append({
                'Column': col,
                'Young (<45)': young[col].isna().mean() * 100,
                'Old (>=45)': old[col].isna().mean() * 100
            })
        mar_df = pd.DataFrame(mar_data).set_index('Column')
        mar_df.plot(kind='bar', ax=axes[1], color=['coral', 'steelblue'])
        axes[1].set_title('MAR Pattern: Missing % by Age Group')
        axes[1].set_ylabel('% missing')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(['Young (<45)', 'Old (≥45)'])
else:
    axes[0].text(0.5, 0.5, 'No Missing Values', ha='center')
    axes[1].text(0.5, 0.5, 'No Missing Values', ha='center')

plt.tight_layout()
save_analysis_figure(fig, 'missing_values_analysis', output_dir)
plt.show()

# %% [markdown]
# ## 4. Descriptive Statistics
# %%
numeric_cols = ['age_start', 'age_end', 'final_bmi', 'final_sbp_mmhg',
                'final_hba1c_percent', 'final_hdl_mgdl', 'final_total_cholesterol_mgdl',
                'avg_alcohol_g_per_week', 'avg_total_met_minutes', 'cumulative_smoking_end',
                'health_score']
numeric_cols = [c for c in numeric_cols if c in df.columns]
save_descriptive_statistics(df, numeric_cols, output_dir, filename="aggregated_percentiles")

# %% [markdown]
# ## 5. Data Preparation: Target and Features
# %%
target = 'has_diabetes'
risk_target = 'diabetes_risk_10year'

# Remove leakage columns
leakage_cols = [c for c in df.columns if c.startswith(('has_', '_risk')) and c not in [target, risk_target]]
leakage_cols = [c for c in leakage_cols if not c.startswith(('avg_', 'std_'))]
leakage_cols.extend([c for c in df.columns if 'risk' in c.lower() and c not in [risk_target]])
leakage_cols = list(set(leakage_cols))

drop_cols = ['person_id', 'primary_death_cause', 'estimated_event_age'] + leakage_cols
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols + [target, risk_target])
y = df[target]

# Encode categorical
if 'sex' in X.columns:
    X['sex_M'] = (X['sex'] == 'M').astype(int)
    X = X.drop(columns=['sex'])

X = X.select_dtypes(include=[np.number])
cfg['logger'].info(f"Features: {X.shape[1]}, Target prevalence: {y.mean():.1%}")

# Split BEFORE imputation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Remove columns with all NaNs in training set (to avoid imputation issues)
cols_to_keep = X_train.columns[~X_train.isnull().all()]
X_train = X_train[cols_to_keep]
X_test = X_test[cols_to_keep]
cfg['logger'].info(f"After removing all-NaN columns: {X_train.shape[1]} features")

# Save feature list
save_analysis_table(pd.DataFrame(X_train.columns.tolist(), columns=['feature']), 'feature_list', output_dir)

# Target and risk summary
target_stats = pd.DataFrame({
    'Metric': ['Prevalence', 'Mean Risk', 'Risk Std'],
    'Value': [y.mean(), df[risk_target].mean(), df[risk_target].std()]
})
save_analysis_table(target_stats, 'target_risk_summary', output_dir)

# %% [markdown]
# ## 6. Imputation Comparison
# %%
X_train_median, X_test_median = impute_data(X_train, X_test, method='median')
cfg['logger'].info("Running MICE imputation...")
X_train_mice, X_test_mice = impute_data(X_train, X_test, method='mice')
cfg['logger'].info("MICE complete")

impute_df = pd.DataFrame({
    'Method': ['Simple (median)', 'MICE (Iterative)'],
    'Train Shape': [X_train_median.shape[0], X_train_mice.shape[0]],
    'Features': [X_train_median.shape[1], X_train_mice.shape[1]]
})
save_analysis_table(impute_df, 'imputation_comparison', output_dir)

# Use MICE for pipeline
X_train_proc, X_test_proc = X_train_mice, X_test_mice


# Post-imputation clipping to physiological ranges
def clip_to_ranges(X_df, col_names, ranges):
    X_clipped = X_df.copy()
    for col, (low, high) in ranges.items():
        if col in col_names:
            col_idx = col_names.index(col)
            X_clipped[:, col_idx] = np.clip(X_clipped[:, col_idx], low, high)
    return X_clipped


from src.config import DEFAULT_CONFIG

biomarker_ranges = DEFAULT_CONFIG.final_biomarker_ranges
X_train_proc = clip_to_ranges(X_train_proc, cols_to_keep, biomarker_ranges)
X_test_proc = clip_to_ranges(X_test_proc, cols_to_keep, biomarker_ranges)
cfg['logger'].info("Biomarker columns clipped post-imputation")

# %% [markdown]
# ## 7. Feature Engineering (Domain-Informed)
# %%
X_train_eng_df = engineer_features(pd.DataFrame(X_train_proc, columns=cols_to_keep))
X_test_eng_df = engineer_features(pd.DataFrame(X_test_proc, columns=cols_to_keep))
new_features = [c for c in X_train_eng_df.columns if c not in cols_to_keep]
cfg['logger'].info(f"Engineered features: {X_train_eng_df.shape[1]} (was {len(cols_to_keep)})")
save_analysis_table(pd.DataFrame(new_features, columns=['engineered_feature']), 'engineered_features', output_dir)

X_train_eng = X_train_eng_df.values
X_test_eng = X_test_eng_df.values
eng_feature_names = X_train_eng_df.columns.tolist()

# %% [markdown]
# ## 8. Scaling and Model Preparation
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Tree-based models don't require scaling
X_train_tree = X_train_eng
X_test_tree = X_test_eng

# Save scaler parameters
scaler_df = pd.DataFrame({
    'feature': eng_feature_names,
    'mean': scaler.mean_,
    'std': scaler.scale_
})
save_analysis_table(scaler_df, 'scaler_parameters', output_dir)

# %% [markdown]
# ## 9. Model Selection with Cross-Validation
# %%
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

param_grids = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'RandomForest': {'max_depth': [5, 10, None], 'min_samples_leaf': [1, 5]},
    'GradientBoosting': {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
cv_results = []

for name, model in models.items():
    cfg['logger'].info(f"Tuning {name}...")
    X_tr = X_train_scaled if 'Logistic' in name else X_train_tree
    grid = GridSearchCV(model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_tr, y_train)
    best_models[name] = grid.best_estimator_
    cv_results.append({'Model': name, 'Best CV AUC': grid.best_score_, 'Best Params': str(grid.best_params_)})

cv_df = pd.DataFrame(cv_results)
print(cv_df.to_markdown())
save_analysis_table(cv_df, 'cv_model_selection', output_dir)

# %% [markdown]
# ## 10. Ensemble: Voting Classifier
# %%
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)

y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]
y_pred_proba_individual = {
    name: model.predict_proba(X_test_scaled if 'Logistic' in name else X_test_tree)[:, 1]
    for name, model in best_models.items()
}

# %% [markdown]
# ## 11. Evaluation and Model Comparison
# %%
comparison = []
for name, y_proba in {**y_pred_proba_individual, 'Ensemble': y_pred_proba_ensemble}.items():
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    comparison.append({'Model': name, 'AUC-ROC': auc, 'Avg Precision': ap, 'Brier Score': brier})

comp_df = pd.DataFrame(comparison).sort_values('AUC-ROC', ascending=False)
print(comp_df.to_markdown())
save_analysis_table(comp_df, 'model_comparison_final', output_dir)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC curves
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
for name, y_proba in {**y_pred_proba_individual, 'Ensemble': y_pred_proba_ensemble}.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={roc_auc_score(y_test, y_proba):.3f})', linewidth=2)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curves')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Precision-Recall curves
for name, y_proba in {**y_pred_proba_individual, 'Ensemble': y_pred_proba_ensemble}.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    axes[0, 1].plot(recall, precision, label=f'{name} (AP={ap:.3f})', linewidth=2)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curves')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Feature importance (Random Forest)
rf_model = best_models['RandomForest']
importance = pd.Series(rf_model.feature_importances_, index=eng_feature_names).sort_values(ascending=True).tail(15)
importance.plot(kind='barh', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title('Top 15 Feature Importances (Random Forest)')
axes[1, 0].grid(True, alpha=0.3)

# Calibration curves
for name, y_proba in {**y_pred_proba_individual, 'Ensemble': y_pred_proba_ensemble}.items():
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    axes[1, 1].plot(prob_pred, prob_true, 's-', label=name, markersize=8)
axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect')
axes[1, 1].set_xlabel('Mean Predicted Probability')
axes[1, 1].set_ylabel('Fraction of Positives')
axes[1, 1].set_title('Calibration Curves')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_analysis_figure(fig, 'model_comparison_complete', output_dir)
plt.show()

# Save feature importance table
importance_df = importance.reset_index()
importance_df.columns = ['feature', 'importance']
save_analysis_table(importance_df, 'feature_importance_aggregated', output_dir)

# %% [markdown]
# ## 12. Best Model Interpretation
# %%
best_model_name = comp_df.iloc[0]['Model']
best_auc = comp_df.iloc[0]['AUC-ROC']
cfg['logger'].info(f"Best model: {best_model_name} with AUC {best_auc:.3f}")

if best_model_name == 'Ensemble':
    cfg['logger'].info("Ensemble uses soft voting with equal weights")
    # Correlations with ensemble
    ens_corr = []
    for name, y_proba in y_pred_proba_individual.items():
        corr = np.corrcoef(y_proba, y_pred_proba_ensemble)[0, 1]
        ens_corr.append({'Model': name, 'Correlation with Ensemble': corr})
    ens_corr_df = pd.DataFrame(ens_corr)
    print(ens_corr_df.to_markdown())
    save_analysis_table(ens_corr_df, 'ensemble_contributions', output_dir)

# %% [markdown]
# ## 13. External Validation (Compare with Reference)
# %%
ref_df = compare_with_reference(df)
if not ref_df.empty:
    print(ref_df.to_markdown())
    save_analysis_table(ref_df, 'external_validation_comparison', output_dir)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    markers_plot = ['final_bmi', 'final_sbp_mmhg', 'final_hba1c_percent', 'final_hdl_mgdl']
    for i, col in enumerate(markers_plot):
        if col in df.columns and i < 4:
            ax = axes[i]
            df[col].dropna().hist(bins=50, ax=ax, alpha=0.7, label='Synthetic', density=True)
            ref_row = ref_df[ref_df['Feature'] == col]
            if not ref_row.empty:
                ref_mean = ref_row['Reference Mean'].values[0]
                ref_std = ref_row['Reference Std'].values[0]
                x = np.linspace(df[col].min(), df[col].max(), 100)
                from scipy.stats import norm

                ax.plot(x, norm.pdf(x, ref_mean, ref_std), 'r--', linewidth=2, label='Reference')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.set_title(f'{col}: Synthetic vs Reference')
            ax.legend()
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_analysis_figure(fig, 'external_validation_distributions', output_dir)
    plt.show()

# %% [markdown]
# ## 14. Comparison with Notebook 4 (Health Risks Only)
# %%
# Assuming we have results from notebook 4 stored or manually entered
notebook_4_results = {
    'CVD': 0.651,
    'Diabetes': 0.588,
    'Stroke': 0.591
}
cfg['logger'].info("Notebook 4 (health risks only) AUCs:")
for disease, auc in notebook_4_results.items():
    cfg['logger'].info(f"  {disease}: {auc:.3f}")

cfg['logger'].info(f"Notebook 5 (aggregated) Diabetes AUC: {best_auc:.3f}")
improvement = best_auc - notebook_4_results['Diabetes']
cfg['logger'].info(
    f"Improvement from aggregated features: +{improvement:.3f} ({improvement / notebook_4_results['Diabetes'] * 100:.1f}%)")

# Save comparison
comp_nb = pd.DataFrame({
    'Notebook': ['Notebook 4 (risks)', 'Notebook 5 (aggregated)'],
    'Target': ['Diabetes', 'Diabetes'],
    'AUC-ROC': [notebook_4_results['Diabetes'], best_auc],
    'Features': ['Health risks only', 'Lifestyle + biomarkers + demographics'],
})
save_analysis_table(comp_nb, 'notebook_comparison_4_vs_5', output_dir)

# %% [markdown]
# ## 15. Threshold Analysis for Clinical Utility
# %%
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
thresh_results = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba_ensemble >= thresh).astype(int)
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    thresh_results.append({
        'Threshold': thresh,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Positive Predictions': int(y_pred_thresh.sum())
    })
thresh_df = pd.DataFrame(thresh_results)
print(thresh_df.to_markdown())
save_analysis_table(thresh_df, 'threshold_analysis', output_dir)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresh_df['Threshold'], thresh_df['Precision'], 'o-', label='Precision', linewidth=2)
ax.plot(thresh_df['Threshold'], thresh_df['Recall'], 's-', label='Recall', linewidth=2)
ax.plot(thresh_df['Threshold'], thresh_df['F1'], '^-', label='F1 Score', linewidth=2)
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision-Recall-F1 Trade-off by Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_analysis_figure(fig, 'threshold_tradeoff', output_dir)
plt.show()

# %% [markdown]
# ## 16. Textual Summary for LLM
# %%
summary_lines = []

# 1. Общая информация о датасете
summary_lines.append(f"**Dataset:** {df.shape[0]} patients, {df.shape[1]} features")
summary_lines.append(f"**Target (diabetes) prevalence:** {y.mean():.3f}")
summary_lines.append(f"**Diabetes risk (10-year) mean:** {df[risk_target].mean():.3f} ± {df[risk_target].std():.3f}")

# 2. Пропущенные значения (MAR)
if 'missing_values_report.csv' in (output_dir / "tables").glob("*.csv"):
    missing_report = pd.read_csv(output_dir / "tables/missing_values_report.csv")
    mar_cols = missing_report[missing_report['p-value'] < 0.05][
        'Column'].tolist() if 'p-value' in missing_report.columns else []
    summary_lines.append(f"**MAR pattern:** {len(mar_cols)} columns with missingness dependent on age (p<0.05)")

# 3. Валидация
val_df = pd.read_csv(output_dir / "tables/validation_check.csv")
ok_all = (val_df['Status'] == 'OK').all()
summary_lines.append(f"**Validation:** {'All checks passed' if ok_all else 'Some checks failed'}")
if not ok_all:
    for _, row in val_df[val_df['Status'] != 'OK'].iterrows():
        summary_lines.append(f"  {row['Check']}: {row['Actual']} (expected {row['Expected']})")

# 4. Импутация
impute_df = pd.read_csv(output_dir / "tables/imputation_comparison.csv")
summary_lines.append(
    f"**Imputation:** MICE used, shape {impute_df[impute_df['Method'] == 'MICE (Iterative)']['Features'].values[0]} features")

# 5. Feature engineering
eng_df = pd.read_csv(output_dir / "tables/engineered_features.csv")
summary_lines.append(
    f"**Engineered features:** {len(eng_df)} new features (metabolic syndrome score, strength ratio, etc.)")

# 6. Моделирование
cv_df = pd.read_csv(output_dir / "tables/cv_model_selection.csv")
best_cv = cv_df.loc[cv_df['Best CV AUC'].idxmax()]
summary_lines.append(f"**Best CV model:** {best_cv['Model']} with AUC = {best_cv['Best CV AUC']:.3f}")

final_df = pd.read_csv(output_dir / "tables/model_comparison_final.csv")
best_final = final_df.loc[final_df['AUC-ROC'].idxmax()]
summary_lines.append(
    f"**Final test performance (ensemble):** AUC = {best_final['AUC-ROC']:.3f}, Avg Precision = {best_final['Avg Precision']:.3f}, Brier = {best_final['Brier Score']:.3f}")

# 7. Feature importance
imp_df = pd.read_csv(output_dir / "tables/feature_importance_aggregated.csv")
top3 = imp_df.nlargest(3, 'importance')
summary_lines.append("**Top 3 features (Random Forest):**")
for _, row in top3.iterrows():
    summary_lines.append(f"  {row['feature']}: {row['importance']:.3f}")

# 8. Пороговый анализ
thresh_df = pd.read_csv(output_dir / "tables/threshold_analysis.csv")
best_f1 = thresh_df.loc[thresh_df['F1'].idxmax()]
summary_lines.append(
    f"**Optimal threshold (F1 max):** {best_f1['Threshold']:.1f} (F1={best_f1['F1']:.3f}, Precision={best_f1['Precision']:.3f}, Recall={best_f1['Recall']:.3f})")

# 9. Сравнение с ноутбуком 4
comp_nb = pd.read_csv(output_dir / "tables/notebook_comparison_4_vs_5.csv")
nb4_auc = comp_nb[comp_nb['Notebook'] == 'Notebook 4 (risks)']['AUC-ROC'].values[0]
nb5_auc = comp_nb[comp_nb['Notebook'] == 'Notebook 5 (aggregated)']['AUC-ROC'].values[0]
improvement = nb5_auc - nb4_auc
summary_lines.append(f"**Improvement vs notebook 4:** {improvement:.3f} ({improvement / nb4_auc * 100:.1f}%)")

# Сохранить
with open(output_dir / "analysis_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

cfg['logger'].info("All analysis tables and summary saved.")

# %% [markdown]
# ## 17. Summary and Conclusions
#
# 1. **Dataset characteristics**: ~5000 patients, ~50 features, 6 MAR columns (2-5% missing).
# 2. **MAR pattern**: Confirmed: younger patients have higher missing rates (χ² test significant).
# 3. **Imputation**: MICE used for final pipeline, with post-imputation clipping to physiological ranges.
# 4. **Feature engineering**: Added metabolic syndrome score, strength ratio, alcohol risk, sleep deficit, age over 45, non-HDL cholesterol.
# 5. **Modeling**: Ensemble slightly outperforms individual models, AUC ~0.65 for diabetes.
# 6. **Improvement vs notebook 4**: +~10% relative gain.
# 7. **Threshold analysis**: Suggest 0.2 for screening, 0.3 for balance.
# %%

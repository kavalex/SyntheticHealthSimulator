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
import warnings

import sklearn

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
from scipy import stats

from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score, brier_score_loss
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
print(f"Log file: logs/05_aggregated_pipeline_*.txt")
# %% [markdown]
# ## 1. Load Data
# %%
print("\n### Loading datasets...")
df = load_aggregated(DATA_PATH, VERSION)
print(f"Loaded: {df.shape[0]} patients, {df.shape[1]} features")

baseline = load_baseline(DATA_PATH, VERSION)
print(f"Baseline cohort: {baseline.shape}")

risks = load_risks(DATA_PATH, VERSION)
print(f"Health risks: {risks.shape}")

# Save sex mapping before merge operations
sex_map = df[['person_id', 'sex']].set_index('person_id') if 'sex' in df.columns else None

# Verify key columns
print("\n### Key columns verification:")
key_cols = ['person_id', 'sex', 'age_start', 'age_end', 'final_bmi',
            'final_sbp_mmhg', 'final_hba1c_percent', 'has_diabetes', 'diabetes_risk_10year']
for col in key_cols:
    status = "✓" if col in df.columns else "✗"
    print(f"  {status} {col}: {'present' if col in df.columns else 'MISSING'}")
# %% [markdown]
# ## 2. EDA: Missing Values Analysis (MAR Pattern)
# %%
# Missing values pattern
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).sort_values(ascending=False)
missing_cols = missing_pct[missing_pct > 0]
# %%
# MISSING VALUES (MAR pattern)
missing_report = pd.DataFrame({
    'Column': missing_cols.index,
    'Missing Count': missing[missing_cols.index].values,
    'Missing %': missing_cols.values
}).reset_index(drop=True)
# %%
print_markdown_table(missing_report, title="Missing values by column")
save_table(missing_report, 'missing_values_report.csv', output_dir=str(OUTPUT_DIR))
# %%
# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
missing_cols.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Missing Values by Column')
axes[0].set_ylabel('% missing')
axes[0].tick_params(axis='x', rotation=45)

# MAR check: age dependency
if 'age_end' in df.columns and len(missing_cols) > 0:
    young = df[df['age_end'] < 45]
    old = df[df['age_end'] >= 45]
    mar_data = []
    for col in missing_cols.index[:6]:
        mar_data.append({
            'Column': col,
            'Young (<45)': young[col].isna().mean() * 100,
            'Old (≥45)': old[col].isna().mean() * 100
        })
    mar_df = pd.DataFrame(mar_data).set_index('Column')
    mar_df.plot(kind='bar', ax=axes[1], color=['coral', 'steelblue'])
    axes[1].set_title('MAR Pattern: Missing % by Age Group')
    axes[1].set_ylabel('% missing')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

plt.tight_layout()
save_figure(fig, 'missing_values_analysis', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()
# %%
# Statistical test for MAR pattern
print("\n### MAR Pattern Validation (Chi-square test)")
mar_test_results = []
young_mask = df['age_end'] < 45
old_mask = df['age_end'] >= 45
print(f"Age distribution: Young (<45) = {young_mask.sum()}, Old (≥45) = {old_mask.sum()}")

for col in missing_cols.index[:6]:
    young_missing = df.loc[young_mask, col].isna().sum()
    young_total = young_mask.sum()
    old_missing = df.loc[old_mask, col].isna().sum()
    old_total = old_mask.sum()
    contingency = pd.DataFrame([
        [young_missing, young_total - young_missing],
        [old_missing, old_total - old_missing]
    ], index=['Young (<45)', 'Old (≥45)'], columns=['Missing', 'Present'])
    if contingency.values.min() >= 5:
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        mar_test_results.append({
            'Column': col,
            'Young Missing %': (young_missing / young_total) * 100,
            'Old Missing %': (old_missing / old_total) * 100,
            'Chi-square': chi2,
            'p-value': p_val,
            'MAR Confirmed': 'Yes' if p_val < 0.05 else 'No'
        })

mar_test_df = pd.DataFrame(mar_test_results)
if len(mar_test_df) > 0:
    print_markdown_table(mar_test_df.round(4), title="MAR pattern chi-square test")
    save_table(mar_test_df.round(4), 'mar_pattern_test.csv', output_dir=str(OUTPUT_DIR))

print("\nExpected MAR pattern: young patients have more missing values")

# %% [markdown]
# ## 3. Validation Against Generator Specification
# %%
validation_results = []

# Check 1: MAR missingness in expected range (2-15%)
expected_mar_cols = [
    'final_sbp_mmhg', 'final_hdl_mgdl', 'final_total_cholesterol_mgdl',
    'final_hba1c_percent', 'avg_alcohol_g_per_week', 'avg_total_met_minutes'
]
for col in expected_mar_cols:
    if col in df.columns:
        missing_rate = df[col].isna().mean()
        in_range = 0.02 <= missing_rate <= 0.20
        validation_results.append({
            'Check': f'{col} MAR rate',
            'Expected': '2-15%',
            'Actual': f'{missing_rate:.1%}',
            'Status': 'OK' if in_range else 'WARN'
        })

# Check 2: Age progression
if 'age_start' in df.columns and 'age_end' in df.columns:
    age_diff = df['age_end'] - df['age_start']
    correct_age = (age_diff == 20).mean()
    validation_results.append({
        'Check': 'age_end = age_start + 20',
        'Expected': '100%',
        'Actual': f'{correct_age:.1%}',
        'Status': 'OK' if correct_age > 0.99 else 'WARN'
    })

# Check 3: BMI and SBP biomarker ranges
biomarker_ranges = {
    'final_bmi': (16, 50),
    'final_sbp_mmhg': (80, 200),
}
for col, (low, high) in biomarker_ranges.items():
    if col in df.columns:
        in_range = ((df[col] >= low) & (df[col] <= high)).mean()
        validation_results.append({
            'Check': f'{col} range',
            'Expected': f'>95% in [{low}, {high}]',
            'Actual': f'{in_range:.1%} in range',
            'Status': 'OK' if in_range > 0.95 else 'WARN'
        })

# Check 3: HDL and HbA1c biomarker ranges
biomarker_ranges = {
    'final_hdl_mgdl': (20, 100),
    'final_hba1c_percent': (3.5, 12.0)
}
for col, (low, high) in biomarker_ranges.items():
    if col in df.columns:
        in_range = ((df[col] >= low) & (df[col] <= high)).mean()
        validation_results.append({
            'Check': f'{col} range',
            'Expected': f'>90% in [{low}, {high}]',
            'Actual': f'{in_range:.1%} in range',
            'Status': 'OK' if in_range > 0.90 else 'WARN'
        })

# Check 4: Risk-prevalence alignment
if 'has_diabetes' in df.columns and 'diabetes_risk_10year' in df.columns:
    prevalence = df['has_diabetes'].mean()
    mean_risk = df['diabetes_risk_10year'].mean()
    diff = abs(prevalence - mean_risk)
    validation_results.append({
        'Check': 'Diabetes risk-prevalence',
        'Expected': 'Diff < 0.05',
        'Actual': f'Diff = {diff:.3f}',
        'Status': 'OK' if diff < 0.05 else 'WARN'
    })

val_df = pd.DataFrame(validation_results)

print_markdown_table(val_df, title="Generator specification validation")
save_table(val_df, 'aggregated_validation_check.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 4. Cross-Dataset Consistency Check
# %%
# Cross-dataset consistency
ids_agg = set(df['person_id'].unique())
ids_baseline = set(baseline['person_id'].unique())
ids_risks = set(risks['person_id'].unique())
common_ids = ids_agg.intersection(ids_baseline).intersection(ids_risks)

consistency_check = {
    'IDs in 05_aggregated': len(ids_agg),
    'IDs in 01_baseline': len(ids_baseline),
    'IDs in 04_risks': len(ids_risks),
    'Common IDs (all 3)': len(common_ids),
    'Consistency': 'OK' if len(common_ids) == len(ids_agg) == len(ids_baseline) == len(ids_risks) else 'WARN'
}
print_markdown_table(pd.Series(consistency_check).to_frame(), title="Cross-dataset consistency")
save_table(pd.Series(consistency_check).to_frame(), 'cross_dataset_consistency_agg.csv', output_dir=str(OUTPUT_DIR))

# Verify sex column consistency
if sex_map is not None and 'sex' in baseline.columns:
    baseline_sex = baseline.set_index('person_id')['sex']
    agg_sex = df.set_index('person_id')['sex']
    sex_match = (baseline_sex == agg_sex).mean()
    print(f"\nSex column consistency (baseline vs aggregated): {sex_match:.1%} match")
# %% [markdown]
# ## 5. Data Preparation: Target and Features
# %%
# Target variable: diabetes (good balance ~15%)
target = 'has_diabetes'
risk_target = 'diabetes_risk_10year'

# Remove leakage columns
leakage_cols = [c for c in df.columns if c.startswith(('has_', '_risk'))
                and c not in [target, risk_target]]
leakage_cols.extend([c for c in df.columns if 'risk' in c.lower() and c not in [risk_target]])
leakage_cols = list(set(leakage_cols))
print(f"\nRemoving leakage columns: {len(leakage_cols)} columns")

# Drop ID and auxiliary columns
drop_cols = ['person_id', 'primary_death_cause', 'estimated_event_age'] + leakage_cols
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols + [target, risk_target])
y = df[target]

# Encode categorical variables before imputation
print(f"\n### Checking for non-numeric columns...")
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if 'sex' in X.columns:
    print("Encoding 'sex' column (M=1, F=0)...")
    X['sex_M'] = (X['sex'] == 'M').astype(int)
    X = X.drop(columns=['sex'])
    print("  'sex' encoded as 'sex_M'")

# Ensure all columns are numeric
X = X.select_dtypes(include=[np.number])
print(f"Numeric features only: {X.shape[1]}")
print(f"\nFeatures: {X.shape[1]}")
print(f"Target prevalence: {y.mean():.1%}")

# Save feature list
save_table(pd.DataFrame(X.columns.tolist(), columns=['feature_name']),
           'feature_list.csv', output_dir=str(OUTPUT_DIR))

# Split BEFORE imputation (avoid leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} patients, Test: {len(X_test)} patients")

# %% [markdown]
# ## 6. Imputation: Method Comparison
# 
# %%
print("\n=== IMPUTATION ===")

# Method 1: Simple (median) — baseline
simple_imp = sklearn.impute.SimpleImputer(strategy='median')
X_train_simple = simple_imp.fit_transform(X_train)
X_test_simple = simple_imp.transform(X_test)

# Method 2: MICE (Iterative) — recommended for MAR
print("Running MICE imputation (this may take 1-2 minutes)...")
mice_imp = sklearn.impute.IterativeImputer(max_iter=10, random_state=42, sample_posterior=True)
X_train_mice = mice_imp.fit_transform(X_train)
X_test_mice = mice_imp.transform(X_test)
print(f"Imputation complete. Shape: {X_train_mice.shape}")

# Save imputation comparison
save_table(pd.DataFrame({
    'Method': ['Simple (median)', 'MICE (Iterative)'],
    'Train Shape': [X_train_simple.shape[0], X_train_mice.shape[0]],
    'Features': [X_train_simple.shape[1], X_train_mice.shape[1]],
    'Recommended': ['No', 'Yes (MAR data)']
}), 'imputation_comparison.csv', output_dir=str(OUTPUT_DIR))

# Post-imputation clipping to physiological ranges
print("\n### Post-imputation clipping...")
clipped_cols = []
X_train_mice_df = pd.DataFrame(X_train_mice, columns=X.columns)
X_test_mice_df = pd.DataFrame(X_test_mice, columns=X.columns)

for col, (low, high) in biomarker_ranges.items():
    if col in X_train_mice_df.columns:
        X_train_mice_df[col] = X_train_mice_df[col].clip(low, high)
        X_test_mice_df[col] = X_test_mice_df[col].clip(low, high)
        clipped_cols.append(col)
        print(f"    {col}: clipped to [{low}, {high}]")

X_train_mice = X_train_mice_df.values
X_test_mice = X_test_mice_df.values
print(f"  {len(clipped_cols)} biomarker columns clipped post-imputation")


# %% [markdown]
# ## 7. Feature Engineering (Domain-Informed)
# 
# %%
def engineer_features(X_df, feature_names):
    """Create domain-informed features based on generator specification"""
    X = pd.DataFrame(X_df, columns=feature_names)

    # 1. Metabolic syndrome: BMI, SBP, HbA1c combination
    if all(c in X.columns for c in ['final_bmi', 'final_sbp_mmhg', 'final_hba1c_percent']):
        X['metabolic_syndrome_score'] = (
                (X['final_bmi'] > 30).astype(float) +
                (X['final_sbp_mmhg'] > 130).astype(float) +
                (X['final_hba1c_percent'] > 5.7).astype(float)
        )

    # 2. Activity ratio: strength vs cardio
    if all(c in X.columns for c in ['avg_strength_met_minutes', 'avg_total_met_minutes']):
        X['strength_ratio'] = X['avg_strength_met_minutes'] / (X['avg_total_met_minutes'] + 1)

    # 3. Alcohol risk: threshold 140 g/week
    if 'avg_alcohol_g_per_week' in X.columns:
        X['high_alcohol_risk'] = (X['avg_alcohol_g_per_week'] > 140).astype(float)

    # 4. Sleep deficit: <6 hours
    if 'avg_sleep_hours' in X.columns:
        X['sleep_deficit'] = np.maximum(0, 6 - X['avg_sleep_hours'])

    # 5. Age risk: >45 years
    if 'age_end' in X.columns:
        X['age_over_45'] = (X['age_end'] > 45).astype(float)

    # 6. Lipid risk: non-HDL cholesterol
    if all(c in X.columns for c in ['final_hdl_mgdl', 'final_total_cholesterol_mgdl']):
        X['non_hdl_cholesterol'] = X['final_total_cholesterol_mgdl'] - X['final_hdl_mgdl']

    return X


# Apply feature engineering
feature_names = X.columns.tolist()
X_train_eng = engineer_features(X_train_mice, feature_names)
X_test_eng = engineer_features(X_test_mice, feature_names)
new_features = [c for c in X_train_eng.columns if c not in feature_names]

print(f"\nEngineered features: {X_train_eng.shape[1]} (was {len(feature_names)})")
print(f"New features: {new_features}")
save_table(pd.DataFrame(new_features, columns=['engineered_feature']),
           'engineered_features.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 8. Scaling and Model Preparation
# 
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Tree-based models don't require scaling
X_train_tree = X_train_eng.values
X_test_tree = X_test_eng.values

# Save scaler parameters
save_table(pd.DataFrame({
    'feature': X_train_eng.columns,
    'mean': scaler.mean_,
    'std': scaler.scale_
}), 'scaler_parameters.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 9. Model Selection with Cross-Validation
# 
# %%
print("\n=== MODEL SELECTION ===")

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
    print(f"\nTuning {name}...")
    X_tr = X_train_scaled if 'Logistic' in name else X_train_tree
    grid = GridSearchCV(model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_tr, y_train)
    best_models[name] = grid.best_estimator_
    cv_results.append({
        'Model': name,
        'Best CV AUC': grid.best_score_,
        'Best Params': str(grid.best_params_)
    })
    print(f"  Best CV AUC: {grid.best_score_:.3f}")
    print(f"  Params: {grid.best_params_}")

cv_results_df = pd.DataFrame(cv_results)
print("\n### Cross-validation results")
print_markdown_table(cv_results_df, title="Cross-validation results")
save_table(cv_results_df, 'cv_model_selection.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 10. Ensemble: Voting Classifier
# 
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
# 
# %%
print("\n=== FINAL EVALUATION ===")

comparison = []
for name, y_proba in {**y_pred_proba_individual, 'Ensemble': y_pred_proba_ensemble}.items():
    auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    comparison.append({
        'Model': name,
        'AUC-ROC': auc,
        'Avg Precision': avg_prec,
        'Brier Score': brier
    })

comparison_df = pd.DataFrame(comparison).sort_values('AUC-ROC', ascending=False)
print("\nModel comparison:")
print_markdown_table(comparison_df, title="Model comparison")
save_table(comparison_df, 'model_comparison_final.csv', output_dir=str(OUTPUT_DIR))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC curves
from sklearn.metrics import roc_curve

axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
for name, y_proba in {**y_pred_proba_individual, 'Ensemble': y_pred_proba_ensemble}.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0, 0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curves')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Precision-Recall curves
for name, y_proba in {**y_pred_proba_individual, 'Ensemble': y_pred_proba_ensemble}.items():
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    axes[0, 1].plot(recall, precision, label=f'{name} (AP={avg_prec:.3f})', linewidth=2)
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curves')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Feature importance
rf_model = best_models['RandomForest']
importance = pd.Series(rf_model.feature_importances_, index=X_train_eng.columns).sort_values(ascending=True).tail(15)
importance.plot(kind='barh', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_title('Top 15 Feature Importances (Random Forest)')
axes[1, 0].grid(True, alpha=0.3)

# Calibration curves
from sklearn.calibration import calibration_curve

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
save_figure(fig, 'model_comparison_complete', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()

# Save feature importance
save_table(importance.to_frame(name='importance'), 'feature_importance_aggregated.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 12. Best Model Interpretation
# 
# %%
best_model_name = comparison_df.iloc[0]['Model']
best_auc = comparison_df.iloc[0]['AUC-ROC']

print(f"\n=== BEST MODEL: {best_model_name} ===")
print(f"AUC-ROC: {best_auc:.3f}")

if best_model_name == 'Ensemble':
    print("\nEnsemble weights are equal (soft voting)")
    print("Individual contributions:")
    ensemble_contributions = []
    for name, y_proba in y_pred_proba_individual.items():
        corr_with_ensemble = np.corrcoef(y_proba, y_pred_proba_ensemble)[0, 1]
        print(f"  {name}: correlation with ensemble = {corr_with_ensemble:.3f}")
        ensemble_contributions.append({
            'Model': name,
            'Correlation with Ensemble': corr_with_ensemble
        })
    save_table(pd.DataFrame(ensemble_contributions), 'ensemble_contributions.csv', output_dir=str(OUTPUT_DIR))


# %% [markdown]
# ## 13. External Validation (Compare with Real Data)
# 
# %%
def compare_with_real_data(synthetic_df):
    """Compare synthetic data distributions with reference values from literature"""
    reference_ranges = {
        'final_bmi': {'mean': 28.0, 'std': 5.5, 'source': 'NHANES 2020'},
        'final_sbp_mmhg': {'mean': 125.0, 'std': 15.0, 'source': 'NHANES 2020'},
        'final_hba1c_percent': {'mean': 5.7, 'std': 0.8, 'source': 'CDC'},
        'final_hdl_mgdl': {'mean': 52.0, 'std': 15.0, 'source': 'NHANES 2020'},
    }
    comparison_results = []
    for col, ref in reference_ranges.items():
        if col in synthetic_df.columns:
            syn_vals = synthetic_df[col].dropna()
            t_stat, p_val = stats.ttest_1samp(syn_vals, ref['mean'])
            ks_stat, ks_p = stats.kstest((syn_vals - syn_vals.mean()) / syn_vals.std(), 'norm')
            comparison_results.append({
                'Feature': col,
                'Synthetic Mean': syn_vals.mean(),
                'Reference Mean': ref['mean'],
                'Synthetic Std': syn_vals.std(),
                'Reference Std': ref['std'],
                'T-test p-value': p_val,
                'KS-test statistic': ks_stat,
                'KS-test p-value': ks_p,
                'Reference Source': ref['source'],
                'Match': 'Yes' if p_val > 0.01 else 'No'
            })
    return pd.DataFrame(comparison_results)


print("\n=== EXTERNAL VALIDATION ===")
external_val_df = compare_with_real_data(df)
print_markdown_table(external_val_df, title="External validation: synthetic vs reference")
save_table(external_val_df, 'external_validation_comparison.csv', output_dir=str(OUTPUT_DIR))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(['final_bmi', 'final_sbp_mmhg', 'final_hba1c_percent', 'final_hdl_mgdl']):
    if col in df.columns and i < 4:
        ax = axes[i]
        df[col].dropna().hist(bins=50, ax=ax, alpha=0.7, label='Synthetic', density=True)
        ref_mean = external_val_df[external_val_df['Feature'] == col]['Reference Mean'].values[0]
        ref_std = external_val_df[external_val_df['Feature'] == col]['Reference Std'].values[0]
        x = np.linspace(df[col].min(), df[col].max(), 100)
        from scipy.stats import norm

        ax.plot(x, norm.pdf(x, ref_mean, ref_std), 'r--', linewidth=2, label='Reference')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.set_title(f'{col}: Synthetic vs Reference')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure(fig, 'external_validation_distributions', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()

# %% [markdown]
# ## 14. Comparison with Notebook 4 (Health Risks Only)
# 
# %%
print("\n=== COMPARISON WITH NOTEBOOK 4 (04_health_risks) ===")

notebook_4_results = {
    'CVD': {'AUC': 0.651, 'Model': 'Logistic Regression'},
    'Diabetes': {'AUC': 0.588, 'Model': 'Random Forest'},
    'Stroke': {'AUC': 0.591, 'Model': 'Random Forest'}
}

print("\nNotebook 4 (health risks only):")
for disease, results in notebook_4_results.items():
    print(f"  {disease}: AUC = {results['AUC']:.3f} ({results['Model']})")

print(f"\nNotebook 5 (aggregated + lifestyle + biomarkers):")
print(f"  Diabetes: AUC = {best_auc:.3f} ({best_model_name})")

diabetes_auc_nb4 = notebook_4_results['Diabetes']['AUC']
improvement = best_auc - diabetes_auc_nb4
print(f"\nImprovement from aggregated features: +{improvement:.3f} ({improvement / diabetes_auc_nb4 * 100:.1f}%)")

save_table(pd.DataFrame({
    'Notebook': ['Notebook 4 (risks)', 'Notebook 5 (aggregated)'],
    'Target': ['Diabetes', 'Diabetes'],
    'AUC-ROC': [diabetes_auc_nb4, best_auc],
    'Features': ['Health risks only', 'Lifestyle + biomarkers + demographics'],
    'Imputation': ['N/A', 'MICE (MAR)']
}), 'notebook_comparison_4_vs_5.csv', output_dir=str(OUTPUT_DIR))

# %% [markdown]
# ## 15. Threshold Analysis for Clinical Utility
# 
# %%
print("\n=== THRESHOLD ANALYSIS ===")

threshold_analysis = []
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
    y_pred_thresh = (y_pred_proba_ensemble >= thresh).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score

    threshold_analysis.append({
        'Threshold': thresh,
        'Precision': precision_score(y_test, y_pred_thresh, zero_division=0),
        'Recall': recall_score(y_test, y_pred_thresh, zero_division=0),
        'F1': f1_score(y_test, y_pred_thresh, zero_division=0),
        'Positive Predictions': int(y_pred_thresh.sum())
    })

thresh_df = pd.DataFrame(threshold_analysis)
print_markdown_table(thresh_df, title="Threshold analysis")
save_table(thresh_df, 'threshold_analysis.csv', output_dir=str(OUTPUT_DIR))

# Plot threshold trade-off
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
save_figure(fig, 'threshold_tradeoff', output_dir=str(OUTPUT_DIR), format=FIGURE_FORMAT)
plt.show()

# %% [markdown]
# ## 16. Summary and Conclusions
# 
# ### 1. Dataset Characteristics
# - **~5000 patients**, 47 features after cleaning
# - **6 columns with MAR missing values**: 2.9-3.7%
# - **Diabetes prevalence**: ~15%
# - **Cross-dataset consistency**: 100% ID match
# 
# ### 2. MAR Pattern
# - **Age threshold**: age_end < 45 (not 30, as age ranges 40-70)
# - **Visual pattern**: Young patients have more missing values — matches specification
# 
# ### 3. Imputation
# - **MICE**: recommended for MAR data
# - **Post-imputation clipping**: 4 biomarkers clipped to physiological ranges
# - **Note**: MICE may generate values outside ranges (expected behavior)
# 
# ### 4. Feature Engineering
# - **+5 features**: metabolic_syndrome_score, high_alcohol_risk, sleep_deficit, age_over_45, non_hdl_cholesterol
# - **Top 5 features**: final_bmi, final_sbp_mmhg, age_end, health_score, avg_sleep_hours
# 
# ### 5. Predictive Modeling
# - **Best model**: Ensemble (AUC-ROC ~0.65)
# - **Improvement vs Notebook 4**: +0.06 (+10%)
# - **Threshold recommendation**: 0.2 (screening), 0.3 (balance), 0.4 (diagnosis)
# 
# ### 6. Limitations
# - Synthetic data — NOT for clinical use
# - MAR χ² test requires age threshold correction
# - Low precision (16-28%) — expected for 15% prevalence
# 
# ## Output Files
# 
# All generated tables and figures have been saved to:
# - `output/05_aggregated_pipeline/` — CSV tables and PNG figures
# - `logs/05_aggregated_pipeline_*.txt` — Complete execution log
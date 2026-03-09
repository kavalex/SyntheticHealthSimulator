#!/usr/bin/env python3
"""
Sensitivity Analysis: Test generator stability across different random seeds
"""
import sys

import pandas as pd

sys.path.append('..')
from generator import DataGenerator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def run_sensitivity_analysis(n_seeds=5, n_people=5000, years=20):
    seeds = [42, 123, 456, 789, 101112]
    results = []

    print("-" * 50)
    print("SENSITIVITY ANALYSIS: Testing generator stability across seeds")
    print("-" * 50)

    for seed in seeds[:n_seeds]:
        print(f"\nRunning seed={seed}...")

        # Generate data
        generator = DataGenerator(seed=seed, diabetes_threshold=18.7)
        cohort = generator.generate_cohort(n_people)
        lifestyle = generator.generate_lifestyle_history(cohort, years=years)
        biomarkers = generator.calculate_biomarkers(cohort, lifestyle, years=years)
        risks = generator.calculate_health_risks(biomarkers, cohort, years=years)

        #  Use column names from biomarkers DataFrame
        last_year = years - 1  # 19 for 20 years

        # Build feature DataFrame with column names
        features_dict = {
            'final_bmi': biomarkers[f'bmi_{last_year}'].values,
            'final_sbp_mmhg': biomarkers[f'sbp_mmhg_{last_year}'].values,
            'final_hba1c_percent': biomarkers[f'hba1c_percent_{last_year}'].values,
            'avg_alcohol_g_per_week': biomarkers[[f'alcohol_g_per_week_{y}' for y in range(years)]].mean(axis=1).values,
            'age_start': cohort['age_start'].values,  # From cohort, not biomarkers
        }

        X = pd.DataFrame(features_dict)
        y = risks['has_diabetes']

        # Handle any missing values (shouldn't be any in raw biomarkers)
        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        results.append({
            'seed': seed,
            'diabetes_prevalence': risks['has_diabetes'].mean(),
            'cvd_prevalence': risks['has_cvd'].mean(),
            'stroke_prevalence': risks['has_stroke'].mean(),
            'mean_bmi': biomarkers[f'bmi_{last_year}'].mean(),
            'mean_sbp': biomarkers[f'sbp_mmhg_{last_year}'].mean(),
            'ml_auc': auc
        })

        print(f"  Diabetes: {risks['has_diabetes'].mean():.1%}, AUC: {auc:.3f}")

    results_df = pd.DataFrame(results)
    print("\n" + "-" * 50)
    print("SUMMARY: Stability across seeds")
    print("-" * 50)
    print(results_df.to_markdown(index=False))

    # Statistics
    print("\nCoefficient of Variation (CV):")
    for col in ['diabetes_prevalence', 'cvd_prevalence', 'ml_auc']:
        cv = results_df[col].std() / results_df[col].mean() * 100
        print(f"  {col}: {cv:.2f}%")

    results_df.to_csv('sensitivity_analysis_results.csv', index=False)

    # Visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for i, col in enumerate(['diabetes_prevalence', 'cvd_prevalence', 'mean_bmi', 'ml_auc']):
        ax = axes[i // 2, i % 2]
        ax.bar(range(len(results_df)), results_df[col], color='steelblue')
        ax.axhline(results_df[col].mean(), color='red', linestyle='--', label='Mean')
        ax.set_xlabel('Seed Index')
        ax.set_ylabel(col)
        ax.set_title(f'{col} across seeds')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sensitivity_analysis_plots.png', dpi=150)
    plt.show()

    return results_df


if __name__ == "__main__":
    run_sensitivity_analysis()

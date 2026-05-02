import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from typing import Literal


def compute_log_p_values(X: pd.DataFrame, y: pd.Series,
                    test_type: Literal['t-test', 'wilcoxon'],
                    multiple_test_method: Literal[None, 'bonferroni', 'fdr_bh'],
                    equal_var: bool,
                    p_threshold: float):

    group_healthy = X[y == 0]
    group_disease = X[y == 1]
    p_values = []
 
    if test_type == 't-test':
        for feature in X.columns:
            try:
                # Remove NaN values
                healthy_vals = group_healthy[feature].dropna()
                disease_vals = group_disease[feature].dropna()
                
                # Check if we have enough data points
                if len(healthy_vals) < 2 or len(disease_vals) < 2:
                    p_values.append(1.0)  # No significant difference
                else:
                    t_stat, p_value = ttest_ind(healthy_vals, disease_vals, equal_var=equal_var)
                    # Ensure p_value is a scalar
                    p_values.append(float(p_value))
            except Exception as e:
                print(f"Warning: t-test failed for feature {feature}: {e}")
                p_values.append(1.0)

    elif test_type == 'wilcoxon':
        for feature in X.columns:
            try:
                # Remove NaN values
                healthy_vals = group_healthy[feature].dropna()
                disease_vals = group_disease[feature].dropna()
                
                # Check if we have enough data points
                if len(healthy_vals) < 2 or len(disease_vals) < 2:
                    p_values.append(1.0)  # No significant difference
                else:
                    # Use Mann-Whitney U test (Wilcoxon rank-sum test) for independent groups
                    stat, p_value = mannwhitneyu(healthy_vals, disease_vals, alternative='two-sided')
                    # Ensure p_value is a scalar
                    p_values.append(float(p_value))
            except Exception as e:
                print(f"Warning: Mann-Whitney U test failed for feature {feature}: {e}")
                p_values.append(1.0)

    p_values = np.array(p_values)
    
    # Apply multiple test correction if specified
    if multiple_test_method is not None:
        reject_multi, p_values_multi, _, _ = multipletests(p_values, method=multiple_test_method, alpha=p_threshold)
    else:
        p_values_multi = p_values
    
    # Return as pandas Series with feature names as index
    return pd.Series(-np.log10(p_values_multi), index=X.columns)
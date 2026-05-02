import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from common.modeling_utils import train_test_split_df

from maternal_HIV_effects.significant_biomarkers.utils import (
    prepare_data,
    load_maternal_results,
    save_json,
    load_lasso_path,
)
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.cv_1se import find_1se_lambda

from maternal_HIV_effects.plotting_utils import (
    plot_lasso_path,
)
from maternal_HIV_effects.significant_biomarkers.utils import (
    evaluate_model,
    statsmodels_train_model,
)

from common.feature_engineering import FeatureEngineer

quality_control = [{'IQR_median_min': 0.05},
                   {'RSD_min': 0.05},
                   {'too_many_missing_values': 0.5}, 
                   {'RSD_max': 100}
                    ]
cv_metric = 'auc'
L1_wt = 0.5

def lasso_path_from_saved_results(biomarkers_type: str, significant_features: list = None):
    path_results = load_lasso_path(biomarkers_type)
    print(f'loaded lasso path shape {len(path_results)}, num lambdas {len(path_results[0].keys())}')
    full_data = prepare_data(biomarkers_type)
    X, y, demographics = _intial_quality_control(full_data)
    conf_cols = list(set(demographics.columns).intersection(X.columns))
    print(f'step 0 done with X shape {X.shape}, y shape {y.shape}, demographics shape {demographics.shape}')
    plot_lasso_path(path_results, conf_cols, biomarkers_type, significant_features=significant_features)
    return path_results


def lasso_path(save_results_flag: bool = True, biomarkers_type: str = None):
    """Train logistic regression with elastic net regularization and balanced class weights."""
    # 1) prepare the raw data 
    # 1) prepare the raw data 
    full_data = prepare_data(biomarkers_type)
    X, y, demographics = _intial_quality_control(full_data)
    conf_cols = list(set(demographics.columns).intersection(X.columns))


    # 2) set the number of trials and the seeds
    rng = np.random.default_rng(42)
    n_trials = int(input('Enter number of trials: '))
    old_path_results = load_lasso_path(biomarkers_type)
    old_n_trials = len(old_path_results)
    trial_seeds = rng.integers(low=0, high=1_00_000, size=500)  # Generate independent seeds
    print(f'step 2 done with n_trials {n_trials}, old_path_results shape {len(old_path_results)}')

    # 3) train the model for each trial
    path_results = []
    for iter in range(n_trials):
        t = iter + old_n_trials
        trial_results = trial_worker(t, X, y, trial_seeds, conf_cols)
        path_results.append(trial_results)

        # save the results (save individual trial, not entire list)
        if save_results_flag:
            save_json(trial_results, filename=f'lasso_path_{biomarkers_type}.json', dir=f'maternal_HIV_effects/significant_biomarkers/results/lasso_path/{biomarkers_type}')

    # 4) plot the stability path
    plot_lasso_path(path_results, conf_cols, biomarkers_type)

    return path_results

def trial_worker(t, X, y, trial_seeds, conf_cols, reg_inv_list=None):

    # 1) split the data into training and testing
    X_tr, X_tst, y_tr, y_tst = train_test_split_df(X, y, 0.7, stratify=True, seed=trial_seeds[t])
    X_tr, X_tst = _normalize_and_impute_data(X_tr.copy(), X_tst.copy())

    reg_weights = _reg_weights(X_tr, conf_cols)
    class_w_tr = _class_weights(y_tr) 
    X_tr, X_tst, reg_weights = _add_intercept(X_tr, X_tst, reg_weights)

    trial_path = _trial_path_worker(X_tr, y_tr, reg_weights, L1_wt, class_w_tr, reg_inv_list) 
    print(f'   trial {t} done with trial_path shape {len(trial_path)}')
    return trial_path
    

def _trial_path_worker(X_tr, y_tr, reg_weights, L1_wt, class_w_tr, reg_inv_list=None):
    if reg_inv_list is None:
        reg_inv_list = np.logspace(-0.11, 0.8, 100) 
    res_dict = {}
    for i, reg_inv in enumerate(reg_inv_list):
        print(f'   training model with {i+1}th regularization value {1/reg_inv}')
        res_dict[reg_inv] = statsmodels_train_model(X_tr, y_tr, 1/reg_inv * reg_weights,
                                                    L1_wt, class_w_tr)
    trial_path = {}
    for i, reg_inv in enumerate(reg_inv_list):
        reg = 1/reg_inv
        trial_path[reg] = res_dict[reg_inv].params
    return trial_path




########################################################
# helper functions
########################################################
def _reg_weights(X_tr, conf_cols):
    # returns the regularzation weight of 1 for all features except the confounders
    num_confounders = len([col for col in X_tr.columns if col in conf_cols])
    if num_confounders == 0:
        return [1]*X_tr.shape[1]
    reg_weights = np.ones(X_tr.shape[1]) * 1
    reg_weights[-num_confounders:] = 0
    return reg_weights

def _normalize_and_impute_data(X_tr, X_tst):
    feature_engineer_1 = FeatureEngineer(missing_values='Left_censored_min', feature_normalization='mean_std', verbose=False)
    X_tr= feature_engineer_1.fit_transform(X_tr.copy(), verbose=False)
    X_tst= feature_engineer_1.transform(X_tst.copy(), verbose=False)
    return X_tr, X_tst

def _add_intercept(X_tr, X_tst, reg_weights):
    reg_weights = np.insert(reg_weights.copy(), 0, 0) 
    X_tr = sm.add_constant(X_tr.copy())
    X_tst = sm.add_constant(X_tst.copy())
    return X_tr, X_tst, reg_weights

def _intial_quality_control(full_data):
    (X, y, demographics) = full_data
    feature_engineer_0 = FeatureEngineer(quality_control=quality_control)
    X = feature_engineer_0.fit_transform(X, verbose=False)

    return X, y, demographics

def _class_weights(y_tr):
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    n_total = len(y_tr)
    w1 = n_total / (2 * n_pos) if n_pos > 0 else 1.0  # Weight for positive class
    w0 = n_total / (2 * n_neg) if n_neg > 0 else 1.0  # Weight for negative class
    return np.where(y_tr == 1, w1, w0)


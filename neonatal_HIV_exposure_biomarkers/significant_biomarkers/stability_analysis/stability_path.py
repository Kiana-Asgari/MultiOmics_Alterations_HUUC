
import numpy as np
import pandas as pd
import statsmodels.api as sm

from joblib import (
    Parallel,
    delayed,
)

from neonatal_HIV_exposure_biomarkers.significant_biomarkers.cv_1se import find_1se_lambda
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.evaluation_and_plots import (
    plot_stability_path,
)
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.utils import (
    prepare_data,
    statsmodels_train_model,
)
from common.feature_engineering import FeatureEngineer
from common.modeling_utils import train_test_split_df
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.stability_analysis_utils import (
    save_json,
    load_stability_path
)

# global variables
confounders = ["sex", "birth.weight.g", "alcohol", "smoking", "diabetes", "hypertension", "gravity",
                "gadelivery", "parity", "multiple.birth", "maternal.age.delivery", "prepregnancy.bmi"]
categorical_confounders = ["site"]
quality_control = [{'IQR_median_min': 0.05},
                   {'RSD_min': 0.05},
                   {'too_many_missing_values': 0.5}, 
                   {'RSD_max': 100}
                    ]
cv_metric = 'auc'
L1_wt = 0.5

def stability_path_from_saved_results(country: str, gender: str, significant_features: list):
    path_results = load_stability_path(country=country, gender=gender)
    plot_stability_path(path_results, country=country, gender=gender, significant_features=significant_features)
    return path_results

def stability_path(save_results_flag: bool = True):
    """Train logistic regression with elastic net regularization and balanced class weights."""
    # 1) prepare the raw data 
    full_data, country, gender = prepare_data(confounders)
    X, y, _ = _intial_quality_control(full_data, country)

    # 2) set the number of trials and the seeds
    rng = np.random.default_rng(42)
    n_trials = int(input('Enter number of trials: '))
    old_stability_path = load_stability_path(country, gender)
    old_n_trials = len(old_stability_path)
    trial_seeds = rng.integers(low=0, high=1_00_000, size=500)  # Generate independent seeds

    # 3) train the model for each trial
    results = []
    for iter in range(n_trials):
        t = iter + old_n_trials
        print(f'[Trial {t}] starting stability path training ...')
        path_results = trial_worker(t, X, y, trial_seeds, country, gender)
        last_lambda = list(path_results.keys())[-1]
        c3_coef = path_results[last_lambda].get("c3", None)
        print(f'            stability path training done; coef of c3 at last lambda ({last_lambda}): {c3_coef}')
        results.append(path_results)

        # 3.5) save the results
        if save_results_flag:
            save_json(path_results, filename=f'stability_path_{country}_{gender}.json', dir='neonatal_HIV_exposure_biomarkers/results/stability_path')

    # 4) plot the stability path
    plot_stability_path(results, country, gender)

    return results

def trial_worker(t, X, y, trial_seeds, country, gender):

    # 1) split the data into training and testing
    X_tr, X_tst, y_tr, y_tst = train_test_split_df(X, y, 0.7, stratify=True, seed=trial_seeds[t])
    X_tr, X_tst = _normalize_and_impute_data(X_tr.copy(), X_tst.copy())

    conf_cols = [c for c in confounders + categorical_confounders if c in X_tr.columns]
    reg_weights = _reg_weights(X_tr, conf_cols)
    class_w_tr = _class_weights(y_tr) 
    X_tr, X_tst, reg_weights = _add_intercept(X_tr, X_tst, reg_weights)

    trial_path = _trial_path_worker(X_tr, y_tr, reg_weights, L1_wt, class_w_tr) 
    return trial_path
    

def _trial_path_worker(X_tr, y_tr, reg_weights, L1_wt, class_w_tr):
    reg_inv_list = np.logspace(0.2, 2.4, 40)
    res_dict = Parallel(n_jobs=2)(
        delayed(statsmodels_train_model)(X_tr, y_tr, 1/reg_inv * reg_weights,
                                        L1_wt, class_w_tr)
        for reg_inv in reg_inv_list
    )
    trial_path = {}
    for i, reg_inv in enumerate(reg_inv_list):
        reg = 1/reg_inv
        trial_path[reg] = res_dict[i].params
    return trial_path




########################################################
# helper functions
########################################################
def _reg_weights(X_tr, conf_cols):
    # returns the regularzation weight of 1 for all features except the confounders
    num_confounders = len([col for col in X_tr.columns if col in conf_cols])
    reg_weights = np.ones(X_tr.shape[1]) * 1
    reg_weights[-num_confounders:] = 0 
    return reg_weights

def _normalize_and_impute_data(X_tr, X_tst):
    feature_engineer_1 = FeatureEngineer(missing_values='Left_censored_min', feature_normalization='mean_std', verbose=False)
    X_tr= feature_engineer_1.fit_transform(X_tr, verbose=False)
    X_tst= feature_engineer_1.transform(X_tst, verbose=False)
    return X_tr, X_tst

def _add_intercept(X_tr, X_tst, reg_weights):
    reg_weights = np.insert(reg_weights, 0, 0) 
    X_tr = sm.add_constant(X_tr)
    X_tst = sm.add_constant(X_tst)
    return X_tr, X_tst, reg_weights

def _intial_quality_control(full_data, country):
    (X, y, demographics) = full_data
    feature_engineer_0 = FeatureEngineer(quality_control=quality_control)
    X = feature_engineer_0.fit_transform(X, verbose=False)

    if country == "Both":
        X = pd.concat([X, demographics[categorical_confounders]], axis=1)
    return X, y, demographics

def _class_weights(y_tr):
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    n_total = len(y_tr)
    w1 = n_total / (2 * n_pos) if n_pos > 0 else 1.0  # Weight for positive class
    w0 = n_total / (2 * n_neg) if n_neg > 0 else 1.0  # Weight for negative class
    return np.where(y_tr == 1, w1, w0)


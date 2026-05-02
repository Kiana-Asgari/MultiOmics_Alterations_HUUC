from maternal_HIV_effects.FDR.thresholding_FDP import select_knockoff
from maternal_HIV_effects.FDR.X_knockoff import make_train_knockoffs, make_test_knockoffs
from maternal_HIV_effects.significant_biomarkers.utils import prepare_data


import numpy as np
import pandas as pd
import statsmodels.api as sm
from common.modeling_utils import train_test_split_df
from maternal_HIV_effects.significant_biomarkers.utils import (
    prepare_data,
    load_maternal_results,
    save_results,
)
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.cv_1se import find_1se_lambda

from maternal_HIV_effects.plotting_utils import (
    plot_delta_ROC,
    plot_significant_features,
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
reg_inv_list = np.logspace(-0.1, 1, 40)





def FDR_knockoff(save_results_flag: bool = True):
    """Train logistic regression with elastic net regularization and balanced class weights."""
    # 1) prepare the raw data 
    full_data = prepare_data()
    X, y, demographics = _intial_quality_control(full_data)
    conf_cols = list(set(demographics.columns).intersection(X.columns))
    print(f'FDR step 1 done with X shape {X.shape}, y shape {y.shape}, demographics shape {demographics.shape}')
    print(f'    conf_cols: {conf_cols}')

    # 2) set the number of trials and the seeds
    rng = np.random.default_rng(42)
    n_trials = int(input('Enter number of trials: '))

    trial_seeds = rng.integers(low=0, high=1_00_000, size=500)  # Generate independent seeds
    print(f'step 2 done with n_trials {n_trials}')

    # 3) train the model for each trial
    knockoff_scores = []
    significant_biomarkers = {}

    for iter in range(n_trials):
        t = iter
        trial_knockoff_scores = trial_worker(t, X, y, trial_seeds, conf_cols)
        optimal_S, optimal_T, optimal_fdp_hat = select_knockoff(trial_knockoff_scores, q=0.3, knockoff_plus=True)
        significant_biomarkers[f'trial_{t}'] = {'significant_biomarkers': optimal_S, 'T': optimal_T, 'fdp_hat': optimal_fdp_hat}
        print(f'trial {t} done with significant biomarkers: {optimal_S}, T: {optimal_T}, fdp_hat: {optimal_fdp_hat}')
    
    print(f'significant_biomarkers: {significant_biomarkers}')

    return significant_biomarkers

def trial_worker(t, X, y, trial_seeds, conf_cols):
    # 1) split the data into training and testing
    X_tr, _, y_tr, _ = train_test_split_df(X, y, 0.9, stratify=True, seed=trial_seeds[t])
    X_tr, _ = _normalize_and_impute_data(X_tr.copy(), X_tr.copy())
    # 2) add the knockoff of the start of the training data
    Xb_tilde_tr = _get_knockoff_copies(X_tr.copy(), conf_cols, trial_seeds[t])
    X_tr_knockoff = pd.concat([Xb_tilde_tr, X_tr], axis=1)
    X_tr_knockoff, _ = _normalize_and_impute_data(X_tr_knockoff.copy(), X_tr_knockoff.copy())
    print(f'knockoff_m295t300.neg.rplc: {Xb_tilde_tr['knockoff_m295t300.neg.rplc'][0:5]}')
    print(f'\nm295t300.neg.rplc: {X_tr['m295t300.neg.rplc'][0:5]}\n')

    reg_weights = _reg_weights(X_tr_knockoff, conf_cols)
    class_w_tr = _class_weights(y_tr) 
    X_tr, _, reg_weights = _add_intercept(X_tr_knockoff, X_tr_knockoff, reg_weights)
    print(f'  Final training data shape: {X_tr.shape}')


    trial_path = _trial_path_worker(X_tr, y_tr, reg_weights, L1_wt, class_w_tr) 
    trial_knockoff_scores = _compute_path_scores(trial_path)
    print(f'   trial {t} done with trial_knockoff_scores shape {len(trial_knockoff_scores)}')
    return trial_knockoff_scores
    

def _trial_path_worker(X_tr, y_tr, reg_weights, L1_wt, class_w_tr):
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


def _compute_path_scores(trial_path):
    # Get the first regularization value to extract feature names

    first_reg = min(trial_path.keys())
    feature_path_scores = {feature: 0 for feature in trial_path[first_reg].keys() if feature not in ['const', 'Intercept']}
    feature_knockoff_scores = {}

    for reg in trial_path.keys():
        for feature in trial_path[reg].keys():
            coef = trial_path[reg][feature]
            if feature in ['const', 'Intercept'] or coef == 0: # skip the constant or intercept terms
                continue
            if reg >= feature_path_scores[feature]: # keep the largest regularization value for each feature
                feature_path_scores[feature] = reg
    
    for feature in feature_path_scores.keys():
        if 'knockoff_' not in feature:
            _score = feature_path_scores[feature]
            _knockoff_feature = f'knockoff_{feature}'
            if _knockoff_feature in feature_path_scores.keys():
                _knockoff_score = feature_path_scores[_knockoff_feature]
                feature_knockoff_scores[feature] = _score - _knockoff_score
                if _score > 0:
                    print(f'   feature {feature} path score: {_score:.3f}, knockoff score: {_knockoff_score:.3f}, difference: {feature_knockoff_scores[feature]:.3f}')

    return feature_knockoff_scores


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



def _get_knockoff_copies(X_tr, conf_cols, seed):
    num_confounders = len([col for col in X_tr.columns if col in conf_cols])
    num_biomarkers = X_tr.shape[1] - num_confounders

    # Convert pandas DataFrames to numpy arrays for knockoff functions
    col_names = [col for col in X_tr.columns if (col not in conf_cols)]

    X_tr_array = X_tr.values

    X_tr_biom = X_tr_array[:,:num_biomarkers] # only the biomarkers column
    
    X_tr_knockoff, ko_params = make_train_knockoffs(X_tr_biom, seed)
    # Convert numpy arrays back to DataFrames for concatenation
    X_tr_knockoff_df = pd.DataFrame(X_tr_knockoff, index=X_tr.index, 
                                    columns=[f'knockoff_{biom}' for biom in col_names])

    return X_tr_knockoff_df
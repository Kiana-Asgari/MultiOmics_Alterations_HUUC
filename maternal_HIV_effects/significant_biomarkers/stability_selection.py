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
    delta_ROC_tables,
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
reg_inv_list = np.logspace(0, 1.65, 30)


def maternal_stability_score_from_saved_results(biomarkers_type, ROC=False, freq_threshold=0.8):
    results, unadjusted_results, baseline_results, se_results = load_maternal_results(biomarkers_type)
    feature_names = results[0]['feature_names']
    full_data = prepare_data(biomarkers_type)
    X, y, demographics = _intial_quality_control(full_data)
    conf_cols = list(set(demographics.columns).intersection(X.columns))

    if ROC:
        plot_delta_ROC(results, unadjusted_results, baseline_results, biomarkers_type, title='se_false')
        delta_ROC_tables(results, baseline_results, biomarkers_type, title='delta_AUC_tests')

    important_features = plot_significant_features(results, feature_names, conf_cols, biomarkers_type, title='se_false', freq_threshold=freq_threshold)

    return important_features



def stability_score(save_results_flag: bool = True,
                    se_rule: str = 'include_both',
                    adjusted: bool = True,
                    unadjusted: bool = False,
                    baseline: bool = True,
                    biomarkers_type: str = 'Both'):
    """Train logistic regression with elastic net regularization and balanced class weights."""
    # 1) prepare the raw data 
    full_data = prepare_data(biomarkers_type)
    breakpoint()
    X, y, demographics = _intial_quality_control(full_data)
    conf_cols = list(set(demographics.columns).intersection(X.columns))
    X_unadjusted = X.drop(columns=conf_cols)
    print(f'step 1 done with X shape {X.shape}, y shape {y.shape}, demographics shape {demographics.shape}')
    print(f'conf_cols: {conf_cols}')

    # 2) set the number of trials and the seeds
    rng = np.random.default_rng(42)
    n_trials = int(input('Enter number of trials: '))
    old_results, old_unadjusted_results, old_baseline_results, old_test_result_se = load_maternal_results(biomarkers_type)
    old_n_trials = len(old_results) if adjusted else len(old_unadjusted_results)
    trial_seeds = rng.integers(low=0, high=1_00_000, size=500)  # Generate independent seeds
    print(f'step 2 done with n_trials {n_trials}, old_results shape {len(old_results)}, old_baseline_results shape {len(old_baseline_results)}')

    # 3) train the model for each trial
    results, unadjusted_results, baseline_results, se_results = [], [], [], []

    for iter in range(n_trials):
        t = iter + old_n_trials
        trial_results, unadjusted_trial_results, baseline_trial_results, trial_se_results = trial_worker(t, X, y, trial_seeds, conf_cols, se_rule=se_rule, adjusted=adjusted, unadjusted=unadjusted, baseline=baseline)
        results.append(trial_results)
        se_results.append(trial_se_results)
        baseline_results.append(baseline_trial_results)
        unadjusted_results.append(unadjusted_trial_results)
        # 3.5) save the results
        if save_results_flag:
            save_results(trial_results, unadjusted_trial_results, baseline_trial_results, trial_se_results, biomarkers_type)

    # 4) plot the significant features and the delta ROC curve
    plot_significant_features(results, X.columns, conf_cols, 'se_false')
    plot_significant_features(se_results, X.columns, conf_cols, 'se_true')
    plot_delta_ROC(results, unadjusted_results, baseline_results, 'se_false')
    plot_delta_ROC(se_results, unadjusted_results, baseline_results, 'se_true')

    return results

def trial_worker(t, X, y, trial_seeds, conf_cols, se_rule, adjusted=True, unadjusted=True, baseline=True):
    # 1) split the data into training and testing
    X_tr, X_tst, y_tr, y_tst = train_test_split_df(X, y, 0.7, stratify=True, seed=trial_seeds[t])
    X_tr_unadjusted, X_tst_unadjusted = X_tr.copy().drop(columns=conf_cols), X_tst.copy().drop(columns=conf_cols)
    print(f'[Trial {t}] training: {X_tr.shape}, {y_tr.shape}, testing: {X_tst.shape}, {y_tst.shape}')
    print(f'            unadjusted training: {X_tr_unadjusted.shape}, {y_tr.shape}, testing: {X_tst_unadjusted.shape}, {y_tst.shape}')
    
    # 3.1) train the adjusted model
    test_results, test_result_se, baseline_test_results, unadjusted_test_results = None, None, None, None
    if adjusted:
        test_results, test_result_se = train_trial_model(X_tr, y_tr, X_tst, y_tst,
                                        conf_cols, t, CV=True, se_rule=se_rule)
        print(f'[Trial {t}] adjusted AUC: {test_results["auc"]:.3f}')
    if baseline:
        X_tr_conf, X_tst_conf = X_tr[conf_cols], X_tst[conf_cols]
        baseline_test_results, _ = train_trial_model(X_tr_conf, y_tr, X_tst_conf, y_tst,
                                                    conf_cols, t, CV=False, lambda_value=0)
        print(f'[Trial {t}] baseline AUC: {baseline_test_results["auc"]:.3f}, adjusted AUC: {test_results["auc"]:.3f}')
    # 3.2) train the unadjusted model
    if unadjusted:
        unadjusted_test_results, _ = train_trial_model(X_tr_unadjusted, y_tr, X_tst_unadjusted, y_tst,
                                        conf_cols, t, CV=True, se_rule=False)
        print(f'[Trial {t}] unadjusted AUC: {unadjusted_test_results["auc"]:.3f}')

    return test_results, unadjusted_test_results, baseline_test_results, test_result_se




    
def train_trial_model(X_tr, y_tr, X_tst, y_tst, conf_cols, t, CV, se_rule=None, lambda_value=None):
    
        # 1) Normalize and impute the data wihtout leakage
        X_tr, X_tst = _normalize_and_impute_data(X_tr, X_tst)

        # 2) remove the regularization of the confounders
        reg_weights = _reg_weights(X_tr, conf_cols)

        # 3) handle imbalance and intercept
        class_w_tr = _class_weights(y_tr) # w1 is the weight for the positive class, w0 is the weight for the negative class
        X_tr, X_tst, reg_weights = _add_intercept(X_tr, X_tst, reg_weights)


        # 4) find the optimal lambda with cross-validation
        if CV and se_rule=='include_both':
            optimal_lambda, se_lambda = find_1se_lambda(X_tr, y_tr,reg_weights,
                                                        L1_wt, cv_metric,
                                                        reg_inv_list=reg_inv_list,
                                                        se_rule=se_rule,
                                                        n_jobs=1) 
        elif CV and se_rule==False:
            optimal_lambda = find_1se_lambda(X_tr, y_tr,reg_weights,
                                                        L1_wt, cv_metric,
                                                        reg_inv_list=reg_inv_list,
                                                        se_rule=se_rule,
                                                        n_jobs=1) 
        else:
            optimal_lambda = lambda_value


        # 5) train the model with the optimal lambda
        res = statsmodels_train_model(X_tr, y_tr, optimal_lambda*reg_weights, L1_wt, class_w_tr)
        test_results = evaluate_model(res, X_tst, y_tst)
        if se_rule=='include_both':
            res_se = statsmodels_train_model(X_tr, y_tr, se_lambda*reg_weights, L1_wt, class_w_tr)
            se_test_results = evaluate_model(res_se, X_tst, y_tst)
            print(f'[Trial info] AUC: {test_results["auc"]:.3f}, SE AUC: {se_test_results["auc"]:.3f}')
        else:
            se_test_results = None
            print(f'     baseline AUC: {test_results["auc"]:.3f}')

        # 6) evaluate the model to get the ROC and AUC
        return test_results, se_test_results


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


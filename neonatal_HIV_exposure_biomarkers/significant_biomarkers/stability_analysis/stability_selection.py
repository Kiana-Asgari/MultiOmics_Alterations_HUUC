
import numpy as np
import pandas as pd
import statsmodels.api as sm

from neonatal_HIV_exposure_biomarkers.significant_biomarkers.cv_1se import find_1se_lambda
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.evaluation_and_plots import (
    plot_delta_ROC,
    delta_ROC_tables,
    plot_significant_features,
    table_significant_features_all_genders,
    table_significant_features_all_genders_all_countries,
)
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.utils import (
    evaluate_model,
    prepare_data,
    statsmodels_train_model,
)
from common.feature_engineering import FeatureEngineer
from common.modeling_utils import train_test_split_df
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.stability_analysis_utils import (
    save_results,
    load_neonatal_delta_ROC_results,
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

def neonatal_genderwise_score_from_saved_results(country: str):
    results_all, baseline_results_all, _ = load_neonatal_delta_ROC_results(country, -1, se_rule=False)
    results_male, baseline_results_male, _ = load_neonatal_delta_ROC_results(country, 1,   se_rule=False)
    results_female, baseline_results_female, _ = load_neonatal_delta_ROC_results(country, 2, se_rule=False)
    table_significant_features_all_genders(results_all, baseline_results_all,
                                            results_male, baseline_results_male,
                                            results_female, baseline_results_female,
                                            country)
def neonatal_genderwise_score_from_saved_results_all_countries():
    results_male_Both, baseline_results_male_Both, _ = load_neonatal_delta_ROC_results("Both", 1,   se_rule=False)
    results_female_Both, baseline_results_female_Both, _ = load_neonatal_delta_ROC_results("Both", 2, se_rule=False)
    results_male_Kenya, baseline_results_male_Kenya, _ = load_neonatal_delta_ROC_results("Kenya", 1,   se_rule=False)
    results_female_Kenya, baseline_results_female_Kenya, _ = load_neonatal_delta_ROC_results("Kenya", 2, se_rule=False)
    table_significant_features_all_genders_all_countries(results_male_Both,
                                            results_female_Both,
                                            results_male_Kenya,
                                            results_female_Kenya)



def stability_score_from_saved_results(country: str, gender: str, se_rule: bool=False, ROC: bool=False,
significant_features: list=None, freq_threshold: float=0.8):
    results, baseline_results, unadjusted_results = load_neonatal_delta_ROC_results(country, gender, se_rule=se_rule)

    feature_names = results[0]['feature_names']
    if ROC:
        plot_delta_ROC(results, baseline_results,unadjusted_results, country, gender)
        delta_ROC_tables(results, baseline_results, country, gender)

    significant_features = plot_significant_features(results, feature_names, country, gender, freq_threshold=freq_threshold)
    #plot_significant_features(unadjusted_results, feature_names, country, gender, title='unadjusted')
    return significant_features



def neonatal_stability_score(se_rule, save_results_flag: bool = True, unadjusted: bool = False):
    """Train logistic regression with elastic net regularization and balanced class weights."""
    # 1) prepare the raw data 
    full_data, country, gender = prepare_data(confounders)
    breakpoint()       
    X, y, _ = _intial_quality_control(full_data, country)

    # 2) set the number of trials and the seeds
    rng = np.random.default_rng(42)
    n_trials = int(input('Enter number of trials: '))
    old_results, old_baseline_results, old_unadjusted_results = load_neonatal_delta_ROC_results(country, gender, se_rule=se_rule)
    old_n_trials = len(old_results) if unadjusted else len(old_unadjusted_results)
    trial_seeds = rng.integers(low=0, high=1_00_000, size=500)  # Generate independent seeds

    # 3) train the model for each trial
    results, baseline_results, unadjusted_results = [], [], []
    for iter in range(n_trials):
        t = iter + old_n_trials
        test_results, baseline_test_results, unadjusted_test_results = trial_worker(t, X, y, trial_seeds, se_rule, unadjusted=unadjusted)
        results.append(test_results)
        baseline_results.append(baseline_test_results)
        unadjusted_results.append(unadjusted_test_results)
        # 3.5) save the results
        if save_results_flag:
            save_results(test_results, baseline_test_results, unadjusted_test_results, country, gender, se_rule=se_rule)

    # 4) plot the significant features and the delta ROC curve
    plot_significant_features(results, X.columns, country, gender)
    plot_delta_ROC(results, baseline_results, country, gender)

    return results

def trial_worker(t, X, y, trial_seeds, se_rule, unadjusted):

    # 1) split the data into training and testing
    X_tr, X_tst, y_tr, y_tst = train_test_split_df(X, y, 0.7, stratify=True, seed=trial_seeds[t])

    # 2) prepare the confounders only set for the baseline model
    conf_cols = [c for c in confounders + categorical_confounders if c in X_tr.columns]
    X_tr_conf, X_tst_conf = X_tr[conf_cols], X_tst[conf_cols]
    X_tr_unadjusted, X_tst_unadjusted = X_tr.copy().drop(columns=conf_cols), X_tst.copy().drop(columns=conf_cols)
   
    # 3) train the models
    test_results, baseline_test_results, unadjusted_test_results = None, None, None
    if unadjusted:
        test_results = train_trial_model(X_tr, y_tr, X_tst, y_tst,
                                        conf_cols, t, CV=True, se_rule=se_rule)

        baseline_test_results = train_trial_model(X_tr_conf, y_tr, X_tst_conf, y_tst,
                                                    conf_cols, t, CV=False, lambda_value=0)

    else:
        unadjusted_test_results = train_trial_model(X_tr_unadjusted, y_tr, X_tst_unadjusted, y_tst,
                                        conf_cols, t, CV=True, se_rule=se_rule)


    return test_results, baseline_test_results, unadjusted_test_results
    
def train_trial_model(X_tr, y_tr, X_tst, y_tst, conf_cols, t, CV, se_rule=None, lambda_value=None):
    
        # 1) Normalize and impute the data wihtout leakage
        X_tr, X_tst = _normalize_and_impute_data(X_tr, X_tst)

        # 2) remove the regularization of the confounders
        reg_weights = _reg_weights(X_tr, conf_cols)

        # 3) handle imbalance and intercept
        class_w_tr = _class_weights(y_tr) # w1 is the weight for the positive class, w0 is the weight for the negative class
        X_tr, X_tst, reg_weights = _add_intercept(X_tr, X_tst, reg_weights)

        # 4) find the optimal lambda with cross-validation
        if CV:
            optimal_lambda = find_1se_lambda(X_tr, y_tr,reg_weights, L1_wt, cv_metric, se_rule=se_rule) 
        else:
            optimal_lambda = lambda_value
        reg_weights = optimal_lambda * reg_weights

        # 5) train the model with the optimal lambda
        res = statsmodels_train_model(X_tr, y_tr, reg_weights, L1_wt, class_w_tr)  

        # 6) evaluate the model to get the ROC and AUC
        test_results = evaluate_model(res, X_tst, y_tst)
        print(f'trial {t} done with pr_auc {test_results["pr_auc"]:.3f}, auc {test_results["auc"]:.3f}')
        
        return test_results


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

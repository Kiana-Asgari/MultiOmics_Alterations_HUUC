import numpy as np
import pandas as pd
import statsmodels.api as sm
from common.modeling_utils import train_test_split_df
from common.feature_engineering import FeatureEngineer
from joblib import Parallel, delayed
from itertools import combinations
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler

import sys
from neonatal_HIV_exposure_biomarkers.load_neonatal_data import load_neonatal_data
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.FDR.FDR_threshold import select_knockoff
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.FDR.X_knockoff import make_train_knockoffs, make_test_knockoffs
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.lasso_path import lasso_path
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.utils import _class_weights

confounders = ["sex", "birth.weight.g", "alcohol", "smoking", "diabetes", "hypertension",
                "gadelivery","parity", "gravity", "multiple.birth", "maternal.age.delivery", "prepregnancy.bmi"]
categorical_confounders = ["site"]
quality_control = [{'IQR_median_min': 0.05},
                   {'RSD_min': 0.05},
                   {'too_many_missing_values': 0.5}, 
                   {'RSD_max': 100}
                    ]
L1_wt = 0.3
rng = np.random.default_rng(42) 

def _stratify_gender(X, y, demographics, gender):
    if gender != -1:
        male_mask = demographics["sex"] == gender
        X = X[male_mask]
        y = y[male_mask]
        X = X.drop(columns=["sex"])
        demographics = demographics[male_mask]
    return X, y, demographics


def _prepare_data(country):
    biomarkers, HIV_exposure, demographics = load_neonatal_data(country)
    gender = int(input("Enter gender (1 for male, 2 for female, -1 for both): "))
    X = pd.concat([biomarkers, demographics[confounders]], axis=1)
    X = X.drop(columns=[col for col in X.columns if "hgb" in col]) # TODO
    y = HIV_exposure
    X, y, demographics = _stratify_gender(X, y, demographics, gender)

    if country == "Both":
        X = pd.concat([X, demographics[categorical_confounders]], axis=1)

    return X, y, demographics


def _add_knockoff_and_normalize(X, y, trial_seed):
    feature_engineer_0 = FeatureEngineer(quality_control=quality_control,
                                       missing_values='Left_censored_min', # note! leakage! TODO
                                       sample_normalization=None,
                                       data_transform=None, 
                                       feature_normalization='mean_std', #'mean_std', #note!
                                       verbose=False
                                       )
    X = feature_engineer_0.fit_transform(X, verbose=False) 
    # produing the knockoff only once 
    X_tr_knockoff = _get_knockoff_copies(X, seed=trial_seed)
    X = pd.concat([X_tr_knockoff, X], axis=1)
    return X, y

def _trial_worker(t, X_df, y_series, L1_wt, path_score, trial_seeds ):
    seed = trial_seeds[t]
    X_tr, X_val, y_tr, y_val = train_test_split_df(X_df, y_series, 0.8, stratify=True, seed=seed)
    X_tr, y_tr = _add_knockoff_and_normalize(X_tr, y_tr, seed)
    scores = train_lasso_path(X_tr, y_tr, L1_wt=L1_wt, score=path_score)
    trial_significant_biomarkers, T, fdp_hat = select_knockoff(scores, q=0.1, knockoff_plus=False)
    print(f'\n[FDR] trial {t} done with significant biomarkers: {trial_significant_biomarkers}, T: {T}')
    return t, trial_significant_biomarkers, T, fdp_hat





def FDR_knockoff():
    n_trials = int(input('Enter number of trials: '))
    country = input('Enter country (Kenya, Both, Zambia): ')
    path_score = input('Enter path score (lasso_path, frequency, max_coef): ')
    X, y, demographics = _prepare_data(country)
    
    results = []
    significant_biomarkers = {}
    fdp_hats = []
    trial_seeds = rng.integers(low=0, high=1_00_000, size=n_trials)  # Generate 100 independent seeds
    
    trial_outputs = Parallel(n_jobs=15)(delayed(_trial_worker)(t, X, y, L1_wt, path_score, trial_seeds) for t in range(n_trials))


    for t, trial_significant_biomarkers, T, fdp_hat in trial_outputs:
        if len(trial_significant_biomarkers) >= 0: # TODO
            i = len(significant_biomarkers.keys())
            significant_biomarkers[f'trial_{i}'] = trial_significant_biomarkers
            fdp_hats.append(fdp_hat)
    
    final_significant_biomarkers = aggregate_significant_biomarkers(significant_biomarkers, X)
    print(f'fdp_hats: {fdp_hats}')
    print(f'mean_fdp_hat: {np.mean(fdp_hats)},std_fdp_hat: {np.std(fdp_hats)}')

    return final_significant_biomarkers


def aggregate_significant_biomarkers(significant_biomarkers, X):
    feature_frequency = {}
    n_trials = len(significant_biomarkers.keys())
    for feature in X.columns:
        feature_frequency[feature] = 0

    for key in significant_biomarkers.keys():
        print(f'[FDR] trial {key} done with significant biomarkers: {significant_biomarkers[key]}')
        for feature in significant_biomarkers[key]:
            feature_frequency[feature] += 1
        feature_frequency[feature] = feature_frequency[feature]

    for feature in feature_frequency.keys():
         feature_frequency[feature] = feature_frequency[feature] / n_trials * 100
         if feature_frequency[feature] > 0:
            print(f'[FDR] {feature}, {feature_frequency[feature]}')

    # compute Jaccard similarity between chosen markers and significant features of each trial
    threshold_vs_markers = {}
    for threshold in np.arange(0, 100, 5):
        chosen_markers = [] 
        for feature in X.columns:
            if feature_frequency[feature] > threshold:
                chosen_markers.append(feature)
        
        jaccard_scores = []
        for key in significant_biomarkers.keys():
            trial_markers = set(significant_biomarkers[key])
            chosen_set = set(chosen_markers)
            if len(chosen_set | trial_markers) > 0:
                # Create binary vectors for all features
                all_features = list(chosen_set | trial_markers)
                chosen_vec = [1 if f in chosen_set else 0 for f in all_features]
                trial_vec = [1 if f in significant_biomarkers[key] else 0 for f in all_features]
                jaccard_scores.append(jaccard_score(chosen_vec, trial_vec, average='macro'))
        
        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
        threshold_vs_markers[threshold] = {
            'chosen_markers': chosen_markers,
            'avg_jaccard': avg_jaccard
        }
        #print(f'[FDR] threshold {threshold}, avg_jaccard: {avg_jaccard}, biomarkers: {chosen_markers}')
    
    return threshold_vs_markers[50]['chosen_markers']





def train_lasso_path(X_tr, y_tr, L1_wt, score='lasso_path'):
    scaler  = FeatureEngineer(quality_control=None,
                                       missing_values='Left_censored_min', 
                                       treat_zero_values=False,
                                       sample_normalization=None,
                                       data_transform= None, 
                                       feature_normalization='mean_std', 
                                       verbose=False
                                       )
    X_tr = scaler.fit_transform(X_tr)

    num_confounders = len([col for col in X_tr.columns if col in confounders])

    reg_weights = np.ones(X_tr.shape[1]) * 1
    reg_weights[-num_confounders:] = 0 # don't regularize confounders (appearning at the end of the array)

    # 4) train the model
    class_w_tr = _class_weights(y_tr) 

    # Add constant term for intercept
    reg_weights = np.insert(reg_weights, 0, 0) 
    X_tr = sm.add_constant(X_tr)


    path_results, largest_coef, frequency = lasso_path(X_tr, y_tr, class_w_tr, reg_weights, L1_wt)

    scores = {}
    for feature in X_tr.columns:
        if 'knockoff' not in feature and f'knockoff_{feature}' in path_results:
            original_lambda, original_coef, original_frequency = path_results[feature], largest_coef[feature], frequency[feature]
            knockoff_lambda, knockoff_coef, knockoff_frequency = path_results[f'knockoff_{feature}'], largest_coef[f'knockoff_{feature}'], frequency[f'knockoff_{feature}']

            if score == 'lasso_path':
                scores[feature] = original_lambda - knockoff_lambda #original_coef - knockoff_coef
            elif score == 'frequency':
                scores[feature] = original_frequency - knockoff_frequency #original_lambda - knockoff_lambda #original_coef - knockoff_coef
            elif score == 'max_coef':
                scores[feature] = original_coef - knockoff_coef #original_lambda - knockoff_lambda #original_coef - knockoff_coef
            else:
                raise ValueError(f'Invalid score: {score}')


    return scores



def _get_knockoff_copies(X_tr, seed):
    num_confounders = len([col for col in X_tr.columns if col in confounders or col in categorical_confounders])
    num_biomarkers = X_tr.shape[1] - num_confounders

    # Convert pandas DataFrames to numpy arrays for knockoff functions
    col_names = [col for col in X_tr.columns if (col not in confounders ) and (col not in categorical_confounders)]

    X_tr_array = X_tr.values

    X_tr_biom = X_tr_array[:,:num_biomarkers] # only the biomarkers column
    
    X_tr_knockoff, ko_params = make_train_knockoffs(X_tr_biom, seed)

    # Convert numpy arrays back to DataFrames for concatenation
    X_tr_knockoff_df = pd.DataFrame(X_tr_knockoff, index=X_tr.index, 
                                    columns=[f'knockoff_{biom}' for biom in col_names])

    return X_tr_knockoff_df








########################################################
# utils
########################################################


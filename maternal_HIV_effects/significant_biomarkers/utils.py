import numpy as np
import pandas as pd
import os
import json

from maternal_HIV_effects.load_maternal_data import load_maternal_data
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import shap

# Helpers for preparing the data
def prepare_data(biomarkers_type: str):
    biomarkers, clinical_data, y = load_maternal_data(biomarkers_type=biomarkers_type)
    X = pd.concat([biomarkers, clinical_data], axis=1)
    return (X, y, clinical_data)
    

def statsmodels_train_model(X_tr, y_tr, reg_weights, L1_wt, class_w_tr, tol=1e-6):
    """Train logistic regression with elastic net regularization and balanced class weights."""
    mod = sm.GLM(y_tr, X_tr, family=sm.families.Binomial(), freq_weights=class_w_tr)
    res = mod.fit_regularized(method="elastic_net",
                                alpha=reg_weights,
                                L1_wt=L1_wt,
                                maxiter=3000, cnvrg_tol=tol)
    return res


def evaluate_model(res, X_tst, y_tst):
    proba_te = res.predict(X_tst)  # predicted P(Y=1)
    y_pred = (proba_te >= 0.5).astype(int)

    # Calculate metrics
    precision = precision_score(y_tst, y_pred, zero_division=0)
    recall = recall_score(y_tst, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_tst, proba_te)

    # Calculate Type 1 Error (False Positive Rate) and Type 2 Error (False Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_tst, y_pred).ravel()
    type_1_error = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    type_2_error = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # Calculate PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_tst, proba_te)
    pr_auc = auc(recall_curve, precision_curve)
  
    results = {
        'model': res.params,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc,
        'pr_auc': pr_auc,
        'type 1 error': type_1_error,
        'type 2 error': type_2_error,
        'log_odds_ratio': res.params[1:],
        'feature_names': X_tst.columns[1:],
    }
    fpr, tpr, thresholds_roc = roc_curve(y_tst, proba_te, drop_intermediate=False)
    results["ROC_curve"] = (fpr, tpr, thresholds_roc)
    return results
    







# Helper functions for saving and loading results
def _convert_to_serializable(obj):
    """Convert numpy arrays, pandas Series/DataFrames/Index to JSON-serializable formats."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='list')
    elif isinstance(obj, pd.Index):
        return obj.tolist()
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {_convert_to_serializable(k): _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    return obj


def save_results(test_results: dict, 
                unadjusted_test_results: dict,
                baseline_test_results: dict,
                test_result_se: dict,
                biomakers_type: str):

    save_json(test_results, filename=f'delta_ROC_test_results_{biomakers_type}.json', dir=f'maternal_HIV_effects/significant_biomarkers/results/stability_selection/{biomakers_type}')
    
    save_json(unadjusted_test_results, filename=f'delta_ROC_unadjusted_test_results_{biomakers_type}.json', dir=f'maternal_HIV_effects/significant_biomarkers/results/stability_selection/{biomakers_type}')
    
    save_json(baseline_test_results, filename=f'baseline_ROC_test_results_{biomakers_type}.json', dir=f'maternal_HIV_effects/significant_biomarkers/results/stability_selection/{biomakers_type}')

    save_json(test_result_se, filename=f'delta_ROC_test_result_se_{biomakers_type}.json', dir=f'maternal_HIV_effects/significant_biomarkers/results/stability_selection/{biomakers_type}')

def load_maternal_results(biomarkers_type: str):
    dir = f'maternal_HIV_effects/significant_biomarkers/results/stability_selection/{biomarkers_type}'
    name = f'delta_ROC_test_results_{biomarkers_type}.json'
    baseline_name = f'baseline_ROC_test_results_{biomarkers_type}.json'
    se_name = f'delta_ROC_test_result_se_{biomarkers_type}.json'
    unadjusted_name = f'delta_ROC_unadjusted_test_results_{biomarkers_type}.json'

    results = load_json(name, dir)
    baseline_results = load_json(baseline_name, dir)
    se_results = load_json(se_name, dir)    
    unadjusted_results = load_json(unadjusted_name, dir)

    return results, unadjusted_results, baseline_results, se_results







def load_lasso_path(biomakers_type: str):
    dir = f'maternal_HIV_effects/significant_biomarkers/results/lasso_path/{biomakers_type}'
    name = f'lasso_path_{biomakers_type}.json'
    name2 = f'lasso_path_{biomakers_type}2.json'
    name3 = f'lasso_path_{biomakers_type}3.json'
    stability_path = load_json(name, dir)
    stability_path2 = load_json(name2, dir)
    stability_path3 = load_json(name3, dir)
    
    if not stability_path and not stability_path2 and not stability_path3:
        return []
    
    # Flatten if nested (handling both list of dicts and list of lists)
    if stability_path and isinstance(stability_path[0], list):
        # Flatten nested list structure
        flattened = []
        for item in stability_path:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        stability_path = flattened
    
    # Convert lambda keys from strings to floats and coef dicts to Series
    converted_stability_path_1 = []
    for trial_path in stability_path:
        if isinstance(trial_path, dict):
            converted_stability_path_1.append(
                {float(lambda_): pd.Series(coefs) for lambda_, coefs in trial_path.items()}
            )
    
    if stability_path2 and isinstance(stability_path2[0], list):
        # Flatten nested list structure
        flattened = []
        for item in stability_path2:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        stability_path2 = flattened
    converted_stability_path2 = []
    for trial_path in stability_path2:
        if isinstance(trial_path, dict):
            converted_stability_path2.append(
                {float(lambda_): pd.Series(coefs) for lambda_, coefs in trial_path.items()}
            )

    if stability_path3 and isinstance(stability_path3[0], list):
        # Flatten nested list structure
        flattened = []
        for item in stability_path3:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        stability_path3 = flattened
    converted_stability_path3 = []
    for trial_path in stability_path3:
        if isinstance(trial_path, dict):
            converted_stability_path3.append(
                {float(lambda_): pd.Series(coefs) for lambda_, coefs in trial_path.items()}
            )
    converted_stability_path_12 = converted_stability_path_1 + converted_stability_path2

    converted_stability_path = _combine_stability_paths_lambdas(converted_stability_path_12, converted_stability_path3)
    
    return list(converted_stability_path)


def _combine_stability_paths_lambdas(stability_path_12, stability_path3):
    combined_stability_path = []
    if len(stability_path_12) == 0:
        return stability_path3
    if len(stability_path3) == 0:
        return stability_path_12

    n_trials = np.min([len(stability_path_12), len(stability_path3)])
    for t in range(n_trials):
        trial_path_12 = stability_path_12[t]
        trial_path_3 = stability_path3[t]
        # concatinate the two dictionary trilas
        full_trial_path = {**trial_path_12, **trial_path_3}
        # sort the dictionary by lambda (keys)
        full_trial_path = dict(sorted(full_trial_path.items()))
        combined_stability_path.append(full_trial_path)

    return combined_stability_path



def save_json(data: dict, filename: str, dir: str):

    if data is None or data == [] or None in data:
        print(f'[warning] saving {filename} failed because data is None or data == [] or None in data')
        return 

    os.makedirs(dir, exist_ok=True)
    # Convert results to JSON-serializable format (exclude the model object)
    data_serializable = _convert_to_serializable(data)
    # Load existing results if they exist
    previous_data_list = []

    if os.path.exists(os.path.join(dir, filename)):
        try:
            with open(os.path.join(dir, filename), 'r') as f:
                previous_data_list = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted JSON file found at {filename}. Starting fresh.")
            previous_data_list = []
    
    # Append new results
    previous_data_list.append(data_serializable)

    # Save to JSON with indentation for readability
    with open(os.path.join(dir, filename), 'w') as f:
        json.dump(previous_data_list, f)


def load_json(filename: str, dir: str):
    # if exists, load the data, otherwise return an empty list
    if os.path.exists(os.path.join(dir, filename)):
        with open(os.path.join(dir, filename), 'r') as f:
            data = json.load(f)
        return data
    else:
        print(f'no json file found at {filename}')
        return []
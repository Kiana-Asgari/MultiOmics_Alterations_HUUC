import numpy as np
import pandas as pd
import os
import json

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
                baseline_test_results: dict, 
                unadjusted_test_results: dict,
                country: str, 
                gender: str, se_rule: bool):


    save_json(test_results, filename=f'delta_ROC_{country}_{gender}_test_results_se_rule_{se_rule}.json', dir='neonatal_HIV_exposure_biomarkers/results/stability_selection')

    save_json(baseline_test_results, filename=f'baseline_ROC_{country}_{gender}_test_results_se_rule_{se_rule}.json', dir='neonatal_HIV_exposure_biomarkers/results/stability_selection')

    save_json(unadjusted_test_results, filename=f'delta_ROC_{country}_{gender}_test_results_unadjusted.json', dir='neonatal_HIV_exposure_biomarkers/results/stability_selection')

def load_neonatal_delta_ROC_results(country: str, gender: str, se_rule: bool):
    dir = 'neonatal_HIV_exposure_biomarkers/results/stability_selection'
    name = f'delta_ROC_{country}_{gender}_test_results_se_rule_{se_rule}.json'
    baseline_name = f'baseline_ROC_{country}_{gender}_test_results_se_rule_{se_rule}.json'
    unadjusted_name = f'delta_ROC_{country}_{gender}_test_results_unadjusted.json'

    results = load_json(name, dir)
    baseline_results = load_json(baseline_name, dir)
    unadjusted_results = load_json(unadjusted_name, dir)

    return results, baseline_results, unadjusted_results


def save_stability_path(stability_path: dict, country: str, gender: str):
    #stability_path : {lambda: {feature: coef}}
    save_json(stability_path, filename=f'stability_path_{country}_{gender}.json', dir='neonatal_HIV_exposure_biomarkers/results/stability_path')
    

def load_stability_path(country: str, gender: str):
    dir = 'neonatal_HIV_exposure_biomarkers/results/stability_path'
    name = f'stability_path_{country}_{gender}.json'
    stability_path = load_json(name, dir)
    
    # Convert lambda keys from strings to floats and coef dicts to Series
    converted_stability_path = [
        {float(lambda_): pd.Series(coefs) for lambda_, coefs in trial_path.items()}
        for trial_path in stability_path
    ]
    
    return converted_stability_path



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
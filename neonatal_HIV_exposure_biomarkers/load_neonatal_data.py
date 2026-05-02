import numpy as np
import pandas as pd
import sys
from typing import Dict
from configs import config
from common.data_loader import MaternalDataLoader, get_lipidome_metabolome_mom
from common.data_loader import MaternalDataLoader, MaternalDataProcessor, divide_heel_blood_sample

from utils import Verbose

def _raw_data(country, patient_type, genom_type, target_column, year_collected):
    loader = MaternalDataLoader(country=country, genom_type=genom_type, year_collected=year_collected)
    processor = MaternalDataProcessor(loader, target_column=target_column, verbose=False)
    
    full_raw_info = processor.process_data(drop_unlabelled=True)
    neonatal_data = {
        'biomarkers': full_raw_info[0][patient_type],
        'HIV_exposure': full_raw_info[2][patient_type], 
        'clinical_data': full_raw_info[1][patient_type]
    }
    # if Zambiya, change the column getational.age.estimate to gadelivery
    if country == "Zambia":
        neonatal_data['clinical_data']['gadelivery'] = neonatal_data['clinical_data']['getational_age_estimate']
        neonatal_data['clinical_data'] = neonatal_data['clinical_data'].drop(columns=['getational_age_estimate'])
   
    return neonatal_data


def load_neonatal_data(country, source="HEEL", drop_ratios=True):
    """Load the neonatal data for Kenya or Zambia"""
    year_collected, genom_type, patient_type, target_column = config.neonatal_sources
    neonatal_data_kenya = _raw_data("Kenya", patient_type, genom_type, target_column, year_collected)
    neonatal_data_zambia = _raw_data("Zambia", patient_type, genom_type, target_column, year_collected)      

    
    # # Clean the data
    neonatal_data_zambia = _clean_data(neonatal_data_zambia, drop_ratios)
    neonatal_data_kenya = _clean_data(neonatal_data_kenya, drop_ratios)
    neonatal_data_kenya['clinical_data']['site'] = 1
    neonatal_data_zambia['clinical_data']['site'] = 2


    if country == "Both": 
        neonatal_info = {
            'biomarkers': pd.concat([neonatal_data_kenya['biomarkers'], neonatal_data_zambia['biomarkers']], axis=0, join='inner'),
            'HIV_exposure': pd.concat([neonatal_data_kenya['HIV_exposure'], neonatal_data_zambia['HIV_exposure']], axis=0, join='inner'),
            'clinical_data': pd.concat([neonatal_data_kenya['clinical_data'], neonatal_data_zambia['clinical_data']], axis=0, join='inner')
        }  
    elif country == "Kenya":
        neonatal_info = neonatal_data_kenya
    elif country == "Zambia":
        neonatal_info = neonatal_data_zambia
    else:
        raise ValueError(f"Invalid country: {country}. Aborting...")
    
    neonatal_info = divide_heel_blood_sample(neonatal_info)

    return neonatal_info[source]


def _clean_data(neonatal_data: Dict[str, pd.DataFrame], drop_ratios=True):
    """Clean the names and remove the ratio columns"""
    neonatal_data['biomarkers'].columns = _clean_columns(neonatal_data['biomarkers'].columns)
    neonatal_data['clinical_data'].columns = _clean_columns(neonatal_data['clinical_data'].columns)
    if drop_ratios:
        neonatal_data['biomarkers'] = _remove_ratios(neonatal_data['biomarkers'])
    neonatal_data['clinical_data'] = _handle_categorical_columns(neonatal_data['clinical_data'])


    return neonatal_data


########################################################
# Helper functions to clean the data
########################################################

def _handle_categorical_columns(X: pd.DataFrame):
    """Handle the categorical columns with safe mapping."""
    mapping_dict = {'Male': 1, 'Female': 2, 'No': 1, 'Yes': 2, 'Ye': 2}
    
    for col in X.columns:
        mask = X[col].isin(mapping_dict.keys())
        if mask.any():
            X[col] = X[col].astype(object)
            X.loc[mask, col] = X.loc[mask, col].map(mapping_dict)
    
    return X


def _clean_columns(cols):
    cols = cols.str.replace('_', '.', regex=False).str.replace('-', '.', regex=False).str.replace(' ', '.', regex=False).str.lower()
    cols = cols.str.replace('preeclampsia.eclampsia', 'preeclampsia', regex=False)
    return cols

def _lipid_ratio_columns(words_list):
    multi_word = words_list.str.len() > 1
    c_counts = words_list.apply(lambda words: sum('c' in word for word in words))
    has_multiple_c = c_counts >= 2
    return multi_word & has_multiple_c

def _nonlipid_ratio_columns(words_list, X):
    two_words = words_list.str.len() == 2
    no_digits = ~X.columns.str.contains(r'\d')
    return two_words & no_digits


def _remove_ratios(X):
    """Remove ratio columns based on naming patterns."""
    words_list = pd.Series(X.columns.str.split('.'))
    
    # Condition 1: Multiple words with 2+ 'c' characters
    lipid_ratio_columns = _lipid_ratio_columns(words_list)
    
    # Condition 2: Exactly 2 words with no digits
    nonlipid_ratio_columns = _nonlipid_ratio_columns(words_list, X)



    # Combine conditions
    ratio_mask = lipid_ratio_columns | nonlipid_ratio_columns# TODO
    
    # Define columns to keep and drop
    keep_ratios = ['orn.arg', 'tyr.phe', 'met.phe'] #TODO: need to check if this is correct
    
    # Extract unique amino acids from ratio names
    #drop_raw = ['phe']
    
    # Add columns to drop (set to True for columns in drop_raw)
    # for col in drop_raw:
    #     if col in X.columns:
    #         ratio_mask[X.columns.get_loc(col)] = True
    
    # # Remove columns to keep (set to False for columns in keep_ratios)
    # for col in keep_ratios:
    #     if col in X.columns:
    #         ratio_mask[X.columns.get_loc(col)] = False
    X_cleaned = X.drop(columns=X.columns[ratio_mask]) # TODO

    
    return X_cleaned
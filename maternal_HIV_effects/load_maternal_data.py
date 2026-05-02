import numpy as np
import pandas as pd
import sys
from typing import Dict, Literal
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


def load_maternal_data(drop_gadelivery: bool=True, biomarkers_type: Literal['Lipidome', 'Metabolome', 'Both'] = 'Both'):
    """Load the maternal lipidom and metabolome data for Kenya"""

    maternal_lipidome_data = _raw_data("Kenya", 'Mom', 'Lipidome', 'HIV', None)
    maternal_metabolome_data = _raw_data("Kenya", 'Mom', 'Metabolome', 'HIV', None)

    biomarkers, clinical_data, HIV_status = _clean_data(maternal_lipidome_data=maternal_lipidome_data,
                                                        maternal_metabolome_data=maternal_metabolome_data, 
                                                        drop_gadelivery=drop_gadelivery,
                                                        biomarkers_type=biomarkers_type)
    return biomarkers, clinical_data, HIV_status





def _clean_data(maternal_lipidome_data: Dict[str, pd.DataFrame],
                maternal_metabolome_data: Dict[str, pd.DataFrame],
                drop_gadelivery: bool, biomarkers_type: str):
    """Clean the names and remove the ratio columns"""
    biomarkers, clinical_data, HIV_status = _merge_data(maternal_lipidome_data, maternal_metabolome_data, biomarkers_type)

    biomarkers.columns = _clean_columns(biomarkers.columns)
    clinical_data.columns = _clean_columns(clinical_data.columns)
    biomarkers = _remove_ratios(biomarkers)
    clinical_data = _handle_categorical_columns(clinical_data, drop_gadelivery)


    return (biomarkers, clinical_data, HIV_status)


########################################################
# Helper functions to clean the data
########################################################

def _merge_data(maternal_data_lipidome, maternal_data_metabolome, biomarkers_type: str):
    """Merge the lipidome and metabolome data for the same patients"""   
    # Set STUDY.ID as index for all dataframes to ensure alignment
    lipid_clinical = maternal_data_lipidome['clinical_data'].set_index('STUDY.ID')
    metabol_clinical = maternal_data_metabolome['clinical_data'].set_index('STUDY.ID')
    
    # Set index for biomarkers using STUDY.ID from clinical_data
    lipid_biomarkers = maternal_data_lipidome['biomarkers'].copy()
    lipid_biomarkers.index = maternal_data_lipidome['clinical_data']['STUDY.ID'].values
    
    metabol_biomarkers = maternal_data_metabolome['biomarkers'].copy()
    metabol_biomarkers.index = maternal_data_metabolome['clinical_data']['STUDY.ID'].values
    
    # Set index for HIV_exposure (it's a Series)
    lipid_HIV = maternal_data_lipidome['HIV_exposure'].copy()
    lipid_HIV.index = maternal_data_lipidome['clinical_data']['STUDY.ID'].values
    
    # Get shared patient IDs
    shared_patients_id = lipid_clinical.index.intersection(metabol_clinical.index)
    
    # Filter by shared IDs (already aligned by index)
    _lipidome = lipid_biomarkers.loc[shared_patients_id]
    _metabolome = metabol_biomarkers.loc[shared_patients_id]
    
    # Concatenate biomarkers (rows are aligned by STUDY.ID index)
    biomarkers = pd.concat([_lipidome, _metabolome], axis=1).reset_index(drop=True)
    
    # Get clinical data and HIV status (already aligned by index)
    clinical_data = lipid_clinical.loc[shared_patients_id].reset_index(drop=True)
    HIV_status = lipid_HIV.loc[shared_patients_id].reset_index(drop=True)
    if biomarkers_type == 'Both':
        return biomarkers, clinical_data, HIV_status
    elif biomarkers_type == 'Lipidome':
        return _lipidome.reset_index(drop=True), clinical_data, HIV_status
    elif biomarkers_type == 'Metabolome':
        return _metabolome.reset_index(drop=True), clinical_data, HIV_status




def _handle_categorical_columns(X: pd.DataFrame, drop_gadelivery: bool):
    """Handle the categorical columns with safe mapping."""
    mapping_dict = {'Male': 1, 'Female': 2, 'No': 1, 'Yes': 2, 'Ye': 2, 'True': 1, 'False': 2, 'true': 1, 'false': 2}
    if drop_gadelivery:
        drop_columns = ['hiv', 'site', 'study.id', 'gadelivery', 'birthweight'] 
    else:
        drop_columns = ['hiv', 'site', 'study.id'] 
    if 'site' in X.columns and len(X['site'].unique()) != 1:
        raise ValueError(f"site column has {len(X['site'].unique())} unique values, expected 1")

    for col in drop_columns:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    for col in X.columns:
        mask = X[col].isin(mapping_dict.keys())
        if mask.any():
            X[col] = X[col].astype(object)
            X.loc[mask, col] = X.loc[mask, col].map(mapping_dict)
    
    return X


def _clean_columns(cols):
    return cols.str.replace(r'[._-]', '.', regex=True).str.lower()

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
    return X
    #TODO: remove this
    words_list = pd.Series(X.columns.str.split('.'))
    
    # Condition 1: Multiple words with 2+ 'c' characters
    lipid_ratio_columns = _lipid_ratio_columns(words_list)
    
    # Condition 2: Exactly 2 words with no digits
    nonlipid_ratio_columns = _nonlipid_ratio_columns(words_list, X)



    # Combine conditions
    ratio_mask = lipid_ratio_columns | nonlipid_ratio_columns# TODO
    
    print(f'\n[INFO data loader] dropping {X.columns[ratio_mask]}\n')

    X_cleaned = X.drop(columns=X.columns[ratio_mask]) # TODO
    return X_cleaned
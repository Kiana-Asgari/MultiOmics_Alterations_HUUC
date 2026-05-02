from data.utils import read_full_data, standardize_columns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import os
from configs import config
from typing import Literal, Tuple, Dict
from utils import Verbose

class MaternalDataLoader:
    """Handles data loading and basic preprocessing"""
    
    def __init__(self, country: Literal[config.countries], genom_type: Literal['Lipidome', 'Metabolome'],
                 year_collected: Literal['2019-2020', '2019-2021']=None):
        self.country = country
        self.genom_type = genom_type
        self.year_collected = year_collected
        self.raw_data = {}
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data files"""
        if self.country == "Zambia":
            self.raw_data['Newborn'] = read_full_data(self.country, data_type=self.year_collected)
            self.raw_data['Mom'] = None
            self.raw_data['Dyad'] = None

        elif self.country == "Kenya":
            self.raw_data['Newborn'] = read_full_data(self.country, "Newborn", self.genom_type)
            self.raw_data['Mom'] = read_full_data(self.country, "Mom", self.genom_type)
            self.raw_data['Dyad'] = self._get_maternal_dyad_data()
        else:
            raise ValueError(f"Invalid country: {self.country}. Aborting...")

        return self.raw_data
    
    def _get_maternal_dyad_data(self) -> pd.DataFrame:
        """Get merged maternal-infant data"""
        if self.raw_data['Newborn'] is None or self.raw_data['Mom'] is None:
            return None

        merged = pd.merge(
            self.raw_data['Newborn'].add_prefix('newborn_info:'),
            self.raw_data['Mom'].add_prefix('mom_info:'),
            left_on='newborn_info:New.Id',
            right_on='mom_info:STUDY.ID',
            how='inner'
        )
        
        # Remove duplicate columns       
        merged = merged.drop(columns=[
            'mom_info:STUDY.ID', 
            'newborn_info:StudyID', 
            'newborn_info:Hiv.Status'
        ])
        
        return merged

########################################################################################
class MaternalDataProcessor:
    """Handles data cleaning and splitting"""
    
    def __init__(self, data_loader: MaternalDataLoader, target_column: str = 'HIV', verbose: bool = True):
        self.data_loader = data_loader
        self.target_column = target_column
        self.country = data_loader.country
        self.genom_type = data_loader.genom_type
        self.printer = Verbose(verbose=verbose)

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data files"""
        raw_data = self.data_loader.load_raw_data()
        self.printer.print_table_raw_data(raw_data)
        return raw_data
    

    def calculate_SGA(self, X_newborn, weight_col="Birth.Weight.G", GA_col="GAdelivery", sex_col="Sex") -> pd.Series:
        # Growth chart data (10th percentile weights by gestational age and sex)
        chart_df = pd.DataFrame(config.growth_chart())
        sga_labels = []
        X_newborn.columns = X_newborn.columns.str.lower()
        
        for i in range(len(X_newborn)):
            weight = X_newborn[weight_col.lower()].iloc[i]
            ga = X_newborn[GA_col.lower()].iloc[i]
            sex = X_newborn[sex_col.lower()].iloc[i]
            
            # Round GA to nearest integer for lookup
            ga_rounded = round(ga)
            if ga_rounded < 25:
                ga_rounded = 25
            elif ga_rounded > 46:
                ga_rounded = 46
            
                sga_labels.append(0)  
                continue
            
            # Get threshold weight for this GA and sex
            if sex.lower() == 'male':
                threshold_weight = chart_df[chart_df['GAdelivery'] == ga_rounded]['BWboys'].iloc[0]
            elif sex.lower() == 'female':
                threshold_weight = chart_df[chart_df['GAdelivery'] == ga_rounded]['BWgirls'].iloc[0]
            else:
                raise ValueError(f"Invalid sex: {sex}. Aborting...")
            
            sga_status = 1 if weight <= threshold_weight else 0
            sga_labels.append(sga_status)
        
        return pd.Series(sga_labels, index=X_newborn.index, name='SGA')


    def process_data(self, drop_unlabelled: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Process and split data"""
        self.raw_data = self.load_raw_data()        
        cleaned_data = self._clean_data(self.raw_data)
        split_data = self._split_data(cleaned_data, drop_unlabelled)
          
        # Standardize labels
        for key in split_data[2].keys():
            split_data[2][key] = self._standardize_labels(split_data[2][key])
        if self.genom_type == "Metabolome" and self.country == "Kenya":
            split_data[0]['Mom'] = annotate_metabolomes_kenya(split_data[0]['Mom'])


        # fix the string columns
        if self.country == "Kenya":
            split_data[0]['Newborn'] = self._fix_string_columns(split_data[0]['Newborn'])
            split_data[0]['Mom'] = self._fix_string_columns(split_data[0]['Mom'])
        elif self.country == "Zimbabwe":
            pass
        
        # Create table data
        self.printer.print_table_split_data(split_data)
        return split_data


    
    def _clean_data(self, raw_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Clean and standardize the data"""
        cleaned_data = {}
        for key, df in raw_data_dict.items():
            if df is None:
                cleaned_data[key] = None
            else:
                cleaned_data[key] = standardize_columns(df)
                cleaned_data[key] = cleaned_data[key].dropna(subset=[self.target_column])
        return cleaned_data
    
    def _split_data(self, cleaned_data_dict: Dict[str, pd.DataFrame], drop_unlabelled: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Split data into features, demographics, and target"""
        X_dict = {}
        y_dict = {}
        demographic_data_dict = {}
        
        for key, df in cleaned_data_dict.items():
            if df is not None:
                if drop_unlabelled:
                    dropped_row_indices = df[self.target_column][df[self.target_column] == 0].index
                    df = df.drop(dropped_row_indices, axis=0)
                
                # TODO: I am using Kenya for Zambia as well here
                demographic_cols_config = config.extract_features(self.country, key, 'demographic_columns', self.genom_type) 
                demographic_cols = [col for col in df.columns if col in demographic_cols_config]
                metabolic_cols = [col for col in df.columns if col not in demographic_cols and col != "HIV"]
                X_dict[key], y_dict[key], demographic_data_dict[key] = df[metabolic_cols], df[self.target_column], df[demographic_cols]
            else:
                # Handle None values
                X_dict[key], y_dict[key], demographic_data_dict[key] = None, None, None 


        return X_dict, demographic_data_dict, y_dict
    

    def _standardize_labels(self, y: pd.Series) -> pd.Series:  
        if y is None:
            return y
        value_counts = y.value_counts()
        values = value_counts.index.tolist()
        if len(value_counts) == 2 and "No" in values and "Yes" in values:
            y = y.map({"No": 0, "Yes": 1})
            self.printer.print(f"[standardize_labels] No->0, Yes->1")
        elif len(value_counts) == 2:
            most_common = value_counts.index[0]
            less_common = value_counts.index[1]

            y = y.map({most_common: 0, less_common: 1})
            self.printer.print(f"[standardize_labels] {value_counts.index[0]}->0, {value_counts.index[-1]}->1")
        else:
            print(f"Warning: Target column {self.target_column} has {len(value_counts)} unique values, expected 2; Standardizing failed")
            y = y.map({value_counts.index[0]: 0, value_counts.index[-1]: 1, value_counts.index[1]: -1})
            self.printer.print(f"[standardize_labels] {value_counts.index[0]}->0, {value_counts.index[-1]}->1, {value_counts.index[1]}->-1")
        return y

    def _fix_string_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        if X is None or X.empty:
            return X
            
        X_fixed = X.copy()
        string_columns = X_fixed.select_dtypes(exclude=['int64', 'int32', 'float64', 'float32']).columns
        
        for col in string_columns:            
            unique_values = X_fixed[col].dropna().unique()
            if len(unique_values) > 0:
                if col == 'Source':
                    continue
                # mapping to 1,2 instead of 0,1 to avoid confusion with missing values
                one_hot_encoded = 1 + pd.get_dummies(X_fixed[col], prefix=col, dummy_na=False)
                X_fixed = X_fixed.drop(columns=[col])
                X_fixed = pd.concat([X_fixed, one_hot_encoded], axis=1)
        
        return X_fixed

    def _anonymize_data(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """1) add a column named: ID as the first column
            This column contains the row index of the data
            2) add the target column to the dataframe after the ID column, named as label
            3) remove all the other column names, and replace with the M+column index
            4) Saves as a csv file with the name: anonymized_data.csv in the data/processed folder
        """
        X_anon = X.copy().reset_index().rename(columns={'index': 'ID'})
        X_anon.insert(1, 'label', y.values)
        X_anon.columns = ['ID', 'label'] + [f'M{i}' for i in range(len(X.columns))]

        # keep the same number of negative samples as the number of positive samples
        num_positive_samples = y.value_counts()[1]
        negative_samples_to_keep = 0

        for i, row in X_anon.iterrows():
            if negative_samples_to_keep >= num_positive_samples:
                if row['label'] == 0:
                    X_anon = X_anon.drop(i, axis=0)
            elif row['label'] == 0:
                negative_samples_to_keep += 1

        X_anon.to_csv('data/processed/anonymized_data.csv', index=False)
        self.printer.print('shape of anonymized data after removing extra negative samples: ', X_anon.shape)

        return X_anon

########################################################################################

def divide_heel_blood_sample(full_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    # Get heel blood samples
    X, y, demographics = full_data["biomarkers"], full_data["HIV_exposure"], full_data["clinical_data"]
    heel_samples_indices = X["source"] == "HEEL"
    X_heel = X[heel_samples_indices].copy()
    y_heel = y[heel_samples_indices]
    demographics_heel = demographics[heel_samples_indices]
    
    # Get cord blood samples
    cord_samples_indices = X["source"] == "CORD"
    X_cord = X[cord_samples_indices].copy()
    y_cord = y[cord_samples_indices]
    demographics_cord = demographics[cord_samples_indices]
    
    # Remove Source column from both datasets
    X_heel = X_heel.drop(columns=["source"])
    X_cord = X_cord.drop(columns=["source"])
    info = {
        'HEEL': (X_heel, y_heel, demographics_heel),
        'CORD': (X_cord, y_cord, demographics_cord)
    }

    return info




 
def get_lipidome_metabolome_mom(country: Literal[config.countries], patient_type: Literal["Mom"], target_column: Literal["HIV"]):
       
        processor = MaternalDataProcessor(MaternalDataLoader(country=country, genom_type="Metabolome"), target_column=target_column)
        X_dict, demographics_dict, y_dict = processor.process_data(drop_unlabelled=True)
        X_metabolome = X_dict[patient_type]
        y_metabolome = y_dict[patient_type]
        demographics_metabolome = demographics_dict[patient_type]

        processor = MaternalDataProcessor(MaternalDataLoader(country=country, genom_type="Lipidome"), target_column=target_column)
        X_dict, demographics_dict, y_dict = processor.process_data(drop_unlabelled=True)
        X_libidome = X_dict[patient_type]
        y_libidome = y_dict[patient_type]
        demographics_libidome = demographics_dict[patient_type]
        # merge the dataframes by keeping only the rows that share the same Study_ID in the demographics
        # First, merge features with demographics to get STUDY.ID for merging
        X_metabolome_with_id = pd.merge(X_metabolome, demographics_metabolome[['STUDY.ID']], left_index=True, right_index=True, how='inner')
        X_libidome_with_id = pd.merge(X_libidome, demographics_libidome[['STUDY.ID']], left_index=True, right_index=True, how='inner')
        # Now merge the two datasets on STUDY.ID
        X = pd.merge(X_metabolome_with_id, X_libidome_with_id, on='STUDY.ID', how='inner')
        
        y_mapping = pd.DataFrame({
            'STUDY.ID': demographics_metabolome['STUDY.ID'],
            'target': y_metabolome
        })
        
        # Merge to get the target for the samples in X
        y = pd.merge(X[['STUDY.ID']], y_mapping, on='STUDY.ID', how='inner')['target']
        
        # Now drop the STUDY.ID column from X
        X = X.drop(columns=['STUDY.ID'])

        return X, y, demographics_libidome


########################################################
# Helper functions to annotate the data
########################################################




def annotate_metabolomes_kenya(X: pd.DataFrame | list) -> pd.DataFrame:
    """Annotate metabolomes with Kenyan specific information"""
    # replace the column names with the annotation file mapping: Variable_id -> Comound.name

    if isinstance(X, list):
        X_df = pd.DataFrame(columns=X)
    else:
        X_df = X.copy()

    annotation_df = read_full_data(country_name="Kenya", data_type="Mom", genom_type="Annotation")
    for col in X_df.columns:
        col_label = col.upper().replace('.', '_`')

        if col_label in annotation_df['variable_id'].values:
            X_df = X_df.rename(columns={col: annotation_df.set_index('variable_id').loc[col_label]['Compound.name']})
    if isinstance(X, list):
        final_features = [feature.lower().replace('-', '.') for feature in X_df.columns.tolist()]
        return final_features
    else:
        return X_df
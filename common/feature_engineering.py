import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Literal
from scipy import stats
from utils import Verbose
from tabulate import tabulate
import sys

class FeatureEngineer:
    def __init__(self, 
                 quality_control: list[Literal[None,{'RSD_min': float},
                                                    {'RSD_max': float},
                                                    {'IQR_median_min': float},
                                                    {'too_many_missing_values': float}]] = [None],
                 missing_values: Literal['mean', 'median', {'knn': int}, 'Left_censored_min'] = None,
                 treat_zero_values: bool = True,
                 sample_normalization: Literal[None, 'mean', 'median', 'pqn', 'sum'] = None,
                 data_transform: Literal[None, 'log_10', 'log_2', 'sqrt', 'power_2'] = None,
                 feature_normalization: Literal[None, 'mean_std', 'mean_centering',
                                                    'mean_scaling', 'std_scaling', 'min_max_scaling', 'pareto_scaling'] = None,
                 final_data_transform: Literal[None, 'log_10', 'log_2', 'sqrt', 'power_2', 'quantile_normalization'] = None,
                 verbose: bool = True):

        self.printer = Verbose(verbose=verbose)
        self.quality_control = quality_control
        self.missing_values = missing_values
        self.sample_normalization = sample_normalization
        self.data_transform = data_transform
        self.feature_normalization = feature_normalization
        self.final_data_transform = final_data_transform
        self.treat_zero_values = treat_zero_values

        self.dropped_columns_ = []
        self.impute_values_ = {}
        self.sample_normalization_values_ = []
        self.feature_normalization_values_ = {}
        self.final_data_transform_values_ = {}

        self.pipeline = [
            ('quality_control', self._fit_quality_control, self._transform_quality_control),
            ('missing_values', self._fit_missing_values, self._transform_missing_values),
            ('sample_normalization', self._fit_sample_normalization, self._transform_sample_normalization),
            ('data_transform', self._fit_data_transform, self._transform_data_transform),
            ('feature_normalization', self._fit_feature_normalization, self._transform_feature_normalization),
            ('final_data_transform', self._fit_final_data_transform, self._transform_final_data_transform),
        ]


        
        
        
    def fit_transform(self, X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        self.printer.set_verbose(verbose)
        if self.treat_zero_values: #replacing un-repoted, missing or negative values with nan
            X = X.replace([0.0, 'NA', 'NaN', 'N/A','nan'], np.nan)
            X = X.mask(X < 0, np.nan)
        # reset the fitted parameters
        self.dropped_columns_ = []
        self.impute_values_ = {}
        self.sample_normalization_values_ = []
        self.feature_normalization_values_ = {}
        self.final_data_transform_values_ = {}
        # Fit and store the transformation parameters
        X_transformed = X.copy()
        for name, fit_func, transform_func in self.pipeline:
            fit_func(X_transformed)
            X_transformed = transform_func(X_transformed)
        self.printer.print_stats(X_transformed)
        return X_transformed
    
    def transform(self, X: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        self.printer.set_verbose(verbose)
        # Use the fitted parameters for transformation
        if self.treat_zero_values: #replacing un-repoted, missing or negative values with nan
            X = X.replace([0.0, 'NA', 'NaN', 'N/A'], np.nan)
            X = X.mask(X < 0, np.nan)
        X_transformed = X.copy()
        for name, _, transform_func in self.pipeline:
            X_transformed = transform_func(X_transformed)
        self.printer.print_stats(X_transformed)
        return X_transformed


########################################################################################
# Private methods for transforming data
########################################################################################

    # quality control: drop columns with low quality, low variance, or low repeatability
    def _fit_quality_control(self, X: pd.DataFrame) -> None:
        if self.quality_control is None or self.quality_control == [None]:
            return
        for filter in self.quality_control:
            name, value = list(filter.items())[0]
            if name == 'RSD_min': # drop columns with RSD<value
                self.dropped_columns_.append(X.columns[X.std(axis=0) < X.mean(axis=0) * value])
            elif name == 'RSD_max': # drop columns with RSD>value
                self.dropped_columns_.append(X.columns[X.std(axis=0) > X.mean(axis=0) * value])
            elif name == 'IQR_median_min': # drop columns with IQR<value
                self.dropped_columns_.append(X.columns[X.quantile(axis=0, q=0.75) - X.quantile(axis=0, q=0.25) < value * X.median(axis=0)])
            elif name == 'too_many_missing_values':
                self.dropped_columns_.append(X.columns[X.isnull().sum(axis=0) > X.shape[0] * value])
            else:
                raise ValueError(f'Invalid quality control filter: {name}')
        
        all_dropped = [col for idx in self.dropped_columns_ if len(idx) > 0 for col in idx]
        self.dropped_columns_ = list(dict.fromkeys(all_dropped)) if all_dropped else []
        self.printer.print(f'[data] quality control removed columns: {self.dropped_columns_}')


    def _transform_quality_control(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(columns=self.dropped_columns_)
        return X

    # missing values: impute missing values with mean, median, or knn
    def _fit_missing_values(self, X: pd.DataFrame) -> None:
        if self.missing_values is None:
            return
        if self.treat_zero_values:
            X = X.replace(0, np.nan)
        self.printer.print(f'\n[missing values] number of missing values before {self.missing_values} imputation: {X.isnull().sum().sum()}\n')
        if self.missing_values == 'mean':
            self.impute_values_ = X.mean(axis=0)
        elif self.missing_values == 'median':
            self.impute_values_ = X.median(axis=0)
        elif self.missing_values == 'Left_censored_min': # 1.5 of the min positive value
            X_positive = X[X > 0]
            self.impute_values_ = X_positive.min(axis=0) * 1.5
        elif isinstance(self.missing_values, dict) and 'knn' in self.missing_values:
            imputer = KNNImputer(n_neighbors=self.missing_values['knn'])
            self.imputer_ = imputer.fit(X)
            self.impute_values_ = None

        else:
            raise ValueError(f'Invalid missing values method: {self.missing_values}')


    def _transform_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.missing_values is None:
            return X
        if isinstance(self.missing_values, str):
            X = X.fillna(self.impute_values_, axis=0)
        elif isinstance(self.missing_values, dict) and 'knn' in self.missing_values:
            X = pd.DataFrame(self.imputer_.transform(X), columns=X.columns, index=X.index)
        else:
            raise ValueError(f'Invalid missing values method: {self.missing_values}')
        return X

    # sample normalization: normalize samples by sum, weight, or BMI
    def _fit_sample_normalization(self, X: pd.DataFrame) -> None:
        if self.sample_normalization == 'sum':
            self.sample_normalization_values_ = X.sum(axis=1)
        elif self.sample_normalization == 'mean':
            self.sample_normalization_values_ = X.mean(axis=1)
        elif self.sample_normalization == 'median':
            self.sample_normalization_values_ = X.median(axis=1)
        elif self.sample_normalization == 'pqn':
            self.sample_normalization_values_ = X.quantile(axis=1, q=0.5)
        elif self.sample_normalization is None:
            self.sample_normalization_values_ = 1*np.ones(X.shape[0])
        else:
            raise ValueError(f'Invalid sample normalization method: {self.sample_normalization}')


    def _transform_sample_normalization(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.sample_normalization is None:
            return X
        elif len(X) != len(self.sample_normalization_values_):
            raise ValueError('Sample normalization values are not fitted due to no shape mismatch')
        X = X.div(self.sample_normalization_values_, axis=0)
        return X

    # data transform: log transform, square root transform, or log2 transform
    def _fit_data_transform(self, X: pd.DataFrame) -> None:
        min_value = X.min(axis=0)
        max_value = X.max(axis=0)
        self.data_transform_values_ = {
            'min_value': min_value,
            'max_value': max_value,
            'epsilon': 1/2 * min(min_value[min_value > 0], default=0) + 1e-6 # half the minimum positive value of the data
        }

    def _transform_data_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.data_transform == 'log_10':
            X = np.log10(X + self.data_transform_values_['epsilon'])
        elif self.data_transform == 'log_2':
            X = np.log2(X + self.data_transform_values_['epsilon'])
        elif self.data_transform == 'sqrt':
            X = np.sqrt(X + self.data_transform_values_['epsilon'])
        elif self.data_transform == 'power_2':
            X = np.power(X, 2)
        elif self.data_transform is None:
            pass
        else:
            raise ValueError(f'Invalid data transform method: {self.data_transform}')
        return X

    # feature normalization: normalize features by mean and standard deviation, or mean and centering
    def _fit_feature_normalization(self, X: pd.DataFrame) -> None:
        if self.feature_normalization == 'mean_std':
            self.feature_normalization_values_ = {
                'mean': X.mean(axis=0),
                'std': X.std(axis=0)
            }
        elif self.feature_normalization == 'mean_centering':
            self.feature_normalization_values_ = {
                'mean': X.mean(axis=0),
            }
        elif self.feature_normalization == 'mean_scaling':
            self.feature_normalization_values_ = {
                'mean': X.mean(axis=0),
            }
        elif self.feature_normalization == 'std_scaling':
            self.feature_normalization_values_ = {
                'std': X.std(axis=0),
            }
        elif self.feature_normalization == 'min_max_scaling':
            self.feature_normalization_values_ = {
                'min': X.min(axis=0),
                'max': X.max(axis=0)
            }
        elif self.feature_normalization == 'pareto_scaling':
            self.feature_normalization_values_ = {
                'mean': X.mean(axis=0),
                'std_root': np.sqrt(X.std(axis=0)),
            }
        elif self.feature_normalization is None:
            self.feature_normalization_values_ = None
        else:
            raise ValueError(f'Invalid feature normalization method: {self.feature_normalization}')

    def _transform_feature_normalization(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_normalization == 'mean_std':
            X = (X - self.feature_normalization_values_['mean']) / self.feature_normalization_values_['std']
        elif self.feature_normalization == 'mean_centering':
            X = X - self.feature_normalization_values_['mean']
        elif self.feature_normalization == 'mean_scaling':
            X = X / (self.feature_normalization_values_['mean'] + 1e-6)
        elif self.feature_normalization == 'std_scaling':
            X = X / (self.feature_normalization_values_['std'] + 1e-6)
        elif self.feature_normalization == 'min_max_scaling':
            X = (X - self.feature_normalization_values_['min']) / (self.feature_normalization_values_['max'] - self.feature_normalization_values_['min'] + 1e-6)
        elif self.feature_normalization == 'pareto_scaling':
            X = (X -self.feature_normalization_values_['mean']) / (self.feature_normalization_values_['std_root'] + 1e-6)
        elif self.feature_normalization is None:
            pass
        else:
            raise ValueError(f'Invalid feature normalization method: {self.feature_normalization}')
        return X

    def _fit_final_data_transform(self, X: pd.DataFrame) -> None:
        self.final_data_transform_values_ = {
            'epsilon': 1e-6
        }
    
    def _transform_final_data_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.final_data_transform == 'log_10':
            X = np.log10(X + self.final_data_transform_values_['epsilon'])
        elif self.final_data_transform == 'log_2':
            X = np.log2(X + self.final_data_transform_values_['epsilon'])
        elif self.final_data_transform == 'sqrt':
            X = np.sqrt(X + self.final_data_transform_values_['epsilon'])
        elif self.final_data_transform == 'power_2':
            X = np.power(X, 2)
        elif self.final_data_transform == 'quantile_normalization':
            X = X.rank(axis=0, method='first').sub(0.5).div(X.shape[0]).mul(X.std(axis=0)).add(X.mean(axis=0))
        elif self.final_data_transform is None:
            pass
        else:
            raise ValueError(f'Invalid final data transform method: {self.final_data_transform}')
        return X








########################################################################################
# Helper functions
########################################################################################



    
    
    
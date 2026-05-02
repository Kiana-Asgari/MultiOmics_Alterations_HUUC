from maternal_HIV_effects.load_maternal_data import load_maternal_data
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from common.feature_engineering import FeatureEngineer
quality_control = [{'IQR_median_min': 0.05},
                   {'RSD_min': 0.05},
                   {'too_many_missing_values': 0.5}, 
                   {'RSD_max': 100}
                    ]

def maternal_vs_birthweight():
    biomarkers, clinical_data, HIV_status = load_maternal_data(drop_gadelivery=False)
    X, y, demographics = _intial_quality_control((biomarkers, HIV_status, clinical_data))


    birthweight = clinical_data['birthweight']
    GAdelivery = clinical_data['gadelivery']
    clinical_data = clinical_data.drop(columns=['birthweight', 'gadelivery'])

    significant_biomarkers = ['fa.20.0', 'fa.22.4', 'm333t291.2.pos.rplc', 'm295t300.neg.rplc',\
        'pc.18.2.18.2', 'pc.18.2.20.4']
    X = pd.concat([biomarkers[significant_biomarkers], clinical_data], axis=1)
    # appending the GAdelivery and Birthweight together as Y vector
    Y = pd.concat([GAdelivery, birthweight], axis=1)

    print('starting training OLS')
    r_2_list = train_OLS(X, birthweight, 10)
    print('mean r2', np.mean(r_2_list), 'std r2', np.std(r_2_list))
    r_2_baseline = train_OLS(clinical_data, birthweight, 10)

    print('starting training RidgeCV')
    r_2_unadjusted = train_OLS(biomarkers, birthweight, 10)
    print('mean r2 baseline', np.mean(r_2_baseline), 'std r2 baseline', np.std(r_2_baseline))
    print('mean r2 unadjusted', np.mean(r_2_unadjusted), 'std r2 unadjusted', np.std(r_2_unadjusted))
    print('mean r2 adjusted', np.mean(r_2_list), 'std r2 adjusted', np.std(r_2_list))




def train_OLS(X, y, n_trials):
    # linear regression for predicting birthweight from maternal biomarkers
    rng = np.random.default_rng(42)
    trial_seeds = rng.integers(low=0, high=1_00_000, size=n_trials)
    r2_list = []

    for t in range(n_trials):
        X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, train_size=0.7, random_state=trial_seeds[t], shuffle=True)
        
        # Restore column names and index
        X_tr = pd.DataFrame(X_tr, columns=X.columns)
        X_tst = pd.DataFrame(X_tst, columns=X.columns)
        if isinstance(y, pd.DataFrame):
            y_tr = pd.DataFrame(y_tr, columns=y.columns)
            y_tst = pd.DataFrame(y_tst, columns=y.columns)
        else:
            y_tr = pd.Series(y_tr, name=y.name if hasattr(y, 'name') else None)
            y_tst = pd.Series(y_tst, name=y.name if hasattr(y, 'name') else None)
        y_pred, r2, (coefs, pvalues) = _train_trial_OLS(X_tr, y_tr, X_tst, y_tst)
        r2_list.append(r2)
        print(f'Trial {t} done with r2 {r2:.4f}')
    return r2_list


def _train_trial_OLS(X_tr: pd.DataFrame, y_tr: pd.Series, X_tst: pd.DataFrame, y_tst: pd.Series):
    X_tr, X_tst = _normalize_and_impute_data(X_tr, X_tst)

    # Add intercept (constant term) to include it in the model
    X_tr_with_const = add_constant(X_tr)
    X_tst_with_const = add_constant(X_tst)
    
    # Using statsmodels OLS - note the order is OLS(y, X)
    model = OLS(y_tr, X_tr_with_const)
    result = model.fit()
    y_pred = result.predict(X_tst_with_const)
    
    # Calculate R² on test data (performance on unseen data)
    r2_test = r2_score(y_tst, y_pred)
    
    # Training R² 
    r2_train = result.rsquared
    
    # Get p-values and coefficients
    pvalues = result.pvalues
    coefs = result.params
    
    return y_pred, r2_test, (coefs, pvalues)




def _intial_quality_control(full_data):
    (X, y, demographics) = full_data
    feature_engineer_0 = FeatureEngineer(quality_control=quality_control)
    X = feature_engineer_0.fit_transform(X, verbose=False)

    return X, y, demographics


def _normalize_and_impute_data(X_tr, X_tst):
    for col in X_tr.columns:
        if X_tr[col].var() <= 1e-4:
            print('variance of', col, f'is {X_tr[col].var()}')


    feature_engineer_1 = FeatureEngineer(quality_control=[{'RSD_min': 1e-2}], missing_values='Left_censored_min', feature_normalization='mean_std', verbose=False)
    X_tr= feature_engineer_1.fit_transform(X_tr.copy(), verbose=False)
    X_tst= feature_engineer_1.transform(X_tst.copy(), verbose=False)
    return X_tr, X_tst
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from common.modeling_utils import train_test_split_df
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import KNNImputer
from common.feature_engineering import FeatureEngineer
from sklearn.metrics import confusion_matrix
import shap
from sklearn.model_selection import StratifiedKFold
import sys
from neonatal_HIV_exposure_biomarkers.load_neonatal_data import load_neonatal_data
from joblib import Parallel, delayed
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.utils import statsmodels_train_model, evaluate_model, _class_weights
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.cv_1se import find_1se_lambda


def _fit_single_regression(cv_reg, X_tr, y_tr, class_w_tr, reg_weights, L1_wt):
    """Helper function to fit a single regularization value - used for parallelization"""
    val_mod = sm.GLM(y_tr, X_tr, family=sm.families.Binomial(), freq_weights=class_w_tr)
    res = val_mod.fit_regularized(method="elastic_net",
                                    alpha=cv_reg * reg_weights,
                                    L1_wt=L1_wt,
                                    maxiter=5000, cnvrg_tol=1e-8)
    return cv_reg, res.params


def lasso_path(X_tr, y_tr, class_w_tr, reg_weights, L1_wt, n_jobs=3, fit_alpha=False):
    cv_regs = np.arange(0.002, 0.5, 0.005) # TODO: need to check if this is correct
    if fit_alpha:
        cv_alpha = np.arange(0.2, 0.8, 0.1)
    else:
        cv_alpha = [L1_wt]

    feature_path = {}
    largest_coef = {}
    frequency = {}
    for feature in X_tr.columns:
        feature_path[feature] = 0
        largest_coef[feature] = 0
        frequency[feature] = 0

    # Parallel computation over regularization parameters using multiprocessing
    # backend='loky' ensures actual CPU core utilization (not just threading)
    for cv_alpha in cv_alpha:
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_fit_single_regression)(cv_reg, X_tr, y_tr, class_w_tr, reg_weights, L1_wt=cv_alpha)
            for cv_reg in cv_regs
        )
        # Process results sequentially (this part is fast)
        for cv_reg, params in results:
            for feature in X_tr.columns:
                coef = params[feature]
                if coef != 0:
                    feature_path[feature] = cv_reg
                    largest_coef[feature] = max(np.abs(largest_coef[feature]), np.abs(coef))
                    frequency[feature] += 1

    return feature_path, largest_coef, frequency
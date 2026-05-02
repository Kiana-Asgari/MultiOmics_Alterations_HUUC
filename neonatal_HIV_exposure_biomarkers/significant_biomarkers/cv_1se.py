import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
import sys
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.utils import statsmodels_train_model, evaluate_model, _class_weights


def _evaluate_lambda(cv_reg, X_tr, y_tr, reg_weights, L1_wt, cv_metric, cv_folds, skf):
    """Evaluate a single lambda value with cross-validation."""
    fold_indices = list(skf.split(X_tr, y_tr))
    fold_aucs = []
    for train_idx, val_idx in fold_indices:
        acc = _train_fold(train_idx, val_idx, X_tr, y_tr, cv_reg, reg_weights, L1_wt, cv_metric)
        fold_aucs.append(acc)
    return {'lambda': cv_reg, 'auc_mean': np.mean(fold_aucs), 'auc_se': np.std(fold_aucs) / np.sqrt(cv_folds)}

def _evaluate_all_lambdas(reg_inv_list, X_tr, y_tr, reg_weights, L1_wt, cv_metric, cv_folds, skf, n_jobs):
    """Evaluate all lambda values either in parallel or sequentially."""
    if n_jobs == 1:
        # Sequential evaluation
        auc_dict = []
        for reg_inv in reg_inv_list:
            result = _evaluate_lambda(1/reg_inv, X_tr, y_tr, reg_weights, L1_wt, cv_metric, cv_folds, skf)
            auc_dict.append(result)
        return auc_dict
    else:
        # Parallel evaluation
        return Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_lambda)(1/reg_inv, X_tr, y_tr, reg_weights, L1_wt, cv_metric, cv_folds, skf)
            for reg_inv in reg_inv_list
        )

def find_1se_lambda(X_tr, y_tr, reg_weights, L1_wt, cv_metric,reg_inv_list=None, se_rule='include_both', n_jobs=1):
    
    """Find optimal lambda using 1-SE rule with cross-validation."""
    #return 0



    if reg_inv_list is None:
        reg_inv_list = np.logspace(-0.2, 2.5, 30)
    cv_folds = 5
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Evaluate all lambdas (parallel or sequential based on n_jobs)
    auc_dict = _evaluate_all_lambdas(reg_inv_list, X_tr, y_tr, reg_weights, L1_wt, cv_metric, cv_folds, skf, n_jobs)
    
    # Find best lambda using 1-SE rule
    max_auc = max(auc_dict, key=lambda x: x['auc_mean'])
    max_auc_se = max_auc['auc_se']
    max_auc_mean = max_auc['auc_mean']
    max_lambda = max_auc['lambda']

    se_lambda = 0

    for i in range(len(auc_dict)):
        if auc_dict[i]['auc_mean'] >= max_auc_mean - max_auc_se:
            if auc_dict[i]['lambda'] > se_lambda:
                se_lambda = auc_dict[i]['lambda']


    if se_rule==True:
        print(f'1-SE lambda: {se_lambda:.4f} (optimal lambda: {max_lambda:.4f}))')
        return se_lambda
    elif se_rule== False:
        print(f'optimal lambda: {max_lambda:.4f} (AUC={max_auc_mean:.3f})')
        return max_lambda   
    elif se_rule== 'include_both':
        print(f'1-SE lambda: {se_lambda:.4f} (optimal lambda: {max_lambda:.4f}))')
        return se_lambda, max_lambda
    else:
        raise ValueError(f'Invalid se_rule: {se_rule}, choose from True, False, or include_both')


def _train_fold(train_idx, val_idx, X_tr, y_tr, cv_reg, reg_weights, L1_wt, cv_metric):
    """Train a single fold - helper for parallelization."""
    X_tr_fold, y_tr_fold = X_tr.iloc[train_idx], y_tr.iloc[train_idx]
    X_val_fold, y_val_fold = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
    
    class_w_fold = _class_weights(y_tr_fold)
    res = statsmodels_train_model(X_tr_fold, y_tr_fold, cv_reg * reg_weights,
                                 L1_wt, class_w_fold, tol=1e-4)
    try:
        val_results = evaluate_model(res, X_val_fold, y_val_fold)
    except Exception as e:
        print(f'Error evaluating had nans in X_val_fold or y_val_fold:', X_val_fold.isna().sum(), y_val_fold.isna().sum())
        return 0
    acc = val_results[cv_metric]
    return acc

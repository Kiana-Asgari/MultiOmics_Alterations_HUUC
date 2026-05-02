import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

from neonatal_HIV_exposure_biomarkers.load_neonatal_data import load_neonatal_data

def _clear_ART(biomarkers, y, demographics):
    #first, convert '99' values to NAN to fix Zambia
    # using -1 to indicate ART negative 
    demographics['antiretroviral.therapy'].replace('99', -1, inplace=True)
    demographics['antiretroviral.therapy'].replace(np.nan, -1, inplace=True)
    demographics['antiretroviral.therapy'].replace(1, -1, inplace=True)
    # using p to indicate Prior to pregnancy and D to indicate During pregnancy before screening; 2 to unclear timing
    demographics['antiretroviral.therapy'].replace('Prior to pregnancy', 'Y:P', inplace=True)
    demographics['antiretroviral.therapy'].replace('During pregnancy before screening', 'Y:D', inplace=True)
    demographics['antiretroviral.therapy'].replace(2, 'Y:NA', inplace=True)

    HIV_pos_ART_neg = (y == 1) & (demographics['antiretroviral.therapy']== -1)
    biomarkers, y, demographics = biomarkers[~HIV_pos_ART_neg], y[~HIV_pos_ART_neg], demographics[~HIV_pos_ART_neg]
    return biomarkers, y, demographics




def prepare_data(confounders, country=None, gender=None, clear_ART=True):
    if country is None:
        country = input('Enter country (Kenya, Both, Zambia): ')
    biomarkers, y, demographics = load_neonatal_data(country)


    if clear_ART:
        biomarkers, y, demographics  = _clear_ART(biomarkers, y, demographics)
    X = pd.concat([biomarkers, demographics[confounders]], axis=1)
    if gender is None:
        gender = int(input("Enter gender (1 for male, 2 for female, -1 for both): "))
    X, y, demographics = stratify_gender(X, y, demographics, gender)
    X = X.drop(columns=[col for col in X.columns if "hgb" in col])

    return (X, y, demographics), country, gender


def statsmodels_train_model(X_tr, y_tr, reg_weights, L1_wt, class_w_tr, tol=1e-5):
    """Train logistic regression with elastic net regularization and balanced class weights."""
    print(f'  Reg_weights  {reg_weights[1]:.4f}')

    mod = sm.GLM(y_tr, X_tr, family=sm.families.Binomial(), freq_weights=class_w_tr)
    res = mod.fit_regularized(method="elastic_net",
                                alpha=reg_weights,
                                L1_wt=L1_wt,
                                maxiter=3000, cnvrg_tol=tol)
    return res

def stratify_gender(X, y, demographics, gender):
    if gender != -1:
        male_mask = demographics["sex"] == gender
        X = X[male_mask]
        y = y[male_mask]
        X = X.drop(columns=["sex"])
        demographics = demographics[male_mask]
    return X, y, demographics

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
    l_SHAP = shap_analysis(res, X_tst, y_tst)
    g_SHAP = global_SAHP(l_SHAP, X_tst.columns[1:])  # Exclude intercept
  
    results = {
        'model': res.params,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc,
        'pr_auc': pr_auc,
        'type 1 error': type_1_error,
        'type 2 error': type_2_error,
        'global_SHAP': g_SHAP,
        'log_odds_ratio': res.params[1:],
        'feature_names': X_tst.columns[1:],
    }
    fpr, tpr, thresholds_roc = roc_curve(y_tst, proba_te, drop_intermediate=False)
    results["ROC_curve"] = (fpr, tpr, thresholds_roc)
    return results
    



def global_SAHP(shap_values, feature_names):
    """Calculate feature importance from SHAP values."""
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create DataFrame with feature names and importance scores
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    return importance_df




def shap_analysis(sm_model, X_test, y_test):
    coef = sm_model.params[1:].values    # exclude intercept and convert to numpy array
    intercept = sm_model.params.iloc[0]

    # Wrap in sklearn-like object
    class SklearnLikeLogit:
        def __init__(self, coef, intercept):
            self.coef_ = coef.reshape(1, -1)
            self.intercept_ = np.array([intercept])

    model_like = SklearnLikeLogit(coef, intercept)

    # Background = sample from train (exclude intercept column)
    sample_indices = np.random.choice(X_test.shape[0],
                                     size=min(200, X_test.shape[0]),
                                     replace=False)
    background = X_test.iloc[sample_indices].iloc[:, 1:]  # Remove intercept column

    # Exact, fast SHAP for linear logit
    explainer = shap.LinearExplainer(model_like, background)
    shap_values = explainer.shap_values(X_test.iloc[:, 1:])  # Remove intercept column
    return shap_values



def _class_weights(y_tr):
    # Calculate balanced class weights
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)
    n_total = len(y_tr)
    
    # Balanced weights: inverse of class frequency
    w1 = n_total / (2 * n_pos) if n_pos > 0 else 1.0  # Weight for positive class
    w0 = n_total / (2 * n_neg) if n_neg > 0 else 1.0  # Weight for negative class
    return np.where(y_tr == 1, w1, w0)

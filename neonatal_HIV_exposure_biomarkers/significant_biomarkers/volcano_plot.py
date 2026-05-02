from neonatal_HIV_exposure_biomarkers.load_neonatal_data import load_neonatal_data
from common.feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.plot_utils import set_seaborn_style, annotate_significant_features, get_publication_colors
from common.log_p_values import compute_log_p_values
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.evaluation_and_plots import _get_feature_display_name

import os

confounders = ["sex", "birth.weight.g", "alcohol", "smoking", "diabetes", "hypertension", "gravity",
                "gadelivery", "parity", "multiple.birth", "maternal.age.delivery", "prepregnancy.bmi"]

both_ratio_features_Kenya = [ 'c0', 'biot' ,'ala.leu', 'tyr.phe', 'asa.orn', 'orn.phe', 'orn.arg', 'cit.tyr', 'c3.c4dc', 'c5.c3']
female_ratio_features_Kenya = [ 'c0', 'biot' ,'ala.leu','tyr.phe', 'orn.phe', 'orn.arg', 'cit.tyr']
male_ratio_features_Kenya =  ['c0', 'biot' ,'ala.leu', 'orn.phe', 'cit.tyr', 'cit.orn', 'asa.orn', 'tyr.phe', 'orn.arg']

both_ratio_features_Both = ['c0', 'biot' ,'ala.leu', 'tyr.phe', 'asa.orn', 'orn.phe', 'orn.arg', 'cit.tyr','c3.c4dc', 'c5.c3']
female_ratio_features_Both = ['c0', 'biot' ,'ala.leu','tyr.phe', 'orn.phe', 'orn.arg', 'cit.tyr']
male_ratio_features_Both = ['c0', 'biot' ,'ala.leu', 'orn.phe', 'cit.tyr', 'cit.orn', 'asa.orn', 'tyr.phe', 'orn.arg']







def neonatal_volcano_plot(country, gender, significant_features_list: list=[]):
    print(f'Plotting volcano plot for {country} with gender {gender}')
    X, y, demographics = _prepare_data(country, gender, drop_ratios=False)
    feature_engineer = FeatureEngineer(
                                        missing_values='Left_censored_min',
                                        verbose=False)
    X = feature_engineer.fit_transform(X) 
    data_transform = None
    feature_normalization = 'mean_std'
    feature_engineer = FeatureEngineer(data_transform=data_transform, 
                                        feature_normalization=feature_normalization)

    X_filtered = feature_engineer.fit_transform(X)

    fc_threshold = 2**0.11
    p_threshold = 10**-0.9    


    saving_name = f'volcano_neonatal_{country}_{gender}'
    if country == 'Both':
        if gender == -1:
            sig_ratio_features_list = both_ratio_features_Both
            dont_plot_features = ['arg', 'c4dc', 'c5']
        elif gender == 1:
            sig_ratio_features_list = male_ratio_features_Both
            dont_plot_features = ['orn', 'arg']
        elif gender == 2:
            sig_ratio_features_list = female_ratio_features_Both
            dont_plot_features = ['arg']

    elif country == 'Kenya':
        if gender == -1:
            sig_ratio_features_list = both_ratio_features_Kenya
            dont_plot_features = ['orn', 'c5','c4dc']
        elif gender == 1:
            sig_ratio_features_list = male_ratio_features_Kenya
            dont_plot_features = ['orn', 'c5','c4dc', 'arg']
        elif gender == 2:
            sig_ratio_features_list = female_ratio_features_Kenya
            dont_plot_features = ['orn', 'c5','c4dc']

    significant_features_ID, significant_features_ID_no_ratio, fixed_col_names = _fix_ratio_id(X.columns, 
                        significant_features_list, sig_ratio_features_list, dont_plot_features)
    X_filtered.columns = fixed_col_names
    X.columns = fixed_col_names


    volcano_plot(X_filtered, X, y,
                 fc_threshold=fc_threshold,
                 p_threshold=p_threshold,
                 saving_name=saving_name,
                 significant_features_list=significant_features_ID,
                 significant_features_list_no_ratio=significant_features_ID_no_ratio,
                 country=country,
                 gender=gender)

def volcano_plot(X: pd.DataFrame, X_original: pd.DataFrame, y: pd.Series,
                        equal_var: bool=False,
                        p_threshold: float=0.1,
                        fc_threshold: float=2,
                        saving_name: str=None,
                        significant_features_list: list=[],
                        significant_features_list_no_ratio: list=[],
                        country: str=None,
                        gender: int=None):    
    
    set_seaborn_style()
    pub_face_colors, pub_edge_colors = get_publication_colors(3, "qualitative")
    log_p_values = compute_log_p_values(X, y, 'wilcoxon', 'fdr_bh', equal_var, p_threshold)
    log_fold_change = _log_fold_change(X_original, y)
    feature_names = X.columns


    # Create significance mask vectorized
    p_sig = log_p_values > -np.log10(p_threshold)
    fc_sig_up = log_fold_change > np.log2(fc_threshold)
    fc_sig_down = log_fold_change < -np.log2(fc_threshold)

    """
    Print the number of significant features for each country and genderexit

    """
    gender_name = {1: "Male", 2: "Female", -1: "Both"}
    up_regulated_indices = (p_sig & fc_sig_up)
    down_regulated_indices = (p_sig & fc_sig_down)
    up_regulated_features = feature_names[up_regulated_indices]
    down_regulated_features = feature_names[down_regulated_indices]
    print(f"\n \n Country: {country} -- {gender_name[gender]} only {up_regulated_indices.sum()} up-regulated and {down_regulated_indices.sum()} down-regulated")
    print(f"\n \n Country: {country} -- {gender_name[gender]} only {(p_sig & fc_sig_up).sum()} up-regulated and {(p_sig & fc_sig_down).sum()} down-regulated")
    #breakpoint()


    edge_colors = np.where(p_sig & fc_sig_up, pub_edge_colors[0], 
                     np.where(p_sig & fc_sig_down, pub_edge_colors[1], pub_edge_colors[2]))
    face_colors = np.where(p_sig & fc_sig_up, pub_face_colors[0], 
                     np.where(p_sig & fc_sig_down, pub_face_colors[1], pub_face_colors[2]))
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({'log2_fc': log_fold_change, 'neg_log10_p': log_p_values,'edge_color': edge_colors,
                            'face_color': face_colors, 'feature': feature_names})
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for color in pub_edge_colors:
        subset = plot_df[plot_df['edge_color'] == color]
        ax.scatter(
            subset['log2_fc'], subset['neg_log10_p'],
            s=22, alpha=1,
            facecolors=subset['face_color'],
            edgecolors=subset['edge_color'],
            linewidths=1
        )
    
    # Add threshold lines
    ax.axhline(y=-np.log10(p_threshold), color='blue', linestyle='--', alpha=0.6, linewidth=2)
    ax.axvline(x=np.log2(fc_threshold), color='red', linestyle='--', alpha=0.6, linewidth=2)
    ax.axvline(x=-np.log2(fc_threshold), color='red', linestyle='--', alpha=0.6, linewidth=2)
    

    significant_mask = _feature_mask(plot_df['feature'], significant_features_list)
    significant_mask_no_ratio = _feature_mask(plot_df['feature'], significant_features_list_no_ratio)
    significant_mask = significant_mask & (p_sig & (fc_sig_up | fc_sig_down))
    significant_mask = significant_mask | (significant_mask_no_ratio & (p_sig & (fc_sig_up | fc_sig_down)))

    significant_features = plot_df[significant_mask]

    annotate_significant_features(
        ax=ax,
        plot_df=plot_df,
        significant_mask=significant_mask
    )
    
    # Create publication-ready legend and grid
    ax.grid(True, alpha=0.3)
    _set_axes_style(ax)
    plt.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)
    _save_or_show(fig, saving_name)
    
    # Print significant features
    _significant_features_list = [dict([(row['feature'], row['neg_log10_p'])]) 
                               for _, row in significant_features.iterrows()]    
    
    return _significant_features_list, np.atleast_1d(ax)


def _fix_ratio_id(all_features: list, significant_features_list: list=[], 
                 significant_ratio_features_list: list=[], 
                 dont_plot_features: list=[]) -> str:

    significant_features_list = [feature for feature in significant_features_list if feature not in dont_plot_features]
    significant_ratio_features_list = [feature for feature in significant_ratio_features_list if feature not in dont_plot_features]
    all_significant_features_list = significant_features_list + significant_ratio_features_list
    significant_features_ID_no_ratio = [_get_feature_display_name(feature) for feature in significant_features_list]
    significant_features_ID = significant_features_ID_no_ratio + [_get_feature_display_name(feature) for feature in significant_ratio_features_list]
    fixed_col_names = []

    for feature in all_features:
        feature_ID = feature
        for sig_feature in all_significant_features_list:
            if sig_feature.lower() in feature.lower():
                feature_ID = _get_feature_display_name(feature)
                break
        fixed_col_names.append(feature_ID)

    
           
    return significant_features_ID, significant_features_ID_no_ratio, fixed_col_names


def _prepare_data(country, gender, drop_ratios):
    X, y, demographics = load_neonatal_data(country, drop_ratios=drop_ratios)
    if gender != -1:
        male_mask = demographics["sex"] == gender
        X = X[male_mask]
        y = y[male_mask]
        demographics = demographics[male_mask]

    return X, y, demographics


def _show_figure(fig: plt.Figure) -> None:
    plt.show()


def _save_figure(fig: plt.Figure, save_dir: str) -> None:
    fig.savefig(save_dir + '.pdf', dpi=300, facecolor='white')


def _save_or_show(fig: plt.Figure, saving_name: str = None) -> None:
    relative_path = 'neonatal_HIV_exposure_biomarkers/figures/volcano_plots/'
    os.makedirs(relative_path, exist_ok=True)
    save_dir = os.path.join(relative_path, saving_name or '')
    action, args = {
        True: (_show_figure, (fig,)),
        False: (_save_figure, (fig, save_dir))
    }[saving_name is None]
    action(*args)


def _set_axes_style(ax: plt.Axes) -> None:
    ax.set_ylabel(r'-log$_{10}$(adjusted p-value)', fontsize=30, fontweight='normal')
    ax.set_xlabel(r'log$_{2}$(fold change)', fontsize=30, fontweight='normal')
    ax.tick_params(axis='both', which='major', labelsize=30)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')


def _feature_mask(feature_series: pd.Series, features: list) -> pd.Series:
    return feature_series.isin(set(features))







def _log_fold_change(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    group_healthy = X[y == 0]
    group_disease = X[y == 1]
    FC = []
    
    for feature in X.columns:
        healthy_mean = np.mean(group_healthy[feature])
        disease_mean = np.mean(group_disease[feature])

        if healthy_mean == 0:
            if disease_mean == 0:
                FC.append(0.0)
            else:
                FC.append(10.0)  
        elif disease_mean == 0:
            FC.append(-10.0)  
        else:
            if disease_mean > 0 and healthy_mean > 0:
                FC.append(np.log2(disease_mean) - np.log2(healthy_mean))
            elif disease_mean < 0 and healthy_mean < 0:
                FC.append(np.log2(-disease_mean) - np.log2(-healthy_mean))
            else:
                # This shouldn't happen with positive means, but just in case
                print(f"Warning: Negative fold change for feature {feature}, setting to 0")
                FC.append(0.0)
    
    return pd.Series(FC, index=X.columns)





        
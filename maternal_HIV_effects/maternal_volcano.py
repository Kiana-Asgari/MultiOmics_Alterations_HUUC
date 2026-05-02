from common.feature_engineering import FeatureEngineer
from maternal_HIV_effects.significant_biomarkers.utils import prepare_data
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.plot_utils import set_seaborn_style, annotate_significant_features, get_publication_colors
from common.log_p_values import compute_log_p_values

from maternal_HIV_effects.plotting_utils import _get_feature_display_name

import os
confounders = ["age", "Age", "sex", "birth.weight.g", "alcohol", "smoking", "diabetes", "hypertension", "gravity",
                "gadelivery", "parity", "multiple.birth", "maternal.age.delivery", "prepregnancy.bmi"]


# significant_features_Both_ID = ['fa.22.4', 'pc.18.0.20.3', 'pe.p.16.0.18.2', 'M227T84_POS_RPLC', 'M333T291_2_POS_RPLC', 'M295T300_NEG_RPLC']
# significant_features_Lipidome_ID = ['ce.20.3', 'fa.20.0', 'fa.20.1', 'fa.22.4', 'fa.22.6', 'pc.16.0.18.2', 'pc.16.0.20.3', 'pc.18.2.18.2', 'pc.18.2.20.4', 'pe.p.16.0.18.2', 'pe.p.16.0.22.4', 'pe.p.18.0.18.2', 'tg.50.5.fa16.0', 'tg.58.6.fa16.0']
# significant_features_Metabolome_ID = ['M227T84_POS_RPLC', 'M333T291_2_POS_RPLC', 'M295T300_NEG_RPLC']






def maternal_volcano_plot(biomarkers_type: str, significant_features_list: list=[]):
    full_data = prepare_data(biomarkers_type)
    (X, y, demographics) = full_data
    
    feature_engineer = FeatureEngineer(quality_control=[{'IQR_median_min': 0.05},
                                                        {'RSD_min': 0.05},
                                                        {'too_many_missing_values': 0.5}, 
                                                        {'RSD_max': 100}],
                                        missing_values='Left_censored_min',
                                        verbose=False)
    X = feature_engineer.fit_transform(X)
    feature_engineer = FeatureEngineer(feature_normalization='mean_std')
    X_filtered = feature_engineer.fit_transform(X)

    fc_threshold = 2**0.11
    p_threshold = 10**-0.9    

    saving_name = f'volcano_maternal_{biomarkers_type}'

    #significant_features_list = X.columns

    significant_features_list = [_get_feature_display_name(feature) for feature in significant_features_list]
    if biomarkers_type == 'Metabolome' or biomarkers_type == 'Both':
        significant_features_list = significant_features_list + ['alpha.d.glucose']
    volcano_plot(X_filtered, X, y,
                 fc_threshold=fc_threshold,
                 p_threshold=p_threshold,
                 saving_name=saving_name,
                 significant_features_list=significant_features_list,
                 biomarkers_type=biomarkers_type)




def volcano_plot(X: pd.DataFrame, X_original: pd.DataFrame, y: pd.Series,
                        equal_var: bool=False,
                        p_threshold: float=0.1,
                        fc_threshold: float=2,
                        saving_name: str=None,
                        significant_features_list: list=[],
                        biomarkers_type: str=None):    
    
    set_seaborn_style()
    log_p_values = compute_log_p_values(X, y, 'wilcoxon', 'fdr_bh', equal_var, p_threshold)
    log_fold_change = _log_fold_change(X_original, y)
    feature_names_raw = X_original.columns
    feature_names = [_get_feature_display_name(feature) for feature in feature_names_raw]


    # Create significance mask vectorized
    p_sig = log_p_values > -np.log10(p_threshold)
    fc_sig_up = log_fold_change > np.log2(fc_threshold)
    fc_sig_down = log_fold_change < -np.log2(fc_threshold)

    for f in significant_features_list:
        if f not in feature_names:
            continue
        idx = feature_names.index(f)
        print(f'significant feature: {f}, idx: {idx}, log_p_values: {log_p_values.iloc[idx]} log_fold_change: {log_fold_change.iloc[idx]}')

    print(f"\n \n Biomarkers type: {biomarkers_type}--  {(p_sig & fc_sig_up).sum()} up-regulated and {(p_sig & fc_sig_down).sum()} down-regulated")
    fc_sig_down_arr = np.asarray(fc_sig_down)
    fc_sig_up_arr = np.asarray(fc_sig_up)
    p_sig_arr = np.asarray(p_sig)
    all_down_regulated_features = [f for t, f in enumerate(feature_names) if fc_sig_down_arr[t] & p_sig_arr[t]]
    all_up_regulated_features = [f for t, f in enumerate(feature_names) if fc_sig_up_arr[t] & p_sig_arr[t]]
    print(f"all down-regulated features: {all_down_regulated_features}")
    print(f"all up-regulated features: {all_up_regulated_features}")


    pub_face_colors, pub_edge_colors = get_publication_colors(3, "qualitative")
    edge_colors = np.where(p_sig & fc_sig_up, pub_edge_colors[0], 
                    np.where(p_sig & fc_sig_down, pub_edge_colors[1], pub_edge_colors[2]))
    face_colors = np.where(p_sig & fc_sig_up, pub_face_colors[0], 
                    np.where(p_sig & fc_sig_down, pub_face_colors[1], pub_face_colors[2]))

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'log2_fc': log_fold_change,
        'neg_log10_p': log_p_values,
        'edge_color': edge_colors,
        'face_color': face_colors,
        'feature': feature_names
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))
    # Filter out extreme outliers before plotting
    # Keep rows where condition is False (i.e., not extreme outliers)
    outlier_mask = (plot_df['log2_fc'].abs() > 4) & (plot_df['neg_log10_p'] < 1e-1)
    if outlier_mask.any():
        print(f"Removing {outlier_mask.sum()} outlier features from plot")
        plot_df = plot_df[~outlier_mask].copy()
    
    # Plot all points with publication-ready colors
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

    non_gray_mask = plot_df['edge_color'] != pub_edge_colors[2]
    significant_mask = _feature_mask(plot_df['feature'], significant_features_list) & non_gray_mask
    significant_features = plot_df[significant_mask]

    # Use the helper function to annotate significant features
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
    significant_features_list = [dict([(row['feature'], row['neg_log10_p'])]) 
                               for _, row in significant_features.iterrows()]    
    
    return significant_features_list, np.atleast_1d(ax)


def _show_figure(fig: plt.Figure) -> None:
    plt.show()


def _save_figure(fig: plt.Figure, save_dir: str) -> None:
    fig.savefig(save_dir + '.pdf', dpi=300, facecolor='white')


def _save_or_show(fig: plt.Figure, saving_name: str = None) -> None:
    relative_path = 'maternal_HIV_effects/significant_biomarkers/figures/volcano_plots/'
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
    ax.tick_params(axis='both', which='major', labelsize=27)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')


def _feature_mask(feature_series: pd.Series, features: list) -> pd.Series:
    feature_set = set(features)
    return feature_series.apply(lambda name: any(sig == name or sig in name for sig in feature_set))



def _log_fold_change(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    group_healthy = X[y == 0]
    group_disease = X[y == 1]
    FC = []
    
    for feature in X.columns:
        healthy_mean = np.mean(group_healthy[feature])
        disease_mean = np.mean(group_disease[feature])
        
        # Handle edge cases
        if healthy_mean == 0:
            if disease_mean == 0:
                # Both means are 0, set fold change to 1 (log2(1) = 0)
                FC.append(0.0)
            else:
                # Healthy mean is 0 but disease mean is not, set to a large positive value
                FC.append(10.0)  # log2(1024)
        elif disease_mean == 0:
            # Disease mean is 0 but healthy mean is not, set to a large negative value
            FC.append(-10.0)  # log2(1/1024)
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





        
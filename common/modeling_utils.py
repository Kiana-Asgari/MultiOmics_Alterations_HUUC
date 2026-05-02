import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Literal
try:
    from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, SVMSMOTE  # type: ignore
    from imblearn.under_sampling import RandomUnderSampler  # type: ignore
except Exception:  # pragma: no cover
    SMOTE = ADASYN = KMeansSMOTE = SVMSMOTE = RandomUnderSampler = None
import matplotlib.pyplot as plt
import numpy as np
import os
from common.plot_utils import set_seaborn_style








def train_test_split_df(X_full: pd.DataFrame, y_full: pd.Series,
                     train_test_split_ratio: float=0.8, stratify: bool=True, seed: int=42):
    """
    Split the data into training and test sets and attache the column names to the data.
    """
    stratify_arg = y_full if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(X_full, 
                                                        y_full, 
                                                        test_size=1 - train_test_split_ratio, 
                                                        stratify=stratify_arg,
                                                        random_state=seed)

    X_train.columns = X_full.columns
    X_test.columns = X_full.columns
    y_train.name = y_full.name
    y_test.name = y_full.name
    return X_train, X_test, y_train, y_test



def balance_data(X_train: pd.DataFrame, y_train: pd.Series,
                 balancing_method: Literal["SMOTE_baseline",
                                          "ADASYN", "SVMSMOTE", "KMeans_SMOTE",
                                          "under_sampling"]="SMOTE_baseline"):
    if SMOTE is None:
        raise ModuleNotFoundError(
            "imblearn is required for balance_data(). Install with `pip install imbalanced-learn` "
            "or avoid calling balance_data."
        )
    # balancing the data with the balancing method
    print(f"Balancing the data with the {balancing_method} method")
    if balancing_method == "SMOTE_baseline":
        sampler = SMOTE(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif balancing_method == "ADASYN":
        sampler = ADASYN(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif balancing_method == "SVMSMOTE":
        sampler = SVMSMOTE(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif balancing_method == "KMeans_SMOTE":
        sampler = KMeansSMOTE(random_state=42)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    elif balancing_method == "under_sampling":
        sampler = RandomUnderSampler(random_state=42, replacement=True)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    else:
        raise ValueError(f"Invalid balancing method: {balancing_method}")
    return X_train, y_train




def plot_ROC_curve(AUC_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]], 
                  AUC_scores: list[float],
                  ROC_baseline_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]]=None,
                  ROC_baseline_scores: list[float]=None,
                   saving_name: str=None, caption: str=None,
                   confusion_matrix: pd.DataFrame=None,
                   important_features: list[pd.DataFrame]=None):
    # plot the ROC curves
    set_seaborn_style()
    
    # Determine figure size and layout based on whether confusion matrix is provided
    if confusion_matrix is not None:
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])  # ROC curve (left, spans both rows)
        ax2 = fig.add_subplot(gs[0, 1])  # Confusion matrix (top right)
        ax3 = fig.add_subplot(gs[1, 1])  # Box plot (bottom right)
        
        # Position the axes
        ax1.set_position([0.04, 0.1, 0.53, 0.8])  # left, bottom, width, height
        ax2.set_position([0.673, 0.56, 0.33, 0.35])  # confusion matrix
        ax3.set_position([0.635, 0.17, 0.33, 0.35])   # box plot
        
        ax_roc = ax1
        ax_cm = ax2
        ax_box = ax3
    else:
        fig, ax_roc = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a common FPR grid for averaging
    fpr_grid = np.linspace(0, 1, 1000)
    tpr_interpolated = []
    
    # Interpolate each curve to the common grid
    for fpr, tpr, thresholds in AUC_curves:
        if len(fpr) > 1:
            # Remove duplicates and sort
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            df = df.drop_duplicates(subset=['fpr']).sort_values('fpr')
            
            if len(df) > 1:
                # Interpolate to common grid
                tpr_interp = np.interp(fpr_grid, df['fpr'], df['tpr'])
                tpr_interpolated.append(tpr_interp)
            else:
                tpr_interpolated.append(np.zeros_like(fpr_grid))
        else:
            tpr_interpolated.append(np.zeros_like(fpr_grid))
    
    # Calculate mean and standard error
    tpr_interpolated = np.array(tpr_interpolated)
    tpr_mean = np.mean(tpr_interpolated, axis=0)
    tpr_se = np.std(tpr_interpolated, axis=0) / np.sqrt(len(tpr_interpolated))
    
    # Plot baseline curves if provided
    if ROC_baseline_curves is not None and ROC_baseline_scores is not None:
        # Interpolate baseline curves
        tpr_baseline_interpolated = []
        for fpr, tpr, thresholds in ROC_baseline_curves:
            if len(fpr) > 1:
                df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
                df = df.drop_duplicates(subset=['fpr']).sort_values('fpr')
                if len(df) > 1:
                    tpr_interp = np.interp(fpr_grid, df['fpr'], df['tpr'])
                    tpr_baseline_interpolated.append(tpr_interp)
                else:
                    tpr_baseline_interpolated.append(np.zeros_like(fpr_grid))
            else:
                tpr_baseline_interpolated.append(np.zeros_like(fpr_grid))
        
        # Calculate mean and standard error for baseline
        tpr_baseline_interpolated = np.array(tpr_baseline_interpolated)
        tpr_baseline_mean = np.mean(tpr_baseline_interpolated, axis=0)
        tpr_baseline_se = np.std(tpr_baseline_interpolated, axis=0) / np.sqrt(len(tpr_baseline_interpolated))
        
        # Plot individual baseline curves (thin, transparent) with orange/red palette
        baseline_colors = plt.cm.autumn(np.linspace(0.2, 0.8, len(ROC_baseline_curves)))
        for i, ((fpr, tpr, thresholds), color) in enumerate(zip(ROC_baseline_curves, baseline_colors)):
            if len(fpr) > 1:
                df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
                df = df.drop_duplicates(subset=['fpr']).sort_values('fpr')
                if len(df) > 1:
                    ax_roc.plot(df['fpr'], df['tpr'], color=color, alpha=0.3, linewidth=0.8)
        
        # Plot baseline mean curve with confidence interval
        ax_roc.plot(fpr_grid, tpr_baseline_mean, color='orangered', linewidth=1, 
                 label=f'Mean Baseline ROC (AUC = {np.mean(ROC_baseline_scores):.3f} ± {np.std(ROC_baseline_scores):.3f})')
        
        # Shade the area below the baseline mean + std curve with orange
        ax_roc.fill_between(fpr_grid, 0, tpr_baseline_mean + tpr_baseline_se, 
                         color='orangered', alpha=0.05)
        
        # Shade the baseline confidence interval (mean ± std)
        ax_roc.fill_between(fpr_grid, tpr_baseline_mean - tpr_baseline_se, tpr_baseline_mean + tpr_baseline_se, 
                         color='orangered', alpha=0.1)
    
    # Plot individual curves (thin, transparent)
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(AUC_curves)))
    for i, ((fpr, tpr, thresholds), color) in enumerate(zip(AUC_curves, colors)):
        if len(fpr) > 1:
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
            df = df.drop_duplicates(subset=['fpr']).sort_values('fpr')
            if len(df) > 1:
                ax_roc.plot(df['fpr'], df['tpr'], color=color, alpha=0.3, linewidth=0.8)
    
    # Plot mean curve with confidence interval
    ax_roc.plot(fpr_grid, tpr_mean, color='blue', linewidth=1, 
             label=f'Mean ROC (AUC = {np.mean(AUC_scores):.3f} ± {np.std(AUC_scores):.3f})')
    
    # Shade the area below the mean + std curve with blue
    ax_roc.fill_between(fpr_grid, 0, tpr_mean + tpr_se, 
                     color='blue', alpha=0.05)
    
    # Shade the confidence interval (mean ± std)
    ax_roc.fill_between(fpr_grid, tpr_mean - tpr_se, tpr_mean + tpr_se, 
                     color='blue', alpha=0.1)
    
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random classifier', linewidth=1, alpha=0.7)
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title(caption, fontsize=14, fontweight='bold')
    ax_roc.legend(fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    
    # Add confusion matrix table if provided
    if confusion_matrix is not None:
        cm_mean = confusion_matrix.mean(axis=0)
        cm_se = confusion_matrix.std(axis=0) / np.sqrt(len(confusion_matrix))
     
        table_data = [
            [f'{cm_mean.iloc[0]:.3f} ± {cm_se.iloc[0]:.3f}', f'{cm_mean.iloc[1]:.3f} ± {cm_se.iloc[1]:.3f}'],
            [f'{cm_mean.iloc[2]:.3f} ± {cm_se.iloc[2]:.3f}', f'{cm_mean.iloc[3]:.3f} ± {cm_se.iloc[3]:.3f}']
        ]
        table = ax_cm.table(cellText=table_data,
                           rowLabels=['Actual\nNegative', 'Actual\nPositive'],
                           colLabels=['Predicted\nNegative', 'Predicted\nPositive'],
                           cellLoc='center',
                           loc='center')
        desired_w, desired_h = 0.25, 0.25   # same values → square cells

        for (row, col), cell in table.get_celld().items():
            cell.set_width(desired_w)
            cell.set_height(desired_h)
            cell.set_alpha(0.2)
            
            # Add borders to match other plots
            cell.set_edgecolor('black')
            cell.set_linewidth(1.0)
            # Set different alpha for face vs edge
            cell.set_alpha(0.2)  # Face alpha (background)
            cell.set_edgecolor((0, 0, 0, 0.7))  # Edge alpha (borders) - RGBA format
            
        table.set_fontsize(15)

        
        # Style table cells with appropriate colors
        # TN (top-left): light green
        table[(1, 0)].set_facecolor('green')
        table[(1, 0)].set_alpha(0.2)
        table[(1, 0)].set_text_props(weight='bold')
        
        # FP (top-right): light red
        table[(1, 1)].set_facecolor('red')
        table[(1, 1)].set_alpha(0.2)
        table[(1, 1)].set_text_props(weight='bold')
        
        # FN (bottom-left): light red
        table[(2, 0)].set_facecolor('red')
        table[(2, 0)].set_alpha(0.2)
        table[(2, 0)].set_text_props(weight='bold')
        
        # TP (bottom-right): light green
        table[(2, 1)].set_facecolor('green')
        table[(2, 1)].set_alpha(0.2)
        table[(2, 1)].set_text_props(weight='bold')
        
        ax_cm.axis('off')
        
        # Add box plot of the 5 most significant features if provided
        if important_features is not None and len(important_features) > 0:
            # Get the top 10 features across all trials
            all_features = []
            for trial_features in important_features:
                if not trial_features.empty:
                    # Get top 10 features from this trial
                    top_10 = trial_features.head(10)
                    all_features.append(top_10)
            
            if all_features:
                # Combine all trials and get unique top features
                combined_df = pd.concat(all_features, ignore_index=True)
                # Get the 10 most frequently appearing features
                feature_counts = combined_df['feature_name'].value_counts()
                top_10_features = feature_counts.head(10).index.tolist()
                
                # Prepare data for box plot
                box_data = []
                feature_labels = []
                
                for feature in top_10_features:
                    feature_scores = []
                    for trial_features in important_features:
                        if not trial_features.empty:
                            feature_row = trial_features[trial_features['feature_name'] == feature]
                            if not feature_row.empty:
                                feature_scores.append(feature_row['importance_score'].iloc[0])
                    
                    if feature_scores:
                        box_data.append(feature_scores)
                        feature_labels.append(feature)
                
                
                if box_data:
                    # Sort by absolute importance for ranking, but plot actual coefficients with signs
                    # Use abs_importance_score if available, otherwise fall back to absolute of importance_score
                    if 'abs_importance_score' in important_features[0].columns:
                        # Calculate mean absolute importance for ranking
                        mean_abs_data = []
                        for feature in top_10_features:
                            abs_scores = []
                            for trial_features in important_features:
                                if not trial_features.empty:
                                    feature_row = trial_features[trial_features['feature_name'] == feature]
                                    if not feature_row.empty:
                                        abs_scores.append(trial_features[trial_features['feature_name'] == feature]['abs_importance_score'].iloc[0])
                            if abs_scores:
                                mean_abs_data.append(np.mean(abs_scores))
                        # Sort by absolute importance
                        sorted_by_abs = np.argsort(mean_abs_data)[::-1]
                    else:
                        # Fallback: sort by absolute value of actual coefficients
                        mean_box_data = [np.mean(np.abs(box_data[i])) for i in range(len(box_data))]
                        sorted_by_abs = np.argsort(mean_box_data)[::-1]
                    
                    box_data = [box_data[i] for i in sorted_by_abs]
                    feature_labels = [feature_labels[i] for i in sorted_by_abs]
                    # Create box plot
                    bp = ax_box.boxplot(box_data, labels=feature_labels, patch_artist=True)
                    
                    # Color the boxes
                    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightcyan', 'lightsalmon', 'lightsteelblue', 'lightcoral']
                    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Add background color below x=0 to distinguish positive/negative coefficients
                    y_min, y_max = ax_box.get_ylim()
                    abs_max = max(abs(y_min), abs(y_max))
                    if y_min < 0:
                        # Fill the area below x=0 with a light gray background
                        ax_box.axhspan(-abs_max, 0, alpha=0.08, color='red', zorder=0)
                        ax_box.axhspan(0, abs_max, alpha=0.08, color='blue', zorder=0)
                        ax_box.axhline(y=0, color='black', linestyle='dashed', alpha=0.3, linewidth=1, zorder=1)
                    
                    # Make y-axis limits symmetric around zero for better visual balance
                    if y_min < 0:
                        ax_box.set_ylim(-abs_max, abs_max)
                    else:
                        ax_box.set_ylim(0, abs_max)
                    
                    ax_box.set_title('Top 10 Most Significant Features', fontsize=10, fontweight='bold')
                    ax_box.set_ylabel('Coefficient Value (with sign)', fontsize=9)
                    ax_box.tick_params(axis='x', rotation=30, labelsize=8)
                    ax_box.grid(True, alpha=0.3)
                else:
                    ax_box.text(0.5, 0.5, 'No feature data available', ha='center', va='center', 
                              transform=ax_box.transAxes, fontsize=10)
                    ax_box.set_title('Top 10 Most Significant Features', fontsize=10, fontweight='bold')
            else:
                ax_box.text(0.5, 0.5, 'No feature data available', ha='center', va='center', 
                          transform=ax_box.transAxes, fontsize=10)
                ax_box.set_title('Top 10 Most Significant Features', fontsize=10, fontweight='bold')
        else:
            ax_box.text(0.5, 0.5, 'No feature data provided', ha='center', va='center', 
                      transform=ax_box.transAxes, fontsize=10)
            ax_box.set_title('Top 10 Most Significant Features', fontsize=10, fontweight='bold')
        
        ax_box.axis('on')
    
    plt.tight_layout()
    if saving_name is not None:
        relative_path = 'results/ROC/'
        # Create directory if it doesn't exist
        os.makedirs(relative_path, exist_ok=True)
        save_dir = os.path.join(relative_path, saving_name)
        plt.savefig(save_dir, dpi=300)
    else:
        plt.show()

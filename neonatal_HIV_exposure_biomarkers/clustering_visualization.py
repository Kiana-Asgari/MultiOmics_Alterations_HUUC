from neonatal_HIV_exposure_biomarkers.load_neonatal_data import load_neonatal_data
from common.feature_engineering import FeatureEngineer
from common.sPLSDA import plot_PLSR
import umap
import matplotlib.pyplot as plt
import os

def neonatal_UMAP_visualization(country, saving_name=None):
    X, y, demographics = load_neonatal_data(country)
    feature_engineer = FeatureEngineer(quality_control=[{'IQR_median_min': 0.05},
                                                        {'RSD_min': 0.05},
                                                        {'too_many_missing_values': 0.5}, 
                                                        {'RSD_max': 100},
                                                        ],
                                        missing_values='Left_censored_min',
                                        sample_normalization=None,
                                        data_transform=None,
                                        feature_normalization='mean_std',
                                        verbose=False)
    X_processed = feature_engineer.fit_transform(X, verbose=False)
    gender = demographics["sex"]

    # select HIV positive data
    hiv_positive_mask = y == 1
    X_processed = X_processed[hiv_positive_mask]
    y = y[hiv_positive_mask]
    gender = gender[hiv_positive_mask]
    demographics = demographics[hiv_positive_mask]
    # Compute UMAP embedding with optimized parameters for accuracy
    reducer = umap.UMAP(
        n_components=4,
        n_neighbors=30,      # Larger preserves more global structure (default: 15)
        min_dist=0.0,        # Tighter clusters for better separation (default: 0.1)
        metric='euclidean',
        n_epochs=1000,       # More training iterations for better convergence (default: 200-500)
        random_state=42
    )
    embedding = reducer.fit_transform(X_processed)
    
    # Create plot with 4 colors for gender and HIV status
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = ['Gender1-HIV-', 'Gender1-HIV+', 'Gender2-HIV-', 'Gender2-HIV+']
    
    for hiv_status in [0, 1]:
        for gender_val in [1, 2]:
            mask = (y == hiv_status) & (gender == gender_val)
            color_idx = hiv_status + (gender_val - 1) * 2
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=colors[color_idx], label=labels[color_idx],
                      alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title(f'UMAP Visualization - {country}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if saving_name:
        os.makedirs('results/UMAP/', exist_ok=True)
        plt.savefig(f'results/UMAP/{saving_name}', dpi=300)
    plt.show()
    
    return embedding



def neonatal_PLSR_visualization(country):
    X, y, demographics = load_neonatal_data(country)
    feature_engineer = FeatureEngineer(quality_control=[{'IQR_median_min': 0.05},
                                                        {'RSD_min': 0.05},
                                                        {'too_many_missing_values': 0.5}, 
                                                        {'RSD_max': 100},
                                                        ],
                                        missing_values='Left_censored_min',
                                        sample_normalization=None,
                                        data_transform=None,
                                        feature_normalization=None,
                                        verbose=False)
    X = feature_engineer.fit_transform(X, verbose=False)


    final_data_transform = None
    sample_normalization = None
    data_transform = None
    feature_normalization = 'mean_std'
    missing_values = None

    feature_engineer = FeatureEngineer(sample_normalization=sample_normalization, 
                                        data_transform=data_transform, 
                                        feature_normalization=feature_normalization,
                                        final_data_transform=final_data_transform,
                                        missing_values=missing_values)

    X_filtered = feature_engineer.fit_transform(X,  verbose=False)
    #X_filtered = X_filtered.drop(columns=['FA.20.1'])
    #X = X.drop(columns=['FA.20.1'])

    fc_threshold = 2**0.15

    saving_name = None #f'volcano_neonatal_{country}_{gender}'
    plot_PLSR(X_filtered, y, gender=demographics["sex"],saving_name=saving_name)




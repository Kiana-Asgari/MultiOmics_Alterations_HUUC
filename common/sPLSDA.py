from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_PLSR(X: pd.DataFrame, y: pd.Series,
            gender: pd.Series, n_components: int = 5, saving_name: str=None):
    """
    Plot the results of the PLSR algorithm, color by HIV status and gender.
    Creates a 2x2 grid showing different component combinations.
    Four colors represent: Gender1-HIV-, Gender1-HIV+, Gender2-HIV-, Gender2-HIV+
    """
    pls = PLSRegression(n_components=n_components)
    # chose only HIV positive data
    # hiv_pos_mask = y == 1
    # X = X[hiv_pos_mask]
    # y = y[hiv_pos_mask]
    # gender = gender[hiv_pos_mask]
    pls.fit(X, y) 
    X_scores = pls.transform(X)
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Define colors for 4 groups: [Gender1-HIV-, Gender1-HIV+, Gender2-HIV-, Gender2-HIV+]
    colors = ['blue', 'red', 'green', 'cyan']
    labels = ['Gender1-HIV-', 'Gender1-HIV+', 'Gender2-HIV-', 'Gender2-HIV+']
    
    # Plot component combinations: (0,0)
    component_pairs = [(0, 1), (0, 2), (1, 2), (0, 3)]
    
    for idx, (comp_x, comp_y) in enumerate(component_pairs):
        if comp_x >= n_components or comp_y >= n_components:
            continue
            
        ax = axes[idx // 2, idx % 2]
        
        # Plot each group with different color
        for hiv_status in [0, 1]:
            for gender_val in [1, 2]:
                mask = (y == hiv_status) & (gender == gender_val)
                color_idx = hiv_status + (gender_val - 1) * 2
                
                ax.scatter(X_scores[mask, comp_x], X_scores[mask, comp_y],
                          c=colors[color_idx], label=labels[color_idx],
                          alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(f'Component {comp_x + 1}', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Component {comp_y + 1}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if saving_name is not None:
        relative_path = 'results/PLSR/'
        os.makedirs(relative_path, exist_ok=True)
        save_dir = os.path.join(relative_path, saving_name)
        plt.savefig(save_dir, dpi=300)
    plt.show()
    
    return pls, X_scores
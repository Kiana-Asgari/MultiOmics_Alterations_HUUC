import numpy as np
import pandas as pd
"""
After computing the Knockoff scores for random trials,
we need to compute the threshold for the Knockoff scores.
To get FDR<=0.05.
"""

def select_knockoff(scores:dict, q=0.1, knockoff_plus=True):
    # scores is a dictionary of scores where the key is the biomarker and the value is the score
    W = np.array(list(scores.values()))
    knockoff_plus=True

    optimal_T, min_T = np.inf, np.inf
    optimal_fdp_hat, min_fdp_hat = np.inf, np.inf
    optimal_S, min_S = [], []
    t_grid = np.sort(np.unique(np.abs(W[W != 0.0])))
    if t_grid.size == 0:
        return np.inf  # selects nothing
    for t in t_grid:
        num = (W <= -t).sum()
        den = max((W >=  t).sum(), 1)
        fdp_hat = (num + (1 if knockoff_plus else 0)) / den
        S = np.where(W >= t)[0] if np.isfinite(t) else np.array([], dtype=int)
        significant_biomarkers = [list(scores.keys())[i] for i in S]
        print(f' \n\nfor t {t:.3f}, fdp_hat: {fdp_hat:.3f}, \nsignificant_biomarkers: {significant_biomarkers}')
        if fdp_hat <= q:
            if len(significant_biomarkers) > len(optimal_S):
                optimal_T = t
                optimal_S = significant_biomarkers
                optimal_fdp_hat = fdp_hat
        if fdp_hat < min_fdp_hat:
            min_T = t
            min_S = significant_biomarkers
            min_fdp_hat = fdp_hat
    print(f'\noptimal_T: {optimal_T}, optimal_S: {optimal_S}, optimal_fdp_hat: {optimal_fdp_hat}')
    print(f'\nmin_T: {min_T}, min_S: {min_S}, min_fdp_hat: {min_fdp_hat}')
    if len(optimal_S) == 0:
        return min_S, min_T, min_fdp_hat
    return optimal_S, optimal_T, optimal_fdp_hat


def knockoff_threshold(W, q=0.1, knockoff_plus=True):
    # W: array of importance stats for one run
    W = np.array(W)  # Ensure W is a numpy array
    t_grid = np.sort(np.unique(np.abs(W[W != 0.0])))
    if t_grid.size == 0:
        return np.inf  # selects nothing
    for t in t_grid:
        num = (W <= -t).sum()
        den = max((W >=  t).sum(), 1)
        fdp_hat = (num + (1 if knockoff_plus else 0)) / den
        if fdp_hat <= q:
            return t
    return np.inf  # no threshold meets q




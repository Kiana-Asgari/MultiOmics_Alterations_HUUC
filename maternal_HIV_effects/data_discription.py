"""
This file is used to comput the mean and se 
for several variables in the maternal data
Functions print the mean and se in a Latex table formal
"""
from maternal_HIV_effects.load_maternal_data import load_maternal_data
import numpy as np
from scipy.stats import sem, ttest_ind
from statsmodels.stats.multitest import multipletests
from maternal_HIV_effects.significant_biomarkers.utils import load_maternal_results
from prettytable import PrettyTable

from common.feature_engineering import FeatureEngineer





def stats_counts(biomarkers_type: str):
    biomarkers, clinical_data, HIV_status = load_maternal_data(drop_gadelivery=False, biomarkers_type=biomarkers_type)
    male_mask, female_mask = clinical_data['sex'] == 1, clinical_data['sex'] == 2
    table = PrettyTable(['Category', 'Count', 'HIV+', 'Control', 'HIV+ prevalence'])
    table.add_row(['Total', len(HIV_status), (HIV_status == 1).sum(), (HIV_status == 0).sum(), (HIV_status == 1).sum() / len(HIV_status)])
    table.add_row(['Male', male_mask.sum(), (HIV_status[male_mask] == 1).sum(), (HIV_status[male_mask] == 0).sum(), (HIV_status[male_mask] == 1).sum() / male_mask.sum()])
    table.add_row(['Female', female_mask.sum(), (HIV_status[female_mask] == 1).sum(), (HIV_status[female_mask] == 0).sum(), (HIV_status[female_mask] == 1).sum() / female_mask.sum()])
    table.add_row(['# biomarkers', len(biomarkers.columns), '', '', ''])
    table.add_row(['# clinical variables', len(clinical_data.columns), '', '', ''])
    print(f"\nData Counts\n{table}")


def mean_se_clinical_data(biomarkers_type: str):
    print('Computing the mean and se of the clinical data for both genders, male and female')
    biomarkers, clinical_data, HIV_status = load_maternal_data(drop_gadelivery=False, biomarkers_type=biomarkers_type)
    female_mask, male_mask = clinical_data['sex'] == 2, clinical_data['sex'] == 1
    clinical_data = clinical_data.drop(columns=['sex'])

    data = {'All': _compute_stats(clinical_data, HIV_status), 
            'Male': _compute_stats(clinical_data[male_mask], HIV_status[male_mask]),
            'Female': _compute_stats(clinical_data[female_mask], HIV_status[female_mask])}
    _print_latex_table(data, 'Clinical Data Statistics') 

def mean_se_significant_biomarkers(biomarkers_type: str):
    print('Computing the mean and se of significant biomarkers (with BH correction across ALL biomarkers)')
    biomarkers, clinical_data, HIV_status = load_maternal_data(drop_gadelivery=False, biomarkers_type=biomarkers_type)

    female_mask, male_mask = clinical_data['sex'] == 2, clinical_data['sex'] == 1
    feature_engineer_1 = FeatureEngineer(feature_normalization='std_scaling', verbose=False)
    biomarkers= feature_engineer_1.fit_transform(biomarkers, verbose=False)
    
    # Get significant biomarker names
    filtered_features, n_trials = significant_biomarkers_statistics(biomarkers_type=biomarkers_type)
    significant_biomarker_names = [f for f in filtered_features.keys() if f in biomarkers.columns]
    
    # Compute stats for ALL biomarkers (for proper BH correction), then filter to significant ones
    all_data_all = _compute_stats(biomarkers, HIV_status)
    all_data_male = _compute_stats(biomarkers[male_mask], HIV_status[male_mask])
    all_data_female = _compute_stats(biomarkers[female_mask], HIV_status[female_mask])
    
    # Filter to only significant biomarkers for display
    data = {
        'All': {k: v for k, v in all_data_all.items() if k in significant_biomarker_names},
        'Male': {k: v for k, v in all_data_male.items() if k in significant_biomarker_names},
        'Female': {k: v for k, v in all_data_female.items() if k in significant_biomarker_names}
    }
    
    _print_latex_table(data, 'Significant Biomarkers Statistics')


def significant_biomarkers_statistics(biomarkers_type: str):
    results, unadjusted_results, _, _ = load_maternal_results(biomarkers_type=biomarkers_type)
    all_features = results[0]['log_odds_ratio'].keys()

    n_trials, per_feature = len(results), {f: {'adj': [], 'unadj': [], 'adj_count': 0, 'unadj_count': 0} for f in all_features}
    for trial in range(n_trials):
        for feature in all_features:
            if feature in results[trial]['log_odds_ratio'] != 0 and results[trial]['log_odds_ratio'][feature] != 0: 
                per_feature[feature]['adj'].append(results[trial]['log_odds_ratio'][feature])
                per_feature[feature]['adj_count'] += 1
            if feature in unadjusted_results[trial]['log_odds_ratio'] and unadjusted_results[trial]['log_odds_ratio'][feature] != 0: 
                per_feature[feature]['unadj'].append(unadjusted_results[trial]['log_odds_ratio'][feature])
                per_feature[feature]['unadj_count'] += 1
    
    # Filter features and restructure data so biomarkers are rows (not columns)
    filtered_features = {f: v for f, v in per_feature.items() if v['adj_count']/n_trials >= 0.95}
    return filtered_features, n_trials

def adjusted_unadjusted_OR(biomarkers_type: str):
    print('Computing the adjusted and unadjusted OR (with BH correction across ALL features)')
    
    # Load ALL features statistics first
    results, unadjusted_results, _, _ = load_maternal_results(biomarkers_type=biomarkers_type)
    all_features = results[0]['log_odds_ratio'].keys()
    n_trials = len(results)
    
    # Compute statistics for ALL features
    per_feature = {f: {'adj': [], 'unadj': [], 'adj_count': 0, 'unadj_count': 0} for f in all_features}
    for trial in range(n_trials):
        for feature in all_features:
            if feature in results[trial]['log_odds_ratio'] != 0 and results[trial]['log_odds_ratio'][feature] != 0: 
                per_feature[feature]['adj'].append(results[trial]['log_odds_ratio'][feature])
                per_feature[feature]['adj_count'] += 1
            if feature in unadjusted_results[trial]['log_odds_ratio'] and unadjusted_results[trial]['log_odds_ratio'][feature] != 0: 
                per_feature[feature]['unadj'].append(unadjusted_results[trial]['log_odds_ratio'][feature])
                per_feature[feature]['unadj_count'] += 1
    
    def fmt_pval(p):
        if p < 0.001:
            return "p<0.001"
        else:
            return f"p={p:.3f}"
    
    # Collect p-values for ALL features
    from scipy.stats import ttest_1samp
    adj_p_values = []
    unadj_p_values = []
    all_feature_names = list(all_features)
    
    for f in all_feature_names:
        v = per_feature[f]
        # Adjusted log OR p-values
        if len(v['adj']) > 1:
            _, p_val = ttest_1samp(v['adj'], 0)
            adj_p_values.append(p_val)
        else:
            adj_p_values.append(1.0)
        
        # Unadjusted log OR p-values
        if len(v['unadj']) > 1:
            _, p_val = ttest_1samp(v['unadj'], 0)
            unadj_p_values.append(p_val)
        else:
            unadj_p_values.append(1.0)
    
    # Apply BH correction across ALL features
    if len(adj_p_values) > 0:
        _, adj_p_adjusted, _, _ = multipletests(adj_p_values, alpha=0.05, method='fdr_bh')
        _, unadj_p_adjusted, _, _ = multipletests(unadj_p_values, alpha=0.05, method='fdr_bh')
    else:
        adj_p_adjusted = []
        unadj_p_adjusted = []
    
    # Create a mapping from feature name to adjusted p-values
    p_value_map = {
        all_feature_names[i]: {
            'adj_p': adj_p_values[i],
            'adj_p_adj': adj_p_adjusted[i],
            'unadj_p': unadj_p_values[i],
            'unadj_p_adj': unadj_p_adjusted[i]
        }
        for i in range(len(all_feature_names))
    }
    
    # Filter to significant features only (for display)
    filtered_features = {f: v for f, v in per_feature.items() if v['adj_count']/n_trials >= 0.95}
    feature_names = list(filtered_features.keys())
    
    def compute_logOR_with_pval(values, p_value, p_adj):
        # One-sample t-test against 0 (testing if log OR is significantly different from 0)
        if len(values) > 1:
            return f"{np.mean(values):.2f}±{sem(values):.2f} (p={fmt_pval(p_value)}, p_adj={fmt_pval(p_adj)})"
        else:
            return f"{np.mean(values):.2f}±{sem(values):.2f} (N/A)"
    
    restructured_data = {
        'adj_logOR': {f: compute_logOR_with_pval(filtered_features[f]['adj'], 
                                                  p_value_map[f]['adj_p'], 
                                                  p_value_map[f]['adj_p_adj']) 
                      for f in feature_names},
        'adj_freq': {f: f"{v['adj_count']/n_trials:.2f}±{_freq_se(v['adj_count'], n_trials):.2f}" 
                     for f, v in filtered_features.items()},
        'unadj_logOR': {f: compute_logOR_with_pval(filtered_features[f]['unadj'], 
                                                    p_value_map[f]['unadj_p'], 
                                                    p_value_map[f]['unadj_p_adj']) 
                        for f in feature_names},
        'unadj_freq': {f: f"{v['unadj_count']/n_trials:.2f}±{_freq_se(v['unadj_count'], n_trials):.2f}" 
                       for f, v in filtered_features.items()}
    }
    _print_latex_table(restructured_data, 'Log Odds Ratios')

# Helper function to calculate SE for frequency (proportion)
def _freq_se(count, n): 
    p = count / n
    return np.sqrt(p * (1 - p) / n)


def _compute_stats(df, HIV_status):
    def fmt(val): 
        val = float(val) if hasattr(val, '__float__') else val
        return f"{val:.2e}" if abs(val) >= 1000 else f"{val:.2f}"
    
    def fmt_pval(p):
        if p < 0.001:
            return "p<0.001"
        else:
            return f"p={p:.3f}"
    
    hiv_pos, hiv_neg = df[HIV_status == 1], df[HIV_status == 0]
    stats_dict = {}
    
    # First pass: collect all p-values
    columns = df.select_dtypes(include=[np.number]).columns
    p_values = []
    t_stats = []
    
    for col in columns:
        # Perform t-test
        t_stat, p_value = ttest_ind(hiv_pos[col].dropna(), hiv_neg[col].dropna(), equal_var=False)
        t_stats.append(t_stat)
        p_values.append(p_value)
    
    # Apply BH correction
    if len(p_values) > 0:
        rejected, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    else:
        p_adjusted = []
    
    # Second pass: create formatted strings with adjusted p-values
    for i, col in enumerate(columns):
        stats_dict[col] = (
            f"HIV+: {fmt(hiv_pos[col].mean())}±{fmt(hiv_pos[col].sem())} "
            f"(p={fmt_pval(p_values[i])}, p_adj={fmt_pval(p_adjusted[i])})\n"
            f"Control: {fmt(hiv_neg[col].mean())}±{fmt(hiv_neg[col].sem())}"
        )
    
    return stats_dict

def full_maternal_data_discription(biomarkers_type: str = 'Both'):
    """
    Print a comprehensive data description table comparing All patients, HIV+, and Control groups.
    Shows number of markers, cohort size, clinical data statistics, and HIV status.
    Outputs in LaTeX table format.
    
    Args:
        biomarkers_type: Type of biomarkers to load ('Lipidome', 'Metabolome', or 'Both')
    """
    print(f'Computing full data description for {biomarkers_type} across HIV status groups')
    
    biomarkers, clinical_data, HIV_status = load_maternal_data(drop_gadelivery=False, biomarkers_type=biomarkers_type)
    
    # Create masks for HIV+ and Control groups
    hiv_pos_mask = HIV_status == 1
    hiv_neg_mask = HIV_status == 0
    
    table_data = {}
    
    # Process three groups: All, HIV+, Control
    groups = {
        'All patients': (clinical_data, HIV_status, slice(None)),
        'HIV+': (clinical_data[hiv_pos_mask], HIV_status[hiv_pos_mask], hiv_pos_mask),
        'Control': (clinical_data[hiv_neg_mask], HIV_status[hiv_neg_mask], hiv_neg_mask)
    }
    
    for group_name, (group_clinical, group_hiv_status, mask) in groups.items():
        data_dict = {}
        
        # Number of markers (same for all groups)
        data_dict['Number of markers'] = str(len(biomarkers.columns))
        
        # Cohort (total samples in this group)
        data_dict['Cohort'] = str(len(group_hiv_status))
        
        # Number of Female and Male infants
        if 'sex' in group_clinical.columns:
            female_count = (group_clinical['sex'] == 2).sum()
            male_count = (group_clinical['sex'] == 1).sum()
            data_dict['Number of Female infants'] = str(female_count)
            data_dict['Number of Male infants'] = str(male_count)
        
        # Clinical data statistics (mean ± SE for each variable, excluding sex)
        clinical_numeric = group_clinical.select_dtypes(include=[np.number])
        for col in clinical_numeric.columns:
            if col == 'sex':  # Skip sex as we handle it separately
                continue
            mean_val = clinical_numeric[col].mean()
            se_val = clinical_numeric[col].sem()
            # Format large numbers with scientific notation
            if abs(mean_val) >= 1000:
                data_dict[col] = f"{mean_val:.2e}±{se_val:.2e}"
            else:
                data_dict[col] = f"{mean_val:.2f}±{se_val:.2f}"
        
        # HIV positive count and prevalence
        hiv_num = (group_hiv_status == 1).sum()
        hiv_prevalence = hiv_num / len(group_hiv_status) * 100 if len(group_hiv_status) > 0 else 0
        data_dict['HIV positive'] = f"{hiv_num} ({hiv_prevalence:.2f}%)"
        
        table_data[group_name] = data_dict
    
    # Print the table in LaTeX format
    _print_latex_code_table(table_data, f'Full Data Description - {biomarkers_type}')


def _print_latex_table(data, title):
    table = PrettyTable()
    markers = list(next(iter(data.values())).keys())
    table.field_names = ["Marker"] + list(data.keys())
    for marker in markers: table.add_row([marker] + [data[group][marker] for group in data.keys()])
    print(f"\n{title}\n{table}")

def _print_latex_code_table(data, title):
    """
    Print a LaTeX formatted table with proper code.
    """
    markers = list(next(iter(data.values())).keys())
    columns = list(data.keys())
    n_cols = len(columns) + 1  # +1 for the row label column
    
    print(f"\n{title}")
    print("=" * 80)
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{{title}}}")
    print(f"\\begin{{tabular}}{{l{'c' * len(columns)}}}")
    print("\\hline")
    
    # Header row
    header = " & ".join([""] + columns) + " \\\\"
    print(header)
    print("\\hline")
    
    # Data rows
    for marker in markers:
        row_data = [marker] + [data[col][marker] for col in columns]
        # Escape underscores for LaTeX
        row_data = [str(item).replace("_", "\\_") for item in row_data]
        row = " & ".join(row_data) + " \\\\"
        print(row)
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\label{tab:data_description}")
    print("\\end{table}")
    print("=" * 80)

"""
This file is used to comput the mean and se 
for several variables in the maternal data
Functions print the mean and se in a Latex table formal
"""
import numpy as np
from scipy.stats import sem, ttest_ind
from prettytable import PrettyTable
import sys

from common.feature_engineering import FeatureEngineer

from neonatal_HIV_exposure_biomarkers.load_neonatal_data import load_neonatal_data
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.stability_analysis_utils import load_neonatal_delta_ROC_results

confounders = ["sex", "birth.weight.g", "alcohol", "smoking", "diabetes", "hypertension", "gravity",
                "gadelivery", "parity", "multiple.birth", "maternal.age.delivery", "prepregnancy.bmi"]


significant_biomarker_names =['tyr', 'c3', 'arg', 'met', 'phe', 'irt', 'trec', 'gly', 'orn', 'tsh',\
                                'c14.1', 'c14.2', 'c6', 'c10.1', 'c18', 'c5oh', 'c14.2']


def stats_counts_neonatal(country, gender):
    biomarkers, HIV_status, clinical_data  = load_neonatal_data(country=country)
    male_mask, female_mask = clinical_data['sex'] == 1, clinical_data['sex'] == 2
    table = PrettyTable(['Category', 'Count', 'HIV+', 'Control', 'HIV+ prevalence'])
    table.add_row(['Total', len(HIV_status), (HIV_status == 1).sum(), (HIV_status == 0).sum(), (HIV_status == 1).sum() / len(HIV_status)])
    table.add_row(['Male', male_mask.sum(), (HIV_status[male_mask] == 1).sum(), (HIV_status[male_mask] == 0).sum(), (HIV_status[male_mask] == 1).sum() / male_mask.sum()])
    table.add_row(['Female', female_mask.sum(), (HIV_status[female_mask] == 1).sum(), (HIV_status[female_mask] == 0).sum(), (HIV_status[female_mask] == 1).sum() / female_mask.sum()])
    table.add_row(['# biomarkers', len(biomarkers.columns), '', '', ''])
    table.add_row(['# clinical variables', len(clinical_data.columns), '', '', ''])
    print(f"\nData Counts\n{table}")


def mean_se_clinical_data_neonatal(country, gender):
    print('Computing the mean and se of the clinical data for both genders, male and female')
    biomarkers, HIV_status, clinical_data  = load_neonatal_data(country=country)
    clinical_data = clinical_data[confounders]
    female_mask, male_mask = clinical_data['sex'] == 2, clinical_data['sex'] == 1
    clinical_data = clinical_data.drop(columns=['sex'])


    data = {'All': _compute_stats(clinical_data, HIV_status), 
            'Male': _compute_stats(clinical_data[male_mask], HIV_status[male_mask]),
            'Female': _compute_stats(clinical_data[female_mask], HIV_status[female_mask])}
    _print_latex_table(data, 'Clinical Data Statistics')

def mean_se_significant_biomarkers_neonatal(country, gender):
    print('Computing the mean and se of the significant biomarkers for both genders, male and female')
    biomarkers, HIV_status, clinical_data  = load_neonatal_data(country=country)
    clinical_data = clinical_data[confounders]

    female_mask, male_mask = clinical_data['sex'] == 2, clinical_data['sex'] == 1

    feature_engineer_1 = FeatureEngineer(feature_normalization='std_scaling', verbose=False)
    biomarkers= feature_engineer_1.fit_transform(biomarkers, verbose=False)
    sig_bio = biomarkers.drop(columns=[col for col in biomarkers.columns if col not in significant_biomarker_names])
    data = {'All': _compute_stats(sig_bio, HIV_status),
             'Male': _compute_stats(sig_bio[male_mask], HIV_status[male_mask]),
            'Female': _compute_stats(sig_bio[female_mask], HIV_status[female_mask])}
    
    _print_latex_table(data, 'Significant Biomarkers Statistics')


def adjusted_unadjusted_OR_neonatal(country, gender):
    print('Computing the adjusted and unadjusted OR for the significant biomarkers from saved data')
    results_female, baseline_results_female, unadjusted_results_female = load_neonatal_delta_ROC_results(country, 2, se_rule=False)    
    results_male, baseline_results_male, unadjusted_results_male = load_neonatal_delta_ROC_results(country, 1, se_rule=False)  
    
    def fmt_pval(p):
        if p < 0.001:
            return "p<0.001"
        else:
            return f"p={p:.3f}"
    
    def compute_logOR_with_pval(values):
        # One-sample t-test against 0 (testing if log OR is significantly different from 0)
        from scipy.stats import ttest_1samp
        if len(values) > 1:
            t_stat, p_value = ttest_1samp(values, 0)
            return f"{np.mean(values):.2f}±{sem(values):.2f} ({fmt_pval(p_value)})"
        else:
            return f"{np.mean(values):.2f}±{sem(values):.2f} (N/A)"
    
    # Helper function to process results for a given gender
    def process_gender_results(results, unadjusted_results, gender_name):
        all_features = results[0]['log_odds_ratio'].keys()
        n_trials = len(results)
        per_feature = {f: {'adj': [], 'unadj': [], 'adj_count': 0, 'unadj_count': 0} for f in all_features}
        
        for trial in range(n_trials):
            for feature in all_features:
                if feature in results[trial]['log_odds_ratio'] and results[trial]['log_odds_ratio'][feature] != 0: 
                    per_feature[feature]['adj'].append(results[trial]['log_odds_ratio'][feature])
                    per_feature[feature]['adj_count'] += 1
                if feature in unadjusted_results[trial]['log_odds_ratio'] and unadjusted_results[trial]['log_odds_ratio'][feature] != 0: 
                    per_feature[feature]['unadj'].append(unadjusted_results[trial]['log_odds_ratio'][feature])
                    per_feature[feature]['unadj_count'] += 1
        
        # Filter features and restructure data
        features_to_include = significant_biomarker_names + confounders
        filtered_features = {f: v for f, v in per_feature.items() if f in features_to_include and len(v['adj']) > 0}
        
        return {
            f'{gender_name}_adj_logOR': {f: compute_logOR_with_pval(v['adj']) for f, v in filtered_features.items()},
            f'{gender_name}_adj_freq': {f: f"{v['adj_count']/n_trials:.2f}±{_freq_se(v['adj_count'], n_trials):.2f}" for f, v in filtered_features.items()},
            f'{gender_name}_unadj_logOR': {f: compute_logOR_with_pval(v['unadj']) if len(v['unadj']) > 0 else 'N/A' for f, v in filtered_features.items()},
            f'{gender_name}_unadj_freq': {f: f"{v['unadj_count']/n_trials:.2f}±{_freq_se(v['unadj_count'], n_trials):.2f}" if v['unadj_count'] > 0 else 'N/A' for f, v in filtered_features.items()}
        }
    
    # Process both genders
    female_data = process_gender_results(results_female, unadjusted_results_female, 'Female')
    male_data = process_gender_results(results_male, unadjusted_results_male, 'Male')
    
    # Combine both gender results
    restructured_data = {**female_data, **male_data}
    
    _print_latex_table(restructured_data, f'Log Odds Ratios - {country}')

# Helper function to calculate SE for frequency (proportion)
def _freq_se(count, n): 
    p = count / n
    return np.sqrt(p * (1 - p) / n)


def _compute_stats(df, HIV_status):
    def fmt(val): 
        # Convert to scalar if it's a Series
        if hasattr(val, 'item'):
            val = val.item()
        return f"{val:.2e}" if abs(val) >= 1000 else f"{val:.2f}"
    
    def fmt_pval(p):
        if p < 0.001:
            return "p<0.001"
        else:
            return f"p={p:.3f}"
    
    hiv_pos, hiv_neg = df[HIV_status == 1], df[HIV_status == 0]
    stats_dict = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        # Perform t-test
        t_stat, p_value = ttest_ind(hiv_pos[col].dropna(), hiv_neg[col].dropna(), equal_var=False)
        
        stats_dict[col] = (
            f"HIV+: {fmt(hiv_pos[col].mean())}±{fmt(hiv_pos[col].sem())} ({fmt_pval(p_value)})\n"
            f"Control: {fmt(hiv_neg[col].mean())}±{fmt(hiv_neg[col].sem())}"
        )
    
    return stats_dict

def full_neonatal_data_discription(country: str = 'Both'):
    """
    Print a comprehensive data description table comparing All patients, HIV+, and Control groups.
    Each row is repeated three times for All, Male, and Female sex groups (stacked together).
    Outputs in LaTeX table format.
    
    Args:
        country: Country to load data for (default: 'Both')
    """
    print(f'Computing full data description for {country} across HIV status and sex groups')
    
    biomarkers, HIV_status, clinical_data = load_neonatal_data(country=country)
    clinical_data = clinical_data[confounders]
    
    # Create masks for sex groups
    male_mask = clinical_data['sex'] == 1
    female_mask = clinical_data['sex'] == 2
    
    # Process three HIV status groups: All, HIV+, Control
    hiv_pos_mask = HIV_status == 1
    hiv_neg_mask = HIV_status == 0
    
    groups = {
        'All patients': (clinical_data, HIV_status, slice(None)),
        'HIV+': (clinical_data[hiv_pos_mask], HIV_status[hiv_pos_mask], hiv_pos_mask),
        'Control': (clinical_data[hiv_neg_mask], HIV_status[hiv_neg_mask], hiv_neg_mask)
    }
    
    # Collect data for each group and sex
    group_sex_data = {}
    for group_name, (group_clinical, group_hiv_status, hiv_mask) in groups.items():
        group_sex_data[group_name] = {}
        
        # For each sex category (All, Male, Female)
        sex_groups = {
            'All': slice(None),
            'Male': male_mask[hiv_mask] if not isinstance(hiv_mask, slice) else male_mask,
            'Female': female_mask[hiv_mask] if not isinstance(hiv_mask, slice) else female_mask
        }
        
        for sex_name, sex_mask in sex_groups.items():
            # Get data for this sex group
            if isinstance(sex_mask, slice):
                sex_clinical = group_clinical
                sex_hiv_status = group_hiv_status
            else:
                sex_clinical = group_clinical[sex_mask]
                sex_hiv_status = group_hiv_status[sex_mask]
            
            group_sex_data[group_name][sex_name] = {
                'n_markers': len(biomarkers.columns),
                'cohort': len(sex_hiv_status),
                'clinical': sex_clinical,
                'hiv_status': sex_hiv_status
            }
    
    # Now reorganize into table format with stacked rows
    table_data = {}
    for group_name in groups.keys():
        data_dict = {}
        
        # Number of markers - stacked by sex
        for sex_name in ['All', 'Male', 'Female']:
            data_dict[f'Number of markers - {sex_name}'] = str(group_sex_data[group_name][sex_name]['n_markers'])
        
        # Cohort - stacked by sex
        for sex_name in ['All', 'Male', 'Female']:
            data_dict[f'Cohort - {sex_name}'] = str(group_sex_data[group_name][sex_name]['cohort'])
        
        # Clinical variables - stacked by sex
        # Get list of clinical variables (excluding sex)
        sample_clinical = group_sex_data[group_name]['All']['clinical']
        clinical_vars = [col for col in sample_clinical.select_dtypes(include=[np.number]).columns if col != 'sex']
        
        for col in clinical_vars:
            for sex_name in ['All', 'Male', 'Female']:
                sex_clinical = group_sex_data[group_name][sex_name]['clinical']
                clinical_numeric = sex_clinical.select_dtypes(include=[np.number])
                if col in clinical_numeric.columns:
                    mean_val = clinical_numeric[col].mean()
                    se_val = clinical_numeric[col].sem()
                    # Format large numbers with scientific notation
                    if abs(mean_val) >= 1000:
                        data_dict[f'{col} - {sex_name}'] = f"{mean_val/1000:.2f}±{se_val/1000:.2f}"
                    
                    else:
                        data_dict[f'{col} - {sex_name}'] = f"{mean_val:.2f}±{se_val:.2f}"
        
        # HIV positive - stacked by sex
        for sex_name in ['All', 'Male', 'Female']:
            sex_hiv_status = group_sex_data[group_name][sex_name]['hiv_status']
            hiv_num = (sex_hiv_status == 1).sum()
            hiv_prevalence = hiv_num / len(sex_hiv_status) * 100 if len(sex_hiv_status) > 0 else 0
            data_dict[f'HIV positive - {sex_name}'] = f"{hiv_num} ({hiv_prevalence:.2f}\%)"
        
        table_data[group_name] = data_dict
    
    # Print the table in LaTeX format
    _print_latex_code_table(table_data, f'Full Data Description - {country}')
    
    # Print additional summary table by country and gender
    _print_country_gender_summary()


def _print_country_gender_summary():
    """
    Print a summary table showing cohort statistics by country and gender.
    Rows: Kenya (All/Male/Female), Zambia (All/Male/Female), All sites (All/Male/Female)
    Columns: All patients, HIV+, Control
    """
    print('\nComputing country and gender summary')
    
    countries = ['Kenya', 'Zambia', 'Both']
    country_display_names = {'Kenya': 'Kenya', 'Zambia': 'Zambia', 'Both': 'All sites'}
    
    # Table with Country/Site and Gender as part of row labels
    table_data = {}
    
    # Process each HIV status group
    for hiv_group_name in ['All patients', 'HIV+', 'Control']:
        data_dict = {}
        
        for country in countries:
            # Load data for this country
            biomarkers, HIV_status, clinical_data = load_neonatal_data(country=country)
            
            # Create HIV status mask
            if hiv_group_name == 'All patients':
                hiv_mask = slice(None)
                filtered_HIV_status = HIV_status
                filtered_clinical = clinical_data
            elif hiv_group_name == 'HIV+':
                hiv_mask = HIV_status == 1
                filtered_HIV_status = HIV_status[hiv_mask]
                filtered_clinical = clinical_data[hiv_mask]
            else:  # Control
                hiv_mask = HIV_status == 0
                filtered_HIV_status = HIV_status[hiv_mask]
                filtered_clinical = clinical_data[hiv_mask]
            
            country_display = country_display_names[country]
            
            # All gender
            total_count = len(filtered_HIV_status)
            hiv_pos_count = (filtered_HIV_status == 1).sum()
            hiv_prevalence = hiv_pos_count / total_count * 100 if total_count > 0 else 0
            data_dict[f'{country_display} - All'] = f"{total_count} ({hiv_prevalence:.1f}\%)"
            
            # Male
            if 'sex' in filtered_clinical.columns:
                male_mask = filtered_clinical['sex'] == 1
                male_count = male_mask.sum()
                male_hiv_pos = (filtered_HIV_status[male_mask] == 1).sum() if male_count > 0 else 0
                male_hiv_prevalence = male_hiv_pos / male_count * 100 if male_count > 0 else 0
                data_dict[f'{country_display} - Male'] = f"{male_count} ({male_hiv_prevalence:.1f}\%)"
                
                # Female
                female_mask = filtered_clinical['sex'] == 2
                female_count = female_mask.sum()
                female_hiv_pos = (filtered_HIV_status[female_mask] == 1).sum() if female_count > 0 else 0
                female_hiv_prevalence = female_hiv_pos / female_count * 100 if female_count > 0 else 0
                data_dict[f'{country_display} - Female'] = f"{female_count} ({female_hiv_prevalence:.1f}\%)"
        
        table_data[hiv_group_name] = data_dict
    
    # Print the table in LaTeX format
    _print_latex_code_table(table_data, 'Country and Gender Summary')


def _print_latex_table(data, title):
    table = PrettyTable()
    # Collect all unique markers from all groups
    all_markers = set()
    for group_data in data.values():
        all_markers.update(group_data.keys())
    markers = sorted(all_markers)
    
    table.field_names = ["Marker"] + list(data.keys())
    for marker in markers: 
        table.add_row([marker] + [data[group].get(marker, 'N/A') for group in data.keys()])
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
    print("\\label{tab:neonatal_data_description}")
    print("\\end{table}")
    print("=" * 80)

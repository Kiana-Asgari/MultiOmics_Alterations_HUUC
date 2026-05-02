import numpy as np
import scipy.stats as stats
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
import os
import pandas as pd
from configs import config
import matplotlib.colors as mcolors
from common.modeling_utils import set_seaborn_style
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.utils import (
    prepare_data,
    statsmodels_train_model,
)

confounders = ["const","sex", "birth.weight.g", "alcohol", "smoking", "diabetes", "hypertension", "gravity",
                "gadelivery", "parity", 
                "multiple.birth", "maternal.age.delivery", "prepregnancy.bmi", "site"]
confounders_nice_name_mapping = {"const": "constant", 
                                    "sex": "Sex", 
                                    "birth.weight.g": "Birth Weight",
                                    "alcohol": "Alcohol", 
                                    "smoking": "Smoking",
                                    "diabetes": "Diabetes",
                                    "hypertension": "Hypertension",
                                        "gravity": "Gravity", 
                                        "gadelivery": "GA", 
                                        "parity": "Parity", 
                                        "multiple.birth": "Multiple Birth",
                                        "maternal.age.delivery": "Maternal age", 
                                        "prepregnancy.bmi": "Pre-preg BMI", 
                                        "site": "Site"}

groups = {
  "free_carnitine":      ["c0"],
  "total_pool_c2":    ["c2"],

  "short_chain":         ["c3", "c4", "c5"],

  "medium_chain_sat":    ["c6", "c8","c10", "c12"],
  "medium_chain_unsat":  ["c8.1", "c10.1", "c12.1"],
  "medium_chain_all":    ["c6", "c8","c10", "c12", "c8.1", "c10.1", "c12.1"],

  "long_chain_sat":      ["c14", "c16", "c18"],
  "long_chain_unsat":    ["c14.2", "c18.1", "c18.2"],
  "long_chain_all":      ["c14", "c16", "c18", "c14.2", "c18.1", "c18.2"],

  "hydroxy_acyl":        ["c4oh", "c14oh", "c16oh", "c16.1oh", "c18oh", "c18.1oh"],
  "dicarboxyl_dc":       ["c3dc","c4dc","c5dc","c6dc"],
  "unsat":               ["c8.1", "c10.1", "c12.1", "c14.2", "c18.1", "c18.2"],
  "sat":                 ["c3", "c4", "c5", "c6", "c8","c10", "c12", "c14", "c16", "c18"],

  "LPUFA":               ["c14.2", "c18.2"],
  "LC-MUFA":             ["c14.1", "c18.1"],
}
def _is_sex_feature(feature_name) -> bool:
    return str(feature_name).strip().lower() == "sex"

def _get_feature_display_name(feature_code) -> str:
    """Neonatal display name used across plots/tables (confounder aliases first, then marker map)."""
    if feature_code is None:
        return ""
    raw = str(feature_code).strip()
    key = raw.lower()
    if key in confounders_nice_name_mapping:
        return confounders_nice_name_mapping[key]
    return config.get_neonatal_marker_full_name(raw)


def _normalize_sex(series):
    """Map sex column (numeric or string) to 'Female' / 'Male'."""
    lookup = {"female": "Female", "f": "Female", "2": "Female",
              "male": "Male",   "m": "Male",   "1": "Male"}
    return series.astype(str).str.strip().str.lower().map(lookup)


def _normalize_hiv(series):
    """Map HIV status column (numeric or string) to 'Neg' / 'Pos'."""
    lookup = {"neg": "Neg", "0": "Neg", "pos": "Pos", "1": "Pos", "2": "Pos"}
    return pd.Series(series).astype(str).str.strip().str.lower().map(lookup)


def _remove_outliers(df, group_cols, value_col, k=2.0):
    """Drop rows outside k*IQR per group."""
    g = df.groupby(group_cols)[value_col]
    q1, q3 = g.transform("quantile", 0.25), g.transform("quantile", 0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    mask = df[value_col].between(q1 - k * iqr, q3 + k * iqr) | iqr.isna()
    return df.loc[mask].copy()


def violin_plots_vs_features(country, features):
    confs = ["sex", "birth.weight.g", "alcohol", "smoking", "diabetes",
             "hypertension", "gravity", "gadelivery", "parity",
             "multiple.birth", "maternal.age.delivery", "prepregnancy.bmi"]
    (X, y, demographics), country, gender = prepare_data(confs, country=country, gender=-1)
    if features in groups.keys():
        stem = features
        features = groups[features]
    else:
        if isinstance(features, str):
            features = [features]
        features = [str(f).strip() for f in (features or []) if str(f).strip()]
        stem = "_".join(features)
    df = X[features].apply(pd.to_numeric, errors="coerce").copy()
    df["value"] = df[features].sum(axis=1)
    df["Sex"] = _normalize_sex(demographics["sex"])
    df["HIV"] = _normalize_hiv(y)
    df = df[["Sex", "HIV", "value"]].dropna()

    df = _remove_outliers(df, ["Sex", "HIV"], "value")
    df["value"] = (df["value"] - df["value"].mean()) / df["value"].std()

    df["HIV"] = df["HIV"].map({"Neg": "HUUC", "Pos": "HEUC"})
    palette = {"HUUC": "#7BAFD4", "HEUC": "#D45B5B"}

    # --- Professional plot styling ---
    set_seaborn_style()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')
    fig.tight_layout(pad=3.0, h_pad=2.5, w_pad=2.5)

    sns.violinplot(
        data=df, x="Sex", y="value", hue="HIV",
        order=["Female", "Male"], hue_order=["HUUC", "HEUC"],
        inner="quartile",
        inner_kws={"color": "0.25", "linewidth": 1},
        cut=0, linewidth=0.8, palette=palette, saturation=0.9, bw_adjust=0.8, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="x", length=3, width=0.6)


    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color("black")


    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if handles and 'free' in stem:
        ax.legend(
            handles, labels,
            loc="upper left", fontsize=22, frameon=True, fancybox=False,
            edgecolor="0.4", facecolor="white", framealpha=0.8,
        )

    save_dir = "neonatal_HIV_exposure_biomarkers/figures/violin_plots"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"violin_{stem}_{country}+{gender}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\nViolin plot saved to: {out_path}")
    plt.show()
   


def plot_stability_path(path_results:list, country:str, gender:str, significant_features:list):
    # results:[{lambda: {feature: coef}}]
    gender = str(gender)
    gender = 'male' if gender == '1' else 'female' if gender == '2' else 'both'
    all_lambdas = list(path_results[0].keys())
    first_lambda = all_lambdas[0]
    all_features = path_results[0][first_lambda].keys()

    feature_per_lambda_frequency = pd.DataFrame(0.0, columns=all_lambdas, index=list(all_features), dtype=float)
    feature_per_lambda_coef = pd.DataFrame(0.0, columns=all_lambdas, index=list(all_features), dtype=float)

    for trial_path in path_results:
        for lambda_ in all_lambdas:
            for feature in all_features:
                #if feature in confounders:
                #    continue
                if feature in trial_path[lambda_].keys() and trial_path[lambda_][feature] != 0:
                    feature_per_lambda_frequency.loc[feature, lambda_] += 1
                    feature_per_lambda_coef.loc[feature, lambda_] += trial_path[lambda_][feature]

    feature_per_lambda_frequency = feature_per_lambda_frequency / len(path_results)
    feature_per_lambda_coef = feature_per_lambda_coef / len(path_results)

    
    # Format gender and country labels to match ROC plots
    gender_label = 'Male neonates' if gender == 'male' else 'Female neonates' if gender == 'female' else 'Neonates'
    country_text = "Kenya and Zambia" if country == "Both" else country

    _plot_feature_per_lambda(feature_per_lambda_frequency, 
                             f'{gender_label} from {country_text}',
                             significant_features=significant_features,
                             confounders=confounders,
                             country=country,
                             gender=gender)


def _plot_feature_per_lambda(
    feature_per_lambda_frequency,
    title,
    significant_features: list | None = None,
    confounders: list | None = None,
    legend_kwargs: dict | None = None,
    country: str = None,
    gender: str = None,
):
    """Neonatal stability-path plot styled to match maternal `plot_lasso_path`."""

    # Visual template aligned with maternal lasso-path plot (ln(lambda), subtle y-grid, framed legend on the right).
    fig, ax = plt.subplots(figsize=(8, 6.5))
    x = np.log(np.asarray(feature_per_lambda_frequency.columns, dtype=float))  # natural log (ln)
    ylabel = "Coefficient" if "coefficient" in str(title).lower() else "Selection Frequency"

    conf_set = set(confounders or [])
    existing_significant = []
    if significant_features is not None:
        for f in significant_features:
            # Be tolerant of accidental whitespace in curated lists (e.g. " c5oh").
            f_clean = str(f).strip()
            if f_clean in feature_per_lambda_frequency.index and (f_clean not in conf_set) and (not _is_sex_feature(f_clean)):
                existing_significant.append(f_clean)

    # Fast background rendering: draw many paths using LineCollection (much faster than per-feature ax.plot).
    features = np.asarray(feature_per_lambda_frequency.index, dtype=object)
    y_all = feature_per_lambda_frequency.to_numpy(dtype=float, copy=False)

    sig_set = set(existing_significant or [])
    sex_mask = np.asarray([_is_sex_feature(f) for f in features], dtype=bool)
    conf_mask = np.isin(features, list(conf_set)) if conf_set else np.zeros(features.shape[0], dtype=bool)
    sig_mask = np.isin(features, list(sig_set)) if sig_set else np.zeros(features.shape[0], dtype=bool)
    bg_mask = ~(sex_mask | conf_mask | sig_mask)

    y_bg = y_all[bg_mask]
    if y_bg.size:
        chunk_size = 5000
        n_lines = y_bg.shape[0]
        for start in range(0, n_lines, chunk_size):
            end = min(start + chunk_size, n_lines)
            ys_chunk = y_bg[start:end]
            xs = np.tile(x, (ys_chunk.shape[0], 1))
            segments = np.stack((xs, ys_chunk), axis=2)  # (n_lines, n_points, 2)
            lc = LineCollection(segments, colors="grey", linewidths=1.5, alpha=0.5, zorder=1)
            ax.add_collection(lc)
        ax.autoscale_view()

    # Highlight significant features
    if existing_significant:
        colors = cm.get_cmap("gist_rainbow")(np.linspace(0, 1, 2 * max(len(existing_significant), 1)))
        for i, feature in enumerate(existing_significant):
            ax.plot(
                x,
                feature_per_lambda_frequency.loc[feature],
                linewidth=1.7,
                alpha=0.85,
                label=_get_feature_display_name(feature),
                color=colors[2 * i % len(colors)],
                zorder=2,
            )

    # Styling (match maternal)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    ax.set_xlabel(r"ln($\lambda$)", fontsize=22)
    if gender == 1 or str(gender).lower() == 'male':
        ax.set_ylabel(ylabel, fontsize=22)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5, color="#CCC")
    ax.set_xlim(right=-1.05)

    if existing_significant:
        legend_params = {
            "bbox_to_anchor": (1.02, 1),
            "loc": "upper left",
            "fontsize": 17,
            "frameon": True,
            "fancybox": False,
            "shadow": False,
            "ncol": 1,
            "edgecolor": "black",
            "facecolor": "white",
            "framealpha": 0.8,
        }
        if legend_kwargs:
            legend_params.update(legend_kwargs)
        legend = ax.legend(**legend_params)
        legend.get_frame().set_linewidth(1.5)

    ax.tick_params(axis="both", which="major", labelsize=18)

    save_dir = "neonatal_HIV_exposure_biomarkers/figures/stability_path"
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    out_path = f"{save_dir}/{title}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\nStability path saved to: {out_path}")
    plt.show()


def delta_ROC_tables(results, baseline_results, country, gender):
    # Mirror plot_delta_ROC logic: delta is per-trial adjusted - baseline.
    auc_adj = np.asarray([r["auc"] for r in results], dtype=float)
    auc_base = np.asarray([r["auc"] for r in baseline_results], dtype=float)
    n = int(min(len(auc_adj), len(auc_base)))
    auc_adj = auc_adj[:n]
    auc_base = auc_base[:n]
    delta_auc = auc_adj - auc_base

    # Relative Discriminative Gain (RDG): per-trial relative gain relative to a random baseline (AUC=0.5).
    # RDG = (AUC_adj - AUC_base) / (AUC_base - 0.5)
    denom = (auc_base - 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        rdg = np.where(np.abs(denom) > 0, delta_auc / denom, np.nan)

    std_delta = float(np.std(delta_auc)) if n > 0 else np.nan
    z_score = float(np.mean(delta_auc) / std_delta) if (n > 1 and std_delta != 0) else np.nan

    def _summary(x: np.ndarray, null_mean: float):
        x = np.asarray(x, dtype=float)
        mean = float(np.mean(x)) if x.size else np.nan
        if x.size > 1:
            se = float(np.std(x, ddof=1) / np.sqrt(x.size))
            ci_low, ci_high = mean - 1.96 * se, mean + 1.96 * se
            p = float(stats.ttest_1samp(x, popmean=null_mean).pvalue)
        else:
            se, ci_low, ci_high, p = np.nan, np.nan, np.nan, np.nan
        return mean, se, ci_low, ci_high, p

    df = pd.DataFrame(
        [
            ("Baseline AUC", *_summary(auc_base, 0.5)),
            ("Adjusted AUC", *_summary(auc_adj, 0.5)),
            ("ΔAUC (Adj−Base)", *_summary(delta_auc, 0.0)),
        ],
        columns=["Model", "Mean", "SE", "CI95_low", "CI95_high", "p_value"],
    ).set_index("Model")

    # Add RDG + its CI as extra columns (only meaningful for the delta row).
    df["Relative Discriminative Gain"] = np.nan
    df["RDG_CI95_low"] = np.nan
    df["RDG_CI95_high"] = np.nan

    rdg_finite = rdg[np.isfinite(rdg)]
    if rdg_finite.size:
        rdg_mean = float(np.mean(rdg_finite))
        if rdg_finite.size > 1:
            rdg_se = float(np.std(rdg_finite, ddof=1) / np.sqrt(rdg_finite.size))
            rdg_ci_low, rdg_ci_high = rdg_mean - 1.96 * rdg_se, rdg_mean + 1.96 * rdg_se
        else:
            rdg_ci_low, rdg_ci_high = np.nan, np.nan
        df.loc["ΔAUC (Adj−Base)", "Relative Discriminative Gain"] = rdg_mean
        df.loc["ΔAUC (Adj−Base)", "RDG_CI95_low"] = rdg_ci_low
        df.loc["ΔAUC (Adj−Base)", "RDG_CI95_high"] = rdg_ci_high

    latex = df.to_latex(float_format=lambda v: f"{v:.3f}" if pd.notna(v) else "")
    print(latex)

    save_dir = f"neonatal_HIV_exposure_biomarkers/figures/ROC/{country}/tables"
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"delta_ROC_table_neonatal_{country}+{gender}"
    df.to_csv(os.path.join(save_dir, base_name + ".csv"), index=True)
    with open(os.path.join(save_dir, base_name + ".tex"), "w") as f:
        f.write(latex)
    print(f"Saved delta ROC table to: {save_dir}/{base_name}.tex (+ .csv)")
    return df


def plot_delta_ROC(results, baseline_results, unadjusted_results, country, gender):
    delta_AUC = [result["auc"] - baseline_result["auc"] for result, baseline_result in zip(results, baseline_results)]
    z_score = np.mean(delta_AUC) / np.std(delta_AUC)
    print(f'z_score: {z_score}')

    # plotting the ROC curve and PR curve
    ROC_curves = [trial["ROC_curve"] for trial in results]
    ROC_scores = [trial["auc"] for trial in results]
    ROC_baseline_curves = [trial["ROC_curve"] for trial in baseline_results]
    ROC_baseline_scores = [trial["auc"] for trial in baseline_results]

    # Keep existing behavior: unadjusted curves are intentionally disabled.
    unadjusted_curves = False
    unadjusted_scores = False
    save_name = f'ROC_neonatal_{country}+{gender}'
    _plot_ROC_curve(ROC_curves, ROC_scores, ROC_baseline_curves, ROC_baseline_scores, unadjusted_curves, unadjusted_scores, saving_name=save_name, country=country, gender=gender)






def _to_series(log_odds_ratio, feature_names):
    """Convert log_odds_ratio to pandas Series."""
    return pd.Series(log_odds_ratio, index=feature_names) if isinstance(log_odds_ratio, list) else pd.Series(log_odds_ratio)

def plot_significant_features(results, feature_names, country, gender, freq_threshold):
    per_feature_results = {f: {'mean_log_odds': 0, 'mean_selection_frequency': 0} for f in feature_names}
    for trial in results:
        log_odds = _to_series(trial['log_odds_ratio'], trial['feature_names'])
        for feature in feature_names:
            val = float(log_odds.get(feature, 0.0))
            per_feature_results[feature]['mean_log_odds'] += val  # include zeros for unselected/missing
            per_feature_results[feature]['mean_selection_frequency'] += (val != 0)
    n_trials = len(results)
    significant_features = []
    for feature in feature_names:
        freq = per_feature_results[feature]['mean_selection_frequency']
        per_feature_results[feature]['mean_log_odds'] /= freq#n_trials if n_trials > 0 else 1
        per_feature_results[feature]['mean_selection_frequency'] /= n_trials
        if per_feature_results[feature]['mean_selection_frequency'] >= freq_threshold:
            significant_features.append(feature)
        print(f"{feature}: frequency {freq}, log_odds {per_feature_results[feature]['mean_log_odds']:.3f}")

    

    _box_plot_features(results, per_feature_results, country, gender, confounders, freq_threshold)
    return significant_features


def _box_plot_features(results, per_feature_results, country, gender, confounders, freq_threshold):
    def _add_feature_background(ax, feats):
        for i, _feat in enumerate(feats, start=1):
            # Ensure the stripe behind the "sex" feature is light blue for consistency/visibility.
            if str(_feat).strip().lower() == "sex":
                facecolor = "lightblue"
            else:
                facecolor = "white" if i % 2 else "lightgrey"
            ax.axvspan(i - 0.5, i + 0.5, facecolor=facecolor, zorder=0)

    def _save_show(fig, path):
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

    def _filter_by_quantile_interval(sorted_feat_vals, q_low=0.25, q_high=0.75, zero_tol=1e-12):
        """Drop features whose [q_low, q_high] quantile interval includes 0."""
        kept, dropped = [], []
        for feat, vals in sorted_feat_vals:
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            lo = float(np.quantile(arr, q_low))
            hi = float(np.quantile(arr, q_high))
            if (lo - zero_tol) <= 0.0 <= (hi + zero_tol):
                dropped.append(feat)
            if feat.lower() in confounders: # do not include confounders in the box plot
                dropped.append((feat, vals))
            else:
                kept.append((feat, vals))
        return kept, dropped, q_low, q_high

    gender_label = {'1': 'Male neonates', '2': 'Female neonates', '-1': 'Neonates', 
                    'male': 'Male neonates', 'female': 'Female neonates', 'both': 'Neonates'}.get(str(gender), 'Neonates')
    frequent_features = {f: d for f, d in per_feature_results.items() if d['mean_selection_frequency'] >= freq_threshold}
    feature_log_odds = {f: [] for f in frequent_features.keys()}
    
    for trial in results:
        log_odds = _to_series(trial['log_odds_ratio'], trial['feature_names'])
        for feature in frequent_features.keys():
            feature_log_odds[feature].append(float(log_odds.get(feature, 0.0)))  # include zeros for unselected/missing
    
    if not feature_log_odds:
        print(f"No features with selection frequency >= {freq_threshold}")
        return
    
    sorted_features = sorted(feature_log_odds.items(), key=lambda x: np.median(x[1]), reverse=True)
    filtered_sorted_features, dropped_due_to_zero_in_iqr, q_low, q_high = _filter_by_quantile_interval(sorted_features)


    original_feature_names = [f for f, _ in filtered_sorted_features]
    plot_data = [vals for _, vals in filtered_sorted_features]
    feature_labels = [confounders_nice_name_mapping.get(f, config.get_neonatal_marker_full_name(f)) for f in original_feature_names]
    medians = [np.median(vals) for vals in plot_data]

    if gender == 2:
        fig_size = (6,4.5)
    elif gender == 1:
        fig_size = (8,4.5)
    else:
        fig_size = (8,4.5)  
    
    fig, ax = plt.subplots(figsize=fig_size)#(0.3*len(plot_data), 5)) #plt.subplots(figsize=(max(11, len(plot_data) * 0.45 + 2), 6))
    _add_feature_background(ax, original_feature_names)
    bp = ax.boxplot(plot_data, vert=True, patch_artist=True, labels=feature_labels, showfliers=False,
                    widths=0.7,
                    medianprops={'color': 'black', 'linewidth': 1.5}, whiskerprops={'color': 'black', 'linewidth': 1.5},
                    capprops={'color': 'black', 'linewidth': 1.5}, flierprops={'marker': 'o', 'markerfacecolor': 'darkgray', 
                    'markersize': 5, 'alpha': 0.5, 'markeredgecolor': 'none'})
    
    for i, (orig_name, patch, med) in enumerate(zip(original_feature_names, bp['boxes'], medians)):
        if orig_name in confounders:
            patch.set(facecolor='lavender', edgecolor='black', linewidth=1.5, alpha=0.7)
            for j in [i*2, i*2+1]:
                bp['whiskers'][j].set_color('black')
                bp['caps'][j].set_color('black')
            bp['medians'][i].set_color('black')
        else:
            patch.set(facecolor=('salmon' if med >= 0 else 'lightskyblue'), edgecolor='black', linewidth=1, alpha=0.85)
    
    ax.axhline(0, color='#333', linestyle='--', linewidth=1.2, alpha=0.8, zorder=0)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5, color='#CCC')
    ax.set_axisbelow(True)
    # Force 0.5 spacing on major y ticks without manually setting tick positions.
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=26)
    ax.set_xticklabels(feature_labels, rotation=90, ha='center', fontsize=27)

    
    save_dir = f'neonatal_HIV_exposure_biomarkers/figures/box_plots/{country}'
    os.makedirs(save_dir, exist_ok=True)
    _save_show(fig, f'{save_dir}/stability_selection_boxplot_{gender_label}_{country}.pdf')
    print(f"\nBox plot saved to: {save_dir}/stability_selection_boxplot_{gender_label}_{country}.pdf for freq threshold {freq_threshold}")

    


def _interpolate_roc_curves(curves, fpr_grid):
    """Helper to interpolate ROC curves to common grid."""
    tpr_interp = []
    for fpr, tpr, *_ in curves:
        if len(fpr) > 1:
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr}).drop_duplicates(subset=['fpr']).sort_values('fpr')
            tpr_interp.append(np.interp(fpr_grid, df['fpr'], df['tpr']) if len(df) > 1 else np.zeros_like(fpr_grid))
        else:
            tpr_interp.append(np.zeros_like(fpr_grid))
    tpr_mean = np.mean(tpr_interp, axis=0)
    tpr_se = np.std(tpr_interp, axis=0) / np.sqrt(len(tpr_interp))
    return tpr_mean, tpr_se

def _plot_ROC_curve(AUC_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]], 
                    AUC_scores: list[float],
                    ROC_baseline_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]]=None,
                    ROC_baseline_scores: list[float]=None,
                    unadjusted_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]]=None,
                    unadjusted_scores: list[float]=None,
                    saving_name: str=None, 
                    country: str=None,
                    gender: str=None):
    set_seaborn_style()
    gender_label = 'Male neonates' if gender == 1 else 'Female neonates' if gender == 2 else 'Neonates'
    
    fig, ax_roc = plt.subplots(1, 1, figsize=(2.5, 2))
    fpr_grid = np.linspace(0, 1, 1000)
    
    def _curve_alpha(score, mean_score, std_score, min_alpha=0.01, max_alpha=0.18):
        if std_score == 0:
            return max_alpha
        distance = abs(score - mean_score) / std_score
        return max(min_alpha, max_alpha * np.exp(-2 * distance**2))

    def _plot_model(curves, scores, colormap_name, mean_color, label):
        colors = plt.cm.get_cmap(colormap_name)(np.linspace(0.2, 0.8, len(curves)))
        mean_score, std_score = np.mean(scores), np.std(scores)

        for (fpr, tpr, *_), color, score in zip(curves, colors, scores):
            if len(fpr) <= 1:
                continue
            df = pd.DataFrame({'fpr': fpr, 'tpr': tpr}).drop_duplicates(subset=['fpr']).sort_values('fpr')
            if len(df) > 1:
                ax_roc.plot(df['fpr'], df['tpr'], color=color, alpha=_curve_alpha(score, mean_score, std_score),
                            linewidth=0.6, zorder=1)

        tpr_mean, tpr_se = _interpolate_roc_curves(curves, fpr_grid)
        ax_roc.fill_between(fpr_grid, tpr_mean - tpr_se, tpr_mean + tpr_se,
                            color=mean_color, alpha=0.15, linewidth=0, zorder=2)
        ax_roc.plot(
            fpr_grid, tpr_mean, color=mean_color, linewidth=0.8,
            label=f'{label}', zorder=3
        )
    
    # Plot all model types
    if ROC_baseline_curves and ROC_baseline_scores:
        _plot_model(ROC_baseline_curves, ROC_baseline_scores, 'autumn', 'orangered', 'Clinical\n Factors')
    
    _plot_model(AUC_curves, AUC_scores, 'viridis', 'blue', 'All Markers')
    
    if unadjusted_curves and unadjusted_scores:
        _plot_model(unadjusted_curves, unadjusted_scores, 'summer', 'green', 'Unadjusted Biomarkers')

    # Style plot
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Random \n classifier', linewidth=0.8, alpha=0.7, zorder=2)

    # ax_roc.set_xlabel('FP Rate', fontsize=20)
    # ax_roc.set_ylabel('TP Rate', fontsize=20)
    ax_roc.set_xticks([])
    ax_roc.set_yticks([])
    ax_roc.set_xlabel('FP Rate', fontsize=8)
    ax_roc.set_ylabel('TP Rate', fontsize=8)
    
    # Style legend with clear border #TODO: add legend
    # if country == "Both" and gender == -1:
    #     legend = ax_roc.legend(fontsize=10.5, edgecolor='black', facecolor='white',
    #                         framealpha=1, fancybox=False, shadow=False, 
    #                         )
    #     ax_roc.set_xlabel('FP Rate', fontsize=10)
    #     ax_roc.set_ylabel('TP Rate', fontsize=10)
    #     legend.get_frame().set_linewidth(0.5)
    
    ax_roc.grid(True, alpha=0.4)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    ax_roc.tick_params(axis='both', which='major', labelsize=20)
    
    # Match border styling
    for spine in ax_roc.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('black')
    
    country_text = "Kenya and Zambia" if country == "Both" else country
    #ax_roc.set_title(f'{gender_label} from {country_text}', fontsize=20, pad=20, fontweight='bold')
    
    plt.tight_layout()
    if saving_name is not None:
        relative_path = f'neonatal_HIV_exposure_biomarkers/figures/ROC/{country}'
        os.makedirs(relative_path, exist_ok=True)
        plt.savefig(os.path.join(relative_path, saving_name) + '.pdf', dpi=300)
    else:
        plt.show()


def table_significant_features_all_genders(results_all, baseline_results_all,
                                            results_male, baseline_results_male,
                                            results_female, baseline_results_female,
                                            country, freq_threshold=0.79):
                                            
    # TODO: This function first finds a list of union of all significant features across all genders, with calculations similar to plot_significant_features function.
    # Then prints a latex table with rows of each significant feature and columns of the mean log odds ratio \pm SE and selection frequency for each gender as the columns.
    # Keep signature stable; baselines are currently unused for this table.
    _ = baseline_results_all, baseline_results_male, baseline_results_female, results_all

    def _per_feature_stats(results: list[dict], feature_names: list[str]) -> dict:
        """Mirror `plot_significant_features` logic.

        - Mean log-odds is averaged over all trials (including zeros when not selected).
        - Selection frequency is fraction of trials where coefficient != 0.
        - SE is computed over all trials (including zeros).
        """
        n_trials = len(results) if results else 0
        out = {f: {"mean": np.nan, "se": np.nan, "freq": 0.0, "n_selected": 0} for f in feature_names}
        if not n_trials:
            return out

        vals_by_feat = {f: [] for f in feature_names}  # all trials (zeros included)
        for trial in results:
            log_odds = _to_series(trial["log_odds_ratio"], trial["feature_names"])
            for f in feature_names:
                v = float(log_odds.get(f, 0.0))
                vals_by_feat[f].append(v)

        for f, vals in vals_by_feat.items():
            arr = np.asarray(vals, dtype=float)
            n_sel = int(np.count_nonzero(arr))
            freq = n_sel / n_trials
            mean = float(np.mean(arr)) if arr.size else np.nan
            se = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else np.nan
            out[f] = {"mean": mean, "se": se, "freq": float(freq), "n_selected": n_sel}
        return out

    def _is_significant(stat: dict, min_abs_mean: float = 0.01) -> bool:
        return stat is not None and np.isfinite(stat.get("mean", np.nan)) and abs(float(stat["mean"])) > min_abs_mean and float(stat.get("freq", 0.0)) >= float(freq_threshold)

    # Compute union of available features across male + female results (we only report those).
    all_features = sorted({f for r in (results_male or []) for f in (r.get("feature_names", []) or [])} | {f for r in (results_female or []) for f in (r.get("feature_names", []) or [])})

    stats_male = _per_feature_stats(results_male, all_features)
    stats_female = _per_feature_stats(results_female, all_features)

    def _fmt_mean_ci(mean: float, se: float, z: float = 1.96) -> str:
        """Format mean with 95% CI using normal approximation: mean ± z * SE."""
        if not np.isfinite(mean):
            return ""
        if not np.isfinite(se):
            return f"{mean:.3f}"
        lo, hi = mean - z * se, mean + z * se
        return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

    def _sort_key(stats: dict, f: str) -> tuple[float, float]:
        s = stats.get(f, {})
        freq = float(s.get("freq", np.nan))
        mean = float(s.get("mean", np.nan))
        return ((freq if np.isfinite(freq) else float("-inf")), (mean if np.isfinite(mean) else float("-inf")))

    male_sig = sorted(
        [f for f in all_features if f not in confounders and _is_significant(stats_male.get(f))],
        key=lambda f: _sort_key(stats_male, f),
        reverse=True,
    )
    female_sig = sorted(
        [f for f in all_features if f not in confounders and _is_significant(stats_female.get(f))],
        key=lambda f: _sort_key(stats_female, f),
        reverse=True,
    )
    male_set, female_set = set(male_sig), set(female_sig)
    both = sorted(male_set & female_set, key=lambda f: _sort_key(stats_male, f), reverse=True)  # order by male freq
    male_only = sorted(male_set - female_set, key=lambda f: _sort_key(stats_male, f), reverse=True)
    female_only = sorted(female_set - male_set, key=lambda f: _sort_key(stats_female, f), reverse=True)
    sig_features = both + male_only + female_only

    if not sig_features:
        print(
            f"No significant features found with |mean log-odds| > 0.01 and "
            f"selection frequency >= {freq_threshold} in any gender."
        )
        return pd.DataFrame()

    rows = []
    for f in sig_features:
        label = confounders_nice_name_mapping.get(f, config.get_neonatal_marker_full_name(f))
        group = "Both" if f in (male_set & female_set) else "Male only" if f in male_set else "Female only"
        rows.append(
            {
                "Feature": label,
                "feature_code": f,
                "Group": group,
               # "All: mean±SE": _fmt_mean_se(stats_all[f]["mean"], stats_all[f]["se"]),
               # "All: freq": stats_all[f]["freq"],
                "Male: mean (CI95)": _fmt_mean_ci(stats_male[f]["mean"], stats_male[f]["se"]),
                "Male: freq": stats_male[f]["freq"],
                "Female: mean (CI95)": _fmt_mean_ci(stats_female[f]["mean"], stats_female[f]["se"]),
                "Female: freq": stats_female[f]["freq"],
            }
        )

    df = pd.DataFrame(rows).set_index("Feature")

    # Pretty print to console
    pretty_df_console = df.drop(columns=["feature_code"]).copy()
    pretty_df_latex = df.drop(columns=["feature_code"]).copy()

    def _pct_int(v) -> int | None:
        if pd.isna(v):
            return None
        return int(round(float(v) * 100))

    for c in ("Male: freq", "Female: freq"):
        if c not in df.columns:
            continue

        pretty_df_console[c] = df[c].apply(
            lambda v: (
                (lambda p: (f"**{p}%**" if p > 80 else f"{p}%"))(_pct_int(v))
                if _pct_int(v) is not None else ""
            )
        )
        pretty_df_latex[c] = df[c].apply(
            lambda v: (
                (lambda p: (f"\\textbf{{{p}}}" if p > 80 else f"{p}"))(_pct_int(v))
                if _pct_int(v) is not None else ""
            )
        )
    # Dependency-free fallback
    fmt = {}
    def _print_block(title: str, group_name: str):
        block = pretty_df_console[pretty_df_console["Group"] == group_name].drop(columns=["Group"])
        if block.empty:
            return
        print(f"\n{title} (n={len(block)}):\n{block.to_string(max_colwidth=48, justify='left', formatters=fmt)}\n")

    _print_block("Significant in both", "Both")
    _print_block("Male significant only", "Male only")
    _print_block("Female significant only", "Female only")


    # Print + save
    latex = pretty_df_latex.to_latex(escape=False)
    print(latex)

    save_dir = f"neonatal_HIV_exposure_biomarkers/figures/tables/{country}"
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"significant_features_all_genders_{country}_freq{freq_threshold}"
    df.to_csv(os.path.join(save_dir, base_name + ".csv"), index=True)
    with open(os.path.join(save_dir, base_name + ".tex"), "w") as f:
        f.write(latex)
    print(f"Saved significant-features table to: {save_dir}/{base_name}.tex (+ .csv)")
    return df



def table_significant_features_all_genders_all_countries(
    results_male_Both,
    results_female_Both,
    results_male_Kenya,
    results_female_Kenya,
    freq_threshold=0.8,
):
    def _per_feature_stats(results: list[dict], feature_names: list[str]) -> dict:
        n_trials = len(results) if results else 0
        out = {f: {"mean": np.nan, "freq": 0.0} for f in feature_names}
        if not n_trials:
            return out

        vals_by_feat = {f: [] for f in feature_names}
        for trial in results:
            log_odds = _to_series(trial["log_odds_ratio"], trial["feature_names"])
            for f in feature_names:
                vals_by_feat[f].append(float(log_odds.get(f, 0.0)))

        for f, vals in vals_by_feat.items():
            arr = np.asarray(vals, dtype=float)
            out[f] = {"mean": float(np.mean(arr)) if arr.size else np.nan, "freq": float(np.count_nonzero(arr) / n_trials)}
        return out

    def _is_significant(stat: dict, min_abs_mean: float = 0.01) -> bool:
        return stat is not None and np.isfinite(stat.get("mean", np.nan)) and abs(float(stat["mean"])) > min_abs_mean and float(stat.get("freq", 0.0)) >= float(freq_threshold)

    # Kenya (both sexes) is represented elsewhere as gender=-1. If those results exist, use them
    # to annotate biomarker names with an up/down arrow when coefficient sign is consistent across
    # Kenya Male/Female/Both. If unavailable, fall back to no arrows.
    results_all_Kenya = []
    try:
        from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.stability_analysis_utils import (
            load_neonatal_delta_ROC_results,
        )
        results_all_Kenya, _, _ = load_neonatal_delta_ROC_results("Kenya", -1, se_rule=False)
    except Exception:
        results_all_Kenya = []

    all_features = sorted(
        {f for r in (results_male_Both or []) for f in (r.get("feature_names", []) or [])}
        | {f for r in (results_female_Both or []) for f in (r.get("feature_names", []) or [])}
        | {f for r in (results_male_Kenya or []) for f in (r.get("feature_names", []) or [])}
        | {f for r in (results_female_Kenya or []) for f in (r.get("feature_names", []) or [])}
    )

    mB = _per_feature_stats(results_male_Both, all_features)
    fB = _per_feature_stats(results_female_Both, all_features)
    mK = _per_feature_stats(results_male_Kenya, all_features)
    fK = _per_feature_stats(results_female_Kenya, all_features)
    aK = _per_feature_stats(results_all_Kenya, all_features) if results_all_Kenya else {f: {"mean": np.nan, "freq": np.nan} for f in all_features}

    male_sig = {
        f
        for f in all_features
        if f not in confounders and (_is_significant(mB.get(f)) or _is_significant(mK.get(f)))
    }
    female_sig = {
        f
        for f in all_features
        if f not in confounders and (_is_significant(fB.get(f)) or _is_significant(fK.get(f)))
    }
    both = male_sig & female_sig
    male_only = male_sig - female_sig
    female_only = female_sig - male_sig

    def _max_male_freq(f: str) -> float:
        return max(
            float(mB.get(f, {}).get("freq", 0.0)),
            float(mK.get(f, {}).get("freq", 0.0)),
        )

    def _max_female_freq(f: str) -> float:
        return max(
            float(fB.get(f, {}).get("freq", 0.0)),
            float(fK.get(f, {}).get("freq", 0.0)),
        )

    ordered = (
        sorted(both, key=_max_male_freq, reverse=True)
        + sorted(male_only, key=_max_male_freq, reverse=True)
        + sorted(female_only, key=_max_female_freq, reverse=True)
    )

    if not ordered:
        print(
            f"No significant features found with |mean log-odds| > 0.01 and "
            f"selection frequency >= {freq_threshold} in any country/gender."
        )
        return pd.DataFrame()

    def _sign(mean: float) -> int | None:
        if mean is None or (not np.isfinite(mean)):
            return None
        if mean > 0:
            return 1
        if mean < 0:
            return -1
        return 0

    def _label(f: str) -> str:
        return confounders_nice_name_mapping.get(f, config.get_neonatal_marker_full_name(f))

    df = pd.DataFrame(
        [
            {
                "Feature": _label(f),
                "feature_code": f,
                "Male Both: freq": float(mB.get(f, {}).get("freq", 0.0)),
                "Female Both: freq": float(fB.get(f, {}).get("freq", 0.0)),
                "Male Kenya: freq": float(mK.get(f, {}).get("freq", 0.0)),
                "Female Kenya: freq": float(fK.get(f, {}).get("freq", 0.0)),
            }
            for f in ordered
        ]
    ).set_index("Feature")

    def _pct_int(v) -> int | None:
        if pd.isna(v):
            return None
        return int(round(float(v) * 100))

    def _console_pct(v) -> str:
        p = _pct_int(v)
        if p is None:
            return ""
        return f"**{p}%**" if p > 80 else f"{p}%"

    def _latex_pct(v) -> str:
        p = _pct_int(v)
        if p is None:
            return ""
        return f"\\textbf{{{p}}}" if p > 80 else f"{p}"

    def _arrow_for_mean(mean: float, latex: bool) -> str:
        s = _sign(mean)
        if s == 1:
            return "($\\uparrow$) " if latex else "(↑) "
        if s == -1:
            return "($\\downarrow$) " if latex else "(↓) "
        return ""

    def _freq_with_reg_arrow(feature_code: str, freq_val: float, stats_by_feat: dict, latex: bool) -> str:
        """Format frequency as: (↑) 85%  using sign(mean coefficient) for that group."""
        p = _pct_int(freq_val)
        if p is None:
            return ""
        mean = float(stats_by_feat.get(feature_code, {}).get("mean", np.nan))
        arrow = _arrow_for_mean(mean, latex=latex)
        pct = _latex_pct(freq_val) if latex else _console_pct(freq_val)
        return f"{arrow}{pct}"

    pretty_console = df.copy()
    pretty_latex = df.copy()
    freq_col_to_stats = {
        "Male Both: freq": mB,
        "Female Both: freq": fB,
        "Male Kenya: freq": mK,
        "Female Kenya: freq": fK,
    }
    for c, stats_map in freq_col_to_stats.items():
        if c not in df.columns:
            continue
        pretty_console[c] = df.apply(
            lambda row, col=c, sm=stats_map: _freq_with_reg_arrow(str(row["feature_code"]), row[col], sm, latex=False),
            axis=1,
        )
        pretty_latex[c] = df.apply(
            lambda row, col=c, sm=stats_map: _freq_with_reg_arrow(str(row["feature_code"]), row[col], sm, latex=True),
            axis=1,
        )

    def _print_block(title: str, feats: set[str]):
        mask = df["feature_code"].isin(feats)
        block = pretty_console.loc[mask].drop(columns=["feature_code"])
        if block.empty:
            return
        print(f"\n{title} (n={len(block)}):\n{block.to_string(max_colwidth=48, justify='left')}\n")

    _print_block("Significant in both (any country)", both)
    _print_block("Male significant only (any country)", male_only)
    _print_block("Female significant only (any country)", female_only)

    latex = pretty_latex.drop(columns=["feature_code"]).to_latex(escape=False)
    print(latex)

    save_dir = "neonatal_HIV_exposure_biomarkers/figures/tables/all_countries"
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"significant_features_all_countries_freq{freq_threshold}"
    df.to_csv(os.path.join(save_dir, base_name + ".csv"), index=True)
    with open(os.path.join(save_dir, base_name + ".tex"), "w") as f:
        f.write(latex)
    print(f"Saved all-countries frequency table to: {save_dir}/{base_name}.tex (+ .csv)")
    return df.drop(columns=["feature_code"])
                                            
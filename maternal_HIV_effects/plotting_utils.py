import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
import os
import re
import math
import pandas as pd
from configs import config
from common.modeling_utils import set_seaborn_style
from common.data_loader import annotate_metabolomes_kenya
from maternal_HIV_effects.load_maternal_data import load_maternal_data

significant_features_Both = ['fa.22.4', 'pc.18.0.20.3', 'pe.p.16.0.18.2', 'm227t84.pos.rplc', 'm333t291.2.pos.rplc', 'm295t300.neg.rplc']
significant_features_Lipidome = ['ce.20.3', 'fa.20.0', 'fa.20.1', 'fa.22.4', 'fa.22.6', 'pc.16.0.22.4', 'pc.16.0.18.2',
                                 'pc.18.2.18.2', 'pc.18.2.20.4', 'pe.p.16.0.18.2', 'pe.p.16.0.22.4', 'pe.p.18.0.18.2',
                                  'tg.50.5.fa16.0', 'tg.58.6.fa16.0']
significant_features_Metabolome = ['m227t84.pos.rplc', 'm333t291.2.pos.rplc', 'm295t300.neg.rplc']
confounders_maternal_nice_name_mapping = {"bmi": "Pre-preg BMI", 
"sex": "Sex", 
"gaultrasound": "GA (US)",
"parity": "Parity", 
"gadelivery": "GA", 
"agedelivery": "Age",
"maternal.age.delivery": "Age",
 "prepregnancy.bmi": "Pre-preg BMI",
  "site": "Site"}

_KNOWN_BIOMARKER_TYPES = {"Both", "Lipidome", "Metabolome"}

_METABOLOME_ID_TO_REALNAME: dict[str, str] | None = None


def _normalize_feature_id_for_matching(feature_id) -> str:
    """
    Normalize a feature id for fuzzy matching across pipelines.
    - case-insensitive
    - ignore differences between space/'-','_','.' (and any other non-alphanumeric)
    """
    if feature_id is None:
        return ""
    s = str(feature_id).strip().lower()
    # Remove separators + any other punctuation so IDs like:
    #   m333t291.2.pos.rplc  <->  M333T291_2_POS_RPLC
    # match.
    return re.sub(r"[^a-z0-9]+", "", s)


def _load_metabolome_id_to_realname() -> dict[str, str]:
    """Load a normalized-id -> real-name mapping from the Kenya metabolome variable info CSV."""
    global _METABOLOME_ID_TO_REALNAME
    if _METABOLOME_ID_TO_REALNAME is not None:
        return _METABOLOME_ID_TO_REALNAME

    # Resolve path robustly regardless of current working dir.
    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "raw",
            "variable_info_mom_metabolom_kenya.csv",
        )
    )
    if not os.path.exists(csv_path):
        _METABOLOME_ID_TO_REALNAME = {}
        return _METABOLOME_ID_TO_REALNAME

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        _METABOLOME_ID_TO_REALNAME = {}
        return _METABOLOME_ID_TO_REALNAME

    if df.shape[1] < 2:
        _METABOLOME_ID_TO_REALNAME = {}
        return _METABOLOME_ID_TO_REALNAME

    # First column is the id (per request); prefer explicit name column if present.
    id_col = df.columns[0]
    preferred_name_col = "Compound.name" if "Compound.name" in df.columns else df.columns[-1]

    # If the "last column" isn't actually a string name column (e.g., Level), fall back to the last string column.
    name_col = preferred_name_col
    if name_col in df.columns and not pd.api.types.is_string_dtype(df[name_col]):
        for col in reversed(df.columns):
            if col == id_col:
                continue
            if pd.api.types.is_string_dtype(df[col]):
                name_col = col
                break

    mapping: dict[str, str] = {}
    for raw_id, raw_name in zip(df[id_col], df[name_col]):
        if pd.isna(raw_id) or pd.isna(raw_name):
            continue
        norm_id = _normalize_feature_id_for_matching(raw_id)
        if not norm_id:
            continue
        # Keep first non-empty mapping.
        if norm_id not in mapping:
            name = str(raw_name).strip()
            if name:
                mapping[norm_id] = name

    _METABOLOME_ID_TO_REALNAME = mapping
    return _METABOLOME_ID_TO_REALNAME


def _get_feature_display_name(feature_code) -> str:
    """Best-effort display name for plots: confounder alias -> maternal map -> metabolome CSV -> raw."""
    if feature_code is None:
        return ""

    raw = str(feature_code)
    code = raw.strip()

    # Confounders have explicit publication-friendly names.
    if code in confounders_maternal_nice_name_mapping:
        return confounders_maternal_nice_name_mapping[code]
    code_l = code.lower()
    if code_l in confounders_maternal_nice_name_mapping:
        return confounders_maternal_nice_name_mapping[code_l]

    # First try the existing maternal mapping config.
    mapped = config.get_maternal_feature_full_name(code)
    if mapped != code:
        return mapped

    # Fall back to metabolome CSV mapping (fuzzy ID matching).
    meta_map = _load_metabolome_id_to_realname()
    meta = meta_map.get(_normalize_feature_id_for_matching(code))
    return meta if meta else code


def _is_sex_feature(feature_name) -> bool:
    """Return True if a feature corresponds to sex (we always exclude from maternal plots)."""
    return str(feature_name).strip().lower() == "sex"


def _normalize_biomarkers_type_and_title(biomarkers_type, title):
    """
    Backward compatible helper:
    Some older call sites pass (results, feature_names, confounders, title) instead of providing biomarkers_type.
    If biomarkers_type doesn't look like a biomarker type and title is empty, treat biomarkers_type as title.
    """
    title = "" if title is None else str(title)
    if biomarkers_type is None:
        return "Both", title
    if (str(biomarkers_type) not in _KNOWN_BIOMARKER_TYPES) and title == "":
        return "Both", str(biomarkers_type)
    return str(biomarkers_type), title


def _to_series(log_odds_ratio, feature_names):
    """Convert log_odds_ratio (list/dict/Series-like) to pandas Series."""
    if isinstance(log_odds_ratio, pd.Series):
        return log_odds_ratio
    return pd.Series(log_odds_ratio, index=feature_names) if isinstance(log_odds_ratio, list) else pd.Series(log_odds_ratio)

def plot_lasso_path(
    path_results: list,
    conf_cols: list,
    biomarkers_type: str,
    significant_features: list | None = None,
    legend_kwargs: dict | None = None,
):

    # Lambdas (columns) - faster: collect once, then de-dupe/sort in NumPy (C-level).
    lam_arr = np.fromiter(
        (float(lam) for trial_path in path_results for lam in trial_path.keys()),
        dtype=float,
    )
    if lam_arr.size == 0:
        raise ValueError("plot_lasso_path found no lambdas in path_results")
    lam_arr = np.unique(lam_arr)
    lam_arr.sort()
    all_lambdas = lam_arr.tolist()
    lambda_to_col = {lam: j for j, lam in enumerate(all_lambdas)}

    # Features (rows): preserve the first trial's ordering, then append any unseen features from other trials/lambdas.
    # (first_lambda might be float while loaded JSON keys might be str; take the first available dict)
    first_trial_any_lambda = next(iter(path_results[0].values()))
    features_list = list(first_trial_any_lambda.keys())
    features_set = set(features_list)
    for trial_path in path_results:
        for _, coef_dict in trial_path.items():
            for feat in coef_dict.keys():
                if feat not in features_set:
                    features_set.add(feat)
                    features_list.append(feat)

    all_features = features_list
    feature_to_row = {f: i for i, f in enumerate(all_features)}

    # Keep confounder list consistent with available columns.
    if "const" in feature_to_row:
        conf_cols = conf_cols + ["const"]
    if "birthweight" in feature_to_row:
        conf_cols = conf_cols + ["birthweight"]
    conf_cols = list(dict.fromkeys(conf_cols))

    # Fast accumulation using numpy arrays (updates only for non-zero coefficients).
    n_features = len(all_features)
    n_lambdas = len(all_lambdas)
    freq = np.zeros((n_features, n_lambdas), dtype=float)
    coef = np.zeros((n_features, n_lambdas), dtype=float)

    for trial_path in path_results:
        for lam, d in trial_path.items():
            col = lambda_to_col.get(float(lam))
            if col is None or d is None:
                continue
            if isinstance(d, pd.Series) and d.empty:
                continue
            if isinstance(d, dict) and len(d) == 0:
                continue
            idxs = []
            vals = []
            # `d` may be a dict-like or a pandas Series (statsmodels params). Preserve old behavior:
            # count a feature if its coefficient is non-zero at this lambda for this trial.
            for feat, c in (d.items() if hasattr(d, "items") else []):
                if c != 0:
                    idxs.append(feature_to_row[feat])
                    vals.append(float(c))
            if idxs:
                idxs_arr = np.asarray(idxs, dtype=int)
                vals_arr = np.asarray(vals, dtype=float)
                freq[idxs_arr, col] += 1.0
                coef[idxs_arr, col] += vals_arr

    n_trials = float(len(path_results))
    feature_per_lambda_frequency = pd.DataFrame(freq / n_trials, columns=all_lambdas, index=all_features, dtype=float)
    feature_per_lambda_coef = pd.DataFrame(coef / n_trials, columns=all_lambdas, index=all_features, dtype=float)


    
    _plot_feature_per_lambda(
        feature_per_lambda_frequency,
        title=f"Maternal {biomarkers_type}",
        confounders=conf_cols,
        biomarkers_type=biomarkers_type,
        significant_features=significant_features,
        legend_kwargs=legend_kwargs,
    )
    _plot_feature_per_lambda(
        feature_per_lambda_coef,
        title=f"Selection Coefficient: Maternal {biomarkers_type}",
        confounders=conf_cols,
        biomarkers_type=biomarkers_type,
        significant_features=significant_features,
        legend_kwargs=legend_kwargs,
    )


def _plot_feature_per_lambda(
    feature_per_lambda_frequency,
    title,
    confounders: list,
    biomarkers_type: str,
    significant_features: list = None,
    legend_kwargs: dict | None = None,
):
    """Maternal version aligned with neonatal plotting style (ln(lambda), background/foreground rendering)."""

    # Match the general visual template used by the box plots (fonts, black spines, subtle grid, striped background).
    fig_size = (8, 6)
    fig, ax = plt.subplots(figsize=fig_size)
    x = np.log(np.asarray(feature_per_lambda_frequency.columns, dtype=float))  # natural log (ln)
    ylabel = "Coefficient" if "coefficient" in str(title).lower() else "Selection Frequency"

    # Confounders should never be plotted (neither as background nor as highlighted "significant" lines).
    conf_set = set(confounders or [])
    existing_significant = []
    if significant_features is not None:
        for f in significant_features:
            if f in feature_per_lambda_frequency.index and (f not in conf_set) and (not _is_sex_feature(f)):
                existing_significant.append(f)

    # Fast background rendering: draw many feature paths via a single LineCollection (much faster than per-feature ax.plot).
    features = np.asarray(feature_per_lambda_frequency.index, dtype=object)
    y_all = feature_per_lambda_frequency.to_numpy(dtype=float, copy=False)

    sig_set = set(existing_significant or [])
    sex_mask = np.asarray([_is_sex_feature(f) for f in features], dtype=bool)
    conf_mask = np.isin(features, list(conf_set)) if conf_set else np.zeros(features.shape[0], dtype=bool)
    sig_mask = np.isin(features, list(sig_set)) if sig_set else np.zeros(features.shape[0], dtype=bool)
    bg_mask = ~(sex_mask | conf_mask | sig_mask)

    y_bg = y_all[bg_mask]
    if y_bg.size:
        # Chunk to avoid large temporary arrays when there are many features.
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
            if biomarkers_type == 'Lipidome':
                ax.set_xlim(right=-0.8)
            elif biomarkers_type == 'Metabolome':
                ax.set_xlim(right=-0.05)

    # Box-plot-like styling (fonts, grid, spines)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")
    if biomarkers_type == 'Metabolome':
        ax.set_ylabel(ylabel, fontsize=22)
    ax.set_xlabel(r"ln($\lambda$)", fontsize=22)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5, color="#CCC")


    # Legend only if significant features were plotted
    if existing_significant:
        legend_params = {
            "bbox_to_anchor": (1.02, 1.02),
            "loc": "upper left",
            "fontsize": 19,
            "frameon": True,
            "fancybox": False,
            "shadow": False,
            "ncol": 1,
            "edgecolor": "black",
            "facecolor": "white",
            "framealpha": 0.5,
            # Shorten the colored line sample shown in the legend (units are ~font-size).
            "handlelength": 0.8,
            "handletextpad": 0.4,

        }
        if legend_kwargs:
            legend_params.update(legend_kwargs)
        legend = ax.legend(**legend_params)
        legend.get_frame().set_linewidth(1.5)

    ax.tick_params(axis="both", which="major", labelsize=20)

    plt.tight_layout()
    save_dir = f"maternal_HIV_effects/significant_biomarkers/figures/lasso_path/{biomarkers_type}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{title}.pdf", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\nLasso path saved to: {save_dir}/{title}.pdf")
    plt.show()





def delta_ROC_tables(results, baseline_results, biomarkers_type=None, title=None):
    # This function is similar to the plot_delta_ROC, except it prints a table in latex format that includes
    # The columns are mean, standard error, confidence interval, and t-test p-value for AUC over the trials
    # the rows are baseline, adjusted, and delta AUC between the adjusted and baseline AUC; calculated over each trial
    # average AUC for adusted abd baseline similar to the plot_delta_ROC function,
    # and additionally inclused the delta AUC between the adjusted and baseline AUC; calculated over each trial

    biomarkers_type, title = _normalize_biomarkers_type_and_title(biomarkers_type, title)

    # Match plot_delta_ROC logic: delta is per-trial adjusted - baseline.
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
    print(f"z_score: {z_score}")

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

    rows = [
        ("Baseline AUC", *_summary(auc_base, 0.5)),
        ("Adjusted AUC", *_summary(auc_adj, 0.5)),
        ("ΔAUC (Adj−Base)", *_summary(delta_auc, 0.0)),
    ]
    df = pd.DataFrame(
        rows,
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

    # Print a compact LaTeX table and also save it (alongside figures).
    latex = df.to_latex(float_format=lambda v: f"{v:.3f}" if pd.notna(v) else "")
    print(latex)

    save_dir = f"maternal_HIV_effects/significant_biomarkers/figures/{biomarkers_type}/tables"
    os.makedirs(save_dir, exist_ok=True)
    base_name = f"delta_ROC_table_maternal_{biomarkers_type}_{title}".strip("_")
    df.to_csv(os.path.join(save_dir, base_name + ".csv"), index=True)
    with open(os.path.join(save_dir, base_name + ".tex"), "w") as f:
        f.write(latex)
    print(f"Saved delta ROC table to: {save_dir}/{base_name}.tex (+ .csv)")
    return df




def plot_delta_ROC(results, unadjusted_results, baseline_results, biomarkers_type=None, title=None):
    biomarkers_type, title = _normalize_biomarkers_type_and_title(biomarkers_type, title)
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

    save_name = f"ROC_maternal_{biomarkers_type}_{title}".strip("_")
    _plot_ROC_curve(
        ROC_curves,
        ROC_scores,
        ROC_baseline_curves,
        ROC_baseline_scores,
        unadjusted_curves,
        unadjusted_scores,
        caption=f"ROC and PR curves {title}",
        saving_name=save_name,
        biomarkers_type=biomarkers_type,
        title=title,
    )






def plot_significant_features(results, feature_names, confounders, biomarkers_type=None, title=None, freq_threshold=0.8):
    biomarkers_type, title = _normalize_biomarkers_type_and_title(biomarkers_type, title)

    # Always drop sex from maternal plots (disregarded by request).
    feature_names = [f for f in feature_names if not _is_sex_feature(f)]
    confounders = [c for c in confounders if not _is_sex_feature(c)]

    per_feature_results = {f: {"mean_log_odds": 0, 
                                "CI95_low_log_odds": 0,
                                "CI95_high_log_odds": 0,
                                "mean_selection_frequency": 0,
                                "sign_stability_frequency": 0} for f in feature_names}
    per_feature_log_odds = {f: [] for f in feature_names}
    for trial in results:
        log_odds = _to_series(trial["log_odds_ratio"], trial["feature_names"])
        for feature in feature_names:
            if feature in log_odds.index:
                val = log_odds.loc[feature]
                per_feature_results[feature]["mean_log_odds"] += val
                per_feature_results[feature]["mean_selection_frequency"] += (val != 0)
                if val != 0:
                    per_feature_log_odds[feature].append(val)
        
    for trial in results:
        log_odds = _to_series(trial["log_odds_ratio"], trial["feature_names"])
        for feature in feature_names:
            if feature in log_odds.index:
                val = log_odds.loc[feature]
                if np.sign(per_feature_results[feature]["mean_log_odds"]) == np.sign(val):
                    per_feature_results[feature]["sign_stability_frequency"] += (val != 0)

    n_trials = len(results)
    important_features = []
    for feature in feature_names:
        freq = per_feature_results[feature]["mean_selection_frequency"]
        per_feature_results[feature]["mean_log_odds"] /= freq if freq > 0 else 1
        per_feature_results[feature]["mean_selection_frequency"] /= n_trials
        per_feature_results[feature]["sign_stability_frequency"] /= freq if freq > 0 else 1
        if per_feature_log_odds[feature]:
            ci_low, ci_high = np.percentile(per_feature_log_odds[feature], [2.5, 97.5])
        else:
            ci_low, ci_high = np.nan, np.nan
        per_feature_results[feature]["CI95_low_log_odds"] = ci_low
        per_feature_results[feature]["CI95_high_log_odds"] = ci_high
        if feature not in confounders and per_feature_results[feature]["mean_selection_frequency"] >= freq_threshold:
            important_features.append(feature)
            print(
                f'{_get_feature_display_name(feature)}, '
                f'{per_feature_results[feature]["mean_log_odds"]:.3f}, '
                f'CI=[{per_feature_results[feature]["CI95_low_log_odds"]:.3f}, {per_feature_results[feature]["CI95_high_log_odds"]:.3f}], '
                f'{per_feature_results[feature]["mean_selection_frequency"]:.3f}',
                f'{per_feature_results[feature]["sign_stability_frequency"]:.3f}'
            )


    print(f"{'-'*10} Important {biomarkers_type} {'-'*10}")
    print(important_features)
    
    _box_plot_features(
        results,
        per_feature_results,
        biomarkers_type=biomarkers_type,
        title=title,
        confounders=confounders,
        freq_threshold=freq_threshold,
    )
    return important_features


def _box_plot_features(results, 
                        per_feature_results,
                        biomarkers_type,
                        title, 
                        confounders,
                        freq_threshold):
    def _add_feature_background(ax, feats):
        # Column-wise alternating bands so each box/bar sits entirely within a single band.
        # Use white / light grey stripes (requested).
        for i, _feat in enumerate(feats, start=1):
            facecolor = "white" if i % 2 else "lightgrey"
            ax.axvspan(i - 0.5, i + 0.5, facecolor=facecolor, zorder=0)

    def _save_show(fig, path):
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
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
            else:
                kept.append((feat, vals))
        return kept, dropped, q_low, q_high

    # Drop sex no matter what (even if it passes frequency threshold).
    frequent_features = {
        f: d for f, d in per_feature_results.items()
        if (d["mean_selection_frequency"] >= freq_threshold) and (not _is_sex_feature(f))
    }
    feature_log_odds = {f: [] for f in frequent_features.keys()}

    for trial in results:
        log_odds = _to_series(trial["log_odds_ratio"], trial["feature_names"])
        for feature in frequent_features.keys():
            if feature in log_odds.index and (val := log_odds.loc[feature]) != 0:
                feature_log_odds[feature].append(val)

    if not feature_log_odds:
        print(f"No features with selection frequency >= {freq_threshold}")
        return

    sorted_features = sorted(feature_log_odds.items(), key=lambda x: np.median(x[1]), reverse=True)
    filtered_sorted_features, dropped_due_to_zero_in_iqr, q_low, q_high = _filter_by_quantile_interval(sorted_features)
    filtered_sorted_features = [(f, v) for f, v in filtered_sorted_features if f not in confounders]
    if dropped_due_to_zero_in_iqr:
        # Prefer the unified display-name logic; fall back to metabolome annotation if still unmapped.
        dropped_annotated = annotate_metabolomes_kenya(dropped_due_to_zero_in_iqr)
        dropped_display = []
        for code, ann in zip(dropped_due_to_zero_in_iqr, dropped_annotated):
            disp = _get_feature_display_name(code)
            if disp == str(code) and ann and str(ann) != str(code):
                disp = str(ann)
            dropped_display.append(disp)
        print(
            f"Dropping {len(dropped_due_to_zero_in_iqr)} feature(s) whose "
            f"{int(q_low*100)}–{int(q_high*100)}% quantile interval includes 0: "
            + ", ".join(dropped_display)
        )
    if not filtered_sorted_features:
        print(
            f"All frequently selected features were dropped because their "
            f"{int(q_low*100)}–{int(q_high*100)}% quantile interval includes 0."
        )
        return

    original_feature_names = [f for f, _ in filtered_sorted_features]
    plot_data = [vals for _, vals in filtered_sorted_features]

    annotated = annotate_metabolomes_kenya(original_feature_names)
    feature_labels = []
    for code, ann in zip(original_feature_names, annotated):
        # Prefer unified display-name logic; fall back to metabolome annotation if still unmapped.
        disp = _get_feature_display_name(code)
        if disp == str(code) and ann and str(ann) != str(code):
            disp = str(ann)
        feature_labels.append(disp)
    medians = [np.median(vals) for vals in plot_data]

    # Box plot (match neonatal template exactly)
    if biomarkers_type == 'Lipidome':
        fig_size = (9, 6)
    elif biomarkers_type == 'Metabolome':
        fig_size = (3, 6)
    else:
        fig_size = (4, 6)
    fig, ax = plt.subplots(figsize=fig_size)
    _add_feature_background(ax, original_feature_names)
    bp = ax.boxplot(
        plot_data,
        vert=True,
        patch_artist=True,
        labels=feature_labels,
        showfliers=False,
        widths=0.7,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markerfacecolor": "darkgray", "markersize": 5, "alpha": 0.5, "markeredgecolor": "none"},
    )

    for i, (orig_name, patch, med) in enumerate(zip(original_feature_names, bp["boxes"], medians)):
        if orig_name in confounders:
            # Match neonatal confounder styling exactly.
            patch.set(facecolor="lavender", edgecolor="black", linewidth=1.5, alpha=0.7)
            for j in [i * 2, i * 2 + 1]:
                bp["whiskers"][j].set_color("black")
                bp["caps"][j].set_color("black")
            bp["medians"][i].set_color("black")
        else:
            patch.set(facecolor=("salmon" if med >= 0 else "lightskyblue"), edgecolor="black", linewidth=1, alpha=0.85)

    ax.axhline(0, color="#333", linestyle="--", linewidth=1.2, alpha=0.8, zorder=0)

    title = f'Adjusted Effect Size for {biomarkers_type}' if biomarkers_type != 'Both' else 'Adjusted Effect Size for All biomarkers'
    #ax.set_title(title, fontsize=18)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5, color="#CCC")
    ax.set_axisbelow(True)
    # Force 0.5 spacing on major y ticks without manually setting tick positions.
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", labelsize=25)
    ax.set_xticklabels(feature_labels, rotation=90, ha="center", fontsize=22)

    save_dir = f"maternal_HIV_effects/significant_biomarkers/figures/{biomarkers_type}/box_plots"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"log_OR_Maternal_{biomarkers_type}_{title}".strip("_")
    _save_show(fig, f"{save_dir}/{save_name}.pdf")
    print(f"\nBox plot saved to: {save_dir}/{save_name}.pdf for freq threshold {freq_threshold}")






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
                    saving_name: str=None, caption: str=None,
                    confusion_matrix: pd.DataFrame=None,
                    important_features: list[pd.DataFrame]=None,
                    biomarkers_type: str=None,
                    title: str=None):
    set_seaborn_style()

    # Keep signature stable, but the neonatal style does not use these right now.
    _ = caption, confusion_matrix, important_features, title

    # Match neonatal ROC styling exactly (size, ticks/labels off, thin lines, subtle grid, thin border).
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
            df = pd.DataFrame({"fpr": fpr, "tpr": tpr}).drop_duplicates(subset=["fpr"]).sort_values("fpr")
            if len(df) > 1:
                ax_roc.plot(
                    df["fpr"],
                    df["tpr"],
                    color=color,
                    alpha=_curve_alpha(score, mean_score, std_score),
                    linewidth=0.6,
                    zorder=1,
                )

        tpr_mean, tpr_se = _interpolate_roc_curves(curves, fpr_grid)
        ax_roc.fill_between(
            fpr_grid,
            tpr_mean - tpr_se,
            tpr_mean + tpr_se,
            color=mean_color,
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )
        ax_roc.plot(
            fpr_grid,
            tpr_mean,
            color=mean_color,
            linewidth=0.8,
            label=f"{label}",
            zorder=3,
        )

    # Plot all model types (same labels/colors as neonatal).
    if ROC_baseline_curves and ROC_baseline_scores:
        _plot_model(
            ROC_baseline_curves,
            ROC_baseline_scores,
            "autumn",
            "orangered",
            "Clinical/Demographic Factors",
        )
    _plot_model(AUC_curves, AUC_scores, "viridis", "blue", "Integrated Multi-Omics")

    # Style plot (match neonatal: no axis labels/ticks, no legend).
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random classifier", linewidth=0.8, alpha=0.7, zorder=2)
    ax_roc.set_xticks([])
    ax_roc.set_yticks([])

    ax_roc.grid(True, alpha=0.4)
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1])
    ax_roc.tick_params(axis="both", which="major", labelsize=20)
    ax_roc.set_xlabel('FP Rate', fontsize=8)
    ax_roc.set_ylabel('TP Rate', fontsize=8)

    if biomarkers_type == "Both":
        legend = ax_roc.legend(fontsize=8, edgecolor='black', facecolor='white',
                            framealpha=1, fancybox=False, shadow=False
                            )
        legend.get_frame().set_linewidth(0.5)


    for spine in ax_roc.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("black")

    plt.tight_layout()
    if saving_name is not None:
        relative_path = f"maternal_HIV_effects/significant_biomarkers/figures/{biomarkers_type}/ROC"
        os.makedirs(relative_path, exist_ok=True)
        plt.savefig(os.path.join(relative_path, saving_name) + ".pdf", dpi=300)
    else:
        plt.show()


def _normalize_hiv_maternal(series):
    """Map maternal HIV status (numeric or string) to 'Neg' / 'Pos'."""
    lookup = {"neg": "Neg", "0": "Neg", "pos": "Pos", "1": "Pos", "2": "Pos"}
    return pd.Series(series).astype(str).str.strip().str.lower().map(lookup)


def _remove_outliers(df, group_cols, value_col, k=2.0):
    """Drop rows outside k*IQR per group."""
    g = df.groupby(group_cols)[value_col]
    q1, q3 = g.transform("quantile", 0.25), g.transform("quantile", 0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    mask = df[value_col].between(q1 - k * iqr, q3 + k * iqr) | iqr.isna()
    return df.loc[mask].copy()


def violin_plots_vs_features(features, biomarkers_type="Lipidome"):
    """Violin plot of maternal lipid (or metabolome) features split by HIV status.

    Parameters
    ----------
    features : str or list[str]
        Feature code(s) to plot (must exist in the biomarkers columns).
    biomarkers_type : {'Lipidome', 'Metabolome', 'Both'}
        Which maternal dataset to load.
    """
    biomarkers, clinical_data, y = load_maternal_data(biomarkers_type=biomarkers_type)

    if isinstance(features, str):
        features = [features]
    features = [str(f).strip() for f in (features or []) if str(f).strip()]

    df = biomarkers[features].copy()
    df["HIV"] = _normalize_hiv_maternal(y)
    df = (df.melt(id_vars=["HIV"], value_vars=features,
                  var_name="Feature", value_name="value")
            .assign(value=lambda d: pd.to_numeric(d["value"], errors="coerce"))
            .dropna(subset=["value", "HIV", "Feature"]))

    df = _remove_outliers(df, ["Feature", "HIV"], "value")
    df["value"] = (df["value"] - df["value"].mean()) / df["value"].std()

    df["HIV"] = df["HIV"].map({"Neg": "HIV−", "Pos": "HIV+"})
    df["FeatureLabel"] = df["Feature"].map(_get_feature_display_name)
    palette = {"HIV−": "#7BAFD4", "HIV+": "#D45B5B"}

    set_seaborn_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))

    sns.violinplot(
        data=df, x="FeatureLabel", y="value", hue="HIV",
        order=[_get_feature_display_name(f) for f in features],
        hue_order=["HIV−", "HIV+"],
        inner="quartile",
        inner_kws={"color": "0.25", "linewidth": 0.8},
        cut=0, linewidth=0.7, palette=palette, saturation=0.9, ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.tick_params(axis="x", labelsize=13, rotation=45)
    ax.tick_params(axis="both", length=3, width=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.35")

    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5, color="0.75")
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc="upper left", fontsize=12, frameon=True, fancybox=False,
            edgecolor="0.4", facecolor="white", framealpha=0.85,
        )

    save_dir = f"maternal_HIV_effects/significant_biomarkers/figures/{biomarkers_type}/violin_plots"
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    stem = "multi" if len(features) > 3 else "_".join(features)
    out_path = os.path.join(save_dir, f"violin_maternal_{stem}_{biomarkers_type}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\nViolin plot saved to: {out_path}")
    plt.show()

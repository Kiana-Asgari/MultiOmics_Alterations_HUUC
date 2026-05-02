"""Generate analysis figures from cached selection results.

Every command consumes the JSON files cached under

    maternal_HIV_effects/significant_biomarkers/results/
    neonatal_HIV_exposure_biomarkers/results/

and re-renders the corresponding figures. No model training is performed.

Run ``python script.py --help`` for the list of subcommands.
"""

import argparse
import sys
from itertools import product

from maternal_HIV_effects.significant_biomarkers.stability_selection import (
    maternal_stability_score_from_saved_results,
)
from maternal_HIV_effects.significant_biomarkers.lasso_path import (
    lasso_path_from_saved_results,
)
from maternal_HIV_effects.maternal_volcano import maternal_volcano_plot

from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.stability_selection import (
    stability_score_from_saved_results,
)
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.stability_path import (
    stability_path_from_saved_results,
)
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.stability_analysis.evaluation_and_plots import (
    violin_plots_vs_features,
)
from neonatal_HIV_exposure_biomarkers.significant_biomarkers.volcano_plot import (
    neonatal_volcano_plot,
)


MATERNAL_BIOMARKERS = ("Both", "Metabolome", "Lipidome")
NEONATAL_COUNTRIES = ("Both", "Kenya", "Zambia")
NEONATAL_GENDERS = (-1, 1, 2)
DEFAULT_THRESHOLD = 0.9
DEFAULT_VIOLIN_FEATURE = "ala"

NO_STABILITY_PATH = {("Zambia", -1), ("Zambia", 1), ("Zambia", 2)}


def _maternal_significant_features(biomarkers_type, threshold):
    """Re-derive the per-biomarker significant-feature list from cached results."""
    return maternal_stability_score_from_saved_results(
        biomarkers_type, ROC=False, freq_threshold=threshold
    )


def _neonatal_significant_features(country, gender, threshold):
    return stability_score_from_saved_results(
        country=country, gender=gender, ROC=False, freq_threshold=threshold
    )


def cmd_maternal_boxplot(args):
    for bt in args.biomarkers:
        _maternal_significant_features(bt, args.threshold)


def cmd_maternal_lasso_path(args):
    for bt in args.biomarkers:
        sig = _maternal_significant_features(bt, args.threshold)
        lasso_path_from_saved_results(bt, significant_features=sig)


def cmd_maternal_volcano(args):
    for bt in args.biomarkers:
        sig = _maternal_significant_features(bt, args.threshold)
        maternal_volcano_plot(bt, significant_features_list=sig)


def cmd_maternal_all(args):
    for bt in args.biomarkers:
        sig = _maternal_significant_features(bt, args.threshold)
        lasso_path_from_saved_results(bt, significant_features=sig)
        maternal_volcano_plot(bt, significant_features_list=sig)


def cmd_neonatal_boxplot(args):
    for c, g in product(args.countries, args.genders):
        _neonatal_significant_features(c, g, args.threshold)


def cmd_neonatal_stability_path(args):
    for c, g in product(args.countries, args.genders):
        if (c, g) in NO_STABILITY_PATH:
            print(f"[skip] no stability_path JSON saved for country={c}, gender={g}")
            continue
        sig = _neonatal_significant_features(c, g, args.threshold)
        stability_path_from_saved_results(country=c, gender=g, significant_features=sig)


def cmd_neonatal_volcano(args):
    for c, g in product(args.countries, args.genders):
        sig = _neonatal_significant_features(c, g, args.threshold)
        neonatal_volcano_plot(country=c, gender=g, significant_features_list=sig)


def cmd_neonatal_violin(args):
    for c in args.countries:
        violin_plots_vs_features(country=c, features=args.feature)


def cmd_neonatal_all(args):
    for c in args.countries:
        violin_plots_vs_features(country=c, features=args.feature)
        for g in args.genders:
            sig = _neonatal_significant_features(c, g, args.threshold)
            neonatal_volcano_plot(country=c, gender=g, significant_features_list=sig)
            if (c, g) in NO_STABILITY_PATH:
                print(f"[skip] no stability_path JSON saved for country={c}, gender={g}")
                continue
            stability_path_from_saved_results(country=c, gender=g, significant_features=sig)


def cmd_all(args):
    cmd_maternal_all(args)
    cmd_neonatal_all(args)


def _add_threshold_arg(parser):
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Selection-frequency threshold for 'significant' features (default: 0.9).",
    )


def _add_maternal_args(parser, threshold=True):
    parser.add_argument(
        "--biomarkers",
        nargs="+",
        default=list(MATERNAL_BIOMARKERS),
        choices=MATERNAL_BIOMARKERS,
        help="Maternal biomarker dataset(s) to render. Default: all three.",
    )
    if threshold:
        _add_threshold_arg(parser)


def _add_neonatal_args(parser, with_feature=False, threshold=True):
    parser.add_argument(
        "--country",
        dest="countries",
        nargs="+",
        default=list(NEONATAL_COUNTRIES),
        choices=NEONATAL_COUNTRIES,
        help="Neonatal cohort(s) to render. Default: Both, Kenya, Zambia.",
    )
    parser.add_argument(
        "--gender",
        dest="genders",
        nargs="+",
        type=int,
        default=list(NEONATAL_GENDERS),
        choices=NEONATAL_GENDERS,
        help="Sex stratum/strata to render: -1=both, 1=male, 2=female.",
    )
    if threshold:
        _add_threshold_arg(parser)
    if with_feature:
        parser.add_argument(
            "--feature",
            default=DEFAULT_VIOLIN_FEATURE,
            help=(
                "Marker code or group name for the violin plot. Single codes like "
                "'ala' or group names from neonatal stability-analysis (e.g. "
                "'free_carnitine', 'short_chain'). Default: 'ala'."
            ),
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="script.py",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "maternal-boxplot",
        help="Stability-selection box plot of significant maternal biomarkers.",
    )
    _add_maternal_args(p)
    p.set_defaults(func=cmd_maternal_boxplot)

    p = sub.add_parser(
        "maternal-lasso-path",
        help="Lasso coefficient/selection path for maternal biomarkers.",
    )
    _add_maternal_args(p)
    p.set_defaults(func=cmd_maternal_lasso_path)

    p = sub.add_parser(
        "maternal-volcano",
        help="Volcano plot of HIV+ vs HIV- mothers (maternal biomarkers).",
    )
    _add_maternal_args(p)
    p.set_defaults(func=cmd_maternal_volcano)

    p = sub.add_parser(
        "maternal-all",
        help="Box plot + lasso path + volcano plot for every selected biomarker set.",
    )
    _add_maternal_args(p)
    p.set_defaults(func=cmd_maternal_all)

    p = sub.add_parser(
        "neonatal-boxplot",
        help="Stability-selection box plot of significant neonatal biomarkers.",
    )
    _add_neonatal_args(p)
    p.set_defaults(func=cmd_neonatal_boxplot)

    p = sub.add_parser(
        "neonatal-stability-path",
        help="Stability-selection path of neonatal biomarkers vs lambda.",
    )
    _add_neonatal_args(p)
    p.set_defaults(func=cmd_neonatal_stability_path)

    p = sub.add_parser(
        "neonatal-volcano",
        help="Volcano plot of HEU vs HUU neonates.",
    )
    _add_neonatal_args(p)
    p.set_defaults(func=cmd_neonatal_volcano)

    p = sub.add_parser(
        "neonatal-violin",
        help="Violin plot of one feature (or feature group) vs HIV exposure x sex.",
    )
    _add_neonatal_args(p, with_feature=True)
    p.set_defaults(func=cmd_neonatal_violin)

    p = sub.add_parser(
        "neonatal-all",
        help="Box plot + stability path + volcano plot + violin plot for every (country, gender).",
    )
    _add_neonatal_args(p, with_feature=True)
    p.set_defaults(func=cmd_neonatal_all)

    p = sub.add_parser(
        "all",
        help="Run every figure: maternal-all + neonatal-all.",
    )
    _add_maternal_args(p, threshold=False)
    _add_neonatal_args(p, with_feature=True)
    p.set_defaults(func=cmd_all)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

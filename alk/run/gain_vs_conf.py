"""Script to export average gain upon interruption at confidence thresholds as a LaTeX table for multiple experiments.

Average confidence efficiency for each experiment is also provided.

usage: gain_vs_conf.py [-h] [-p FPATH] [-c CONFTHOLD [CONFTHOLD ...]] [-z Z]
                       [-d DEC] [--knni KNNI] [--rtrim [RTRIM [RTRIM ...]]]
                       [experiments [experiments ...]]

positional arguments:
  experiments           Interruption experiment result file path(s) (default:
                        None)

optional arguments:
  -h, --help            show this help message and exit
  -p FPATH, --fpath FPATH
                        Full path of the folder containing interruption
                        experiment result files. Optionally used instead of
                        the 'experiments' argument. (default: None)
  -c CONFTHOLD [CONFTHOLD ...], --confthold CONFTHOLD [CONFTHOLD ...]
                        Confidence thresholds (default: [1.0, 0.98, 0.95, 0.9,
                        0.8])
  -z Z, --z Z           z factor of the efficiency measure (default: 1.0)
  -d DEC, --dec DEC     Decimal digits to be used in gain percentage (default:
                        2)
  --knni KNNI           Zero-based index of the kNN for which the average
                        'confidence performance' is calculated. 'None' to
                        calculate for all kNNs. Normally, it makes sense
                        either for the last or all NNs. (default: None)
  --rtrim [RTRIM [RTRIM ...]]
                        Remove given strings at the end of dataset names; e.g.
                        _TRAIN, _TEST (default: None)


Examples
    # First, copy all interruption experiment output files to ~/Desktop/results
    # Export the LaTeX table for experiments that used z=-1 setting, use knni[8] for the average gain, trim _TRAIN _TEST tags from dataset names
    $ python -m alk.run.gain_vs_conf -p ~/Desktop/results -c 1. .98 .95 .92 .9 .85 .8 .75 .7 --knni 8 --z -1 --rtrim _TRAIN _TEST

 """

import argparse
import os
import re
import sys

import pandas as pd

from alk import common, helper
from alk.exp import intrpt
from alk.run import run_common


def _conf_col_label(conf, formatter, z=None):
    # lbl = "Conf={}".format(formatter(conf))
    # if conf < 1. and z:  # not(None or 0)
    #     z = helper.is_float_int(z)
    #     lbl = "{}-{}$\sigma$={}".format(lbl, z if z > 1 else "")
    lbl = "{}".format(formatter(conf) if formatter else conf)
    return lbl


def main(argv=None):
    # Configure argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiments", nargs="*", type=str,
                        help="Interruption experiment result file path(s)")
    parser.add_argument("-p", "--fpath", type=str,
                        help="Full path of the folder containing interruption experiment result files. "
                             "Optionally used instead of the 'experiments' argument.")
    parser.add_argument("-c", "--confthold", nargs="+", type=float, default=[1., .98, .95, .9, .8],
                        help="Confidence thresholds")
    parser.add_argument("-z", "--z", type=float, default=1.,
                        help="z factor of the efficiency measure")
    parser.add_argument("-d", "--dec", type=int, default=2,
                        help="Decimal digits to be used in gain percentage")
    parser.add_argument("--knni", type=int,
                        help="Zero-based index of the kNN for which the average 'confidence performance' is calculated."
                             " 'None' to calculate for all kNNs."
                             " Normally, it makes sense either for the last or all NNs.")
    parser.add_argument("--rtrim", nargs="*", type=str,
                       help="Remove given strings at the end of dataset names; e.g. _TRAIN, _TEST")
    # Parse arguments
    args = parser.parse_args(argv)
    # Required params check
    if args.experiments and args.fpath is not None:
        parser.error("'experiments' and 'fpath' arguments may not coexist!")
    if not args.experiments and args.fpath is None:
        parser.error("Either 'experiments' or 'fpath' argument should be given!")
    # Get experiment files
    if args.fpath:
        fpath = os.path.expanduser(args.fpath)
        files_ = common.listdir_non_hidden(fpath)
        experiments = [os.path.join(fpath, f) for f in files_]
    else:
        experiments = [os.path.expanduser(e) for e in args.experiments]
    conf_thold = args.confthold
    arg_z = args.z
    dec_digits = args.dec
    knn_i = args.knni
    rtrim = args.rtrim
    float_formatter = lambda x: "{0:.{1}f}".format(x, dec_digits)

    LBL_DATASET = "Dataset"
    LBL_FWIDTH = "Width"
    LBL_FSTEP = "Step"
    LBL_CONF_PERF = "Effcy"
    LBL_CONF_PERF_STD = "\u03C3"

    # Create output dataframe
    df_output = pd.DataFrame(columns=[LBL_DATASET, LBL_FWIDTH, LBL_FSTEP] +
                                     [_conf_col_label(c, float_formatter, arg_z) for c in conf_thold] +
                                     [LBL_CONF_PERF, LBL_CONF_PERF_STD])
    # Populate summary dictionary
    for exp in experiments:
        print("Exp: {}".format(exp))
        # Load interruption experiment
        exp_intrpt_output = common.load_obj(exp)  # type: intrpt.ExpIntrptOutput
        exp_z = exp_intrpt_output.settings.z
        if arg_z != exp_z:
            print("Ignored. The 'z' command line argument ({}) is different from "
                  "the experiment 'z' setting ({}).".format(helper.is_float_int(arg_z), helper.is_float_int(exp_z)))
        else:
            dataset_name = common.file_name_wo_ext(exp_intrpt_output.settings.dataset)
            print("...Dataset: {}".format(dataset_name))
            # Get the average gains in the interruption experiment
            df_avg_gains = intrpt.get_avg_gain_for_intrpt_exp(exp, conf_thold)  # Average gain is calculated for the last kNN member
            # Add the results to the output dataframe
            if rtrim:
                for tag in rtrim:
                    # Trim end tags
                    dataset_name = re.sub("{}$".format(tag), "", dataset_name)
            dataset_name = "\textit{{{0}}}".format(dataset_name)
            dict_avg_gains = {LBL_DATASET: dataset_name,
                              LBL_FWIDTH: run_common.time_window_width_str(exp_intrpt_output.settings.tw_width),
                              LBL_FSTEP: exp_intrpt_output.settings.tw_step}
            avg_gain_keys = [_conf_col_label(c, float_formatter, exp_z) for c in df_avg_gains.index.tolist()]
            avg_gain_values = df_avg_gains["gain"].values
            dict_avg_gains.update(dict(zip(avg_gain_keys, avg_gain_values)))
            # Add average efficiency and its std deviation columns too
            avg_conf_perf, avg_conf_perf_std = intrpt.get_avg_effcy_for_intrpt_exp(exp, knn_i=knn_i)
            dict_avg_gains.update({LBL_CONF_PERF: avg_conf_perf, LBL_CONF_PERF_STD: avg_conf_perf_std})
            df_output = df_output.append(dict_avg_gains, ignore_index=True)

    # Export the df to LaTeX
    if len(df_output) > 0:
        #  Create a multiindex for a sorted (and prettier) output
        df_output = df_output.set_index([LBL_DATASET, LBL_FWIDTH, LBL_FSTEP])
        df_output = df_output.sort_index()
        # df_output = df_output.round(dec_digits)
        save_fn = "gain_vs_conf_(x{})_[{}]_sd_{}_ki_{}".format(len(df_output),
                                                              "_".join([str(c) for c in conf_thold]),
                                                               helper.is_float_int(arg_z),
                                                               knn_i if knn_i is not None else "All")
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.tex".format(save_fn))
        df_output.to_latex(buf=save_fpath, float_format=float_formatter, escape=False, multirow=True, index=True)
        print("Avg Gain vs Confidence table saved as LaTeX table into '{}'.".format(save_fpath))
    else:
        print("No average gain results could be calculated.")


if __name__ == '__main__':
    main(sys.argv[1:])

"""Script to export average solution hit % upon interruptions at confidence thresholds and upon guaranteeing exact solution
as a LaTeX table for multiple classification experiments.

Of course, the average hit for interruptions with exact solution will be 100%.

usage: hit_classify.py [-h] [-p FPATH] [-c CONFTHOLD [CONFTHOLD ...]] [-z Z]
                       [-d DEC] [--knni KNNI] [--wsoln {0,1}]
                       [--rtrim [RTRIM [RTRIM ...]]]
                       [experiments [experiments ...]]

positional arguments:
  experiments           Classification experiment result file path(s)
                        (default: None)

optional arguments:
  -h, --help            show this help message and exit
  -p FPATH, --fpath FPATH
                        Full path of the folder containing classification
                        experiment result files. Optionally used instead of
                        the 'experiments' argument. (default: None)
  -c CONFTHOLD [CONFTHOLD ...], --confthold CONFTHOLD [CONFTHOLD ...]
                        Confidence thresholds (default: [1.0, 0.98, 0.95, 0.9,
                        0.8])
  -z Z, --z Z           z factor of the efficiency measure (default: -1.0)
  -d DEC, --dec DEC     Decimal digits to be used in gain percentage (default:
                        2)
  --knni KNNI           Zero-based index of the kNN for which the average
                        'confidence performance' is calculated. 'None' to
                        calculate for all kNNs. Normally, it makes sense
                        either for the last or all NNs. (default: None)
  --wsoln {0,1}         1 to display hits upon interruption w/ exact solution
                        (default: 0)
  --rtrim [RTRIM [RTRIM ...]]
                        Remove given strings at the end of dataset names; e.g.
                        _TRAIN, _TEST (default: None)

Examples
    # First, copy all classification experiment output files to ~/Desktop/results
    # Export the LaTeX table for the solution hits upon both type of interruptions (by conf and by exact solution)
    $ python -m alk.run.hit_intrpt -p ~/Desktop/results -c .98 .95 .92 .9 .85 .8 .75 .7 --z -1 --wsoln 1 --rtrim _TRAIN _TEST

 """

import argparse
import os
import sys
from typing import Union

import pandas as pd

import alk.exp.classify
from alk import common, helper
from alk.exp import intrpt, ts
from alk.run import gain_intrpt_classify, run_common


def main(argv=None):
    # Configure argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiments", nargs="*", type=str,
                        help="Classification experiment result file path(s)")
    parser.add_argument("-p", "--fpath", type=str,
                        help="Full path of the folder containing classification experiment result files. "
                             "Optionally used instead of the 'experiments' argument.")
    parser.add_argument("-c", "--confthold", nargs="+", type=float, default=[1., .98, .95, .9, .8],
                        help="Confidence thresholds")
    parser.add_argument("-z", "--z", type=float, default=-1.,
                        help="z factor of the efficiency measure")
    parser.add_argument("-d", "--dec", type=int, default=2,
                        help="Decimal digits to be used in gain percentage")
    parser.add_argument("--knni", type=int,
                        help="Zero-based index of the kNN for which the average 'confidence performance' is calculated."
                             " 'None' to calculate for all kNNs."
                             " Normally, it makes sense either for the last or all NNs.")
    parser.add_argument("--wsoln", choices=[0, 1], type=int, default=0,
                        help="1 to display hits upon interruption w/ exact solution")
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
    float_formatter_hit = lambda x: "{0:.{1}f}".format(x * 100, dec_digits) if isinstance(x, (int, float)) else x
    wsoln = args.wsoln
    wsoln_tag = "_ws_{}".format(wsoln) if wsoln else ""

    LBL_DATASET = "Dataset"
    LBL_FWIDTH = "Width"
    LBL_FSTEP = "Step"
    LBL_STOP_W_SOLN = "w\u2215Soln"

    # Create output dataframe
    df_hits_output = pd.DataFrame(columns=[LBL_DATASET, LBL_FWIDTH, LBL_FSTEP] +
                                          ([LBL_STOP_W_SOLN] if wsoln else []) +
                                          [gain_intrpt_classify.conf_col_label(c, float_formatter, arg_z) for c in conf_thold if c != 1])  # Exclude conf=1.00 for hits, makes no sense for uninterruption

    # Populate summary dictionary
    for exp in experiments:
        print("Exp: {}".format(exp))
        # Load classification experiment
        exp_output = common.load_obj(exp)  # type: Union[intrpt.ExpIntrptOutput, classify.ExpClassifierOutput]
        exp_z = exp_output.settings.z
        if arg_z != exp_z:
            print("Ignored. The 'z' command line argument ({}) is different from "
                  "the experiment 'z' setting ({}).".format(helper.is_float_int(arg_z), helper.is_float_int(exp_z)))
        else:
            dataset_name = common.file_name_wo_ext(exp_output.settings.dataset)
            print("...Dataset: {}".format(dataset_name))
            # Get the average hits in the classification experiment
            if rtrim:
                dataset_name = ts.rtrim_dataset_name(dataset_name, rtrim, latex_it=True)
            # Avg hit dict
            df_avg_hits = alk.exp.classify.get_avg_hit_for_classify_exp(exp, conf_thold, wsoln, lblwsoln=LBL_STOP_W_SOLN)
            dict_avg_hits = {LBL_DATASET: dataset_name,
                             LBL_FWIDTH: run_common.time_window_width_str(exp_output.settings.tw_width),
                             LBL_FSTEP: exp_output.settings.tw_step}
            avg_hits_keys = [gain_intrpt_classify.conf_col_label(c, float_formatter, exp_z) if isinstance(c, float) else c for c in df_avg_hits.index.tolist()]
            avg_hits_values = df_avg_hits["hit"].values
            dict_avg_hits.update(dict(zip(avg_hits_keys, avg_hits_values)))
            # Add the results to the output dataframe
            df_hits_output = df_hits_output.append(dict_avg_hits, ignore_index=True)

    # Export the df_hits to LaTeX
    if len(df_hits_output) > 0:
        #  Create a multiindex for a sorted (and prettier) output
        df_hits_output = df_hits_output.set_index([LBL_DATASET, LBL_FWIDTH, LBL_FSTEP])
        df_hits_output = df_hits_output.sort_index()
        save_fn_hit = "soln_hit_(x{})_[{}]_sd_{}_ki_{}{}".format(len(df_hits_output),
                                                                 "_".join([str(c) for c in conf_thold]),
                                                                 helper.is_float_int(arg_z),
                                                                 knn_i if knn_i is not None else "All",
                                                                 wsoln_tag)
        save_fpath_hit = os.path.join(common.APP.FOLDER.FIGURE, "{}.tex".format(save_fn_hit))
        df_hits_output.to_latex(buf=save_fpath_hit, float_format=float_formatter_hit, escape=False, multirow=True, index=True)
        print("Avg Solution Hit %s saved as LaTeX table into '{}'.".format(save_fpath_hit))
    else:
        print("No average solution hit results could be calculated.")


if __name__ == '__main__':
    main(sys.argv[1:])

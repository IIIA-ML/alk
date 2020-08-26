"""Script to export average gain for insights experiments.

usage: gain_insights.py [-h] [-p FPATH] [-d DEC] [--rtrim [RTRIM [RTRIM ...]]]
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
  -d DEC, --dec DEC     Decimal digits to be used in gain percentage (default:
                        2)
  --rtrim [RTRIM [RTRIM ...]]
                        Remove given strings at the end of dataset names; e.g.
                        _TRAIN, _TEST (default: None)

Examples:
    $ python -m alk.run.gain_insights -p ~/Desktop/results --rtrim _TRAIN _TEST

"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

from alk import common
from alk.exp import insights, ts


def time_window_width_str(w):
    """String for the time window"""
    return "Expanding" if w == 0 else w


def main(argv=None):
    # Configure argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiments", nargs="*", type=str,
                        help="Interruption experiment result file path(s)")
    parser.add_argument("-p", "--fpath", type=str,
                        help="Full path of the folder containing interruption experiment result files. "
                             "Optionally used instead of the 'experiments' argument.")
    parser.add_argument("-d", "--dec", type=int, default=2,
                        help="Decimal digits to be used in gain percentage")
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
    dec_digits = args.dec
    rtrim = args.rtrim
    float_formatter = lambda x: "{0:.{1}f}".format(x, dec_digits) if isinstance(x, (int, float)) else x
    int_formatter = lambda x: '{:,}'.format(x)

    LBL_DATASET = "Dataset"
    LBL_FWIDTH = "Width"
    LBL_FSTEP = "Step"
    LBL_UPDATES = "Updates"
    LBL_CB_SIZE = "\u007CCB\u007C"
    LBL_GAIN = "Gain"

    # Create output dataframe
    df_gains_output = pd.DataFrame(columns=[LBL_DATASET, LBL_FWIDTH, LBL_FSTEP, LBL_UPDATES, LBL_CB_SIZE, LBL_GAIN])

    # Populate summary dictionaries
    for exp in experiments:
        print("Exp: {}".format(exp))
        # Load insights experiment
        exp_insights_output = common.load_obj(exp)  # type: insights.ExpInsightsOutput
        dataset_name = common.file_name_wo_ext(exp_insights_output.settings.dataset)
        print("...Dataset: {}".format(dataset_name))
        # Add the results to the output dataframe
        if rtrim:
            dataset_name = ts.rtrim_dataset_name(dataset_name, rtrim, latex_it=True)
        # Get the average gain for the insights experiment
        avg_gain = insights.get_avg_gain_for_exp(exp)
        n_updates = np.max([u[0] for u in exp_insights_output.data.gain])
        # Avg gain dict
        dict_avg_gains = {LBL_DATASET: dataset_name,
                          LBL_FWIDTH: time_window_width_str(exp_insights_output.settings.tw_width),
                          LBL_FSTEP: exp_insights_output.settings.tw_step,
                          LBL_UPDATES: n_updates + 1,
                          LBL_CB_SIZE: exp_insights_output.settings.cb_size,
                          LBL_GAIN: avg_gain}
        # Add the results to the output dataframe
        df_gains_output = df_gains_output.append(dict_avg_gains, ignore_index=True)

    # Export the df to LaTeX
    if len(df_gains_output) > 0:
        #  Create a multiindex for a sorted (and prettier) output
        df_gains_output = df_gains_output.set_index([LBL_DATASET, LBL_FWIDTH, LBL_FSTEP])
        df_gains_output = df_gains_output.sort_index()
        # df_gains_output = df_gains_output.round(dec_digits)
        save_fn = "gain_insights_(x{})".format(len(df_gains_output))
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.tex".format(save_fn))
        df_gains_output.to_latex(buf=save_fpath, formatters={LBL_UPDATES: int_formatter, LBL_CB_SIZE: int_formatter}, float_format=float_formatter, escape=False, multirow=True, index=True)
        print("Avg Gain results saved as LaTeX table into '{}'.".format(save_fpath))
    else:
        print("No average gain results could be calculated.")


if __name__ == '__main__':
    main(sys.argv[1:])

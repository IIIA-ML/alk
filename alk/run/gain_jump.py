"""Script to export average gain for multiple 'jumping iteration' experiments as a LaTeX table.

usage: gain_jump.py [-h] [-p FPATH] [-d DEC] [--rtrim [RTRIM [RTRIM ...]]]
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
    # First, copy all jumping iteration experiment output files to ~/Desktop/results
    # Export the LaTeX table for experiments, trim _TRAIN _TEST tags from dataset names
    $ python -m alk.run.gain_jump -p ~/Desktop/results --rtrim _TRAIN _TEST

"""


import argparse
import sys
import os
import re

import pandas as pd

from alk import common
from alk.exp import jump
from alk.run import run_common


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

    LBL_DATASET = "Dataset"
    LBL_FWIDTH = "Width"
    LBL_FSTEP = "Step"

    # Create output dataframe
    df_gains_output = pd.DataFrame(columns=[LBL_DATASET, LBL_FWIDTH, LBL_FSTEP])

    # Populate summary dictionaries
    for exp in experiments:
        print("Exp: {}".format(exp))
        # Load jumping experiment output
        # result = pickle.load(open(exp, "rb"))
        exp_jump_output = common.load_obj(exp)  # type: jump.ExpJumpOutput
        # Get the dataset name
        dataset_name = common.file_name_wo_ext(exp_jump_output.settings.dataset)
        print("...Dataset: {}".format(dataset_name))
        if rtrim:
            for tag in rtrim:
                # Trim end tags
                dataset_name = re.sub("{}$".format(tag), "", dataset_name)
        dataset_name = "\textit{{{0}}}".format(dataset_name)

        # Get the average gains in experiments
        exp_jump_data = exp_jump_output.data
        exp_jump_data = exp_jump_data.loc[exp_jump_data["update"] != 0]  # Filter the initial problem
        df_avg_gains = exp_jump_data[["gain", "jumpat"]] .groupby("jumpat").mean()
        # Avg gain dict
        dict_avg_gains = {LBL_DATASET: dataset_name,
                          LBL_FWIDTH: run_common.time_window_width_str(exp_jump_output.settings.tw_width),
                          LBL_FSTEP: exp_jump_output.settings.tw_step}
        # avg_gain_keys = [str(c) if c is not None else "-" for c in df_avg_gains.index.tolist()]
        avg_gain_keys = df_avg_gains.index.tolist()
        avg_gain_values = df_avg_gains["gain"].values
        # Add the results to the output dataframe
        dict_avg_gains.update(dict(zip(avg_gain_keys, avg_gain_values)))
        df_gains_output = df_gains_output.append(dict_avg_gains, ignore_index=True)

    # Export the df to LaTeX
    if len(df_gains_output) > 0:
        #  Create a multiindex for a sorted (and prettier) output
        df_gains_output = df_gains_output.set_index([LBL_DATASET, LBL_FWIDTH, LBL_FSTEP])
        df_gains_output = df_gains_output.sort_index()
        # Sort columns
        try:
            cols = sorted([int(c) for c in df_gains_output.columns.tolist()], reverse=True)
            if cols[-1] == 0:
                cols = [cols.pop(len(cols) - 1)] + cols
            df_gains_output = df_gains_output[cols]
        except TypeError:  # if int, None or str coexist in columns
            pass
        # df_gains_output = df_gains_output.round(dec_digits)
        jump_at_tag = "jump_[{}]".format("_".join([str(j) for j in df_gains_output.columns]))
        save_fn = "gain_jump_(x{})_{}".format(len(df_gains_output), jump_at_tag)
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.tex".format(save_fn))
        df_gains_output.to_latex(buf=save_fpath, float_format=float_formatter, escape=False, multirow=True, index=True)
        print("Avg Gain for Normal vs Jumping Candidacy Assessment table saved as LaTeX table into '{}'.".format(save_fpath))
    else:
        print("No average gain results could be calculated.")


if __name__ == '__main__':
    main(sys.argv[1:])

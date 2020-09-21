"""Generates a LaTeX table for the similarity distribution between the cases of given case bases

usage: sim_distr.py [-h] [-p FPATH] [-w WIDTH [WIDTH ...]]
                    [-s STEP [STEP ...]] [-b BINS] [-t TESTSIZE] [-u UPDATE]
                    [--dec DEC]
                    [datasets [datasets ...]]

positional arguments:
  datasets              Dataset file path(s) (default: None)

optional arguments:
  -h, --help            show this help message and exit
  -p FPATH, --fpath FPATH
                        Full path of the folder containing dataset files.
                        Optionally used instead of the 'datasets' argument.
                        (default: None)
  -w WIDTH [WIDTH ...], --width WIDTH [WIDTH ...]
                        Time window width(s) (default: None)
  -s STEP [STEP ...], --step STEP [STEP ...]
                        Time window step(s) (default: None)
  -b BINS, --bins BINS  Number of bins for the histogram (default: 10)
  -t TESTSIZE, --testsize TESTSIZE
                        Test size for the dataset. float between (0.0, 1.0) as
                        a proportion; int > 0 for absolute number of test
                        sequences (default: 1)
  -u UPDATE, --update UPDATE
                        Particular update index to use as the query for each
                        sequence (default: None)
  --dec DEC             Decimal digits to be used in percentage values
                        (default: 2)

Examples:
    # Use `PowerCons_TRAIN.arff` dataset
    # Generate four case bases with combinations of time window `width`=[Expanding, 40] and `step`=[1, 10] settings
    # Use 1% of the dataset as test sequences to generate queries, and the rest as the case base
    # Distribute the similarity value densities as percentage in 10 `bins`
    $ python -m alk.run.sim_distr ~/Dev/alk/datasets/PowerCons_TRAIN.arff --width 0 40 --step 1 10 --bins 10 --testsize 0.01
    # Use all datasets
    # Use only 1 sequence from the dataset to generate queries, and the rest of the sequences as the case base
    $ python -m alk.run.sim_distr -p ~/Dev/alk/datasets/ --width 0 40 --step 1 10 --bins 10 --testsize 1
    # Check the similarity distribution only for the 3rd update
    $ python -m alk.run.sim_distr -p ~/Dev/alk/datasets/ --width 0 40 --step 1 10 --bins 10 --testsize 0.01 --update=3
    # Check the similarity distribution only for the last update (i.e. longest case in a sequence)
    $ python -m alk.run.sim_distr -p ~/Dev/alk/datasets/ --width 0 40 --step 1 10 --bins 10 --testsize 0.01 --update=-1

"""

import argparse
import datetime
import logging
import math
import os
import sys
import time
from typing import Callable, Union

import numpy as np
import pandas as pd

from alk import alk, cbr, common
from alk.exp import exp_common, ts
from alk.run import run_common


logger = logging.getLogger("ALK")


def sim_hist(cb, similarity, bins=10, test_size=10, upd_select=None):
    """Calculates similarities between a proportion of cases over the rest of the case base

    Args:
        cb (cbr.TCaseBase): Temporal case base
        similarity (Callable): Similarity metric
        bins (int): Number of bins for the histogram
        test_size(Union[float, int]): test size to split the `cb` into test sequences and CB_train.
            If float, should be between 0.0 and 1.0 and represent the proportion of the `cb` to generate test sequences;
            if int, represents the absolute number of test sequences
            upd_select (int): Particular update index to use as the query for each sequence"

    Returns:
        numpy.ndarray: 1D array of histogram of similarity distribution given as percentage values

    """

    logger.info(".. Calculating similarity distribution for a proportion of cases over the case base")
    CB_train, CB_test = exp_common.split_cb(cb, test_size)
    CB_train = cbr.TCaseBase(cb=CB_train)
    linear_search = alk.LinearSearch(cb=CB_train, similarity=similarity)
    total_queries = sum([seq.n_profiles() for seq in CB_test]) if upd_select is None else len(CB_test)
    logger.info(".... Linear search with {} queries X {} cases".format(total_queries, CB_train.size()))
    distr_arr = np.full((bins,), 0, dtype=int)
    bin_width = 1. / bins
    total_calcs = 0
    try:
        n_features_min, n_features_max = len(CB_test[0].profile(0)), len(CB_test[0].profile(0))
    except IndexError:
        logger.error("No queries to test. Check your case base and test size.")
        sys.exit(1)
    sim_min, sim_max, sim_mean = 1., 0., 0.
    sim_cntr = 0
    for sequence in CB_test:
        for upd_id in range(sequence.n_profiles()) if upd_select is None else [upd_select]:
            query = sequence.profile(upd_id)  # Get query for the sequence update
            n_features = len(query)
            if n_features > n_features_max:
                n_features_max = n_features
            elif n_features < n_features_min:
                n_features_min = n_features
            stage, calcs = linear_search.search(query)  # Search
            for assess in stage.nn:
                sim_cntr += 1
                bin_idx = 0 if assess.sim == 0 else math.ceil(assess.sim / bin_width) - 1
                distr_arr[bin_idx] += 1  # Update sim occurrence array
                if assess.sim < sim_min:
                    sim_min = assess.sim
                if assess.sim > sim_max:
                    sim_max = assess.sim
                sim_mean = sim_mean + (assess.sim - sim_mean) / sim_cntr  # Incrementally update mean
            total_calcs += calcs
    logger.info(".... Number of features min: {}, max: {}".format(n_features_min, n_features_max))
    logger.info(".... Total computations: {}".format(total_calcs))
    logger.info(".... Similarity min: {}, max: {}, avg: {}".format(sim_min, sim_max, sim_mean))
    return distr_arr / distr_arr.sum()


def _parse_args(argv):
    """Helper function that creates a parser and parses the arguments"""
    # Configure argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("datasets", nargs="*", type=str,
                        help="Dataset file path(s)")
    parser.add_argument("-p", "--fpath", type=str,
                        help="Full path of the folder containing dataset files. "
                             "Optionally used instead of the 'datasets' argument.")
    parser.add_argument("-w", "--width", nargs="+", type=int,
                        help="Time window width(s)")
    parser.add_argument("-s", "--step", nargs="+", type=int,
                        help="Time window step(s)")
    parser.add_argument("-b", "--bins", type=int, default=10,
                        help="Number of bins for the histogram")
    parser.add_argument("-t", "--testsize", type=float, default=1, action="store",
                        help="Test size for the dataset. float between (0.0, 1.0) as a proportion; "
                             "int > 0 for absolute number of test sequences")
    parser.add_argument("-u", "--update", type=int, default=None,
                        help="Particular update index to use as the query for each sequence")
    parser.add_argument("--dec", type=int, default=2,
                        help="Decimal digits to be used in percentage values")
    # Parse arguments
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    start_time = time.time()
    # add and parse arguments
    args = _parse_args(argv)

    LBL_DATASET = "Dataset"
    LBL_FWIDTH = "Width"
    LBL_FSTEP = "Step"

    bins = args.bins
    bin_width = 1. / bins
    decimals_bin_width = len(str(bin_width).split('.')[1])
    if decimals_bin_width > 2:
        decimals_bin_width = 2
    cols_bin = ["{0:.{1}f}".format(i, decimals_bin_width) for i in list(np.arange(0, 1. + bin_width, bin_width)[1:])]

    # Create output dataframe
    df_output = pd.DataFrame(columns=[LBL_DATASET, LBL_FWIDTH, LBL_FSTEP] + cols_bin)

    # Get datasets
    if args.fpath:
        fpath = os.path.expanduser(args.fpath)
        files_ = common.listdir_non_hidden(fpath)
        datasets = [os.path.join(fpath, f) for f in files_]
    else:
        datasets = [os.path.expanduser(e) for e in args.datasets]
    test_size = args.testsize if 0. < args.testsize < 1. else int(args.testsize)
    update = args.update
    dec_digits = args.dec
    float_formatter = lambda x: "{0:.{1}f}".format(x, dec_digits) if isinstance(x, (int, float)) else x

    # Set logger
    save_fn = "similarity_distr_(x{n})_w_{w}_s_{s}_t_{t}{u}".format(n=len(datasets), w=str(args.width),
                                                                    s=str(args.step), t=str(test_size),
                                                                    u="_u_{}".format(
                                                                        update) if update is not None else "")
    save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.tex".format(save_fn))
    log_file = common.gen_log_file(save_fpath)
    logger = common.initialize_logger(console_level="INFO", output_dir=common.APP.FOLDER.LOG,
                                      log_file=log_file, file_level="INFO")
    logger.info("Case base similarity distribution script launched with args: {}".format(str(vars(args))))

    # Start computing distributions for CBs
    for dataset in datasets:
        dataset_name = os.path.expanduser(dataset)
        dataset_name = os.path.splitext(os.path.basename(dataset_name))[0]
        logger.info("Dataset: {}".format(dataset_name))
        # get the similarity for the dataset
        similarity = ts.euclidean_similarity_ts_dataset(dataset)
        for tw_width in args.width:
            logger.info(".. TW width: {}".format(tw_width))
            for tw_step in args.step:
                logger.info(".. TW step: {}".format(tw_step))
                # read the dataset -> cb
                cb = ts.gen_cb(dataset=dataset, tw_width=tw_width, tw_step=tw_step)
                distr_hist = sim_hist(cb, similarity, bins, test_size, update)
                # logger.info(".... Distribution:\n{}".format(
                #     pd.DataFrame([distr_hist * 100], columns=cols_bin).to_string(index=False,
                #                                                                  float_format=float_formatter)))
                dict_distr = {LBL_DATASET: dataset_name,
                              LBL_FWIDTH: run_common.time_window_width_str(tw_width),
                              LBL_FSTEP: tw_step}
                dict_distr.update(dict(zip(cols_bin, distr_hist * 100.)))
                # Add the distribution to the output dataframe
                df_output = df_output.append(dict_distr, ignore_index=True)

    # Export the df to LaTeX
    if len(df_output) > 0:
        #  Create a multiindex for a sorted (and prettier) output
        df_output = df_output.set_index([LBL_DATASET, LBL_FWIDTH, LBL_FSTEP])
        df_output = df_output.sort_index()
        df_output.to_latex(buf=save_fpath, float_format=float_formatter, escape=True, multirow=True, index=True)
        logger.info("Similarity distribution saved as LaTeX table into '{}'.".format(save_fpath))
    else:
        logger.info("No similarity distribution data could be calculated.")
    logger.info("Script finished in {}.".format(datetime.timedelta(seconds=(time.time() - start_time))))


if __name__ == '__main__':
    main(sys.argv[1:])

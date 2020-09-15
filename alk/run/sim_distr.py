"""Generates a LaTeX table for the similarity distribution between the cases of given case bases

usage: sim_distr.py [-h] [-p FPATH] [-w WIDTH [WIDTH ...]]
                    [-s STEP [STEP ...]] [-b BINS] [-t TESTSIZE] [--dec DEC]
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
                        sequences; 0 for pairwise similarity calculation
                        between all cases (default: 1)
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
    # Use `PowerCons_TRAIN.arff` dataset
    # Generate one case base with time window `width`=40 and `step`=10 settings
    # Calculate _pairwise_ similarities (Beware of long execution times for large case bases!)
    $ python -m alk.run.sim_distr ~/Dev/alk/datasets/PowerCons_TRAIN.arff --width 40 --step 10 --bins 10 --testsize 0

"""

import argparse
import datetime
import itertools
import logging
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


def pairwise_sim(cb, similarity):
    """Calculates pairwise similarities between all cases

    Args:
        cb (cbr.TCaseBase): Temporal case base
        similarity (Callable): Similarity metric

    Returns:
        numpy.ndarray: Condensed similarity matrix as a 1D array (i.e. only the upper-right triangle of the matrix)

    """
    logger.info(".. Calculating pairwise similarity distribution between all cases")
    queries = []
    computed_arr = []
    for sequence in cb.sequences():
        for upd_id in range(sequence.n_profiles()):
            queries.append(cb.get_case_query(cbr.CaseId(sequence.seq_id, upd_id)))
    logger.info(".... Total cases: {}".format(len(queries)))
    queries_len = [len(q) for q in queries]
    logger.info(".... Number of features min: {}, max: {}".format(min(queries_len), max(queries_len)))
    for i, j in itertools.combinations(range(cb.size()), 2):
        # print(".... Computing pairwise {} for the case: {}".format(metric_tag_plural, i), end="\r")
        computed_arr.append(similarity(queries[i], queries[j]))
    logger.info(".... Total computations: {}".format(len(computed_arr)))
    computed_arr = np.array(computed_arr)
    logger.info(".... Similarity min: {}, max: {}, avg: {}".format(computed_arr.min(), computed_arr.max(),
                                                                   computed_arr.mean()))
    return computed_arr


def proportion_sim(cb, similarity, test_size=10):
    """Calculates similarities between a proportion of cases over the rest of the case base

    Args:
        cb (cbr.TCaseBase): Temporal case base
        similarity (Callable): Similarity metric
        test_size(Union[float, int]): test size to split the `cb` into test sequences and CB_train.
            If float, should be between 0.0 and 1.0 and represent the proportion of the `cb` to generate test sequences;
            if int, represents the absolute number of test sequences.

    Returns:
        numpy.ndarray: 1D array of calculated similarities

    """

    logger.info(".. Calculating similarity distribution for a proportion of cases over the case base")
    CB_train, CB_test = exp_common.split_cb(cb, test_size)
    CB_train = cbr.TCaseBase(cb=CB_train)
    linear_search = alk.LinearSearch(cb=CB_train, similarity=similarity)
    logger.info(".... Linear search with {} queries X {} cases".format(sum([seq.n_profiles() for seq in CB_test]),
                                                                CB_train.size()))
    computed_arr = []
    total_calcs = 0
    try:
        n_features_min, n_features_max = len(CB_test[0].profile(0)), len(CB_test[0].profile(0))
    except IndexError:
        logger.error("No queries to test. Check your case base and test size.")
        sys.exit(1)
    for sequence in CB_test:
        for upd_id in range(sequence.n_profiles()):
            query = sequence.profile(upd_id)
            n_features = len(query)
            if n_features > n_features_max:
                n_features_max = n_features
            elif n_features < n_features_min:
                n_features_min = n_features
            stage, calcs = linear_search.search(query)
            for assess in stage.nn:
                computed_arr.append(assess.sim)
            total_calcs += calcs
    logger.info(".... Number of features min: {}, max: {}".format(n_features_min, n_features_max))
    logger.info(".... Total computations: {}".format(total_calcs))
    computed_arr = np.array(computed_arr)
    logger.info(".... Similarity min: {}, max: {}, avg: {}".format(computed_arr.min(), computed_arr.max(),
                                                                   computed_arr.mean()))
    return computed_arr


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
                             "int > 0 for absolute number of test sequences; "
                             "0 for pairwise similarity calculation between all cases")
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

    bin_width = 1. / args.bins
    bins = np.arange(0, 1. + bin_width, bin_width)
    decimals_bin_width = len(str(bin_width).split('.')[1])
    if decimals_bin_width > 2:
        decimals_bin_width = 2
    cols_bin = ["{0:.{1}f}".format(i, decimals_bin_width) for i in list(bins[1:])]

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
    dec_digits = args.dec
    float_formatter = lambda x: "{0:.{1}f}".format(x, dec_digits) if isinstance(x, (int, float)) else x

    # Set logger
    save_fn = "similarity_distr_(x{n})_w_{w}_s_{s}_t_{t}".format(n=len(datasets), w=str(args.width), s=str(args.step),
                                                                 t=str(test_size))
    save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.tex".format(save_fn))
    log_file = common.gen_log_file(save_fpath)
    logger = common.initialize_logger(console_level="INFO", output_dir=common.APP.FOLDER.LOG, log_file=log_file, file_level="INFO")
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
                if test_size == 0:
                    distr_arr = pairwise_sim(cb, similarity)
                else:
                    distr_arr = proportion_sim(cb, similarity, test_size)
                # Calculate the similarity distribution % over histogram
                distr_arr_hist, _ = np.histogram(distr_arr, weights=np.ones(len(distr_arr)) / len(distr_arr),
                                                 bins=bins, density=False)
                dict_distr = {LBL_DATASET: dataset_name,
                              LBL_FWIDTH: run_common.time_window_width_str(tw_width),
                              LBL_FSTEP: tw_step}
                dict_distr.update(dict(zip(cols_bin, distr_arr_hist * 100.)))
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

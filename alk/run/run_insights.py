"""Script to launch experiment(s) to collect gain and knn insights data.

usage: run_insights.py [-h] [-k K] [-t TESTSIZE] [-w WIDTH] [-s STEP]
                       [-o OUTFILE] [-l LOGFILE] [--runs RUNS] [--iter {td,j}]
                       [--kwargsiter [KWARGSITER [KWARGSITER ...]]]
                       [--logc {DEBUG,INFO}] [--logf {DEBUG,INFO}]
                       dataset

positional arguments:
  dataset               .arff file for time series dataset

optional arguments:
  -h, --help            show this help message and exit
  -k K, --k K           k of kNN (default: None)
  -t TESTSIZE, --testsize TESTSIZE
                        test size for the dataset (default: 0.01)
  -w WIDTH, --width WIDTH
                        frame width for cases as moving frames (default: 0)
  -s STEP, --step STEP  Number of steps (in terms of data points in TS) taken
                        by the window at each update (default: 1)
  -o OUTFILE, --outfile OUTFILE
                        File path for experiment results (default: None)
  -l LOGFILE, --logfile LOGFILE
                        Log file path (default: None)
  --runs RUNS           Number of experiment runs (default: 1)
  --iter {td,j}         Rank iteration -> td: Top-down, j: Jumping (default:
                        td)
  --kwargsiter [KWARGSITER [KWARGSITER ...]]
                        keyword args entered in attribute=value format to set
                        iterator class' class attributes (default: None)
  --logc {DEBUG,INFO}   console logger level (default: INFO)
  --logf {DEBUG,INFO}   file logger level (default: INFO)

Examples:
    # Time window width=40, Time window step=10, test size= 10% of the dataset, log level-> console: INFO, file: DEBUG
    $ python -m alk.run.run_insights "~/Dev/alk/datasets/SwedishLeaf_TEST.arff" -k 9 -t 0.1 --width 40 --step 10 --logc INFO --logf DEBUG

"""

import argparse
import datetime
import logging
import os
import sys
import time

from alk import alk, common, helper
from alk.exp import ts, insights
from alk.run import run_common


logger = logging.getLogger("ALK")


def _create_exp_insights_engine(dataset, tw_width, tw_step, k, test_size, similarity=None,
                                cls_rank_iterator=alk.TopDownIterator, cls_rank_iterator_kwargs={}, n_exp=1, gen_profile=None):
    """ Creates an insights experiment engine.

    Returns:
        ExpInsightsEngine:

    """
    # read the dataset -> cb
    cb = ts.gen_cb(dataset=dataset, tw_width=tw_width, tw_step=tw_step, gen_profile=gen_profile)
    # create an experiment engine
    max_val, min_val = ts.get_max_min(dataset)
    if similarity is None:
        similarity = lambda p1, p2: ts.euclidean_similarity_ts(p1, p2, max_=max_val, min_=min_val)
    engine = insights.ExpInsightsEngine(cb=cb, k=k, similarity=similarity,
                                        cls_rank_iterator=cls_rank_iterator, cls_rank_iterator_kwargs=cls_rank_iterator_kwargs,
                                        test_size=test_size, n_exp=n_exp)
    return engine


def _parse_args(argv):
    """Helper function that creates a parser and parses the arguments"""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset",
                        help=".arff file for time series dataset")
    parser.add_argument("-k", "--k", type=int,
                        help="k of kNN")
    parser.add_argument("-t", "--testsize", type=float, default=0.01, action="store",
                        help="test size for the dataset")
    parser.add_argument("-w", "--width", type=int, default=0, action="store",
                        help="frame width for cases as moving frames")
    parser.add_argument("-s", "--step", type=int, default=1,
                        help="Number of steps (in terms of data points in TS) taken by the window at each update")
    parser.add_argument("-o", "--outfile",
                        help="File path for experiment results")
    parser.add_argument("-l", "--logfile", type=str,
                        help="Log file path")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of experiment runs")
    parser.add_argument("--iter", choices=run_common.RANK_ITER_ARG_CHOICES, default=run_common.RANK_ITER_ARG_CHOICES[0],
                        help="Rank iteration -> " + run_common.RANK_ITER_ARG_HELP)
    parser.add_argument("--kwargsiter", nargs="*", type=str,
                        help="keyword args entered in attribute=value format to set iterator class' class attributes")
    parser.add_argument("--logc", choices=["DEBUG", "INFO"], default="INFO",
                        help="console logger level")
    parser.add_argument("--logf", choices=["DEBUG", "INFO"], default="INFO",
                        help="file logger level")
    # Parse arguments
    args = parser.parse_args(argv)

    # Eval iterator class attributes
    kwargs_iter = {}
    if args.kwargsiter:
        for kwa in args.kwargsiter:
            # Try to convert a string to its real type
            k = kwa.split("=")[0]
            v = kwa.split("=")[1]
            kwargs_iter[k] = helper.eval_str(v)
    args.kwargsiter = kwargs_iter

    return args


def main(argv=None):
    start_time = time.time()
    # add and parse arguments
    args = _parse_args(argv)
    dataset = os.path.expanduser(args.dataset)
    cls_rank_iterator = run_common.RANK_ITER_OPTIONS[args.iter]["cls"]
    cls_rank_iterator_kwargs = args.kwargsiter
    # Generate file names
    out_file = args.outfile if args.outfile else insights.gen_insights_ouput_f_path(dataset, args.width, args.step, args.k, args.testsize, cls_rank_iterator)
    log_file = args.logfile if args.logfile else common.gen_log_file(out_file)
    # Set logger
    logger = common.initialize_logger(console_level=args.logc, output_dir=common.APP.FOLDER.LOG, log_file=log_file, file_level=args.logf)
    logger.info("Insights experiment script launched with args: {}".format(str(vars(args))))
    # create the experiment engine
    engine = _create_exp_insights_engine(dataset, args.width, args.step, args.k, args.testsize, n_exp=args.runs,
                                         cls_rank_iterator=cls_rank_iterator, cls_rank_iterator_kwargs=cls_rank_iterator_kwargs)
    # engine.run -> processed data
    processed_insights = engine.run()
    # create a result obj
    output = insights.ExpInsightsOutput(settings=insights.ExpInsightsSettings(dataset, args.width, args.step, args.k, args.testsize,
                                                                              engine.cb.size(), cls_rank_iterator, cls_rank_iterator_kwargs),
                                        data=processed_insights)
    logger.info("Insights experiment finished in {}.".format(datetime.timedelta(seconds=(time.time() - start_time))))
    # save the output
    output.save(out_file=out_file)


if __name__ == '__main__':
    main(sys.argv[1:])

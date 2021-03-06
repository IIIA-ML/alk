"""Script to launch interruption experiment to collect gain with confidence and efficiency of confidence data.

usage: run_intrpt.py [-h] [-t TESTSIZE] [-o OUTFILE] [-l LOGFILE]
                     [-c CONFTHOLD [CONFTHOLD ...]] [-z Z]
                     [--logc {DEBUG,INFO}] [--logf {DEBUG,INFO}]
                     dataset pdpfile

positional arguments:
  dataset               .arff file for time series dataset
  pdpfile               PDP file path

optional arguments:
  -h, --help            show this help message and exit
  -t TESTSIZE, --testsize TESTSIZE
                        test size for the dataset (default: 0.01)
  -o OUTFILE, --outfile OUTFILE
                        File path for experiment results (default: None)
  -l LOGFILE, --logfile LOGFILE
                        Log file path (default: None)
  -c CONFTHOLD [CONFTHOLD ...], --confthold CONFTHOLD [CONFTHOLD ...]
                        Confidence thresholds (default: None)
  -z Z, --z Z           z factor in the confidence efficiency (default: -1.0)
  --logc {DEBUG,INFO}   console logger level (default: INFO)
  --logf {DEBUG,INFO}   file logger level (default: INFO)

Examples:
    # Use SwedishLeaf_TRAIN.arff, use 1% of the dataset for test sequences
    # Use PDP generated for the ~_TEST Cb of the same dataset
    # Interrupt at confidence thresholds .98 .95 .92 .9 .85 .8 .75 .7 (in reverse order)
    # Set z factor in the efficiency measure to -1 for the standard deviation
    $ python -m alk.run.run_intrpt ~/Dev/alk/datasets/SwedishLeaf_TRAIN.arff ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1__cs_0.0025_qs_0.05.pk -t 0.01 -c .98 .95 .92 .9 .85 .8 .75 .7 -z -1 --logc DEBUG --logf DEBUG

"""

import argparse
import datetime
import logging
import os
import sys
import time

from alk import common, rank
from alk.exp import ts, pdp, intrpt


logger = logging.getLogger("ALK")


def _create_exp_intrpt_engine(dataset, pdp_file, tw_width, tw_step, k, conf_tholds, z, test_size, similarity=None,
                              cls_rank_iterator=rank.TopDownIterator, cls_rank_iterator_kwargs={}, gen_profile=None):
    """ Creates an interruption experiment engine.

    Returns:
        intrpt.ExpIntrptEngine:

    """
    # read the dataset -> cb
    cb = ts.gen_cb(dataset=dataset, tw_width=tw_width, tw_step=tw_step, gen_profile=gen_profile)
    # create an experiment engine
    if similarity is None:
        similarity = ts.euclidean_similarity_ts_dataset(dataset)
    engine = intrpt.ExpIntrptEngine(pdp_file=pdp_file, cb=cb, k=k, similarity=similarity, conf_tholds=conf_tholds,
                                    cls_rank_iterator=cls_rank_iterator, cls_rank_iterator_kwargs=cls_rank_iterator_kwargs,
                                    z=z, test_size=test_size)
    return engine


def _parse_args(argv):
    """Helper function that creates a parser and parses the arguments"""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset",
                        help=".arff file for time series dataset")
    parser.add_argument("pdpfile", type=str,
                        help="PDP file path")
    parser.add_argument("-t", "--testsize", type=float, default=0.01, action="store",
                        help="test size for the dataset")
    parser.add_argument("-o", "--outfile",
                        help="File path for experiment results")
    parser.add_argument("-l", "--logfile", type=str,
                        help="Log file path")
    parser.add_argument("-c", "--confthold", nargs='+', type=float, default=None,
                        help="Confidence thresholds")
    parser.add_argument("-z", "--z", type=float, default=-1.,
                        help="z factor in the confidence efficiency")
    parser.add_argument("--logc", choices=["DEBUG", "INFO"], default="INFO",
                        help="console logger level")
    parser.add_argument("--logf", choices=["DEBUG", "INFO"], default="INFO",
                        help="file logger level")
    # Parse arguments
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    start_time = time.time()
    # add and parse arguments
    args = _parse_args(argv)
    dataset = os.path.expanduser(args.dataset)
    # Read k, tw_width, tw_step and cls_rank_iterator of the insights experiment to use the same settings
    exp_ins_settings = pdp.get_exp_ins_settings(args.pdpfile)
    k = exp_ins_settings.k
    tw_width = exp_ins_settings.tw_width
    tw_step = exp_ins_settings.tw_step
    cls_rank_iterator = exp_ins_settings.cls_rank_iterator
    cls_rank_iterator_kwargs = exp_ins_settings.cls_rank_iterator_kwargs
    # Generate file names
    out_file = args.outfile if args.outfile else intrpt.gen_intrpt_output_f_path(dataset, args.pdpfile, tw_width, tw_step, k, args.confthold, args.z, args.testsize, cls_rank_iterator)
    log_file = args.logfile if args.logfile else common.gen_log_file(out_file)
    # Set logger
    logger = common.initialize_logger(console_level=args.logc, output_dir=common.APP.FOLDER.LOG, log_file=log_file, file_level=args.logf)
    logger.info("Interruption experiment script launched with args: {}".format(str(vars(args))))
    # Create an interruption experiment engine
    engine = _create_exp_intrpt_engine(dataset, args.pdpfile, tw_width, tw_step, k, args.confthold, args.z, args.testsize,
                                       cls_rank_iterator=cls_rank_iterator, cls_rank_iterator_kwargs=cls_rank_iterator_kwargs)
    # engine.run -> collected data
    processed_data = engine.run()
    # create a result obj
    output = intrpt.ExpIntrptOutput(settings=intrpt.ExpIntrptSettings(dataset, args.pdpfile, tw_width, tw_step, k, args.confthold, args.z, args.testsize,
                                                                      cls_rank_iterator=cls_rank_iterator, cls_rank_iterator_kwargs=cls_rank_iterator_kwargs),
                                    data=processed_data)
    logger.info("Interruption experiment finished in {}.".format(datetime.timedelta(seconds=(time.time() - start_time))))
    # save the output
    output.save(out_file=out_file)


if __name__ == "__main__":
    main(sys.argv[1:])

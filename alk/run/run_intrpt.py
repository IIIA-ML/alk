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
    $ python -m alk.run.run_intrpt ~/Dev/alk/datasets/SwedishLeaf_TRAIN.arff ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1__cs_0.0025_qs_0.05.pk -t 0.01 -c .98 .95 .92 .9 .85 .8 .75 .7 -z -1 --logc DEBUG --logf DEBUG

"""

import argparse
import datetime
import logging
import os
import sys
import time

from alk import alk, common
from alk.exp import ts, pdp, intrpt


logger = logging.getLogger("ALK")


def _create_exp_intrpt_engine(dataset, pdp_file, tw_width, tw_step, k, conf_tholds, z, test_size, similarity=None,
                              cls_rank_iterator=alk.TopDownIterator, gen_profile=None):
    """ Creates an interruption experiment engine.

    Returns:
        intrpt.ExpIntrptEngine:

    """
    # read the dataset -> cb
    cb = ts.gen_cb(dataset=dataset, tw_width=tw_width, tw_step=tw_step, gen_profile=gen_profile)
    # create an experiment engine
    max_val, min_val = ts.get_max_min(dataset)
    if similarity is None:
        similarity = lambda p1, p2: ts.euclidean_similarity_ts(p1, p2, max_=max_val, min_=min_val)
    engine = intrpt.ExpIntrptEngine(pdp_file=pdp_file, cb=cb, k=k, similarity=similarity, cls_rank_iterator=cls_rank_iterator,
                                    conf_tholds=conf_tholds, z=z, test_size=test_size)
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
    cls_rank_iterator.set_cls_attr(**exp_ins_settings.cls_rank_iterator_attrs)   # Set class attributes. We do it like this because class attrs are not pickled.
    # Generate file names
    out_file = args.outfile if args.outfile else intrpt.gen_intrpt_output_f_path(dataset, args.pdpfile, tw_width, tw_step, k, args.confthold, args.z, args.testsize, cls_rank_iterator)
    log_file = args.logfile if args.logfile else common.gen_log_file(out_file)
    # Set logger
    logger = common.initialize_logger(console_level=args.logc, output_dir=common.APP.FOLDER.LOG, log_file=log_file, file_level=args.logf)
    logger.info("Interruption experiment script launched with args: {}".format(str(vars(args))))
    # Create an interruption experiment engine
    engine = _create_exp_intrpt_engine(dataset, args.pdpfile, tw_width, tw_step, k, args.confthold, args.z, args.testsize, cls_rank_iterator=cls_rank_iterator)
    # engine.run -> collected data
    processed_data = engine.run()
    # create a result obj
    output = intrpt.ExpIntrptOutput(settings=intrpt.ExpIntrptSettings(dataset, args.pdpfile, tw_width, tw_step, k, args.confthold,
                                                                      args.z, args.testsize, cls_rank_iterator=cls_rank_iterator),
                                    data=processed_data)
    logger.info("Interruption experiment finished in {}.".format(datetime.timedelta(seconds=(time.time() - start_time))))
    # save the output
    output.save(out_file=out_file)


if __name__ == "__main__":
    main(sys.argv[1:])

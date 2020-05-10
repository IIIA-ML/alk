"""Script to generate PDP of an insights experiment

usage: gen_pdp.py [-h] [-o OUTFILE] [-l LOGFILE] [--logc {DEBUG,INFO}]
                  [--logf {DEBUG,INFO}]
                  expfile calcstep qstep

positional arguments:
  expfile               Anytime Lazy KNN Insights experiment result pickle
                        file path
  calcstep              Either steps of no of calculations for discrete PDP or
                        the precision as the ratio of the max `total_calc`
  qstep                 Steps to scale quality within (0.0, 1.0)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        PDP output pickle file path (default: None)
  -l LOGFILE, --logfile LOGFILE
                        Log file path (default: None)
  --logc {DEBUG,INFO}   console logger level (default: INFO)
  --logf {DEBUG,INFO}   file logger level (default: INFO)

Examples:
    # quality step=0.05, calc_step=0.0025 -> 1/400 of the max(calc) encountered for a test sequence during the experiment
    $ python -m alk.run.gen_pdp ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1.pk 0.0025 0.05 --logc DEBUG --logf DEBUG

"""

import argparse
import logging
import os
import sys
import time
import datetime

from alk import common
from alk.exp import pdp


logger = logging.getLogger("ALK")


def _parse_args(argv):
    """Helper function that creates a parser and parses the arguments"""
    # Configure argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("expfile", type=str,
                        help="Anytime Lazy KNN Insights experiment result pickle file path")
    parser.add_argument("calcstep", type=float,
                        help="Either steps of no of calculations for discrete PDP or the precision as the ratio of the max `total_calc`")
    parser.add_argument("qstep", type=float,
                        help="Steps to scale quality within (0.0, 1.0)")
    parser.add_argument("-o", "--outfile", type=str,
                        help="PDP output pickle file path")
    parser.add_argument("-l", "--logfile", type=str,
                        help="Log file path")
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
    pdp_file = args.outfile
    log_file = args.logfile
    # Generate file names
    if not pdp_file:
        pdp_file = pdp.gen_pdp_f_path(args.expfile, args.calcstep, args.qstep)
    if not log_file:
        log_file = common.gen_log_file(pdp_file)
    # Set logger
    logger = common.initialize_logger(console_level=args.logc, output_dir=common.APP.FOLDER.LOG, log_file=log_file, file_level=args.logf)
    logger.info("PDP generation script launched with args: {}".format(str(vars(args))))
    # build the PDP
    pdp_data, calc_step_actual = pdp.build_pdp(os.path.expanduser(args.expfile), calc_step=args.calcstep, q_step=args.qstep)
    # create an output object
    pdp_output = pdp.PDPOutput(settings=pdp.PDPSettings(experiment=args.expfile, calc_step_arg=args.calcstep, calc_step=calc_step_actual, q_step=args.qstep),
                               data=pdp_data)
    logger.info("PDP generation finished in {}.".format(datetime.timedelta(seconds=(time.time() - start_time))))
    # save the pdp
    pdp_output.save(out_file=pdp_file)


if __name__ == '__main__':
    main(sys.argv[1:])

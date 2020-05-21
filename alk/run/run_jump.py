"""Script to launch experiment with the `JumpingIterator` to collect gain data for various `jump_at` settings"""
import argparse
import datetime
import logging
import os
import sys
import time

from alk import common
from alk.exp import jump, ts


logger = logging.getLogger("ALK")


def _create_exp_jump_engine(dataset, tw_width, tw_step, k, test_size, jump_at_lst, similarity=None, gen_profile=None):
    """ Creates an jumping iteration experiment engine.

    Returns:
        jump.ExpJumpEngine:

    """
    # read the dataset -> cb
    cb = ts.gen_cb(dataset=dataset, tw_width=tw_width, tw_step=tw_step, gen_profile=gen_profile)
    # create an experiment engine
    max_val, min_val = ts.get_max_min(dataset)
    if similarity is None:
        similarity = lambda p1, p2: ts.euclidean_similarity_ts(p1, p2, max_=max_val, min_=min_val)
    engine = jump.ExpJumpEngine(cb=cb, k=k, similarity=similarity, jump_at_lst=jump_at_lst, test_size=test_size)
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
    parser.add_argument("-j", "--jumps", nargs='+', type=float, default=None,
                        help="jump_at values")
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
    # Generate file names
    out_file = args.outfile if args.outfile else jump.gen_jump_output_f_path(dataset, args.width, args.step,
                                                                             args.k, args.testsize, args.jum_at)
    log_file = args.logfile if args.logfile else common.gen_log_file(out_file)
    # Set logger
    logger = common.initialize_logger(console_level=args.logc, output_dir=common.APP.FOLDER.LOG, log_file=log_file,
                                      file_level=args.logf)
    logger.info("Jumping Iteration experiment script launched with args: {}".format(str(vars(args))))
    # create the experiment engine
    engine = _create_exp_jump_engine(dataset, args.width, args.step, args.k, args.testsize, args.jumps)
    # engine.run -> experiment data
    jump_data = engine.run()
    # create a result obj
    output = jump.ExpJumpOutput(settings=jump.ExpJumpSettings(dataset, args.width, args.step, args.k, args.testsize,
                                                              engine.cb.size(), args.jumps),
                                data=jump_data)
    logger.info("Jumping Iteration experiment finished in {}.".format(datetime.timedelta(seconds=(time.time() - start_time))))
    # save the output
    output.save(out_file=out_file)


if __name__ == '__main__':
    main(sys.argv[1:])

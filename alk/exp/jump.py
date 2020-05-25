"""The machinery to conduct experiment with the `JumpingIterator` to collect gain data for various `jump_at` settings"""

import logging
import os
from typing import List  # For type hints

import pandas as pd

from alk import cbr, common, rank
from alk.exp import exp_common

logger = logging.getLogger("ALK")


class ExpJumpData:

    def __init__(self):
        self.data = []

    def add(self, update, gain, jump_at):
        """Adds experiment data.

        Args:
            update (int): the index of the sequence update
            gain (float): Percentage of gain in terms of avoided similarity calculations compared to a brute-force search
            jump_at (int): the number of candidates after which the iterator jumped.

        """
        self.data.append([update, gain, jump_at])

    def process(self):
        """Returns dataframe w/o actual processing

        Returns:
            pd.DataFrame: columns -> ["update", "gain", "jumpat"]

        """
        return pd.DataFrame(self.data, columns=["update", "gain", "jumpat"])


class ExpJumpSettings:
    """Settings used in the jumping iteration experiment.

    Attributes:
        dataset (str): Full path of the dataset "arff" file
        tw_width (int): if > 0, width of the 'moving' time window;
            otherwise, 'expanding' time window approach is applied.
        tw_step (int): number of steps (in terms of data points in TS) taken at each update.
            This can also be seen as the number of data points changed at each update.
        k (int): k of kNN
        test_size (float): (0., 1.) ratio of the time series dataset to be used as Test CB
        cb_size (int): Number of cases in the Train CB
        jump_at_lst (List[int]): List of number of candidates after which the iterator will jump

    """
    def __init__(self, dataset, tw_width, tw_step, k, test_size, cb_size, jump_at_lst):
        """

        Args:
            dataset (str): Full path of the dataset "arff" file
            tw_width (int): if > 0, width of the 'moving' time window;
                otherwise, 'expanding' time window approach is applied.
            tw_step (int): number of steps (in terms of data points in TS) taken at each update.
                This can also be seen as the number of data points changed at each update.
            k (int): k of kNN
            test_size (float): (0., 1.) ratio of the time series dataset to be used as Test CB
            cb_size (int): Number of cases in the Train CB
            jump_at_lst (List[int]): List of number of candidates after which the iterator will jump

        """
        self.dataset = dataset
        self.tw_width = tw_width
        self.tw_step = tw_step
        self.k = k
        self.test_size = test_size
        self.cb_size = cb_size
        self.jump_at_lst = jump_at_lst


class ExpJumpOutput(exp_common.Output):
    """Holds and saves jumping iteration experiment settings and result data"""

    def __init__(self, settings, data):
        """Overrides the parent method only to specify type annotations.

        Args:
            settings (ExpJumpSettings):
            data (ExpJumpData):

        """
        # super(ExpIntrptOutput, self).__init__(settings, data)  # commented to help IDE auto-fill below attributes
        self.settings = settings  # type: ExpJumpSettings
        self.data = data

    def save(self, out_file=None):
        """Saves the object itself into a pickle file.

        Args:
            out_file (str): Full path to the dumped pickle file.
                If not given, it is generated automatically out of experiment settings.

        Returns:
            str: Full path to the dumped pickle file

        """
        if out_file is None:
            out_file = gen_jump_output_f_path(self.settings.dataset, self.settings.tw_width, self.settings.tw_step,
                                              self.settings.test_size, self.settings.jump_at_lst)
        common.dump_obj(self, out_file)
        logger.info("Anytime Lazy KNN - Insights experiment output dumped into '{}'.".format(out_file))
        return out_file


class ExpJumpEngine:
    """Jumping Iteration experiment engine"""

    def __init__(self, cb, k, similarity, jump_at_lst, test_size=0.01):
        """

        Args:
            cb (cbr.TCaseBase):
            k (int): k of kNN.
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            jump_at_lst (List[int]): List of number of candidates after which the iterator will jump
            test_size (float): ratio of the number of test sequences to be separated from the `cb`.

        """
        self.cb = cb
        self.k = k
        self.similarity = similarity
        self.jump_at_lst = jump_at_lst
        self.test_size = test_size

        logger.info("Jumping Iteration experiment engine created")

    def run(self):
        """Runs the Jumping Iteration experiment.

        Returns:
            pd.DataFrame: Output of the `ExpJumpData.process()`
                columns -> ["update", "gain", "jumpat"]

        """
        # Create an ExpJumpData instance to save experiment data
        exp_jump_data = ExpJumpData()
        # Generate test problems
        CB_train, CB_test = exp_common.split_cb(self.cb, self.test_size)
        len_test = len(CB_test)
        CB_train = cbr.TCaseBase(cb=CB_train)  # This will be passed to Anytime Lazy KNN, not the `ExpInsightsEngine.cb`
        # Conduct tests for each sequence in cb_test
        for idx, sequence in enumerate(CB_test):
            logger.info(".. Testing with problem sequence {} of {} (seq_id: {})".format(idx + 1, len_test, sequence.seq_id))
            # For every problem create multiple sequence solvers, one for TopDown, others for Jumping Iterator
            top_down_solver = exp_common.SolveSequence(CB_train, self.k, sequence, self.similarity, rank.TopDownIterator())  # Note: 'exp_insights_raw' not provided
            jumping_solvers = {}
            for jump_at in self.jump_at_lst:
                jumping_solvers[jump_at] = exp_common.SolveSequence(CB_train, self.k, sequence, self.similarity, rank.JumpingIterator(jump_at=jump_at))  # Note: 'exp_insights_raw' not provided
            # Run tests for each update
            for stop_update in range(sequence.n_profiles()):
                # Run top_down_solver to stop at the end of the stop_update
                logger.debug(".... TOP_DOWN_solver launching for stop_update: {}".format(stop_update))
                kNN_top_down, _, calc_pct_top_down = top_down_solver.solve(stop_update=stop_update)
                # Append to experiment data
                exp_jump_data.add(stop_update, 100. - calc_pct_top_down, 0)
                # Run jumping_solvers to jump after every `jump_at` calcs
                for jump_at in self.jump_at_lst:
                    logger.debug(
                        ".... JUMPING_solver w/ jump at {} launching for stop_update: {}".format(jump_at, stop_update))
                    kNN_jumping, calc_pct_jumping, _ = jumping_solvers[jump_at].solve(stop_update=stop_update)
                    # Append to experiment data
                    exp_jump_data.add(stop_update, 100. - calc_pct_top_down, jump_at)
            # Help garbage collector to release the memory as soon as possible
            del top_down_solver
            del jumping_solvers
        return exp_jump_data.process()


def gen_jump_output_f_path(dataset, tw_width, tw_step, k, test_size, jump_at_lst, suffix=""):
    """Returns full path of the output file for the insights experiment results"""
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]  # Base file name w/o extension
    jump_at_tag = "[{}]".format("_".join([str(j) for j in jump_at_lst]))
    out_file = os.path.join(common.APP.FOLDER.RESULT,
                            "JUMP_{d}_w_{w}_s_{s}_k_{k}_t_{t}_j_{j}{x}{e}".format(
                                d=dataset_name, w=tw_width, s=tw_step, k=k, t=str(test_size), j=jump_at_tag, x=suffix, e=common.APP.FILE_EXT.PICKLE))
    return out_file

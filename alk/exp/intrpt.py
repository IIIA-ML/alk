"""The machinery to conduct interruption experiments to collect gain and confidence efficiency data"""

import collections
import logging
import os

import pandas as pd

from alk import cbr, common, rank
from alk.exp import pdp, exp_common


logger = logging.getLogger("ALK")


class ExpIntrptSettings:
    """"
    Attributes:
        dataset (str): Full path of the dataset "arff" file
        pdp_file (str): Full path of the PDP file
        tw_width (int): if > 0, width of the 'moving' time window;
            otherwise, 'expanding' time window approach is applied.
        tw_step (int): number of steps (in terms of data points in TS) taken at each update.
            This can also be seen as the number of data points changed at each update.
        k (int): k of kNN
        conf_tholds (List[float]): List of confidence threshold values [0., 1.]
        z (int): z factor of the efficiency measure used in the experiment
        test_size (float): (0., 1.) ratio of the time series dataset to be used as Test CB
        cls_rank_iterator (type): `RankIterator` sub-class used in the experiment
        cls_rank_iterator_kwargs (dict): Keyword arguments for the `RankIterator.__init__`, if any.

    """
    def __init__(self, dataset, pdp_file, tw_width, tw_step, k, conf_tholds, z, test_size, cls_rank_iterator, cls_rank_iterator_kwargs={}):
        """

        Args:
            dataset (str): Full path of the dataset "arff" file
            pdp_file (str): Full path of the PDP file
            tw_width (int): if > 0, width of the 'moving' time window;
                otherwise, 'expanding' time window approach is applied.
            tw_step (int): number of steps (in terms of data points in TS) taken at each update.
                This can also be seen as the number of data points changed at each update.
            k (int): k of kNN
            conf_tholds (List[float]): List of confidence threshold values [0., 1.]
            z (int): z factor of the efficiency measure used in the experiment
            test_size (float): (0., 1.) ratio of the time series dataset to be used as Test CB
            cls_rank_iterator (type): `RankIterator` sub-class used in the experiment
            cls_rank_iterator_kwargs (dict): Keyword arguments for the `RankIterator.__init__`, if any.

        """
        self.dataset = dataset
        self.pdp_file = pdp_file
        self.tw_width = tw_width
        self.tw_step = tw_step
        self.k = k
        self.conf_tholds = conf_tholds
        self.z = z
        self.test_size = test_size
        self.cls_rank_iterator = cls_rank_iterator
        self.cls_rank_iterator_kwargs = cls_rank_iterator_kwargs


class ExpIntrptData:

    RawData = collections.namedtuple("RawData", "update calc knn_i conf std_dev quality gain sim thold")

    def __init__(self, z):
        """

        Args:
            z (int): z factor of the efficiency measure used in the experiment

        """
        self.z = z
        self.data = []

    def add(self, update, calc, knn_i, conf, std_dev, quality, gain, sim, thold=None):
        """Adds interruption experiment data for each interruption and/or termination.

        Args:
            update (int): the update of interruption (i.e. `stop_update`)
            calc (int):  the number of similarity calc used for interruption (i.e. `stop_calc`)
            knn_i (int): the index of the kNN member
            conf (float): confidence (i.e. estimated quality) given py the PDP for the `calc`.
            std_dev (float): standard deviation of the quality estimation
            quality (float): the actual quality of the approximate kNN[knn_i], see `alk.exp.pdp.quality()`
            gain (float): Percentage of gain in terms of avoided similarity calculations compared to a brute-force search
            sim (float): similarity(approximate kNN[knn_i], exact kNN[knn_i])
            thold (float): confidence threshold for the interruption, if it was given.

        """
        self.data.append(ExpIntrptData.RawData(update, calc, knn_i, conf, std_dev, quality, gain, sim, thold))

    def process(self):
        """Process the raw interruption experiment data.

        Calculates various error and efficiency values for estimated confidence.

        Returns:
            pd.DataFrame: columns -> ["update", "calc", "knni", "conf", "std", "quality", "gain", "sim",
                                      "abserr", "abspcterr", "effcysim", "effcyq", "confthold"]

        """
        processed_data = []
        for rd in self.data:  # type: ExpIntrptData.RawData
            # Absolute error
            abs_err = pdp.confidence_absolute_error(conf=rd.conf, q=rd.quality, std=rd.std_dev, z=self.z)
            # Absolute percentage error
            abs_pct_err = pdp.confidence_absolute_pct_error(conf=rd.conf, q=rd.quality, std=rd.std_dev, z=self.z)
            # Efficiency with sim
            effcy_sim = pdp.confidence_efficiency_sim(conf=rd.conf, sim=rd.sim, std=rd.std_dev, z=self.z)
            # Efficiency with quality
            effcy_q = pdp.confidence_efficiency_q(conf=rd.conf, q=rd.quality, std=rd.std_dev, z=self.z)

            processed_data.append([rd.update, rd.calc, rd.knn_i, rd.conf, rd.std_dev, rd.quality, rd.gain, rd.sim,
                                   abs_err, abs_pct_err, effcy_sim, effcy_q, rd.thold])

        return pd.DataFrame(processed_data,
                            columns=["update", "calc", "knni", "conf", "std", "quality", "gain", "sim",
                                     "abserr", "abspcterr", "effcysim", "effcyq", "confthold"])


class ExpIntrptOutput(exp_common.Output):
    """Holds and saves interruption experiment settings and result data"""

    def __init__(self, settings, data):
        """Overrides the parent method only to specify type annotations.

        Args:
            settings (ExpIntrptSettings):
            data (ExpIntrptData):  # TODO (OM, 20200825): Create new ExpIntrptProcessedData and refactor ExpIntrptData as ExpIntrptRawData

        """
        # super(ExpIntrptOutput, self).__init__(settings, data)  # commented to help IDE auto-fill below attributes
        self.settings = settings  # type: ExpIntrptSettings
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
            out_file = gen_intrpt_output_f_path(self.settings.dataset, self.settings.pdp_file, self.settings.tw_width, self.settings.tw_step,
                                                self.settings.k, self.settings.conf_tholds, self.settings.z, self.settings.test_size, self.settings.cls_rank_iterator)
        common.dump_obj(self, out_file)
        logger.info("Anytime Lazy KNN - Interruption experiment output dumped into '{}'.".format(out_file))
        return out_file


class ExpIntrptEngine:
    """Interruption experiment engine"""

    def __init__(self, pdp_file, cb, k, similarity, conf_tholds, cls_rank_iterator=rank.TopDownIterator, cls_rank_iterator_kwargs={}, z=-1, test_size=0.01):
        """

        Args:
            pdp_file (str): Full path of the PDP file
            cb (cbr.TCaseBase):
            k (int): k of kNN.
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            conf_tholds (List[float]): List of confidence threshold values [0., 1.]
            cls_rank_iterator (type): `RankIterator` sub-class used in the experiment
            cls_rank_iterator_kwargs (dict): Keyword arguments for the `RankIterator.__init__`, if any.
            z (int): factor of std deviations to be added or subtracted from the 'confidence'z in the 'efficiency' measure.
                It provides a means of how prudent we want to be with the raw confidence value provided by PDP.
                z=−1 would be a more cautious choice than the neutral z=0, and z=1 would be a more optimistic one.
            test_size (float): ratio of the number of test sequences to be separated from the `cb`.

        """
        self.pdp_file = pdp_file
        self.cb = cb
        self.k = k
        self.similarity = similarity
        self.conf_tholds = sorted(conf_tholds) if conf_tholds is not None else None  # sort in ascending order...
        self.cls_rank_iterator = cls_rank_iterator
        self.cls_rank_iterator_kwargs = cls_rank_iterator_kwargs
        self.z = z
        self.test_size = test_size

        logger.info("Interruption experiment engine created")

    def run(self):
        """Runs the interruption experiment.

        Returns:
            pd.DataFrame: Output of the `ExpIntrptData.process()`
                columns -> ["update", "calc", "knni", "conf", "std", "quality", "gain", "sim",
                            "abserr", "abspcterr", "effcysim", "effcyq", "confthold"]

        """
        # Create an ExpIntrptData instance to save experiment data
        exp_intrpt_data = ExpIntrptData(z=self.z)
        if self.conf_tholds is not None:
            self.conf_tholds = sorted(self.conf_tholds)  # sort just in case the arguments are not given in ascending order...
        # Generate test problems
        CB_train, CB_test = exp_common.split_cb(self.cb, self.test_size)
        len_test = len(CB_test)
        CB_train = cbr.TCaseBase(cb=CB_train)  # This will be passed to Anytime Lazy KNN, not the `ExpInsightsEngine.cb`
        # Conduct tests for each sequence in cb_test
        for idx, sequence in enumerate(CB_test):
            logger.info(".. Testing with problem sequence {} of {} (seq_id: {})".format(idx + 1, len_test, sequence.seq_id))
            # For every problem create two sequence solvers, one for uninterrupted, the other for the interrupted solving
            # instantiate the `RankIterator of choice with its given keyword arguments for both solvers
            rank_iterator_unint = self.cls_rank_iterator(**self.cls_rank_iterator_kwargs)
            rank_iterator_int = self.cls_rank_iterator(**self.cls_rank_iterator_kwargs)
            uninterrupted_solver = exp_common.SolveSequence(CB_train, self.k, sequence, self.similarity, rank_iterator_unint)  # Note: 'exp_insights_raw' not provided
            interrupted_solver = exp_common.SolveSequence(CB_train, self.k, sequence, self.similarity, rank_iterator_int)
            # Run tests for each update
            for stop_update in range(sequence.n_profiles()):
                # ------------------------------------------------------------------------------------------------------
                # 1) Run uninterrupted_solver to stop at the end of the stop_update
                logger.debug(".... UNINTERRUPTED_solver launching for stop_update: {}".format(stop_update))
                kNN_uninterrupted, _, _ = uninterrupted_solver.solve(stop_update=stop_update)
                # Set the stop_calc_list for testing
                if stop_update == 0:
                    stop_calc_list = [None]  # At initial problem (i.e. 0th update) we don't interrupt
                else:
                    stop_calc_list = pdp.get_calcs_for_conf_tholds(pdp_file=self.pdp_file, update=stop_update,
                                                                   conf_tholds=self.conf_tholds, z=self.z, knn_i=-1)  # last kNN member, i.e. kNN[k-1]
                    stop_calc_list.append(None)  # Make sure you iterate whole RANK in the end.
                logger.debug(".... stop_calc_list : {}".format(str(stop_calc_list)))
                # ------------------------------------------------------------------------------------------------------
                # 2) Run interrupted_solver to stop at each stop_calc in the stop_update
                interrupted = common.APP.INTRPT.W_CALC
                for stop_calc_ind, stop_calc in enumerate(stop_calc_list):
                    # Run AnytimeLazyKNN to stop at the `stop_calc`^th calc of the `stop_update`^th update
                    if interrupted == common.APP.INTRPT.W_CALC and (
                            True if (stop_calc_ind == 0 or stop_calc != stop_calc_list[stop_calc_ind - 1]) else False):
                        # This 'if' is needed for occasions when interrupted=False but conf_thold is provided  # TODO (OM, 20200505): ? Check if this info still holds
                        # Only execute if you have been interrupted before
                        # AND if stop_calc is different than the previous one (get_calc_for_confidence can return same calc for different confidence thresholds)
                        logger.debug(".... INTERRUPTED_solver launching for stop_update: {}, stop_calc: {}".format(stop_update, stop_calc))
                        kNN_interrupted, interrupted, calc_pct = interrupted_solver.solve(stop_update=stop_update, stop_calc_at_stop_update=stop_calc)
                    #  For each kNN[i], calculate "confidence", its "std_dev", "quality", "gain" and "sim" values
                    for knn_i in range(self.k):
                        knn_unint_i = kNN_uninterrupted[knn_i]
                        knn_int_i = kNN_interrupted[knn_i]
                        if stop_update != 0:
                            conf, std_dev = pdp.get_conf_for_calc(pdp_file=self.pdp_file, update=stop_update, knn_i=knn_i, calc=stop_calc)
                        else:
                            conf, std_dev = (1.0, 0.0)  # In 0^th update we don't interrupt, so the kNNs should be the same.
                        quality_ = knn_int_i.sim / knn_unint_i.sim if knn_unint_i.sim != 0. else 0.
                        gain = 100.0 - calc_pct
                        sim = self.similarity(CB_train.get_case_query(knn_unint_i.case_id), CB_train.get_case_query(knn_int_i.case_id))
                        if self.conf_tholds is not None:
                            if stop_calc is not None:
                                thold = self.conf_tholds[stop_calc_ind]
                            else:
                                thold = 1.  # uninterrupted (last run of the interruption points)
                        else:
                            thold = None
                        # Append the data to the conf_gain_data for the result file
                        exp_intrpt_data.add(update=stop_update, calc=stop_calc, knn_i=knn_i, conf=conf, std_dev=std_dev,
                                            quality=quality_, gain=gain, sim=sim, thold=thold)
                        # Compare knn_unint_i and knn_int_i given the confidence
                        log_unint_int_comparison(conf, std_dev, stop_update, stop_calc, knn_i, knn_unint_i, knn_int_i, quality_, sim)
            # Help garbage collector to release the memory as soon as possible
            del uninterrupted_solver
            del interrupted_solver
        return exp_intrpt_data.process()


def gen_intrpt_output_f_path(dataset, pdp_file, tw_width, tw_step, k, conf_tholds, z, test_size, cls_rank_iterator, suffix=""):
    """Returns full path of the output file for the interruption experiment results"""
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]  # Base file name w/o extension
    pdp_output = common.load_obj(pdp_file)  # type: pdp.PDPOutput
    calc_step = pdp_output.settings.calc_step_arg
    q_step = pdp_output.settings.q_step
    pdp_dataset = exp_common.get_setting(pdp_output.settings.experiment, "dataset")
    pdp_dataset = common.file_name_wo_ext(pdp_dataset)
    rank_iter_tag = cls_rank_iterator.abbrv
    z_tag = int(z) if int(z) == z else z
    conf_thold_tag = "[{}]".format("_".join([str(ct) for ct in conf_tholds])) if conf_tholds is not None else ""
    out_file = os.path.join(common.APP.FOLDER.RESULT,
                            "INT_{d}_w_{w}_s_{s}_k_{k}_r_{r}_PDP_{dp}_c_{c}_q_{q}__ct_{ct}_z_{z}_t_{t}{x}{e}".format(
                                d=dataset_name, w=tw_width, s=tw_step, k=k, r=rank_iter_tag, dp=pdp_dataset, c=calc_step, q=q_step,
                                ct=conf_thold_tag, z=z_tag, t=str(test_size), x=suffix, e=common.APP.FILE_EXT.PICKLE))
    return out_file


def get_avg_gain_for_intrpt_exp(exp_file, conf_tholds):
    """Gives the average gain throughout the sequence updates of an interruption experiment for given confidence thresholds.

    Args:
        exp_file (str): full path to the interruption experiment result file
        conf_tholds (list): of floats for confidence thresholds that the average gains are to be calculated for.
            If any given threshold was not used in the interruption experiment, it is silently ignored.

    Notes:
        Average gain is calculated for the last kNN member.

    Returns:
        pd.DataFrame: indices are the `conf_tholds` and the only column is the average gain

    """
    # Load interruption experiment
    exp_intrpt_output = common.load_obj(exp_file)  # type: ExpIntrptOutput
    data = exp_intrpt_output.data
    data = data.loc[data["update"] != 0]  # Filter the initial problem
    df = data.loc[data["knni"] == data["knni"].max()][["confthold", "gain"]]  # Use last kNN member only
    conf_tholds_in_df = df["confthold"].unique()
    for ind, thold in enumerate(conf_tholds):
        if thold not in conf_tholds_in_df:
            conf_tholds.pop(ind)
            print("Threshold argument {} is ignored, "
                  "it does not exist in the experiment results {}.".format(thold, exp_file))
    avg_gains = df.loc[df["confthold"].isin(conf_tholds)].groupby("confthold").mean()
    return avg_gains


def get_avg_effcy_for_intrpt_exp(exp_file, knn_i=None):
    """Gives the average confidence efficiency throughout the sequence updates of an interruption experiment.

    Args:
        exp_file (str): full path to the interruption experiment result file
        knn_i (int): Zero-based index of the kNN member;
            if None, the average performance is calculated for *all* kNNs.
    Returns:
        (float, float). (avg conf, std dev)

    """
    # Load interruption experiment
    exp_intrpt_output = common.load_obj(exp_file)  # type: ExpIntrptOutput
    data = exp_intrpt_output.data
    data = data.loc[data["update"] != 0]  # Filter the initial problem
    if knn_i is not None:
        data = data.loc[data["knni"] == knn_i]  # Filter by the given ki
    return data["effcyq"].mean(), data["effcyq"].std()  # μ & σ of all updates and stop calcs


def log_unint_int_comparison(conf, std_dev, stop_update, stop_calc, knn_i, knn_unint_i, knn_int_i, quality, sim, z=-3):
    """Helper function to check the exact and approximate kNNs regarding confidence upon interruption.

    Just generates warning log messages, if deemed necessary.
    """
    DEC_DIGITS = 3
    if conf + z * std_dev > quality:
        logger.warning(
            "····· High confidence warning: stop_update={}, stop_calc={}, k{}, (conf, \u03C3): {}, quality: {}; conf{}{}\u03C3>quality".format(
                stop_update, stop_calc, knn_i, (round(conf, DEC_DIGITS), round(std_dev, DEC_DIGITS)), round(quality, DEC_DIGITS), "+" if z >= 0 else "",  z))
    elif conf == 1. and knn_unint_i.case_id != knn_int_i.case_id:
        logger.warning(
            "·····  stop_update={}, stop_calc={}, k{}, (conf, \u03C3): {}, kNN[{}]s equal?: {} -> uninterrupted: {}, interrupted: {}, sim: {}".format(
                stop_update, stop_calc, knn_i, (round(conf, DEC_DIGITS), round(std_dev, DEC_DIGITS)), knn_i,
                knn_unint_i == knn_int_i,
                knn_unint_i, knn_int_i,
                round(sim, DEC_DIGITS)))
    elif conf < .5 and knn_unint_i == knn_int_i:
        logger.warning(
            "····· Low confidence warning: stop_update={}, stop_calc={}, k{}, (conf, \u03C3): {}, kNN[{}]s equal?: {} -> uninterrupted: {}, interrupted: {}".format(
                stop_update, stop_calc, knn_i, (round(conf, DEC_DIGITS), round(std_dev, DEC_DIGITS)), knn_i,
                knn_unint_i == knn_int_i,
                knn_unint_i, knn_int_i))



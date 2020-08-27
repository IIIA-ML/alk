"""The machinery to conduct classification experiments to collect gain and solution hit data
at interruptions at confidence thresholds and/or upon guaranteeing exact solution with bes-so-far kNN.
"""
import collections
import logging
import os

import pandas as pd

from alk import alk_classifier, cbr, common, rank
from alk.exp import exp_common, intrpt, pdp


logger = logging.getLogger("ALK")


class SolveSequenceClassifier(exp_common.SolveSequence):
    """Conduct interrupted and resumed kNN classification for all updates of a particular problem sequence.

    Attributes:
        cb (cbr.TCaseBase):
        k (int): k of kNN
        sequence (Sequence): A problem sequence with multiple updates
        similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
        exp_insights (ExpInsightsRaw): Optional. If given; (gain, delta, knn_insights) data are appended.
        anytime_lazy_knn (alk_classifier.AnytimeLazyKNNClassifier): `AnytimeLazyKNNClassifier` instance for the `sequence`.
        start_update (int): The update index to resume from after interruption.

    Notes:
        Note that `anytime_lazy_knn` attribute is an `AnytimeLazyKNNClassifier` instance. To interrupt the algorithm
        when best-so-far kNNs guarantee an exact solution, `SolveSequenceClassifier.solve` method's following argument
        should be passed as: `stop_calc_at_stop_update`=`alk_classifier.STOP_CALC_FOR_STOP_W_SOLN`.

    """
    def __init__(self, cb, k, sequence, similarity, rank_iterator, reuse, exp_insights=None):
        """Initialize a SolveSequence object

        Args:
            cb (cbr.TCaseBase):
            k (int): k of kNN
            sequence (cbr.Sequence): A problem sequence with multiple updates
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            rank_iterator (RankIterator): Instance of the `RankIterator` *sub-class* of choice to find kNN candidates
                within the `Rank` of the `Sequence`
            reuse (Type[BaseVote]): The classification method of choice
            exp_insights (ExpInsightsRaw): Optional. If given; (gain, delta, knn_insights) data are appended.

        """

        self.k = k
        self.cb = cb
        self.sequence = sequence
        self.similarity = similarity
        self.exp_insights = exp_insights
        self.anytime_lazy_knn = alk_classifier.AnytimeLazyKNNClassifier(reuse=reuse,
                                                                        seq_id=sequence.seq_id, cb=cb, k=k,
                                                                        similarity=similarity,
                                                                        rank_iterator=rank_iterator)
        self.anytime_lazy_knn_classifier = self.anytime_lazy_knn  # Just as an alias, to recycle self.solve() method
        self.start_update = 0


class ExpClassifierSettings:
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
        reuse (alk_classifier.BaseVote): Classification voting method
        stop_w_soln (bool): if True, ALK  Classifier is also interrupted upon guaranteeing exact solution;
            otherwise, only `conf_tholds` list is used for interruption.
        cls_rank_iterator (type): `RankIterator` sub-class used in the experiment
        cls_rank_iterator_kwargs (dict): Keyword arguments for the `RankIterator.__init__`, if any.

    """

    def __init__(self, dataset, pdp_file, tw_width, tw_step, k, conf_tholds, z, test_size, reuse, stop_w_soln,
                 cls_rank_iterator, cls_rank_iterator_kwargs={}):
        self.dataset = dataset
        self.pdp_file = pdp_file
        self.tw_width = tw_width
        self.tw_step = tw_step
        self.k = k
        self.conf_tholds = conf_tholds
        self.z = z
        self.test_size = test_size
        self.reuse = reuse
        self.stop_w_soln = stop_w_soln
        self.cls_rank_iterator = cls_rank_iterator
        self.cls_rank_iterator_kwargs = cls_rank_iterator_kwargs


class ExpClassifierProcessedData:
    """
    Attributes:
        gain_data (pd.DataFrame): Gain DataFrame columns ->
            ["update", "calc", "knni", "conf", "std", "quality", "gain", "sim", "abserr", "abspcterr", "effcysim",
            "effcyq", "confthold", "stopwsoln", "intrptwsoln", "hit"]
        hit_data (pd.DataFrame): Hit DataFrame columns ->
            ["update", "confthold", "stopwsoln", "intrptwsoln", "hit"]
    """
    def __init__(self, gain_data, hit_data):
        self.gain_data = gain_data
        self.hit_data = hit_data


class ExpClassifierRawData:

    RawGainData = collections.namedtuple("RawData", "update calc knn_i conf std_dev quality gain sim thold stop_w_soln intrpt_w_soln")
    HitData = collections.namedtuple("HitData", "update thold stop_w_soln intrpt_w_soln hit")

    def __init__(self, z):
        """

        Args:
            z (int): z factor of the efficiency measure used in the experiment

        """
        self.z = z
        self.gain_data = []  # type: List[RawGainData]
        self.hit_data = []  # type: List[HitData]

    def add_gain(self, update, calc, knn_i, conf, std_dev, quality, gain, sim, thold=None,
            stop_w_soln=None, intrpt_w_soln=None):
        """Adds classification experiment data for each interruption and/or termination.

        Args:
            update (int): the update of interruption (i.e. `stop_update`)
            calc (int):  the number of similarity calc used for interruption (i.e. `stop_calc`)
            knn_i (int): the index of the kNN member
            conf (float): confidence (i.e. estimated quality) given py the PDP for the `calc`.
            std_dev (float): standard deviation of the quality estimation
            quality (float): the actual quality of the approximate kNN[knn_i], see `alk.exp.pdp.quality()`
            gain (float): Percentage of gain in terms of avoided similarity calculations compared to a brute-force search
            sim (float): similarity(approximate kNN[knn_i], exact kNN[knn_i])
            thold (float): confidence threshold for the interruption, if it was given
            stop_w_soln (bool): True if stop_w_soln was applied, i.e. if interruption with exact solution was requested
            intrpt_w_soln (bool): True if interruption w/ exact solution actually occurred

        """
        self.gain_data.append(ExpClassifierRawData.RawGainData(update, calc, knn_i, conf, std_dev, quality, gain, sim, thold,
                                                               stop_w_soln, intrpt_w_soln))

    def add_hit(self, update, thold=None, stop_w_soln=None, intrpt_w_soln=None, hit=None):
        """Adds classification experiment data for each interruption and/or termination.

        Args:
            update (int): the update of interruption (i.e. `stop_update`)
            thold (float): confidence threshold for the interruption, if it was given
            stop_w_soln (bool): True if stop_w_soln was applied, i.e. if interruption with exact solution was requested
            intrpt_w_soln (bool): True if interruption w/ exact solution actually occurred
            hit (bool): True if the solution w/ best-so-far kNN = solution w/ exact kNN
        """
        self.hit_data.append(ExpClassifierRawData.HitData(update, thold, stop_w_soln, intrpt_w_soln, hit))

    def process(self):
        """Process the raw classification experiment data.

        Calculates various error and efficiency values for estimated confidence.

        Returns:
            ExpClassifierProcessedData:

        """
        processed_gain_data = []
        for rd in self.gain_data:  # type: ExpClassifierRawData.RawGainData
            if rd.conf is not None:
                # Absolute error
                abs_err = pdp.confidence_absolute_error(conf=rd.conf, q=rd.quality, std=rd.std_dev, z=self.z)
                # Absolute percentage error
                abs_pct_err = pdp.confidence_absolute_pct_error(conf=rd.conf, q=rd.quality, std=rd.std_dev, z=self.z)
                # Efficiency with sim
                effcy_sim = pdp.confidence_efficiency_sim(conf=rd.conf, sim=rd.sim, std=rd.std_dev, z=self.z)
                # Efficiency with quality
                effcy_q = pdp.confidence_efficiency_q(conf=rd.conf, q=rd.quality, std=rd.std_dev, z=self.z)
            else:
                abs_err, abs_pct_err, effcy_sim, effcy_q = None, None, None, None

            processed_gain_data.append([rd.update, rd.calc, rd.knn_i, rd.conf, rd.std_dev, rd.quality, rd.gain, rd.sim,
                                        abs_err, abs_pct_err, effcy_sim, effcy_q, rd.thold, rd.stop_w_soln, rd.intrpt_w_soln])

        exp_classifier_processed_data = ExpClassifierProcessedData(
            gain_data=pd.DataFrame(processed_gain_data,
                                   columns=["update", "calc", "knni", "conf", "std", "quality", "gain", "sim", "abserr",
                                            "abspcterr", "effcysim", "effcyq", "confthold", "stopwsoln", "intrptwsoln"]),
            hit_data=pd.DataFrame(self.hit_data,
                                  columns=["update", "confthold", "stopwsoln", "intrptwsoln", "hit"]))
        return exp_classifier_processed_data


class ExpClassifierOutput(exp_common.Output):
    """Holds and saves classification experiment settings and result data"""

    def __init__(self, settings, data):
        """Overrides the parent method only to specify type annotations.

        Args:
            settings (ExpClassifierSettings):
            data (ExpClassifierProcessedData):

        """
        # super(ExpIntrptOutput, self).__init__(settings, data)  # commented to help IDE auto-fill below attributes
        self.settings = settings  # type: ExpClassifierSettings
        self.data = data  # type: ExpClassifierProcessedData

    def save(self, out_file=None):
        """Saves the object itself into a pickle file.

                Args:
                    out_file (str): Full path to the dumped pickle file.
                        If not given, it is generated automatically out of experiment settings.

                Returns:
                    str: Full path to the dumped pickle file

                """
        if out_file is None:
            out_file = gen_classifier_output_f_path(self.settings.dataset, self.settings.pdp_file,
                                                    self.settings.tw_width, self.settings.tw_step,
                                                    self.settings.k, self.settings.conf_tholds, self.settings.z,
                                                    self.settings.test_size, self.settings.cls_rank_iterator,
                                                    self.settings.reuse, self.settings.stop_w_soln)
        common.dump_obj(self, out_file)
        logger.info("Anytime Lazy KNN Classifier - Classification experiment output dumped into '{}'.".format(out_file))
        return out_file


class ExpClassifierEngine:
    """Classification experiment engine"""

    def __init__(self, pdp_file, cb, k, similarity, conf_tholds, reuse, stop_w_soln,
                 cls_rank_iterator=rank.TopDownIterator, cls_rank_iterator_kwargs={}, z=-1, test_size=0.01):
        """

        Args:
            pdp_file (str): Full path of the PDP file
            cb (cbr.TCaseBase):
            k (int): k of kNN.
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            conf_tholds (List[float]): List of confidence threshold values [0., 1.]
            reuse (alk_classifier.BaseVote): Classification voting method
            stop_w_soln (bool): if True, ALK  Classifier is also interrupted upon guaranteeing exact solution;
            otherwise, only `conf_tholds` list is used for interruption.
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
        self.reuse = reuse
        self.stop_w_soln = stop_w_soln
        self.cls_rank_iterator = cls_rank_iterator
        self.cls_rank_iterator_kwargs = cls_rank_iterator_kwargs
        self.z = z
        self.test_size = test_size

        logger.info("Classification experiment engine created")

    def run(self):
        """Runs the classification experiment.

        Returns:
            ExpClassifierProcessedData:

        """
        # Create an ExpClassifierRawData instance to save experiment data
        exp_classifier_data = ExpClassifierRawData(z=self.z)
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
            uninterrupted_solver = SolveSequenceClassifier(CB_train, self.k, sequence, self.similarity, rank_iterator_unint, self.reuse)  # Note: 'exp_insights_raw' not provided
            interrupted_solver = SolveSequenceClassifier(CB_train, self.k, sequence, self.similarity, rank_iterator_int, self.reuse)
            if self.stop_w_soln:
                rank_iterator_int_w_soln = self.cls_rank_iterator(**self.cls_rank_iterator_kwargs)
                interrupted_w_soln_solver = SolveSequenceClassifier(CB_train, self.k, sequence, self.similarity, rank_iterator_int_w_soln, self.reuse)
            # Run tests for each update
            for stop_update in range(sequence.n_profiles()):
                # ------------------------------------------------------------------------------------------------------
                # 1) Run uninterrupted_solver to stop at the end of the stop_update
                logger.debug(".... UNINTERRUPTED_solver launching for stop_update: {}".format(stop_update))
                kNN_uninterrupted, _, _ = uninterrupted_solver.solve(stop_update=stop_update)
                # Solution Uninterrupted
                soln_uninterrupted, _ = uninterrupted_solver.anytime_lazy_knn_classifier.suggest_solution()
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
                    # Run AnytimeLazyKNNClassifier to stop at the `stop_calc`^th calc of the `stop_update`^th update
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
                        # Append the data to the gain_data for the result file
                        exp_classifier_data.add_gain(update=stop_update, calc=stop_calc, knn_i=knn_i, conf=conf, std_dev=std_dev,
                                                     quality=quality_, gain=gain, sim=sim, thold=thold,
                                                     stop_w_soln=False, intrpt_w_soln=False)
                        # Compare knn_unint_i and knn_int_i given the confidence
                        intrpt.log_unint_int_comparison(conf, std_dev, stop_update, stop_calc, knn_i, knn_unint_i, knn_int_i, quality_, sim)
                    if interrupted == common.APP.INTRPT.W_CALC:
                        # Solution Interrupted
                        soln_interrupted, _ = interrupted_solver.anytime_lazy_knn_classifier.suggest_solution()
                        hit_interrupted = soln_interrupted == soln_uninterrupted
                        exp_classifier_data.add_hit(update=stop_update, thold=thold, stop_w_soln=False, intrpt_w_soln=False, hit=hit_interrupted)
                        logger.debug("...... Solution hit for conf_thold {}: {}".format(thold, hit_interrupted))
                # ------------------------------------------------------------------------------------------------------
                # 3) Run interrupted_w_soln_solver to stop at the moment when an exact solution is guaranteed
                if self.stop_w_soln:
                    logger.debug(".... INTERRUPTED_W_SOLN_solver launching for stop_update: {}, stop_calc: {}".format(stop_update, alk_classifier.STOP_CALC_FOR_STOP_W_SOLN))
                    kNN_interrupted_w_soln, interrupted, calc_pct = interrupted_w_soln_solver.solve(
                        stop_update=stop_update,
                        stop_calc_at_stop_update=alk_classifier.STOP_CALC_FOR_STOP_W_SOLN if stop_update > 0 else None)
                    # Append the data to the gain_data for the result file
                    exp_classifier_data.add_gain(update=stop_update, calc=None, knn_i=None, conf=None, std_dev=None,
                                                 quality=None, gain=100.0 - calc_pct, sim=None, thold=None,
                                                 stop_w_soln=True, intrpt_w_soln=interrupted == common.APP.INTRPT.W_SOLN)
                    # Save solution hit upon interruption w/ exact soln
                    if interrupted == common.APP.INTRPT.W_SOLN:
                        soln_interrupted_w_soln, _ = interrupted_solver.anytime_lazy_knn_classifier.suggest_solution()
                        hit_interrupted_w_soln = soln_uninterrupted == soln_interrupted_w_soln
                        exp_classifier_data.add_hit(update=stop_update, thold=None, stop_w_soln=True, intrpt_w_soln=True, hit=hit_interrupted_w_soln)
                        logger.debug("...... Solution hit for stop_w_soln: {}".format(hit_interrupted_w_soln))
                        if not hit_interrupted_w_soln:
                            logger.error("...... Solution hit NOT achieved for stop_w_soln !!!")
                        # run interrupted_w_soln_solver to complete the search for the remaining ki's w/o interruption
                        if stop_update > 0:
                            logger.debug(".... INTERRUPTED_W_SOLN_solver *resuming* for stop_update: {}, stop_calc: {}".format(stop_update, stop_calc))
                            interrupted_w_soln_solver.solve(stop_update=stop_update, stop_calc_at_stop_update=None)
            # Help garbage collector to release the memory as soon as possible
            del uninterrupted_solver
            del interrupted_solver
            if self.stop_w_soln:
                del interrupted_w_soln_solver
        return exp_classifier_data.process()


def gen_classifier_output_f_path(dataset, pdp_file, tw_width, tw_step, k, conf_tholds, z, test_size, cls_rank_iterator,
                                 reuse, stop_w_soln, suffix=""):
    """Returns full path of the output file for the classification experiment results"""
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]  # Base file name w/o extension
    pdp_output = common.load_obj(pdp_file)  # type: pdp.PDPOutput
    calc_step = pdp_output.settings.calc_step_arg
    q_step = pdp_output.settings.q_step
    pdp_dataset = exp_common.get_setting(pdp_output.settings.experiment, "dataset")
    pdp_dataset = common.file_name_wo_ext(pdp_dataset)
    rank_iter_tag = cls_rank_iterator.abbrv
    z_tag = int(z) if int(z) == z else z
    conf_thold_tag = "[{}]".format("_".join([str(ct) for ct in conf_tholds])) if conf_tholds is not None else ""
    reuse_tag = reuse.abbrv
    out_file = os.path.join(common.APP.FOLDER.RESULT,
                            "CLS_{d}_w_{w}_s_{s}_k_{k}_r_{r}_PDP_{dp}_c_{c}_q_{q}__ct_{ct}_z_{z}_t_{t}_ws_{ws}_ru_{ru}{x}{e}".format(
                                d=dataset_name, w=tw_width, s=tw_step, k=k, r=rank_iter_tag, dp=pdp_dataset, c=calc_step, q=q_step,
                                ct=conf_thold_tag, z=z_tag, t=str(test_size), ws=int(stop_w_soln), ru=reuse_tag, x=suffix,
                                e=common.APP.FILE_EXT.PICKLE))
    return out_file


def get_avg_gain_for_classify_exp(exp_file, conf_tholds, wsoln=False, lblwsoln="w/Soln"):
    """Gives the average gain throughout the sequence updates in a classification experiment
    for given confidence thresholds and optionally upon stopping w/ exact solution.

    Args:
        exp_file (str): full path to the classification experiment result file
        conf_tholds (list): of floats for confidence thresholds that the average gains are to be calculated for.
            If any given threshold was not used in the classification experiment, it is silently ignored.
        wsoln (bool): If True, gains for 'stop_w_soln=1' rows are also added.
        lblwsoln (str): Label of the column for the interruption with exact solution

    Notes:
        Average gain is calculated for the last kNN member.

    Returns:
        pd.DataFrame: indices are the `conf_tholds` and `w/soln` if requested, and the only column is the average gain

    """
    # Load classification experiment
    exp_intrpt_output = common.load_obj(exp_file)  # type: ExpClassifierOutput
    data = exp_intrpt_output.data.gain_data
    data = data.loc[data["update"] != 0]  # Filter the initial problem
    df = data.loc[(data["stopwsoln"] == 0) & (data["knni"] == data["knni"].max())][["confthold", "gain"]]  # Use intrpt w/conf and last kNN member only
    conf_tholds_in_df = df["confthold"].unique()
    conf_tholds_copy = conf_tholds[:]
    for ind, thold in enumerate(conf_tholds_copy):
        if thold not in conf_tholds_in_df:
            conf_tholds_copy.pop(ind)
            print("Threshold argument {} is ignored, "
                  "it does not exist in the experiment results {}.".format(thold, exp_file))
    avg_gains = df.loc[df["confthold"].isin(conf_tholds_copy)].groupby("confthold").mean()
    if wsoln:
        # Add the avg gain for the experiment w/ stop_w_soln=True
        # Doesn't matter if an interruption has actually occurred w/ exact soln or not, see ignore 'intrptwsoln' column
        s_avg_w_soln = data.loc[data["stopwsoln"] == 1][["gain"]].mean()
        s_avg_w_soln.name = lblwsoln
        avg_gains = avg_gains.append(s_avg_w_soln)
    return avg_gains


def get_avg_effcy_for_classify_exp(exp_file, knn_i=None):
    """Gives the average confidence efficiency throughout the sequence updates of an interruption/classification experiment.

    Args:
        exp_file (str): full path to the classification experiment result file
        knn_i (int): Zero-based index of the kNN member;
            if None, the average performance is calculated for *all* kNNs.
    Returns:
        (float, float). (avg conf, std dev)

    """
    # Load classification experiment
    exp_classify_output = common.load_obj(exp_file)  # type: ExpClassifierOutput
    data = exp_classify_output.data.gain_data
    data = data.loc[(data["stopwsoln"] == 0) & (data["update"] != 0)]  # Filter the stop w/soln records and the initial problem
    if knn_i is not None:
        data = data.loc[data["knni"] == knn_i]  # Filter by the given ki
    return data["effcyq"].mean(), data["effcyq"].std()  # μ & σ of all updates and stop calcs


def get_avg_hit_for_classify_exp(exp_file, conf_tholds, wsoln=False, lblwsoln="w/Soln"):
    """Gives the average solution throughout updates of an classification experiment for given confidence thresholds.

    Args:
        exp_file (str): full path to classification experiment result file
        conf_tholds (list): list of floats for confidence thresholds that the average gains are to be calculated for.
            If any given threshold was not used in the classification experiment, it is silently ignored.
        wsoln (bool): If True, gains for 'stop_w_soln=1' rows are also added.
        lblwsoln (str): Label of the column for the interruption with exact solution

    Returns:
        pandas.DataFrame: indices are the `conf_tholds` and the only column is the average hit

    """
    # Load classification experiment
    exp_intrpt_output = common.load_obj(exp_file)  # type: ExpClassifierOutput
    data = exp_intrpt_output.data.hit_data
    data = data.loc[data["update"] != 0]  # Filter the initial problem
    df = data.loc[data["stopwsoln"] == 0][["confthold", "hit"]]  # Use intrpt w/conf and all kNN members
    conf_tholds_in_df = df["confthold"].unique()
    conf_tholds_copy = conf_tholds[:]
    for ind, thold in enumerate(conf_tholds_copy):
        if thold not in conf_tholds_in_df:
            conf_tholds_copy.pop(ind)
            print("Threshold argument {} is ignored, "
                  "it does not exist in the experiment results {}.".format(thold, exp_file))
    avg_hits = df.loc[df["confthold"].isin(conf_tholds_copy)].groupby("confthold").mean()
    if wsoln:
        # Add the avg hit for the experiment w/ stop_w_soln=True and intrptwsoln=True
        # An interruption has to have actually occurred w/ exact soln to be included in hits
        s_avg_w_soln = data.loc[(data["stopwsoln"] == 1) & (data["intrptwsoln"] == 1)][["hit"]].mean()
        s_avg_w_soln.name = lblwsoln
        avg_hits = avg_hits.append(s_avg_w_soln)
    return avg_hits

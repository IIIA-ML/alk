"""The machinery to conduct experiments to collect gain and knn insights data"""

import logging
import os
from typing import List, Union, Any  # For type hints

import numpy as np

from alk import alk, common, helper, cbr
from alk.exp import exp_common


logger = logging.getLogger("ALK")


class ExpInsightsRaw:
    """Holds raw insight data of experiments.

    Outer list of an attribute is for test sequences, inner list is for updates in each sequence

    Attributes:
        gain (List[List[float]]): For an experiment of 'm' test sequences with 'n' updates each,
            gains are given as:
                     [[g00, g10, ..., gn0], [[g01, g11, ..., gn1]], ..., [[g0m, g1m, ..., gnm]]]
            update:     0    1         n       0    1         n             0    1         n
                       ------------------     ------------------           ------------------
            sequence:           0                      1                            m
        delta (List[List[float]]):
        knn_insights (List[List[alk.KNNInsights]]):

    """

    def __init__(self):
        # replacing CBR.executions
        self.gain = []  # type: List[List[float]]
        # replacing CBR.deltas
        self.delta = []  # type: List[List[float]]
        # replacing CBR.exp_k_calcs
        self.knn_insights = []  # type: List[List[alk.KNNInsights]]

    def add_new_sequence(self):
        """Creates an empty list for the new test sequence in all instance attributes."""
        for attr in common.public_attrs(self):
            self.__dict__[attr].append([])

    def add_insights_for_cur_sequence_update(self, **kwargs):
        """Adds the values of the given insight keywords for the current update of the current test sequence.

        1) Insights should be added per update
        2) Given keyword arguments must match defined 'public' instance attribute names.
        3) Each given key=value is appended as an [update_id, value] to the *latest* test sequence.

        Args:
            **kwargs: key=value pairs where each key corresponds to an instance attribute of an insight data

        Returns:
            None:

        Raises:
            AttributeError: If a non-existing or private attribute is tried to be accessed.
            IndexError: If `add_new_sequence` is not called before adding data for the new sequence.

        Examples:
            .add_insights_for_cur_sequence_update(gain=74.04, delta=0.12, knn_insights=<`alk.KNNInsights` instance for current update>)

        """
        allowed_attrs = common.public_attrs(self)  # public attrs only
        # Validate
        if not set(kwargs.keys()).issubset(allowed_attrs):
            raise AttributeError("{} is not within the allowed attributes: {}.".format(
                set(kwargs.keys()).difference(allowed_attrs), allowed_attrs))
        # Append values to corresponding instance attributes
        for kw, v in kwargs.items():
            if not self.__dict__[kw]:  # Empty list
                raise IndexError("First call 'add_new_sequence' method to be able to save insights for the first sequence")
            else:
                self.__dict__[kw][-1].append(v)  # Append to last sequence

    def get_knn_insight_for_all_updates(self, test_idx, insight):
        """Gives the list of the `alk.KNNInsights.<insight>` values collected at each for the given test sequence.

        Args:
            test_idx (int): the index of the test sequence in the experiment. This is NOT the `Sequence.seq_id`.
            insight (str): name of the knn_insight data

        Returns:
            List[List[Union[int, float]]]: List of 'u' sublists, where 'u' is the number of updates of the sequence.
                And, each sublist contains 'k' `insight` values gathered for kNN members at the corresponding update.
                e.g.: `insights`="total_cals", no of updates=5, k=3
                -> [[300, 0, 0], [118, 4, 1], [240, 0, 0], [145, 0, 3], [239, 0, 1]]
                     k0  k1 k2    k0  k1 k2
                     ---------    ---------
                     update 0     update 1

        """
        knni_for_all_updates = [entry for entry in self.knn_insights[test_idx]]  # entry = <`alk.KNNInsights` instance>
        return [knni.get(insight) for knni in knni_for_all_updates]

    def get_n_upd(self, test_idx):
        """Get the number of update `insight` *entries* for the given test sequence

        Args:
            test_idx (int): the index of the test in the experiment. This is NOT the `Sequence.seq_id`.

        Notes:
            IMPORTANT: This is not the number of updates of the test sequence! Just the number of current entries.

        Raises:
            ValueError: "Insight attributes of the sequence have different number of entries"

        Returns:
            int:

        """
        # Make sure all insight attributes have the same number of entries
        set_n_entries_for_all_insight_attrs = set([len(self.__dict__[attr][test_idx]) for attr in common.public_attrs(self)])
        if len(set_n_entries_for_all_insight_attrs) > 1:
            ValueError("Insight attributes of the {}^th sequence have different number of entries".format(test_idx))
        # There is a single number of entries which is the same for all insights
        return set_n_entries_for_all_insight_attrs.pop()

    def get_n_seq(self):
        """Gives the number of test sequences used in experimentation.

        The number takes into account multiple experiments if that was the case.

        Returns:
            int:

        """
        return len(self.knn_insights)  # Could have been any other data attribute

    def get_max_seq_len(self):
        """Gives the max sequence length

        Returns:
            int:

        """
        return max([len(seq_result) for seq_result in self.knn_insights])  # seq_result: List[alk.KNNInsights]

    def get_k(self):
        """Gives the `k` used in the exp"""
        if not self.knn_insights:
            raise IndexError("No kNN insights data found")
        return len(self.knn_insights[0][0])  # self.knn_insights[0][0] : <`alk.KNNInsights`> object


class ExpInsightsProcessed:
    """Holds processed insight data of experiments"""
    def __init__(self, gain, delta, knn_total_cumsum, knn_total_abs, knn_actual_cumsum, knn_actual_abs,
                 knn_found_stage_id, knn_last_calc_stage_id, knn_case_id, knn_calc_sim_history):
        """

        Args:
            gain (List[List[Union[int, float]]]):
            delta (List[List[Union[int, float]]]):
            knn_total_cumsum (List[List[int]]):
            knn_total_abs (List[List[int]]):
            knn_actual_cumsum (List[List[int]]):
            knn_actual_abs (List[List[int]]):
            knn_found_stage_id (List[List[int]]):
            knn_last_calc_stage_id (List[List[int]]):
            knn_case_id (List[List[int]]):
            knn_calc_sim_history (List[List[Union[int, List[Tuple[int, float]]]]]):

        Notes:
            First item in the first level sublists is the update id.

        Examples:
            gain: [[1, 3.33], [2, 32.33], [3, 39.66], [4, 25.0],
                   [1, 66.33], [2, 20.0], [3, 49.33], [4, 20.0], ...]
            knn_total_abs: [[1, 213, 236, 290], [2, 203, 203, 203], [3, 163, 164, 181], [4, 151, 151, 225],
                            [1, 99, 101, 101], [2, 240, 240, 240], [3, 152, 152, 152], [4, 240, 240, 240], ...]
            knn_calc_sim_history: [[1, [(0, 0.0), (1, 0.43), (61, 0.84), (105, 0.89), ..., (213, 0.98)],
                                       [(0, 0.0), (2, 0.43), (61, 0.43), (105, 0.89), ..., (236, 0.97)],
                                       [(0, 0.0), (3, 0.43), (61, 0.43), (105, 0.89), ..., (290, 0.94)]],
                                   [2, [(0, 0.0), (1, 0.57), (61, 0.82), (117, 0.90), ..., (203, 0.97)],
                                       [(0, 0.0), (2, 0.57), (61, 0.57), (62, 0.57), ..., (203, 0.96)],
                                       [(0, 0.0), (3, 0.57), (61, 0.57), (62, 0.57), ..., (203, 0.94)]], ...
                                  ]
        """
        self.gain = gain
        self.delta = delta
        self.knn_total_cumsum = knn_total_cumsum
        self.knn_total_abs = knn_total_abs
        self.knn_actual_cumsum = knn_actual_cumsum
        self.knn_actual_abs = knn_actual_abs
        self.knn_found_stage_id = knn_found_stage_id
        self.knn_last_calc_stage_id = knn_last_calc_stage_id
        self.knn_case_id = knn_case_id
        self.knn_calc_sim_history = knn_calc_sim_history


class ProcessExpInsightsRaw:
    """Holds gain experiment settings and results."""
    def __init__(self, raw_insights=None):
        """

        Args:
            raw_insights (List[ExpInsightsRaw]): Insights of all experiments conducted in one run
        """
        self.raw_insights = raw_insights

    def _insert_upd_id(self, processed_insight: Union[List[List[float]], np.ndarray]) -> List[List[Any]]:
        """Used in Final processing.

        Insert update id in the beginning of the insight values list for every update and remove 0^th update

        """
        post_processed_insight = []
        for test_seq_result in processed_insight:
            for update_no, update_result in enumerate(test_seq_result):
                if update_no > 0:  # and not (isinstance(update_result, np.ndarray) and np.isnan(update_result).any()):  # exclude 0th update and nan rows
                    # TODO (OM, 20200429): Below, maybe append a tuple instead of a list. In this case, change type hints of `ExpInsightsProcessed`
                    if isinstance(update_result, np.ndarray):  # for knn insights related data
                        post_processed_insight.append([update_no] + update_result.tolist())
                    elif isinstance(update_result, float):  # for gain & delta data
                        post_processed_insight.append([update_no, update_result])
                    else:
                        raise TypeError("Unhandled experiment insight data type {} for the value {}".format(type(update_result), update_result))
        return post_processed_insight

    def process(self):
        """Process the raw insights data.

        Notes:
            Processing the data is conducted in two steps:
            1) Pre-processing:
               1.1) Unite test sequences in one list
               1.2) For knn insights:
                    1.2.1) Form lists of values for each kNN insight by extracting it from the `alk.KNNInsights` object
                    1.2.2) Make cumulative sums for 'total', 'absolute total', 'actual' and 'absolute actual ' calcs for kNNs
            2) Final Processing:
               Insert update id in the beginning of the insight values list for every update and remove 0^th update results

        Returns:
            ExpInsightsProcessed:

        """
        # 1) PRE-PROCESSING
        n_exp = len(self.raw_insights)
        dim_seq = sum([exp.get_n_seq() for exp in self.raw_insights])  # total number of test sequences in all exp runs
        dim_upd = max([exp.get_max_seq_len() for exp in self.raw_insights])  # max number of updates in a sequence
        dim_k = self.raw_insights[0].get_k()  # k of kNN
        pre_gain = []
        pre_delta = []
        pre_knn_total_cumsum = np.full((dim_seq, dim_upd, dim_k), np.nan)
        pre_knn_actual_cumsum = np.full((dim_seq, dim_upd, dim_k), np.nan)
        pre_knn_total_abs = np.full((dim_seq, dim_upd, dim_k), np.nan)
        pre_knn_actual_abs = np.full((dim_seq, dim_upd, dim_k), np.nan)
        pre_knn_found_stage_id = np.full((dim_seq, dim_upd, dim_k), np.nan)
        pre_knn_last_calc_stage_id = np.full((dim_seq, dim_upd, dim_k), np.nan)
        pre_knn_case_id = np.full((dim_seq, dim_upd, dim_k), np.nan, dtype=object)
        pre_knn_calc_sim_history = np.full((dim_seq, dim_upd, dim_k), np.nan, dtype=object)
        # Iterate each experiment
        i = 0  # Counter to number test sequences in all experiments
        for exp_idx in range(n_exp):
            # Iterate each test sequence in the experiment
            for seq_idx_in_exp in range(self.raw_insights[exp_idx].get_n_seq()):
                # Populate gain
                pre_gain.append(self.raw_insights[exp_idx].gain[seq_idx_in_exp])
                # Populate delta
                pre_delta.append(self.raw_insights[exp_idx].delta[seq_idx_in_exp])
                # Populate knn_insights
                # Number of updates of the sequence
                n_upd_i = self.raw_insights[exp_idx].get_n_upd(seq_idx_in_exp)
                # total calcs absolute
                total_i = np.array(self.raw_insights[exp_idx].get_knn_insight_for_all_updates(seq_idx_in_exp, "total_calcs"))
                total_abs_i = np.cumsum(total_i, axis=1)  # sum over columns
                pre_knn_total_abs[i][:n_upd_i] = total_abs_i
                # total calcs cumulative
                total_cumsum_i = helper.cumsum_array(total_i, absolute=True)  # sum over both axes
                pre_knn_total_cumsum[i][:n_upd_i] = total_cumsum_i
                # actual calcs absolute by escalating the calculation numbers to later kNN members
                actual_abs_i = np.array([helper.escalate_list(l) for l in self.raw_insights[exp_idx].get_knn_insight_for_all_updates(seq_idx_in_exp, "actual_calcs")])
                pre_knn_actual_abs[i][:n_upd_i] = actual_abs_i
                # actual calcs cumulative
                actual_cumsum_i = np.cumsum(actual_abs_i, axis=0)  # just sum over rows for each of the ki columns, do not cumsum along both axes.
                pre_knn_actual_cumsum[i][:n_upd_i] = actual_cumsum_i
                # found stage ids
                stage_id_i = np.array(self.raw_insights[exp_idx].get_knn_insight_for_all_updates(seq_idx_in_exp, "found_stage_id"))
                pre_knn_found_stage_id[i][:n_upd_i] = stage_id_i
                # last calculation stage ids
                last_calc_stage_id_i = np.array(self.raw_insights[exp_idx].get_knn_insight_for_all_updates(seq_idx_in_exp, "last_calc_stage_id"))
                pre_knn_last_calc_stage_id[i][:n_upd_i] = last_calc_stage_id_i
                # case ids
                case_id_i = np.full((dim_upd, dim_k), np.nan, dtype=object)
                case_id_i[...] = np.array(self.raw_insights[exp_idx].get_knn_insight_for_all_updates(seq_idx_in_exp, "case_id"))
                pre_knn_case_id[i][:n_upd_i][...] = case_id_i
                # (calc, sim) evolution histories of kNN members
                calc_sim_history_i = np.full((dim_upd, dim_k), np.nan, dtype=object)
                calc_sim_history_i[...] = np.array(self.raw_insights[exp_idx].get_knn_insight_for_all_updates(seq_idx_in_exp, "calc_sim_history"))
                pre_knn_calc_sim_history[i][:n_upd_i][...] = calc_sim_history_i
                i += 1
        # 2) FINAL PROCESSING
        insights_processed = ExpInsightsProcessed(
            gain=self._insert_upd_id(pre_gain),
            delta=self._insert_upd_id(pre_delta),
            knn_total_cumsum=self._insert_upd_id(pre_knn_total_cumsum),
            knn_total_abs=self._insert_upd_id(pre_knn_total_abs),
            knn_actual_cumsum=self._insert_upd_id(pre_knn_actual_cumsum),
            knn_actual_abs=self._insert_upd_id(pre_knn_actual_abs),
            knn_found_stage_id=self._insert_upd_id(pre_knn_found_stage_id),
            knn_last_calc_stage_id=self._insert_upd_id(pre_knn_last_calc_stage_id),
            knn_case_id=self._insert_upd_id(pre_knn_case_id),
            knn_calc_sim_history=self._insert_upd_id(pre_knn_calc_sim_history))

        return insights_processed


class ExpInsightsSettings:
    """Settings used in the insights experiment.

    Attributes:
        dataset (str): Full path of the dataset "arff" file
            tw_width (int): if > 0, width of the 'moving' time window;
                otherwise, 'expanding' time window approach is applied.
            tw_step (int): number of steps (in terms of data points in TS) taken at each update.
                This can also be seen as the number of data points changed at each update.
            k (int): k of kNN
            test_size (float): (0., 1.) ratio of the time series dataset to be used as Test CB
            cb_size (int): Number of cases in the Train CB
            cls_rank_iterator (str): Used RankIterator class's name

    """
    def __init__(self, dataset, tw_width, tw_step, k, test_size, cb_size, cls_rank_iterator, cls_rank_iterator_attrs):
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
            cls_rank_iterator (str): Used RankIterator class's name
            cls_rank_iterator_attrs (dict): Used RankIterator class's class attributes, if any.

        """
        self.dataset = dataset
        self.tw_width = tw_width
        self.tw_step = tw_step
        self.k = k
        self.test_size = test_size
        self.cb_size = cb_size
        self.cls_rank_iterator = cls_rank_iterator
        self.cls_rank_iterator_attrs = cls_rank_iterator_attrs


class ExpInsightsOutput(exp_common.Output):
    """Holds and saves experiment settings and processed result data"""

    def __init__(self, settings, data):
        """Overrides the parent method only to specify type annotations.

        Args:
            settings (ExpInsightsSettings):
            data (ExpInsightsProcessed):

        """
        # super(ExpInsightsOutput, self).__init__(settings, data)  # commented to help IDE auto-fill below attributes
        self.settings = settings
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
            out_file = gen_insights_ouput_f_path(self.settings.dataset, self.settings.tw_width, self.settings.tw_step,
                                                 self.settings.test_size)
        common.dump_obj(self, out_file)
        logger.info("Anytime Lazy KNN - Insights experiment output dumped into '{}'.".format(out_file))
        return out_file


class ExpInsightsEngine:
    """Insights experiment engine"""

    def __init__(self, cb, k, similarity, cls_rank_iterator, test_size=0.01, n_exp=1):
        """

        Args:
            cb (cbr.TCaseBase):
            k (int): k of kNN.
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            cls_rank_iterator (type): The `RankIterator` *sub-class* of choice to find kNN candidates
                within the `Rank` of the `Sequence`
            test_size (float): ratio of the number of test sequences to be separated from the `cb`.
            n_exp (int) = number of experiments to repeat

        """
        self.cb = cb
        self.k = k
        self.similarity = similarity
        self.cls_rank_iterator = cls_rank_iterator
        self.test_size = test_size
        self.n_exp = n_exp
        self.all_exp_insights_raw = None  # type: List[ExpInsightsRaw]

        logger.info("Insights experiment engine created")

    def run(self):
        """Runs a total of `n_exp` insights experiments.

        Returns:
            ExpInsightsProcessed:

        """
        self.all_exp_insights_raw = []
        for exp_id in range(self.n_exp):
            logger.info(".. Experiment {} of {}".format(exp_id + 1, self.n_exp))
            CB_train, CB_test = exp_common.split_cb(self.cb, self.test_size)
            exp_insights_raw = ExpInsightsRaw()  # Create a new insights object for the new experiment
            len_test = len(CB_test)
            CB_train = cbr.TCaseBase(cb=CB_train)  # This will be passed to Anytime Lazy KNN, not the `ExpInsightsEngine.cb`
            for idx, sequence in enumerate(CB_test):
                logger.info(".... Testing with problem sequence {} of {} (seq_id: {})".format(idx + 1, len_test, sequence.seq_id))
                exp_insights_raw.add_new_sequence()  # Save insights of each sequence separately
                solve_sequence = exp_common.SolveSequence(CB_train, self.k, sequence, self.similarity, self.cls_rank_iterator, exp_insights_raw)
                solve_sequence.solve()  # All sequence updates w/o interruption and collect insights data
                del solve_sequence  # Help garbage collector to release the memory as soon as possible
            self.all_exp_insights_raw.append(exp_insights_raw)
        processed_insights = ProcessExpInsightsRaw(self.all_exp_insights_raw).process()
        return processed_insights


def gen_insights_ouput_f_path(dataset, tw_width, tw_step, k, test_size, key_cls_rank_iter, suffix=""):
    """Returns full path of the output file for the insights experiment results"""
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]  # Base file name w/o extension
    out_file = os.path.join(common.APP.FOLDER.RESULT,
                               "INS_{d}_w_{w}_s_{s}_k_{k}_t_{t}_{i}{x}{e}".format(
                                   d=dataset_name, w=tw_width, s=tw_step, k=k, t=str(test_size), i=key_cls_rank_iter, x=suffix, e=common.APP.FILE_EXT.PICKLE))
    return out_file

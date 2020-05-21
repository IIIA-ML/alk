"""Library of common functionality for experimentation with Anytime Lazy KNN"""

import logging
from typing import Tuple, List  # For type hints

import numpy as np
from sklearn.model_selection import train_test_split

from alk import alk, common


logger = logging.getLogger("ALK")


class Output:
    """Base class for experiment results and PDPs to be saved"""

    def __init__(self, settings, data):
        self.settings = settings
        self.data = data

    def save(self):
        """Saves the object itself into a file."""
        raise NotImplementedError("'save' method is not implemented for the '{}' class yet.".format(self.__class__.__name__))


class SolveSequence:
    """Conduct interrupted and resumed kNN search for all updates of a particular problem sequence.

    Attributes:
        cb (cbr.TCaseBase):
        k (int): k of kNN
        sequence (Sequence): A problem sequence with multiple updates
        similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
        exp_insights (ExpInsightsRaw): Optional. If given; (gain, delta, knn_insights) data are appended.
        anytime_lazy_knn (alk.AnytimeLazyKNN): `AnytimeLazyKNN` instance for the `sequence`.
        start_update (int): The update index to resume from after interruption.

    """
    def __init__(self, cb, k, sequence, similarity, rank_iterator, exp_insights=None):
        """Initialize a SolveSequence object

        Args:
            cb (cbr.TCaseBase):
            k (int): k of kNN
            sequence (cbr.Sequence): A problem sequence with multiple updates
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            rank_iterator (RankIterator): Instance of the `RankIterator` *sub-class* of choice to find kNN candidates
                within the `Rank` of the `Sequence`
            exp_insights (ExpInsightsRaw): Optional. If given; (gain, delta, knn_insights) data are appended.

        """

        # TODO (OM, 20204020): Convert some of the below to private attrs
        self.k = k
        self.cb = cb
        self.sequence = sequence
        self.similarity = similarity
        self.exp_insights = exp_insights
        self.anytime_lazy_knn = alk.AnytimeLazyKNN(seq_id=sequence.seq_id, cb=cb, k=k, similarity=similarity,
                                                   rank_iterator=rank_iterator)
        self.start_update = 0

    def solve(self, stop_update=None, stop_calc_at_stop_update=None):
        """

        Args:
            k (int): The k of kNN
            stop_update (int): The update in which the knn search will be stopped;
                if None, the knn search is not interrupted.
            stop_calc_at_stop_update (int): if > 0, no of calcs to stop the kNN search at the `stop_update`;
                if None, the knn search is not interrupted.

        Returns:
            Tuple[List[alk.Assessment], common.APP.INTRPT, float]: (kNN, INTRPT, calc_pct)
                - kNN : best-so-far/exact kNNs of the `stop_update`;
                - interruption flag, see `INTRPT` Enumeration;
                - percentage of the total number of conducted similarity calculations with regard to
                    the number of calculations that would have been carried out by a brute search
                    between the `stop_update` and `start_update` of the problem `sequence`.

        """
        # TODO (OM, 20200420): `reset` arg is removed. Consider again when `CBR.reset_execution_data()` deemed necessary
        total_calcs = 0
        cb_size = self.cb.size()
        logger.debug("...... SolveSequence.solve() started: k: {}, start_update: {}, start_calc: {}, stop_update: {}, stop_calc_at_stop_update: {}:".format(
            self.k, self.start_update, self.anytime_lazy_knn.get_upd_calcs(), stop_update, stop_calc_at_stop_update))
        if stop_update is None:
            stop_update = self.sequence.n_profiles() - 1  # Last update
        for u in range(self.start_update, stop_update + 1):
            query = self.sequence.profile(u)
            interrupt = stop_calc_at_stop_update if u == stop_update else None
            # Call Anytime Lazy KNN
            _, is_interrupted = self.anytime_lazy_knn.search(query=query, interrupt=interrupt)  # Call Anytime Lazy KNN
            calcs = self.anytime_lazy_knn.get_upd_calcs()
            total_calcs += calcs
            calcs_pct = (calcs / cb_size) * 100.  # % of calcs compared to a brute search
            if self.exp_insights is not None and is_interrupted == common.APP.INTRPT.NONE:
                # Add these insights only after the kNN search for an update is completed
                self.exp_insights.add_insights_for_cur_sequence_update(gain=100. - calcs_pct,
                                                                       delta=self.anytime_lazy_knn.get_upd_delta(),
                                                                       knn_insights=self.anytime_lazy_knn.get_upd_knn_insights())
        total_calcs_brute = cb_size * (stop_update + 1 - self.start_update)
        total_calcs_pct = (total_calcs / total_calcs_brute) * 100.0
        logger.debug("......... Total similarity calculations: {:d} ({:5.2f}% of {:d}).".format(total_calcs, total_calcs_pct, total_calcs_brute))
        # Set the start_update for next call
        if is_interrupted == common.APP.INTRPT.NONE:
            self.start_update = stop_update + 1  # NO Interruption: start from the next update
        else:
            self.start_update = stop_update  # Interruption: Next time, resume from the update where you had left
        logger.debug("..... SolveSequence.solve() finished: start_update: {}, start_calc: {}, interrupted: {}.".format(
            self.start_update, self.anytime_lazy_knn.get_upd_calcs(), is_interrupted))
        return self.anytime_lazy_knn.get_knn(self.k), is_interrupted, total_calcs_pct


def get_setting(out_file, attr):
    """Returns the given setting attribute of an output object.

    Args:
        out_file (str): Full path to the output file
        attr (str): name of the setting attribute

    Raises:
        AttributeError: If the given attribute is not within the settings

    Returns:
        Any: the value of the setting.

    """
    output = common.load_obj(out_file)  # type: Output
    try:
        return getattr(output.settings, attr)
    except AttributeError:
        raise AttributeError("'{}' is not found in the settings.".format(attr))


def split_cb(cb, test_size):
    """Splits the CB to CB_train and CB_test

    Args:
        cb (cbr.TCaseBase):
        test_size (float): defines the size of the CB_test

    Returns:
        Tuple[List[cbr.Sequence], List[cbr.Sequence]]: (CB_train, CB_test)

    """
    CB_train, CB_test, _, _ = train_test_split(cb.sequences(), np.zeros(len(cb)), test_size=test_size)
    return CB_train, CB_test

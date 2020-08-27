"""Anytime Lazy KNN core objects"""

import copy
import logging
from collections import UserList
from typing import List  # For type hints

from alk import cbr, common, rank


logger = logging.getLogger("ALK")


class InitialSearch:
    """Base class for the initial search used by `AnytimeLazyKNN`"""
    def __init__(self, cb, similarity):
        """

        Args:
            cb (cbr.TCaseBase):
            similarity (Callable[[array-like, array-like], float]): a normalized similarity measure that should return a `float` in [0., 1.]

        """
        self.cb = cb
        self.similarity = similarity

    def search(self, query):
        """Conducts a kNN search for the first stage of the `Rank`

        Args:
            query: The query formed by an update of the problem sequence

        Returns:
            (rank.Stage, int): (`rank.Stage`, calcs) tuple where
                `Stage` should be sorted in descending order of `Assessment` similarities, and
                calcs is the total number of similarity assessments carried out

        """
        raise NotImplementedError("Subclasses of `InitialSearch` should implement the `search` method")


class LinearSearch(InitialSearch):

    def search(self, query):
        """Conducts a brute force kNN search for the first stage of the `Rank`

        Args:
            query: The query formed by an update of the problem sequence

        (rank.Stage, int): (`rank.Stage`, calcs) tuple where
                `Stage` is sorted in descending order of `Assessment` similarities, and
                calcs is the total number of similarity assessments carried out

        """
        calcs = self.cb.size()
        nn = []
        for sequence in self.cb.sequences():
            for upd_id in range(sequence.n_profiles()):
                case_id = cbr.CaseId(sequence.seq_id, upd_id)
                nn.append(rank.Assessment(case_id=case_id,
                                          sim=self.similarity(query, sequence.profile(upd_id)),  # or self.cb.get_case_query(case_id)
                                          calcs_no=calcs,
                                          found_stage=None))

        stage = rank.Stage(nn=nn, delta=0)
        stage.sort()
        return stage, calcs


class NNInsights:
    """Insights for the assessment of a particular kNN member.

    Updated every time a different NN replaces this kNN member.

    Attributes:
            case_id (cbr.CaseId): the case id of the current NN for this this kNN member
            total_calcs (int): total number of sim assessments made for this kNN member
            actual_calcs (int): actual number of sim assessments made for this kNN member,
                i.e. after which the NN has not changed although the rank iteration for this NN had to
                continue to ascertain its exactness
            found_stage_id (int): In which stage this NN is found
            last_calc_stage_id (int): the stage where the last sim assessment is made for this NN
            calc_sim_history ( `list` of `(int, float)`): list of (calc, sim) values representing the
                evolution of the best-so-far NN that this history belongs to.

    """
    def __init__(self, case_id=None, total_calcs=0, actual_calcs=None, found_stage_id=None,
                 last_calc_stage_id=None, calc_sim_history=None):
        """Initialize a NNInsights object.

        Args:
            case_id (cbr.CaseId): the case id of the NN
            total_calcs (int): total number of sim assessments made for this NN
            actual_calcs (int): actual number of sim assessments made for this NN,
                i.e. after which the NN has not changed although the rank iteration for this NN had to
                continue to ascertain its exactness
            found_stage_id (int): In which stage this NN is found
            last_calc_stage_id (int): the stage where the last sim assessment is made for this NN
            calc_sim_history (`list` of `(int, float)`): list of (calc, sim) values representing the
                evolution of the best-so-far NN that this history belongs to.

        """
        self.case_id = case_id
        self.total_calcs = total_calcs
        self.actual_calcs = actual_calcs
        self.found_stage_id = found_stage_id
        self.last_calc_stage_id = last_calc_stage_id
        if calc_sim_history is None:
            self.calc_sim_history = [(0, 0.)]
        else:
            self.calc_sim_history = calc_sim_history

    def get(self, insight):
        """Gets the value of the given insight keyword.

        Args:
            insight (str): The name of the corresponding instance attribute

        """
        if insight not in common.public_attrs(self):
            raise AttributeError("{} is not within the accessible insights attributes: {}.".format(insight, common.public_attrs(self)))
        else:
            return self.__dict__[insight]

    def set(self, **kwargs):
        """Sets the values of the given insight keywords

        Given keyword arguments must match defined 'public' instance attribute names.

        Args:
            **kwargs: key=value pairs for insight attributes

        Returns:
            None:

        Examples:
            .set(case_id=..., total_calcs=1000, actual_calcs=10, ...)

        Raises:
            AttributeError: If a non-existing or private attribute is tried to be set.

        """
        common.set_public_attrs(self, **kwargs)

    def add_history(self, calc, sim):
        """Add (calc, sim) to the history of this NN.

        Note:
            Used when the current kNN member is replaced by a nearer neighbor.
            Adds if `calc` does not already exist in the history.

        Args:
            calc (int): number of calc for the new NN
            sim (float): similarity of the new NN to the query

        Returns:
             List[Tuple[int, float]]: new `calc_sim_history`
        """
        if calc not in [c for c, _ in self.calc_sim_history]:
            self.calc_sim_history.append((calc, sim))
        return self.calc_sim_history


class KNNInsights(UserList):
    """A wrapper class simulating a built-in `list` for the insights info of all kNN members.

    Exhibits a list-like behaviour to read items. So, it's possible to access it like _upd_knn_insights[0];
    But, NOT the other way around: _upd_knn_insights[0] = ... is not allowed.

    Attributes:
        query_id (cbr.CaseId): The id of the query formed by the corresponding sequence.

    """
    def __init__(self, k, query_id=None):
        """Initialize a KNNInsights object

        Args:
            k (int): k of kNN.
            query_id (cbr.CaseId): (optional) The id of the query formed by the corresponding sequence
        """
        self.query_id = query_id
        self.data = [NNInsights() for _ in range(k)]  # Intrinsic attribute that must be populated

    def __repr__(self):
        return "KNNInsights(|nn|:{}, nn:{})".format(
            len(self.data), str([nn.case_id for nn in self.data]))

    def get(self, insight):
        """Gives the `insight` values of all kNN members.

        Args:
            insight (str): The name of the corresponding instance attribute

        Returns:
            list: List of requested insight data values of all kNN members

        """
        return [nni.get(insight) for nni in self.data]


class AnytimeLazyKNN:
    """Anytime Lazy KNN, a fast interruptable and resumable kNN search.

    An instance should be instantiated for each problem sequence.

    Attributes:
        seq_id (int):
        cb (TCaseBase):
        k (int): k of kNN
        similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
        initial_nns (InitialSearch): `InitialSearch` implementation used for the initial problem of the sequence.
        _upd_knn_insights (KNNInsights): insights data of kNN assessments during the search for an update
        _query: The query for the last search call
        _rank (rank.Rank):
        _rank_iterator (rank.RankIterator): Instance of the `RankIterator` *sub-class* of choice to find kNN candidates
                within the `Rank` of the `Sequence`
        _upd_calcs (int): Number of similarity calculations made during the kNN search for a particular update

    """
    def __init__(self, seq_id, cb, k, similarity, rank_iterator, initial_nns=LinearSearch):
        """Initialize an ALK object for a particular sequence.

        Args:
            seq_id (int):
            cb (TCaseBase):
            k (int): k of kNN
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            rank_iterator (rank.RankIterator): Instance of the `RankIterator` *sub-class* of choice to find kNN candidates
                within the `Rank` of the `Sequence`
            initial_nns (InitialSearch): `InitialSearch` implementation used for the initial problem of the sequence.

        """
        self.seq_id = seq_id
        self.cb = cb  # type: cbr.TCaseBase
        self.k = k
        self.initial_nns = initial_nns
        self.similarity = similarity
        self._upd_knn_insights = None  # holds insights info of the NN search for each kNN member for an update
        self._query = None
        self._rank = None
        self._rank_iterator = rank_iterator
        self._upd_calcs = 0
        self._upd_delta = 0
        self._upd_sorts = 0  # For debugging purposes, sorts made for a particular update

    def _distance(self, a, b):
        """Normalized distance: 1 - sim(a, b) -> [0, 1]"""
        return 1 - self.similarity(a, b)

    def search(self, query, interrupt=None):
        """Dispatcher function for initial, consecutive and resumed kNN search

        Args:
            query: The query for the initial problem of the sequence # TODO (OM, 20200420): A query/problem class ?
            interrupt (int): Optional interruption point given as the number of similarity calculations for fresh and
                resumed searches for updates > 0. When resuming the search, this number should be cumulative.
                e.g. if you previously interrupted the search at 100 calcs and now you want to resume and run for
                100 calcs more; pass `interrupt` as 200, not 100.
        Returns:
            (List[rank.Assessment], INTRPT):
                 - kNNs of the query;
                 - interruption flag, see `INTRPT` Enumeration
        """
        if self._rank is None:  # Initial problem
            if interrupt is not None:
                raise ValueError("kNN search for the initial problem cannot be interrupted.")
            return self._initial_search(query)
        else:  # Following updates
            # Validation of input params and current state
            self._validate_upon_start(query, stop_calc=interrupt)
            if not self._rank.is_being_iterated():  # Fresh search
                output = self._consecutive_search(query, stop_calc=interrupt)
            else:  # Resumed search
                output = self._resume_last_search(stop_calc=interrupt)
            # Validation of output and current state
            self._validate_upon_exit()
            return output

    def _initial_search(self, query):
        """kNN search for the initial problem of the sequence.

        Args:
            query: The query for the initial problem of the sequence

        """
        logger.debug("........ Sequence update 0 (Initial problem)")
        initial_stage, initial_calcs = self.initial_nns(self.cb, self.similarity).search(query)
        initial_stage.sort()  # Just in case, it won't sort if it is already sorted
        self._upd_sorts += 1
        # Initialize the Rank for initial_knn_search this problem sequence
        self._rank = rank.Rank(seq_id=self.seq_id, initial_stage=initial_stage)
        # Set the `rank` of the iterator
        self._rank_iterator.set_rank(rank=self._rank)
        self._upd_calcs = initial_calcs
        self._upd_knn_insights = KNNInsights(self.k)  # Create a new insights object for the initial problem
        # Add kNN[0] insights to _upd_knn_insights: use only kNN[0] for total_calcs in 0^th update.
        self._upd_knn_insights[0].set(case_id=initial_stage[0].case_id, total_calcs=self._upd_calcs, actual_calcs=self._upd_calcs,
                                      found_stage_id=None, last_calc_stage_id=None,
                                      calc_sim_history=[(0, 0.), (self._upd_calcs, initial_stage[0].sim)])
        # Add insights of the rest of the kNN's: for actual calcs, continue to use the same value for all ki's.
        for nn_idx in range(1, self.k):
            self._upd_knn_insights[nn_idx].set(case_id=initial_stage[nn_idx].case_id, total_calcs=0, actual_calcs=self._upd_calcs,
                                               found_stage_id=None, last_calc_stage_id=None,
                                               calc_sim_history=[(0, 0.), (self._upd_calcs, initial_stage[nn_idx].sim)])
        self._query = query
        self._upd_delta = 0
        return self._rank.knn(k=self.k), common.APP.INTRPT.NONE

    def _consecutive_search(self, query, stop_calc=None):
        """Called for every new update to the sequence.

        Args:
            query: The query for the new problem update to the sequence
            stop_calc (int): Optional interruption point given as the number of similarity calculations

        """
        self._upd_delta = self._distance(self._query, query)
        logger.debug("........ Sequence update {:d} (delta: {:.3f})".format(self._rank.n_cur_updates(), self._upd_delta))
        self._rank_iterator.new_update()  # Reset the iterator for the new update
        self._rank.create_new_stage(delta=self._upd_delta)  # This is the only place where we create a new stage
        self._query = query
        self._upd_knn_insights = KNNInsights(self.k)  # Create a new insights object for each fresh update
        self._upd_sorts = 0
        self._upd_calcs = 0
        return self._lazy_core(stop_calc=stop_calc)

    def _resume_last_search(self, stop_calc=None):
        """Called to resume the previously interrupted search.

        Args:
            stop_calc (int): Optional *new* interruption point given as the cumulative number of similarity calculations

        """
        return self._lazy_core(stop_calc=stop_calc)

    def _lazy_core(self, stop_calc=None):
        """

        Args:
            stop_calc (int): Optional interruption point given as the number of similarity calculations

        Returns:
            (List[rank.Assessment], INTRPT):
                 - kNNs of the query;
                 - interruption flag, see `INTRPT` Enumeration

        """
        signal_intrpt = common.APP.INTRPT.NONE
        start_nn_idx = self._rank_iterator.nn_idx  # For resuming
        logger.debug(".......... stage: {}, kNN[{}], calcs: {}, rank: {}".format(self._rank_iterator.stage_idx, start_nn_idx, self._upd_calcs, repr(self._rank)))
        # Iterate rank once for each kNN member
        for nn_idx in range(start_nn_idx, self.k):
            # logger.debug("............ knn_insights[{}]-> total: {}, calcs: {}".format(nn_idx, self._upd_knn_insights[nn_idx].total_calcs, self._upd_calcs))
            # Iterate through true candidates
            candidate = None
            for candidate in self._rank_iterator:
                # Actual similarity calculation of the candidate to the query
                candidate.sim = self.similarity(self._query, self.cb.get_case_query(candidate.case_id))
                self._rank_iterator.feedback(candidate)  # Send the updated similarity as feedback
                self._upd_calcs += 1
                self._upd_knn_insights[nn_idx].total_calcs += 1
                # Add the new assessment to Rank
                candidate.calcs_no = self._upd_calcs
                assess_idx = self._rank.add_assessment(assess=candidate, to_beat_idx=nn_idx, k=self.k)
                if assess_idx < self.k:
                    # Update histories of the shifted kNN members
                    for x in range(assess_idx, min(self._rank.n_cur_assess(), self.k)):
                        self._upd_knn_insights[x].add_history(self._upd_calcs, self._rank.cur_stage[x].sim)
                # CALCS interruption check
                if self._upd_calcs == stop_calc:
                    signal_intrpt = common.APP.INTRPT.W_CALC
                    logger.debug("............ breaking at kNN[{}], calcs: {}, |cur_stage|: {}".format(nn_idx, self._upd_calcs, self._rank.n_cur_assess()))
                    break  # INTERRUPT by calc
            # Sort new assessments regardless of interruption
            actual_sort = self._rank.cur_stage.sort()
            self._upd_sorts = self._upd_sorts + (1 if actual_sort else 0)
            # Update _upd_knn_insights for kNN[nn_idx]
            self._upd_knn_insights[nn_idx].set(case_id=self._rank.cur_stage[nn_idx].case_id,
                                               actual_calcs=self._rank.cur_stage[nn_idx].calcs_no,
                                               found_stage_id=self._rank.cur_stage[nn_idx].found_stage,
                                               last_calc_stage_id=candidate.found_stage if candidate else None)
            # Now calcs=total, add the calcs to the exact kNN[nn_idx]'s calcs_sim_history
            self._upd_knn_insights[nn_idx].add_history(self._upd_calcs, self._rank.cur_stage[nn_idx].sim)
            if signal_intrpt == common.APP.INTRPT.W_CALC:
                break  # INTERRUPT by calc
            logger.debug("............ knn_insights[{}]-> total: {}, calcs: {}, sorts: {}".format(nn_idx, self._upd_knn_insights[nn_idx].total_calcs, self._upd_calcs, self._upd_sorts))
        # Is kNN search completed?
        if signal_intrpt == common.APP.INTRPT.NONE:
            self._rank.knn_completed()
        if self._rank_iterator:
            logger.debug(".......... stage: {}, kNN[{}], calcs: {}, sorts: {}, rank: {}".format(self._rank_iterator.stage_idx, self._rank_iterator.nn_idx, self._upd_calcs, self._upd_sorts, repr(self._rank)))
        return self._rank.knn(k=self.k), signal_intrpt

    def _validate_upon_start(self, query, stop_calc):
        """Validates `rank`status and input parameters upon starting `search`

        Args:
            query
            stop_calc (int):

        Returns:
            bool: if all is OK.

        Raises:
            RuntimeError, ValueError:
        """
        resuming = self._rank.is_being_iterated()
        msg_error = []
        # TODO (OM, 20200419): Revise these validations, extend/reduce if needed
        if resuming and not (self._query is query):
            raise RuntimeError("Cannot continue with a new update. First, you need to resume the previously interrupted kNN search.")
        if stop_calc is not None and 0 < stop_calc <= self._rank.n_cur_assess():
            msg_error.append("interrupt: {} should be > # of already made evaluations: {}".format(stop_calc, self._rank.n_cur_assess()))
        if msg_error:
            logger.error(str(msg_error))
            raise ValueError(str(msg_error))
        return True

    def _validate_upon_exit(self):
        """Validates internal state upon leaving `search`

        Returns:
            bool: if all is OK.

        Raises:
            RuntimeError:
        """
        msg_error = []
        # TODO (OM, 20200419): Revise these validations, extend/reduce if needed
        if self._rank.is_being_iterated() and self._upd_calcs != self._rank.n_cur_assess():
            msg_error.append("calcs != |rank.cur_stage| on leaving lazy_knn_intrprt upon interruption!")
        if not self._rank.is_being_iterated() and self._upd_calcs != len(self._rank[0]):
            msg_error.append("calcs != |rank.stages[0]| on leaving lazy_knn_intrprt after kNN search termination!")
        # for i in range(0, min(self._rank.n_cur_assess(), len(self._upd_knn_insights))):
        #     if self._upd_knn_insights[i].case_id is None:
        #         msg_error.append("_upd_knn_insights[{}] cannot be empty when resuming after {} assessments: {}".format(
        #             i, self._rank.n_cur_assess(), str(self._upd_knn_insights)))
        #         break
        if msg_error:
            logger.error(str(msg_error))
            raise RuntimeError(str(msg_error))
        return True

    def get_knn(self, k):
        """Returns best-so-far/exact kNNs of the current/last sequence update.

        Note:
            Should be used with caution since it may cause costly sorting of the assessments.

        Args:
            k (int): None for all evaluated NNs

        Returns:
            List[rank.Assessment]:
        """
        return copy.copy(self._rank.knn(k=k))  # shallow-copy

    def get_upd_calcs(self):
        """Returns the number of similarity calculations made for the current sequence update till now

        Used as insights raw data by the caller.

        Returns:
            int:

        """
        return self._upd_calcs

    def get_upd_delta(self):
        """Return the delta for the current update

        Used as insights raw data by the caller.

        Returns:
            float:

        """
        return self._upd_delta

    def get_upd_knn_insights(self):
        """Get the insights data of kNN assessments for the current update till now.

        Returns:
            KNNInsights: A shallow-copy of the internal `KNNInsights` object

        """
        return copy.copy(self._upd_knn_insights)  # Shallow-copy


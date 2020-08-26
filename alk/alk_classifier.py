"""Module that extends ALK to ALK Classifier to use for classification tasks"""
import logging
import sys
from collections import Counter
from typing import Any, List, Tuple

from alk import alk, common, rank


logger = logging.getLogger("ALK")


class BaseVote:
    """Base class for classification voting methods"""

    @classmethod
    def _get_votes(cls, competitors, n=None, **kwargs):
        """Makes the voting for the given exact/best-so-far competitor NNs

        Args:
            competitors (List[Tuple[Any, float]]): List of (class, sim) 2-tuples belonging to nearest neighbors.
                        Should be sorted descending with regards to sim values.
            n (int): if None, all votes; otherwise, top `n` votes are returned.

        Returns:
            List[Tuple[Any, float]]: Reverse sorted list of (class label, vote) items.
                  Sorting is made first by vote, then by class (for equal votes), both in descending order

        """
        raise NotImplementedError("Subclasses of `BaseVote` should implement the `_get_votes` method")

    @classmethod
    def is_exact_solution(cls, k, competitors_exact, solution_set):
        """Check if current exact NNs yield an exact solution.

        Checks if the solution with the current exact NNs (NNe) in best-so-far kNNs is equal to
        the solution with the supposed kNNs (kNN_s); kNN_s consists of current exact NNs and supposed exact NNs;
        Supposed exact NNs are the clones of an imaginary best future neighbor (N_s).

        Args:
            k (int): k of kNN
            competitors_exact (List[Tuple[Any, float]]): List of (class, sim) 2-tuples belonging to 'exact' nearest neighbors,
                        (!!! NOT best-so-far kNN !!!) Should be sorted descending with regards to sim values.
            solution_set (list) :  list of all unique classes in the solutions pace

        Returns:
            bool: True, if exact solution is guaranteed; False, otherwise.

        """
        n_exact = len(competitors_exact)
        if n_exact == 0:
            return False  # Why are you even here?
        if k == n_exact:
            return True  # Why are you even here?
        # Get top two exact competitors
        top_two = cls._get_votes(competitors_exact, 2)
        # Get the imaginary best rival
        if len(top_two) > 1:
            if top_two[0][1] == top_two[1][1]:  # Leading and runner_up classes have the same votes
                return False  # There has to be a single leading class for exactness
            runner_up_class = top_two[1][0]
        else:  # len(top_two) == 1
            # Get the runner-up from available solutions in CB
            leading_class = top_two[0][0]
            runner_up_class = (set(solution_set) - {leading_class}).pop()
        # similarity upper-bound set by the furthest exact NN
        sim_ub = competitors_exact[-1][1]
        imaginary_best_rival = (runner_up_class, sim_ub)
        # Clone the best rival for the supposed competitors with best rivals
        competitors_supposed = competitors_exact + [imaginary_best_rival] * (k - n_exact)
        # Check if the exact soln is guaranteed
        return cls.reuse(competitors_exact) == cls.reuse(competitors_supposed)

    @classmethod
    def vote(cls, competitors):
        """Conducts voting among kNN of a target query to label it

        competitors (list): List of (class, sim) 2-tuples belonging to nearest neighbors.
                        Should be sorted descending with regards to sim values.

        Returns:
            Any: The winner class
        """
        winner_class, _ = cls._get_votes(competitors, n=1)[0]  # Top voted class, even if it is the same with the runner-up
        return winner_class


class Plurality(BaseVote):
    """Plurality (a.k.a. Relative Majority) vote among kNN which subsumes 'Simple Majority' vote"""
    # Implement base class' method
    @classmethod
    def _get_votes(cls, competitors, n=None):
        """Plurality voting"""
        classes_ = [comp[0] for comp in competitors]
        votes = Counter(classes_).most_common()  # Don't use 'n' here to avoid problems caused by arbitrarily sorted equal votes
        # Sort first by vote, then by class (for equal votes), both in descending order
        votes = sorted(votes, key=lambda i: (i[1], i[0]), reverse=True)
        return votes[:n]


class DistanceWeighted(BaseVote):

    # Implement base class' method
    @classmethod
    def _get_votes(cls, competitors, n=None, epsilon=sys.float_info.epsilon):
        """Distance-weighted voting

        Args:
            epsilon (float): Used for the distance d in 1/d when d=0 (i.e. sim = 1.0) for a competitor,
                default = sys.float_info.epsilon (i.e. "difference between 1.0 and the least value greater than 1.0
                that is representable as a float")
        """
        votes = {}
        for comp in competitors:
            class_ = comp[0]
            d = 1. - comp[1]  # normalized distance
            vote = 1. / (d if d > 0 else epsilon)  # if sim == 1.0
            if class_ in votes.keys():
                votes[class_] += vote
            else:
                votes[class_] = vote
        votes = sorted(votes.items(), key=lambda i: (i[1], i[0]), reverse=True)
        return votes[:n]


class BaseSolutionConfidence:
    """Base class for solution confidence"""
    @classmethod
    def get_confidence(cls, nn, k):
        """Calculates the confidence for the NNs and kNNs of a query

        nn (:obj:`list` of :obj:`Assessment`): a list of `Assessment` objects.
            Typically, Current stage in `Rank`'s `Stage.nn` would be sent as a parameter.
            k (int): k of kNN

        Returns:
            float:

        """
        raise NotImplementedError("Subclasses of `BaseSolutionConfidence` should implement the `get_confidence` method")


class AnytimeLazyKNNClassifier(alk.AnytimeLazyKNN, confidence_soln=None):
    """Extends ALK to ALK Classifier to use for classification tasks

    An instance should be instantiated for each problem sequence.

    Attributes:
        reuse (BaseVote): The classification method of choice,
        confidence_soln (BaseSolutionConfidence)

    """

    def __init__(self, reuse, confidence_soln=None, **kwargs):
        self.reuse = reuse  # type: BaseVote
        self.confidence_soln = confidence_soln  # type: BaseSolutionConfidence
        super().__init__(**kwargs)  # Invoke `AnytimeLazyKNN` class init with the keyword argument dictionary

    def _nn_to_soln_competitors(self):
        """Form a list of (solution, sim) 2-tuples for the top `n` nearest neighbors

        Args:
            k (int): k of kNN, number of neighbors to be used in voting

        Returns:
            List[Tuple[Any, float]]: list of (solution, sim) 2-tuples

        """
        competitors = []
        knn = self.get_knn(k=self.k)
        for assess in knn:
            case_id = assess.case_id
            soln = self.cb[case_id.seq_id].solution
            competitors.append((soln, assess.sim))
        return competitors

    def suggest_solution(self):
        """Returns the  suggested solution for the best-so-far/exact kNNs and the confidence for this solution.

        Returns:
            Tuple[Any, float]: The (solution, confidence) tuple.
        """
        soln = self.reuse.vote(self._nn_to_soln_competitors())
        conf = self.confidence_soln(self._rank.cur_stage.nn, self.k) if self.confidence_soln is not None else None
        return soln, conf

    def _lazy_core(self, stop_calc=None, stop_w_soln=None):
        """Overrides parent's core method to extend it for classification purposes in ALK Classifier

        Args:
            stop_calc (int): Optional interruption point given as the number of similarity calculations
            stop_w_soln (bool): if True, interrupts when best-so-far kNNs guarantee an exact solution;
                otherwise, exact solution is not checked.

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
            # Check if exact solution can be achieved
            if stop_w_soln and nn_idx < self.k - 1:  # nn_idx=self.k-1 -> The search has already terminated
                # Check if best so far kNNs can guarantee an exact solution
                competitors_exact = self._nn_to_soln_competitors(self.get_knn(nn_idx + 1))  # Exact NNs and their sims
                if self._reuse.is_exact_solution(k, competitors_exact, self.solution_set):
                    logger.debug("............ breaking for * EXACT SOLUTION * at kNN[{}], calcs: {}, |cur_stage|: {}".format(nn_idx, self._upd_calcs, self._rank.n_cur_assess()))
                    logger.debug(".............. competitors were {}".format([(c[0], round(c[1], 3)) for c in  # class and sim
                                                                               self._nn_to_soln_competitors(self.get_knn(nn_idx + 1))]))
                    signal_intrpt = common.APP.INTRPT.W_SOLN  # Raise the interruption flag for interruption w/ exact solution
                    break  # INTERRUPT by exact solution * * * *
        # Is kNN search completed?
        if signal_intrpt == common.APP.INTRPT.NONE:
            self._rank.knn_completed()
        if self._rank_iterator:
            logger.debug(".......... stage: {}, kNN[{}], calcs: {}, sorts: {}, rank: {}".format(self._rank_iterator.stage_idx, self._rank_iterator.nn_idx, self._upd_calcs, self._upd_sorts, repr(self._rank)))
        return self._rank.knn(k=self.k), signal_intrpt


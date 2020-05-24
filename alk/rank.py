"""Module for `Rank`-specific entities and various `RankIterator` implementations"""

import logging

from collections import UserDict, namedtuple, UserList
from typing import Dict, List, NamedTuple, Union

from alk import cbr


logger = logging.getLogger("ALK")


class Assessment:
    """Holds the assessment info for a neighbor of a query update.

    Attributes:
        case_id (cbr.CaseId): (seq_id, upd_id)
        sim (float): Similarity to the problem update related to the stage
        calcs_no (int): number of calcs made till this assessment (inclusive)
        found_stage (int): İndex of the stage where this candidate was found

    """

    __slots__ = ("case_id", "sim", "calcs_no", "found_stage")  # Save memory, avoid __dict__ for attributes

    def __init__(self, case_id, sim, calcs_no=None, found_stage=None):
        """Initiate an Assessment object

        Args:
            case_id (cbr.CaseId): (seq_id, upd_id)
            sim (float): Similarity to the problem update related to the stage
            calcs_no (int): number of calcs made till this assessment (inclusive)
            found_stage (int): İndex of the stage where this candidate was found

        """
        self.case_id = case_id
        self.sim = sim
        # Below attributes are for statistical purposes
        self.calcs_no = calcs_no  # Set by ALK: How many calculations made until this case was assessed for this update
        self.found_stage = found_stage  # Set by RankIterator: In which stage this case was found as a candidate

    def __repr__(self):
        return "Assessment({}, {})".format(self.case_id, round(self.sim, 3) if self.sim is not None else None)


class Stage:
    """Holds the assessments for a particular update of a query sequence.

    Attributes:
        nn (:obj:`list` of :obj:`Assessment`): a list of `Assessment` objects
        delta (float): distance(q^u-1, q^u), distance between the current and previous query update

    """

    def __init__(self, nn=None, delta=0.):
        """Initialize a Stage object

        Args:
            nn (`list` of `Assessment`): a list of `Assessment` objects
            delta (float): distance(q^u-1, q^u), distance between the current and previous query update

        Raises:
            TypeError:

        """
        if nn:
            if not isinstance(nn, list):
                raise TypeError("Expected `list` for `nn`, got {}".format(type(nn)))
            for item in nn:
                if not isinstance(item, Assessment):
                    raise TypeError("Expected `Assessment` for `nn` item, got {}".format(type(item)))
            self.nn = nn[:]  # shallow copy
        else:
            self.nn = []  # Avoid the gotcha for mutable argument default value
        self.delta = delta
        self._is_sorted = False  # True, if the `nn` is sorted (see `add_assessment` and `sort` methods)

    def __getitem__(self, idx):
        """Provides list like behaviour to access `nn`, e.g. stage[0]

        Args:
            idx (int): index of the wanted `Assessment` within `nn`

        Returns:
            Union[Assessment, List[Assessment]]:

        Raises:
            TypeError:  if `idx` is not an integer>=0 or a slice

        """
        if not (isinstance(idx, slice)
                or (isinstance(idx, int) and idx >= 0)):
            raise TypeError("{} is not a valid index".format(str(idx)))
        return self.nn[idx]

    def __repr__(self):
        return "Stage(|nn|:{}, \u0394:{})".format( len(self.nn), round(self.delta, 3) if self.delta is not None else None)

    def __len__(self):
        """Returns the number of assessments in `nn`"""
        return len(self.nn)

    def sort(self, force=False):
        """Sorts the `nn` in a descending order.

        Note:
            If not `force`d, sorting is done only if necessary.

        Args:
            force (bool): If True, does not check the internal `_is_sorted` attribute.

        Returns:
            bool: True, if an actual sorting is made; False otherwise.

        """
        actual_sort = False
        if not self._is_sorted or force:
            self.nn.sort(key=lambda assess: assess.sim, reverse=True)
            self._is_sorted = True
            actual_sort = True
        return actual_sort

    def add_assessment(self, assess, idx=None):
        """Appends/inserts a neighbor to the stage

        Args:
            assess (Assessment): neighbor case w/ sim
            idx (int): If given, inserts at the `idx` index; otherwise appends

        Returns:
            int: The zero-based index of the added `assess` in the `nn`. Note that this index is
                liable to change as new assessments are inserted on top it.
        """
        if idx is not None:
            self.nn.insert(idx, assess)
        else:
            self.nn.append(assess)
            self._is_sorted = False
            idx = len(self.nn) - 1
        return idx

    def pop(self, idx=0):
        """Pop the idx^th assessment from the stage

        Args:
            idx (int): index of the assessment within the stage. Defaults to 0.

        Returns:
            Assessment: requested assessment
        """
        return self.nn.pop(idx)

    def is_empty(self):
        """Return True if the stage_idx^th stage is empty"""
        return not self.nn


class Rank:
    """Holds the Rank of assessments made through out the updates of a particular query sequence.

    Attributes:
        stages (List[Stage]): Each `Stage` holds `Assessments` made for a particular sequence update
        seq_id (int): id of the sequence of temporally related cases that this Rank object belongs to
        cur_stage (Stage): Holds assessments during all kNN[i] iterations for a query update;
            later added to the `stages`
    """

    def __init__(self, seq_id, initial_stage):
        """ Initialize a Rank object fo a particular problem sequence

        Args:
            seq_id (int): id of the sequence of temporally related cases that this Rank object belongs to
            initial_stage (Stage): assessed neighbors of the first query update, sorted descending by their sim
        """
        # TODO (OM, 20200417): Consider having Rank.k
        if not isinstance(initial_stage, Stage):
            raise TypeError("Expected `Stage` for `initial_stage`, got {}".format(type(initial_stage)))
        elif initial_stage.is_empty():
            raise ValueError("First stage of a `Rank` cannot be empty upon initialization.")
        for item in initial_stage:
            if not isinstance(item, Assessment):
                raise TypeError("Expected `Assessment` for `nn` item, got {}".format(type(item)))
        self.seq_id = seq_id
        self.stages = [initial_stage]
        self.cur_stage = None

    def __getitem__(self, idx):
        """Provides list like behaviour to access `stages`, e.g. rank[0]

        Note:
            `Rank.cur_stage` is not accessible with this method.

        Args:
            idx (int): index of the wanted `Stage` within all `stages`

        Returns:
            Stage:

        Raises:
            TypeError: if `idx` is not an integer

        """
        if not(isinstance(idx, int)):  # and idx >= 0): in RankHash the stage index are -ive
            raise TypeError("{} is not a valid index; only an integer index is accepted.".format(idx))
        return self.stages[idx]

    def __repr__(self):
        str_stages = str([repr(stage) for stage in self.stages]) if self.stages is not None else None
        return "Rank(seq:{})-> cur_stage:{}, stages:{}".format(self.seq_id, repr(self.cur_stage), str_stages)

    def __len__(self):
        """Return the number of stages *excluding* `cur_stage` if exists"""
        return len(self.stages)

    def n_cur_updates(self):
        """Return the number of updates so far *including* `cur_stage` if exists"""
        return len(self.stages) + (1 if self.cur_stage is not None else 0)

    def create_new_stage(self, delta):
        """Creates a new stage for the current iteration assessments.

        Called by the RankIterator constructor.

        Args:
            delta(float): distance(q^u-1, q^u)

        Returns:
            None:

        Raises:
             RuntimeError: If `rank` already has a `cur_stage`

        """
        if self.cur_stage is None:
            self.cur_stage = Stage(delta=delta)
        else:
            raise RuntimeError("New stage already exists for a previous unterminated kNN search. "
                               "Call `Rank.knn_completed()` before, if the search is finished.")

    def add_assessment(self, assess, to_beat_idx=0, k=1):
        """Appends/inserts a neighbor to the stage for the ongoing kNN search (i.e. `cur_stage`).

        If `assess.sim` wins over an existing assessment within indices [to_beat_idx, k),
        it is inserted at the index of that assessment; otherwise, it is appended to `cur_stage.nn`.

        Args:
            assess (Assessment): neighbor case w/ sim
            to_beat_idx (int): zero-based index of the existing assessment to beat in `cur_stage.nn`.
                should be < k.
            k (int): k of kNN.

        Note:
            If `assess` wins over an existing assessment, it is inserted at the index of that assessment;
            otherwise, `assess` is appended to `cur_stage.nn`.

        Raises:
            TypeError: If `assess` is not an `Assessment`object
            ValueError: If `k` OR `to_beat_idx` are not valid
            RuntimeError: If `cur_stage` is None

        Returns:
            int: The zero-based index of the added `assess` in the `cur_stage.nn`. Note that this index is
                liable to change as new assessments are inserted on top it.

        """
        if not isinstance(assess, Assessment):
            raise TypeError("Expected `Assessment` for `assess`, got ", type(assess))
        if self.cur_stage is None:
            raise RuntimeError("Attempt to add an assessment to the `cur_stage` before it is created.")
        if not k > 0:
            raise ValueError("'k' has to be > 0, now got {}".format(k))
        if not 0 <= to_beat_idx < k:
            raise ValueError("'to_beat_idx' has to be [0, {0}) for 'k'={0}, now got {1}".format(k, to_beat_idx))
        if to_beat_idx > self.n_cur_assess():
            raise ValueError("'to_beat_idx'={} > |'cur_stage.nn'|={}. Something is wrong.".format(to_beat_idx, self.n_cur_assess()))

        assess_idx = None  # The index of the assess after adding
        # Check if `assess` can win over any current top assessments within index range [to_beat_idx, k)
        if not self.cur_stage.is_empty():
            for competitor_idx in range(to_beat_idx, min(self.n_cur_assess(), k)):
                if assess.sim > self.cur_stage[competitor_idx].sim:
                    assess_idx = competitor_idx
                    break
        if assess_idx is None:  # `assess` cannot win over current top k Assessments
            assess_idx = self.cur_stage.add_assessment(assess)
        else:
            self.cur_stage.add_assessment(assess, assess_idx)
        return assess_idx

    def is_stage_empty(self, stage_idx):
        """Return True if the stage_idx^th stage is empty"""
        return self.stages[stage_idx].is_empty()

    def n_cur_assess(self):
        """Return the number of current assessments made for the ongoing kNN search"""
        return len(self.cur_stage) if self.cur_stage is not None else 0
        # return len(self.cur_stage) if self.cur_stage is not None else None

    def is_being_iterated(self):
        """Return True if there is an iteration going on the Rank"""
        return self.cur_stage is not None

    def pop(self, stage_idx, assess_idx=0):
        """Pop the assess_idx^th assessment from the stage_idx^th stage in Rank

        Args:
            stage_idx (int): index of the stage in `Rank`
            assess_idx (int): index of the assessment inside the given stage

        Returns:
            Assessment: requested assessment
        """
        return self.stages[stage_idx].pop(assess_idx)

    def knn_completed(self):
        """Should be called by Anytime Lazy KNN when kNN search is finished, i.e. when all exact kNNs are found.

        Inserts the sorted new stage to the top of the `Rank.stages`
        """
        if self.cur_stage:  # Update #0 does not create `cur_stage`
            self.cur_stage.sort()
            self.stages.insert(0, self.cur_stage)
            self.cur_stage = None

    def knn(self, k=1):
        """Returns best-so-far/exact kNNs of the current/last sequence update.

        Note:
            Should be used with caution since it may cause costly sorting of the assessments.

        Args:
            k (int): None for all evaluated NNs

        Returns:
            List[Assessment]:
        """
        if self.cur_stage is not None:
            self.cur_stage.sort()
            return self.cur_stage[:k]
        else:
            return self.stages[0][:k]


class RankHash(UserDict):
    """Serves as a hash table to locate an `Assessment` in the `Rank` by its `Assessment.case_id`

    Used by rank iterators that needs to access assessments arbitrarily, e.g. `ExploitCandidatesIterator`.

    Internally, it is a dictionary of `CaseId`=(stage_end_idx, assess_idx) key=value pairs where stage_end_idx is the
    *reverse* index of the stage (i.e., -ive). This is for the sake of not having to update the hash when a new `Stage`
    is added to the `Rank`.

    Notes:
        Hash maintenance method `popped_from_rank`, `new_stage_inserted` have to be called appropriately by the client
        `RankIterator` subclass.

    """

    RankIndex = namedtuple("RankIndex", "stage_end_idx assess_idx")

    def __init__(self, rank):
        """

        Args:
            rank (Rank):
        """
        if not isinstance(rank, Rank):
            raise TypeError("Expected a `Rank` object for `rank`, got {}".format(type(rank)))
        if rank.is_being_iterated():
            raise RuntimeError("{} cannot be instantiated with a `Rank`"
                               "that is currently being iterated.".format(type(self).__name__))
        self.data = {}  # type: Dict[cbr.CaseId, RankHash.RankIndex] # Intrinsic dict object of UserDict
        self.rank = rank
        self._gen_hash()

    def _gen_stage_hash(self, stage_idx):
        """Generates/updates the hash for the given `Stage` in `Rank`

        Args:
            stage_idx (int): >= 0 index of the stage, where 0 is the index of the stage for the last update.

        Returns:
            None

        """
        len_rank = len(self.rank)
        for assess_idx, assess in enumerate(self.rank[stage_idx]):
            self.data[assess.case_id] = RankHash.RankIndex(stage_idx - len_rank, assess_idx)

    def _gen_hash(self):
        """Generates the hash for the `self.rank`

        Returns:
            dict: A dictionary of CaseId=(stage_end_idx, assess_idx) key=value pairs for the rank

        """
        for stage_idx in range(len(self.rank)):
            self._gen_stage_hash(stage_idx)

    def popped_from_rank(self, case_id):
        """Maintains the hash after `Rank.pop()` occurs.

        It decreases the assess_idx values of the cases that are shifted up in position by 1.

        Args:
            case_id (cbr.CaseId): The `case_id` of the popped_from_rank `Assessment` from the `Rank`

        """
        popped = self.data[case_id]
        self.data[case_id] = None  # This value is to be set later by the `new_stage_inserted` method
        len_rank = len(self.rank)
        for assess in self.rank[len_rank + popped.stage_end_idx][popped.assess_idx:]:
            self.data[assess.case_id] = RankHash.RankIndex(self.data[assess.case_id].stage_end_idx,
                                                           self.data[assess.case_id].assess_idx - 1)

    def new_stage_inserted(self):
        """Updates the hash after `RankIterator.new_update()` is called"""
        self._gen_stage_hash(0)  # The new stage is always the 0^th stage in Rank


class RankIterator:
    """Base class for rank iterators.

    A RankIterator instance must be re-initiated for every query update.

    Attributes:
        rank (Rank): `Rank` instance created for a particular sequence of cases
        nn_idx (int): Zero-based ordinal of the iteration to find kNN[nn_idx] in incremental kNN search.
            Should be incremented by the RankIterator after each full rank iteration.
        stage_idx (int): Index of the stage currently being iterated in `Rank.stages`.
            It should be set in the `__iter__` method.
    """

    abbrv = "Not set"  # Abbreviation of the iterator; sub-classes are expected to overwrite it.

    def __init__(self, rank=None):
        """Initialize a RankIterator object

        Args:
            rank (Rank): Optional. `Rank` instance created for a particular sequence of cases

        """
        self.rank = None  # type: Rank  # Although we'll call set_rank(), this line is for the sake of explicit definition of the instance attr inside __init__
        self.set_rank(rank=rank)
        self.nn_idx = 0
        self.stage_idx = None
        self._fback = None  # see `self.feedback` method

    def set_rank(self, rank):
        """Sets the `Rank` of the iterator, if it doesn't already have one.

        Args:
            rank (Rank): `Rank` instance created for a particular sequence of cases

        Raises:
            TypeError: If `rank`is not a `Rank` object
            RuntimeError:
                i) The iterator object already has a `Rank` associated; OR
                ii) If the `Rank` is currently being iterated OR

        """
        if self.rank is not None:
            raise RuntimeError("The iterator object already has a `Rank` associated.")
        if rank is not None:
            if not isinstance(rank, Rank):
                raise TypeError("Expected a `Rank` object for `rank`, got {}".format(type(rank)))
            if rank.is_being_iterated():
                raise RuntimeError("{} cannot be instantiated with a `Rank`"
                                   "that is currently being iterated.".format(type(self).__name__))
        self.rank = rank

    def new_update(self):
        """Resets the iterator for iteration for a new update.

        Should be called by the `AnytimeLazyKNN._consecutive_search()`.

        Returns:
            None

        Raises:
            RuntimeError: If the `Rank` is currently being iterated.

        """
        if self.rank.is_being_iterated():
            raise RuntimeError("Cannot reset iterator for a new update while the `Rank` is still being iterated. "
                               "Call `Rank.knn_completed()` before, if the search is finished.")
        self.nn_idx = 0
        self.stage_idx = None
        self._fback = None

    def is_candidate(self, assess, stage_idx):
        """Check if a neighbor of a previous query update is a kNN candidate for the current update

        Candidacy assessment is made by leveraging the triangle inequality (see:https://bit.ly/2VxIA0n).

        Args:
            assess (Assessment): An evaluated case for (i.e. neighbor of) a previous update of the query
            stage_idx (int): The stage where the `assess` currently resides

        Returns:
            bool: True if that case is a kNN candidate for the current update; False otherwise.

        """
        # TODO (OM, 20200524): Change the signature to (stage_idx, assess_idx)
        if self.rank.n_cur_assess() <= self.nn_idx:  # No one to beat yet in the cur_stage
            return True
        accum_delta = self.rank.cur_stage.delta
        for j in range(0, stage_idx):  # Sum up the deltas until (excluding) the current stage
            accum_delta += self.rank.stages[j].delta
        return assess.sim + accum_delta >= self.rank.cur_stage[self.nn_idx].sim

    def feedback(self, candidate_assess):
        """To be called by `AnytimeLazyKNN` after assessing the yielded candidate by the RankIterator.

        By default sets `self._fback` to `candidate_assess.sim`.

        If another behaviour is wished, the method should be overridden accordingly by sub-classes
        that demands feedback for the choosing of the next candidate.

        Args:
            candidate_assess (Assessment): Updated Assessment of the yielded candidate for the current query update.

        Returns:
            None.
        """
        self._fback = candidate_assess.sim

    def __iter__(self):
        """Should be implemented by sub-classes.

        Yields:
            Assessment: the previous assessment that contains the candidate case to be re-evaluated
        """
        raise NotImplementedError

    def __repr__(self):
        return "{} Iterator over Rank(seq:{})".format(type(self).__name__, self.rank.seq_id)


class TopDownIterator(RankIterator):
    """Yields candidates in a top-down fashion along the rank stages."""

    abbrv = "td"  # Abbreviation of the iterator

    def __iter__(self):
        logger.debug(".......... {} TopDown RANK iteration for kNN[{}]".format(
            "*resuming*" if self.stage_idx is not None and self.rank.is_being_iterated() else "starting",
            self.nn_idx))
        len_rank = len(self.rank)
        for self.stage_idx in range(len_rank):  # Note! Uses instance attribute as the loop variable to update it.
            while (not self.rank.is_stage_empty(self.stage_idx)
                   and self.is_candidate(self.rank[self.stage_idx][0], self.stage_idx)):
                candidate = self.rank.pop(self.stage_idx, 0)  # Pop the top
                candidate.found_stage = len_rank - self.stage_idx - 1
                yield candidate
        logger.debug("............ TopDown RANK iteration ended for kNN[{}]".format(self.nn_idx))
        # Iteration completed, set the index of the kNN member to beat at the next iteration
        self.nn_idx += 1
        # Reset the iterated stage_idx
        self.stage_idx = None


class JumpingIterator(RankIterator):
    """After every ​n^th candidates, assesses the candidacy of top of the next stage in ​`Rank`

    If that is a true candidate, evaluate it, and continue back with the current stage.
    It is implemented to check if this random behavior improves the gain by reducing the number of candidates.

    """

    abbrv = "j"  # Abbreviation of the iterator

    def __init__(self, jump_at, **kwargs):
        self.jump_at = jump_at
        super().__init__(**kwargs)  # Invoke parent class init with the keyword argument dictionary

    def __iter__(self):
        logger.debug(".......... {} Jumping RANK iteration for kNN[{}] w/ jump_at: {}".format(
            "*resuming*" if self.stage_idx is not None and self.rank.is_being_iterated() else "starting",
            self.nn_idx, self.jump_at))
        if self.jump_at is None or not(isinstance(self.jump_at, int) and self.jump_at > 0):
            raise ValueError("`jump_at` should be an integer > 0, got {}".format(self.jump_at))
        len_rank = len(self.rank)
        for self.stage_idx in range(len_rank):  # Note! Uses instance attribute as the loop variable to update it.
            candidates_in_stage_idx = 0  # counter for the candidates found in stage
            just_jumped = False  # reset the flag for every stage
            while (not self.rank.is_stage_empty(self.stage_idx)
                   and self.is_candidate(self.rank[self.stage_idx][0], self.stage_idx)):
                ready_to_jump = (not just_jumped  # Don't jump if you are already coming from there
                                 and self.stage_idx < len_rank - 1  # There exists a stage to jump to
                                 and candidates_in_stage_idx > 0  # Don't jump too early
                                 and candidates_in_stage_idx % self.jump_at == 0  # Only after every `jump_at` candidates
                                 and not self.rank.is_stage_empty(self.stage_idx + 1)  # Don't jump to a blank stage
                                 and self.is_candidate(self.rank[self.stage_idx + 1][0], self.stage_idx + 1))  # Jump if it's worth
                if ready_to_jump:
                    just_jumped = True
                    candidate = self.rank.pop(self.stage_idx + 1, 0)  # Pop the top from the next stage
                    candidate.found_stage = len_rank - self.stage_idx
                else:
                    just_jumped = False
                    candidate = self.rank.pop(self.stage_idx, 0)  # Pop the top from the currently searched stage
                    candidate.found_stage = len_rank - self.stage_idx - 1
                    candidates_in_stage_idx += 1
                yield candidate
        logger.debug("............ Jumping RANK iteration ended for kNN[{}]".format(self.nn_idx))
        # Iteration completed, set the index of the kNN member to beat at the next iteration
        self.nn_idx += 1
        # Reset the iterated stage_idx
        self.stage_idx = None


class ExploitCandidatesIterator(RankIterator):
    """Exploits updates of approaching NNs that were yielded as candidates.

    Formally:

    If sim(s^i, q^{u}) > sim(s^i, q^{u-1}), then give priority to previous and later ​candidate​ updates of the sequence s

    { s^j | candidate(s^j, q^{u}) ⋀ sim(s^j, q^{u}) >= sim(s^i, q^{u}), for all j in [a, b] } where

    a, b are the indices of the first prior and posterior updates of s, respectively, that give lower similarity than
    sim(s^i, q^{u}).

    Attributes:
        _rank_hash (dict): A `dict` of `(seq_id, upd_id): (stage_end_idx, assess_idx)` key/value pairs where stage_end_idx is the
            reverse index of the stage (i.e., -ive). This for the sake of not having to maintain the hash when a new
            Stage is added to the Rank.
            The dict will be used as a hash table to access an assessment a in `self.rank` by its `a.case_id` as:

            `self.rank.pop(self._rank_hash[a.case_id][0], self._rank_hash[a.case_id][1])`

    """

    abbrv = "xc"  # Abbreviation of the iterator

    class ExploitBag(UserList):
        """Container for the `cbr.CaseId`'s of the updates of an exploited candidate."""
        def __init__(self, exploited=None):
            """
            Args:
                exploited (Assessment): Exploited candidate where `exploited.sim` is its similarity to the current query update
            """
            self.exploited = exploited
            self.data = []  # type: List[cbr.CaseId]

    def __init__(self, rank=None):
        self.rank = None  # type: Rank
        self._rank_hash = None  # type: RankHash
        self._exploit_bag = ExploitCandidatesIterator.ExploitBag()
        super().__init__(rank)
        if self.rank:
            self._set_rank_hash()

    def set_rank(self, rank):
        super().set_rank(rank)
        if self.rank:
            self._set_rank_hash()

    def _set_rank_hash(self):
        """Creates the rank hash"""
        self._rank_hash = RankHash(rank=self.rank)

    def _get_adjacent_case_id(self, case_id, next_=True):
        """Get's the case_id for the next sequence update in the Rank,

        Args:
            case_id (cbr.CaseId):
            next_ (bool): If True, CaseId of the later sequence update is returned; otherwise; of the previous update.

        Returns:
            Union[cbr.CaseId, None]: None is returned if there is no update in the given direction.

        """
        adj_upd_id = case_id.upd_id + (1 if next_ else - 1)
        if adj_upd_id < 0:
            return None
        adj_case_id = cbr.CaseId(case_id.seq_id, adj_upd_id)
        if adj_case_id in self._rank_hash and self._rank_hash[adj_case_id]:  # It exists in the hash and it is not popped from rank
            return adj_case_id
        else:
            return None

    def __iter__(self):
        logger.debug(".......... {} ExploitCandidates RANK iteration for kNN[{}]".format(
            "*resuming*" if self.stage_idx is not None and self.rank.is_being_iterated() else "starting",
            self.nn_idx))
        len_rank = len(self.rank)
        for self.stage_idx in range(len_rank):  # Note! Uses instance attribute as the loop variable to update it.
            while (not self.rank.is_stage_empty(self.stage_idx)
                   and self.is_candidate(self.rank[self.stage_idx][0], self.stage_idx)):
                candidate = self.rank.pop(self.stage_idx, 0)  # Pop the top
                self._rank_hash.popped_from_rank(candidate.case_id)  # Update rank hash
                candidate.found_stage = len_rank - self.stage_idx - 1
                candidate_sim = candidate.sim
                yield candidate  # yield TOP DOWN
                if self._fback > candidate_sim:  # The candidate is closer to this query update than it was to query's previous update
                    # logger.debug("............ Exploiting candidate {}".format(candidate.case_id))
                    # Exploit the candidate
                    self._exploit_bag.exploited = Assessment(candidate.case_id, sim=self._fback)  # Create a tmp Assessment; do not trust the client has set the candidate.sim before calling `feedback()`
                    # Add prev and next updates of the candidate to the exploit bag
                    prev_case_id = self._get_adjacent_case_id(candidate.case_id, next_=False)
                    if prev_case_id:
                        self._exploit_bag.append(prev_case_id)
                    next_case_id = self._get_adjacent_case_id(candidate.case_id, next_=True)
                    if next_case_id:
                        self._exploit_bag.append(next_case_id)
                    while self._exploit_bag:
                        update_case_id = self._exploit_bag.pop(0)
                        update_stage_idx, update_assess_idx = self._rank_hash[update_case_id]
                        if self.is_candidate(self.rank[update_stage_idx][update_assess_idx], update_stage_idx):
                            candidate = self.rank.pop(update_stage_idx, update_assess_idx)  # Pop
                            self._rank_hash.popped_from_rank(candidate.case_id)  # Update rank hash
                            # logger.debug(".............. found a candidate update {}".format(candidate.case_id))
                            yield candidate  # yield EXPLOITATION
                            if self._fback > self._exploit_bag.exploited.sim:
                                # logger.debug("................ exploited update {} has proved even closer.".format(candidate.case_id))
                                logger.debug(".............. Exploited candidate w/ {}'s update w/ {} has proved even closer.".format(self._exploit_bag.exploited.case_id, candidate.case_id))
                                # Add the adjacent update in the same direction of the assessed update to the exploit bag
                                adj_case_id = self._get_adjacent_case_id(candidate.case_id,
                                                                         next_=candidate.case_id.upd_id > self._exploit_bag.exploited.case_id.upd_id)
                                if adj_case_id:
                                    self._exploit_bag.append(adj_case_id)
                    self._exploit_bag.exploited = None  # Leave the bag clean
        logger.debug("............ ExploitCandidates RANK iteration ended for kNN[{}]".format(self.nn_idx))
        # Iteration completed, set the index of the kNN member to beat at the next iteration
        self.nn_idx += 1
        # Reset the iterated stage_idx
        self.stage_idx = None

    def new_update(self):
        super().new_update()
        self._rank_hash.new_stage_inserted()  # Update rank hash for the newly inserted stage



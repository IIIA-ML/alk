"""Anytime Lazy KNN core objects"""

import copy
import logging
from collections import UserList
from typing import List  # For type hints

from alk import common, cbr


logger = logging.getLogger("ALK")


class InitialSearch:
    """Base class for the initial search used by `AnytimeLazyKNN`"""
    def __init__(self, cb, similarity):
        """

        Args:
            cb (cbr.TCaseBase):
            similarity (Callable[[array-like, arrat-like], float]): a normalized similarity measure that should return a `float` in [0., 1.]

        """
        self.cb = cb
        self.similarity = similarity

    def search(self, query):
        """Conducts a kNN search for the first stage of the `Rank`

        Args:
            query: The query formed by an update of the problem sequence

        Returns:
            (Stage, int): (`Stage`, calcs) tuple where
                `Stage` should be sorted in descending order of `Assessment` similarities, and
                calcs is the total number of similarity assessments carried out

        """
        raise NotImplementedError("Sublcasses of `InitialSearch` should implement the `search`method")


class LinearSearch(InitialSearch):

    def search(self, query):
        """Conducts a brute force kNN search for the first stage of the `Rank`

        Args:
            query: The query formed by an update of the problem sequence

        (Stage, int): (`Stage`, calcs) tuple where
                `Stage` is sorted in descending order of `Assessment` similarities, and
                calcs is the total number of similarity assessments carried out

        """
        calcs = self.cb.size()
        nn = []
        for sequence in self.cb.sequences():
            for upd_id in range(sequence.n_profiles()):
                case_id = cbr.CaseId(sequence.seq_id, upd_id)
                nn.append(Assessment(case_id=case_id,
                                     sim=self.similarity(query, sequence.profile(upd_id)),  # or self.cb.get_case_query(case_id)
                                     calcs_no=calcs,
                                     found_stage=None))

        stage = Stage(nn=nn, delta=0)
        stage.sort()
        return stage, calcs


class Assessment:
    """Holds the assessment info for a neighbor of a query update.

    Attributes:
        case (CaseId): (seq_id, upd_id)
        sim (float): Similarity to the problem update related to the stage
        calcs_no (int): number of calcs made till this assessment (inclusive)
        stage_id (int): the stage where this candidate was found

    """

    __slots__ = ("case_id", "sim", "calcs_no", "found_stage")  # Save memory, avoid __dict__ for attributes

    def __init__(self, case_id, sim, calcs_no=None, found_stage=None):
        """Initiate an Assessment object

        Args:
            case (CaseId): (seq_id, upd_id)
            sim (float): Similarity to the problem update related to the stage
            calcs_no (int): number of calcs made till this assessment (inclusive)
            found_stage (int): İndex of the stage where this candidate was found

        """
        self.case_id = case_id
        self.sim = sim
        # Below attributes are for statistical purposes
        # TODO (OM, 20200417): Think on delegating the maintenance of this info to ALK
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
            Assessment:

        Raises:
            ValueError:

        """
        if not (isinstance(idx, slice)
                or (isinstance(idx, int) and idx >= 0)):
            raise ValueError("{} is not a valid index".format(str(idx)))
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
        stages (`list` of `Stage`): Each `Stage` holds `Assessments` made for a particular sequence update
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
        elif not initial_stage.nn:
            raise ValueError("First stage of a `Rank` cannot be empty upon initialization.")
        for item in initial_stage.nn:
            if not isinstance(item, Assessment):
                raise TypeError("Expected `Assessment` for `nn` item, got {}".format(type(item)))
        self.seq_id = seq_id
        self.stages = [initial_stage]
        self.cur_stage = None  # TODO (OM, 20200420): rename -> _cur_stage ?

    def __getitem__(self, idx):
        """Provides list like behaviour to access `stages`, e.g. rank[0]

        Note:
            `Rank.cur_stage` is not accessible with this method.

        Args:
            idx (int): index of the wanted `Stage` within all `stages`

        Returns:
            Stage:

        Raises:
            ValueError:
        """
        if not(isinstance(idx, int) and idx >= 0):
            raise ValueError("{} is not a valid index >= 0".format(idx))
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
            raise RuntimeError("New stage already exists for a previous unterminated kNN search")

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
    def __init__(self, rank, delta):
        """Initialize a RankIterator object

        Args:
            rank (Rank): `Rank` instance created for a particular sequence of cases
            delta (float): distance(q^u-1, q^u)

        Raises:
            TypeError: If `rank`is not a `Rank` object
            RuntimeError: If `rank` already has a `cur_stage`

        """
        if not isinstance(rank, Rank):
            raise TypeError("Expected a `Rank` object for `rank`, got {}".format(type(rank)))
        if rank.cur_stage is not None:
            raise RuntimeError("{} cannot be instantiated with a Rank "
                               "that has a current stage.".format(type(self).__name__))
        self.rank = rank
        self.nn_idx = 0
        self.stage_idx = None
        self.rank.create_new_stage(delta=delta)  # This is the only place where we create a new stage
        self._fback = None  # see `self.feedback` method

    def is_candidate(self, assess, stage_idx):
        """Check if a neighbor of a previous query update is a kNN candidate for the current update

        Candidacy assessment is made by leveraging the triangle inequality (see:https://bit.ly/2VxIA0n).

        Args:
            assess (Assessment): An evaluated case for (i.e. neighbor of) a previous update of the query
            stage_idx (int): The stage where the `assess` currently resides

        Returns:
            bool: True if that case is a kNN candidate for the current update; False otherwise.

        """
        if self.rank.n_cur_assess() <= self.nn_idx:  # No one to beat yet in the cur_stage
            return True
        accum_delta = self.rank.cur_stage.delta
        for j in range(0, stage_idx):  # Sum up the deltas until (excluding) the current stage
            accum_delta += self.rank.stages[j].delta
        return assess.sim + accum_delta >= self.rank.cur_stage[self.nn_idx].sim

    def feedback(self, upd_assess, *args):
        """Should be implemented by sub-classes that demands feedback for the choosing of the next candidate.

        Anytime Lazy KNN sends the *updated* Assessment of the yielded candidate
        after calculating its distance to the current query update.
        And, this function sets the internal `self._fback` attribute accordingly.

        Args:
            upd_assess (Assessment): Updated Assessment of the yielded candidate for the current query update.
            *args: Any positional arguments that is used by the implementing class.

        Returns:
            None.
        """
        pass

    @classmethod
    def set_cls_attr(cls, **kwargs):
        """Sets public 'class' attributes

        Used when a subclass has class attribute(s) that need(s) to be set before creating its instances.
        Typical callees are the `run_~` experiment scripts which call this method with arguments passed from the command
        line.

        Given keyword arguments must match defined 'public' class attribute names.

        Args:
            **kwargs: key=value pairs for class attributes

        Returns:
            None:

        Raises:
            AttributeError: If a non-existing or private attribute is tried to be set.

        """
        common.set_public_attrs(cls, **kwargs)

    def __iter__(self):
        """Should be implemented by sub-classes.

        Yields:
            Assessment: the previous assessment that contains the candidate case to be re-evaluated
        """
        raise NotImplementedError

    def __repr__(self):
        return "{} Iterator over Rank(seq:{})".format(type(self).__name__, self.rank.seq_id)


class TopDownCandidates(RankIterator):
    """Yields candidates in a top-down fashion along the rank stages."""

    def __iter__(self):
        logger.debug(".......... {} TopDown RANK iteration for kNN[{}]".format(
            "*resuming*" if self.stage_idx is not None and self.rank.is_being_iterated() else "starting",
            self.nn_idx))
        for self.stage_idx in range(len(self.rank)):  # Note! Uses instance attribute as the loop variable to update it.
            # while stage_not_empty and (top_case_is_candidate or no_one_to_beat_yet)
            while (not self.rank.is_stage_empty(self.stage_idx)
                   and self.is_candidate(self.rank[self.stage_idx][0], self.stage_idx)):
                candidate = self.rank.pop(self.stage_idx, 0)  # Pop the top
                candidate.found_stage = self.stage_idx
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

    # `jump_at` class attribute has to be set by the callee via `set_cls_attr` class method
    # And, it's advised to be used as read-only by class instances; if overwritten, this would shadow the class attr.
    jump_at = None

    def __iter__(self):
        logger.debug(".......... {} Jumping RANK iteration for kNN[{}]".format(
            "*resuming*" if self.stage_idx is not None and self.rank.is_being_iterated() else "starting",
            self.nn_idx))
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
                    candidate.found_stage = self.stage_idx + 1
                else:
                    just_jumped = False
                    candidate = self.rank.pop(self.stage_idx, 0)  # Pop the top from the currently searched stage
                    candidate.found_stage = self.stage_idx
                    candidates_in_stage_idx += 1
                yield candidate
        logger.debug("............ Jumping RANK iteration ended for kNN[{}]".format(self.nn_idx))
        # Iteration completed, set the index of the kNN member to beat at the next iteration
        self.nn_idx += 1
        # Reset the iterated stage_idx
        self.stage_idx = None


class NNInsights:
    """Insights for the assessment of a particular kNN member.

    Updated every time a different NN replaces this kNN member.

    Attributes:
            case_id (CaseId): the case id of the current NN for this this kNN member
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
            case_id (CaseId): the case id of the NN
            total_calcs (int): total number of sim assessments made for this NN
            actual_calcs (int): actual number of sim assessments made for this NN,
                i.e. after which the NN has not changed although the rank iteration for this NN had to
                continue to ascertain its exactness
            found_stage_id (int): In which stage this NN is found
            last_calc_stage_id (int): the stage where the last sim assessment is made for this NN
            calc_sim_history ( `list` of `(int, float)`): list of (calc, sim) values representing the
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
        query_id (CaseId): The id of the query formed by the corresponding sequence.

    """
    def __init__(self, k, query_id=None):
        """Initialize a KNNINsights object

        Args:
            k (int): k of kNN.
            query_id (CaseId): (optional) The id of the query formed by the corresponding sequence
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
        cls_rank_iterator (type): The `RankIterator` *sub-class* of choice to find kNN candidates
                within the `Rank` of the `Sequence`
        _upd_knn_insights (KNNInsights): insights data of kNN assessments during the search for an update
        _query: The query for the last search call
        _rank (Rank):
        _upd_rank_iter (RankIterator): `cls_rank_iterator` instance created once for each update
        _upd_calcs (int): Number of similarity calculations made during the kNN search for a particular update

    """
    def __init__(self, seq_id, cb, k, similarity, cls_rank_iterator, initial_nns=LinearSearch):
        """Initialize an ALK object for a particular sequence.

        Args:
            seq_id (int):
            cb (TCaseBase):
            k (int): k of kNN
            similarity (Callable): a normalized similarity measure that should return a `float` in [0., 1.]
            cls_rank_iterator (type): The `RankIterator` *sub-class* of choice to find kNN candidates
                within the `Rank` of the `Sequence`
            initial_nns (InitialSearch): `InitialSearch` implementation used for the initial problem of the sequence.

        """
        self.seq_id = seq_id
        self.cb = cb
        self.k = k
        self.initial_nns = initial_nns
        self.similarity = similarity
        self.cls_rank_iterator = cls_rank_iterator
        self._upd_knn_insights = None  # holds insights info of the NN search for each kNN member for an update
        self._query = None
        self._rank = None
        self._upd_start_nn_idx = 0  # The index of the iteration for the kNN[_upd_start_nn_idx] for an update
        self._upd_rank_iter = None  # Will be reset for every sequence update
        self._upd_start_nn_idx = 0  # The index of the kNN member
        self._upd_calcs = 0
        self._upd_delta = 0
        self._upd_sorts = 0  # For debugging purposes sorts made for a particular update

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
            (List[Assessment], INTRPT):
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
        initial_stage, initial_calcs = self.initial_nns(self.cb, self.similarity).search(query)
        initial_stage.sort()  # Just in case, it won't sort if it is already sorted
        self._upd_sorts += 1
        # Initialize the Rank for initial_knn_search this problem sequence
        self._rank = Rank(seq_id=self.seq_id, initial_stage=initial_stage)
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
        self._upd_start_nn_idx = 0
        self._upd_delta = self._distance(self._query, query)
        logger.debug("........ Sequence update {:d} (delta: {:.3f})".format(self._rank.n_cur_updates(), self._upd_delta))
        self._query = query
        self._upd_rank_iter = self.cls_rank_iterator(self._rank, self._upd_delta)  # One iterator instance per update
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
            (List[Assessment], INTRPT):
                 - kNNs of the query;
                 - interruption flag, see `INTRPT` Enumeration

        """
        signal_intrpt = common.APP.INTRPT.NONE
        start_nn_idx = self._upd_rank_iter.nn_idx  # For resuming
        logger.debug(".......... stage: {}, kNN[{}], calcs: {}, rank: {}".format(self._upd_rank_iter.stage_idx, start_nn_idx, self._upd_calcs, repr(self._rank)))
        # Iterate rank once for each kNN member
        for nn_idx in range(start_nn_idx, self.k):
            # logger.debug("............ knn_insights[{}]-> total: {}, calcs: {}".format(nn_idx, self._upd_knn_insights[nn_idx].total_calcs, self._upd_calcs))
            # Iterate through true candidates
            candidate = None
            for candidate in self._upd_rank_iter:
                # Actual similarity calculation of the candidate to the query
                candidate.sim = self.similarity(self._query, self.cb.get_case_query(candidate.case_id))
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
            self._upd_knn_insights[nn_idx].set(case_id=self._rank.cur_stage.nn[nn_idx].case_id,
                                               actual_calcs=self._rank.cur_stage.nn[nn_idx].calcs_no,
                                               found_stage_id=self._rank.cur_stage.nn[nn_idx].found_stage,
                                               last_calc_stage_id=candidate.found_stage if candidate else None)
            # Now calcs=total, add the calcs to the exact kNN[nn_idx]'s calcs_sim_history
            self._upd_knn_insights[nn_idx].add_history(self._upd_calcs, self._rank.cur_stage.nn[nn_idx].sim)
            if signal_intrpt == common.APP.INTRPT.W_CALC:
                break  # INTERRUPT by calc
            logger.debug("............ knn_insights[{}]-> total: {}, calcs: {}, sorts: {}".format(nn_idx, self._upd_knn_insights[nn_idx].total_calcs, self._upd_calcs, self._upd_sorts))
        # Is kNN search completed?
        if signal_intrpt == common.APP.INTRPT.NONE:
            self._rank.knn_completed()
        if self._upd_rank_iter:
            logger.debug(".......... stage: {}, kNN[{}], calcs: {}, sorts: {}, rank: {}".format(self._upd_rank_iter.stage_idx, self._upd_rank_iter.nn_idx, self._upd_calcs, self._upd_sorts, repr(self._rank)))
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
        if stop_calc is not None and stop_calc <= self._rank.n_cur_assess():
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
            List[Assessment]:
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


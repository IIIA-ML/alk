"""Library for CBR-related classes"""

import logging
import math
from collections import UserDict

import numpy as np


logger = logging.getLogger("ALK")


class Sequence:
    """The class to represent a sequence of temporally related cases

    Attributes:
        seq_id (int): sequence id
        _profiles (list): The list of cases  # TODO (OM, 20200501): Rename profile -> case

    """
    def __init__(self, seq_id, profiles, solution=None):
        self.seq_id = seq_id
        self._profiles = profiles
        self.solution = solution

    def n_profiles(self):
        """Number of profiles"""
        return len(self._profiles)

    def profile(self, idx=None):
        """Returns the profile of the idx^th update.

        Args:
            idx (int): The index of the update throughout the history of the problem `Sequence`

        Returns:
            iterable: A list of features defining the query (i.e.) part of the idx^th update

        """
        if idx is None:
            idx = self.idx - 1
        return self._profiles[idx]

    def __repr__(self):
        return "{} (id:{}, |profiles|:{})".format(self.__class__.__name__, self.seq_id, self.n_profiles())


class TSSequence(Sequence):
    """The class to represent a sequence of temporally related cases for a time series

    Attributes:
        seq_id (int): sequence id
        data (iterable): # THIS CORRESPONDS TO THE *** DATA SEQUENCE for the TIME SERIES ***  ITSELF.
        tw_width (int): if > 0, width of the moving time window;
            otherwise expanding windows approach is applied.
        tw_step (int): number of steps (in terms of data points in TS) taken at each update.
            This can also be seen as the number of data points changed in each update.
        gen_profile (callable): (optional) profile generator function with a (updates, tw_width, tw_step) signature
        solution: solution/class of the sequence

    """

    @staticmethod
    def _get_range_profile(i, len_data, tw_width, tw_step):
        """Return the (start, end) indices for the profile in a ts.

        Helper function.

        Args:
            i (int): (zero-based) index of the profile
            len_data (int): Length of time series.
            tw_width (int): if > 0, width of the moving time window; otherwise expanding windows approach is applied.
            tw_step (int): number of steps (in terms of data points in TS) taken by the window at each update.
                This can also be seen as the number of data points changed in each update.
        """
        i_x_step = (i + 1) * tw_step
        start_ = 0 if i_x_step < tw_width or tw_width == 0 else i_x_step - tw_width
        end_ = i_x_step if i_x_step < len_data else len_data
        return start_, end_

    @staticmethod
    def _gen_profile(data, tw_width=0, tw_step=1):
        """Generates the list of updates (cases) for a sequence as moving or expanding time windows.

        Args:
            data (iterable): A sequence of data points like a time series.
            if > 0, width of the 'moving' time window;
                otherwise, 'expanding' time window approach is applied.
            tw_step (int): number of steps (in terms of data points in TS) taken by the window at each update.
                This can also be seen as the number of data points changed at each update.

        Returns:
            list of numpy.ndarray: Each update in the list is a fragment of the time series data
                that is contained by the frame for that particular update.
                Both fixed-tw_width and expanding window approaches generate `math.ceil(len_data / tw_step)` cases

        Examples:
             >>> gen_profile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tw_width=5, tw_step=1)
             [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7],
              [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9, 10]]
             >>> gen_profile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tw_width=0, tw_step=1)
             [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6],
              [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
             >>> gen_profile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tw_width=5, tw_step=2)
             [[0, 1], [0, 1, 2, 3], [1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 6, 7, 8, 9], [7, 8, 9, 10]]
             >>> gen_profile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tw_width=0, tw_step=2)
             [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
             >>> gen_profile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tw_width=5, tw_step=5)
             [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10]]

        """
        # List comprehension version
        # profiles = [np.array(data[0 if i * tw_step < tw_width or tw_width == 0 else i * tw_step - tw_width:
        #                         i * tw_step if i * tw_step < len(data) else len(data)])
        #             for i in range(1, math.ceil(len(data) / tw_step) + 1)]
        # Loop version for less calculation
        profiles = []
        len_data = len(data)
        for i in range(0, math.ceil(len_data / tw_step)):
            start_, end_ = TSSequence._get_range_profile(i, len_data, tw_width, tw_step)
            profiles.append(np.array(data[start_:end_]))
        return profiles

    def __init__(self, data, tw_width=0, tw_step=1, gen_profile=None, solution=None, seq_id=None):
        """

        Args:
            data (iterable): # THIS CORRESPONDS TO THE *** DATA SEQUENCE for the TIME SERIES ***  ITSELF.
            tw_width (int): if > 0, width of the 'moving' time window;
                otherwise, 'expanding' time window approach is applied.
            tw_step (int): number of steps (in terms of data points in TS) taken at each update.
                This can also be seen as the number of data points changed in each update.
            gen_profile (Callable[[List[], int, int], List[numpy.ndarray]]): Function to generate problem profiles
                and/or queries out of a data sequence. Its signature should be (data, tw_width, tw_step).
            solution: solution/class of the sequence
            seq_id (int): sequence id

        """
        self.seq_id = seq_id
        self.solution = solution  # One solution/class for all profiles
        self.data = data  # THIS CORRESPONDS TO THE *** DATA SEQUENCE for the TIME SERIES ***  ITSELF.
        self.tw_width = tw_width
        self.tw_step = tw_step
        if gen_profile is None:
            gen_profile = TSSequence._gen_profile
        self._profiles = gen_profile(self.data, tw_width, tw_step)


class CaseId:
    """Pointer to the actual case in the case base.

    Attributes:
        seq_id (int): Sequence id in CB that the case belongs to
        upd_id (int): id of the sequence update that the case represents (f.k.a. profile id)

    """

    __slots__ = ("seq_id", "upd_id")  # For memory efficiency

    def __init__(self, seq_id, upd_id):
        """Initiate a CaseId object

        Args:
            seq_id (int): Sequence id in CB that the case belongs to
            upd_id (int): id of the sequence update that the case represents (f.k.a. profile id)

        """
        self.seq_id = seq_id
        self.upd_id = upd_id

    def __repr__(self):
        return "CaseId({}, {})".format(self.seq_id, self.upd_id)

    def __eq__(self, other):
        """For case_id_1 == case_id_2 comparison, and for !="""
        return self.seq_id == other.seq_id and self.upd_id == other.upd_id


class TCaseBase(UserDict):
    """A simple temporal case-base.

    The sequences of temporally related cases are held in a `dict` where
    keys are `Sequence.seq_id`s and values are `Sequence`s.

    Attributes:
        data (dict):

    """

    def __init__(self, cb=None):
        """A wrapper class simulating a built-in `list` for the `Sequence`s in the case base.

        Args:
            cb (list of Sequence): If not None, `Sequence.seq_id`'s for all sequences should be non-empty.

        Raises:
            TypeError: `cb` items are not `Sequence` objects
            KeyError: `Sequence.seq_id` is empty for at least one sequence.

        """
        self.data = {}
        if cb is not None:
            for sequence in cb:
                if not isinstance(sequence, Sequence):
                    raise TypeError("Expected `Sequence` for `cb` item, got {}".format(type(sequence)))
                if sequence.seq_id is None:
                    raise KeyError("`seq_id`s of the sequences in the `cb` cannot be empty.")
                self.data[sequence.seq_id] = sequence

    def size(self):
        """Gives the number of cases, i.e. the total number of all updates of all sequences."""
        return sum([sequence.n_profiles() for sequence in self.sequences()])

    def __repr__(self):
        return "CB: {} sequences, total cases: {} ".format(self.__len__(), self.size())

    def get_case_query(self, case_id):
        """Get the query the given sequence update.

        The query is essentially the problem part of the corresponding case.

        Args:
            case_id (CaseId): id of the sequence update that the case represents

        Raises:
            KeyError: The sequence `seq_id` does not match its key in the cb.

        Returns:
            numpy.ndarray: The array of feature values of the case.problem

        """
        sequence = self.data[case_id.seq_id]
        if sequence.seq_id != case_id.seq_id:
            raise KeyError("The sequence id {} does not match its index {} in the cb".format(sequence.seq_id, case_id.seq_id))
        return sequence.profile(case_id.upd_id)

    def sequences(self):
        """Returns a list of all sequences in the CB.

        Returns:
            list of Sequence:

        """
        return list(self.values())

    def solution_set(self):
        """Get the set of distinct solutions (e.g. classes) in the CB.

        Returns:
            set:
        """
        solutions = [sequence.solution for sequence in self.sequences()]
        return set(sorted(solutions))


"""Module for helper functions"""

import ast
import math
import random

import numpy as np


def cumsum_array(a, absolute=True):
    """Cumulative sum of an array either absolute or relative to the 1st col.

    E.g. absolute=True --> Cumsums ``a`` along both axes:

        c0   c1   c2          c0   c1   c2          c0   c1   c2
        ------------          ------------          ------------
    r0| 10    3    3      r0| 10    3    3      r0| 10   13   16
    r1|  8    2    4 ==>  r1| 18    5    7 ==>  r1| 18   23   30
    r2| 16    5    2      r2| 34   10    9      r2| 34   44   53

    E.g. absolute=False --> Sums 1st column along rows and then all rows along
    columns cumulatively:

        c0   c1   c2          c0   c1   c2          c0   c1   c2
        ------------          ------------          ------------
    r0| 10    3    3      r0| 10    3    3      r0| 10   13   16
    r1|  8    2    4 ==>  r1| 18    2    4 ==>  r1| 18   20   24
    r2| 16    5    2      r2| 34    5    2      r2| 34   39   41

    Args:
        a (np.ndarray): A ``numpy`` 2d-array.
        absolute (bool): If True, uses absolute; otherwise, relative cumsum.

    Returns:
        (np.ndarray): A ``numpy`` 2d-array.

    Examples:
        >>> a = np.array([[10, 3, 3], [8, 2, 4], [16, 5, 2]])
        >>> a
        array([[10,  3,  3],
               [ 8,  2,  4],
               [16,  5,  2]])
        >>> cumsum_array(a, absolute=True)
        array([[10, 13, 16],
               [18, 23, 30],
               [34, 44, 53]])
        >>> cumsum_array(a, absolute=False)
        array([[10, 13, 16],
               [18, 20, 24],
               [34, 39, 41]])

    """
    if absolute:
        return np.cumsum(np.cumsum(a[:, ], axis=0)[:, ], axis=1)  # Cumsum along both axes
    else:
        return np.cumsum(  # Cumulative sum across the columns
                        np.concatenate(  # Concatenate processed 1st column with the rest of the columns
                            (np.cumsum(a[:, 0]).reshape(-1, 1),  # The cumsum of the 1st column as an array
                             a[:, 1:]),
                            axis=1),
                        axis=1)


def escalate_list(l):
    """Equalize list items' values until the first leftmost greater value.

   Non-destructive, does not manipulate the list given as an argument.

    Args:
        l (list): A numerical list.

    Returns:
        list:

    Examples:
        >>> escalate_list([2, 3, 1, 5, 4, 8])
        [2, 3, 3, 5, 5, 8]
        >>> escalate_list([2, 3, 1, 5, 4, 0])
        [2, 3, 3, 5, 5, 5]
        >>> escalate_list([8, 3, 1, 5, 4, 0])
        [8, 8, 8, 8, 8, 8]
        >>> escalate_list([1, 2, 3, 4, 5])
        [1, 2, 3, 4, 5]

    """
    # Create a shallow copy
    l_copy = l[:]
    i = 0
    last_i = len(l_copy) - 1
    while i < last_i:
        try:
            # Find the first item greater than the current item
            gt_i = next(ind for ind, val in enumerate(l_copy) if val > l_copy[i])
            # Set the same values until the found item
            l_copy[i:gt_i] = [l_copy[i]] * (gt_i - i)
            # Continue from the found item
            i = gt_i
        except StopIteration:
            # No element found greater than the current item
            # Set the same values until the end of the list
            l_copy[i:] = [l_copy[i]] * (last_i - i + 1)
            break
    return l_copy


def to_arr(points, fill=True):
    """ Convert list of (x, y) tuples to X, Y arrays.

    Args:
        points (list): list of (x, y) tuples where x is a positive integer and y is a numerical value
        fill (bool): if True, treats points as a sparse list and fills the intermediate values.

    Returns:
        (numpy.ndarray, numpy.ndarray): A 2-tuple of 1D `numpy.ndarray`s

    Examples:
        >>> to_arr([(1, .1), (4, .4), (6, .6)], fill=False)
        (array([1, 4, 6]), array([0.1, 0.4, 0.6]))
        >>> to_arr([(1, .1), (4, .4), (6, .6)], fill=True)
        (array([0, 1, 2, 3, 4, 5, 6]), array([0. , 0.1, 0.1, 0.1, 0.4, 0.4, 0.6]))
    """
    if fill:
        X, Y = to_arr_fill(points)
    else:
        X = np.array([p[0] for p in points])
        Y = np.array([p[1] for p in points])
    return X, Y


def to_arr_fill(points):
    """ Convert sparse list of (x, y) tuples to X, Y arrays with filled intermediate values

    Args:
        points (list): list of (x, y) tuples where x is a positive integer and y is a numerical value

    Returns:
        tuple: A 2-tuple of 1D `numpy.ndarray`s

    Examples:
        >>> to_arr_fill([(0, .0), (2, .2), (4, .4), (6, .6)])
        (array([0, 1, 2, 3, 4, 5, 6]), array([0. , 0. , 0.2, 0.2, 0.4, 0.4, 0.6]))
        >>> to_arr_fill([(1, .1), (4, .4), (6, .6)])
        (array([0, 1, 2, 3, 4, 5, 6]), array([0. , 0.1, 0.1, 0.1, 0.4, 0.4, 0.6]))

    """
    max_x = points[-1][0]
    X = np.arange(max_x + 1)
    Y = np.zeros(max_x + 1)
    for point in points:
        x, y = point
        # propagate the intermediate values (trivial method overwriting right values)
        Y[x:] = y
    return X, Y


def eval_str(s):
    """Attempts to convert a string literal to its real type.

    Args:
        s (str):

    Returns:
        Any: A value of an adequate type if possible

    Examples:
        >>> [eval_str(e) for e in ["hi", "-1", "3.14", "True", "[0, 4, 9]", "1/2"]]
        ['hi', -1, 3.14, True, [0, 4, 9], '1/2']

    """
    try:
        s = ast.literal_eval(s)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return s


def cull_arr(a, pct=None, ind=None):
    """Culls an array with a given indices list or by randomly removing items by a given a percentage.

    Args:
        a (array-like): the array-like list to be culled
        pct (float): (0., 1.] the percentage of the items to be randomly removed
        ind (list): The list of items to be removed. When given, `pct` is ignored

    Returns:
        tuple: (np.array, np.array) (Culled `a`, array of the indices of the removed items)

    Raises: ValueError
        When `ind` is None and `pct` is not in (0., 1.]

    """
    if ind is None:
        if pct is None or pct <= 0. or pct > 1.:
            raise ValueError("pct has to be in (0., 1.]")
        ind = np.array(sorted(random.sample(list(range(len(a))), k=int(len(a) * pct))))
    # culled_a = [e for i, e in enumerate(a) if i not in ind]  # Too slow !
    culled_a = np.delete(a, ind)
    return culled_a, ind


def weighted_std(x, w, sample=False):
    """Calculate weighted standard deviation.

    Args:
        x (array-like): values
        w (array-like): weights (i.e. probabilities)
        sample (bool): If True, std dev is calculated for sample; otherwise for population.

    Returns:
        float:

    Notes:
        Ref NIST: http://bit.ly/34BKqR8 (Sample)
        Ref MIT: http://bit.ly/2L7xDhM  (Population)
        Ref WISC: http://bit.ly/2OXemRk (Both)

    """
    xwm = np.average(x, weights=w)  # weighted mean
    if sample and np.count_nonzero(w) > 1:
        n = np.count_nonzero(w)  # the number of non-zero weights
        stdw = math.sqrt(np.sum(w * (x - xwm) ** 2) / ((n - 1) * np.sum(w) / n))
    elif not(sample) or (sample and np.count_nonzero(w) <= 1):  # population
        stdw = math.sqrt(np.sum(w * (x - xwm) ** 2))
    return stdw


def is_float_int(v):
    """Returns integer if the float argument is an integer"""
    return int(v) if int(v) == v else v

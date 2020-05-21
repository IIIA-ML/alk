"""Module with time series related functions"""

import logging
import os

import numpy as np
from scipy.io import arff

from alk import cbr


logger = logging.getLogger("ALK")


def read_ts(dataset):
    """Reads 'arff'-format time series classification data

    Args:
        dataset (str): Full path to the 'arff' file

    Raises:
        TypeError: If dataset file extension is not 'arff'.
        ValueError: If dataset contains missing values.

    Returns:
        (`numpy.ndarray`, `numpy.ndarray`): (instances, classes)

    """
    _, ext = os.path.splitext(dataset)
    if ext != ".arff":
        raise TypeError("For now, only 'arff' files are supported; got '{}'.".format(ext))
    with open(os.path.expanduser(dataset), "r") as f:
        data, _ = arff.loadarff(f)
    # TODO (OM): clean the redundant list<->ndarray code
    TS = np.asarray(data.tolist(), dtype=np.float32)[:, :-1]  # time series instances
    C = np.asarray(data.tolist(), dtype=int)[:, -1]  # target classes
    # Check missing values
    if np.isnan(TS).any() or np.isnan(C).any():
        raise ValueError("The dataset contains instance(s) with missing value(s) which is not handled yet!")
    return TS, C


def euclidean_similarity_ts(a, b, max_, min_):
    """Euclidean similarity for two frames of univariate time series data
    Args:
        a (np.ndarray): 1D array. frame 1
        b (np.ndarray): 1D array. frame 2
        max_ (float): max value a slot can have
        min_ (float): min value a slot can have
    Returns:
        float: [0. 1.]
    Notes:
        If one of the arrays is shorter than the other, the shorter is extended
        so that the abs(a[j]-b[j]) equals 'max_diff' for the extended slots.
    Examples:
        >>> euclidean_similarity_ts(np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1]), max_=10, min_=0)
        1.0
        >>> euclidean_similarity_ts(np.array([5, 5, 5, 5, 5]), np.array([10, 10, 10, 10, 10]), max_=10, min_=0)
        0.5
        >>> euclidean_similarity_ts(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), max_=10, min_=0)
        0.0
        >>> euclidean_similarity_ts(np.array([0, 0, 0]), np.array([10, 10, 10, 10, 10]), max_=10, min_=0)
        0.0
        >>> euclidean_similarity_ts(np.array([1, 1, 1, 0, 0, 0]), np.array([1, 1, 1, 1, 1, 1]), max_=1, min_=0)
        0.2928932188134524
        >>> euclidean_similarity_ts(np.array([1, 1, 1]), np.array([1, 1, 1, 1, 1, 1]), max_=1, min_=0)
        0.2928932188134524
        >>> euclidean_similarity_ts(np.array([-1, -1, -1]), np.array([8, 8, 8, 8, 8]), max_=8, min_=-2)
        0.05872426993999269
        >>> euclidean_similarity_ts(np.array([-1, -1, -1, -2, -2]), np.array([8, 8, 8, 8, 8]), max_=8, min_=-2)
        0.05872426993999269
        >>> euclidean_similarity_ts(np.array([-1, -1, -1]), np.array([8, 8, 8, 8, 8]), max_=10, min_=-10)
        0.3325421361613904
        >>> euclidean_similarity_ts(np.array([-1, -1, -1, -10, -10]), np.array([8, 8, 8, 8, 8]), max_=10, min_=-10)
        0.3325421361613904
        >>> euclidean_similarity_ts(np.array([0, 1, 2, -3, 0]), np.array([4, 5]), max_=10, min_=-10)
        0.5283009433971698
        >>> euclidean_similarity_ts(np.array([0, 1, 2, -3, 0]), np.array([4, 5, -10, 10, 10]), max_=10, min_=-10)
        0.5283009433971698
        >>> euclidean_similarity_ts(np.array([ -10, 10, -10, 10]), np.array([4, 5]), max_=10, min_=-10)
        0.2011727345664771
        >>> euclidean_similarity_ts(np.array([ -10, 10, -10, 10]), np.array([4, 5, 10, -10]), max_=10, min_=-10)
        0.2011727345664771

    """
    avg_ = (max_ + min_) / 2
    len_a = a.size
    len_b = b.size

    if len_a == len_b:
        diff = a - b
    elif len_a > len_b:
        bx = np.where(a[len_b:] <= avg_, max_, min_)
        bx = np.concatenate((b, bx))
        diff = a - bx
    else:  # len_a < len_b:
        ax = np.where(b[len_a:] <= avg_, max_, min_)
        ax = np.concatenate((a, ax))
        diff = ax - b
    # normalized distance
    dist = np.linalg.norm(diff) / (np.sqrt(max(len_a, len_b)) * abs(max_ - min_))
    return 1. - dist if dist < 1. else 0.


def euclidean_similarity_ts_dataset(dataset):
    """Gives the custom euclidean_similarity_ts for a dataset

    Returns:
        Callable[[np.ndarray, np.ndarray], float]
    """
    max_val, min_val = get_max_min(dataset)
    return lambda p1, p2: euclidean_similarity_ts(p1, p2, max_=max_val, min_=min_val)


def gen_cb(dataset, gen_profile=None, tw_width=0, tw_step=1):
    """Generate a case base out of a TS dataset using given time-window settings

    Args:
        dataset (str): Full path of the dataset "arff" file
        gen_profile (Callable[[int, int], List[numpy.ndarray]]): Function to generate problem profiles
                and/or queries out of a data sequence. Its signature should be (data, tw_width, tw_step).
        tw_width (int): if > 0, width of the 'moving' time window;
            otherwise, 'expanding' time window approach is applied.
        tw_step (int): number of steps (in terms of data points in TS) taken at each update.
            This can also be seen as the number of data points changed at each update.

    Returns:
        cbr.TCaseBase:

    """
    # read dataset
    logger.info("Loading time series dataset: {}".format(dataset))
    ts_instances, ts_classes = read_ts(dataset)
    # create an empty CB
    cb = cbr.TCaseBase()
    # loop data appending sequences to the CB
    logger.info("Generating cb")
    for idx, instance in enumerate(ts_instances):
        cb[idx] = cbr.TSSequence(data=instance, tw_width=tw_width, tw_step=tw_step,
                                 gen_profile=gen_profile, solution=ts_classes[idx], seq_id=idx)
    logger.info(".. CB unique solutions: {}".format(cb.solution_set()))
    logger.info(".. CB generated: {} sequences containing {} cases".format(len(cb), cb.size()))
    return cb


def get_max_min(dataset):
    """Get max and min values for instance data points in the time-series dataset

    Args:
        dataset: Time series dataset file path

    Returns:
        (float, float): (max, min)

    """
    TS, _ = read_ts(dataset)
    return np.nanmax(TS), np.nanmin(TS)

"""Module for Performance Distribution Profile (PDP) related functionality"""

import logging
import os

import numpy as np
import math

from alk import common, helper
from alk.exp import exp_common, insights
from alk.run import run_insights


logger = logging.getLogger("ALK")


def quality(calc_x_sim):
    """Returns the quality at every given calc by normalizing the sim list

    Args:
        calc_x_sim (List[Tuple(int, float)]): list of (calc, sim) 2-tuples

    Returns:
        List[Tuple(int, float)]: list of (calc, quality) 2-tuples

    """
    max_sim = max(calc_x_sim, key=lambda t: t[1])[1]  # i.e. the exact NN's sim to the query
    return [(calc, sim/max_sim) for calc, sim in calc_x_sim]


def build_pdp(ins_exp, calc_step=0.01, q_step=0.05):
    """Generates the performance distribution profile for the experiment data.

    Args:
        ins_exp (str): Path of the insight experiment result file.
        calc_step (Union[int, float]): The number of similarity calculations to discretize the calculation range.
            If 0<calc_step<1, the value is used as a ratio to the max calc value, and multiplied by it to find the
            `calc_step` to be used in the PDP.
        q_step (float): Steps to discretize quality range within [0.0, ..., 1.0].

    Returns:
        (numpy.ndarray, int): (pdp, calc_step)
            - pdp is a 4D-array (update, knn_i, calc_range, quality_range) where each cell is the probability of quality
            - calc_step is the actual integer `calc_step` value used in the PDP

    """

    logger.info("In build_pdp: ")
    # Read experiment data
    result = common.load_obj(ins_exp)  # type: run_insights.ExpInsightsOutput
    logger.info(".. Insight experiment result file loaded: {}".format(ins_exp))
    knn_total_abs = result.data.knn_total_abs
    knn_calc_sim_history = result.data.knn_calc_sim_history

    # Get the dimensions
    d_iter = max(i[0] for i in knn_total_abs)  # max # of problem updates(aka updates) except the 0th initial problem
    d_k = len(knn_total_abs[0]) - 1  # k of kNN
    max_calc = int(max(i[-1] for i in knn_total_abs))  # max # of calc searched at the biggest knn_i
    if 0 < calc_step < 1:  # Use calc_step as a ratio
        calc_step = math.ceil(max_calc * calc_step)
    d_calc_range = math.ceil(max_calc / calc_step)
    calc_range = [calc_step * i for i in range(1, d_calc_range + 1)]
    d_quality_range = math.ceil(1.0 / q_step)

    # Quality occurrence distribution matrix
    distr_matrix = np.full((d_iter, d_k, d_calc_range, d_quality_range), 0, dtype=int)
    len_knn_calc_sim_history = len(knn_calc_sim_history)
    for test_ind, test_history in enumerate(knn_calc_sim_history):
        update = test_history[0]
        logger.debug("Test: {} of {}".format(test_ind, len_knn_calc_sim_history))
        for nn_ind, nn_i_history in enumerate(test_history[1:]):
            # Get the (calc, quality) tuples
            calc_x_quality = quality(nn_i_history)  # type: Tuple[int, float]
            if max_calc not in [c for c, _ in calc_x_quality]:
                calc_x_quality.append((max_calc, 1.))
            calcs, qualities = helper.to_arr(calc_x_quality, fill=False)
            calc_ind = 0
            for idx, calc in enumerate(calcs):
                q = qualities[idx]
                q_ind = 0 if q == 0 else math.ceil(q / q_step) - 1
                for c in range(calc, calcs[idx + 1] if idx < len(calcs) - 1 else calc + 1):
                    if c > calc_range[calc_ind]:  # if c falls into another calc_range cell, find it.
                        calc_ind = 0 if c == 0 else math.ceil(c / calc_step) - 1
                    # Increment related cell's value (i.e. occurrence +1)
                    distr_matrix[update - 1, nn_ind, calc_ind, q_ind] += 1

    # occurrence distribution matrix -> Performance distribution profile
    pdp = distr_matrix/distr_matrix.sum(axis=3, keepdims=True)  # Each cell is probability of quality

    return pdp, calc_step


class PDPSettings:
    """Settings used in building the PDP.

    Attributes:
        experiment (str): Path of the insight experiment result file.
        calc_step_arg (Union[int, float]): calc_step given as an argument to the script.
            (see `build_pdp` for the usage of float value as a ratio)
        calc_step (int): actual integer `calc_step` value used to discretize calculation range of the PDP
        q_step (float): quality step used to discretize quality range of the PDP

    """
    def __init__(self, experiment, calc_step_arg, calc_step, q_step):
        """

        Args:
            experiment (str): Path of the insight experiment result file.
            calc_step_arg (Union[int, float]): calc_step given as an argument to the script.
                (see `build_pdp` for the usage of float value as a ratio)
            calc_step (int): actual integer `calc_step` value used to discretize calculation range of the PDP
            q_step (float): quality step used to discretize quality range of the PDP

        """
        self.experiment = experiment
        self.calc_step_arg = calc_step_arg
        self.calc_step = calc_step
        self.q_step = q_step


class PDPOutput(exp_common.Output):
    """Holds and saves PDP settings and data"""

    def __init__(self, settings, data):
        """Overrides the parent method only to specify the type annotations.

        Args:
            settings (PDPSettings):
            data (numpy.ndarray): 4d array (update, knn_i, calc_range, quality_range) where each
                cell is the probability of quality.

        """
        # super(PDPOutput, self).__init__(settings, data)  # commented to help IDE auto-fill below attributes
        self.settings = settings
        self.data = data

    def save(self, out_file=None):
        """Saves the object itself into a pickle file.

        Args:
            out_file (str): Full path to the dumped pickle file.
                If not given, it is generated automatically out of PDP settings.

        Returns:
            str: Full path to the dumped pickle file

        """
        if out_file is None:
            out_file = gen_pdp_f_path(self.settings.experiment, self.settings.calc_step, self.settings.q_step)
        common.dump_obj(self, out_file)
        logger.info("Anytime Lazy KNN - PDP output dumped into '{}'.".format(out_file))
        return out_file


def gen_pdp_f_path(ins_exp, calc_step, q_step, suffix=""):
    """Generates the full path of the PDP file for the given insight experiment."""
    ins_exp_name = os.path.splitext(os.path.basename(ins_exp))[0]  # Base file name w/o extension
    pdp_file = os.path.join(common.APP.FOLDER.PDP,
                            "PDP_{i}__cs_{c}_qs_{q}{x}{e}".format(
                                i=ins_exp_name,
                                c=str(calc_step if isinstance(calc_step, float) else int(calc_step)),
                                q=str(q_step),
                                x=suffix,
                                e=common.APP.FILE_EXT.PICKLE))
    return pdp_file


def get_pdp_w_conf_2D(pdp_file, update, knn_i):
    """Adds confidence & std dev columns to the quality dimension of the given PDP in 2D (i.e. for `update` and `knn_i`)

    Args:
        pdp_file (str): Full path of the PDP to be used
        update (int): index of the problem sequence update
        knn_i (int): zero-based index of the KNN

    Returns:
        (np.ndarray, int, float):
            - PDP: excerpt for for `update` and `knn_i`;
            - calc_step: (actual integer value used in the discretization of the calculation range)
            - q_step: quality step used in the discretization of the quality range

    """
    # Read PDP data
    pdp_output = common.load_obj(pdp_file)
    pdp_all = pdp_output.data
    calc_step = pdp_output.settings.calc_step
    q_step = pdp_output.settings.q_step

    pdp = pdp_all[update - 1][knn_i]
    ncols = pdp.shape[1]
    decimals_q_step = len(str(q_step).split('.')[1])
    # calculate the weighted mean of probability distributions of quality (i.e. confidence) and std deviation for each row (i.e. calc range)
    q_ind_array = np.array([round(i * q_step, decimals_q_step) for i in range(1, ncols + 1)])
    conf_n_std_dev = np.apply_along_axis(lambda a: (np.average(q_ind_array, weights=a),  # conf
                                                    helper.weighted_std(q_ind_array, a)),  # std_dev
                                         axis=1, arr=pdp)
    # Add the conf and std_dev columns to the original pdp
    pdp = np.column_stack((pdp, conf_n_std_dev))
    return pdp, calc_step, q_step


def get_conf_for_calc(pdp_file, update, knn_i, calc, dec_digits=None):
    """Gives the confidence value and its standard deviation for an `update`, `knn_i` and `calc` using the PDP.

    Confidence is the weighted sum of the probability distribution along
    quality ranges in a Performance Distribution Profile (PDP).

    Interpolation is used when calc_range[i-1] < calc < calc_range[i] for
    both confidence and the std deviation of the probability distribution.

    Weighted std is given where weights are the probability distribution at each calc_range row.

    Args:
        pdp_file (str): Full path to the PDP output file.
        update (int): problem update
        knn_i (int): index in kNN (zero-based)
        calc (int): number of (similarity) calculations made.
        dec_digits (int): precision of confidence in terms of decimal digits.

    Returns:
        (float, float): (confidence, std deviation) where both values in [0., 1.]

    """
    if calc is None:
        return 1., 0.

    # Read PDP data
    pdp_output = common.load_obj(pdp_file)
    pdp_all = pdp_output.data
    calc_step = pdp_output.settings.calc_step
    q_step = pdp_output.settings.q_step

    pdp = pdp_all[update - 1][knn_i]
    d_calc_range = pdp.shape[0]  # |calc range|
    calc_range = [calc_step * i for i in range(1, d_calc_range + 1)]
    d_q_range = pdp.shape[1]  # |quality range|
    decimals_q_step = len(str(q_step).split('.')[1])
    q_ind_array = np.array([round(i * q_step, decimals_q_step) for i in range(1, d_q_range + 1)])
    pdp_weighted = np.multiply(pdp, q_ind_array)

    # Confidence  (i.e. weighted quality sum)
    pdp_weighted_sum = np.sum(pdp_weighted, axis=1)
    # Add the confidence column to the original pdp
    pdp = np.column_stack((pdp, pdp_weighted_sum))
    conf_range = pdp[:, -1]

    # Check if `calc` falls out of the range of the PDP, just in case...
    if calc > calc_range[-1]:
        # Give the last conf value and its std deviation
        pdp_calc_last_ind = pdp_all[update - 1][knn_i][-1]
        std_calc_last_ind = helper.weighted_std(q_ind_array, pdp_calc_last_ind)
        return (round(conf_range[-1], dec_digits), round(std_calc_last_ind, dec_digits)) if dec_digits is not None else (conf_range[-1], std_calc_last_ind)

    # Use interpolation for confidence
    conf = np.interp(calc, calc_range, conf_range)

    # Standard deviation
    # ... Deviation of the row that `calc` falls into
    calc_ind = 0 if calc == 0 else math.ceil(calc / calc_step) - 1
    pdp_calc_ind = pdp_all[update - 1][knn_i][calc_ind]
    std_calc_ind = helper.weighted_std(q_ind_array, pdp_calc_ind)
    # ... Deviation of the previous row
    calc_prev_ind = 0 if calc_ind == 0 else calc_ind - 1
    pdp_calc_prev_ind = pdp_all[update - 1][knn_i][calc_prev_ind]
    std_calc_prev_ind = helper.weighted_std(q_ind_array, pdp_calc_prev_ind)

    # Use interpolation for std deviation as well (or their mean would could also be used?)
    std = np.interp(calc, [calc_range[calc_prev_ind], calc_range[calc_ind]], [std_calc_prev_ind, std_calc_ind])
    return (round(conf, dec_digits), round(std, dec_digits)) if dec_digits is not None else (conf, std)


def get_calc_for_conf_thold(pdp_file, update, conf_thold, z=-1, knn_i=-1):
    """Gives the estimated calc value needed to reach the given confidence threshold.

    The returned calc can be the exact value in the PDP or an interpolated one
    by using the calc_(j-1), calc_j of PDP where conf_(j-1) < conf < conf_j.

    Args:
        pdp_file (str): Full path of the PDP to be used
        update (int): Problem file
        conf_thold (float): Confidence threshold value [0., 1.]
        z (int): factor of std deviations to be added or subtracted from the 'confidence'z in the 'efficiency' measure.
            It provides a means of how prudent we want to be with the raw confidence value provided by PDP.
            z=−1 would be a more cautious choice than the neutral z=0, and z=1 would be a more optimistic one.
        knn_i (int): zero-based index of the KNN; provide -1 for the last kNN member

    Returns:
        int: calc

    """
    pdpc, calc_step, _ = get_pdp_w_conf_2D(pdp_file, update, knn_i)
    dc, dq = pdpc.shape
    c_ind_array = np.array([i * calc_step for i in range(1, dc + 1)])  # calc values for indices
    calc = c_ind_array[-1]  # In some pdpc's conf_thold may not be satisfied, in this case return the last calc
    for c in range(dc):
        conf = pdpc[c, dq - 2]
        std_dev = pdpc[c, dq - 1]
        if conf + z * std_dev >= conf_thold:
            if conf + z * std_dev == conf_thold or c == 0:
                # exact value
                calc = c_ind_array[c]
            else:
                # interpolation
                prev_conf = pdpc[c - 1, dq - 2]
                prev_std_dev = pdpc[c - 1, dq - 1]
                prev_calc = c_ind_array[c - 1]
                calc = math.ceil(np.interp(conf_thold,
                                           [prev_conf + z * prev_std_dev, conf + z * std_dev],
                                           [prev_calc, c_ind_array[c]]))
            break
    return calc


def get_calcs_for_conf_tholds(pdp_file, update, conf_tholds, z=-1, knn_i=-1):
    """Gives the list of estimated calc values needed to reach given confidence thresholds.

    A returned calc can be the exact value in the PDP or an interpolated one
    by using the calc_(j-1), calc_j of PDP where conf_(j-1) < conf < conf_j.

    Args:
        pdp_file (str): Full path of the PDP to be used
        update (int): index of the problem sequence update
        conf_tholds (List[float]): List of confidence threshold values [0., 1.]
        z (int): factor of std deviations to be added or subtracted from the 'confidence'z in the 'efficiency' measure.
            It provides a means of how prudent we want to be with the raw confidence value provided by PDP.
            z=−1 would be a more cautious choice than the neutral z=0, and z=1 would be a more optimistic one.
        knn_i (int): zero-based index of the KNN; provide -1 for the last kNN member

    Returns:
        List[int]: list of calcs

    """
    stop_calc_list = []
    for thold in conf_tholds:
        stop_calc_list.append(get_calc_for_conf_thold(pdp_file=pdp_file,
                                                      update=update,
                                                      conf_thold=thold,
                                                      z=z,
                                                      knn_i=knn_i))
    return stop_calc_list


def confidence_adjusted(conf, std=0., z=0.):
    """Adjust the confidence to be used with standard deviation

    Args:
        conf (float): confidence (i.e., expected quality)
        std (float): standard deviation of confidence
        z (float): number of standard deviations to be taken into account in the performance

    Returns:
        float:

    """
    return conf + z * std


def confidence_absolute_error(conf, q, std=0., z=0.):
    """Absolute error in confidence estimation

    Args:
        conf (float): confidence (i.e., expected quality)
        q (float): observed quality (i.e., sim(kNN_approx, query) / sim(kNN_exact, query))
        std (float): standard deviation of confidence
        z (float): number of standard deviations to be taken into account in the performance

    Returns:
        float:

    """
    return abs(q - confidence_adjusted(conf, std, z))


def confidence_absolute_pct_error(conf, q, std=0., z=0.):
    """Absolute percentage error in confidence estimation

    Args:
        conf (float): confidence (i.e., expected quality)
        q (float): observed quality (i.e., sim(kNN_approx, query) / sim(kNN_exact, query))
        std (float): standard deviation of confidence
        z (float): number of standard deviations to be taken into account in the performance

    Returns:
        float: if q != 0
        `numpy.inf`: if q == 0

    """
    if q != 0.:
        return (confidence_absolute_error(conf, q, std, z) / q) * 100.
    else:
        return np.inf


def confidence_efficiency_sim(conf, sim, std=0., z=0.):
    """Efficiency of the confidence measure (f.k.a., confidence performance) using sim as divisor

    Args:
        conf (float): confidence
        sim (float): result of the sim(kNN_approx, kNN_exact)
        std (float): standard deviation of confidence
        z (float): number of standard deviations to be taken into account in the performance

    Returns:
        float: if sim != 0
        `numpy.inf`: if sim == 0

    """
    if sim != 0:
        return confidence_adjusted(conf, std, z) / sim
    else:
        return np.inf


def confidence_efficiency_q(conf, q, std=0., z=0.):
    """Efficiency of the confidence measure (f.k.a., confidence performance) using quality as divisor

    Args:
        conf (float): confidence
        q (float): quality (i.e. result of the sim(kNN_approx, query) / sim(kNN_exact, query))
        std (float): standard deviation of confidence
        z (float): number of standard deviations to be taken into account in the performance

    Returns:
        float: if q != 0
        `numpy.inf`: if q == 0

    """
    if q != 0:
        return confidence_adjusted(conf, std, z) / q
    else:
        return np.inf


def get_exp_ins_settings(pdp_file):
    """Gives the settings of the insight experiment out of which the PDP is generated

    Args:
        pdp_file (str): Full path of the PDP to be used:

    Returns:
        insights.ExpInsightsSettings:

    """
    pdp_output = common.load_obj(pdp_file)  # type: PDPOutput
    exp_ins_output = common.load_obj(pdp_output.settings.experiment)  # type: insights.ExpInsightsOutput
    return exp_ins_output.settings
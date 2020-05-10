"""Performance Distribution Profile plots"""

import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from alk import common, helper


def pdp(pdp_file, file_format=None, update=1, ki=0, decimals=3, to_latex=False, start_q=0.0):
    """Plots or exports as latex a PDP table for a given update and ki.

    Only top n rows where the n^th row is the first row with conf=1.0 are plotted or exported.

    Args:
        pdp_file (str): pickle file for the performance distribution profile
        file_format (str): One of the file extensions supported by the active backend.
            Most backends support png, pdf, ps, eps and svg.
            if is None, the plot is displayed and not saved.
        update (int): which update to plot.
        ki (int): zero-based index of the NN in kNN list to plot.
        decimals (int): number of decimal digits for rounding probability and confidence values;
        to_latex (bool): if True, PDP 2D excerpt is saved into a .tex file in the pdp folder and the figure is not
            displayed or saved.
        start_q (float): start column for the quality to be used to plot/export the PDP

    Returns:
        None

    """
    # File info
    # fn = basename(pdp_file)
    fn_wo_ext = common.file_name_wo_ext(pdp_file)
    save_fn = "PDP_FIG_{}_ki_{}_u_{}".format(fn_wo_ext, ki, update)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Read experiment data
    pdp_output = common.load_obj(pdp_file)
    pdp_all = pdp_output.data
    calc_step = pdp_output.settings.calc_step
    q_step = pdp_output.settings.q_step
    pdp = pdp_all[update - 1][ki]
    # pdp = pdp * 100
    # Filter PDP in order not to plot redundant rows with conf = 1.0
    pdp_rows_conf_1 = np.where(pdp[:, -1] == 1.0)[0]  # rows with conf=1.0
    if pdp_rows_conf_1.size > 0:  # If the calc_step is very coarse, then there may be no rows with conf = 1
        top_n = pdp_rows_conf_1[0] + 1  # top n rows where the n^th row is the first row with conf=1.0
    else:
        top_n = pdp.shape[0]
    pdp = pdp[:top_n]
    nrows = pdp.shape[0]
    rows = ["{:d}".format(i * calc_step) for i in range(1, nrows + 1)]
    ncols = pdp.shape[1]
    decimals_q_step = len(str(q_step).split('.')[1])
    cols = ["{0:.{1}f}".format(i * q_step, decimals_q_step) for i in range(1, ncols + 1)]
    # calculate the weighted mean of probability distributions of quality (i.e. confidence) and std deviation for each row (i.e. calc range)
    q_ind_array = np.array([round(i * q_step, decimals) for i in range(1, ncols + 1)])
    conf_n_std_dev = np.apply_along_axis(lambda a: (np.average(q_ind_array, weights=a),  # conf
                                                    helper.weighted_std(q_ind_array, a)),  # std_dev
                                         axis=1, arr=pdp)

    # Add the conf and std_dev columns to the original pdp
    pdp = np.column_stack((pdp, conf_n_std_dev))
    cols = cols + ["Confidence", "\u03C3"]
    pdp = pdp.round(decimals)
    if start_q:
        start_col_ind = math.ceil(start_q / q_step) - 1
        pdp = pdp[:, start_col_ind:]  # Show only the quality columns >= start_q
        ncols = ncols - start_col_ind
        cols = cols[start_col_ind:]
        save_fn = "{}{}".format(save_fn, "_sq_{}".format(cols[0]) if start_q else "")
    cell_colors = plt.cm.Oranges(pdp)
    ########################
    # PDP to LaTeX
    if to_latex:
        # Make a table for the top n rows where the n^th row is the first row with conf=1.0
        pdp = pd.DataFrame(data=pdp, index=rows, columns=cols)
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.tex".format(save_fn))
        pdp.to_latex(buf=save_fpath, index=True, float_format=lambda x: "{0:.{1}f}".format(x, decimals) if x != 0. else "")
        print("PDP saved as LaTeX table into '{}'.".format(save_fpath))
        return None
    ########################
    # Clear 0.0 cells
    pdp = pdp.astype("U{}".format(2 + decimals))  # len("0.") = 2
    pdp[pdp == "0.0"] = ""
    # Create a figure
    hcell, wcell = 0.3, .8
    hpad, wpad = 0, 0
    fig = plt.figure(figsize=(ncols * wcell + wpad, (nrows * hcell + hpad)))
    # fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.axis('off')
    # Add a table at the bottom of the axes
    table = ax2.table(cellText=pdp,
                      rowLabels=rows,
                      rowLoc='right',
                      rowColours=plt.cm.BuPu(np.linspace(0, 0.5, len(rows))),
                      colColours=plt.cm.YlGn(np.linspace(0, 0.5, len(cols))),
                      cellColours=cell_colors,
                      colLabels=cols,
                      loc='center')
    # Set "confidence" column header's color.
    c = table.get_celld()[(0, len(cols)-1)]
    c.set_facecolor("w")
    title_ = "Performance Distribution Profile\n"
    title_ = "{}PDP: {}\n".format(title_, fn_wo_ext)
    title_ = "{}ki: {}, update: {}\n".format(title_, ki, update)
    # plt.subplots_adjust(left=0.2, top=0.8)
    plt.title(title_)
    plt.tight_layout()
    if file_format:
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.{}".format(save_fn, file_format))
        plt.savefig(save_fpath, dpi=150, bbox_inches="tight")
        print("PDP figure saved into '{}'.".format(save_fpath))
    else:
        # Update the title of the plot window
        plt.gcf().canvas.set_window_title(save_fn)
        plt.show()
    plt.close()

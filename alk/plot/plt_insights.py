"""Library to plot insights data"""

import math
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from alk import common, helper
from alk.exp import pdp
from alk.plot import plt_common


def gains_multiple(experiments, file_format="pdf", marker_size=1., color_ind=None, with_title=True, signature=True):
    """Plots the gains for a list of insights experiments.

    Args:
        experiments (list): List of full paths to the `run.run_insights` experiment(s) result file(s).
        file_format (str): One of the file extensions supported by the active backend.
            Most backends support png, pdf, ps, eps and svg.
        marker_size (float): size of the marker in scatter plot
        color_ind (int): Index in the `sns.color_palette()` for plotting single experiment
        with_title (bool): if True, shows the figure title.
        signature (bool): If True, name of the plotting function is also displayed.

    Returns:
        None

    """
    COLORS_ = sns.color_palette()
    title_ = "Gain in similarity calcs compared to Brute Search"
    save_fn = ""
    sns.set(style='whitegrid')  # , font_scale=.9)
    plt.figure(num=1, figsize=(8, 6))
    for exp_id, experiment in enumerate(experiments):
        result = common.load_obj(experiment)
        fn_wo_ext = common.file_name_wo_ext(experiment)
        # title_ = "{}\nExp #{}: {}".format(title_, exp_id, fn)
        dd = pd.DataFrame(result.data.gain)
        dd.columns = ['Update', '% Gain']

        g = sns.regplot(x='Update', y='% Gain', data=dd, scatter=True,
                        fit_reg=True, scatter_kws={"s": marker_size}, order=3,
                        ci=None, line_kws={"label":"#{}: {}".format(exp_id, fn_wo_ext)},
                        color=COLORS_[exp_id] if len(experiments) > 1 or not color_ind else COLORS_[color_ind],  # color hack for presentations
                        truncate=True)
        plt.ylim(np.min(dd['% Gain']), 100.)  # np.max(dd['% Gain']))
        g.set(ylabel="Gain (%)")
        plt.xlim(np.min(dd['Update']), np.max(dd['Update']))
        g.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        # plt.xticks(range(np.min(dd['Update']), np.max(dd['Update']) + 1))
        plt.gca().grid(True, linestyle="dashed", linewidth=0.4)
        plt.xlabel('Problem index')
        if not save_fn:
            save_fn = 'GAINS_{}'.format(fn_wo_ext)
    save_fn = "{}{}{}".format(save_fn,
                              "_and_{}_more".format(len(experiments) - 1) if len(experiments) > 1 else "",
                              "_MARKERS" if marker_size else "")
    if signature:
        plt_common.sign_plot(plt, gains_multiple.__name__)
    plt.legend(title="Experiments", frameon=True, loc="best", fontsize="small")
    if with_title:
        plt.title(title_ + "\n")
    if file_format:
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.{}".format(save_fn, file_format))
        plt.savefig(save_fpath, dpi=300, bbox_inches="tight")
        print("Gains figure saved into '{}'.".format(save_fpath))
    else:
        # Update the title of the plot window
        plt.gcf().canvas.set_window_title(save_fn)
        plt.show()
    plt.close()


def insights_multiple(experiments, file_format="pdf", total=True, actual=True, all_k=False, marker_size=0,
                      all_ticks=True, with_title=True, signature=True):
    """Plots the total and actual calculations made to find kNNs for multiple experiments on the same figure.

    Args:
        experiments (list): List of full paths to the `run.run_insights` experiment(s) result file(s).
        file_format (str): One of the file extensions supported by the active backend.
            Most backends support png, pdf, ps, eps and svg.
        total (bool): If True, plots the total number of calculations made for the ki's.
        actual (bool): If True, plots the actual number of calculations made for the ki's.
        all_k (bool): If False, plots all ki's in the first experiment and then plots
            only the calcs for the kth NN for the experiments with index>=1;
            otherwise, plots all ki's for all experiments. If there is only one
            experiment to plot, this argument is ignored.
        marker_size (float): size of the marker in scatter plot
        all_ticks (bool): If True, all x-ticks are displayed.
        with_title (bool): if True, shows the figure title.
        signature (bool): If True, name of the plotting function is also displayed.

    Returns:
        None

    """
    LINE_STYLE = {0: "-", 1: "--", 2: "-.", 3: ":"}
    COLORS_ = sns.color_palette()
    LW = 1
    experiments_k = []  # k's of the experiment for the output file name
    from matplotlib.ticker import MaxNLocator
    # Create a figure
    plt.figure(num=1, figsize=(12, 10))
    title_ = "Anytime Lazy KNN Search\nTotal vs Actual # of Similarity Assessments\n\n"
    sns.set_style("whitegrid")
    for exp_id, experiment in enumerate(experiments):
        result = common.load_obj(experiment)
        fn_wo_ext = common.file_name_wo_ext(experiment)
        # Read experiment data
        data = result.data
        if total:
            k_calcs_total = data.knn_total_cumsum
            k = len(k_calcs_total[0]) - 1  # k of kNN
        if actual:
            k_calcs_actual = data.knn_actual_cumsum
            if not total:
                k = len(k_calcs_actual[0]) - 1  # k of kNN
        experiments_k.append(str(k))
        for i in range(k):
            # If all_k is False, plot k_calcs of all ki's of the 1st experiment but only the 0^th and k'th of the other experiments.
            if all_k or exp_id == 0 or i == (k - 1) or i == 0:
                if total:
                    # Total calculation number for each kNN
                    dd = pd.DataFrame(np.array(k_calcs_total, dtype=int)[:, [0, i + 1]].tolist())
                    dd.columns = ["Update", "Comps"]
                    data = dd
                    label_str = "kNN[{}] total".format(i) if len(experiments) == 1 else "kNN[{}] total (Exp #{})".format(i, exp_id)
                    g = sns.regplot(x='Update', y='Comps', data=data,
                                    scatter=True, fit_reg=True,
                                    scatter_kws={"s": marker_size}, order=3, ci=None,  # ci=100,
                                    color=COLORS_[i],
                                    truncate=True,
                                    line_kws={"linestyle": LINE_STYLE[exp_id % len(LINE_STYLE)],
                                              "label": label_str,
                                              "lw": LW})
                    plt.legend(frameon=True, loc="best", fontsize="large")
                if actual:
                    # Actual calculation number for each kNN
                    dd = pd.DataFrame(np.array(k_calcs_actual, dtype=int)[:, [0, i + 1]].tolist())
                    dd.columns = ["Update", "Comps"]
                    data = dd
                    label_str = "kNN[{}] actual".format(i) if len(experiments) == 1 else "kNN[{}] actual (Exp #{})".format(i, exp_id)
                    g = sns.regplot(x='Update', y='Comps', data=data,
                                    scatter=True, fit_reg=True,
                                    scatter_kws={"s": marker_size}, order=3, ci=None,  # ci=100,
                                    marker="x",
                                    color=COLORS_[i],
                                    truncate=True,
                                    line_kws={"linestyle": LINE_STYLE[(exp_id % len(LINE_STYLE)) + 1],  # Actual dashed
                                              "label": label_str,
                                              "lw": LW})
                    plt.legend(frameon=True, loc="best", fontsize="large")
        # Update the title
        title_ = "{}{}Exp #{}: {}".format(title_, "\n" if exp_id > 0 else "", exp_id, fn_wo_ext)

    #g.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax = plt.gca()
    if all_ticks:
        x_ticks_ = dd["Update"].unique()
        ax.set_xticks(x_ticks_)
    # if not all_ticks:
    #     for ind, label in enumerate(g.get_xticklabels()):
    #         if ind % (len(x_ticks_) % 10) == 0:  # only 10% of the ticks is shown
    #             label.set_visible(True)
    #         else:
    #             label.set_visible(False)
    ax.grid(True, linestyle="dashed", linewidth=0.4)
    plt.xlabel("Problem index", fontsize="x-large")
    plt.ylabel("Cumulative # of sim calculations for the i^th NN", fontsize="x-large")
    #plt.rcParams['xtick.labelsize'] = "x-large"
    #plt.rcParams['ytick.labelsize'] = "x-large"
    ax.tick_params(axis='both', which='major', labelsize="large")
    if signature:
        plt.figtext(0.99, 0.01, 'rendered by \'insights_multiple\'.', horizontalalignment='right', alpha=0.5, size="small")
    save_fn = 'CALCS_{}_{}{}{}-{}{}'.format(fn_wo_ext,
                                            "+".join(experiments_k),
                                            "-TOTAL" if total else "",
                                            "-ACTUAL" if actual else "",
                                            "ALL_K" if all_k or len(experiments) == 1 else "SELECT_K",
                                            "-MARKERS" if marker_size else "")
    # Udate the title of the plot window
    plt.gcf().canvas.set_window_title(save_fn)
    if with_title:
        plt.title(title_ + "\n")
    if file_format:
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.{}".format(save_fn, file_format))
        plt.savefig(save_fpath, dpi=300, bbox_inches="tight")
        print("Total vs Actual Calcs figure saved into '{}'.".format(save_fpath))
    else:
        # Update the title of the plot window
        plt.gcf().canvas.set_window_title(save_fn)
        plt.show()
    plt.close()


def quality_map(experiment, file_format=None, ki=None, urange=None, ustep=1,
                     colored=True, fill=False, with_title=True, signature=True,
                     q_full_scale=True, start_calc=1, cull=None):
    """Plots the log scale quality map for a given kNN[j] of all sample test sequences.

    Plots all or ppi (calc, sim) points between each major tick interval

    Args:
        experiment (str): insights experiment results full file path
        file_format (str): One of the file extensions supported by the active backend.
            Most backends support png, pdf, ps, eps and svg.
            if is None, the plot is displayed and not saved.
        ki (int): zero-based index of the NN in kNN list, if None all kNNs are plotted.
        ustep (int): a different color is chosen in the color palette for every `ustep` number of updates;
            irrelevant for colored=False.
        urange (tuple): range of updates to plot given as (start, end) where both is inclusive;
            if given as (start, ), max update is used for end;
            if None, all updates are plotted.
        colored (bool): If True, each update is plotted in a different color; otherwise all plotted black,
        fill (bool): if True, propagates the quality for the intermediate calc values;
            if False, plots only the given points provided as sparse list.
        with_title (bool): if True, shows the figure title.
        signature (bool): If True, name of the plotting function is also displayed.
        q_full_scale (bool): if True, quality (i.e. y) axis starts from 0.0; otherwise minimum quality is used.
        start_calc (int): The start value for the calculations (i.e. x) axis.
        cull (float): The percentage (.0, 1.] to cull the data points to be plotted

    Returns:
        None

    Note:
        Start: 20190918, End:20191023
    """
    # Create a figure
    plt.figure(num=1)  # , figsize=(10, 8))
    # Read experiment data
    result = common.load_obj(experiment)
    knn_calc_sim_history = result.data.knn_calc_sim_history
    k = len(knn_calc_sim_history[0]) - 1  # k of kNN
    max_update = max([test[0] for test in knn_calc_sim_history])
    if urange is None:
        urange = (1, max_update)
    elif len(urange) == 1 or urange[1] > max_update:
        urange = (urange[0], max_update)
    max_X = 0
    # Fill in plot data
    CALCS = np.array([])
    QUALITY = np.array([])
    UPDATE = np.array([])
    knn_calc_sim_history = sorted(knn_calc_sim_history, key=lambda point: point[0], reverse=False)  # sort by updates
    for test_history in knn_calc_sim_history:
        # print(test_history)
        update = test_history[0]
        if urange[0] <= update <= urange[1]:
            # print("update: ", update)
            for nn_ind, nn_i_history in enumerate(test_history[1:]):
                if ki is None or nn_ind == ki:
                    points = pdp.quality(nn_i_history)
                    # print(points)
                    X, Y = helper.to_arr(points, fill=fill)
                    # Eliminate (0, 0.0) entries
                    if X[0] == 0:
                        X = X[1:]
                    if Y[0] == 0:
                        Y = Y[1:]
                    if max(X) > max_X:
                        max_X = max(X)
                    # Eliminate the entries < start_calc
                    X = X[X >= start_calc]
                    Y = Y[-len(X):]
                    CALCS = np.concatenate((CALCS, X))
                    QUALITY = np.concatenate((QUALITY, Y))
                    # UPDATE = np.concatenate((UPDATE, np.full((len(X),), math.ceil(update / ustep), dtype=np.int)))
                    UPDATE = np.concatenate((UPDATE, np.full((len(X),), update, dtype=np.int)))
    if cull:
        CALCS, ind_removed = helper.cull_arr(CALCS, pct=cull)
        QUALITY, _ = helper.cull_arr(QUALITY, ind=ind_removed)
        UPDATE, _ = helper.cull_arr(UPDATE, ind=ind_removed)
    if colored:
        # Color palette
        cmap = "autumn"  # "autumn"  "tab20"  # "Blues_r"
        cmap_size = math.ceil(max_update / ustep)
        my_palette = plt.cm.get_cmap(cmap, cmap_size)
        _ = plt.scatter(CALCS, QUALITY, marker=".", s=1, c=UPDATE, cmap=my_palette, vmin=1, vmax=max_update, alpha=1.)
        cbar = plt.colorbar(orientation="vertical")
        cbar.set_label("updates")
    else:
        _ = plt.scatter(CALCS, QUALITY, marker=".", s=1, c="black")
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(.05))
    plt.grid(True, which="major", linestyle="-", linewidth=1, color="lightgrey")
    plt.grid(True, which="minor", linestyle=":", linewidth=1, color="lightgrey")
    xticks_ = plt_common.get_ticks_log_scale(max_X, start=start_calc)
    y_min = 0.0 if q_full_scale else math.floor(np.nanmin(QUALITY[start_calc - 1:]) * 10) / 10
    yticks_ = np.arange(y_min, 1.01, .1)
    plt.rcParams['axes.axisbelow'] = True
    plt.xscale('log')
    plt.xticks(xticks_)
    plt.xlim(left=start_calc, right=max(xticks_))
    plt.yticks(yticks_)
    if signature:
        plt_common.sign_plot(plt, quality_map.__name__)
    fn_wo_ext = common.file_name_wo_ext(experiment)
    lbl_ki = str(ki if ki is not None else list([0, k - 1]))  # zero-based
    lbl_update = str(list(urange) if urange[0] != urange[1] else urange[0])
    if with_title:
        title_ = "Quality Map\n"
        title_ = "{}Exp: {}\n".format(title_, fn_wo_ext)
        title_ = "{}ki:{}, update:{}".format(title_, lbl_ki, lbl_update)
        if colored:
            title_ = "{}, color step:{}".format(title_, ustep)
        if cull:
            title_ = "{}, cull:{:.0%}".format(title_, cull)
        plt.title(title_ + "\n\n")
    save_fn = "QUALITY_MAP_{}_ki_{}_u_{}{}{}{}{}{}".format(fn_wo_ext,
                                                         lbl_ki,
                                                         lbl_update,
                                                         "_s_{}".format(ustep) if colored else "",
                                                         "_f" if fill else "",
                                                         "_t" if with_title else "",
                                                         "_c_{:.2f}".format(cull) if cull else "",
                                                         "_z" if not q_full_scale else "")
    # axis labels
    matplotlib.rcParams['text.usetex'] = True  # Allow LaTeX in text
    ax.set_xlabel("$\#$ of similarity calculations ($c$)")
    ax.set_ylabel(r"quality ($\mathcal{Q}_c$)")
    # Tight layout
    plt.tight_layout()
    if file_format:
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.{}".format(save_fn, file_format))
        plt.savefig(save_fpath, dpi=300, bbox_inches="tight")
        print("Quality Map figure saved into '{}'.".format(save_fpath))
    else:
        # Update the title of the plot window
        plt.gcf().canvas.set_window_title(save_fn)
        plt.show()
    plt.close()

    return None


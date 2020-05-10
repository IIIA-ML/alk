"""Library to plot interruption experiment data"""

import os
import re
from typing import List  # For type hints

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from alk import common, helper
from alk.exp import intrpt
from alk.plot import plt_common


def efficiency(experiments, file_format="png", y_to_use="effcyq",
               rtrim=None, outliers=False, with_title=True, signature=True,
               palette="tab20", maximized=True, aspect=None):
    """Plots efficiency (f.k.a. confidence performance) for given interruption experiments.

    Args:
        experiments (List[str]): list of experiment file paths
        file_format (str): One of the file extensions supported by the active backend.
            Most backends support png, pdf, ps, eps and svg.
            if is None, the plot is displayed and not saved.
        y_to_use (str): ['abspcterr', 'abserr', 'effcysim', 'effcyq'] fields in experiment result dataframe
            standing for absolute error, absolute error percentage, efficiency (using sim) and efficiency (using quality)
            of confidence.
        rtrim (list): Remove given strings at the end of dataset names; e.g. ['_TRAIN', '_TEST']
        outliers (bool): If True, outliers are plotted.
        with_title (bool): If True, show generated plot title.
        signature (bool): If True, write the name of this function.
        palette (str): A matplotlib colormap. e.g. 'tab20", 'Purples_d'
        maximized (bool): If True, maximize plot to full screen.
        aspect (float): Desired aspect ratio (i.e. height/width) of the figure.
            If not None, the width is re-adjusted for this ratio while the height remains the same.

    Returns:
        None

    Raises:
        ValueError:
            i) If dataset names for the experiments are different;
            ii) If y_to_use not in valid options.

    """

    Y_OPTIONS = {"abspcterr": "absolute percentage error (%)",
                 "abserr": "absolute error",
                 "effcysim": "efficiency ($\eta$)",  # using sim
                 "effcyq": "efficiency ($\eta$)"}  # using quality
    if y_to_use not in Y_OPTIONS.keys():
        raise ValueError("Non-valid value for y_to_use argument. Should be in {}".format(list(Y_OPTIONS.keys())))
    df_output = None
    # Variables for output file name
    dataset_name = None
    w_list = []
    step_list = []
    z_list = []  # z=-nstd in the new efficiency definition
    # Populate summary dictionary
    for exp in experiments:
        # Read experiment data
        result = common.load_obj(exp)  # type: intrpt.ExpIntrptOutput
        # Update output DataFrame
        data = result.data
        data["setting"] = "$w$:{}, $step$:{}, $z$:{}".format(result.settings.tw_width,
                                                             result.settings.tw_step,
                                                             helper.is_float_int(result.settings.z))
        df_output = pd.concat([df_output, data])
        # Update variables for output file name
        if dataset_name is None:
            dataset_name = common.file_name_wo_ext(result.settings.dataset)
            if rtrim:
                for tag in rtrim:
                    # Trim end tags
                    dataset_name = re.sub("{}$".format(tag), "", dataset_name)
        elif result.settings.dataset.find(dataset_name) == -1:
            # dataset_name = "Misc_{}".format(time.strftime("%Y%m%d"))
            raise ValueError("Plotting different datasets not allowed yet: {}, {}.".format(dataset_name,
                                                                                           result.settings.dataset))
        w_list.append(result.settings.tw_width)
        step_list.append(result.settings.tw_step)
        z_list.append(helper.is_float_int(result.settings.z))

    # Plot
    plt.figure()
    ax = plt.gca()
    # Show grid
    ax.grid(True, linestyle="dashed", linewidth=.5)
    # Grouped boxplot
    sns.boxplot(x="confthold", y=y_to_use, hue="setting", data=df_output,
                palette=palette, linewidth=.5, showfliers=outliers, fliersize=.2)
    if y_to_use in ["effcysim", "effcyq"]:
        # Draw a horizontal line for y=1  (efficiency approx= 1)
        ax.axhline(1, linestyle="--", color="black", linewidth=1.)
    # axis labels
    matplotlib.rcParams['text.usetex'] = True  # Allow LaTeX in text
    ax.set_xlabel("confidence thresholds ($\mu\!+\!z\sigma$) for interruption")
    ax.set_ylabel("{}".format(Y_OPTIONS[y_to_use]))
    # plot title
    if with_title:
        plt.title("{} of Confidence in Interruption Tests\nfor ${}$".format(Y_OPTIONS[y_to_use].capitalize(),
                                                                            dataset_name.replace("_", "\_")))
    # Add the signature
    if signature:
        plt_common.sign_plot(plt, efficiency.__name__)
    # Maximize plot to full screen
    if maximized:
        plt.legend(fontsize="medium")
        manager = plt.get_current_fig_manager()
        backend_ = matplotlib.get_backend()
        if backend_.upper == "TKAGG":
            manager.resize(*manager.window.maxsize())
        elif backend_.upper().startswith("QT"):
            manager.window.showMaximized()
        elif backend_.find("interagg") != -1:  # Hack for PyCharm SciView
            pass
        else:  # Try your chance
            manager.resize(*manager.window.maxsize())
    else:
        plt.legend(fontsize="small")
    # File name/Window title
    w_list = [str(_) for _ in set(sorted(w_list))]
    step_list = [str(_) for _ in set(sorted(step_list))]
    z_list = [str(_) for _ in set(sorted(z_list))]
    save_fn = "EFF_{}(x{})_w_[{}]_s_[{}]_z_[{}]_{}{}{}{}".format(dataset_name,
                                                               len(experiments),
                                                               "_".join(w_list),
                                                               "_".join(step_list),
                                                               "_".join(z_list),
                                                               y_to_use,
                                                               "_a_{}".format(aspect) if aspect else "",
                                                               "_o" if outliers else "",
                                                               "_t" if with_title else "")
    # Aspect ratio
    if aspect:
        # figw, figh = plt.rcParams["figure.figsize"]
        # plt.rcParams["figure.figsize"] = [figw / aspect, figh]
        fig = plt.gcf()
        figw, figh = fig.get_size_inches()
        fig.set_size_inches(figh / aspect, figh, forward=True)
    if file_format:
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.{}".format(save_fn, file_format))
        plt.savefig(save_fpath, dpi=300, bbox_inches="tight")
        print("Confidence efficiency figure saved into '{}'.".format(save_fpath))
    else:
        # Update the title of the plot window
        plt.gcf().canvas.set_window_title(save_fn)
        plt.show()
    plt.close()

    return None

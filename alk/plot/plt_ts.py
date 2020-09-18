"""Library to plot time series"""

import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from alk import common
from alk.exp import ts
from alk.plot import plt_common
from alk.run import run_common


def cb_sequences(dataset, file_format="pdf", width=0, step=1, seqs=[0], upd_ind=None,
                 full=True, with_title=True, signature=True, **kwargs):
    """Plots given sequences of a given time series CB.

    Args:
        dataset (str): The sequence dataset file to be converted to CB.
        file_format (str): One of the file extensions supported by the active backend.
            Most backends support png, pdf, ps, eps and svg.
            if is None, the plot is displayed and not saved.
        width (int): if > 0, width of the moving time window; otherwise expanding windows approach is applied.
        step (int): number of steps (in terms of data points in TS) taken by the window at each update.
                This can also be seen as the number of data points changed in each update.
        seqs (list or int): if is a `list`, then the items are the indexes of the sequences in the CB to be plotted;
            if is an `int`, `seq_ind` number of randomly selected sequences are plotted
        upd_ind (int): the update (i.e. the upd_ind^th problem profile) of the sequences to be plotted;
            if None, *last* update of the sequences are plotted
        full (bool): if True, plots full sequence in light grey color under the given `update`
        with_title (bool): if True, shows the figure title.
        signature (bool): If True, name of the plotting function is also displayed.
        **kwargs: to passed over to the `matplotlib.pyplot.plot` function
    Returns:
        list : indices of the plotted sequences.

    """
    COLORS_ = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cb = ts.gen_cb(dataset=dataset, tw_width=width, tw_step=step)
    max_updates = max([len(seq.data) for seq in cb.sequences()])  # Data points
    max_profiles = max([seq.n_profiles() for seq in cb.sequences()])  # Cases, max_updates=max_profiles if step=1
    Y_max, Y_min = ts.get_max_min(dataset)
    X = list(range(max_updates))
    xticks_ = list(np.arange(0, max_updates, step=math.ceil(max_updates / 10)))
    if [max_updates - 1] not in xticks_:
        xticks_.append(max_updates - 1)
    if upd_ind is None:
        upd_ind = max_profiles - 1

    if not isinstance(seqs, list):
        seq_ind = random.sample(range(len(cb.sequences())), seqs)
    else:
        seq_ind = seqs
    for ix, ind in enumerate(seq_ind):
        case_features = cb[ind].profile(idx=upd_ind)
        start_, end_ = cb[ind]._get_range_profile(upd_ind, len(cb[ind].data), width, step)
        pad_right = start_
        pad_left = max_updates - end_
        Y = [None] * pad_right + list(case_features) + [None] * pad_left  # do padding to the sides too
        if full and upd_ind < max_updates:
            full_seq = cb[ind].data
            plt.plot(X, list(full_seq), label=None, color=COLORS_[2 + ix], alpha=0.2)  # Hack not to repeat gain plot colors
        plt.plot(X, Y, color=COLORS_[2 + ix], label="Seq {}".format(ind), alpha=0.8 if ix > 0 else 1, **kwargs)
    plt.xticks(xticks_)
    plt.xlim(left=min(xticks_))
    plt.ylim(Y_min, Y_max)
    # Show both gridlines
    ax = plt.gca()
    ax.grid(True, linestyle=":", linewidth=.5)
    plt.ylabel("value", fontsize="large")
    plt.xlabel("data point", fontsize="large")
    # plt.margins(x=0)  # ! This removes the None Y values from the plot, which we do NOT want.
    plt.legend(fontsize="medium", ncol=2)
    fn_wo_ext = common.file_name_wo_ext(dataset)
    if with_title:
        title_ = "Case Base Sequences\n"
        title_ = "{}CB: {}\n".format(title_, fn_wo_ext)
        title_ = "{}Time-window width:{}, step:{},  Update:{}".format(title_, run_common.time_window_width_str(width), step, upd_ind)
        plt.title(title_ + "\n")
    if signature:
        plt_common.sign_plot(plt, cb_sequences.__name__)
    plt.tight_layout()
    save_fn = "CB_SEQUENCES_{}_w_{}_s_{}_seqs_{}_u_{}{}".format(fn_wo_ext,
                                                                width,
                                                                step,
                                                                str(seqs),
                                                                upd_ind,
                                                                "_t" if with_title else "")
    if file_format:
        save_fpath = os.path.join(common.APP.FOLDER.FIGURE, "{}.{}".format(save_fn, file_format))
        plt.savefig(save_fpath, dpi=150, bbox_inches="tight")
        print("CB sequences figure saved into '{}'.".format(save_fpath))
    else:
        # Update the title of the plot window
        plt.gcf().canvas.set_window_title(save_fn)
        plt.show()
    plt.close()
    return seq_ind


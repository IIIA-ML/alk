"""Script for Anytime Lazy KNN plots

usage: fig.py [-h] [-p {g,qm,i,p,e}] [-f {pdf,png,ps,eps,svg}] [--dir DIR]
              [--kwargs [KWARGS [KWARGS ...]]]
              experiments [experiments ...]

positional arguments:
  experiments           Experiment result file path(s)

optional arguments:
  -h, --help            show this help message and exit
  -p {g,qm,i,p,e}, --plot {g,qm,i,p,e}
                        Plot type: g: gains, qm: quality map, i: calc
                        insights, p: PDP, e: confidence efficiency (default:
                        g)
  -f {pdf,png,ps,eps,svg}, --fileformat {pdf,png,ps,eps,svg}
                        File format for saving plots. If not supplied, plot is
                        not saved. (default: None)
  --dir DIR             Experiment results folder (default: None)
  --kwargs [KWARGS [KWARGS ...]]
                        kwargs entered in arg=value format to be passed to the
                        plot func (default: None)

Examples:
    # --- Gain ---
    $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1.pk -p g -f png --kwargs marker_size=0.5
    # -- Calc Insights --
    $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1.pk -p i -f png --kwargs total=True actual=True all_k=True all_ticks=False with_title=True signature=True marker_size=0
    # --- Quality Map ---
    # kNN[2], Update=10, save to PNG file, by filling in between calcs where kNN[2] has changed, culling 60% of data points
    $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1.pk -p qm --kwargs with_title=False signature=False colored=False urange="(10, 10)" ki=2 start_calc=3 ustep=1 fill=True q_full_scale=False cull=.6
    # kNN[2], Update=[10, 12], in color, without filling in between calcs where kNN[2] has changed, without culling data points, full quality scale [0., 1.]
    $ python -m alk.run.fig ~/Dev/alk/results/INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1.pk -p qm --kwargs with_title=True signature=True colored=True urange="(10, 12)" ki=2 start_calc=3 ustep=1 fill=True q_full_scale=True cull=.6
    # --- PDP ---
    # Export to LaTeX
    $ python -m alk.run.fig ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1__cs_0.03_qs_0.025.pk -p p --kwargs update=10 ki=2 to_latex=True decimals=3 start_q=.75
    # As figure
    $ python -m alk.run.fig ~/Dev/alk/pdps/PDP_INS_SwedishLeaf_TEST_w_40_s_10_k_9_t_0.1__cs_0.03_qs_0.025.pk -p p -f png --kwargs update=10 ki=2 to_latex=False decimals=3 start_q=.75
    # -- Confidence Efficiency --
    $ python -m alk.run.fig "INT_SwedishLeaf_TRAIN_w_0_s_1_k_9_r_td_PDP_SwedishLeaf_TEST_c_0.0025_q_0.05__ct_[0.98_0.95_0.92_0.9_0.85_0.8_0.75_0.7]_z_-1_t_0.01.pk" "INT_SwedishLeaf_TRAIN_w_0_s_10_k_9_r_td_PDP_SwedishLeaf_TEST_c_0.0025_q_0.05__ct_[0.98_0.95_0.92_0.9_0.85_0.8_0.75_0.7]_z_-1_t_0.01.pk" "INT_SwedishLeaf_TRAIN_w_40_s_1_k_9_r_td_PDP_SwedishLeaf_TEST_c_0.0025_q_0.05__ct_[0.98_0.95_0.92_0.9_0.85_0.8_0.75_0.7]_z_-1_t_0.01.pk" "INT_SwedishLeaf_TRAIN_w_40_s_10_k_9_r_td_PDP_SwedishLeaf_TEST_c_0.0025_q_0.05__ct_[0.98_0.95_0.92_0.9_0.85_0.8_0.75_0.7]_z_-1_t_0.01.pk" -p e -f png --dir ~/Dev/alk/results/ --kwargs maximized=False with_title=True signature=True outliers=False aspect=.5

"""

import argparse
import os
import sys

from alk import helper
from alk.plot import plt_insights, plt_pdp, plt_intrpt


PLOT_FUNC_DICT = {"g": {"func": plt_insights.gains_multiple, "help": "gains"},
                  "qm": {"func": plt_insights.quality_map, "help": "quality map"},
                  "i": {"func": plt_insights.insights_multiple, "help": "calc insights"},
                  "p": {"func": plt_pdp.pdp, "help": "PDP"},
                  "e": {"func": plt_intrpt.efficiency, "help": "confidence efficiency"}}

PLOT_FILE_FORMATS = ["pdf", "png", "ps", "eps", "svg"]


def main(argv=None):
    plot_func_choices = list(PLOT_FUNC_DICT.keys())
    plot_func_help = ", ".join(["{}: {}".format(f, PLOT_FUNC_DICT[f]["help"]) for f in PLOT_FUNC_DICT.keys()])
    # Args parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiments", nargs="+", type=str,
                        help="Experiment result file path(s)")
    parser.add_argument("-p", "--plot", choices=plot_func_choices,
                        default=plot_func_choices[0],
                        help="Plot type: " + plot_func_help)
    parser.add_argument("-f", "--fileformat", choices=PLOT_FILE_FORMATS,
                        default=None,
                        help="File format for saving plots. If not supplied, plot is not saved.")
    parser.add_argument("--dir", help="Experiment results folder")
    parser.add_argument("--kwargs",  nargs="*", type=str, help="kwargs entered in arg=value format to be passed to the plot func")

    # Parse the args
    args = parser.parse_args(argv)
    if args.dir:
        dir_results = os.path.expanduser(args.dir)
        experiments = [os.path.join(dir_results, e) for e in args.experiments]
    else:
        experiments = [os.path.expanduser(e) for e in args.experiments]
    plot_type = args.plot
    plot_func = PLOT_FUNC_DICT[plot_type]["func"]
    file_format = args.fileformat
    kwargs = {}
    if args.kwargs:
        for kwa in args.kwargs:
            # Try to convert a string to its real type
            k = kwa.split("=")[0]
            v = kwa.split("=")[1]
            kwargs[k] = helper.eval_str(v)

    # Check
    # print("plot_type: {}, plot_func: {}".format(plot_type, plot_func.__name__))
    # print("to_file:{}, file_format: {}, kwargs: {}".format(to_file, file_format, kwargs))
    # for e in experiments:
    #     print(e)
    # return

    # TODO: Find a better way to dispatch kwargs to single experiment expecting plots.
    plot_func(experiments if plot_type not in ["qm", "qp", "p", "cb", "ir"] else experiments[0],
              file_format=file_format,
              **kwargs)


if __name__ == '__main__':
    main(sys.argv[1:])

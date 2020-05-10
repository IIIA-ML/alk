"""Common plotting funcionality"""

import math


def sign_plot(plt, renderer, x=0.99, y=0.01):
    """Adds 'rendered by' text to figure.

    Args:
        plt (matplotlib.pyplot):
        renderer (str): plotting function name
        x (float): x position to place the text
        y (float): y position to place the text

    """
    plt.figtext(x, y, "rendered by \'{}\'.".format(renderer), horizontalalignment='right',
                alpha=0.5, size="small")


def get_ticks_log_scale(max_val, start=1):
    """Get ticks for log10 scale.

    Args:
        max_val (int): Maximum value for which to create the log scale ticks
        start (int): The start value (not the exponent) for the ticks

    Returns:

    """
    start_exp = math.floor(math.log10(start))
    end_exp = math.ceil(math.log10(max_val))
    ticks = [10 ** i for i in range(start_exp, end_exp + 1)]
    return ticks

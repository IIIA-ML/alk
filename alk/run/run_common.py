"""Library of common functionality for Anytime Lazy KNN scripts"""

from alk import alk

# To be used in script argument choices for the `alk.RankIterator` classes
RANK_ITER_OPTIONS = {"td": {"cls": alk.TopDownCandidates, "help": "Top-down"}}
RANK_ITER_ARG_CHOICES = list(RANK_ITER_OPTIONS.keys())
RANK_ITER_ARG_HELP = ", ".join(["{}: {}".format(c, RANK_ITER_OPTIONS[c]["help"]) for c in RANK_ITER_OPTIONS.keys()])


def get_rank_iter_option_key(cls_rank_iterator):
    """Gives the argument option for the given `RankIterator` class

    Args:
        cls_rank_iterator (alk.RankIterator): sub-class

    Raises:
        KeyError: If the given class is not within the options.

    Returns:
        str:

    """
    k = None
    for k, v in RANK_ITER_OPTIONS.items():
        if v["cls"] == cls_rank_iterator:
            break
    if k is None:
        raise KeyError("{} is not within the RankIterator options.".format(cls_rank_iterator.__class__.__name__))
    return k


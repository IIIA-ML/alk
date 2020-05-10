"""Module for shared objects among sub-packages"""

import inspect
import logging
import os
import pickle
from enum import Enum


class APP:
    """Application-wide constants"""

    class INTRPT(Enum):
        """Enumeration for defining interruption flags"""
        NONE = 0  # No interruption
        W_CALC = 1  # Intrpt w/ calc (i.e. after a given number of sim assessments)
        W_SOLN = 2  # Intrpt w/ solution (i.e. when the best-so-far kNNs guarantee an exact solution)

    class FOLDER:
        """Constants for application folders at the same level with the `alk.common` module"""
        ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # Parent folder of the parent of this module
        DATASET = os.path.join(ROOT, "datasets")
        RESULT = os.path.join(ROOT, "results")
        PDP = os.path.join(ROOT, "pdps")
        FIGURE = os.path.join(ROOT, "figures")
        LOG = os.path.join(ROOT, "logs")

    class FILE_EXT:
        """"Constants for file extensions ('.' included)."""
        PICKLE = ".pk"
        LOG = ".log"

    class SUFFIX:
        """Constants for suffixes in file names"""
        HIT = "HIT"


def public_attrs(obj):
    """Return public instance attributes of a class

    Returns:
        set:
    """
    return set(a for a in obj.__dict__.keys() if not a.startswith("_"))  # public attrs only


def dump_obj(obj, f_path):
    """Dumps the `obj` into a pickle file at `f_path`

    Args:
        obj (object): Any object
        f_path (str): Full path to the dumped pickle file.

    """
    with open(os.path.expanduser(f_path), "wb") as f:
        pickle.dump(obj, f)


def load_obj(f_path):
    """Loads the pickle file at `f_path`

    Args:
        f_path (str): Full path to the dumped pickle file.

    Returns:
        object: Dumped object

    """

    with open(os.path.expanduser(f_path), "rb") as f:
        obj = pickle.load(f)
    return obj


def file_name_wo_ext(f_path):
    """Gives the `basename` of the file without extension

    Args:
        f_path (str): Full path or only the name with extentsion of a file.

    Returns:
        str: File name w/o extension and w/o path

    """
    fn = os.path.basename(f_path)
    return os.path.splitext(fn)[0]


def gen_log_file(out_file):
    """Returns full path for the corresponding log file for an output file by copying its `basename`"""
    out_file_name_wo_ext = file_name_wo_ext(out_file)
    log_file_name = out_file_name_wo_ext + APP.FILE_EXT.LOG
    log_file = os.path.join(APP.FOLDER.LOG, log_file_name)
    return log_file


def initialize_logger(console_level=logging.INFO,
                      output_dir=None, log_file=None,
                      file_level=logging.ERROR):
    """Initializes logging handlers for the console and log file.

    If any handler already exists, it updates its levels if necessary.

    """
    logger = logging.getLogger("ALK")
    logger.setLevel(logging.DEBUG)

    file_name = os.path.join(output_dir, log_file) if (output_dir and log_file) else None
    console_handler_exists = False
    stream_handler_exists = False
    # Check if the handlers already exist
    if len(logger.handlers) > 0:
        for handler in logger.handlers:
            if file_name and isinstance(handler, logging.FileHandler):
                if handler.baseFilename == file_name:
                    stream_handler_exists = True
                    if handler.level != file_level:
                        # if the same file handler exists with a different level, set it to the new level.
                        logger.info("File logger level changed from {} to: {}".format(logging.getLevelName(handler.level), logging.getLevelName(file_level)))
                        handler.setLevel(file_level)
                else:
                    # Remove the current file handler
                    logger.removeHandler(handler)
            elif isinstance(handler, logging.StreamHandler):
                console_handler_exists = True
                if handler.level != console_level:
                    # if the stream handler exists with a different level, set it to the new level.
                    logger.info("Console logger level changed from {} to {}".format(logging.getLevelName(handler.level), logging.getLevelName(console_level)))
                    handler.setLevel(console_level)
    if not console_handler_exists:
        # create console handler
        handler = logging.StreamHandler()
        handler.setLevel(console_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("Console logger created (level={}).".format(logging.getLevelName(console_level)))
    if file_name and not stream_handler_exists:
        # create debug file handler
        handler = logging.FileHandler(file_name, "w")
        handler.setLevel(file_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("Log file created: {}, (level={})".format(file_name, logging.getLevelName(file_level)))

    return logger


def log_auto_indent(logger, level, message, indent="."):
    """Automatically indents log messages depending on the current level of call stack"""
    depth = len(inspect.getouterframes(inspect.currentframe())) - 1   # depth of call stack
    message = "{} {}".format(indent * depth, message)
    logger.log(getattr(logging, level), message)


def listdir_non_hidden(path):
    """Generator extending `os.listdir` to avoid hidden files.

    Notes:
        ref: https://stackoverflow.com/a/7099342

    Yields:
        str: File name
    """
    for f in os.listdir(os.path.expanduser(path)):
        if not f.startswith(".") or f.startswith("~"):
            yield f

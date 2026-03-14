import logging


def rank_zero_only(fn):
    return fn


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger

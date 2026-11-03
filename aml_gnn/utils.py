"""
utils.py — shared logging setup.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Console logger with the project's timestamped format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

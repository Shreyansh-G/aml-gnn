"""
graph_features.py — pure helpers used to build the transaction graph.

  temporal_split     train/val/test assignment by calendar day
  port_numbers       Multi-GNN port number of every edge
  load_pattern_types laundering typology of each transaction in Patterns.txt
"""

import itertools

import numpy as np
import pandas as pd

from . import config


def temporal_split(seconds: np.ndarray, fractions=config.SPLIT_FRACTIONS) -> np.ndarray:
    """
    Assign each transaction to train (0), val (1) or test (2) by calendar day.

    Same procedure as IBM Multi-GNN: try every pair of day boundaries and keep
    the one whose split sizes are closest to *fractions*.
    """
    target = np.array(fractions)
    day = seconds // (24 * 3600)
    per_day = np.bincount(day)
    best, best_err = None, np.inf
    for i, j in itertools.combinations(range(len(per_day)), 2):
        sizes = np.array([per_day[:i].sum(), per_day[i:j].sum(), per_day[j:].sum()])
        err = np.max(np.abs(sizes / sizes.sum() - target) / target)
        if err < best_err:
            best, best_err = (i, j), err
    i, j = best
    return np.where(day < i, 0, np.where(day < j, 1, 2)).astype(np.int8)


def port_numbers(node: np.ndarray, nbr: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    For each edge, the rank of `nbr` among `node`'s distinct neighbours,
    ordered by the time of their first transaction together.
    """
    df = pd.DataFrame({"node": node, "nbr": nbr, "t": t})
    first = df.groupby(["node", "nbr"], sort=False)["t"].min().reset_index()
    first = first.sort_values(["node", "t", "nbr"])
    first["port"] = first.groupby("node").cumcount()
    return df.merge(first, on=["node", "nbr"], how="left")["port"].to_numpy(np.int32)


def load_pattern_types(path) -> dict:
    """
    Map each transaction line listed in Patterns.txt to its typology.

    The file is a sequence of blocks:
        BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
        <transaction line, same format as the CSV>
        ...
        END LAUNDERING ATTEMPT - FAN-OUT
    """
    types, current = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                current = line.split(" - ", 1)[1].split(":")[0].strip()
            elif line.startswith("END LAUNDERING ATTEMPT"):
                current = None
            elif line and current:
                types[line] = current
    return types

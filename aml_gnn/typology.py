"""
typology.py — explain *what kind* of laundering a case looks like.

Rule-based matching against the eight patterns of the IBM AMLworld generator,
using the shape of the case's transaction graph (self-transfers ignored,
parallel transactions collapsed):

  CYCLE           funds return to the account they started from
  FAN-OUT         one account pays many            (no intermediaries)
  FAN-IN          many accounts pay one            (no intermediaries)
  BIPARTITE       many pay many directly           (no intermediaries)
  SCATTER-GATHER  one source → several intermediaries → one target
  GATHER-SCATTER  many → one hub → many
  STACK           layers of accounts passing funds on (stacked bipartite)
  RANDOM          funds hop along a chain of accounts

Each match comes with a plain-English evidence sentence for the investigator.
"""

from typing import Tuple

import networkx as nx
import pandas as pd

TYPOLOGIES = ["CYCLE", "FAN-OUT", "FAN-IN", "BIPARTITE", "SCATTER-GATHER",
              "GATHER-SCATTER", "STACK", "RANDOM"]


def detect_typology(txs: pd.DataFrame, labels: pd.Series = None) -> Tuple[str, str]:
    """(typology, evidence) for transactions with `src` and `dst` account ids."""
    def name(node):
        return labels.get(node, str(node)) if labels is not None else str(node)

    g = nx.DiGraph()
    g.add_edges_from((s, d) for s, d in zip(txs["src"], txs["dst"]) if s != d)
    if g.number_of_edges() == 0:
        return "SELF-TRANSFER", "Only transfers between an account and itself."

    try:
        cycle = [u for u, _ in nx.find_cycle(g)]
        return "CYCLE", (f"Funds move in a closed loop through {len(cycle)} accounts "
                         f"and return to {name(cycle[0])}.")
    except nx.NetworkXNoCycle:
        pass

    in_deg, out_deg = dict(g.in_degree()), dict(g.out_degree())
    sources = [n for n in g if in_deg[n] == 0]
    sinks = [n for n in g if out_deg[n] == 0]
    middle = [n for n in g if in_deg[n] > 0 and out_deg[n] > 0]

    if not middle:                                     # one hop: who pays whom
        if len(sources) == 1 and len(sinks) >= 2:
            return "FAN-OUT", f"{name(sources[0])} sends funds to {len(sinks)} different accounts."
        if len(sinks) == 1 and len(sources) >= 2:
            return "FAN-IN", f"{name(sinks[0])} receives funds from {len(sources)} different accounts."
        if len(sources) >= 2 and len(sinks) >= 2:
            return "BIPARTITE", (f"{len(sources)} sender accounts pay {len(sinks)} receiver "
                                 "accounts directly, with no intermediaries.")
        return "PAIR", f"A single transfer from {name(sources[0])} to {name(sinks[0])}."

    # One source splits through intermediaries that (at least partly) converge again
    target = max(sinks, key=in_deg.get)
    if len(sources) == 1 and len(middle) >= 2 and in_deg[target] >= 2:
        return "SCATTER-GATHER", (f"{name(sources[0])} splits funds through {len(middle)} "
                                  f"intermediary accounts that converge on {name(target)}.")
    hubs = [n for n in middle if in_deg[n] >= 2 and out_deg[n] >= 2]
    if len(hubs) == 1:
        hub = hubs[0]
        return "GATHER-SCATTER", (f"{name(hub)} collects from {in_deg[hub]} accounts and "
                                  f"redistributes to {out_deg[hub]} accounts.")
    if len(sources) >= 2 and len(sinks) >= 2 and len(middle) >= 2:
        layers = nx.dag_longest_path_length(g) + 1
        return "STACK", (f"Funds pass through {layers} layers of accounts "
                         f"({len(sources)} senders, {len(middle)} intermediaries, {len(sinks)} receivers).")
    hops = nx.dag_longest_path_length(g)
    return "RANDOM", f"Funds hop along a chain of {hops + 1} accounts ({hops} transfers)."

import numpy as np

from aml_gnn.graph_features import load_pattern_types, port_numbers, temporal_split

DAY = 24 * 3600


def test_temporal_split_cuts_at_day_boundaries():
    seconds = np.repeat(np.arange(10) * DAY, 100)        # 10 days, 100 transactions each
    split = temporal_split(seconds)
    assert np.bincount(split).tolist() == [600, 200, 200]
    for d in range(10):                                   # a day never straddles two splits
        assert len(set(split[d * 100:(d + 1) * 100])) == 1


def test_port_numbers_follow_first_contact_order():
    # account 0 pays 5 (t=1), 7 (t=2), 5 again (t=3), then 9 (t=4)
    node, nbr, t = np.array([0, 0, 0, 0]), np.array([5, 7, 5, 9]), np.array([1, 2, 3, 4])
    assert port_numbers(node, nbr, t).tolist() == [0, 1, 0, 2]


def test_load_pattern_types(tmp_path):
    path = tmp_path / "patterns.txt"
    path.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out\n"
        "a,b\nc,d\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops\n"
        "e,f\n"
        "END LAUNDERING ATTEMPT - CYCLE\n"
    )
    assert load_pattern_types(path) == {"a,b": "FAN-OUT", "c,d": "FAN-OUT", "e,f": "CYCLE"}

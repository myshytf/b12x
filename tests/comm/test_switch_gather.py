from collections import Counter

import pytest

from b12x.comm.pcie._switch_gather import normalize_switch_groups, packet_routes


@pytest.mark.parametrize(
    "groups", (((0, 1, 2, 3, 8), (4, 5, 6, 7)), ((8, 7, 6, 5), (4, 3, 2, 1, 0)))
)
def test_switch_tree_delivers_each_packet_once_without_reading_unpublished_data(groups):
    groups = normalize_switch_groups(groups, 9)
    tx, rx, cut = Counter(), Counter(), Counter()
    count = 1280
    for source in range(9):
        for packet in range(count):
            available = {source}
            first, second = packet_routes(groups, source, packet)
            for stage in (first, second):
                before = available.copy()
                for sender, receiver in stage:
                    assert sender in before
                    assert receiver not in available
                    available.add(receiver)
                    tx[sender] += 1
                    rx[receiver] += 1
                    a = next(i for i, g in enumerate(groups) if sender in g)
                    b = next(i for i, g in enumerate(groups) if receiver in g)
                    if a != b:
                        cut[(a, b)] += 1
            assert available == set(range(9))
            assert len(first) + len(second) == 8
    assert set(rx.values()) == {8 * count}
    assert max(tx.values()) == 8.0625 * count
    assert sorted(cut.values()) == [5 * count, 5.25 * count]


@pytest.mark.parametrize(
    "groups", ("0,1,2,3;4,5,6,7", "0,1,2,3,8;4,5,6,6", "0,1,2;3,4,5;6,7,8")
)
def test_switch_group_contract_rejects_missing_duplicate_and_unsupported_groups(groups):
    with pytest.raises(ValueError, match="partition"):
        normalize_switch_groups(groups, 9)

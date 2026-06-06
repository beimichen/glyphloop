"""Leakage-safe splitting — the rail that kept self-training honest."""

from __future__ import annotations

import pytest

from glyphloop.data.splits import (
    assert_test_is_human_verified,
    split_by_group,
)


def _items():
    # 200 items across 40 groups (5 variants each) — variants share a group key.
    return [{"id": f"{g}-{v}", "group": f"scene{g}"} for g in range(40) for v in range(5)]


def test_no_group_straddles_the_split():
    split = split_by_group(_items(), lambda it: it["group"], test_frac=0.25)
    train_groups = {it["group"] for it in split.train}
    test_groups = {it["group"] for it in split.test}
    assert train_groups.isdisjoint(test_groups)
    assert len(split) == 200


def test_split_is_deterministic():
    a = split_by_group(_items(), lambda it: it["group"], test_frac=0.25)
    b = split_by_group(_items(), lambda it: it["group"], test_frac=0.25)
    assert [it["id"] for it in a.test] == [it["id"] for it in b.test]


def test_split_fraction_is_in_the_right_ballpark():
    split = split_by_group(_items(), lambda it: it["group"], test_frac=0.25)
    assert 0.1 < len(split.test) / len(split) < 0.4


def test_assert_test_is_human_verified_rejects_pseudo_labels():
    test_items = [{"id": "a", "pseudo": False}, {"id": "b", "pseudo": True}]
    with pytest.raises(ValueError, match="leaked"):
        assert_test_is_human_verified(test_items, lambda it: it["pseudo"])


def test_assert_test_is_human_verified_passes_clean_set():
    test_items = [{"id": "a", "pseudo": False}]
    # must not raise
    assert_test_is_human_verified(test_items, lambda it: it["pseudo"])

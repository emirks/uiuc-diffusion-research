"""Unit tests for scripts/store_per_gen.py — the per-generation transport metric.

Synthetic items.jsonl + a known ceiling exercise the load-bearing pieces: the item-id split, the
pool-row dedup + app_ref pooling + per-field reduction of collect_generations, and the capped
pooled-% arithmetic (transport). No repo data, no GPU. Run: pytest tests/test_store_per_gen.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import store_per_gen as spg  # noqa: E402


def test_parse_item_id():
    # <grid item_id>__s<seed>__ref_<poolstem>  ->  (grid item_id, seed)
    assert spg.parse_item_id(
        "G-fit__ic_gen_neutral_v3__animalization_3__ref_animalization_1__s42__ref_animalization_0"
    ) == ("G-fit__ic_gen_neutral_v3__animalization_3__ref_animalization_1", 42)


def test_transport_uncapped():
    pct, capped = spg.transport(0.30, 0.60)   # ratio 0.5
    assert pct == 50.0 and capped is False


def test_transport_capped():
    pct, capped = spg.transport(0.50, 0.40)   # ratio 1.25 -> capped at 100
    assert pct == 100.0 and capped is True


def test_transport_exactly_at_ceiling():
    pct, capped = spg.transport(0.50, 0.50)   # ratio 1.0 -> 100 but NOT flagged capped
    assert pct == 100.0 and capped is False


def _write_items(tmp_path: Path):
    """Two generations of one grid item across two seeds, plus a dup row and an app_ref=None row."""
    base = "G-x__arm__ep__ref_r1"
    rows = [
        # ITEM_A seed 42: three pool refs; copy_max varies; near_copy on one; one ref out-of-population
        {"item_id": f"{base}__s42__ref_a", "app_ref": 0.4, "copy_max": 0.1, "near_copy": False,
         "cam_zpr": 0.2, "obj_csls": 0.10, "prefix_seam_z": 1.5, "cross_high": True,
         "ref_in_v4_population": True},
        {"item_id": f"{base}__s42__ref_b", "app_ref": 0.5, "copy_max": 0.2, "near_copy": False,
         "cam_zpr": 0.4, "obj_csls": 0.20, "prefix_seam_z": 1.5, "cross_high": True,
         "ref_in_v4_population": True},
        {"item_id": f"{base}__s42__ref_c", "app_ref": 0.6, "copy_max": 0.3, "near_copy": True,
         "cam_zpr": 0.6, "obj_csls": 0.30, "prefix_seam_z": 1.5, "cross_high": True,
         "ref_in_v4_population": False},
        # a duplicate of the first row (same full item_id) — must be counted once (dedup)
        {"item_id": f"{base}__s42__ref_a", "app_ref": 0.4, "copy_max": 0.1, "near_copy": False,
         "cam_zpr": 0.2, "obj_csls": 0.10, "prefix_seam_z": 1.5, "cross_high": True,
         "ref_in_v4_population": True},
        # an app_ref=None row — dropped from the pool entirely
        {"item_id": f"{base}__s42__ref_d", "app_ref": None, "copy_max": 0.9, "near_copy": True,
         "prefix_seam_z": 1.5, "ref_in_v4_population": True},
        # ITEM_A seed 43: two pool refs
        {"item_id": f"{base}__s43__ref_a", "app_ref": 0.2, "copy_max": 0.05, "near_copy": False,
         "cam_zpr": 0.1, "obj_csls": 0.05, "prefix_seam_z": 2.5, "cross_high": False,
         "ref_in_v4_population": True},
        {"item_id": f"{base}__s43__ref_b", "app_ref": 0.4, "copy_max": 0.15, "near_copy": False,
         "cam_zpr": 0.3, "obj_csls": 0.15, "prefix_seam_z": 2.5, "cross_high": False,
         "ref_in_v4_population": True},
    ]
    shard = tmp_path / "c0"
    shard.mkdir()
    (shard / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return base


def test_collect_generations(tmp_path):
    import statistics as st

    base = _write_items(tmp_path)
    gens = spg.collect_generations(tmp_path)

    # two generations (one grid item x two seeds); the app_ref=None row produced no extra generation
    assert set(gens) == {(base, 42), (base, 43)}

    a = gens[(base, 42)]
    # dedup: ref_a counted once -> three pooled app_ref values, not four
    assert sorted(a["app_ref"]) == [0.4, 0.5, 0.6]
    # MEAN fields averaged over the (deduped) pool references
    assert abs(st.mean(a["mean"]["copy_max"]) - 0.2) < 1e-12
    assert abs(st.mean(a["mean"]["cam_zpr"]) - 0.4) < 1e-12
    # ANY field: near_copy True because one reference tripped it
    assert any(a["any"]["near_copy"]) is True
    # CONST field taken verbatim
    assert a["const"]["prefix_seam_z"] == 1.5
    assert a["const"]["cross_high"] is True
    # ref_in_v4_population reduced with all(): False because ref_c is out-of-population
    assert all(a["ref_in_pop"]) is False

    b = gens[(base, 43)]
    assert sorted(b["app_ref"]) == [0.2, 0.4]
    assert all(b["ref_in_pop"]) is True


def test_transport_from_collected(tmp_path):
    base = _write_items(tmp_path)
    gens = spg.collect_generations(tmp_path)
    import statistics as st

    ceil = 0.4  # deliberately below seed-42's mean (0.5) so it caps, above seed-43's mean (0.3)
    raw42 = st.mean(gens[(base, 42)]["app_ref"])   # 0.5
    raw43 = st.mean(gens[(base, 43)]["app_ref"])   # 0.3
    assert spg.transport(raw42, ceil) == (100.0, True)   # 0.5/0.4 = 1.25 -> capped
    pct43, capped43 = spg.transport(raw43, ceil)         # 0.3/0.4 = 0.75
    assert capped43 is False and abs(pct43 - 75.0) < 1e-9


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

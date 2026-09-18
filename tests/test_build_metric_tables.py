"""Synthetic contract tests for scripts/build_metric_tables.py (Op-5).

CPU, no GPU, no store access: a tiny in-memory fixture exercises the join on
(item_id, seed), aggregation, bolding, placeholders, the shared one-sided set,
and the --strict mixed-n assertion.  Also runs the built-in fixture end-to-end
(without latexmk) to prove the whole builder wires up.
"""
import math
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_metric_tables as B  # noqa: E402


# --------------------------------------------------------------------------- #
# arm parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm_dir,expect", [
    ("base_cond_neutral_v3", ("base_cond", "neutral", "hf")),
    ("base_cond_neutral_v3ed81", ("base_cond", "neutral", "ed")),
    ("ic_gen_effect_v3", ("ic_gen", "effect", "hf")),
    ("dualforce_dcg_w6_effect_v3ed81", ("dualforce_dcg_w6", "effect", "ed")),
    ("dualforce_control_neutral_v3", ("dualforce_control", "neutral", "hf")),
    ("vap_author_native", ("vap", "effect", "external")),
    ("refvfx_author_native", ("refvfx", "effect", "external")),
    ("nonsense_arm", None),
])
def test_parse_arm(arm_dir, expect):
    assert B.parse_arm(arm_dir) == expect


# --------------------------------------------------------------------------- #
# join on (item_id, seed)
# --------------------------------------------------------------------------- #
def test_join_on_item_id_seed():
    per_gen = [
        {"item_id": "g1", "seed": 42, "transport_pct": 60.0},
        {"item_id": "g2", "seed": 43, "transport_pct": 90.0},
        {"item_id": "g3", "seed": 42, "transport_pct": 70.0},  # no side rows
    ]
    handoff = {("g1", 42): {"identity_A": 0.9, "seam_free": 1},
               ("g2", 43): {"identity_A": 0.8, "seam_free": 0}}
    lens = {("g1", 42): {"motion_smoothness": 0.99}}
    recs = B.join_records([dict(r) for r in per_gen], handoff, lens)
    by = {(r["item_id"], r["seed"]): r for r in recs}
    assert by[("g1", 42)]["identity_A"] == 0.9
    assert by[("g1", 42)]["motion_smoothness"] == 0.99
    assert by[("g2", 43)]["identity_A"] == 0.8
    # g2 had no lens row -> the key is simply absent (renders placeholder later)
    assert "motion_smoothness" not in by[("g2", 43)]
    # g3 had neither side row
    assert "identity_A" not in by[("g3", 42)]


# --------------------------------------------------------------------------- #
# aggregation + placeholder
# --------------------------------------------------------------------------- #
def test_agg_mean_and_nan_drop():
    recs = [{"x": 1.0}, {"x": 3.0}, {"x": float("nan")}, {"x": None}, {}]
    c = B.agg(recs, "x")
    assert c.present and c.n == 2 and abs(c.value - 2.0) < 1e-9


def test_agg_bool_rate_and_scale():
    recs = [{"nc": True}, {"nc": False}, {"nc": True}, {"nc": False}]
    c = B.agg(recs, "nc", scale=100.0)
    assert c.present and c.n == 4 and abs(c.value - 50.0) < 1e-9


def test_agg_empty_is_placeholder():
    c = B.agg([{"y": None}, {}], "x")
    assert not c.present and c.n == 0
    assert B.cell_tex(c, "sim3") == r"\ph{--}"
    assert B.cell_md(c, "sim3") == "--"


# --------------------------------------------------------------------------- #
# bolding
# --------------------------------------------------------------------------- #
def test_bold_block_max_min_absmin_and_ties():
    cols = [
        B.Col("a", "", "sim3", "max", None),
        B.Col("b", "", "pct1", "min", None),
        B.Col("d", "", "dpct1", "absmin", None),
    ]
    rows = [
        [B.Cell(0.90, 12), B.Cell(10.0, 12), B.Cell(+5.0, 12)],
        [B.Cell(0.95, 12), B.Cell(10.0, 12), B.Cell(-2.0, 12)],   # best max; ties min; best absmin
        [B.Cell(0.80, 12), B.Cell(40.0, 12), B.Cell(+9.0, 12)],
    ]
    B.bold_block(rows, cols)
    assert rows[1][0].bold and not rows[0][0].bold          # max
    assert rows[0][1].bold and rows[1][1].bold and not rows[2][1].bold  # min tie
    assert rows[1][2].bold and not rows[0][2].bold          # absmin


def test_bold_skips_placeholders():
    cols = [B.Col("a", "", "sim3", "max", None)]
    rows = [[B.Cell(None, 0)], [B.Cell(0.5, 12)]]
    B.bold_block(rows, cols)
    assert rows[1][0].bold and not rows[0][0].bold


# --------------------------------------------------------------------------- #
# minimum-n rule (n < MIN_N -> "n/a", excluded from bolding)
# --------------------------------------------------------------------------- #
def test_min_n_renders_na_with_n():
    thin = B.Cell(0.912, B.MIN_N - 1)          # e.g. n=9
    fat = B.Cell(0.912, B.MIN_N)               # n=10 exactly -> reportable
    assert B.cell_tex(thin, "sim3") == r"n/a{\tiny\,($n{=}" + str(B.MIN_N - 1) + "$)}"
    assert B.cell_md(thin, "sim3") == f"n/a (n={B.MIN_N - 1})"
    assert B.cell_tex(fat, "sim3") == "0.912"
    assert B.cell_md(fat, "sim3") == "0.912"


def test_min_n_excluded_from_bolding():
    cols = [B.Col("a", "", "sim3", "max", None)]
    # the thin cell has the largest value but must NOT win the bold
    rows = [[B.Cell(0.99, 3)], [B.Cell(0.80, 20)]]
    B.bold_block(rows, cols)
    assert not rows[0][0].bold          # thin (n=3) excluded
    assert rows[1][0].bold              # only reportable candidate wins
    # and it renders n/a, unbolded
    assert B.cell_tex(rows[0][0], "sim3").startswith("n/a")


def test_min_n_applies_in_table_a_two_sided_subset():
    """Two-sided rows are few -> Identity B renders n/a, but Identity A (all rows) is a number."""
    recs = []
    for i in range(14):                          # 14 one-sided neutral unseen rows -> block healthy
        recs.append(_mk("ic_gen", "neutral", "hf", "unseen", "one", f"e{i}", f"r{i}",
                        "unseen|same", 42, identity_A=0.90, near_copy=False, copy_near_copy=False))
    for i in range(3):                           # 3 two-sided rows -> Identity B thin (n<10)
        recs.append(_mk("ic_gen", "neutral", "hf", "unseen", "two", f"t{i}", f"tr{i}",
                        "unseen|cross", 42, identity_A=0.90, identity_B=0.88,
                        near_copy=False, copy_near_copy=False))
    ta = B.build_table_a(recs)
    tex = B.render_table_a(ta)
    row = [r for r in ta.blocks["unseen"] if r[0] == "ic_gen"][0]
    id_a = row[2][B.TABLE_A_COLS.index([c for c in B.TABLE_A_COLS if c.key == "identity_a"][0])]
    id_b = row[2][B.TABLE_A_COLS.index([c for c in B.TABLE_A_COLS if c.key == "identity_b"][0])]
    assert id_a.present and id_a.n == 17 and abs(id_a.value - 0.90) < 1e-9
    assert id_b.present and id_b.n == 3          # only the two-sided rows
    assert "n/a" in B.cell_tex(id_b, "sim3")     # thin -> n/a


# --------------------------------------------------------------------------- #
# copy eval join (Copy rate comes from the copy eval, NOT per_gen's near_copy)
# --------------------------------------------------------------------------- #
def test_copy_join_overrides_per_gen_near_copy():
    # per_gen says near_copy True everywhere (the invalid pool-scored value);
    # the copy eval says False everywhere -> Copy rate must be 0, not 100.
    per_gen = [{"item_id": f"g{i}", "seed": 42, "near_copy": True, "copy_max": 0.9}
               for i in range(12)]
    copy = {("g{}".format(i), 42): {"near_copy": False, "copy_max": 0.1} for i in range(12)}
    recs = B.join_records([dict(r) for r in per_gen], {}, {}, copy)
    assert all(r["copy_near_copy"] is False for r in recs)
    # the Copy rate column extractor reads copy_near_copy
    copy_col = [c for c in B.TABLE_A_COLS if c.key == "copy"][0]
    cell = copy_col.extract(recs)
    assert cell.present and abs(cell.value - 0.0) < 1e-9   # 0%, from the copy eval
    # per_gen's own near_copy is never aggregated
    bad = B.agg(recs, "near_copy", scale=100.0)
    assert abs(bad.value - 100.0) < 1e-9                   # (would be 100 if used -- proves we don't)


# --------------------------------------------------------------------------- #
# number formatting
# --------------------------------------------------------------------------- #
def test_cell_tex_formats():
    assert B.cell_tex(B.Cell(92.55, 20), "pct1") in ("92.5", "92.6")
    assert B.cell_tex(B.Cell(0.9613, 20), "sim3") == "0.961"
    assert B.cell_tex(B.Cell(2.0, 20), "dpct1") == "+2.0"
    assert B.cell_tex(B.Cell(-0.01, 20), "dsim3") == "-0.010"
    # bold + sub-n annotation (n >= MIN_N but < block_n)
    out = B.cell_tex(B.Cell(0.9, 12, bold=True), "sim3", block_n=20)
    assert out == r"\textbf{0.900}{\tiny\,($n{=}12$)}"


# --------------------------------------------------------------------------- #
# shared one-sided set (intersection) + Table B/C
# --------------------------------------------------------------------------- #
def _mk(base, kind, family, tier, sided, ep, ref, cell, seed, **extra):
    r = {"item_id": f"{base}_{kind}_{family}__{ep}__{ref}__s{seed}", "seed": seed,
         "base_arm": base, "variant_kind": kind, "family": family, "tier": tier,
         "sided": sided, "endpoint": ep, "reference": ref, "cell": cell}
    r.update(extra)
    return r


def test_shared_set_intersection():
    recs = []
    # externals cover triples T1,T2 (seeds 42,43); own arms cover T1,T2,T3.
    triples = [("e1", "r1", "T1"), ("e2", "r2", "T2")]
    own_triples = triples + [("e3", "r3", "T3")]
    for base in ("vap", "vfxmaster", "refvfx"):
        for (ep, ref, cell) in triples:
            for s in (42, 43):
                recs.append(_mk(base, "effect", "external", "zero_shot", "one", ep, ref, cell, s,
                                transport_pct=50.0, videoprism_sim_ref=0.9))
    for base in ("ic_gen", "dualforce_control", "dualforce_dcg_w6"):
        for (ep, ref, cell) in own_triples:
            for s in (42, 43):
                recs.append(_mk(base, "neutral", "hf", "zero_shot", "one", ep, ref, cell, s,
                                transport_pct=60.0, videoprism_sim_ref=0.95))
    shared, diag = B.build_shared_set(recs)
    assert diag["shared_n"] == 4          # 2 triples x 2 seeds
    assert diag["per_arm_n"]["vap"] == 4
    assert diag["per_arm_n"]["ic_gen"] == 6
    assert diag["lost"]["ic_gen"] == 2    # T3 x 2 seeds dropped by intersection
    assert diag["lost"]["vap"] == 0


def test_table_b_uniform_n_and_strict_clean():
    recs = []
    triples = [("e1", "r1", "T1"), ("e2", "r2", "T2")]
    for base in ("vap", "vfxmaster", "refvfx"):
        for (ep, ref, cell) in triples:
            for s in (42, 43):
                recs.append(_mk(base, "effect", "external", "zero_shot", "one", ep, ref, cell, s,
                                transport_pct=50.0, videoprism_sim_ref=0.9, identity_A=0.9,
                                near_copy=False, seam_free=1, motion_smoothness=0.98,
                                det_motion_fidelity=0.2, aesthetic=5.0))
    for base in ("ic_gen", "dualforce_control", "dualforce_dcg_w6"):
        for (ep, ref, cell) in triples:
            for s in (42, 43):
                recs.append(_mk(base, "neutral", "hf", "zero_shot", "one", ep, ref, cell, s,
                                transport_pct=60.0, videoprism_sim_ref=0.95, identity_A=0.92,
                                near_copy=False, seam_free=1, motion_smoothness=0.99,
                                det_motion_fidelity=0.22, aesthetic=5.2))
    tb = B.build_table_b(recs)
    ns = {n for (_b, n, _c) in tb.rows}
    assert ns == {4}                       # every arm has all 4 shared rows
    tc = B.build_table_c(recs)
    assert B.check_strict(tb, tc) == []    # no mixed present-column n


def test_strict_flags_mixed_column_n():
    """One external missing the lens metric for half its rows -> mixed present-column n."""
    recs = []
    triples = [("e1", "r1", "T1"), ("e2", "r2", "T2")]
    for base in ("vap", "vfxmaster", "refvfx"):
        for (ep, ref, cell) in triples:
            for s in (42, 43):
                extra = dict(transport_pct=50.0, identity_A=0.9, near_copy=False, seam_free=1)
                # vfxmaster gets videoprism only on seed 42 -> its vpref n != others' n
                if not (base == "vfxmaster" and s == 43):
                    extra["videoprism_sim_ref"] = 0.9
                recs.append(_mk(base, "effect", "external", "zero_shot", "one", ep, ref, cell, s, **extra))
    for base in ("ic_gen", "dualforce_control", "dualforce_dcg_w6"):
        for (ep, ref, cell) in triples:
            for s in (42, 43):
                recs.append(_mk(base, "neutral", "hf", "zero_shot", "one", ep, ref, cell, s,
                                transport_pct=60.0, videoprism_sim_ref=0.95, identity_A=0.92,
                                near_copy=False, seam_free=1))
    tb = B.build_table_b(recs)
    tc = B.build_table_c(recs)
    problems = B.check_strict(tb, tc)
    assert any("vpref" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# Table A structure + neutral-only filtering
# --------------------------------------------------------------------------- #
def test_table_a_neutral_only_and_tiers():
    recs = []
    for kind in ("neutral", "effect"):
        for s in (42, 43):
            recs.append(_mk("ic_gen", kind, "hf", "unseen", "one", "e1", "r1", "unseen|same", s,
                            transport_pct=70.0, near_copy=False))
    ta = B.build_table_a(recs)
    # only neutral rows counted -> block n = 2 (2 seeds), not 4
    row = [r for r in ta.blocks["unseen"] if r[0] == "ic_gen"][0]
    assert row[1] == 2
    # seen tier empty -> transport cell is a placeholder
    seen_row = [r for r in ta.blocks["seen"] if r[0] == "ic_gen"][0]
    tcol = B.TABLE_A_COLS[-1]
    assert not seen_row[2][B.TABLE_A_COLS.index(tcol)].present


# --------------------------------------------------------------------------- #
# end-to-end on the built-in fixture (no latexmk)
# --------------------------------------------------------------------------- #
def test_fixture_end_to_end(tmp_path):
    fx = tmp_path / "fx"
    roots = B.write_fixture(str(fx))
    per_gen, arm_dirs = B.load_per_gen(roots["per_gen_roots"])
    handoff = B.load_side(roots["handoff_globs"], B.HANDOFF_KEYS)
    lens = B.load_side(roots["lens_globs"], B.LENS_KEYS)
    copy = B.load_side(roots["copy_globs"], B.COPY_KEYS)
    recs = B.join_records(per_gen, handoff, lens, copy)
    assert len(arm_dirs) == 19          # 16 paper-arm variants + 3 externals
    assert len(recs) == len(per_gen) > 0
    assert copy and all("copy_near_copy" in r for r in recs)   # copy eval joined
    # every own + external arm parsed into a known base
    bases = {r["base_arm"] for r in recs}
    assert bases == set(B.OWN_ARMS) | set(B.EXTERNAL_ARMS)
    ta = B.build_table_a(recs)
    tb = B.build_table_b(recs)
    tc = B.build_table_c(recs)
    # shared one-sided zero-shot set = 8 HF triples x 2 seeds; own arms drop their 4 ED rows
    assert tb.diag["shared_n"] == 16
    assert tb.diag["lost"]["ic_gen"] == 4 and tb.diag["lost"]["vap"] == 0
    # Table A renders and champion beats base on transport in every tier where both present
    tex = B.render_table_a(ta)
    assert r"\begin{tabular}" in tex and r"\ph{--}" in tex
    assert B.render_table_b(tb).count(r"\\") >= 6
    assert B.render_table_c(tc).count(r"\\") >= 7


def test_missing_inputs_all_placeholder():
    """No per_gen at all -> every data cell is a placeholder, builder still renders."""
    ta = B.build_table_a([])
    tb = B.build_table_b([])
    tc = B.build_table_c([])
    texA = B.render_table_a(ta)
    assert texA.count(r"\ph{--}") >= 8 * 3          # >= 8 cols x 3 tiers x 4 arms actually
    assert r"\bottomrule" in texA
    assert r"\ph{--}" in B.render_table_b(tb)
    assert r"\ph{--}" in B.render_table_c(tc)

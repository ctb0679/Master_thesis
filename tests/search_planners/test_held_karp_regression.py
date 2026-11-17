"""Held-Karp regression: the extracted search_planners.held_karp (and the
rank_candidates re-export) must give exactly the outputs the original
rank_candidates.optimal_visit_order gave before the refactor.

held_karp_baseline.json was written by running this file as a script
(`python tests/search_planners/test_held_karp_regression.py`) against the
code BEFORE the extraction (commit a544c1a). Do not regenerate it from the
new code: that would make the test compare the new code with itself."""

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "held_karp_baseline.json"


def make_cases():
    """Deterministic cases: (name, kwargs). Distances that are not the
    default straight line are given as a matrix and turned into a callback."""
    rng = random.Random(20261117)
    cases = [("empty", {"candidates": [], "centroids": {"S": [0, 0]}, "start": "S"}),
             ("one_room", {"candidates": [{"room": "A", "probability": 0.7, "likelihood": 0.9,
                                           "reasoning": "only one"}],
                           "centroids": {"S": [0, 0], "A": [3, 4]}, "start": "S"}),
             ("start_is_candidate", {"candidates": [{"room": "S"}, {"room": "A"}],
                                     "centroids": {"S": [0, 0], "A": [1, 0]}, "start": "S"}),
             ("missing_start", {"candidates": [{"room": "A"}], "centroids": {"A": [1, 1]}, "start": "S"})]
    # ties: a square around the start, all four rooms the same distance apart
    cases.append(("ties_square", {"candidates": [{"room": r} for r in "ABCD"],
                                  "centroids": {"S": [0, 0], "A": [1, 0], "B": [0, 1], "C": [-1, 0],
                                                "D": [0, -1]}, "start": "S"}))
    cases.append(("ties_matrix", {"candidates": [{"room": r} for r in "ABC"],
                                  "centroids": {r: [0, 0] for r in "SABC"}, "start": "S",
                                  "matrix": {a: {b: (0 if a == b else 1.0) for b in "SABC"} for a in "SABC"}}))
    for n in range(2, 10):
        names = [f"R{i}" for i in range(n)]
        cents = {r: [round(rng.uniform(0, 30), 3), round(rng.uniform(0, 20), 3)] for r in ["S", *names]}
        cands = [{"room": r, "probability": round(rng.random(), 4), "likelihood": round(rng.random(), 2),
                  "reasoning": f"why {r}"} for r in names]
        cases.append((f"euclid_{n}", {"candidates": cands, "centroids": cents, "start": "S"}))
        # asymmetric, non-metric matrix through a custom callback
        nodes = ["S", *names]
        matrix = {a: {b: (0.0 if a == b else round(rng.uniform(0.5, 25), 3)) for b in nodes} for a in nodes}
        cases.append((f"matrix_{n}", {"candidates": cands, "centroids": cents, "start": "S", "matrix": matrix}))
    # greedy fallback above the exact threshold
    for n, max_exact in ((5, 3), (20, 18), (19, 18)):
        names = [f"F{i}" for i in range(n)]
        cents = {r: [round(rng.uniform(0, 50), 3), round(rng.uniform(0, 50), 3)] for r in ["S", *names]}
        cands = [{"room": r, "probability": 0.1} for r in names]
        cases.append((f"fallback_{n}_{max_exact}", {"candidates": cands, "centroids": cents, "start": "S",
                                                    "max_exact": max_exact}))
        nodes = ["S", *names]
        matrix = {a: {b: (0.0 if a == b else round(rng.uniform(1, 40), 3)) for b in nodes} for a in nodes}
        cases.append((f"fallback_matrix_{n}_{max_exact}", {"candidates": cands, "centroids": cents, "start": "S",
                                                           "max_exact": max_exact, "matrix": matrix}))
    return cases


def run_case(func, case):
    kw = dict(case)
    matrix = kw.pop("matrix", None)
    args = dict(candidates=kw["candidates"], centroid_lookup={k: tuple(v) for k, v in kw["centroids"].items()},
                start_room=kw["start"])
    if "max_exact" in kw:
        args["max_exact"] = kw["max_exact"]
    if matrix is not None:
        args["distance"] = lambda a, b: matrix[a][b]
    try:
        return {"result": func(**args)}
    except Exception as e:                       # the error itself is part of the behaviour
        return {"error": type(e).__name__, "message": str(e)}


def _check(func):
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    cases = make_cases()
    assert [name for name, _ in cases] == list(expected)
    for name, case in cases:
        got = json.loads(json.dumps(run_case(func, case)))
        assert got == expected[name], name


def test_extracted_held_karp_matches_the_pre_refactor_baseline():
    from search_planners.held_karp import optimal_visit_order
    _check(optimal_visit_order)


def test_rank_candidates_reexport_matches_the_baseline():
    from rank_candidates import optimal_visit_order
    _check(optimal_visit_order)


def test_reexport_is_the_same_function():
    import rank_candidates
    from search_planners import held_karp
    assert rank_candidates.optimal_visit_order is held_karp.optimal_visit_order
    assert rank_candidates._greedy_nearest_neighbor is held_karp._greedy_nearest_neighbor


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parents[1] / "scripts" / "vlm_reasoning"))
    from rank_candidates import optimal_visit_order
    out = {name: run_case(optimal_visit_order, case) for name, case in make_cases()}
    BASELINE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {BASELINE}: {len(out)} cases, "
          f"{sum('error' in v for v in out.values())} error cases, "
          f"max rooms {max(len(c['candidates']) for _, c in make_cases())}")
    _ = math

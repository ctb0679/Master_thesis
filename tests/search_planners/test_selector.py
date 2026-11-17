"""search_planners.selector: held_karp is the default, blr goes to the BLR
module, both get the same candidates and the same distance provider."""

import pytest

import search_planners.selector as selector
from search_planners import ALGORITHMS, DEFAULT_ALGORITHM, check_algorithm, order_rooms
from search_planners.held_karp import optimal_visit_order

CENT = {"S": (0, 0), "A": (4, 0), "B": (0, 3), "C": (5, 5)}
CANDS = [{"room": "A", "probability": 0.5, "likelihood": 0.8, "reasoning": "a"},
         {"room": "B", "probability": 0.3, "likelihood": 0.6, "reasoning": "b"},
         {"room": "C", "probability": 0.2, "likelihood": 0.4, "reasoning": "c"}]
AREAS = {"A": 10.0, "B": 20.0, "C": 5.0}


def test_default_is_held_karp():
    assert DEFAULT_ALGORITHM == "held_karp" and ALGORITHMS == ("held_karp", "blr")
    assert order_rooms(CANDS, CENT, "S") == optimal_visit_order(CANDS, CENT, "S")
    assert order_rooms(CANDS, CENT, "S", "held_karp") == optimal_visit_order(CANDS, CENT, "S")


def test_held_karp_needs_no_areas():
    assert order_rooms(CANDS, CENT, "S", "held_karp", areas=None) == optimal_visit_order(CANDS, CENT, "S")
    assert order_rooms(CANDS, CENT, "S", "held_karp", areas={}) == optimal_visit_order(CANDS, CENT, "S")


def test_invalid_name_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown search algorithm 'nearest'; choose one of held_karp, blr"):
        order_rooms(CANDS, CENT, "S", "nearest")
    with pytest.raises(ValueError):
        check_algorithm("Held_Karp")


def test_both_get_the_same_candidates_and_distance(monkeypatch):
    seen = {}

    def spy(name, real):
        def f(candidates, centroid_lookup, start_room, distance=None, **kw):
            seen[name] = (candidates, centroid_lookup, start_room, distance)
            return real(candidates, centroid_lookup, start_room, distance=distance, **kw)
        return f

    monkeypatch.setattr(selector, "optimal_visit_order", spy("held_karp", selector.optimal_visit_order))
    monkeypatch.setattr(selector, "blr_visit_order", spy("blr", selector.blr_visit_order))

    def dist(a, b):
        return 1.0 if a != b else 0.0

    hk = order_rooms(CANDS, CENT, "S", "held_karp", distance=dist, areas=AREAS)
    blr = order_rooms(CANDS, CENT, "S", "blr", distance=dist, areas=AREAS)
    assert seen["held_karp"] == seen["blr"] == (CANDS, CENT, "S", dist)
    assert sorted(c["room"] for c in hk) == sorted(c["room"] for c in blr) == ["A", "B", "C"]
    assert all("blr_score" in c for c in blr) and not any("blr_score" in c for c in hk)


def test_blr_does_not_change_the_candidates():
    before = [dict(c) for c in CANDS]
    out = order_rooms(CANDS, CENT, "S", "blr", areas=AREAS)
    assert CANDS == before
    assert {c["room"]: c["probability"] for c in out} == {c["room"]: c["probability"] for c in CANDS}


def test_example_where_the_two_algorithms_differ():
    """Deterministic example for the report. Same probabilities, areas,
    start and distances; Held-Karp minimises the total tour, BLR takes the
    best P / (d * sqrt(A)) step by step."""
    m = {"S": {"A": 10.0, "B": 4.0, "C": 6.0},
         "A": {"S": 10.0, "B": 7.0, "C": 5.0},
         "B": {"S": 4.0, "A": 7.0, "C": 3.0},
         "C": {"S": 6.0, "A": 5.0, "B": 3.0}}
    cands = [{"room": "A", "probability": 0.6}, {"room": "B", "probability": 0.25},
             {"room": "C", "probability": 0.15}]
    areas = {"A": 9.0, "B": 25.0, "C": 16.0}
    dist = lambda a, b: 0.0 if a == b else m[a][b]                    # noqa: E731
    hk = [c["room"] for c in order_rooms(cands, CENT, "S", "held_karp", distance=dist)]
    blr = [c["room"] for c in order_rooms(cands, CENT, "S", "blr", distance=dist, areas=areas)]
    assert hk == ["B", "C", "A"]          # 4 + 3 + 5 = 12 m, the shortest tour
    assert blr == ["A", "C", "B"]         # A: 0.6/(10*3)=0.020 > B: 0.25/(4*5)=0.0125 > C: 0.15/(6*4)=0.00625

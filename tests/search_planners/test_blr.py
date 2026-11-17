"""BLR (search_planners.blr): Eq. (1) H = P / (d * sqrt(A)), Algorithm 1
(greedy, re-scored from the latest chosen room), and the input checks."""

import math

import pytest

from search_planners.blr import (BLRInputError, BLRUnreachableError, blr_next_room, blr_score,
                                 blr_visit_order, check_candidates, invalid_areas)


def matrix_distance(m):
    return lambda a, b: m[a][b]


def sym(pairs, rooms):
    """Symmetric distance matrix from {(a, b): d}; 0 on the diagonal."""
    m = {a: {b: 0.0 for b in rooms} for a in rooms}
    for (a, b), d in pairs.items():
        m[a][b] = m[b][a] = d
    return m


CENT = {r: (0.0, 0.0) for r in "SABCDE"}     # only the start's presence matters with a matrix


def order(cands, m, areas, start="S"):
    return [c["room"] for c in blr_visit_order(cands, CENT, start, matrix_distance(m), areas)]


# --- Eq. (1) -----------------------------------------------------------------------

def test_formula_is_p_over_d_times_sqrt_area():
    assert blr_score(0.5, 2.0, 4.0) == 0.5 / (2.0 * 2.0)
    assert blr_score(0.3, 7.0, 10.0) == 0.3 / (7.0 * math.sqrt(10.0))
    assert blr_score(0.0, 3.0, 9.0) == 0.0


def test_score_reported_is_eq1_at_the_time_of_choice():
    m = sym({("S", "A"): 2.0, ("S", "B"): 5.0, ("A", "B"): 4.0}, "SAB")
    out = blr_visit_order([{"room": "A", "probability": 0.6}, {"room": "B", "probability": 0.4}],
                          CENT, "S", matrix_distance(m), {"A": 4.0, "B": 9.0})
    assert out[0]["blr_score"] == blr_score(0.6, 2.0, 4.0)
    assert out[1]["blr_score"] == blr_score(0.4, 4.0, 9.0)          # d from A, not from S


# --- each input matters ---------------------------------------------------------------

def test_higher_probability_first_when_distance_and_area_equal():
    m = sym({("S", "A"): 3.0, ("S", "B"): 3.0, ("A", "B"): 1.0}, "SAB")
    cands = [{"room": "A", "probability": 0.2}, {"room": "B", "probability": 0.7}]
    assert order(cands, m, {"A": 10.0, "B": 10.0}) == ["B", "A"]


def test_smaller_area_first_when_probability_and_distance_equal():
    m = sym({("S", "A"): 3.0, ("S", "B"): 3.0, ("A", "B"): 1.0}, "SAB")
    cands = [{"room": "A", "probability": 0.5}, {"room": "B", "probability": 0.5}]
    assert order(cands, m, {"A": 50.0, "B": 8.0}) == ["B", "A"]


def test_nearer_first_when_probability_and_area_equal():
    m = sym({("S", "A"): 9.0, ("S", "B"): 2.0, ("A", "B"): 1.0}, "SAB")
    cands = [{"room": "A", "probability": 0.5}, {"room": "B", "probability": 0.5}]
    assert order(cands, m, {"A": 10.0, "B": 10.0}) == ["B", "A"]


def test_uses_the_given_distance_not_straight_lines():
    """Centroids say A is near, the (navigable) distance says B is near."""
    cent = {"S": (0, 0), "A": (1, 0), "B": (10, 0)}
    m = sym({("S", "A"): 30.0, ("S", "B"): 2.0, ("A", "B"): 5.0}, "SAB")
    cands = [{"room": "A", "probability": 0.5}, {"room": "B", "probability": 0.5}]
    areas = {"A": 10.0, "B": 10.0}
    assert [c["room"] for c in blr_visit_order(cands, cent, "S", None, areas)] == ["A", "B"]
    assert [c["room"] for c in blr_visit_order(cands, cent, "S", matrix_distance(m), areas)] == ["B", "A"]


def test_scores_are_recomputed_from_the_latest_room():
    """From S: A first. Scored from S, C (3.0) would beat B (8.0) next; from A,
    B is 1 m away and C 20 m, so Algorithm 1 takes B second."""
    m = sym({("S", "A"): 1.0, ("S", "B"): 8.0, ("S", "C"): 3.0,
             ("A", "B"): 1.0, ("A", "C"): 20.0, ("B", "C"): 20.0}, "SABC")
    cands = [{"room": r, "probability": p} for r, p in (("A", 0.4), ("B", 0.3), ("C", 0.3))]
    areas = {"A": 4.0, "B": 4.0, "C": 4.0}
    from_start_only = sorted("BC", key=lambda r: -blr_score(0.3, m["S"][r], 4.0))
    assert from_start_only == ["C", "B"]
    assert order(cands, m, areas) == ["A", "B", "C"]


def test_full_order_visits_every_candidate_once():
    rooms = "ABCDE"
    m = sym({(a, b): float(1 + (ord(a) * 7 + ord(b) * 3) % 11) for a in "S" + rooms for b in "S" + rooms
             if a < b}, "S" + rooms)
    cands = [{"room": r, "probability": 0.1 + 0.1 * i} for i, r in enumerate(rooms)]
    out = order(cands, m, {r: 5.0 + i for i, r in enumerate(rooms)})
    assert sorted(out) == list(rooms)


def test_next_room_is_one_step_of_the_order():
    m = sym({("S", "A"): 2.0, ("S", "B"): 1.0, ("A", "B"): 6.0}, "SAB")
    cands = [{"room": "A", "probability": 0.8}, {"room": "B", "probability": 0.2}]
    areas = {"A": 4.0, "B": 4.0}
    c, score, d = blr_next_room(cands, "S", matrix_distance(m), areas)
    assert c["room"] == order(cands, m, areas)[0] == "A"
    assert score == blr_score(0.8, 2.0, 4.0) and d == 2.0
    c, _, _ = blr_next_room(cands, "B", matrix_distance(m), areas)   # from another planning location
    assert c["room"] == "B"                                           # zero distance first
    assert blr_next_room([], "S", matrix_distance(m), areas) is None


# --- output schema ---------------------------------------------------------------------

def test_output_keeps_fields_adds_distance_and_score_and_does_not_mutate():
    m = sym({("S", "A"): 2.345, ("S", "B"): 5.0, ("A", "B"): 1.0}, "SAB")
    cands = [{"room": "A", "probability": 0.6, "likelihood": 0.9, "reasoning": "why A"},
             {"room": "B", "probability": 0.4, "likelihood": 0.5, "reasoning": "why B"}]
    before = [dict(c) for c in cands]
    out = blr_visit_order(cands, CENT, "S", matrix_distance(m), {"A": 4.0, "B": 4.0})
    assert cands == before
    assert out[0] == {**cands[0], "distance_from_previous_m": 2.35, "blr_score": blr_score(0.6, 2.345, 4.0)}
    assert out[1]["distance_from_previous_m"] == 1.0
    assert blr_visit_order([], CENT, "S", matrix_distance(m), {}) == []


def test_one_candidate():
    m = sym({("S", "A"): 3.0}, "SA")
    out = blr_visit_order([{"room": "A", "probability": 1.0}], CENT, "S", matrix_distance(m), {"A": 9.0})
    assert out == [{"room": "A", "probability": 1.0, "distance_from_previous_m": 3.0,
                    "blr_score": 1.0 / (3.0 * 3.0)}]


# --- conventions (not from the paper) ---------------------------------------------------

def test_zero_distance_rooms_come_first_by_p_over_sqrt_area():
    """The robot stands in A (d = 0): Eq. (1) is undefined there, A goes first."""
    m = sym({("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "C"): 1.0}, "ABC")
    cands = [{"room": "B", "probability": 0.9}, {"room": "A", "probability": 0.05},
             {"room": "C", "probability": 0.05}]
    out = blr_visit_order(cands, CENT, "A", matrix_distance(m), {"A": 4.0, "B": 4.0, "C": 4.0})
    assert [c["room"] for c in out] == ["A", "B", "C"]
    assert out[0]["blr_score"] is None and out[0]["blr_rule"] == "zero_distance_first"
    assert out[0]["distance_from_previous_m"] == 0.0
    # two rooms at zero distance (e.g. a shared-door table): higher P / sqrt(A) first
    z = {"S": {"A": 0.0, "B": 0.0}, "A": {"B": 2.0, "S": 0.0}, "B": {"A": 2.0, "S": 0.0}}
    c2 = [{"room": "A", "probability": 0.5}, {"room": "B", "probability": 0.5}]
    assert order(c2, z, {"A": 16.0, "B": 4.0}) == ["B", "A"]


def test_equal_scores_keep_the_candidate_order():
    m = sym({("S", "A"): 2.0, ("S", "B"): 2.0, ("A", "B"): 1.0}, "SAB")
    cands = [{"room": "B", "probability": 0.5}, {"room": "A", "probability": 0.5}]
    assert order(cands, m, {"A": 4.0, "B": 4.0}) == ["B", "A"]
    assert order(list(reversed(cands)), m, {"A": 4.0, "B": 4.0}) == ["A", "B"]


def test_unreachable_from_here_is_skipped_until_reachable():
    """B cannot be reached from S directly (None / inf) but from A it can."""
    m = {"S": {"A": 2.0, "B": None}, "A": {"B": 3.0, "S": 2.0}, "B": {"A": 3.0, "S": None}}
    cands = [{"room": "B", "probability": 0.9}, {"room": "A", "probability": 0.1}]
    assert order(cands, m, {"A": 4.0, "B": 4.0}) == ["A", "B"]
    m["S"]["B"] = math.inf
    assert order(cands, m, {"A": 4.0, "B": 4.0}) == ["A", "B"]


def test_all_remaining_unreachable_is_an_error():
    m = {"S": {"A": None, "B": math.inf}}
    with pytest.raises(BLRUnreachableError, match="reachable"):
        order([{"room": "A", "probability": 0.5}, {"room": "B", "probability": 0.5}], m,
              {"A": 4.0, "B": 4.0})


# --- invalid input ------------------------------------------------------------------------

@pytest.mark.parametrize("cands, areas, message", [
    ([{"probability": 0.5}], {}, "has no 'room'"),
    ([{"room": "A", "probability": 0.5}, {"room": "A", "probability": 0.5}], {"A": 4.0}, "twice"),
    ([{"room": "A"}], {"A": 4.0}, "has no probability"),
    ([{"room": "A", "probability": 1.5}], {"A": 4.0}, "invalid probability"),
    ([{"room": "A", "probability": -0.1}], {"A": 4.0}, "invalid probability"),
    ([{"room": "A", "probability": float("nan")}], {"A": 4.0}, "invalid probability"),
    ([{"room": "A", "probability": "0.5"}], {"A": 4.0}, "invalid probability"),
    ([{"room": "A", "probability": 0.5}], {}, "no area"),
    ([{"room": "A", "probability": 0.5}], None, "no area"),
    ([{"room": "A", "probability": 0.5}], {"A": 0.0}, "not positive"),
    ([{"room": "A", "probability": 0.5}], {"A": -3.0}, "not positive"),
    ([{"room": "A", "probability": 0.5}], {"A": float("inf")}, "not a finite number"),
    ([{"room": "A", "probability": 0.5}], {"A": None}, "no area"),
])
def test_invalid_input_is_a_clear_error(cands, areas, message):
    with pytest.raises(BLRInputError, match=message):
        check_candidates(cands, areas)
    with pytest.raises(BLRInputError, match=message):
        blr_visit_order(cands, {"S": (0, 0), "A": (1, 1)}, "S", None, areas)


def test_missing_area_message_says_what_to_do():
    with pytest.raises(BLRInputError, match="--search_algorithm held_karp"):
        check_candidates([{"room": "A", "probability": 0.5}], {})


@pytest.mark.parametrize("bad", [float("nan"), -1.0, "3"])
def test_invalid_distance_is_an_error(bad):
    m = {"S": {"A": bad}}
    with pytest.raises(BLRInputError, match="distance"):
        order([{"room": "A", "probability": 0.5}], m, {"A": 4.0})


def test_missing_start_room_is_the_same_error_as_held_karp():
    with pytest.raises(ValueError, match="start room 'S' not found"):
        blr_visit_order([{"room": "A", "probability": 0.5}], {"A": (0, 0)}, "S", None, {"A": 4.0})


def test_invalid_areas_lists_every_bad_room():
    assert invalid_areas(["A", "B", "C", "D"], {"A": 4.0, "B": 0, "C": float("nan")}) == {
        "B": "area 0 m2 is not positive", "C": "area nan is not a finite number", "D": "no area"}

"""BestLocalRatio (BLR) room ordering.

Source: R. Izquierdo-Cordova, E. F. Morales, L. E. Sucar, R. Murrieta-Cid,
"Searching Objects in Known Environments: Empowering Simple Heuristic
Strategies", RoboCup 2016: Robot World Cup XX, Springer, 2017,
Section 3.1, Eq. (1) and Algorithm 1.

Eq. (1):   H(R_i) = P_i / (d_i * sqrt(A_i))
    P_i  probability of finding the object in room i
    A_i  area of room i
    d_i  distance from the current position of the robot to room i

Algorithm 1 (greedy): compute H for every room not yet in the order, add the
room with the maximum H, continue "from that room" (the planning location
becomes the chosen room), until every room is in the order. The paper does
not normalise P, d or A and does not change P while ordering; neither does
this module. Only the room ordering of the paper is implemented (not the
in-room search / viewpoint generation).

Conventions of THIS implementation, not of the paper (the paper leaves
these cases open):
  * d_i = 0 (the planning location is the room itself, e.g. the robot is
    already in a candidate room): Eq. (1) is undefined. Such rooms are taken
    before every room with d_i > 0 (the limit of H for d -> 0+ is +inf);
    among several of them, the higher P_i / sqrt(A_i) first. Their
    blr_score is reported as None with blr_rule "zero_distance_first".
  * Equal H (or equal P/sqrt(A) for zero-distance rooms): the room that
    comes first in the candidate list wins (deterministic). Callers pass
    candidates sorted by probability, so this is the more probable room.
  * A room the distance provider cannot reach from the current planning
    location (distance None or +inf) gets no score in that step. If every
    remaining room is unreachable, BLRUnreachableError is raised: distances
    are never invented.
  * No epsilon is added anywhere; invalid inputs raise BLRInputError.

Dynamic use: blr_next_room() is one step of Algorithm 1 from a given
planning location. The HTN executor re-plans after every searched room with
the robot's new room and the updated probabilities (the belief update stays
in the executor), so BLR is re-applied to the new state each time. That
repeated use is an execution strategy built around the published greedy
heuristic; the heuristic itself is unchanged.
"""

import math

from .common import euclidean


class BLRInputError(ValueError):
    """Missing or invalid input for BLR (room ids, probabilities, areas, distances)."""


class BLRUnreachableError(ValueError):
    """No remaining candidate room can be reached from the planning location."""


def blr_score(probability, distance, area):
    """Eq. (1): H = P / (d * sqrt(A)). Defined for d > 0 and A > 0 only."""
    return probability / (distance * math.sqrt(area))


# --- input checks (kept apart from the formula) ----------------------------------

def _finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def invalid_areas(rooms, areas):
    """{room: problem} for every room whose area cannot be used by BLR."""
    areas = areas or {}
    bad = {}
    for room in rooms:
        a = areas.get(room)
        if a is None:
            bad[room] = "no area"
        elif not _finite_number(a):
            bad[room] = f"area {a!r} is not a finite number"
        elif a <= 0:
            bad[room] = f"area {a} m2 is not positive"
    return bad


def check_candidates(candidates, areas):
    """Raise BLRInputError for missing/duplicate room ids, missing or
    invalid probabilities (must be finite and in [0, 1]) and missing or
    invalid areas (must be finite and > 0)."""
    seen, problems = set(), []
    for i, c in enumerate(candidates):
        room = c.get("room") if isinstance(c, dict) else None
        if room is None:
            problems.append(f"candidate #{i} has no 'room'")
            continue
        if room in seen:
            problems.append(f"room {room!r} is a candidate twice")
        seen.add(room)
        p = c.get("probability")
        if p is None:
            problems.append(f"room {room!r} has no probability")
        elif not _finite_number(p) or not 0.0 <= p <= 1.0:
            problems.append(f"room {room!r} has an invalid probability {p!r} (needs a number in [0, 1])")
    bad_areas = invalid_areas([c["room"] for c in candidates if isinstance(c, dict) and "room" in c], areas)
    problems += [f"room {r!r}: {why}" for r, why in bad_areas.items()]
    if problems:
        hint = (" BLR needs a positive floor area (area_m2 from the BIM) for every candidate room;"
                " use --search_algorithm held_karp for buildings without areas." if bad_areas else "")
        raise BLRInputError("BLR cannot order these rooms: " + "; ".join(problems) + "." + hint)


def _checked_distance(distance, a, b):
    """The provider's distance a -> b; None when unreachable (None or +inf).
    Raises for NaN, negative or non-numeric values."""
    d = distance(a, b)
    if d is None or d == math.inf:
        return None
    if not isinstance(d, (int, float)) or isinstance(d, bool) or math.isnan(d) or d < 0:
        raise BLRInputError(f"distance {a!r} -> {b!r} is invalid: {d!r}")
    return float(d)


# --- Algorithm 1 -------------------------------------------------------------------

def blr_next_room(candidates, current, distance, areas):
    """One step of Algorithm 1: the candidate with the maximum H from the
    planning location `current`. Returns (candidate, blr_score or None,
    distance) or None when there are no candidates. Inputs are assumed
    checked (check_candidates)."""
    if not candidates:
        return None
    scored, zero, unreachable = [], [], []
    for idx, c in enumerate(candidates):
        d = _checked_distance(distance, current, c["room"])
        if d is None:
            unreachable.append(c["room"])
        elif d == 0:                                   # convention, see module docstring
            zero.append((c["probability"] / math.sqrt(areas[c["room"]]), idx, c, d))
        else:
            scored.append((blr_score(c["probability"], d, areas[c["room"]]), idx, c, d))
    if zero:
        _, _, c, d = max(zero, key=lambda t: (t[0], -t[1]))
        return c, None, d
    if not scored:
        raise BLRUnreachableError(f"no remaining candidate room is reachable from {current!r}: {unreachable}")
    score, _, c, d = max(scored, key=lambda t: (t[0], -t[1]))   # equal H: earlier candidate
    return c, score, d


def blr_visit_order(candidates, centroid_lookup, start_room, distance=None, areas=None):
    """The full BLR order from start_room, in the output format of
    held_karp.optimal_visit_order: each candidate dict plus
    distance_from_previous_m (rounded to 2), and additionally blr_score (H
    when the room was chosen; None under the zero-distance convention) and,
    for zero-distance rooms, blr_rule. `distance(a, b) -> metres` defaults to
    the straight line between centroid_lookup centroids (as in Held-Karp);
    `areas` maps room -> area in m2. Candidate dicts are not changed."""
    if not candidates:
        return []
    if distance is None:
        def distance(a, b):
            return euclidean(centroid_lookup[a], centroid_lookup[b])
    if centroid_lookup.get(start_room) is None:
        raise ValueError(f"start room '{start_room}' not found or has no centroid")
    check_candidates(candidates, areas)

    remaining, ordered, current = list(candidates), [], start_room
    while remaining:
        c, score, d = blr_next_room(remaining, current, distance, areas)
        entry = dict(c, distance_from_previous_m=round(d, 2), blr_score=score)
        if score is None:
            entry["blr_rule"] = "zero_distance_first"
        ordered.append(entry)
        remaining.remove(c)
        current = c["room"]                            # "from that room the next best room is selected"
    return ordered

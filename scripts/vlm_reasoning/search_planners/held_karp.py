"""Held-Karp room ordering: the shortest open tour from the start room over
the candidate rooms, with the greedy nearest-neighbour fallback above
max_exact rooms.

Moved here unchanged from rank_candidates.py (2026-11-17); rank_candidates
re-exports optimal_visit_order and _greedy_nearest_neighbor, so existing
imports keep working. This is the only implementation of Held-Karp.
Regression test: tests/search_planners/test_held_karp_regression.py."""

import sys

from .common import euclidean


def optimal_visit_order(candidates, centroid_lookup, start_room, max_exact=18, distance=None):
    """Shortest open tour from start_room over the candidates (Held-Karp).
    `distance(room_a, room_b) -> metres` defaults to the straight line
    between centroid_lookup centroids; the HTN passes door-graph route
    lengths instead."""
    n = len(candidates)
    if n == 0:
        return []
    if distance is None:
        def distance(a, b):
            return euclidean(centroid_lookup[a], centroid_lookup[b])
    if n > max_exact:
        print(f"WARNING: {n} candidates exceeds max_exact={max_exact}, falling back to greedy.", file=sys.stderr)
        return _greedy_nearest_neighbor(candidates, centroid_lookup, start_room, distance)

    start_xy = centroid_lookup.get(start_room)
    if start_xy is None:
        raise ValueError(f"start room '{start_room}' not found or has no centroid")

    rooms = [c["room"] for c in candidates]
    dist_from_start = [distance(start_room, r) for r in rooms]
    dist = [[distance(rooms[i], rooms[j]) for j in range(n)] for i in range(n)]

    FULL = (1 << n) - 1
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = dist_from_start[i]
    for mask in range(1 << n):
        for i in range(n):
            if dp[mask][i] == INF or not (mask & (1 << i)):
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist[i][j]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i

    best_end = min(range(n), key=lambda i: dp[FULL][i])
    best_cost = dp[FULL][best_end]
    order = []
    mask, i = FULL, best_end
    while i != -1:
        order.append(i)
        prev = parent[mask][i]
        mask ^= (1 << i)
        i = prev
    order.reverse()

    ordered = []
    prev = start_room
    raw_total = 0.0
    for idx in order:
        c = candidates[idx]
        d = distance(prev, rooms[idx])
        raw_total += d
        ordered.append(dict(c, distance_from_previous_m=round(d, 2)))
        prev = rooms[idx]
    assert abs(raw_total - best_cost) < 1e-6, "path reconstruction mismatch"
    return ordered


def _greedy_nearest_neighbor(candidates, centroid_lookup, start_room, distance):
    remaining = list(candidates)
    ordered = []
    current = start_room
    while remaining:
        remaining.sort(key=lambda c: distance(current, c["room"]))
        nxt = remaining.pop(0)
        d = distance(current, nxt["room"])
        ordered.append(dict(nxt, distance_from_previous_m=round(d, 2)))
        current = nxt["room"]
    return ordered

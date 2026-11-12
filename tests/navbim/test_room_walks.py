"""Tests for option B (room centre to room centre): scripts/navbim/navbim_room_walks.py
and navbim_distance_table.build_table(method="centre"), on small hand-made rooms.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "navbim"))
from navbim_distance_table import build_table  # noqa: E402
from navbim_room_walks import (RoomGrid, room_polygon, room_start_point,  # noqa: E402
                               walkable_area)


def room(i, name, x0, y0, x1, y1):
    return {"id": i, "type": "room", "name": name, "ifc_guid": f"G{name}",
            "position": {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2, "z": 0.0},
            "polygon": {"coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


def door(i, x, y):
    return {"id": i, "type": "transition", "position": {"x": x, "y": y, "z": 0.0}}


def two_rooms():
    """A (0..4 x 0..4) | wall 0.125 | B (4.125..8.125 x 0..4), door 10 in the
    wall at y = 2. Both centres are on the line y = 2."""
    nodes = {1: room(1, "A", 0, 0, 4, 4), 2: room(2, "B", 4.125, 0, 8.125, 4),
             10: door(10, 4.0625, 2.0)}
    links = [{"source": 1, "target": 10, "type": "room"},
             {"source": 2, "target": 10, "type": "room"}]
    return nodes, links


def test_neighbours_are_not_zero_any_more():
    nodes, links = two_rooms()
    door_table = build_table(nodes, links)                    # option A
    centre_table = build_table(nodes, links, method="centre")  # option B
    assert door_table["matrix_m"][0][1] == 0.0
    d = centre_table["matrix_m"][0][1]
    assert abs(d - 4.125) < 0.1                    # straight line centre to centre
    assert centre_table["matrix_m"][1][0] == d     # symmetric
    route = centre_table["routes"]["GA->GB"]
    assert route["doors"] == [10]
    assert abs(route["start_room_walk_m"] + route["end_room_walk_m"] - d) < 1e-6


def test_walk_goes_around_a_room_inside_a_room():
    """Workspace-like hall 0..10 x 0..4 with a Lab-like room 4..6 x 1..3
    inside it: walking across the hall must go around the inner room."""
    hall, inner = room(1, "Hall", 0, 0, 10, 4), room(2, "Inner", 4, 1, 6, 3)
    hp, ip = room_polygon(hall), room_polygon(inner)
    free = walkable_area(hp, [ip], wall=0.125, clearance=0.2)
    m, path = RoomGrid(hp, 0.05, free_area=free).walks_from((1, 2), {"x": (9, 2)})["x"]
    assert m > 8.0 + 0.2            # longer than the straight line through the inner room
    assert m < 9.0
    assert all(not (4 < x < 6 and 1 < y < 3) for x, y in path)
    # the hall centre (5, 2) lies in the inner room -> moved to a walkable point
    start, how = room_start_point(hall, free)
    assert free.contains(__import__("shapely").geometry.Point(start))
    assert "nearest" in how
    # the inner room itself is not cut away by the hall that surrounds it
    assert math.isclose(walkable_area(ip, [hp], clearance=0).area, ip.area)


def test_clearance_keeps_away_from_walls():
    a = room_polygon(room(1, "A", 0, 0, 4, 4))
    free = walkable_area(a, clearance=0.2)
    assert math.isclose(free.area, 3.6 * 3.6, rel_tol=1e-3)

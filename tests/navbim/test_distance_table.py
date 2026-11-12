"""Tests for scripts/navbim/navbim_distance_table.py on a small hand-made map:

    room A --door 10-- room B --door 11-- room C
                       (B: door 10 -> door 11 planned 4.0 m)
    room D: no doors (unreachable)
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "navbim"))
from navbim_distance_table import build_table, load_distance_function, write_outputs  # noqa: E402


def make_map(planned=4.0):
    nodes = {i: {"id": i, "type": "room", "name": n, "ifc_guid": f"G{n}"}
             for i, n in [(1, "A"), (2, "B"), (3, "C"), (4, "D")]}
    nodes[10] = {"id": 10, "type": "transition"}
    nodes[11] = {"id": 11, "type": "transition"}
    links = [
        {"source": 1, "target": 10, "type": "room"},
        {"source": 2, "target": 10, "type": "room"},
        {"source": 2, "target": 11, "type": "room"},
        {"source": 3, "target": 11, "type": "room"},
        {"source": 10, "target": 11, "type": "transition", "room_id": 2,
         "planned_distance": planned},
    ]
    return nodes, links


def lookup(table):
    idx = {r["name"]: i for i, r in enumerate(table["rooms"])}
    return lambda a, b: table["matrix_m"][idx[a]][idx[b]]


def test_distances():
    d = lookup(build_table(*make_map()))
    assert d("A", "A") == 0.0
    assert d("A", "B") == 0.0          # shared door, walk inside rooms not counted
    assert d("A", "C") == 4.0          # through B on the planned path
    assert d("C", "A") == 4.0          # symmetric
    assert d("A", "D") is None         # no door -> unreachable


def test_route_is_recorded():
    table = build_table(*make_map())
    route = table["routes"]["GA->GC"]
    assert route["doors"] == [10, 11]
    assert route["walks_inside"] == ["B"]


def test_missing_preplan_is_an_error():
    with pytest.raises(ValueError, match="PrePlanEdges"):
        build_table(*make_map(planned=-1))


def test_load_distance_function(tmp_path):
    nodes, links = make_map()
    map_path = tmp_path / "toy.json"
    map_path.write_text(json.dumps({"nodes": list(nodes.values()), "links": links}))
    json_path, csv_path = write_outputs(build_table(nodes, links), str(map_path), str(tmp_path))
    dist = load_distance_function(json_path)
    assert dist("GA", "GC") == 4.0
    assert os.path.exists(csv_path)

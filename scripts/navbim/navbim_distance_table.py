"""navbim_distance_table.py - room-to-room walking distances from a navbim map.

Reads the topological map that navbim (GPP-BIM) writes after PrePlanEdges and
builds an N x N walking-distance table between all rooms, keyed by IfcSpace
GUID, for the Held-Karp visit ordering (rank_candidates.optimal_visit_order).

Distance definition, "option A: door to door" (--method door, the default):
    d(A, B) = shortest walk from any door of room A to any door of room B,
              moving door to door, where every door-to-door step inside one
              room has the length navbim planned on that room's costmap
              (link["planned_distance"], walls and robot size respected).
    The walk inside the start and the goal room is NOT included, so two rooms
    that share a door have distance 0.

Distance definition, "option B: room centre to room centre" (--method centre):
    d(A, B) = shortest walk  centre of A -> a door of A -> ... (door to door
              as in option A) ... -> a door of B -> centre of B.
    The two walks "centre <-> door" are not planned by navbim; they are
    computed from the room outlines in the same JSON (navbim_room_walks.py:
    grid search inside the room, walls and robot clearance respected).
    --check compares that method with the door-to-door walks navbim did plan.

d(A, A) = 0 in both. No ROS needed: only the JSON file is read.

Usage:
    python scripts/navbim/navbim_distance_table.py data/IDAC_Model_preplanned.json
    python scripts/navbim/navbim_distance_table.py <map.json> --method centre [--check]
    options: --out-dir data/distances  --clearance 0.2  --resolution 0.05  --wall 0.125

Writes <out-dir>/<map name>_walking_distances[_centre].json (matrix + the
route of every pair, for checking) and a .csv with room names for reading
by eye. In other code, load the table with load_distance_function().
"""

import argparse
import csv
import heapq
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOM_TYPES = {"room"}
DOOR_TYPES = {"transition"}

DEFINITIONS = {
    "door": "door_to_door (option A): shortest walk from a door of room A to a door "
            "of room B over navbim planned_distance; walk inside A and B not included",
    "centre": "centre_to_centre (option B): walk from the centre of room A to one of "
              "its doors, door to door over navbim planned_distance, then from a door "
              "of B to its centre; the centre-door walks are computed from the room "
              "outlines (navbim_room_walks.py)",
}
WALK_DEFAULTS = {"resolution": 0.05, "wall": 0.125, "clearance": 0.2}


def load_map(path):
    """The navbim JSON as (nodes by id, links). Accepts both the "links" and
    the newer NetworkX "edges" key."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    links = data.get("links", data.get("edges"))
    if links is None:
        raise ValueError(f"{path}: no 'links' or 'edges' in the file")
    return {n["id"]: n for n in data["nodes"]}, links


def rooms_and_doors(nodes, links):
    """Rooms ({id: node}), the doors of each room ({room id: set(door ids)})
    and the door-to-door steps inside rooms ({door id: [(door id, metres,
    room id)]}). A door belongs to a room when a link joins them. Raises when
    a door-to-door link has no planned distance (PrePlanEdges not run)."""
    rooms = {i: n for i, n in nodes.items() if n.get("type") in ROOM_TYPES}
    doors = {i for i, n in nodes.items() if n.get("type") in DOOR_TYPES}
    room_doors = {r: set() for r in rooms}
    steps = {d: [] for d in doors}
    missing = []
    for link in links:
        s, t = link["source"], link["target"]
        if s in rooms and t in doors:
            room_doors[s].add(t)
        elif t in rooms and s in doors:
            room_doors[t].add(s)
        elif s in doors and t in doors and link.get("type") == "transition":
            length = link.get("planned_distance", -1)
            if length is None or length < 0:
                missing.append((s, t))
                continue
            steps[s].append((t, float(length), link.get("room_id")))
            steps[t].append((s, float(length), link.get("room_id")))
    if missing:
        raise ValueError(
            f"{len(missing)} door-to-door links have no planned_distance "
            f"(e.g. {missing[:3]}). Run navbim PrePlanEdges first.")
    return rooms, room_doors, steps


def shortest_from(start_doors, steps):
    """Dijkstra over the doors from all start_doors at once. start_doors is
    a set of doors (all start at 0 m) or a dict {door: metres already
    walked}. Returns ({door: metres}, {door: previous (door, room id)})."""
    if not isinstance(start_doors, dict):
        start_doors = {d: 0.0 for d in start_doors}
    dist = dict(start_doors)
    prev = {}
    heap = [(m, d) for d, m in start_doors.items()]
    heapq.heapify(heap)
    while heap:
        d0, door = heapq.heappop(heap)
        if d0 > dist[door]:
            continue
        for nxt, length, room in steps.get(door, []):
            nd = d0 + length
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = (door, room)
                heapq.heappush(heap, (nd, nxt))
    return dist, prev


def _room_grids(nodes, resolution, wall, clearance):
    """Outline grid of every room: {room id: (RoomGrid, walkable area)}."""
    from navbim_room_walks import RoomGrid, room_polygon, walkable_area
    rooms = {i: n for i, n in nodes.items() if n.get("type") in ROOM_TYPES}
    polys = {r: room_polygon(rooms[r]) for r in rooms}
    floors = {r: set(rooms[r].get("floor") or []) for r in rooms}

    def same_floor(a, b):   # rooms with no floor info are compared with all
        return not floors[a] or not floors[b] or bool(floors[a] & floors[b])

    grids = {}
    for r in rooms:
        others = [q for o, q in polys.items() if o != r and same_floor(o, r)]
        free = walkable_area(polys[r], others, wall=wall, clearance=clearance)
        grids[r] = (RoomGrid(polys[r], resolution, free_area=free), free)
    return grids


def centre_walks(nodes, room_doors, resolution=0.05, wall=0.125, clearance=0.2):
    """Walk inside every room from its centre to each of its doors:
    ({room: {door: metres}}, {room: info about the centre point used})."""
    from navbim_room_walks import point_xy, room_start_point
    walks, info = {}, {}
    for r, (grid, free) in _room_grids(nodes, resolution, wall, clearance).items():
        start, how = room_start_point(nodes[r], free)
        res = grid.walks_from(start, {d: point_xy(nodes[d]) for d in room_doors[r]})
        walks[r] = {d: m for d, (m, _) in res.items() if m is not None}
        info[r] = {"centre_xy": [round(start[0], 3), round(start[1], 3)],
                   "centre_from": how,
                   "centre_to_door_m": {str(d): round(m, 3) for d, m in walks[r].items()}}
    return walks, info


def check_against_navbim(nodes, links, resolution=0.05, wall=0.125, clearance=0.2):
    """Recompute every door-to-door walk navbim planned with the outline
    method of option B: [(room name, door a, door b, navbim m, outline m)]."""
    from navbim_room_walks import point_xy
    grids = _room_grids(nodes, resolution, wall, clearance)
    out = []
    for link in links:
        if link.get("type") != "transition" or link.get("planned_distance", -1) < 0:
            continue
        r = int(link["room_id"])
        a, b = link["source"], link["target"]
        m, _ = grids[r][0].walks_from(point_xy(nodes[a]), {b: point_xy(nodes[b])})[b]
        out.append((nodes[r].get("name", str(r)), a, b, float(link["planned_distance"]), m))
    return out


def build_table(nodes, links, method="door", **walk_options):
    """The full table as a dict: rooms (guid, name, node id, doors), the
    distance matrix in that room order (None = unreachable) and the route of
    every pair as a list of door ids plus the rooms walked through.
    method "door" = option A, "centre" = option B (walk_options: resolution,
    wall, clearance; see WALK_DEFAULTS)."""
    if method not in DEFINITIONS:
        raise ValueError(f"unknown method {method!r} (door or centre)")
    rooms, room_doors, steps = rooms_and_doors(nodes, links)
    order = sorted(rooms, key=lambda r: rooms[r].get("name", ""))
    guid = {r: rooms[r].get("ifc_guid") or str(r) for r in order}
    name = {r: rooms[r].get("name", str(r)) for r in order}
    opts = {**WALK_DEFAULTS, **walk_options}
    if method == "centre":
        inside, centre_info = centre_walks(nodes, room_doors, **opts)
    else:
        inside = {r: {d: 0.0 for d in room_doors[r]} for r in rooms}

    matrix, routes = [], {}
    for a in order:
        dist, prev = shortest_from(inside[a], steps)
        row = []
        for b in order:
            if a == b:
                row.append(0.0)
                continue
            ends = [d for d in inside[b] if d in dist]
            if not ends:
                row.append(None)
                continue
            end = min(ends, key=lambda d: dist[d] + inside[b][d])
            total = dist[end] + inside[b][end]
            path, through = [end], []
            while path[-1] in prev:
                door, room = prev[path[-1]]
                path.append(door)
                through.append(name.get(room, str(room)))
            path.reverse()
            through.reverse()
            row.append(round(total, 3))
            route = {"from": name[a], "to": name[b], "metres": round(total, 3),
                     "doors": path, "door_guids": [nodes[d].get("ifc_guid") for d in path],
                     "walks_inside": through}
            if method == "centre":
                route["start_room_walk_m"] = round(inside[a][path[0]], 3)
                route["end_room_walk_m"] = round(inside[b][end], 3)
            routes[f"{guid[a]}->{guid[b]}"] = route
        matrix.append(row)

    table = {
        "method": method,
        "definition": DEFINITIONS[method],
        "rooms": [{"guid": guid[r], "name": name[r], "node_id": r,
                   "doors": sorted(room_doors[r])} for r in order],
        "matrix_m": matrix,
        "routes": routes,
    }
    if method == "centre":
        table["centre_walk_options"] = opts
        for room in table["rooms"]:
            room.update(centre_info[room["node_id"]])
    return table


def write_outputs(table, map_path, out_dir):
    base = os.path.splitext(os.path.basename(map_path))[0]
    if table.get("method") == "centre":
        base += "_centre"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{base}_walking_distances.json")
    csv_path = os.path.join(out_dir, f"{base}_walking_distances.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"source_map": os.path.basename(map_path), **table}, f, indent=2)
    names = [r["name"] for r in table["rooms"]]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["from \\ to"] + names)
        for n, row in zip(names, table["matrix_m"]):
            w.writerow([n] + ["" if v is None else f"{v:.2f}" for v in row])
    return json_path, csv_path


def load_distance_function(table_path):
    """distance(guid_a, guid_b) -> metres, for optimal_visit_order(...,
    distance=...). Returns None for an unreachable pair; raises KeyError for
    a GUID that is not in the table."""
    with open(table_path, encoding="utf-8") as f:
        table = json.load(f)
    index = {r["guid"]: i for i, r in enumerate(table["rooms"])}
    matrix = table["matrix_m"]

    def distance(a, b):
        return matrix[index[a]][index[b]]
    return distance


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("map", help="navbim topological map JSON (after PrePlanEdges)")
    p.add_argument("--out-dir", default=os.path.join("data", "distances"))
    p.add_argument("--method", choices=sorted(DEFINITIONS), default="door",
                   help="door = option A (door to door), centre = option B")
    p.add_argument("--clearance", type=float, default=WALK_DEFAULTS["clearance"],
                   help="option B: robot radius in m (navbim robot_radius is 0.2)")
    p.add_argument("--resolution", type=float, default=WALK_DEFAULTS["resolution"],
                   help="option B: grid cell size in m")
    p.add_argument("--wall", type=float, default=WALK_DEFAULTS["wall"],
                   help="option B: wall thickness kept around other rooms, in m")
    p.add_argument("--check", action="store_true",
                   help="option B: compare the outline method with navbim's own "
                        "door-to-door walks (slow, about a minute)")
    args = p.parse_args(argv)

    nodes, links = load_map(args.map)
    opts = {"resolution": args.resolution, "wall": args.wall, "clearance": args.clearance}
    table = build_table(nodes, links, method=args.method, **opts)
    if args.method == "door":
        table.pop("centre_walk_options", None)
    if args.check and args.method == "centre":
        rows = check_against_navbim(nodes, links, **opts)
        diffs = [m - nav for *_, nav, m in rows if m is not None]
        c = {"what": "door-to-door walks navbim planned, recomputed with the outline method",
             "pairs": len(rows), "unreachable": sum(m is None for *_, m in rows),
             "mean_diff_m": round(sum(diffs) / len(diffs), 3),
             "mean_abs_diff_m": round(sum(map(abs, diffs)) / len(diffs), 3),
             "min_diff_m": round(min(diffs), 3), "max_diff_m": round(max(diffs), 3),
             "rows": [{"room": r, "doors": [a, b], "navbim_m": round(nav, 3),
                       "outline_m": None if m is None else round(m, 3)}
                      for r, a, b, nav, m in rows]}
        table["check_against_navbim"] = c
        print(f"check: {c['pairs']} navbim walks ({c['unreachable']} unreachable), "
              f"outline - navbim: mean {c['mean_diff_m']:+.3f} m, mean |diff| "
              f"{c['mean_abs_diff_m']:.3f} m, range {c['min_diff_m']:+.2f}..{c['max_diff_m']:+.2f}")
    json_path, csv_path = write_outputs(table, args.map, args.out_dir)

    rooms, m = table["rooms"], table["matrix_m"]
    unreachable = sum(v is None for row in m for v in row)
    zeros = sum(1 for i, row in enumerate(m) for j, v in enumerate(row) if i != j and v == 0)
    print(f"{len(rooms)} rooms, {len(table['routes'])} ordered pairs, method {table['method']}")
    print(f"unreachable pairs: {unreachable}, pairs with distance 0: {zeros}")
    print(f"wrote {json_path}\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

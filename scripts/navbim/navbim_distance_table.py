"""navbim_distance_table.py - room-to-room walking distances from a navbim map.

Reads the topological map that navbim (GPP-BIM) writes after PrePlanEdges and
builds an N x N walking-distance table between all rooms, keyed by IfcSpace
GUID, for the Held-Karp visit ordering (rank_candidates.optimal_visit_order).

Distance definition, "option A: door to door" (see navbim_setup_log.md, F):
    d(A, B) = shortest walk from any door of room A to any door of room B,
              moving door to door, where every door-to-door step inside one
              room has the length navbim planned on that room's costmap
              (link["planned_distance"], walls and robot size respected).
    d(A, A) = 0.
    The walk inside the start and the goal room is NOT included, so two rooms
    that share a door have distance 0. Option B (room centre to room centre)
    will add that part.

No ROS needed: only the JSON file is read.

Usage:
    python scripts/navbim/navbim_distance_table.py data/IDAC_Model_preplanned.json
    python scripts/navbim/navbim_distance_table.py <map.json> --out-dir data/distances

Writes <out-dir>/<map name>_walking_distances.json (matrix + the route of
every pair, for checking) and a .csv with room names for reading by eye.
In other code, load the table with load_distance_function().
"""

import argparse
import csv
import heapq
import json
import os
import sys

ROOM_TYPES = {"room"}
DOOR_TYPES = {"transition"}


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
    """Dijkstra over the doors from all start_doors at once (distance 0).
    Returns ({door: metres}, {door: previous (door, room id)})."""
    dist = {d: 0.0 for d in start_doors}
    prev = {}
    heap = [(0.0, d) for d in start_doors]
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


def build_table(nodes, links):
    """The full table as a dict: rooms (guid, name, node id, doors), the
    distance matrix in that room order (None = unreachable) and the route of
    every pair as a list of door ids plus the rooms walked through."""
    rooms, room_doors, steps = rooms_and_doors(nodes, links)
    order = sorted(rooms, key=lambda r: rooms[r].get("name", ""))
    guid = {r: rooms[r].get("ifc_guid") or str(r) for r in order}
    name = {r: rooms[r].get("name", str(r)) for r in order}

    matrix, routes = [], {}
    for a in order:
        dist, prev = shortest_from(room_doors[a], steps)
        row = []
        for b in order:
            if a == b:
                row.append(0.0)
                continue
            ends = [d for d in room_doors[b] if d in dist]
            if not ends:
                row.append(None)
                continue
            end = min(ends, key=lambda d: dist[d])
            path, through = [end], []
            while path[-1] in prev:
                door, room = prev[path[-1]]
                path.append(door)
                through.append(name.get(room, str(room)))
            path.reverse()
            through.reverse()
            row.append(round(dist[end], 3))
            routes[f"{guid[a]}->{guid[b]}"] = {
                "from": name[a], "to": name[b], "metres": round(dist[end], 3),
                "doors": path, "walks_inside": through}
        matrix.append(row)

    return {
        "definition": "door_to_door (option A): shortest walk from a door of room A "
                      "to a door of room B over navbim planned_distance; walk inside "
                      "A and B not included",
        "rooms": [{"guid": guid[r], "name": name[r], "node_id": r,
                   "doors": sorted(room_doors[r])} for r in order],
        "matrix_m": matrix,
        "routes": routes,
    }


def write_outputs(table, map_path, out_dir):
    base = os.path.splitext(os.path.basename(map_path))[0]
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
    args = p.parse_args(argv)

    nodes, links = load_map(args.map)
    table = build_table(nodes, links)
    json_path, csv_path = write_outputs(table, args.map, args.out_dir)

    rooms, m = table["rooms"], table["matrix_m"]
    unreachable = sum(v is None for row in m for v in row)
    zeros = sum(1 for i, row in enumerate(m) for j, v in enumerate(row) if i != j and v == 0)
    print(f"{len(rooms)} rooms, {len(table['routes'])} ordered pairs")
    print(f"unreachable pairs: {unreachable}, pairs with distance 0 (rooms share a door): {zeros}")
    print(f"wrote {json_path}\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

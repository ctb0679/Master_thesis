"""navbim_room_walks.py - walking distances INSIDE one room, from its floor outline.

Used by navbim_distance_table.py for "option B: room centre to room centre".
navbim only plans door-to-door inside a room, so the piece "room centre ->
door" is computed here, offline, from the room polygon in the navbim JSON:

  1. Lay a grid (default 5 cm) over the room. A cell is walkable when its
     centre lies inside the room outline, outside every other room (plus
     one wall thickness around it: a room can sit inside another one, like
     the Lab inside the Workspace in IDAC_Model), and at least `clearance`
     (the robot radius, navbim uses 0.2 m) away from those borders.
     IfcSpace outlines stop at the walls, so walls are never walkable;
     furniture is not in the outline and is ignored.
  2. Dijkstra on the grid (8 neighbours) from the start point.
  3. Straighten the grid path with line-of-sight shortcuts (the same idea
     as navbim's ThetaStar), and measure its real length in metres.

Start and end points (room centre, door) that fall outside the walkable
cells, e.g. a door that sits in the wall, are joined to the nearest
walkable cell by a straight step, which is added to the length.
"""

import heapq
import math

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import Point, Polygon

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def room_polygon(node):
    """Shapely polygon from a navbim room node ({"polygon": {"coordinates":
    [[[x, y(, z)], ...]]}}), repaired when the outline crosses itself."""
    ring = [(p[0], p[1]) for p in node["polygon"]["coordinates"][0]]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
        if poly.geom_type != "Polygon":   # keep the biggest piece
            poly = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
    return poly


def walkable_area(polygon, other_rooms=(), wall=0.125, clearance=0.2):
    """Room outline minus the other rooms (grown by one wall thickness),
    shrunk by the clearance."""
    area = polygon
    # a room that surrounds this one (Workspace around the Lab) is not cut
    blocked = [q.buffer(wall) for q in other_rooms
               if q.intersects(polygon.buffer(wall)) and not polygon.within(q.buffer(0.01))]
    if blocked:
        area = area.difference(shapely.union_all(blocked))
    if clearance > 0:
        area = area.buffer(-clearance)
    return area


class RoomGrid:
    """Walkable grid of one room and shortest walks inside it.
    `polygon` is the room outline, `free_area` the part that may be walked
    (default: the whole outline, see walkable_area())."""

    def __init__(self, polygon, resolution=0.05, free_area=None):
        self.res = resolution
        free_area = polygon if free_area is None else free_area
        if free_area.is_empty:
            raise ValueError("room is narrower than twice the clearance")
        minx, miny, maxx, maxy = polygon.bounds
        self.x0, self.y0 = minx, miny
        self.nx = int(math.ceil((maxx - minx) / resolution)) + 1
        self.ny = int(math.ceil((maxy - miny) / resolution)) + 1
        xs = self.x0 + (np.arange(self.nx) + 0.5) * resolution
        ys = self.y0 + (np.arange(self.ny) + 0.5) * resolution
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.free = shapely.contains_xy(free_area, gx, gy)   # [ix, iy]
        self._free_idx = np.argwhere(self.free)
        if len(self._free_idx) == 0:
            raise ValueError("room has no walkable cells at this resolution")

    # --- cells <-> metres -------------------------------------------------
    def centre_of(self, cell):
        return (self.x0 + (cell[0] + 0.5) * self.res,
                self.y0 + (cell[1] + 0.5) * self.res)

    def snap(self, xy):
        """(nearest walkable cell, straight distance in metres to it)."""
        cx = (np.asarray(xy[0]) - self.x0) / self.res - 0.5
        cy = (np.asarray(xy[1]) - self.y0) / self.res - 0.5
        d2 = (self._free_idx[:, 0] - cx) ** 2 + (self._free_idx[:, 1] - cy) ** 2
        k = int(np.argmin(d2))
        cell = (int(self._free_idx[k, 0]), int(self._free_idx[k, 1]))
        return cell, math.sqrt(float(d2[k])) * self.res

    # --- search -----------------------------------------------------------
    def dijkstra(self, start_cell, goals=()):
        """Grid distances (in cells) and predecessors from start_cell. Stops
        early once every cell in `goals` is settled."""
        free, nx, ny = self.free, self.nx, self.ny
        dist = {start_cell: 0.0}
        prev = {}
        heap = [(0.0, start_cell)]
        left = set(goals)
        while heap:
            d0, (i, j) = heapq.heappop(heap)
            if d0 > dist[(i, j)]:
                continue
            left.discard((i, j))
            if goals and not left:
                break
            for di, dj, step in NEIGHBOURS:
                a, b = i + di, j + dj
                if not (0 <= a < nx and 0 <= b < ny) or not free[a, b]:
                    continue
                if di and dj and not (free[i + di, j] and free[i, j + dj]):
                    continue   # no squeezing diagonally past a corner
                nd = d0 + step
                if nd < dist.get((a, b), math.inf):
                    dist[(a, b)] = nd
                    prev[(a, b)] = (i, j)
                    heapq.heappush(heap, (nd, (a, b)))
        return dist, prev

    def line_of_sight(self, c1, c2):
        """True when the straight line between two cell centres only
        crosses walkable cells."""
        (i1, j1), (i2, j2) = c1, c2
        n = int(max(abs(i2 - i1), abs(j2 - j1)) * 2) + 1
        for t in np.linspace(0.0, 1.0, n + 1):
            i = int(round(i1 + (i2 - i1) * t))
            j = int(round(j1 + (j2 - j1) * t))
            if not self.free[i, j]:
                return False
        return True

    def straighten(self, cells):
        """Greedy line-of-sight shortcuts: from each kept cell jump to the
        farthest later cell that is directly visible."""
        out, k = [cells[0]], 0
        while k < len(cells) - 1:
            j = len(cells) - 1
            while j > k + 1 and not self.line_of_sight(cells[k], cells[j]):
                j -= 1
            out.append(cells[j])
            k = j
        return out

    def walks_from(self, start_xy, targets_xy):
        """Walking distance (m) and path (list of (x, y)) from start_xy to
        each target, inside the room. {key: (metres, path)}; a target the
        start cannot reach gets (None, None)."""
        s_cell, _ = self.snap(start_xy)
        t_cells = {key: self.snap(xy)[0] for key, xy in targets_xy.items()}
        dist, prev = self.dijkstra(s_cell, goals=t_cells.values())
        result = {}
        for key, xy in targets_xy.items():
            t_cell = t_cells[key]
            if t_cell not in dist:
                result[key] = (None, None)
                continue
            cells = [t_cell]
            while cells[-1] in prev:
                cells.append(prev[cells[-1]])
            cells.reverse()
            pts = [self.centre_of(c) for c in self.straighten(cells)]
            pts = [tuple(start_xy)] + pts + [tuple(xy)]
            length = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
            result[key] = (length, pts)
        return result


def point_xy(node):
    p = node["position"]
    return (float(p["x"]), float(p["y"]))


def room_start_point(node, free_area):
    """The room centre from navbim ("position"); when it is not walkable
    (L-shaped room, or another room sits there) use the nearest walkable
    point instead."""
    xy = point_xy(node)
    if free_area.contains(Point(xy)):
        return xy, "navbim position"
    near = shapely.ops.nearest_points(free_area, Point(xy))[0]
    # step a little further inside so the point is not on the border
    inside = free_area.buffer(-0.05)
    if not inside.is_empty:
        near = shapely.ops.nearest_points(inside, Point(xy))[0]
    return (near.x, near.y), "nearest walkable point to the navbim position"

"""Helpers shared by the room-ordering algorithms (search_planners)."""

import math


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

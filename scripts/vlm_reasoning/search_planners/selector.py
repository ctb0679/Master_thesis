"""Choose the room-ordering algorithm: --search_algorithm {held_karp, blr}.

Both receive the same candidate list (made beforehand by the existing
select_candidates) and the same distance provider; the selector only
forwards them. It does not change probabilities, filter rooms, navigate or
detect. Held-Karp ignores `areas`, so it works without room areas."""

from .blr import blr_visit_order
from .held_karp import optimal_visit_order

ALGORITHMS = ("held_karp", "blr")
DEFAULT_ALGORITHM = "held_karp"


def check_algorithm(name):
    if name not in ALGORITHMS:
        raise ValueError(f"unknown search algorithm {name!r}; choose one of {', '.join(ALGORITHMS)}")
    return name


def order_rooms(candidates, centroid_lookup, start_room, algorithm=DEFAULT_ALGORITHM, distance=None, areas=None):
    """Visit order of `candidates` from `start_room` with the chosen algorithm,
    as a list of candidate dicts with distance_from_previous_m (BLR adds
    blr_score). `distance(a, b) -> metres` defaults to straight lines between
    centroids for both algorithms; `areas` (room -> m2) is used by BLR only."""
    check_algorithm(algorithm)
    if algorithm == "blr":
        return blr_visit_order(candidates, centroid_lookup, start_room, distance=distance, areas=areas)
    return optimal_visit_order(candidates, centroid_lookup, start_room, distance=distance)

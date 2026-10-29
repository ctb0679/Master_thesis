"""Object-search domain: the methods for find_object, search_space and go_to,
plus the room selection they share with the executor.

The three find_object methods are the three strategies of the old
SearchSession, now as separate HTN methods tried in priority order:
user-restricted room -> rooms where the IFC records the target -> semantic
ranking. When every recorded room has been searched, m_search_recorded is
rejected and the planner falls through to m_search_semantic by ordinary
backtracking."""

from htn.core import CompoundTask, Condition, Method, PrimitiveTask
from rank_candidates import optimal_visit_order, select_candidates

from .building import DoorGraph
from .operators import HasCapability, RobotAt, SpaceReachable


# --- room selection (shared with the executor's stop rule) --------------------

def reachability(state):
    """Every room, split by whether this robot has a passable route to it
    from its current space. Returns (reachable guids, {unreachable guid:
    {"code", "message"}}); codes: unreachable_space, door_too_narrow,
    missing_capability, exterior_not_allowed, other_storey."""
    graph = DoorGraph.from_state(state)
    reachable, unreachable = [], {}
    for guid in state.spaces:
        route = graph.route(state.robot["space"], guid, state.robot)
        if isinstance(route, list):
            reachable.append(guid)
        else:
            unreachable[guid] = {"code": route.reason_code, "message": route.message}
    return reachable, unreachable


def eligible_spaces(state):
    """Rooms that may still be searched: reachable (see reachability) and
    fewer than max_visits_per_room visits. Returns (eligible guids,
    {unreachable guid: {"code", "message"}}). Unreachable rooms keep their
    probability; the executor reports it."""
    visits, max_visits = state.search["visits"], state.settings["max_visits_per_room"]
    reachable, unreachable = reachability(state)
    return [g for g in reachable if visits.get(g, 0) < max_visits], unreachable


def visit_order(state, candidates):
    """Shortest tour over the candidates ({"room": guid, ...} dicts) from the
    robot's space: Held-Karp (rank_candidates.optimal_visit_order) with the
    door-graph distance, or straight lines with settings distance = euclidean."""
    graph = DoorGraph.from_state(state)
    centroids = {g: s["centroid"] for g, s in state.spaces.items()}
    distance = None
    if state.settings.get("distance", "door_graph") == "door_graph":
        def distance(a, b):
            return graph.route_length(a, b, state.robot)
    return optimal_visit_order(candidates, centroids, state.robot["space"], distance=distance)


def select_rooms(state):
    """Rooms to search next, in visit order. The posterior of the eligible
    rooms is renormalised to sum to 1 (the distribution given that the target
    was not in the rooms already searched); select_candidates then takes the
    most likely rooms up to prob_mass, skipping any below min_prob."""
    eligible, _ = eligible_spaces(state)
    posterior = state.belief.get("posterior") or {}
    mass = sum(posterior.get(g, 0.0) for g in eligible)
    if mass <= 0:
        return []
    ranked = sorted(({"room": g, "probability": posterior.get(g, 0.0) / mass} for g in eligible),
                    key=lambda r: (-r["probability"], r["room"]))
    chosen = select_candidates(ranked, state.settings["prob_mass"], state.settings["min_prob"])
    return [c["room"] for c in visit_order(state, chosen)] if chosen else []


def _recorded_eligible(state):
    eligible, _ = eligible_spaces(state)
    return [r for r in state.task.get("recorded", []) if r in eligible]


# --- method preconditions -----------------------------------------------------

class RestrictedRoomGiven(Condition):
    def check(self, state):
        return bool(state.task.get("restricted"))

    def explain(self, state):
        return "the user did not restrict the search to one room"


class NoRestrictedRoom(Condition):
    def check(self, state):
        return not state.task.get("restricted")

    def explain(self, state):
        return "the user restricted the search to one room"


class RecordedRoomsGiven(Condition):
    def check(self, state):
        return bool(state.task.get("recorded"))

    def explain(self, state):
        return "the building model records the target in no room"


class SomeRecordedRoomEligible(Condition):
    def check(self, state):
        return bool(_recorded_eligible(state))

    def explain(self, state):
        return "every room where the building model records the target has been searched"


class PriorAvailable(Condition):
    reason_code = "prior_needed"

    def check(self, state):
        return bool(state.belief.get("posterior"))

    def explain(self, state):
        return "no room probabilities yet (the executor asks the ranking model once)"


class SomeRoomSelected(Condition):
    reason_code = "no_eligible_rooms"

    def check(self, state):
        return bool(select_rooms(state))

    def explain(self, state):
        return "no reachable room is left to search (all visited or below min_prob)"


# --- method bodies ------------------------------------------------------------

def _search(rooms, target):
    return [CompoundTask("search_space", (room, target)) for room in rooms]


def search_restricted(state, target, mode):
    room = state.task["restricted"]
    if state.search["visits"].get(room, 0) >= state.settings["max_visits_per_room"]:
        return []                         # the one room was searched: nothing left to do
    return _search([room], target)


def search_recorded(state, target, mode):
    rooms = [c["room"] for c in visit_order(state, [{"room": r} for r in _recorded_eligible(state)])]
    return _search(rooms, target)


def search_semantic(state, target, mode):
    return _search(select_rooms(state), target)


def search_space(state, space, target):
    return [CompoundTask("go_to", (space,)),
            PrimitiveTask("capture_rgb", (space,)),
            PrimitiveTask("detect", (space, target)),
            PrimitiveTask("update_belief", (space,)),
            PrimitiveTask("record_result", (space, target))]


def already_there(state, space):
    return []


def route_to(state, space):
    """navigate through each door on the route, opening closed doors first."""
    route = DoorGraph.from_state(state).route(state.robot["space"], space, state.robot)
    tasks, here = [], state.robot["space"]
    for nxt, door_guid in route:
        door = state.doors[door_guid]
        pos = tuple(door["position"])
        near, far = door["approach"].get(here), door["approach"].get(nxt)
        near, far = (tuple(near) if near else None), (tuple(far) if far else None)
        if door["state"] != "open":
            tasks.append(PrimitiveTask("manipulate_door", (door_guid, "open", pos, near)))
        centroid = tuple(state.spaces[nxt]["centroid"]) if nxt in state.spaces else None
        tasks.append(PrimitiveTask("navigate", (here, nxt, door_guid, pos, near, far, centroid)))
        here = nxt
    return tasks


def methods():
    """All methods; per task, list order = priority order."""
    return [
        Method("m_search_restricted", "find_object", ("target", "mode"),
               [RestrictedRoomGiven()], search_restricted,
               "search only the room the user named"),
        Method("m_search_recorded", "find_object", ("target", "mode"),
               [NoRestrictedRoom(), RecordedRoomsGiven(), SomeRecordedRoomEligible()], search_recorded,
               "search the rooms where the IFC model records the target, in route order"),
        Method("m_search_semantic", "find_object", ("target", "mode"),
               [NoRestrictedRoom(), PriorAvailable(), SomeRoomSelected()], search_semantic,
               "search the most likely rooms (renormalised posterior, prob_mass), in route order"),
        Method("m_search_space", "search_space", ("space", "target"),
               [HasCapability("rgb"), SpaceReachable("?space")], search_space,
               "go to the space, take an image, detect, update belief, record"),
        Method("m_already_there", "go_to", ("space",),
               [RobotAt("?space")], already_there,
               "the robot is already in the space"),
        Method("m_route", "go_to", ("space",),
               [SpaceReachable("?space")], route_to,
               "shortest passable route through door positions; manipulate_door before closed doors"),
    ]

# search_planners: room ordering

Puts the rooms chosen for a search into a visit order. Which rooms are
chosen is decided before, by `rank_candidates.select_candidates`
(`--prob_mass`, `--min_prob`); both algorithms get that same list and the
same distance function, and only the order differs.

| File | What |
|---|---|
| `held_karp.py` | `optimal_visit_order`: shortest open tour from the start room (Held-Karp, exact up to `max_exact=18` rooms, greedy nearest neighbour above). Moved unchanged from `rank_candidates.py`, which re-exports it. **Default.** |
| `blr.py` | `blr_visit_order`, `blr_next_room`: BestLocalRatio. |
| `selector.py` | `order_rooms(candidates, centroid_lookup, start_room, algorithm, distance, areas)`; `ALGORITHMS = ("held_karp", "blr")`. |
| `common.py` | `euclidean` (default distance of both). |

Choose with `--search_algorithm {held_karp, blr}` in `interactive_search.py`
(HTN and `--no-htn`).

## Held-Karp

Minimises d(s, r1) + d(r1, r2) + ... + d(r(n-1), rn) over all orders of the
n rooms (open path from the start s). C({j}, j) = d(s, j);
C(S, j) = min over i in S without j of C(S without j, i) + d(i, j); answer
min over j of C(all, j). Regression test against the pre-refactor outputs:
`tests/search_planners/test_held_karp_regression.py`.

## BLR (BestLocalRatio)

R. Izquierdo-Cordova, E. F. Morales, L. E. Sucar, R. Murrieta-Cid,
"Searching Objects in Known Environments: Empowering Simple Heuristic
Strategies", RoboCup 2016, Springer 2017, Sec. 3.1, Eq. (1), Algorithm 1.

    H(R_i) = P_i / (d_i * sqrt(A_i))

P_i probability of the room, A_i its area (`area_m2` from the BIM), d_i the
distance from the current planning location. Take the room with the highest
H, continue from that room, until all rooms are ordered. No weights, no
normalisation, probabilities not changed while ordering.

Conventions of this implementation (the paper leaves these open):
- d_i = 0 (the robot is in that room): taken first, before every d_i > 0;
  among several, higher P_i / sqrt(A_i) first; `blr_score` None,
  `blr_rule` "zero_distance_first".
- Equal H: the earlier candidate (candidates come sorted by probability).
- Unreachable from the current location (distance None or inf): not scored
  in that step; all remaining unreachable -> `BLRUnreachableError`.
- Missing room id, duplicate room, probability not in [0, 1], area missing
  / not finite / <= 0, distance NaN or negative -> `BLRInputError`. No
  epsilon, nothing invented.

Output: the candidate dicts plus `distance_from_previous_m` (as Held-Karp)
and `blr_score`.

## Where it is used

Every place that orders rooms follows `--search_algorithm`:

| Path | Ranked rooms | Rooms where the IFC records the target |
|---|---|---|
| HTN (default) | `methods.select_rooms` | `methods.search_recorded` |
| `--no-htn` | `SearchSession.rank_and_plan` | `SearchSession.plan_known_rooms` (also a named room and go_to: one room, nothing to order) |

Recorded rooms are searched before the ranking model is asked, so they have
no model probabilities: they count as equally likely (HTN 1/n each,
`--no-htn` 1.0 each as it always showed; the BLR order is the same for any
equal value). BLR then orders them by 1 / (d * sqrt(A)); Held-Karp is
unchanged.

Distances: the HTN passes `distance_function(state)` (`--distance`, navbim
by default) to both algorithms; `--no-htn` uses straight lines between room
centroids by default and navbim walking distances with `--distance navbim`
(same table; a pair without a walk falls back to the straight line).

Replanning: the HTN executor replans after every searched room with the
robot's new room and the updated probabilities (belief update in the
executor), so BLR is applied again to the new state. That repeated use is an
execution strategy around the published greedy heuristic, not part of the
paper. `--no-htn` plans once, as before.

Note for comparisons: the paper evaluates orders by the expected distance
E(d) = sum_i (D_i + sqrt(A_i)) P_i (its Eq. 2); Held-Karp here minimises the
total walk, not E(d).

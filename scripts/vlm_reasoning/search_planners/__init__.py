"""Room-ordering algorithms (see README.md in this folder).

held_karp  shortest open tour (default; moved unchanged from rank_candidates)
blr        BestLocalRatio, Izquierdo-Cordova et al., RoboCup 2016, Eq. (1)
selector   order_rooms(..., algorithm=...) used by both execution paths
"""

from .selector import ALGORITHMS, DEFAULT_ALGORITHM, check_algorithm, order_rooms  # noqa: F401

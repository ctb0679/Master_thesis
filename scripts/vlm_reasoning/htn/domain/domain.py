"""build_domain(): the object-search domain as operator and method libraries."""

from htn.core import MethodLibrary, OperatorLibrary

from .methods import methods
from .operators import operators


def build_domain():
    """-> (OperatorLibrary, MethodLibrary) with the six operators and the
    methods for find_object, search_space and go_to, in priority order."""
    ops, ms = OperatorLibrary(), MethodLibrary()
    for op in operators():
        ops.register(op)
    for m in methods():
        ms.register(m)
    return ops, ms

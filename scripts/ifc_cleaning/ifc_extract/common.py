"""Constants and small helpers shared by every module of the extractor."""

import json
import math

import numpy as np
import ifcopenshell

OUTSIDE = "outside"          # adjacency node for external doors
COORD_DECIMALS = 3           # mm resolution in the output


def is_any(entity, classes):
    return any(entity.is_a(c) for c in classes)


def by_type(model, cls):
    """model.by_type that returns [] for classes absent from the schema
    (e.g. IfcMapConversion or IfcSpatialZone in IFC2X3)."""
    try:
        return model.by_type(cls)
    except RuntimeError:
        return []


def guid_of(entity):
    return getattr(entity, "GlobalId", None) if entity is not None else None


def rnd(value, nd=COORD_DECIMALS):
    return None if value is None else round(float(value), nd)


def sanitise(obj):
    """JSON-safe: NaN/inf -> null, tuples -> lists, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitise(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, ifcopenshell.entity_instance):
        return str(obj)
    return obj


def write_json(path, data):
    path.write_text(json.dumps(sanitise(data), indent=2, default=str, allow_nan=False))
    return path.stat().st_size

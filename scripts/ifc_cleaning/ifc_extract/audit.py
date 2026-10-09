"""Audit: everything skipped, dropped, guessed geometrically or failed is
counted here, written to the JSON "audit" key and printed. It is the
semantic-quality measure of a model, and the guarantee that nothing is
dropped silently."""

from collections import Counter

from .common import guid_of


class _Guard:
    """Context manager that counts and suppresses exceptions for one step,
    so one malformed entity never aborts the whole extraction, but no failure
    goes unrecorded either."""

    def __init__(self, audit, stage, subject):
        self.audit, self.stage, self.subject = audit, stage, subject

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, _tb):
        if exc_type is None or not issubclass(exc_type, Exception):
            return False
        self.audit.record_error(self.stage, self.subject, exc)
        return True


class AuditReport:
    def __init__(self):
        self.errors = Counter()          # "stage | IfcClass | ErrorType" -> n
        self.error_samples = {}
        self.exported = Counter()        # records per IFC class
        self.skipped = Counter()         # blacklisted classes
        self.folded = Counter()          # relationships / property definitions
        self.pset_names = Counter()      # every pset/qto name seen
        self.dropped_psets = Counter()
        self.tessellation_skipped = Counter()   # by design: site, non-slab openings
        self.geometry_failures = Counter()
        self.duplicate_guids = []
        self.no_storeys = False
        self.rooms_without_function = []
        self.rooms_area_from_footprint = []
        self.doors_without_room = []
        self.doors_geometric_only = []
        self.virtual_boundaries_without_edge = []
        self.equipment_geometric = []
        self.equipment_without_room = []
        self.obstacles_skipped = Counter()      # "reason | IfcClass" -> n

    def guard(self, stage, subject):
        return _Guard(self, stage, subject)

    def record_error(self, stage, subject, exc):
        cls = subject if isinstance(subject, str) else subject.is_a()
        key = f"{stage} | {cls} | {type(exc).__name__}"
        self.errors[key] += 1
        if key not in self.error_samples:
            gid = None if isinstance(subject, str) else guid_of(subject)
            self.error_samples[key] = f"{gid or ''} {exc}".strip()[:300]

    def to_dict(self, model, units):
        return {
            "schema": model.schema_identifier,
            "units": units.describe(),
            "counts_per_class": dict(sorted(self.exported.items())),
            "skipped_classes": dict(sorted(self.skipped.items())),
            "folded_definitions": dict(sorted(self.folded.items())),
            "pset_names": dict(sorted(self.pset_names.items())),
            "dropped_psets": dict(sorted(self.dropped_psets.items())),
            "tessellation_skipped": dict(sorted(self.tessellation_skipped.items())),
            "geometry_failures": dict(sorted(self.geometry_failures.items())),
            "duplicate_guids": self.duplicate_guids,
            "no_storeys": self.no_storeys,
            "rooms_without_function": self.rooms_without_function,
            "rooms_area_from_footprint": self.rooms_area_from_footprint,
            "doors_without_room": self.doors_without_room,
            "doors_geometric_only": self.doors_geometric_only,
            "virtual_boundaries_without_edge": self.virtual_boundaries_without_edge,
            "equipment_geometric": self.equipment_geometric,
            "equipment_without_room": self.equipment_without_room,
            "obstacles_skipped": dict(sorted(self.obstacles_skipped.items())),
            "errors": dict(sorted(self.errors.items())),
            "error_samples": self.error_samples,
        }

    def print_summary(self, model, units, full):
        p = print
        p(f"Schema: {model.schema_identifier}   units: {units.describe()['project_units']}")
        p(f"Records: {len(full['spatial'])} spatial, {len(full['storeys'])} storeys, "
          f"{len(full['spaces'])} spaces, {len(full['elements'])} elements, {len(full['types'])} types, "
          f"{len(full['groups'])} groups, {len(full['space_boundaries'])} space boundaries")
        p(f"Adjacency: {len(full['adjacency']['edges'])} edges")
        for label, counter in (("Skipped (blacklist)", self.skipped),
                               ("Dropped psets", self.dropped_psets),
                               ("Not tessellated (by design)", self.tessellation_skipped),
                               ("Geometry failures", self.geometry_failures),
                               ("Obstacles skipped", self.obstacles_skipped)):
            if counter:
                p(f"{label}: {dict(counter)}")
        if self.no_storeys:
            p("WARNING: model has no IfcBuildingStorey; obstacles use one whole-model layer")
        if self.duplicate_guids:
            p(f"WARNING: {len(self.duplicate_guids)} duplicate GlobalId(s)")
        for label, items in (
            ("rooms without a usable function field", self.rooms_without_function),
            ("rooms with area from footprint only", self.rooms_area_from_footprint),
            ("doors with no room link", self.doors_without_room),
            ("doors linked only geometrically", self.doors_geometric_only),
            ("virtual boundaries that produced no edge", self.virtual_boundaries_without_edge),
            ("equipment placed geometrically", self.equipment_geometric),
            ("equipment with no room", self.equipment_without_room),
        ):
            if items:
                p(f"{len(items):5d} {label}")
        if self.errors:
            p(f"Caught exceptions ({sum(self.errors.values())}):")
            for key, n in sorted(self.errors.items()):
                p(f"  {n:5d}  {key}")

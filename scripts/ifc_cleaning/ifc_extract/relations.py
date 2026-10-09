"""Relationships: one pass over every relationship class, building id-keyed
lookups. Schema-agnostic: IFC2X3 aggregates spaces under storeys and has no
IsNestedBy inverse, so inverse attributes are not used directly."""

import re
from collections import defaultdict

from .common import OUTSIDE, by_type, guid_of


class RelationshipResolver:
    def __init__(self, model, audit):
        self.model, self.audit = model, audit
        self.container = {}
        self.contained = defaultdict(list)
        self.references = defaultdict(list)
        self.referenced = defaultdict(list)
        self.aggregate_parent = {}
        self.aggregate_children = defaultdict(list)
        self.nest_parent = {}
        self.nest_children = defaultdict(list)
        self.groups = defaultdict(list)
        self.group_members = defaultdict(list)
        self.type_instances = defaultdict(list)
        self.filled_opening = {}       # door/window id -> opening
        self.voided_host = {}          # opening id -> host element
        self.boundaries_by_element = defaultdict(list)
        self.space_boundaries = []

        for rel in by_type(model, "IfcRelContainedInSpatialStructure"):
            with audit.guard("relations", rel):
                for e in rel.RelatedElements or ():
                    self.container[e.id()] = rel.RelatingStructure
                    self.contained[rel.RelatingStructure.id()].append(e)
        for rel in by_type(model, "IfcRelReferencedInSpatialStructure"):
            with audit.guard("relations", rel):
                for e in rel.RelatedElements or ():
                    self.references[e.id()].append(rel.RelatingStructure)
                    self.referenced[rel.RelatingStructure.id()].append(e)
        for cls, parent, children in (
            ("IfcRelAggregates", self.aggregate_parent, self.aggregate_children),
            ("IfcRelNests", self.nest_parent, self.nest_children),
        ):
            for rel in by_type(model, cls):
                with audit.guard("relations", rel):
                    for e in rel.RelatedObjects or ():
                        parent[e.id()] = rel.RelatingObject
                        children[rel.RelatingObject.id()].append(e)
        for rel in by_type(model, "IfcRelAssignsToGroup"):
            with audit.guard("relations", rel):
                for e in rel.RelatedObjects or ():
                    self.groups[e.id()].append(rel.RelatingGroup)
                    self.group_members[rel.RelatingGroup.id()].append(e)
        for rel in by_type(model, "IfcRelDefinesByType"):
            with audit.guard("relations", rel):
                for e in rel.RelatedObjects or ():
                    self.type_instances[rel.RelatingType.id()].append(e)
        for rel in by_type(model, "IfcRelFillsElement"):
            with audit.guard("relations", rel):
                self.filled_opening[rel.RelatedBuildingElement.id()] = rel.RelatingOpeningElement
        for rel in by_type(model, "IfcRelVoidsElement"):
            with audit.guard("relations", rel):
                self.voided_host[rel.RelatedOpeningElement.id()] = rel.RelatingBuildingElement
        for rel in by_type(model, "IfcRelSpaceBoundary"):
            with audit.guard("space_boundary", rel):
                rec = self._boundary_record(rel)
                self.space_boundaries.append(rec)
                elem = rel.RelatedBuildingElement
                if elem is not None:
                    self.boundaries_by_element[elem.id()].append(rec)

        self._storey_cache = {}

    @staticmethod
    def _boundary_level(rel):
        if rel.is_a("IfcRelSpaceBoundary2ndLevel"):
            return "2nd"
        if rel.is_a("IfcRelSpaceBoundary1stLevel"):
            return "1st"
        # IFC2X3 implementer agreement: Name "1stLevel"/"2ndLevel", Description "2a"/"2b".
        for text in (rel.Name, rel.Description):
            if text and re.search(r"(1st|2nd|2a|2b)", text, re.I):
                return text
        return "unspecified"

    def _boundary_record(self, rel):
        space = rel.RelatingSpace
        elem = rel.RelatedBuildingElement
        external_space = space is not None and space.is_a("IfcExternalSpatialElement")
        return {
            "guid": rel.GlobalId,
            "ifc_class": rel.is_a(),
            "space": OUTSIDE if external_space else guid_of(space),
            "element": guid_of(elem),
            "element_class": elem.is_a() if elem is not None else None,
            "physical_or_virtual": rel.PhysicalOrVirtualBoundary,
            "internal_or_external": rel.InternalOrExternalBoundary,
            "level": self._boundary_level(rel),
            "corresponding_boundary": guid_of(getattr(rel, "CorrespondingBoundary", None)),
            "parent_boundary": guid_of(getattr(rel, "ParentBoundary", None)),
        }

    def _parent(self, entity):
        i = entity.id()
        return (self.container.get(i) or self.aggregate_parent.get(i)
                or self.nest_parent.get(i) or self.filled_opening.get(i)
                or self.voided_host.get(i))

    def storey_of(self, entity):
        i = entity.id()
        if i in self._storey_cache:
            return self._storey_cache[i]
        cur, seen, found = entity, set(), None
        while cur is not None and cur.id() not in seen:
            if cur.is_a("IfcBuildingStorey"):
                found = cur
                break
            seen.add(cur.id())
            cur = self._parent(cur)
        self._storey_cache[i] = found
        return found

    def space_of(self, entity):
        """Room by relation: containment, spatial reference, or via the
        aggregation/nesting parent chain. Never walks past a storey."""
        cur, seen = entity, set()
        while cur is not None and cur.id() not in seen:
            if cur is not entity and cur.is_a("IfcSpace"):
                return cur
            for ref in self.references.get(cur.id(), ()):
                if ref.is_a("IfcSpace"):
                    return ref
            if cur.is_a("IfcSpatialStructureElement") and not cur.is_a("IfcSpace"):
                return None
            seen.add(cur.id())
            cur = self._parent(cur)
        return None

    def relations(self, entity):
        i = entity.id()
        groups = list(self.groups.get(i, ()))
        groups += [r for r in self.references.get(i, ()) if r.is_a("IfcSpatialZone")]
        return {
            "container": guid_of(self.container.get(i)),
            "references": [guid_of(r) for r in self.references.get(i, ())],
            "aggregate_parent": guid_of(self.aggregate_parent.get(i)),
            "aggregate_children": [guid_of(c) for c in self.aggregate_children.get(i, ())],
            "nest_parent": guid_of(self.nest_parent.get(i)),
            "nest_children": [guid_of(c) for c in self.nest_children.get(i, ())],
            "groups": [guid_of(g) for g in groups],
        }

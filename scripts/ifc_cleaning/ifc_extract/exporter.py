"""The generic per-entity record: attributes, psets/qtos with type
inheritance, type object, resolved type, materials, appearance,
classifications, folded relations and geometry fields.

Resource entities (points, property values, colours, materials) never become
records of their own; they are folded into the object that uses them."""

import ifcopenshell
import ifcopenshell.util.classification as ifc_classification
import ifcopenshell.util.element as ifc_element

from .common import guid_of, rnd

# Direct attributes not dumped. Representation/ObjectPlacement/RepresentationMaps
# are replaced by the geometry fields; HasPropertySets is folded into psets;
# IfcOwnerHistory (not rooted, but referenced everywhere) is administrative.
DROPPED_ATTRIBUTES = {
    "OwnerHistory", "Representation", "ObjectPlacement",
    "RepresentationMaps", "HasPropertySets",
}

# Exact pset/qto names to drop. Fill only after inspecting audit.pset_names;
# extend per run with --drop-pset.
DROP_PSET_EXACT = set()

QUANTITY_VALUES = {
    "IfcQuantityLength": ("LengthValue", "IfcLengthMeasure"),
    "IfcQuantityArea": ("AreaValue", "IfcAreaMeasure"),
    "IfcQuantityVolume": ("VolumeValue", "IfcVolumeMeasure"),
    "IfcQuantityCount": ("CountValue", None),
    "IfcQuantityWeight": ("WeightValue", None),
    "IfcQuantityTime": ("TimeValue", None),
    "IfcQuantityNumber": ("NumberValue", None),
}

UNSET = (None, "USERDEFINED", "NOTDEFINED")


class EntityExporter:
    def __init__(self, units, rels, geom, audit, drop_psets):
        self.units, self.rels, self.geom, self.audit = units, rels, geom, audit
        self.drop_psets = drop_psets
        self._style_cache = {}

    # --- generic values -------------------------------------------------

    def value(self, v, measure=None):
        if isinstance(v, ifcopenshell.entity_instance):
            return self.ref(v)
        if isinstance(v, (tuple, list)):
            return [self.value(x, measure) for x in v]
        if measure and isinstance(v, (int, float)):
            return self.units.convert(v, measure)
        return v

    def ref(self, entity):
        if entity.id() == 0:                      # typed value inside a SELECT
            return self.value(entity.wrappedValue, entity.is_a())
        if entity.is_a("IfcRoot"):
            return entity.GlobalId
        return self.summary(entity)

    def summary(self, entity):
        """Shallow inline summary of a resource entity (no GUID): scalar
        attributes kept, nested resources reduced to their class name."""
        out = {"type": entity.is_a()}
        for k, v in entity.get_info(recursive=False, include_identifier=False).items():
            if k == "type" or v is None:
                continue
            out[k] = self._shallow(v, self.units.attribute_measure(entity.is_a(), k))
        return out

    def _shallow(self, v, measure):
        if isinstance(v, ifcopenshell.entity_instance):
            if v.id() == 0:
                return self.value(v.wrappedValue, v.is_a())
            return v.GlobalId if v.is_a("IfcRoot") else v.is_a()
        if isinstance(v, (tuple, list)):
            return [self._shallow(x, measure) for x in v]
        return self.value(v, measure)

    def attributes(self, entity):
        out = {}
        for k, v in entity.get_info(recursive=False, include_identifier=False).items():
            if k == "type" or k in DROPPED_ATTRIBUTES:
                continue
            out[k] = self.value(v, self.units.attribute_measure(entity.is_a(), k))
        return out

    # --- psets / qtos -----------------------------------------------------

    def property_sets(self, entity):
        """(psets, qtos) with type inheritance: type definitions first, then
        occurrence definitions override per property. Values unit-converted."""
        defs = []
        if entity.is_a("IfcTypeObject"):
            defs += list(entity.HasPropertySets or ())
        else:
            type_obj = ifc_element.get_type(entity) if entity.is_a("IfcObject") else None
            if type_obj is not None:
                defs += list(type_obj.HasPropertySets or ())
            for rel in getattr(entity, "IsDefinedBy", None) or ():
                if rel.is_a("IfcRelDefinesByProperties"):
                    d = rel.RelatingPropertyDefinition
                    defs += list(d) if isinstance(d, (tuple, list)) else [d]
        psets, qtos = {}, {}
        for d in defs:
            name = getattr(d, "Name", None) or d.is_a()
            self.audit.pset_names[name] += 1
            if name in self.drop_psets:
                self.audit.dropped_psets[name] += 1
                continue
            with self.audit.guard("pset", d):
                if d.is_a("IfcElementQuantity"):
                    qtos.setdefault(name, {}).update(self._quantities(d.Quantities))
                elif d.is_a("IfcPropertySet"):
                    psets.setdefault(name, {}).update(
                        {p.Name: self._property(p) for p in d.HasProperties or ()})
                else:   # predefined sets, e.g. IfcDoorLiningProperties
                    psets.setdefault(name, {}).update(self.attributes(d))
        return psets, qtos

    def _quantities(self, quantities):
        out = {}
        for q in quantities or ():
            with self.audit.guard("quantity", q):
                spec = QUANTITY_VALUES.get(q.is_a())
                if spec:
                    attr, measure = spec
                    out[q.Name] = self.units.convert(getattr(q, attr), measure, getattr(q, "Unit", None))
                elif q.is_a("IfcPhysicalComplexQuantity"):
                    out[q.Name] = self._quantities(q.HasQuantities)
                else:
                    out[q.Name] = self.summary(q)
        return out

    def _typed(self, v, unit=None):
        if isinstance(v, ifcopenshell.entity_instance) and v.id() == 0:
            raw = v.wrappedValue
            if isinstance(raw, (tuple, list)):
                return list(raw)
            return self.units.convert(raw, v.is_a(), unit)
        return self.value(v)

    def _property(self, p):
        cls = p.is_a()
        unit = getattr(p, "Unit", None)
        if cls == "IfcPropertySingleValue":
            return self._typed(p.NominalValue, unit)
        if cls == "IfcPropertyEnumeratedValue":
            return [self._typed(v) for v in p.EnumerationValues or ()]
        if cls == "IfcPropertyListValue":
            return [self._typed(v, unit) for v in p.ListValues or ()]
        if cls == "IfcPropertyBoundedValue":
            return {"lower": self._typed(p.LowerBoundValue, unit),
                    "upper": self._typed(p.UpperBoundValue, unit),
                    "set_point": self._typed(getattr(p, "SetPointValue", None), unit)}
        if cls == "IfcPropertyTableValue":
            return {"defining": [self._typed(v) for v in p.DefiningValues or ()],
                    "defined": [self._typed(v) for v in p.DefinedValues or ()]}
        if cls == "IfcPropertyReferenceValue":
            ref = p.PropertyReference
            return self.ref(ref) if ref is not None else None
        if cls == "IfcComplexProperty":
            return {sub.Name: self._property(sub) for sub in p.HasProperties or ()}
        return self.summary(p)

    # --- type, materials, appearance, classification ----------------------

    @staticmethod
    def resolved_type(entity, type_obj):
        """PredefinedType (inherited from the type when the occurrence leaves it
        open), falling back to ObjectType / ElementType for USERDEFINED/NOTDEFINED."""
        pt = getattr(entity, "PredefinedType", None)
        if pt in UNSET and type_obj is not None and getattr(type_obj, "PredefinedType", None) not in UNSET:
            pt = type_obj.PredefinedType
        if pt not in UNSET:
            return pt, "PredefinedType"
        for source, owner, attr in (("ObjectType", entity, "ObjectType"),
                                    ("ElementType", entity, "ElementType"),
                                    ("ElementType", type_obj, "ElementType")):
            val = getattr(owner, attr, None) if owner is not None else None
            if val:
                return val, source
        return pt, "PredefinedType" if pt else None

    def type_object(self, type_obj):
        if type_obj is None:
            return None
        return {
            "guid": type_obj.GlobalId,
            "name": type_obj.Name,
            "predefined_type": getattr(type_obj, "PredefinedType", None),
            "attributes": self.attributes(type_obj),
        }

    def materials(self, entity):
        """(material record, IfcMaterial entities used) — the latter feed appearance."""
        mat = ifc_element.get_material(entity, should_skip_usage=True)
        if mat is None:
            return None, []
        used = []

        def name_of(m):
            if m is not None:
                used.append(m)
                return m.Name
            return None

        cls = mat.is_a()
        if cls == "IfcMaterial":
            out = {"kind": "material", "name": name_of(mat), "category": getattr(mat, "Category", None)}
        elif cls == "IfcMaterialLayerSet":
            layers = [{
                "material": name_of(l.Material),
                "thickness_m": rnd(self.units.convert(l.LayerThickness, "IfcLengthMeasure"), 4),
                "name": getattr(l, "Name", None),
                "category": getattr(l, "Category", None),
                "is_ventilated": l.IsVentilated,
            } for l in mat.MaterialLayers or ()]
            out = {"kind": "layer_set", "name": mat.LayerSetName, "layers": layers,
                   "total_thickness_m": rnd(sum(l["thickness_m"] or 0 for l in layers), 4)}
        elif cls == "IfcMaterialConstituentSet":
            out = {"kind": "constituent_set", "name": mat.Name, "constituents": [{
                "name": c.Name, "material": name_of(c.Material),
                "fraction": c.Fraction, "category": c.Category,
            } for c in mat.MaterialConstituents or ()]}
        elif cls == "IfcMaterialProfileSet":
            out = {"kind": "profile_set", "name": mat.Name, "profiles": [{
                "name": p.Name, "material": name_of(p.Material),
                "profile": {"type": p.Profile.is_a(), "name": p.Profile.ProfileName} if p.Profile else None,
                "category": p.Category,
            } for p in mat.MaterialProfiles or ()]}
        elif cls == "IfcMaterialList":
            out = {"kind": "list", "materials": [name_of(m) for m in mat.Materials or ()]}
        else:
            out = self.summary(mat)
        return out, used

    def _surface_styles(self, styled_item, source, out):
        for s in styled_item.Styles or ():
            inner = s.Styles if s.is_a("IfcPresentationStyleAssignment") else [s]   # IFC2X3 wrapper
            for style in inner or ():
                if not isinstance(style, ifcopenshell.entity_instance) or not style.is_a("IfcSurfaceStyle"):
                    continue
                for elem in style.Styles or ():
                    if not elem.is_a("IfcSurfaceStyleShading"):
                        continue
                    c = elem.SurfaceColour
                    out[(style.id(), source)] = {
                        "name": style.Name,
                        "side": style.Side,
                        "rgb": [rnd(c.Red, 4), rnd(c.Green, 4), rnd(c.Blue, 4)] if c is not None else None,
                        "transparency": getattr(elem, "Transparency", None),
                        "source": source,
                    }

    def _item_styles(self, item, out, depth=0):
        if depth > 8:
            return
        for si in getattr(item, "StyledByItem", None) or ():
            self._surface_styles(si, "element", out)
        if item.is_a("IfcMappedItem"):
            rep = item.MappingSource.MappedRepresentation
            if rep.id() not in self._style_cache:      # shared by every occurrence of a type
                found = {}
                for sub in rep.Items or ():
                    self._item_styles(sub, found, depth + 1)
                self._style_cache[rep.id()] = found
            out.update(self._style_cache[rep.id()])

    def appearance(self, entity, materials_used):
        out = {}
        rep = getattr(entity, "Representation", None)
        for r in (rep.Representations if rep is not None else ()) or ():
            for item in r.Items or ():
                self._item_styles(item, out)
        for mat in materials_used:
            for dr in getattr(mat, "HasRepresentation", None) or ():
                for r in dr.Representations or ():
                    for item in r.Items or ():
                        if item.is_a("IfcStyledItem"):
                            self._surface_styles(item, f"material:{mat.Name}", out)
        return list(out.values())

    def classifications(self, entity):
        out = []
        for ref in ifc_classification.get_references(entity) or ():
            system = None
            with self.audit.guard("classification", ref):
                src = ifc_classification.get_classification(ref)
                system = getattr(src, "Name", None) if src is not None else None
            out.append({
                "system": system,
                "code": getattr(ref, "Identification", None) or getattr(ref, "ItemReference", None),
                "name": ref.Name,
                "description": getattr(ref, "Description", None),
            })
        return out

    # --- the record -------------------------------------------------------

    def export(self, entity):
        rec = {
            "guid": entity.GlobalId,
            "ifc_class": entity.is_a(),
            "name": entity.Name,
            "attributes": {},
            "resolved_type": None,
            "resolved_type_source": None,
            "type_object": None,
            "psets": {}, "qtos": {},
            "materials": None, "appearance": [], "classifications": [],
            "relations": {},
            "storey": None,
            "geometry": None,
        }
        type_obj = None
        with self.audit.guard("attributes", entity):
            rec["attributes"] = self.attributes(entity)
        if entity.is_a("IfcObject"):
            with self.audit.guard("type", entity):
                type_obj = ifc_element.get_type(entity)
                rec["type_object"] = self.type_object(type_obj)
        rec["resolved_type"], rec["resolved_type_source"] = self.resolved_type(entity, type_obj)
        with self.audit.guard("psets", entity):
            rec["psets"], rec["qtos"] = self.property_sets(entity)
        used = []
        if entity.is_a("IfcObjectDefinition"):
            with self.audit.guard("materials", entity):
                rec["materials"], used = self.materials(entity)
            with self.audit.guard("classifications", entity):
                rec["classifications"] = self.classifications(entity)
        with self.audit.guard("appearance", entity):
            rec["appearance"] = self.appearance(entity, used)
        if entity.is_a("IfcTypeObject"):
            rec["relations"] = {"instances": [guid_of(e) for e in self.rels.type_instances.get(entity.id(), ())]}
        else:
            rec["relations"] = self.rels.relations(entity)
            rec["storey"] = guid_of(self.rels.storey_of(entity))
        if entity.is_a("IfcProduct"):
            with self.audit.guard("geometry_fields", entity):
                rec["geometry"] = self.geom.fields(entity)
        return rec

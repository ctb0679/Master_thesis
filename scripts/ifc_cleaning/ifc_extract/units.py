"""Unit conversion to metres / m² / m³ / degrees.

Project units come from IfcUnitAssignment; a property or quantity carrying
its own Unit uses that instead. Direct attributes have no measure type at
runtime (get_info returns a bare float), so their declared type is looked up
in the schema, e.g. IfcDoor.OverallWidth -> IfcPositiveLengthMeasure."""

import math

import ifcopenshell.util.unit as ifc_unit
from ifcopenshell import ifcopenshell_wrapper

SI_PREFIX = {
    "EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9, "MEGA": 1e6,
    "KILO": 1e3, "HECTO": 1e2, "DECA": 1e1, "DECI": 1e-1, "CENTI": 1e-2,
    "MILLI": 1e-3, "MICRO": 1e-6, "NANO": 1e-9, "PICO": 1e-12,
    "FEMTO": 1e-15, "ATTO": 1e-18,
}


class UnitResolver:
    KINDS = {"LENGTHUNIT": "m", "AREAUNIT": "m2", "VOLUMEUNIT": "m3", "PLANEANGLEUNIT": "deg"}

    def __init__(self, model, audit):
        self.model, self.audit = model, audit
        self.schema = ifcopenshell_wrapper.schema_by_name(model.schema_identifier)
        self.factors = {}
        self.names = {}
        for kind in self.KINDS:
            factor = ifc_unit.calculate_unit_scale(model, kind)
            if kind == "PLANEANGLEUNIT":
                factor = math.degrees(factor)   # to radians, then to degrees
            self.factors[kind] = factor
            unit = ifc_unit.get_project_unit(model, kind)
            self.names[kind] = ifc_unit.get_full_unit_name(unit) if unit else None
        self._attr_cache = {}

    def describe(self):
        return {
            "output_units": {"length": "m", "area": "m2", "volume": "m3", "angle": "deg"},
            "project_units": {k.replace("UNIT", "").lower(): self.names[k] for k in self.KINDS},
            "factors_to_output": {k.replace("UNIT", "").lower(): self.factors[k] for k in self.KINDS},
        }

    @property
    def length(self):
        return self.factors["LENGTHUNIT"]

    def _unit_entity_factor(self, unit):
        if unit.is_a("IfcSIUnit"):
            power = {"AREAUNIT": 2, "VOLUMEUNIT": 3}.get(unit.UnitType, 1)
            return SI_PREFIX.get(unit.Prefix, 1.0) ** power
        if unit.is_a("IfcConversionBasedUnit"):
            cf = unit.ConversionFactor
            inner = self._unit_entity_factor(cf.UnitComponent)
            return None if inner is None else float(cf.ValueComponent.wrappedValue) * inner
        return None

    def convert(self, value, measure_class, unit=None):
        """Convert a raw value of measure_class (e.g. 'IfcPositiveLengthMeasure')."""
        if value is None or not measure_class or not measure_class.endswith("Measure"):
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        kind = ifc_unit.get_measure_unit_type(measure_class)
        if kind not in self.factors:
            return value
        factor = self.factors[kind]
        if unit is not None:
            with self.audit.guard("unit", unit):
                own = self._unit_entity_factor(unit)
                if own is not None:
                    factor = math.degrees(own) if kind == "PLANEANGLEUNIT" else own
        return float(value) * factor

    def attribute_measure(self, ifc_class, attr_name):
        """Declared measure type of a direct attribute, or None."""
        key = (ifc_class, attr_name)
        if key not in self._attr_cache:
            measure = None
            with self.audit.guard("schema", ifc_class):
                for attr in self.schema.declaration_by_name(ifc_class).all_attributes():
                    if attr.name() != attr_name:
                        continue
                    t = attr.type_of_attribute()
                    while t.as_aggregation_type():
                        t = t.as_aggregation_type().type_of_element()
                    named = t.as_named_type()
                    if named:
                        name = named.declared_type().name()
                        if name.endswith("Measure"):
                            measure = name
                    break
            self._attr_cache[key] = measure
        return self._attr_cache[key]

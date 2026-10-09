"""ifc_extract — generic, schema-agnostic IFC-to-JSON extraction.

Modules, in pipeline order:
    common      shared constants/helpers and JSON serialisation
    audit       counters for everything skipped, dropped, guessed or failed
    units       project/property units -> m, m², m³, degrees
    relations   one pass over all relationships -> id-keyed lookups
    geometry    tessellation, the four geometry fields, coordinate frame
    exporter    the generic per-entity record
    rooms_doors room function/area, door details, adjacency, equipment rooms
    obstacles   per-storey obstacle layers
    extractor   IfcExtractor: routes entities and runs the passes above

Only ifcopenshell, shapely, numpy (required by both) and the standard library
are used; nothing outside this folder is imported.
"""

EXTRACTOR_VERSION = "2.1"

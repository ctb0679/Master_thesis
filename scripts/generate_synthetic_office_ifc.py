"""
Generate a synthetic, well-defined single-storey office IFC4X3 model
for the object-search thesis pipeline.

Design goals:
- Every IfcSpace has Name / LongName / ObjectType set (matches Stage-2
  classification's expected fields).
- Equipment modeled as IfcFurniture (desks/chairs/tables) and
  IfcBuildingElementProxy (monitors/screens/appliances) — mirrors the
  entity split already used by the extraction pipeline.
- Equipment contained in its IfcSpace via spatial.assign_container
  (-> LOCATED_IN in the graph).
- Doors modeled as IfcDoor, contained in the storey, with explicit
  IfcRelSpaceBoundary links to every space they connect (room linkage).
- Room functions are deliberately differentiated: water sources only
  in Kitchen/Restroom; cup-plausible surfaces only in Offices/Kitchen/
  Reception; Storage/Server room has neither — gives real ground truth
  for the leak-source and cup-collection example tasks.
"""

import ifcopenshell
import ifcopenshell.api.root as root
import ifcopenshell.api.unit as unit
import ifcopenshell.api.context as context
import ifcopenshell.api.aggregate as aggregate
import ifcopenshell.api.spatial as spatial
import ifcopenshell.guid as guid_mod
import datetime

# ---------------------------------------------------------------------
# Model / project setup
# ---------------------------------------------------------------------

model = ifcopenshell.file(schema="IFC4X3")

project = root.create_entity(model, ifc_class="IfcProject", name="Synthetic Office Building")
unit.assign_unit(model, length={"is_metric": True, "raw": "METERS"})  # explicit metres — default is millimetres and was silently wrong

model_ctx = context.add_context(model, context_type="Model")
body_ctx = context.add_context(
    model, context_type="Model", context_identifier="Body",
    target_view="MODEL_VIEW", parent=model_ctx,
)

site = root.create_entity(model, ifc_class="IfcSite", name="Site")
building = root.create_entity(model, ifc_class="IfcBuilding", name="Synthetic Office Building")
storey = root.create_entity(model, ifc_class="IfcBuildingStorey", name="Ground Floor")
storey.Elevation = 0.0

aggregate.assign_object(model, relating_object=project, products=[site])
aggregate.assign_object(model, relating_object=site, products=[building])
aggregate.assign_object(model, relating_object=building, products=[storey])

WALL_T = 0.15
WALL_H = 3.0
DOOR_W = 1.0
DOOR_H = 2.1

# ---------------------------------------------------------------------
# Geometry helpers (all coordinates are absolute world coords; every
# object placement is identity at the origin to keep this simple)
# ---------------------------------------------------------------------

def identity_placement():
    loc = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    ax = model.create_entity("IfcAxis2Placement3D", Location=loc)
    return model.create_entity("IfcLocalPlacement", RelativePlacement=ax)


def rect_profile(x0, y0, x1, y1):
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    ifc_pts = [model.create_entity("IfcCartesianPoint", Coordinates=p) for p in pts]
    poly = model.create_entity("IfcPolyline", Points=ifc_pts)
    return model.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=poly)


def extruded_shape(profile, z_base, height):
    axis_pt = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, z_base))
    axis_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    ref_dir = model.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    pos = model.create_entity("IfcAxis2Placement3D", Location=axis_pt, Axis=axis_dir, RefDirection=ref_dir)
    extrude_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    solid = model.create_entity(
        "IfcExtrudedAreaSolid", SweptArea=profile, Position=pos,
        ExtrudedDirection=extrude_dir, Depth=height,
    )
    shape_rep = model.create_entity(
        "IfcShapeRepresentation", ContextOfItems=body_ctx,
        RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid],
    )
    return model.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])


def make_product(ifc_class, name, x0, y0, x1, y1, z_base=0.0, height=WALL_H, predefined_type=None):
    kwargs = dict(GlobalId=guid_mod.new(), Name=name)
    ent = model.create_entity(ifc_class, **kwargs)
    ent.ObjectPlacement = identity_placement()
    ent.Representation = extruded_shape(rect_profile(x0, y0, x1, y1), z_base, height)
    if predefined_type and hasattr(ent, "PredefinedType"):
        try:
            ent.PredefinedType = predefined_type
        except Exception:
            pass
    return ent


def make_space(name, long_name, object_type, x0, y0, x1, y1, height=WALL_H):
    space = root.create_entity(model, ifc_class="IfcSpace", name=name, predefined_type="INTERNAL")
    space.LongName = long_name
    space.ObjectType = object_type
    space.ObjectPlacement = identity_placement()
    space.Representation = extruded_shape(rect_profile(x0, y0, x1, y1), 0.0, height)
    aggregate.assign_object(model, relating_object=storey, products=[space])

    area = round(abs((x1 - x0) * (y1 - y0)), 3)
    volume = round(area * height, 3)
    qty_area = model.create_entity(
        "IfcQuantityArea", Name="NetFloorArea", AreaValue=area,
    )
    qty_area_gross = model.create_entity(
        "IfcQuantityArea", Name="GrossFloorArea", AreaValue=area,
    )
    qty_vol = model.create_entity(
        "IfcQuantityVolume", Name="NetVolume", VolumeValue=volume,
    )
    qset = model.create_entity(
        "IfcElementQuantity", GlobalId=guid_mod.new(),
        Name="Qto_SpaceBaseQuantities",
        Quantities=[qty_area, qty_area_gross, qty_vol],
    )
    model.create_entity(
        "IfcRelDefinesByProperties", GlobalId=guid_mod.new(),
        RelatedObjects=[space], RelatingPropertyDefinition=qset,
    )
    return space


def make_wall(name, x0, y0, x1, y1):
    return make_product("IfcWall", name, x0, y0, x1, y1, z_base=0.0, height=WALL_H)


def make_perimeter_walls(room_name, x0, y0, x1, y1, space, skip_segments=None):
    """4 thin perimeter walls, centered on the room boundary. skip_segments
    lets us leave a gap (unused here, doors are just overlaid) — kept simple."""
    walls = []
    segs = {
        "S": (x0 - WALL_T / 2, y0 - WALL_T / 2, x1 + WALL_T / 2, y0 + WALL_T / 2),
        "N": (x0 - WALL_T / 2, y1 - WALL_T / 2, x1 + WALL_T / 2, y1 + WALL_T / 2),
        "W": (x0 - WALL_T / 2, y0 - WALL_T / 2, x0 + WALL_T / 2, y1 + WALL_T / 2),
        "E": (x1 - WALL_T / 2, y0 - WALL_T / 2, x1 + WALL_T / 2, y1 + WALL_T / 2),
    }
    skip = skip_segments or []
    for label, (a, b, c, d) in segs.items():
        if label in skip:
            continue
        wall = make_wall(f"{room_name} Wall {label}", a, b, c, d)
        spatial.assign_container(model, relating_structure=storey, products=[wall])
        boundary = model.create_entity(
            "IfcRelSpaceBoundary", GlobalId=guid_mod.new(),
            RelatingSpace=space, RelatedBuildingElement=wall,
            PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="INTERNAL",
        )
        walls.append(wall)
    return walls


def make_door(name, x, y, connects):
    """Door as a simple box at (x,y); `connects` = list of IfcSpace this
    door links (1 = exterior door, 2 = interior door between two spaces)."""
    x0, y0 = x - DOOR_W / 2, y - WALL_T / 2
    x1, y1 = x + DOOR_W / 2, y + WALL_T / 2
    door = make_product("IfcDoor", name, x0, y0, x1, y1, z_base=0.0, height=DOOR_H)
    door.OverallWidth = DOOR_W
    door.OverallHeight = DOOR_H
    spatial.assign_container(model, relating_structure=storey, products=[door])
    for space in connects:
        model.create_entity(
            "IfcRelSpaceBoundary", GlobalId=guid_mod.new(),
            RelatingSpace=space, RelatedBuildingElement=door,
            PhysicalOrVirtualBoundary="PHYSICAL", InternalOrExternalBoundary="INTERNAL",
        )
    return door


def make_equipment(name, ifc_class, x, y, z, dx, dy, dz, space, predefined_type=None):
    ent = make_product(
        ifc_class, name, x, y, x + dx, y + dy, z_base=z, height=dz,
        predefined_type=predefined_type,
    )
    spatial.assign_container(model, relating_structure=space, products=[ent])
    return ent


def auto_layout(space, room_box, items, cols=3, margin=0.4, spacing=0.3):
    """Place floor-standing items in a simple grid inside room_box=(x0,y0,x1,y1).
    Each item: dict(name, ifc_class, dx, dy, dz, predefined_type, on=None).
    Items with on=<name of a placed item> are stacked on top of that item
    (used for monitors sitting on desks)."""
    x0, y0, x1, y1 = room_box
    placed = {}
    cursor_x = x0 + margin
    cursor_y = y0 + margin
    row_h = 0.0
    floor_items = [i for i in items if not i.get("on")]
    stacked_items = [i for i in items if i.get("on")]

    for idx, item in enumerate(floor_items):
        dx, dy, dz = item["dx"], item["dy"], item["dz"]
        if cursor_x + dx > x1 - margin:
            cursor_x = x0 + margin
            cursor_y += row_h + spacing
            row_h = 0.0
        px, py = cursor_x, cursor_y
        ent = make_equipment(
            item["name"], item["ifc_class"], px, py, 0.0, dx, dy, dz,
            space, item.get("predefined_type"),
        )
        placed[item["name"]] = (ent, px, py, dz)
        cursor_x += dx + spacing
        row_h = max(row_h, dy)

    for item in stacked_items:
        base_name = item["on"]
        if base_name not in placed:
            continue
        _, bx, by, bz = placed[base_name]
        dx, dy, dz = item["dx"], item["dy"], item["dz"]
        ent = make_equipment(
            item["name"], item["ifc_class"], bx, by, bz, dx, dy, dz,
            space, item.get("predefined_type"),
        )
        placed[item["name"]] = (ent, bx, by, bz + dz)

    return placed


print("Setup OK — helpers defined.")

# =======================================================================
# ROOM DEFINITIONS
# box = (x0, y0, x1, y1)   — world coordinates in metres
# =======================================================================

ROOMS = {
    "Corridor":  dict(long_name="Main Corridor", object_type="Circulation",
                       box=(0, 5, 21, 7.5)),
    "Office1":   dict(long_name="Office 1 - Shared Open Office", object_type="Office",
                       box=(0, 7.5, 5, 12.5)),
    "Office2":   dict(long_name="Office 2 - Shared Open Office", object_type="Office",
                       box=(5, 7.5, 10, 12.5)),
    "Office3":   dict(long_name="Office 3 - Manager Office", object_type="Office",
                       box=(10, 7.5, 14, 11.5)),
    "Seminar":   dict(long_name="Seminar / Meeting Room", object_type="Meeting",
                       box=(14, 7.5, 21, 13.5)),
    "Reception": dict(long_name="Reception / Entrance Lobby", object_type="Reception",
                       box=(0, 0, 6, 5)),
    "Kitchen":   dict(long_name="Kitchen / Break Room", object_type="Kitchen",
                       box=(6, 0, 11, 5)),
    "Restroom":  dict(long_name="Restroom (Combined)", object_type="Restroom",
                       box=(11, 0, 14, 4)),
    "Storage":   dict(long_name="Storage / Server Room", object_type="Technical",
                       box=(14, 0, 18, 4)),
}

spaces = {}
for key, r in ROOMS.items():
    x0, y0, x1, y1 = r["box"]
    spaces[key] = make_space(key, r["long_name"], r["object_type"], x0, y0, x1, y1)

for key, r in ROOMS.items():
    x0, y0, x1, y1 = r["box"]
    make_perimeter_walls(key, x0, y0, x1, y1, spaces[key])

# =======================================================================
# DOORS — every room connects to the Corridor; Reception also gets an
# exterior entrance door.
# =======================================================================

corridor = spaces["Corridor"]
make_door("Door - Office 1 to Corridor",   2.5,  7.5, [spaces["Office1"], corridor])
make_door("Door - Office 2 to Corridor",   7.5,  7.5, [spaces["Office2"], corridor])
make_door("Door - Office 3 to Corridor",   12.0, 7.5, [spaces["Office3"], corridor])
make_door("Door - Seminar Room to Corridor", 17.5, 7.5, [spaces["Seminar"], corridor])
make_door("Door - Reception to Corridor",  3.0,  5.0, [spaces["Reception"], corridor])
make_door("Door - Kitchen to Corridor",    8.5,  5.0, [spaces["Kitchen"], corridor])
make_door("Door - Restroom to Corridor",   12.5, 4.0, [spaces["Restroom"], corridor])
make_door("Door - Storage to Corridor",    16.0, 4.0, [spaces["Storage"], corridor])
make_door("Main Entrance Door", 3.0, 0.0, [spaces["Reception"]])

# =======================================================================
# EQUIPMENT — deliberately differentiated per room (see design rationale
# in module docstring: water sources only in Kitchen/Restroom; cup-
# plausible surfaces only in Offices/Kitchen/Reception).
# =======================================================================

FURN = "IfcFurniture"
PROXY = "IfcBuildingElementProxy"

auto_layout(spaces["Office1"], ROOMS["Office1"]["box"], [
    {"name": "Office1 Desk A", "ifc_class": FURN, "dx": 1.2, "dy": 0.6, "dz": 0.75, "predefined_type": "DESK"},
    {"name": "Office1 Chair A", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Office1 Desk B", "ifc_class": FURN, "dx": 1.2, "dy": 0.6, "dz": 0.75, "predefined_type": "DESK"},
    {"name": "Office1 Chair B", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Office1 Filing Cabinet", "ifc_class": FURN, "dx": 0.5, "dy": 0.4, "dz": 1.2, "predefined_type": "FILINGCABINET"},
    {"name": "Office1 Monitor A", "ifc_class": PROXY, "dx": 0.5, "dy": 0.15, "dz": 0.4, "on": "Office1 Desk A"},
    {"name": "Office1 Monitor B", "ifc_class": PROXY, "dx": 0.5, "dy": 0.15, "dz": 0.4, "on": "Office1 Desk B"},
])

auto_layout(spaces["Office2"], ROOMS["Office2"]["box"], [
    {"name": "Office2 Desk A", "ifc_class": FURN, "dx": 1.2, "dy": 0.6, "dz": 0.75, "predefined_type": "DESK"},
    {"name": "Office2 Chair A", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Office2 Desk B", "ifc_class": FURN, "dx": 1.2, "dy": 0.6, "dz": 0.75, "predefined_type": "DESK"},
    {"name": "Office2 Chair B", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Office2 Filing Cabinet", "ifc_class": FURN, "dx": 0.5, "dy": 0.4, "dz": 1.2, "predefined_type": "FILINGCABINET"},
    {"name": "Office2 Shared Printer", "ifc_class": PROXY, "dx": 0.5, "dy": 0.5, "dz": 0.4},
    {"name": "Office2 Monitor A", "ifc_class": PROXY, "dx": 0.5, "dy": 0.15, "dz": 0.4, "on": "Office2 Desk A"},
    {"name": "Office2 Monitor B", "ifc_class": PROXY, "dx": 0.5, "dy": 0.15, "dz": 0.4, "on": "Office2 Desk B"},
])

auto_layout(spaces["Office3"], ROOMS["Office3"]["box"], [
    {"name": "Office3 Desk", "ifc_class": FURN, "dx": 1.4, "dy": 0.7, "dz": 0.75, "predefined_type": "DESK"},
    {"name": "Office3 Chair", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Office3 Side Table", "ifc_class": FURN, "dx": 0.6, "dy": 0.6, "dz": 0.5, "predefined_type": "TABLE"},
    {"name": "Office3 Bookshelf", "ifc_class": FURN, "dx": 0.8, "dy": 0.35, "dz": 1.8, "predefined_type": "SHELF"},
    {"name": "Office3 Monitor", "ifc_class": PROXY, "dx": 0.5, "dy": 0.15, "dz": 0.4, "on": "Office3 Desk"},
])

auto_layout(spaces["Seminar"], ROOMS["Seminar"]["box"], [
    {"name": "Seminar Conference Table", "ifc_class": FURN, "dx": 3.0, "dy": 1.2, "dz": 0.75, "predefined_type": "TABLE"},
    {"name": "Seminar Chair 1", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Seminar Chair 2", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Seminar Chair 3", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Seminar Chair 4", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Seminar Chair 5", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Seminar Chair 6", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Seminar Whiteboard", "ifc_class": PROXY, "dx": 1.5, "dy": 0.05, "dz": 1.0},
    {"name": "Seminar Projection Screen", "ifc_class": PROXY, "dx": 2.0, "dy": 0.05, "dz": 1.2},
    {"name": "Seminar Projector", "ifc_class": PROXY, "dx": 0.4, "dy": 0.3, "dz": 0.15},
])

auto_layout(spaces["Reception"], ROOMS["Reception"]["box"], [
    {"name": "Reception Desk", "ifc_class": FURN, "dx": 1.6, "dy": 0.7, "dz": 1.1, "predefined_type": "DESK"},
    {"name": "Reception Chair", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Reception Sofa", "ifc_class": FURN, "dx": 1.8, "dy": 0.8, "dz": 0.8, "predefined_type": "SOFA"},
    {"name": "Reception Waiting Chair 1", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Reception Waiting Chair 2", "ifc_class": FURN, "dx": 0.5, "dy": 0.5, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Reception Coffee Table", "ifc_class": FURN, "dx": 0.8, "dy": 0.5, "dz": 0.4, "predefined_type": "TABLE"},
])

auto_layout(spaces["Kitchen"], ROOMS["Kitchen"]["box"], [
    {"name": "Kitchen Sink", "ifc_class": PROXY, "dx": 0.6, "dy": 0.5, "dz": 0.9},
    {"name": "Kitchen Refrigerator", "ifc_class": PROXY, "dx": 0.7, "dy": 0.7, "dz": 1.8},
    {"name": "Kitchen Coffee Machine", "ifc_class": PROXY, "dx": 0.3, "dy": 0.3, "dz": 0.4},
    {"name": "Kitchen Microwave", "ifc_class": PROXY, "dx": 0.5, "dy": 0.4, "dz": 0.3},
    {"name": "Kitchen Countertop", "ifc_class": FURN, "dx": 2.0, "dy": 0.6, "dz": 0.9, "predefined_type": "TABLE"},
    {"name": "Kitchen Water Dispenser", "ifc_class": PROXY, "dx": 0.35, "dy": 0.35, "dz": 1.1},
    {"name": "Kitchen Wall Cabinets", "ifc_class": FURN, "dx": 1.5, "dy": 0.4, "dz": 0.7, "predefined_type": "SHELF"},
    {"name": "Kitchen Dining Table", "ifc_class": FURN, "dx": 1.4, "dy": 0.8, "dz": 0.75, "predefined_type": "TABLE"},
    {"name": "Kitchen Dining Chair 1", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Kitchen Dining Chair 2", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Kitchen Dining Chair 3", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
    {"name": "Kitchen Dining Chair 4", "ifc_class": FURN, "dx": 0.45, "dy": 0.45, "dz": 0.9, "predefined_type": "CHAIR"},
])

auto_layout(spaces["Restroom"], ROOMS["Restroom"]["box"], [
    {"name": "Restroom Toilet Stall 1", "ifc_class": PROXY, "dx": 0.9, "dy": 0.7, "dz": 0.8},
    {"name": "Restroom Toilet Stall 2", "ifc_class": PROXY, "dx": 0.9, "dy": 0.7, "dz": 0.8},
    {"name": "Restroom Sink 1", "ifc_class": PROXY, "dx": 0.5, "dy": 0.4, "dz": 0.85},
    {"name": "Restroom Sink 2", "ifc_class": PROXY, "dx": 0.5, "dy": 0.4, "dz": 0.85},
    {"name": "Restroom Mirror", "ifc_class": PROXY, "dx": 0.6, "dy": 0.05, "dz": 0.6},
    {"name": "Restroom Hand Dryer", "ifc_class": PROXY, "dx": 0.3, "dy": 0.2, "dz": 0.3},
])

auto_layout(spaces["Storage"], ROOMS["Storage"]["box"], [
    {"name": "Server Rack", "ifc_class": PROXY, "dx": 0.6, "dy": 0.8, "dz": 2.0},
    {"name": "Network Switch", "ifc_class": PROXY, "dx": 0.4, "dy": 0.3, "dz": 0.1, "on": "Server Rack"},
    {"name": "Storage Shelving Unit 1", "ifc_class": FURN, "dx": 1.0, "dy": 0.4, "dz": 2.0, "predefined_type": "SHELF"},
    {"name": "Storage Shelving Unit 2", "ifc_class": FURN, "dx": 1.0, "dy": 0.4, "dz": 2.0, "predefined_type": "SHELF"},
])

# =======================================================================
# WRITE + VALIDATE
# =======================================================================

OUT = "/home/claude/synthetic_office.ifc"
model.write(OUT)
print("Wrote", OUT)
print("Spaces:", len(model.by_type("IfcSpace")))
print("Doors:", len(model.by_type("IfcDoor")))
print("Furniture:", len(model.by_type("IfcFurniture")))
print("Proxy equipment:", len(model.by_type("IfcBuildingElementProxy")))
print("Walls:", len(model.by_type("IfcWall")))
print("Space boundaries:", len(model.by_type("IfcRelSpaceBoundary")))


"""Robot profiles: footprint width and capabilities, derived from a URDF.

The paper derives capabilities from robot description files instead of
hard-coding them. A URDF has no capability field, so they are inferred from
sensor evidence and listed with that evidence. The result is written to
data/robot/profiles/<stem>.json, which then overrides the URDF (edit it by
hand to correct anything)."""

import glob
import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_SAFETY_MARGIN_M = 0.1

# <gazebo><sensor type="..."> -> capabilities it gives
SENSOR_CAPABILITIES = {
    "camera": ["rgb"], "wideanglecamera": ["rgb"],
    "depth": ["depth"], "depth_camera": ["depth"],
    "rgbd": ["rgb", "depth"], "rgbd_camera": ["rgb", "depth"],
    "thermal": ["thermal"], "thermal_camera": ["thermal"],
}


class RobotProfileError(RuntimeError):
    """The robot cannot be set up; the message says why. Never answered by
    guessing a width or inventing a robot."""


@dataclass
class RobotProfile:
    """Robot facts the planner needs: footprint width (for door traversal),
    a safety margin, capabilities with their evidence, and whether it may
    leave the building."""
    id: str
    width_m: float
    capabilities: list
    safety_margin_m: float = DEFAULT_SAFETY_MARGIN_M
    allow_exterior: bool = False
    length_m: float = None
    evidence: dict = field(default_factory=dict)
    source: str = ""

    def to_state(self, space):
        return {"id": self.id, "space": space, "capabilities": sorted(self.capabilities),
                "width_m": self.width_m, "safety_margin_m": self.safety_margin_m,
                "allow_exterior": self.allow_exterior}

    def to_json(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        missing = [k for k in ("id", "width_m", "capabilities") if k not in data]
        if missing:
            raise RobotProfileError(f"robot profile {path} lacks {missing}")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_urdf(cls, path):
        """Width = lateral (y) extent of ALL collision geometry, transformed
        into the root link's frame (docs/htn_design.md, deviation 1: the A300's
        base_link has no collision geometry and its wheels stick out past the
        chassis). Stops with RobotProfileError if no width can be derived."""
        path = Path(path)
        root = ET.parse(path).getroot()
        links = {l.get("name"): l for l in root.findall("link")}
        parent_of = {}
        for j in root.findall("joint"):
            parent_of[j.find("child").get("link")] = (j.find("parent").get("link"), _origin(j.find("origin")))
        roots = [name for name in links if name not in parent_of]
        if len(roots) != 1:
            raise RobotProfileError(f"{path}: expected one root link, found {roots}")
        base = roots[0]

        points, used, problems = [], [], []
        for name, link in links.items():
            to_base = _chain(name, parent_of)
            for col in link.findall("collision"):
                geom = col.find("geometry")
                try:
                    local, kind = _geometry_points(geom)
                except RobotProfileError as e:
                    problems.append(f"link {name}: {e}")
                    continue
                pts = _transform(to_base @ _origin(col.find("origin")), local)
                points.append(pts)
                used.append({"link": name, "geometry": kind,
                             "y_min": round(float(pts[:, 1].min()), 4), "y_max": round(float(pts[:, 1].max()), 4)})
        if problems:
            raise RobotProfileError(f"{path}: cannot derive the footprint width:\n  " + "\n  ".join(problems)
                                    + "\nFix the mesh paths, or write the profile JSON by hand.")
        if not points:
            raise RobotProfileError(f"{path}: no collision geometry at all, cannot derive the footprint width; "
                                    f"write data/robot/profiles/{path.stem}.json by hand")
        allpts = np.vstack(points)
        width = float(allpts[:, 1].max() - allpts[:, 1].min())
        length = float(allpts[:, 0].max() - allpts[:, 0].min())
        widest = sorted(used, key=lambda u: u["y_min"])[0]["link"], sorted(used, key=lambda u: -u["y_max"])[0]["link"]

        caps, evidence = ["navigate"], {}
        wheels = [j.get("name") for j in root.findall("joint") if j.get("type") == "continuous"]
        evidence["navigate"] = ("assumed: mobile base" + (f" (continuous joints: {', '.join(wheels)})" if wheels else ""))
        other = []
        for gz in root.iter("gazebo"):
            for sensor in gz.findall("sensor"):
                kind = (sensor.get("type") or "").lower()
                for cap in SENSOR_CAPABILITIES.get(kind, []):
                    if cap not in caps:
                        caps.append(cap)
                    evidence.setdefault(cap, []).append(
                        f'<sensor name="{sensor.get("name")}" type="{kind}"> on link {gz.get("reference")}')
                if kind not in SENSOR_CAPABILITIES:
                    other.append(f'{sensor.get("name")} ({kind})')
        if "rgb" not in caps:
            cams = [n for n in links if "camera" in n.lower()]
            if cams:
                caps.append("rgb")
                evidence["rgb"] = [f"link name {n}" for n in cams]
        if other:
            evidence["other_sensors (no capability used by the planner)"] = other
        evidence["manipulate_door"] = "not inferred (never inferred from a URDF; add it by hand if the robot can open doors)"
        evidence["width"] = {
            "method": f"y-extent of all collision geometry in the {base} frame",
            "root_link": base, "outermost_links": list(widest), "geometries": used,
        }
        return cls(id=path.stem, width_m=round(width, 3), capabilities=caps,
                   length_m=round(length, 3), evidence=evidence, source=str(path))


def select_robot(urdf_dir, profiles_dir, name=None, log=print):
    """The robot to plan for. --robot <stem> picks one URDF; with exactly one
    URDF it is used; otherwise stop with the list. A profile JSON for the
    stem overrides the URDF; a URDF-derived profile is written there."""
    urdf_dir, profiles_dir = Path(urdf_dir), Path(profiles_dir)
    urdfs = sorted(urdf_dir.glob("*.urdf")) if urdf_dir.is_dir() else []
    xacros = sorted(urdf_dir.glob("*.xacro")) if urdf_dir.is_dir() else []
    if xacros:
        log(f"NOTE: {len(xacros)} .xacro file(s) in {urdf_dir} are not read "
            f"({', '.join(x.name for x in xacros)}); expand them to .urdf with xacro first.")
    stems = [u.stem for u in urdfs]
    if name:
        if name not in stems:
            raise RobotProfileError(f"--robot {name}: no {urdf_dir}/{name}.urdf (available: {stems or 'none'})")
        stem = name
    elif len(urdfs) == 1:
        stem = stems[0]
    elif not urdfs:
        raise RobotProfileError(f"no robot: {urdf_dir} contains no *.urdf file")
    else:
        raise RobotProfileError(f"several robots in {urdf_dir}: {stems}; choose one with --robot <name>")
    profile_path = profiles_dir / f"{stem}.json"
    if profile_path.exists():
        return RobotProfile.from_json(profile_path)
    profile = RobotProfile.from_urdf(urdf_dir / f"{stem}.urdf")
    profile.to_json(profile_path)
    log(f"NOTE: robot profile derived from {stem}.urdf and written to {profile_path} "
        f"(width {profile.width_m} m, capabilities {profile.capabilities}); edit it to correct.")
    return profile


# --- URDF geometry helpers -----------------------------------------------------

def _origin(el):
    """4x4 transform of an <origin xyz rpy> element (identity when absent)."""
    m = np.eye(4)
    if el is None:
        return m
    x, y, z = (float(v) for v in (el.get("xyz") or "0 0 0").split())
    r, p, yw = (float(v) for v in (el.get("rpy") or "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(yw), math.sin(yw)
    m[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    m[:3, 3] = [x, y, z]
    return m


def _chain(link, parent_of):
    """Transform from a link's frame to the root link's frame (joints at zero)."""
    m = np.eye(4)
    while link in parent_of:
        parent, joint = parent_of[link]
        m = joint @ m
        link = parent
    return m


def _transform(m, pts):
    return pts @ m[:3, :3].T + m[:3, 3]


def _geometry_points(geom):
    """Points whose extent equals the shape's extent, in the shape's frame."""
    if geom is None or len(geom) == 0:
        raise RobotProfileError("collision without geometry")
    g = geom[0]
    if g.tag == "box":
        sx, sy, sz = (float(v) / 2 for v in g.get("size").split())
        return np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]), "box"
    if g.tag == "cylinder":
        r, h = float(g.get("radius")), float(g.get("length")) / 2
        a = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        return np.vstack([np.c_[ring, np.full(len(a), z)] for z in (-h, h)]), "cylinder"
    if g.tag == "sphere":
        r = float(g.get("radius"))
        return np.array([[s * r if i == k else 0.0 for i in range(3)] for k in range(3) for s in (-1, 1)]), "sphere"
    if g.tag == "mesh":
        filename = g.get("filename")
        scale = np.array([float(v) for v in (g.get("scale") or "1 1 1").split()])
        return _read_stl(_resolve_mesh(filename)) * scale, f"mesh {os.path.basename(filename)}"
    raise RobotProfileError(f"unsupported collision geometry <{g.tag}>")


def _resolve_mesh(uri):
    """package://pkg/rest -> <ament prefix>/share/pkg/rest; file:// -> path."""
    if uri.startswith("file://"):
        path = Path(uri[len("file://"):])
    elif uri.startswith("package://"):
        pkg, _, rest = uri[len("package://"):].partition("/")
        prefixes = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(":") if p]
        candidates = [Path(p) / "share" / pkg / rest for p in prefixes]
        candidates += [Path(p) / rest for p in sorted(glob.glob(f"/opt/ros/*/share/{pkg}"))]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise RobotProfileError(f"cannot find {uri} (package '{pkg}' not in AMENT_PREFIX_PATH or /opt/ros/*/share)")
    else:
        path = Path(uri)
    if not path.exists():
        raise RobotProfileError(f"mesh file {path} does not exist")
    if path.suffix.lower() != ".stl":
        raise RobotProfileError(f"mesh {path.name}: only STL collision meshes are supported")
    return path


def _read_stl(path):
    """Vertices of a binary or ASCII STL file as an (n, 3) array."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * n:
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].reshape(-1, 3).astype(float)
    verts = [[float(t) for t in line.split()[1:4]] for line in data.decode("ascii", "replace").splitlines()
             if line.strip().startswith("vertex")]
    if not verts:
        raise RobotProfileError(f"{path}: not a readable STL file")
    return np.array(verts)


if __name__ == "__main__":
    print(json.dumps(asdict(RobotProfile.from_urdf(sys.argv[1])), indent=2))

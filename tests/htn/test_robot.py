"""Robot profiles from URDF files."""

import json

import pytest

from htn.domain import RobotProfile, RobotProfileError, select_robot

from conftest import ROOT

BOX_URDF = """<?xml version="1.0"?>
<robot name="boxbot">
  <link name="base_link">
    <collision><geometry><box size="0.6 0.4 0.2"/></geometry></collision>
  </link>
  <link name="left_wheel">
    <collision><origin xyz="0 0 0" rpy="1.5707963 0 0"/>
      <geometry><cylinder radius="0.1" length="0.05"/></geometry></collision>
  </link>
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/><child link="left_wheel"/><origin xyz="0 0.25 0"/>
  </joint>
  <link name="camera_link"/>
  <joint name="camera_joint" type="fixed"><parent link="base_link"/><child link="camera_link"/></joint>
  <gazebo reference="camera_link"><sensor name="cam" type="camera"/></gazebo>
</robot>
"""


def write(dir_, name, text=BOX_URDF):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{name}.urdf").write_text(text)


def test_width_from_a_known_box(tmp_path):
    no_wheel = BOX_URDF.split("<link name=\"left_wheel\">")[0] + "</robot>"
    write(tmp_path, "box", no_wheel)
    profile = RobotProfile.from_urdf(tmp_path / "box.urdf")
    assert profile.width_m == pytest.approx(0.4)
    assert profile.length_m == pytest.approx(0.6)
    assert profile.evidence["width"]["geometries"][0]["geometry"] == "box"


def test_width_includes_a_wheel_sticking_out(tmp_path):
    write(tmp_path, "boxbot")
    profile = RobotProfile.from_urdf(tmp_path / "boxbot.urdf")
    # box reaches y = -0.2; wheel at y = 0.25 with half-length 0.025 -> 0.275
    assert profile.width_m == pytest.approx(0.475, abs=1e-3)
    assert "left_wheel" in profile.evidence["width"]["outermost_links"]


def test_capabilities_and_their_evidence(tmp_path):
    write(tmp_path, "boxbot")
    profile = RobotProfile.from_urdf(tmp_path / "boxbot.urdf")
    assert profile.capabilities == ["navigate", "rgb"]
    assert "left_wheel_joint" in profile.evidence["navigate"]
    assert "manipulate_door" not in profile.capabilities


def test_two_urdfs_without_robot_flag_stop_with_the_list(tmp_path):
    write(tmp_path / "urdf", "alpha")
    write(tmp_path / "urdf", "beta")
    with pytest.raises(RobotProfileError, match=r"\['alpha', 'beta'\]"):
        select_robot(tmp_path / "urdf", tmp_path / "profiles")
    assert select_robot(tmp_path / "urdf", tmp_path / "profiles", "beta", log=lambda m: None).id == "beta"


def test_empty_folder_stops(tmp_path):
    (tmp_path / "urdf").mkdir()
    with pytest.raises(RobotProfileError, match="no \\*.urdf"):
        select_robot(tmp_path / "urdf", tmp_path / "profiles")


def test_profile_json_is_written_and_then_overrides_the_urdf(tmp_path):
    write(tmp_path / "urdf", "boxbot")
    first = select_robot(tmp_path / "urdf", tmp_path / "profiles", log=lambda m: None)
    path = tmp_path / "profiles" / "boxbot.json"
    data = json.loads(path.read_text())
    data["capabilities"].append("manipulate_door")            # corrected by hand
    path.write_text(json.dumps(data))
    second = select_robot(tmp_path / "urdf", tmp_path / "profiles")
    assert "manipulate_door" not in first.capabilities
    assert "manipulate_door" in second.capabilities


def test_unresolvable_mesh_stops_instead_of_guessing(tmp_path):
    text = BOX_URDF.replace('<box size="0.6 0.4 0.2"/>', '<mesh filename="package://no_such_pkg/m.stl"/>')
    write(tmp_path, "meshbot", text)
    with pytest.raises(RobotProfileError, match="no_such_pkg"):
        RobotProfile.from_urdf(tmp_path / "meshbot.urdf")


def test_committed_a300_profile():
    profile = RobotProfile.from_json(ROOT / "data" / "robot" / "profiles" / "a300.json")
    assert profile.width_m == pytest.approx(0.665)
    assert profile.capabilities == ["navigate", "rgb", "depth"]
    assert profile.safety_margin_m == 0.1 and profile.allow_exterior is False

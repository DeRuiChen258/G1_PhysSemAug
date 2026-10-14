#!/usr/bin/env python3
"""Generate auditable G1 physical-semantic sequences from LeRobot Parquet data.

The generator deliberately keeps video out of the hot path.  It maps the common
14 arm joints into the official G1 MJCF, runs MuJoCo forward dynamics, and writes
fixed-shape NumPy arrays plus a manifest.  The generated semantic state is a
kinematic/dynamic projection of the recorded joint state; it is not a claim that
the source dataset contains force-sensor ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_BODY_NAMES = [name.replace("_joint", "_link") for name in ARM_JOINT_NAMES]
LEFT_LEG_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG_NAMES = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
WAIST_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
MAX_CONTACTS = 8
SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class Profile:
    profile_id: int
    seed: int
    mass_scale: float
    friction_scale: float
    delay_steps: int
    force: tuple[float, float, float]
    force_start: int
    force_duration: int
    terrain_slope_rad: float
    sensor_occlusion: bool
    profile_hash: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"配置必须是 mapping: {path}")
    return value


def resolve_env_path(value: str, hint: str) -> str:
    """Expand ``${VAR}`` placeholders in a configured path with a clear error."""

    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise SystemExit(
            f"配置路径 {value!r} 仍包含未解析的环境变量；请先设置 {hint}，"
            "或把该字段直接改成绝对路径。"
        )
    return expanded


def body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise ValueError(f"G1 MJCF 缺少 body: {name}")
    return int(result)


def joint_ids(model: mujoco.MjModel, names: Iterable[str]) -> list[int]:
    result: list[int] = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"G1 MJCF 缺少 joint: {name}")
        result.append(int(jid))
    return result


def arrow_matrix(column: Any, width: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert Arrow list/fixed-list arrays without relying on pandas."""

    values = column.to_pylist()
    if not values:
        return np.empty((0, width), dtype=dtype)
    return np.asarray(values, dtype=dtype).reshape(-1, width)


def arrow_scalar(column: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def dataset_columns(dataset: str) -> list[str]:
    if dataset == "G1_WBT":
        return [
            "observation.state.left_arm",
            "observation.state.right_arm",
            "observation.state.left_leg",
            "observation.state.right_leg",
            "observation.state.waist_state_joint",
            "observation.state.state_base_pose",
            "action.left_arm",
            "action.right_arm",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]
    return ["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"]


def extract_batch(dataset: str, batch: Any) -> dict[str, np.ndarray | None]:
    names = set(batch.schema.names)

    def matrix(name: str, width: int) -> np.ndarray | None:
        return arrow_matrix(batch.column(name), width) if name in names else None

    if dataset == "G1_WBT":
        left_state = matrix("observation.state.left_arm", 7)
        right_state = matrix("observation.state.right_arm", 7)
        left_action = matrix("action.left_arm", 7)
        right_action = matrix("action.right_arm", 7)
        if any(x is None for x in (left_state, right_state, left_action, right_action)):
            raise ValueError("G1_WBT 缺少双臂字段")
        state = np.concatenate([left_state, right_state], axis=1)
        action = np.concatenate([left_action, right_action], axis=1)
        return {
            "state": state,
            "action": action,
            "left_leg": matrix("observation.state.left_leg", 6),
            "right_leg": matrix("observation.state.right_leg", 6),
            "waist": matrix("observation.state.waist_state_joint", 3),
            "base_pose": matrix("observation.state.state_base_pose", 7),
            "timestamp": arrow_scalar(batch.column("timestamp"), np.float32),
            "frame_index": arrow_scalar(batch.column("frame_index"), np.int64),
            "episode_index": arrow_scalar(batch.column("episode_index"), np.int64),
            "index": arrow_scalar(batch.column("index"), np.int64),
            "task_index": arrow_scalar(batch.column("task_index"), np.int64),
        }

    state_raw = matrix("observation.state", 28 if dataset == "G1_Dex3" else 16)
    action_raw = matrix("action", 28 if dataset == "G1_Dex3" else 16)
    if state_raw is None or action_raw is None:
        raise ValueError(f"{dataset} 缺少 observation.state/action")
    return {
        "state": state_raw[:, :14],
        "action": action_raw[:, :14],
        "left_leg": None,
        "right_leg": None,
        "waist": None,
        "base_pose": None,
        "timestamp": arrow_scalar(batch.column("timestamp"), np.float32),
        "frame_index": arrow_scalar(batch.column("frame_index"), np.int64),
        "episode_index": arrow_scalar(batch.column("episode_index"), np.int64),
        "index": arrow_scalar(batch.column("index"), np.int64),
        "task_index": arrow_scalar(batch.column("task_index"), np.int64),
    }


def convex_hull(points: list[np.ndarray]) -> list[np.ndarray]:
    if len(points) <= 1:
        return points
    unique = sorted({(round(float(p[0]), 8), round(float(p[1]), 8)) for p in points})
    if len(unique) <= 2:
        return [np.asarray(p, dtype=np.float32) for p in unique]

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    polygon = lower[:-1] + upper[:-1]
    return [np.asarray(p, dtype=np.float32) for p in polygon]


def point_in_polygon(point: np.ndarray, polygon: list[np.ndarray], margin: float = 1e-6) -> bool:
    if len(polygon) < 3:
        return False
    signs: list[float] = []
    for index, current in enumerate(polygon):
        nxt = polygon[(index + 1) % len(polygon)]
        edge = nxt - current
        rel = point - current
        signs.append(float(edge[0] * rel[1] - edge[1] * rel[0]))
    return all(value >= -margin for value in signs) or all(value <= margin for value in signs)


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(x) for x in quaternion_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def profile_for(dataset_id: int, episode: int, config: dict[str, Any], profile_id: int) -> Profile:
    aug = config["augmentation"]
    seed = int(aug["base_seed"]) + dataset_id * 10_000_000 + int(episode)
    rng = np.random.default_rng(seed)
    force_probability = float(aug["external_force_probability"])
    force_mag = float(rng.uniform(*aug["external_force_newton"])) if rng.random() < force_probability else 0.0
    force_direction = rng.normal(size=3)
    force_direction /= max(np.linalg.norm(force_direction), 1e-9)
    force = tuple((force_direction * force_mag).astype(np.float64).tolist())
    duration = int(rng.integers(aug["external_force_duration_frames"][0], aug["external_force_duration_frames"][1] + 1))
    start = int(rng.integers(0, 90)) if force_mag > 0 else -1
    raw = {
        "seed": seed,
        "mass_scale": round(float(rng.uniform(*aug["mass_scale"])), 8),
        "friction_scale": round(float(rng.uniform(*aug["friction_scale"])), 8),
        "delay_steps": int(rng.integers(aug["delay_steps"][0], aug["delay_steps"][1] + 1)),
        "force": [round(float(v), 8) for v in force],
        "force_start": start,
        "force_duration": duration,
        "terrain_slope_rad": round(float(rng.uniform(*aug["terrain_slope_rad"])), 8),
        "sensor_occlusion": bool(rng.random() < float(aug["sensor_occlusion_probability"])),
    }
    profile_hash = sha256_bytes(canonical_json(raw).encode("utf-8"))
    return Profile(profile_id=profile_id, profile_hash=profile_hash, **raw)


class SemanticExtractor:
    """MuJoCo-only physical semantic calculator; file I/O stays outside this class."""

    def __init__(self, xml_path: Path, keyframe: str, dt: float, max_contacts: int) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.dt = float(dt)
        self.max_contacts = int(max_contacts)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"MuJoCo keyframe 不存在: {keyframe}")
        self.key_id = int(key_id)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

        self.arm_joints = joint_ids(self.model, ARM_JOINT_NAMES)
        self.arm_bodies = [body_id(self.model, name) for name in ARM_BODY_NAMES]
        self.left_leg_joints = joint_ids(self.model, LEFT_LEG_NAMES)
        self.right_leg_joints = joint_ids(self.model, RIGHT_LEG_NAMES)
        self.waist_joints = joint_ids(self.model, WAIST_NAMES)
        self.torso_body = body_id(self.model, "torso_link")
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_geom < 0:
            self.floor_geom = 0

        self.arm_qpos = np.asarray([self.model.jnt_qposadr[jid] for jid in self.arm_joints], dtype=np.int32)
        self.arm_dof = np.asarray([self.model.jnt_dofadr[jid] for jid in self.arm_joints], dtype=np.int32)
        self.arm_limits = np.asarray([self.model.jnt_range[jid] for jid in self.arm_joints], dtype=np.float32)
        self.arm_parent_bodies = np.asarray([self.model.jnt_bodyid[jid] for jid in self.arm_joints], dtype=np.int32)
        self.arm_joint_types = np.asarray([self.model.jnt_type[jid] for jid in self.arm_joints], dtype=np.int8)
        self.base_qpos = self.model.key_qpos[self.key_id].copy()

        base_body_pos = self.data.xpos[self.arm_bodies].copy()
        torso_pos = self.data.xpos[self.torso_body].copy()
        torso_rot = self.data.xmat[self.torso_body].reshape(3, 3).copy()
        local = (torso_rot.T @ (base_body_pos - torso_pos).T).T
        margin = np.asarray([0.35, 0.35, 0.35], dtype=np.float64)
        self.nocs_min = local.min(axis=0) - margin
        self.nocs_max = local.max(axis=0) + margin
        self.nocs_extent = np.maximum(self.nocs_max - self.nocs_min, 1e-6)

        self.base_body_mass = self.model.body_mass.copy()
        self.base_body_inertia = self.model.body_inertia.copy()
        self.base_geom_friction = self.model.geom_friction.copy()
        self.active_profile: Profile | None = None

    def apply_profile(self, profile: Profile) -> None:
        if self.active_profile == profile:
            return
        self.model.body_mass[:] = self.base_body_mass * profile.mass_scale
        self.model.body_inertia[:] = self.base_body_inertia * profile.mass_scale
        self.model.geom_friction[:] = self.base_geom_friction
        self.model.geom_friction[:, 0] *= profile.friction_scale
        mujoco.mj_setConst(self.model, self.data)
        self.active_profile = profile

    def _make_qpos(
        self,
        arm_state: np.ndarray,
        left_leg: np.ndarray | None,
        right_leg: np.ndarray | None,
        waist: np.ndarray | None,
        base_pose: np.ndarray | None,
    ) -> np.ndarray:
        qpos = self.base_qpos.copy()
        if base_pose is not None and len(base_pose) >= 7 and np.all(np.isfinite(base_pose)):
            qpos[:3] = base_pose[:3]
            # Dataset uses (qx, qy, qz, qw), MuJoCo free joint uses (qw, qx, qy, qz).
            qpos[3:7] = [base_pose[6], base_pose[3], base_pose[4], base_pose[5]]
        qpos[self.arm_qpos] = np.clip(arm_state, self.arm_limits[:, 0], self.arm_limits[:, 1])
        if left_leg is not None:
            for jid, value in zip(self.left_leg_joints, left_leg):
                qpos[self.model.jnt_qposadr[jid]] = np.clip(value, *self.model.jnt_range[jid])
        if right_leg is not None:
            for jid, value in zip(self.right_leg_joints, right_leg):
                qpos[self.model.jnt_qposadr[jid]] = np.clip(value, *self.model.jnt_range[jid])
        if waist is not None:
            for jid, value in zip(self.waist_joints, waist):
                qpos[self.model.jnt_qposadr[jid]] = np.clip(value, *self.model.jnt_range[jid])
        return qpos

    def extract(
        self,
        arm_state: np.ndarray,
        joint_velocity: np.ndarray,
        left_leg: np.ndarray | None,
        right_leg: np.ndarray | None,
        waist: np.ndarray | None,
        base_pose: np.ndarray | None,
        profile: Profile,
        frame_in_episode: int,
    ) -> dict[str, np.ndarray | int | float | bool]:
        self.apply_profile(profile)
        qpos = self._make_qpos(arm_state, left_leg, right_leg, waist, base_pose)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.qvel[self.arm_dof] = joint_velocity
        self.data.xfrc_applied[:] = 0.0
        if profile.force_start >= 0 and profile.force_start <= frame_in_episode < profile.force_start + profile.force_duration:
            self.data.xfrc_applied[self.torso_body, :3] = np.asarray(profile.force, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        torso_pos = self.data.xpos[self.torso_body].copy()
        torso_rot = self.data.xmat[self.torso_body].reshape(3, 3).copy()
        body_pos = self.data.xpos[self.arm_bodies].copy()
        body_local = (torso_rot.T @ (body_pos - torso_pos).T).T
        part_nocs = np.clip((body_local - self.nocs_min) / self.nocs_extent, 0.0, 1.0).astype(np.float32)

        parent_offset = np.zeros((len(self.arm_bodies), 3), dtype=np.float32)
        joint_axis_torso = np.zeros((len(self.arm_bodies), 3), dtype=np.float32)
        for index, (jid, body) in enumerate(zip(self.arm_joints, self.arm_bodies)):
            parent = int(self.model.body_parentid[body])
            parent_rot = self.data.xmat[parent].reshape(3, 3)
            parent_offset[index] = (parent_rot.T @ (self.data.xpos[body] - self.data.xpos[parent])).astype(np.float32)
            axis_world = parent_rot @ self.model.jnt_axis[jid]
            joint_axis_torso[index] = (torso_rot.T @ axis_world).astype(np.float32)

        joint_angle = self.data.qpos[self.arm_qpos].astype(np.float32).copy()
        limit_fraction = np.clip(
            (joint_angle - self.arm_limits[:, 0]) / np.maximum(self.arm_limits[:, 1] - self.arm_limits[:, 0], 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)

        contact_points = np.zeros((self.max_contacts, 3), dtype=np.float32)
        contact_normals = np.zeros((self.max_contacts, 3), dtype=np.float32)
        contact_impulses = np.zeros((self.max_contacts,), dtype=np.float32)
        contact_mask = np.zeros((self.max_contacts,), dtype=np.uint8)
        contact_body_ids = np.full((self.max_contacts, 2), -1, dtype=np.int32)
        support_points: list[np.ndarray] = []
        for index in range(min(int(self.data.ncon), self.max_contacts)):
            contact = self.data.contact[index]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, index, force)
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            if int(contact.geom2) == self.floor_geom:
                normal = -normal
            contact_points[index] = contact.pos
            contact_normals[index] = normal
            contact_impulses[index] = max(float(force[0]), 0.0) * self.dt
            contact_mask[index] = 1
            contact_body_ids[index] = [self.model.geom_bodyid[contact.geom1], self.model.geom_bodyid[contact.geom2]]
            if (contact.geom1 == self.floor_geom or contact.geom2 == self.floor_geom) and normal[2] > 0.2:
                support_points.append(np.asarray(contact.pos[:2], dtype=np.float32))

        support_hull = convex_hull(support_points)
        support_polygon = np.zeros((self.max_contacts, 2), dtype=np.float32)
        support_count = min(len(support_hull), self.max_contacts)
        if support_count:
            support_polygon[:support_count] = np.asarray(support_hull[:support_count], dtype=np.float32)

        com = np.asarray(self.data.subtree_com[0], dtype=np.float32).copy()
        com_velocity = np.asarray(self.data.subtree_linvel[0], dtype=np.float32).copy()
        normal_sum = float(sum(contact_impulses[: int(self.data.ncon)]))
        if normal_sum > 1e-8 and support_points:
            weights = contact_impulses[: min(int(self.data.ncon), self.max_contacts)]
            positions = contact_points[: min(int(self.data.ncon), self.max_contacts)]
            # Use all floor contacts; non-floor contacts have zero contribution in the normal direction.
            floor_mask = np.asarray(
                [
                    1.0
                    if (
                        contact_body_ids[i, 0] == self.model.geom_bodyid[self.floor_geom]
                        or contact_body_ids[i, 1] == self.model.geom_bodyid[self.floor_geom]
                    )
                    else 0.0
                    for i in range(min(int(self.data.ncon), self.max_contacts))
                ],
                dtype=np.float32,
            )
            weighted = weights * floor_mask
            zmp = (positions * weighted[:, None]).sum(axis=0) / max(float(weighted.sum()), 1e-8)
            zmp = np.asarray([zmp[0], zmp[1], 0.0], dtype=np.float32)
            zmp_valid = 1
        else:
            zmp = np.asarray([com[0], com[1], 0.0], dtype=np.float32)
            zmp_valid = 0

        stable = bool(support_count >= 3 and point_in_polygon(com[:2], support_hull) and com[2] > 0.45)
        return {
            "joint_angle": joint_angle,
            "joint_velocity": np.asarray(joint_velocity, dtype=np.float32),
            "joint_limit_fraction": limit_fraction,
            "qpos": self.data.qpos.astype(np.float32).copy(),
            "qvel": self.data.qvel.astype(np.float32).copy(),
            "imu": self.data.sensordata.astype(np.float32).copy(),
            "part_world_pos": body_pos.astype(np.float32),
            "part_nocs": part_nocs,
            "part_parent_offset": parent_offset,
            "joint_axis_torso": joint_axis_torso,
            "contact_points": contact_points,
            "contact_normals": contact_normals,
            "contact_impulses": contact_impulses,
            "contact_mask": contact_mask,
            "contact_body_ids": contact_body_ids,
            "support_polygon": support_polygon,
            "support_count": np.int16(support_count),
            "com": com,
            "com_velocity": com_velocity,
            "zmp": zmp,
            "zmp_valid": np.uint8(zmp_valid),
            "stability": np.uint8(stable),
            "qpos_arm": self.data.qpos[self.arm_qpos].astype(np.float32).copy(),
        }


def profile_to_json(profile: Profile) -> dict[str, Any]:
    result = asdict(profile)
    result["force"] = list(result["force"])
    return result


def empty_chunk(max_contacts: int) -> dict[str, list[np.ndarray]]:
    return {
        "observation_state": [],
        "augmented_state": [],
        "action": [],
        "augmented_action": [],
        "next_observation_state": [],
        "next_action": [],
        "joint_angle": [],
        "joint_velocity": [],
        "joint_limit_fraction": [],
        "qpos": [],
        "qvel": [],
        "imu": [],
        "part_world_pos": [],
        "part_nocs": [],
        "part_parent_offset": [],
        "joint_axis_torso": [],
        "contact_points": [],
        "contact_normals": [],
        "contact_impulses": [],
        "contact_mask": [],
        "contact_body_ids": [],
        "support_polygon": [],
        "support_count": [],
        "com": [],
        "com_velocity": [],
        "zmp": [],
        "zmp_valid": [],
        "external_force": [],
        "mass_scale": [],
        "friction_scale": [],
        "delay_steps": [],
        "sensor_occlusion": [],
        "stability": [],
        "future_stability": [],
        "reward": [],
        "termination_reason": [],
        "validity_flags": [],
        "dataset_id": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "task_index": [],
        "profile_id": [],
    }


def append_value(store: dict[str, list[np.ndarray]], key: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        store[key].append(value.copy())
    else:
        store[key].append(np.asarray(value))


def stack_store(store: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, values in store.items():
        if not values:
            result[key] = np.empty((0,), dtype=np.float32)
        else:
            result[key] = np.stack(values, axis=0)
    return result


def frame_reward(stability: np.ndarray, com: np.ndarray, zmp: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(com[:, :2] - zmp[:, :2], axis=1)
    speed_penalty = np.linalg.norm(velocity, axis=1)
    return (stability.astype(np.float32) - 0.5 * distance - 0.002 * speed_penalty).astype(np.float32)


def set_next_fields(arrays: dict[str, np.ndarray], episode: np.ndarray) -> None:
    for key in ("observation_state", "action"):
        source = arrays[key]
        target = np.empty_like(source)
        target[:-1] = source[1:]
        target[-1] = source[-1]
        for index in range(len(source) - 1):
            if episode[index + 1] != episode[index]:
                target[index] = source[index]
        arrays[f"next_{key}"] = target


def process_dataset(
    dataset_name: str,
    dataset_id: int,
    config: dict[str, Any],
    extractor: SemanticExtractor,
    output_root: Path,
    max_frames: int | None,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise SystemExit(
            "pyarrow is required to read G1_Action Parquet files. "
            "Install it into the active environment, e.g. `pip install pyarrow`."
        ) from exc
    source_root = Path(config["data"]["root"]) / dataset_name
    dataset_output = output_root / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)
    files = sorted((source_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有 Parquet 数据: {source_root / 'data'}")

    max_contacts = int(config["data"]["max_contacts"])
    store = empty_chunk(max_contacts)
    profiles: dict[int, Profile] = {}
    episode_profiles: dict[int, Profile] = {}
    profile_index = 0
    current_episode: int | None = None
    last_state: np.ndarray | None = None
    histories: dict[int, list[np.ndarray]] = {}
    frame_in_episode = 0
    total_read = 0
    dt = float(config["robot"]["dt"])
    noise_std = float(config["augmentation"]["state_noise_std_rad"])
    action_noise_std = float(config["augmentation"]["action_noise_std_rad"])

    for parquet_path in files:
        parquet = pq.ParquetFile(parquet_path)
        columns = [name for name in dataset_columns(dataset_name) if name in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=4096, columns=columns, use_threads=True):
            parsed = extract_batch(dataset_name, batch)
            count = len(parsed["state"])
            for row in range(count):
                if max_frames is not None and total_read >= max_frames:
                    break
                episode = int(parsed["episode_index"][row])
                if current_episode != episode:
                    current_episode = episode
                    frame_in_episode = 0
                    last_state = None
                    history = []
                    histories[episode] = history
                    profile = profile_for(dataset_id, episode, config, profile_index)
                    profiles[profile.profile_id] = profile
                    episode_profiles[episode] = profile
                    profile_index += 1
                else:
                    profile = episode_profiles[episode]
                    history = histories[episode]

                raw_state = np.asarray(parsed["state"][row], dtype=np.float32)
                raw_action = np.asarray(parsed["action"][row], dtype=np.float32)
                velocity = np.zeros(14, dtype=np.float32) if last_state is None else (raw_state - last_state) / dt
                delayed_state = history[-profile.delay_steps - 1] if len(history) > profile.delay_steps else raw_state
                rng = np.random.default_rng(profile.seed + frame_in_episode)
                augmented_state = delayed_state + rng.normal(0.0, noise_std, size=14).astype(np.float32)
                augmented_action = raw_action + rng.normal(0.0, action_noise_std, size=14).astype(np.float32)
                semantics = extractor.extract(
                    arm_state=augmented_state,
                    joint_velocity=velocity,
                    left_leg=None if parsed["left_leg"] is None else parsed["left_leg"][row],
                    right_leg=None if parsed["right_leg"] is None else parsed["right_leg"][row],
                    waist=None if parsed["waist"] is None else parsed["waist"][row],
                    base_pose=None if parsed["base_pose"] is None else parsed["base_pose"][row],
                    profile=profile,
                    frame_in_episode=frame_in_episode,
                )
                append_value(store, "observation_state", raw_state)
                append_value(store, "augmented_state", augmented_state)
                append_value(store, "action", raw_action)
                append_value(store, "augmented_action", augmented_action)
                for key in (
                    "joint_angle",
                    "joint_velocity",
                    "joint_limit_fraction",
                    "qpos",
                    "qvel",
                    "imu",
                    "part_world_pos",
                    "part_nocs",
                    "part_parent_offset",
                    "joint_axis_torso",
                    "contact_points",
                    "contact_normals",
                    "contact_impulses",
                    "contact_mask",
                    "contact_body_ids",
                    "support_polygon",
                    "support_count",
                    "com",
                    "com_velocity",
                    "zmp",
                    "zmp_valid",
                ):
                    append_value(store, key, semantics[key])
                force_active = profile.force_start >= 0 and profile.force_start <= frame_in_episode < profile.force_start + profile.force_duration
                append_value(store, "external_force", np.asarray(profile.force if force_active else (0.0, 0.0, 0.0), dtype=np.float32))
                append_value(store, "mass_scale", profile.mass_scale)
                append_value(store, "friction_scale", profile.friction_scale)
                append_value(store, "delay_steps", profile.delay_steps)
                append_value(store, "sensor_occlusion", int(profile.sensor_occlusion))
                append_value(store, "stability", semantics["stability"])
                append_value(store, "dataset_id", dataset_id)
                append_value(store, "episode_index", episode)
                append_value(store, "frame_index", int(parsed["frame_index"][row]))
                append_value(store, "timestamp", float(parsed["timestamp"][row]))
                append_value(store, "task_index", int(parsed["task_index"][row]))
                append_value(store, "profile_id", profile.profile_id)
                append_value(store, "termination_reason", 0)
                append_value(store, "validity_flags", 0b111 if np.all(np.isfinite(semantics["com"])) else 0)
                history.append(raw_state.copy())
                if len(history) > 8:
                    del history[:-8]
                last_state = raw_state
                frame_in_episode += 1
                total_read += 1
            if max_frames is not None and total_read >= max_frames:
                break
        if max_frames is not None and total_read >= max_frames:
            break

    arrays = stack_store(store)
    if not arrays["episode_index"].size:
        raise RuntimeError(f"{dataset_name} 没有成功读取任何帧")
    set_next_fields(arrays, arrays["episode_index"])
    stable = arrays["stability"].astype(np.uint8)
    future = np.zeros_like(stable)
    horizon = int(config["data"]["future_horizon"])
    for start in range(len(stable)):
        end = min(len(stable), start + horizon + 1)
        same_episode = arrays["episode_index"][start:end] == arrays["episode_index"][start]
        future[start] = np.uint8(np.all(stable[start:end][same_episode]))
    arrays["future_stability"] = future
    arrays["reward"] = frame_reward(stable, arrays["com"], arrays["zmp"], arrays["joint_velocity"])
    last_of_episode = np.r_[arrays["episode_index"][1:] != arrays["episode_index"][:-1], True]
    arrays["termination_reason"] = np.where(last_of_episode, 1, np.where(stable == 0, 2, 0)).astype(np.int8)

    # Make scalar arrays compact and all numeric arrays explicit little-endian dtypes.
    for key, value in arrays.items():
        if value.dtype.kind == "f":
            arrays[key] = np.asarray(value, dtype="<f4")
        elif value.dtype.kind in "iu":
            arrays[key] = np.asarray(value)

    npz_path = dataset_output / "samples.npz"
    tmp_path = dataset_output / "samples.npz.tmp"
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(npz_path)

    info_path = source_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    source_file_records = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files]
    profile_records = [profile_to_json(profiles[key]) for key in sorted(profiles)]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "robot": {
            "name": "Unitree_G1",
            "mjcf": str(config["robot"]["xml"]),
            "mujoco_version": mujoco.__version__,
            "model_sha256": sha256_file(Path(config["robot"]["xml"])),
            "coordinate_convention": "world/base/body; quaternion source xyzw converted to MuJoCo wxyz",
        },
        "source": {
            "root": str(source_root),
            "info": info,
            "episodes": int(np.unique(arrays["episode_index"]).size),
            "frames": int(len(arrays["episode_index"])),
            "source_files": source_file_records,
            "videos_read": False,
        },
        "generation": {
            "seed": int(config["augmentation"]["base_seed"]),
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git_commit(Path.cwd()),
            "max_frames_per_dataset": max_frames,
            "cpu_gpu_split": "CPU: Parquet/MuJoCo/NPZ; GPU: training only",
        },
        "arrays": {
            key: {"dtype": str(value.dtype), "shape": list(value.shape), "units": units_for(key)}
            for key, value in arrays.items()
        },
        "semantic_constants": {
            "arm_joint_names": ARM_JOINT_NAMES,
            "arm_body_names": ARM_BODY_NAMES,
            "joint_types": {"0": "free", "1": "ball", "2": "slide", "3": "hinge"},
            "arm_joint_type": extractor.arm_joint_types.tolist(),
            "arm_joint_limits_rad": extractor.arm_limits.tolist(),
            "nocs_min_m": extractor.nocs_min.tolist(),
            "nocs_max_m": extractor.nocs_max.tolist(),
            "nocs_definition": "part body origins relative to torso mapped to [0,1]^3; ANCSH-inspired hierarchy, not category-level object pose",
        },
        "profiles": profile_records,
        "checksums": {"samples.npz": sha256_file(npz_path)},
    }
    manifest_path = dataset_output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset": dataset_name, "frames": len(arrays["episode_index"]), "episodes": manifest["source"]["episodes"], "manifest": str(manifest_path)}


def units_for(key: str) -> str:
    if key in {"observation_state", "augmented_state", "action", "augmented_action", "next_observation_state", "next_action", "joint_angle", "joint_velocity", "joint_limit_fraction", "qpos", "qvel", "imu", "reward"}:
        return "rad/rad_s/unitless (see field)"
    if key in {"part_nocs", "contact_mask", "zmp_valid", "stability", "future_stability", "sensor_occlusion", "validity_flags", "dataset_id", "episode_index", "frame_index", "task_index", "profile_id", "termination_reason", "support_count", "delay_steps"}:
        return "unitless"
    if key in {"part_parent_offset", "part_world_pos", "contact_points", "support_polygon", "com", "com_velocity", "zmp", "external_force"}:
        return "m / m_s / N (see field)"
    if key == "contact_normals" or key == "joint_axis_torso":
        return "unit vector"
    if key == "contact_impulses":
        return "N_s"
    if key in {"mass_scale", "friction_scale"}:
        return "scale"
    if key == "timestamp":
        return "s"
    return "see docs/design/data_schema.md"


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "not-a-git-repository"
    except OSError:
        return "git-unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/g1_physical_semantics.yaml"))
    parser.add_argument("--dataset", action="append", dest="datasets", help="只生成指定数据集，可重复")
    parser.add_argument("--max-frames-per-dataset", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="允许覆盖该数据集已有 samples.npz/manifest")
    parser.add_argument("--limit-datasets", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    config["_config_path"] = str(args.config.resolve())
    config["robot"]["xml"] = resolve_env_path(config["robot"]["xml"], "G1_MJCF")
    config["data"]["root"] = resolve_env_path(config["data"]["root"], "G1_ACTION_ROOT")
    output_root = (args.output or Path(config["data"]["output"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or list(config["data"]["datasets"])
    if args.limit_datasets is not None:
        datasets = datasets[: args.limit_datasets]
    unknown = sorted(set(datasets) - set(config["data"]["datasets"]))
    if unknown:
        raise SystemExit(f"配置未声明的数据集: {unknown}")
    xml_path = Path(config["robot"]["xml"])
    if not xml_path.exists():
        raise SystemExit(f"G1 MJCF 不存在: {xml_path}")
    extractor = SemanticExtractor(xml_path, config["robot"]["keyframe"], config["robot"]["dt"], config["data"]["max_contacts"])
    summaries = []
    for dataset_id, dataset in enumerate(datasets):
        target = output_root / dataset
        if not args.force and (target / "samples.npz").exists():
            raise SystemExit(f"输出已存在：{target}；如需覆盖请显式传 --force")
        max_frames = args.max_frames_per_dataset
        if max_frames is None:
            max_frames = config["data"].get("max_frames_per_dataset")
        summary = process_dataset(dataset, dataset_id, config, extractor, output_root, max_frames)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    combined = {
        "schema_version": SCHEMA_VERSION,
        "robot": "Unitree_G1",
        "datasets": summaries,
        "source_root": config["data"]["root"],
        "config_sha256": sha256_file(args.config),
    }
    (output_root / "index.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_root), "datasets": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

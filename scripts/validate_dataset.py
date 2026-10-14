#!/usr/bin/env python3
"""Read-only schema, checksum and physical-invariant validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED = {
    "observation_state",
    "augmented_state",
    "action",
    "augmented_action",
    "next_observation_state",
    "next_action",
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
    "external_force",
    "mass_scale",
    "friction_scale",
    "delay_steps",
    "sensor_occlusion",
    "stability",
    "future_stability",
    "reward",
    "termination_reason",
    "validity_flags",
    "dataset_id",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "profile_id",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fail(code: int, message: str) -> None:
    print(json.dumps({"status": "failed", "code": code, "error": message}, ensure_ascii=False, indent=2))
    raise SystemExit(code)


def validate_dataset(dataset_root: Path, check_source_hash: bool) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    npz_path = dataset_root / "samples.npz"
    if not manifest_path.exists() or not npz_path.exists():
        fail(2, f"缺少 manifest 或 samples.npz: {dataset_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0.0":
        fail(2, f"不支持的 schema_version: {manifest.get('schema_version')}")
    expected_checksum = manifest.get("checksums", {}).get("samples.npz")
    actual_checksum = sha256_file(npz_path)
    if expected_checksum != actual_checksum:
        fail(3, f"NPZ checksum 不匹配: {dataset_root}")
    if check_source_hash:
        for source in manifest.get("source", {}).get("source_files", []):
            path = Path(source["path"])
            if not path.exists() or sha256_file(path) != source["sha256"]:
                fail(3, f"源 Parquet checksum 不匹配: {path}")

    with np.load(npz_path, allow_pickle=False) as loaded:
        names = set(loaded.files)
        missing = sorted(REQUIRED - names)
        if missing:
            fail(2, f"缺少数组字段: {missing}")
        arrays = {name: loaded[name] for name in loaded.files}
    frames = len(arrays["episode_index"])
    if frames == 0:
        fail(4, f"空数据集: {dataset_root}")
    for name, value in arrays.items():
        if value.shape[0] != frames:
            fail(4, f"首维长度不一致: {name}={value.shape} vs {frames}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            fail(4, f"浮点数组含 NaN/Inf: {name}")
    if not np.logical_and(arrays["part_nocs"] >= -1e-6, arrays["part_nocs"] <= 1.0 + 1e-6).all():
        fail(4, "part_nocs 超出 [0,1]")
    if not np.isin(arrays["contact_mask"], [0, 1]).all():
        fail(4, "contact_mask 非二值")
    if np.any(arrays["contact_impulses"] < -1e-6):
        fail(4, "contact_impulses 为负")
    if np.any(arrays["support_count"] < 0) or np.any(arrays["support_count"] > arrays["support_polygon"].shape[1]):
        fail(4, "support_count 越界")
    if not np.isin(arrays["stability"], [0, 1]).all() or not np.isin(arrays["future_stability"], [0, 1]).all():
        fail(4, "稳定性标签非二值")

    episode = arrays["episode_index"]
    boundary = np.r_[episode[1:] != episode[:-1], True]
    if np.any(arrays["termination_reason"][boundary] == 0):
        fail(4, "episode 末帧 termination_reason 为 0")
    same = ~boundary
    if np.any(same):
        # Explicit loop keeps the boundary index in any diagnostic.
        for index in np.flatnonzero(same):
            if not np.allclose(arrays["next_observation_state"][index], arrays["observation_state"][index + 1], atol=1e-6):
                fail(4, f"next_observation_state 跨帧不一致: index={index}")
    for index in np.flatnonzero(boundary):
        if not np.allclose(arrays["next_observation_state"][index], arrays["observation_state"][index], atol=1e-6):
            fail(4, f"episode 边界 next_observation_state 未保持末帧: index={index}")

    manifest_frames = int(manifest.get("source", {}).get("frames", -1))
    if manifest_frames != frames:
        fail(4, f"manifest frames={manifest_frames} 与数组={frames} 不一致")
    stability_ratio = float(np.mean(arrays["stability"]))
    report = {
        "dataset": manifest.get("dataset"),
        "frames": frames,
        "episodes": int(np.unique(episode).size),
        "checksum": actual_checksum,
        "stability_ratio": stability_ratio,
        "future_stability_ratio": float(np.mean(arrays["future_stability"])),
        "support_count_mean": float(np.mean(arrays["support_count"])),
        "contact_impulse_max_Ns": float(np.max(arrays["contact_impulses"])),
        "warnings": ["stability label >99%" if stability_ratio > 0.99 else "stability label <1%" if stability_ratio < 0.01 else ""],
    }
    report["warnings"] = [warning for warning in report["warnings"] if warning]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/physical_semantics"))
    parser.add_argument("--check-source-hash", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.data.resolve()
    index_path = root / "index.json"
    if not index_path.exists():
        fail(2, f"缺少 index.json: {root}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    reports = [validate_dataset(root / entry["dataset"], args.check_source_hash) for entry in index.get("datasets", [])]
    result = {"status": "passed", "data": str(root), "datasets": reports}
    report_path = args.report or (root / "validation_report.json")
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

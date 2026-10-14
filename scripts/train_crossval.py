#!/usr/bin/env python3
"""GPU-only episode-group cross-validation for the generated G1 semantic data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_features(data: dict[str, np.ndarray]) -> np.ndarray:
    """Build a fixed numeric feature vector from physical semantic fields.

    The target is kept separate so that the model cannot read the current action
    directly.  All coordinate fields retain their documented units/scales; only
    the fold-local mean/std normalization is learned from training episodes.
    """

    parts = [
        data["augmented_state"],
        data["joint_velocity"],
        data["joint_limit_fraction"],
        data["qpos"],
        data["qvel"],
        data["imu"],
        data["part_world_pos"].reshape(len(data["part_world_pos"]), -1),
        data["part_nocs"].reshape(len(data["part_nocs"]), -1),
        data["part_parent_offset"].reshape(len(data["part_parent_offset"]), -1),
        data["joint_axis_torso"].reshape(len(data["joint_axis_torso"]), -1),
        data["contact_points"].reshape(len(data["contact_points"]), -1),
        data["contact_normals"].reshape(len(data["contact_normals"]), -1),
        data["contact_impulses"],
        data["contact_mask"].astype(np.float32),
        data["support_polygon"].reshape(len(data["support_polygon"]), -1),
        data["support_count"].astype(np.float32)[:, None],
        data["com"],
        data["com_velocity"],
        data["zmp"],
        data["zmp_valid"].astype(np.float32)[:, None],
        data["external_force"],
        data["mass_scale"][:, None],
        data["friction_scale"][:, None],
        data["delay_steps"].astype(np.float32)[:, None],
        data["sensor_occlusion"].astype(np.float32)[:, None],
        data["stability"].astype(np.float32)[:, None],
        data["future_stability"].astype(np.float32)[:, None],
    ]
    features = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("特征含 NaN/Inf")
    return features


def load_generated(root: Path, verify_checksums: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    manifests: list[dict[str, Any]] = []
    for record in index["datasets"]:
        dataset_root = root / record["dataset"]
        manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
        npz_path = dataset_root / "samples.npz"
        if verify_checksums:
            expected = manifest["checksums"]["samples.npz"]
            actual = sha256_file(npz_path)
            if actual != expected:
                raise ValueError(f"checksum 不匹配: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            data = {name: loaded[name] for name in loaded.files}
        x = build_features(data)
        y = data["next_action"].astype(np.float32, copy=False)
        group = data["dataset_id"].astype(np.int64) * 1_000_000 + data["episode_index"].astype(np.int64)
        if len(x) != len(y) or len(x) != len(group):
            raise ValueError(f"样本长度不一致: {dataset_root}")
        feature_blocks.append(x)
        target_blocks.append(y)
        group_blocks.append(group)
        manifests.append(manifest)
        print(f"loaded {record['dataset']}: frames={len(x)} feature_dim={x.shape[1]} groups={len(np.unique(group))}", flush=True)
    if not feature_blocks:
        raise ValueError(f"没有生成数据: {root}")
    return np.concatenate(feature_blocks), np.concatenate(target_blocks), np.concatenate(group_blocks), manifests


class ActionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 14) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred - target
    mse = float(np.mean(error * error))
    mae = float(np.mean(np.abs(error)))
    centered = target - target.mean(axis=0, keepdims=True)
    denominator = float(np.sum(centered * centered))
    r2 = float(1.0 - np.sum(error * error) / denominator) if denominator > 1e-12 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def make_folds(groups: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    assignments = {str(int(group)): int(index % folds) for index, group in enumerate(shuffled)}
    fold_ids = np.asarray([assignments[str(int(group))] for group in groups], dtype=np.int64)
    return fold_ids, assignments


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                normalized = torch.clamp((xb - mean) / std, -8.0, 8.0)
                pred = model(normalized)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("GPU inference produced NaN/Inf")
            outputs.append(pred.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_fold(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config: dict[str, Any],
    fold: int,
    output_root: Path,
) -> dict[str, Any]:
    training = config["training"]
    input_dim = x.shape[1]
    hidden_dim = int(training["hidden_dim"])
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    seed = int(training["seed"]) + fold
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    max_train = training.get("max_train_samples_per_fold")
    if max_train is not None and len(train_indices) > int(max_train):
        rng = np.random.default_rng(seed)
        train_indices = np.sort(rng.choice(train_indices, int(max_train), replace=False))

    x_train = torch.from_numpy(x[train_indices])
    y_train = torch.from_numpy(y[train_indices])
    x_test = torch.from_numpy(x[test_indices])
    y_test = torch.from_numpy(y[test_indices])
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": min(4, max(0, (os.cpu_count() or 2) // 2)),
        "pin_memory": True,
    }
    train_loader = DataLoader(TensorDataset(x_train, y_train), **loader_kwargs, persistent_workers=loader_kwargs["num_workers"] > 0)
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_kwargs["num_workers"],
        pin_memory=True,
        persistent_workers=loader_kwargs["num_workers"] > 0,
    )

    device = torch.device("cuda")
    model = ActionMLP(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    requested_amp_dtype = str(training.get("amp_dtype", "bfloat16")).lower()
    use_amp = bool(training.get("amp", True))
    if use_amp and requested_amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif use_amp and requested_amp_dtype == "float16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
        use_amp = False
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    train_mean = x_train.float().mean(dim=0).to(device, non_blocking=True)
    train_std = x_train.float().std(dim=0, unbiased=False).to(device, non_blocking=True).clamp_min(1e-6)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    loss_history: list[float] = []
    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                normalized = torch.clamp((xb - train_mean) / train_std, -8.0, 8.0)
                pred = model(normalized)
                loss = torch.nn.functional.mse_loss(pred, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"GPU training produced NaN/Inf at fold={fold}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(xb)
            total_count += len(xb)
        loss_history.append(total_loss / max(total_count, 1))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    pred = evaluate_model(model, test_loader, train_mean, train_std, device, amp_dtype if use_amp else torch.float32)
    target = y[test_indices]
    metrics = regression_metrics(pred, target)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"fold={fold} metrics contain NaN/Inf: {metrics}")
    baseline = regression_metrics(x[test_indices, :14], target)
    checkpoint = output_root / f"fold_{fold:02d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "mean": train_mean.detach().cpu(),
            "std": train_std.detach().cpu(),
            "seed": seed,
            "fold": fold,
        },
        checkpoint,
    )
    samples_per_second = float(len(train_indices) * epochs / max(elapsed, 1e-9))
    return {
        "fold": fold,
        "train_frames": int(len(train_indices)),
        "test_frames": int(len(test_indices)),
        "test_groups": int(np.unique(groups[test_mask]).size),
        "epochs": epochs,
        "loss_history": loss_history,
        "metrics": metrics,
        "persistence_baseline": baseline,
        "train_wall_seconds": elapsed,
        "train_samples_per_second": samples_per_second,
        "amp_dtype": str(amp_dtype),
        "checkpoint": str(checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/physical_semantics"))
    parser.add_argument("--config", type=Path, default=Path("configs/g1_physical_semantics.yaml"))
    parser.add_argument("--output", type=Path, default=Path("experiments/g1_action_cv"))
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="只用于 smoke test；按固定 seed 抽样")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GPU-only 训练要求 torch.cuda.is_available() == True")
    config = load_yaml(args.config)
    folds = int(args.folds or config["training"]["folds"])
    if folds < 2:
        raise SystemExit("folds 必须 >= 2")
    x, y, groups, manifests = load_generated(args.data.resolve(), args.verify_checksums)
    if args.max_samples is not None and len(x) > args.max_samples:
        rng = np.random.default_rng(int(config["training"]["seed"]))
        indices = np.sort(rng.choice(len(x), int(args.max_samples), replace=False))
        x, y, groups = x[indices], y[indices], groups[indices]
    fold_ids, assignments = make_folds(groups, folds, int(config["training"]["seed"]))
    if len(np.unique(fold_ids)) != folds:
        raise SystemExit("样本不足以生成要求的 fold 数")
    output_root = args.output.resolve()
    if output_root.exists() and not args.force and any(output_root.iterdir()):
        raise SystemExit(f"输出目录非空：{output_root}；如需覆盖请显式传 --force")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "fold_assignments.json").write_text(json.dumps(assignments, indent=2), encoding="utf-8")
    results: list[dict[str, Any]] = []
    for fold in range(folds):
        print(f"training fold {fold + 1}/{folds}", flush=True)
        result = train_fold(x, y, groups, fold_ids != fold, fold_ids == fold, config, fold, output_root)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    metric_names = ["mse", "mae", "r2"]
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "task": "predict next 14D arm action from augmented state + G1 physical semantics",
        "data_root": str(args.data.resolve()),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "frames": int(len(x)),
        "groups": int(len(np.unique(groups))),
        "folds": folds,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "feature_dim": int(x.shape[1]),
        "folds_result": results,
        "aggregate": {
            "model": {name: {"mean": float(np.mean([r["metrics"][name] for r in results])), "std": float(np.std([r["metrics"][name] for r in results], ddof=1))} for name in metric_names},
            "persistence_baseline": {name: {"mean": float(np.mean([r["persistence_baseline"][name] for r in results])), "std": float(np.std([r["persistence_baseline"][name] for r in results], ddof=1))} for name in metric_names},
        },
        "reproducibility": {
            "seed": int(config["training"]["seed"]),
            "command": " ".join(sys.argv),
            "deterministic_algorithms": False,
            "note": "cuDNN/AMP GPU kernel scheduling may produce small last-bit variation; fold groups and config are fixed.",
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

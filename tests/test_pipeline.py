"""Self-contained tests for the G1 physical-semantic pipeline.

The tests rely only on the 120-frame fixture dataset shipped under
``tests/fixtures/mini_dataset`` so they can run without the full G1_Action
download or any generated data directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_physical_semantics as generator  # noqa: E402

try:
    import train_crossval as trainer  # noqa: E402
except ImportError:  # pragma: no cover - torch is optional for CPU-only users
    trainer = None

FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "mini_dataset"


def load_fixture_arrays() -> dict[str, np.ndarray]:
    with np.load(FIXTURE_ROOT / "G1_Sample" / "samples.npz", allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


class GeometryTests(unittest.TestCase):
    def test_convex_hull_and_containment(self) -> None:
        points = [
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.5, 0.5]),
        ]
        hull = generator.convex_hull(points)
        self.assertEqual(len(hull), 4)
        self.assertTrue(generator.point_in_polygon(np.asarray([0.5, 0.5]), hull))
        self.assertFalse(generator.point_in_polygon(np.asarray([1.5, 0.5]), hull))


class FixtureDatasetTests(unittest.TestCase):
    def test_fixture_manifest_and_index(self) -> None:
        index = json.loads((FIXTURE_ROOT / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["schema_version"], "1.0.0")
        self.assertEqual([entry["dataset"] for entry in index["datasets"]], ["G1_Sample"])
        manifest = json.loads(
            (FIXTURE_ROOT / "G1_Sample" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["source"]["frames"], 120)
        self.assertEqual(manifest["source"]["episodes"], 1)

    def test_fixture_semantic_shapes(self) -> None:
        arrays = load_fixture_arrays()
        self.assertEqual(arrays["observation_state"].shape, (120, 14))
        self.assertEqual(arrays["part_nocs"].shape[1:], (14, 3))
        self.assertEqual(arrays["qpos"].shape[1:], (36,))
        self.assertEqual(arrays["qvel"].shape[1:], (35,))
        self.assertEqual(arrays["imu"].shape[1:], (12,))
        self.assertEqual(arrays["part_world_pos"].shape[1:], (14, 3))
        self.assertTrue(np.isfinite(arrays["com"]).all())
        self.assertTrue(np.isfinite(arrays["part_nocs"]).all())
        self.assertTrue(np.isin(arrays["stability"], [0, 1]).all())

    def test_validator_accepts_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = Path(tmp_dir) / "report.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_dataset.py"),
                    "--data",
                    str(FIXTURE_ROOT),
                    "--report",
                    str(report),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(payload["datasets"][0]["frames"], 120)


@unittest.skipIf(trainer is None, "PyTorch 未安装，跳过训练特征测试")
class TrainingFeatureTests(unittest.TestCase):
    def test_feature_builder_is_finite_and_fixed_width(self) -> None:
        arrays = load_fixture_arrays()
        features = trainer.build_features(arrays)
        self.assertEqual(features.shape, (120, 393))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()

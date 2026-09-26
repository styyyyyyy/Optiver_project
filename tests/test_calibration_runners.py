from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


class CalibrationProcessIsolationTests(unittest.TestCase):
    def test_mlp_runner_does_not_import_lightgbm_runtime(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_nested_calibration_mlp_fold.py"
        modules = _imported_modules(path)
        self.assertFalse(
            any(name == "lightgbm" or name.startswith("lightgbm.") for name in modules)
        )
        self.assertNotIn("scripts.run_experiments", modules)

    def test_lightgbm_runner_does_not_import_torch_runtime(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_nested_calibration_fold.py"
        modules = _imported_modules(path)
        self.assertFalse(
            any(name == "torch" or name.startswith("torch.") for name in modules)
        )
        self.assertNotIn("optiver.mlp", modules)


if __name__ == "__main__":
    unittest.main()

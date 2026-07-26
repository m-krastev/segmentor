import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight_navigator_nnunet.py"
SPEC = importlib.util.spec_from_file_location("preflight_navigator_nnunet", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NavigatorPreflightTest(unittest.TestCase):
    def test_deterministic_split_is_disjoint_and_complete(self):
        cases = [f"s{index:04d}" for index in range(20)]
        first = MODULE.deterministic_split(
            cases,
            seed=42,
            train_fraction=0.8,
            validation_fraction=0.1,
        )
        second = MODULE.deterministic_split(
            list(reversed(cases)),
            seed=42,
            train_fraction=0.8,
            validation_fraction=0.1,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            set(first["train"]) | set(first["validation"]) | set(first["test"]),
            set(cases),
        )
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))
        self.assertEqual(
            {name: len(case_ids) for name, case_ids in first.items()},
            {"train": 16, "validation": 2, "test": 2},
        )


if __name__ == "__main__":
    unittest.main()

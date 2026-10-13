from pathlib import Path
import shutil
import subprocess
import unittest


class SageExampleTests(unittest.TestCase):
    EXAMPLES = {
        "01_basic_lll.sage": ("BASIC_LLL_OK", False),
        "02_deep_lll.sage": ("DEEP_LLL_OK", False),
        "03_bkz_qary.sage": ("BKZ_QARY_OK", False),
        "04_column_basis.sage": ("COLUMN_BASIS_OK", False),
        "05_compare_flatter_blaster.sage": ("FLATTER_BLASTER_COMPARE_OK", True),
        "06_large_integer_boundary.sage": ("LARGE_INTEGER_BOUNDARY_OK", True),
    }

    @unittest.skipUnless(shutil.which("sage"), "Sage executable is required")
    def test_sage_examples(self):
        root = Path(__file__).resolve().parents[1]
        example_dir = root / "examples" / "sage"

        for filename, (marker, requires_flatter) in self.EXAMPLES.items():
            with self.subTest(filename=filename):
                if requires_flatter and shutil.which("flatter") is None:
                    self.skipTest("Flatter executable is required for comparison examples")
                result = subprocess.run(
                    ["sage", str(example_dir / filename)],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(marker, result.stdout)


if __name__ == "__main__":
    unittest.main()

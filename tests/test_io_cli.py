import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from blaster.lattice_io import parse_lattice, read_lattice, write_lattice


class LatticeIoTests(unittest.TestCase):
    def setUp(self):
        self.columns = np.array(
            [
                [7, 2, 1],
                [0, 5, 1],
                [0, 0, 1],
            ],
            dtype=np.int64,
        )

    def test_stream_round_trip(self):
        stream = io.StringIO()
        write_lattice(self.columns, stream)
        stream.seek(0)

        actual = read_lattice(stream, qary_order=False)

        np.testing.assert_array_equal(actual, self.columns)

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.lat"
            write_lattice(self.columns, path)
            actual = read_lattice(path, qary_order=False)

        np.testing.assert_array_equal(actual, self.columns)

    def test_parser_accepts_fplll_whitespace(self):
        actual = parse_lattice("[ [1 2]\n [3 -4] ]", qary_order=False)
        expected = np.array([[1, 3], [2, -4]], dtype=np.int64)
        np.testing.assert_array_equal(actual, expected)

    def test_parser_reports_malformed_and_overflow_input(self):
        with self.assertRaisesRegex(ValueError, "lattice"):
            parse_lattice("[[1 2] [3]]", qary_order=False)

        with self.assertRaisesRegex(OverflowError, "int64"):
            parse_lattice(f"[[{2**63} 0] [0 1]]", qary_order=False)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.columns = np.array(
            [
                [10, 1, 0],
                [0, 10, 1],
                [0, 0, 1],
            ],
            dtype=np.int64,
        )
        self.env = os.environ.copy()
        source_dir = str(Path(__file__).resolve().parents[1] / "src")
        self.env["PYTHONPATH"] = os.pathsep.join(
            value for value in (source_dir, self.env.get("PYTHONPATH")) if value
        )

    def run_cli(self, *arguments, input_text=None):
        return subprocess.run(
            [sys.executable, "-m", "blaster", *arguments],
            text=True,
            input=input_text,
            capture_output=True,
            env=self.env,
            check=False,
        )

    def test_module_cli_handles_stdin_and_stdout(self):
        source = io.StringIO()
        write_lattice(self.columns, source)

        result = self.run_cli("--no-seysen", "--cores", "1", input_text=source.getvalue())

        self.assertEqual(result.returncode, 0, result.stderr)
        reduced = parse_lattice(result.stdout, qary_order=False)
        self.assertEqual(reduced.shape, self.columns.shape)

    def test_module_cli_handles_files(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.lat"
            output_path = Path(directory) / "output.lat"
            write_lattice(self.columns, input_path)

            result = self.run_cli(
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--no-seysen",
                "--cores",
                "1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_path.exists())
            reduced = read_lattice(output_path, qary_order=False)
            self.assertEqual(reduced.shape, self.columns.shape)

    def test_file_and_stream_cli_results_match(self):
        source = io.StringIO()
        write_lattice(self.columns, source)
        stream_result = self.run_cli(
            "--no-seysen", "--cores", "1", input_text=source.getvalue()
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.lat"
            output_path = Path(directory) / "output.lat"
            input_path.write_text(source.getvalue(), encoding="utf-8")
            file_result = self.run_cli(
                "--input", str(input_path),
                "--output", str(output_path),
                "--no-seysen", "--cores", "1",
            )
            file_output = output_path.read_text(encoding="utf-8")

        self.assertEqual(stream_result.returncode, 0, stream_result.stderr)
        self.assertEqual(file_result.returncode, 0, file_result.stderr)
        self.assertEqual(stream_result.stdout, file_output)

    def test_cli_reports_invalid_parameters_without_traceback(self):
        result = self.run_cli("--delta", "0.25", input_text="[[1 0] [0 1]]")

        self.assertEqual(result.returncode, 2)
        self.assertIn("delta", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_profiles_rank_one_basis(self):
        result = self.run_cli(
            "--profile", "--no-seysen", "--cores", "1", input_text="[[5]]"
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("slope = 0.000000", result.stderr)

    def test_compatibility_script_exposes_help(self):
        script = Path(__file__).resolve().parents[1] / "src" / "app.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            text=True,
            capture_output=True,
            env=self.env,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("BLASter", result.stdout)


if __name__ == "__main__":
    unittest.main()

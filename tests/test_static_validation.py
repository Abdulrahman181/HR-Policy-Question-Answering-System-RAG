"""Dependency-free repository checks; these do not import the ML application."""

from __future__ import annotations

import ast
import json
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class StaticValidationTests(unittest.TestCase):
    def test_application_parses(self) -> None:
        source = (REPO / "app.py").read_text(encoding="utf-8")
        ast.parse(source, filename="app.py")

    def test_notebook_code_cells_parse_after_removing_ipython_magics(self) -> None:
        notebook = json.loads(
            (REPO / "hr-policy-question-answering-system-rag.ipynb").read_text(
                encoding="utf-8"
            )
        )
        for number, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            python_lines = [
                line for line in source.splitlines() if not line.lstrip().startswith("!")
            ]
            ast.parse("\n".join(python_lines), filename=f"notebook cell {number}")

    def test_notebook_has_no_stale_outputs_or_machine_specific_kaggle_path(self) -> None:
        notebook = json.loads(
            (REPO / "hr-policy-question-answering-system-rag.ipynb").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(any(cell.get("outputs") for cell in notebook["cells"]))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertNotIn("/kaggle/input/", source)

    def test_runtime_requirements_cover_notebook_imports(self) -> None:
        requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
        for package in ("pypdf", "langchain-community", "langchain-text-splitters", "tqdm"):
            self.assertIn(package, requirements)

    def test_index_artifact_is_present(self) -> None:
        self.assertTrue((REPO / "chroma_hr_db").is_dir())


if __name__ == "__main__":
    unittest.main()

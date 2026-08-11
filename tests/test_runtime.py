from __future__ import annotations

import json
from pathlib import Path
import tomllib

from eigenmode_analysis import __version__
from eigenmode_analysis.runtime import main, runtime_info


def test_runtime_info_identifies_source_used_by_tests():
    info = runtime_info()

    assert info.version == __version__
    assert info.install_mode in {"source-tree", "editable"}
    assert Path(info.module_file).name == "runtime.py"
    assert Path(info.python_executable).is_file()


def test_runtime_cli_emits_machine_readable_provenance(capsys):
    assert main(["--json", "--require-mode", "source"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["distribution"] == "eigenmode-analysis"
    assert payload["version"] == __version__
    assert payload["install_mode"] in {"source-tree", "editable"}


def test_pyproject_and_runtime_versions_are_identical():
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert project["project"]["version"] == __version__

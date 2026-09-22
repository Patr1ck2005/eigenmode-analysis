"""Run the shared/A/B compatibility matrix without mutating their environments."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT.parent
FORBIDDEN_IMPORTS = {
    "myplots_app",
    "tests",
    "plotly",
    "campaign_data",
    "plot_workflows",
    "core",
    "jpype",
    "matplotlib",
    "mph",
    "pandas",
    "research_agent_workbench",
}


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    cwd: Path
    pythonpath: tuple[Path, ...]


def _environment(pythonpath: tuple[Path, ...]) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath)
    return environment


def _run(step: Step) -> dict[str, float | str]:
    print(f"\n[{step.name}]")
    print("+ " + subprocess.list2cmdline(step.command))
    started = time.perf_counter()
    subprocess.run(
        step.command,
        cwd=step.cwd,
        env=_environment(step.pythonpath),
        check=True,
    )
    return {"name": step.name, "duration_s": time.perf_counter() - started}


def _scan_dependency_boundary() -> None:
    violations = []
    for path in sorted((ROOT / "src" / "eigenmode_analysis").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name in FORBIDDEN_IMPORTS:
                    violations.append(f"{path}:{node.lineno}: {name}")
    if violations:
        raise SystemExit("Shared dependency boundary violations:\n" + "\n".join(violations))
    print("\n[dependency-boundary]\nno forbidden imports")


def _write_import_blocker(directory: Path) -> Path:
    path = directory / "sitecustomize.py"
    path.write_text(
        """import sys

class _BlockEigenmodeAnalysis:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'eigenmode_analysis' or fullname.startswith('eigenmode_analysis.'):
            raise ModuleNotFoundError(
                'eigenmode_analysis intentionally blocked by compatibility test',
                name=fullname,
            )
        return None

sys.meta_path.insert(0, _BlockEigenmodeAnalysis())
""",
        encoding="ascii",
    )
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--myplots-root", type=Path, default=WORK_ROOT / "myPlots")
    parser.add_argument(
        "--workbench-root",
        type=Path,
        default=WORK_ROOT / "research-agent-workbench",
    )
    parser.add_argument(
        "--myplots-python",
        type=Path,
        default=WORK_ROOT / "myPlots" / ".venv" / "Scripts" / "python.exe",
    )
    parser.add_argument(
        "--workbench-python",
        type=Path,
        default=(
            WORK_ROOT
            / "research-agent-workbench"
            / ".venv"
            / "Scripts"
            / "python.exe"
        ),
    )
    parser.add_argument("--skip-without-package", action="store_true")
    parser.add_argument("--json-report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    myplots_root = args.myplots_root.resolve()
    workbench_root = args.workbench_root.resolve()
    myplots_python = args.myplots_python.resolve()
    workbench_python = args.workbench_python.resolve()
    for path in (myplots_root, workbench_root, myplots_python, workbench_python):
        if not path.exists():
            raise SystemExit(f"Required path does not exist: {path}")

    shared_src = ROOT / "src"
    steps = [
        Step(
            "shared-tests-myplots-env",
            (str(myplots_python), "-m", "pytest", "-q"),
            ROOT,
            (shared_src,),
        ),
        Step(
            "shared-tests-workbench-env",
            (str(workbench_python), "-m", "pytest", "-q"),
            ROOT,
            (shared_src,),
        ),
        Step(
            "myplots-adapter-tests",
            (str(myplots_python), "-m", "pytest", "shared_tests", "-q"),
            myplots_root,
            (myplots_root, shared_src),
        ),
        Step(
            "workbench-with-shared",
            (str(workbench_python), "-m", "pytest", "-q"),
            workbench_root,
            (shared_src,),
        ),
    ]

    results = []
    for step in steps:
        results.append(_run(step))

    if not args.skip_without_package:
        with tempfile.TemporaryDirectory(prefix="eigenmode-import-block-") as raw_temp:
            blocker = _write_import_blocker(Path(raw_temp))
            results.append(
                _run(
                    Step(
                        "workbench-without-shared",
                        (str(workbench_python), "-m", "pytest", "-q"),
                        workbench_root,
                        (blocker,),
                    )
                )
            )

    _scan_dependency_boundary()
    report = {"status": "success", "steps": results}
    if args.json_report:
        report_path = args.json_report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"\nreport: {report_path}")
    print("\ncompatibility matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build and switch eigenmode-analysis in an explicitly selected environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + subprocess.list2cmdline(command))
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def _interpreter_info(python: Path) -> dict[str, str | bool]:
    code = (
        "import json,sys;"
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'is_venv':sys.prefix!=sys.base_prefix}))"
    )
    result = _run([str(python), "-c", code], capture=True)
    return json.loads(result.stdout)


def _assert_mutable_target(python: Path, allow_system: bool) -> None:
    info = _interpreter_info(python)
    if not info["is_venv"] and not allow_system:
        raise SystemExit(
            "Refusing to modify a system interpreter. Select a virtual environment "
            "or pass --allow-system explicitly."
        )


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _default_wheel() -> Path:
    wheel = ROOT / "dist" / f"eigenmode_analysis-{_project_version()}-py3-none-any.whl"
    if not wheel.is_file():
        raise SystemExit(f"Stable wheel does not exist: {wheel}. Run build-wheel first.")
    return wheel


def _runtime_status(python: Path, required_mode: str | None = None) -> dict:
    command = [str(python), "-m", "eigenmode_analysis.runtime", "--json"]
    if required_mode:
        command.extend(["--require-mode", required_mode])
    result = _run(command, capture=True)
    payload = json.loads(result.stdout)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage stable and editable eigenmode-analysis installs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("status", "build-wheel", "install-stable", "install-editable"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--python",
            type=Path,
            required=True,
            help="Exact target Python interpreter.",
        )
        if name.startswith("install-"):
            subparser.add_argument(
                "--allow-system",
                action="store_true",
                help="Allow mutation of a non-venv interpreter.",
            )
        if name == "install-stable":
            subparser.add_argument("--wheel", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    python = args.python.resolve()
    if not python.is_file():
        raise SystemExit(f"Python interpreter does not exist: {python}")

    if args.command == "status":
        _runtime_status(python)
        return 0
    if args.command == "build-wheel":
        _run(
            [
                str(python),
                "-m",
                "pip",
                "wheel",
                str(ROOT),
                "--no-deps",
                "--wheel-dir",
                str(ROOT / "dist"),
            ]
        )
        print(_default_wheel())
        return 0

    _assert_mutable_target(python, args.allow_system)
    if args.command == "install-stable":
        wheel = (args.wheel or _default_wheel()).resolve()
        if not wheel.is_file():
            raise SystemExit(f"Wheel does not exist: {wheel}")
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ]
        )
        _runtime_status(python, "stable")
        return 0
    if args.command == "install-editable":
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                "--editable",
                str(ROOT),
            ]
        )
        _runtime_status(python, "editable")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for scripts/bootstrap.py — the one-command installer and doctor.

The script must run before anything is installed, so it is deliberately
stdlib-only and lives outside the package.  It is loaded here by path, and only
the pure decision-making is tested: version verdicts, wheel-tag matching, the
install plan, and report formatting.  Running an actual install is out of scope
for unit tests (``--dry-run`` prints the plan without touching anything).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_PATH = REPO_ROOT / "scripts" / "bootstrap.py"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("virtgaze_bootstrap", BOOTSTRAP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = _load_bootstrap()


# ── Version parsing ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3.11.9", (3, 11)),
        ("3.10", (3, 10)),
        ("Python 3.12.4", (3, 12)),
        ("3.11.9+something", (3, 11)),
        ("v3.9.18", (3, 9)),
    ],
)
def test_parse_version(text, expected):
    assert bs.parse_version(text) == expected


def test_parse_version_rejects_text_without_a_version():
    with pytest.raises(ValueError):
        bs.parse_version("no version here")


def test_parse_requires_python_reads_the_lower_bound():
    assert bs.parse_requires_python(">=3.10") == (3, 10)


def test_parse_requires_python_falls_back_on_a_foreign_spec():
    assert bs.parse_requires_python("~=x") == bs.parse_version(bs.FALLBACK_REQUIRES_PYTHON)


def test_read_requires_python_from_a_file(tmp_path):
    pyproject = tmp_path / "pyproject.toml"

    pyproject.write_text('[project]\nrequires-python = ">=3.12"\n', encoding="utf-8")

    assert bs.read_requires_python(pyproject) == ">=3.12"


def test_read_requires_python_falls_back_when_the_file_is_missing(tmp_path):
    assert bs.read_requires_python(tmp_path / "nope.toml") == bs.FALLBACK_REQUIRES_PYTHON


def test_read_requires_python_ignores_commented_lines(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('# requires-python = ">=9.9"\n', encoding="utf-8")

    assert bs.read_requires_python(pyproject) == bs.FALLBACK_REQUIRES_PYTHON


# ── The version gate, and its drift guard ─────────────────────────────────────


def test_the_floor_comes_from_pyproject_not_a_duplicate_constant():
    """Guard: the script must not hardcode a floor that drifts from the metadata."""
    declared = bs.parse_requires_python(bs.read_requires_python())

    assert bs.check_python_version((*declared, 0)).ok is True
    assert bs.check_python_version((declared[0], declared[1] - 1, 0)).ok is False


def test_the_tested_range_is_not_inverted():
    declared = bs.parse_requires_python(bs.read_requires_python())

    assert bs.MAX_PYTHON >= declared


def test_too_old_is_a_hard_failure():
    verdict = bs.check_python_version((3, 7, 0), requires_python=">=3.10")

    assert verdict.ok is False
    assert "older than the required" in verdict.message


def test_first_supported_version_passes():
    verdict = bs.check_python_version((3, 10, 0), requires_python=">=3.10")

    assert verdict.ok is True
    assert verdict.outside_tested_range is False


def test_newer_than_tested_is_flagged_but_not_blocked():
    verdict = bs.check_python_version((3, 99, 0), requires_python=">=3.10")

    assert verdict.ok is True
    assert verdict.outside_tested_range is True
    assert "checking" in verdict.message


# ── Wheel-tag matching ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename,version,expected",
    [
        # Exact CPython tag.
        ("torch-2.14.0-cp312-cp312-manylinux_2_28_x86_64.whl", (3, 12), True),
        ("torch-2.14.0-cp312-cp312-manylinux_2_28_x86_64.whl", (3, 13), False),
        ("pygame-2.6.1-cp313-cp313-manylinux_2_17_x86_64.whl", (3, 14), False),
        # "py3" Python tag: works on any Python 3 (this is how mediapipe ships,
        # and treating it as incompatible wrongly blocked a working install).
        ("mediapipe-1.0.1-py3-none-manylinux_2_28_x86_64.whl", (3, 14), True),
        ("thing-1.0-py3-none-any.whl", (3, 14), True),
        ("thing-1.0-py2.py3-none-any.whl", (3, 10), True),
        # Stable ABI: fine on an equal or newer CPython.
        ("opencv_python-5.0.0.93-cp37-abi3-manylinux_2_28_x86_64.whl", (3, 10), True),
        ("opencv_python-5.0.0.93-cp37-abi3-manylinux_2_28_x86_64.whl", (3, 14), True),
        # Source-only and unrelated tags are not usable wheels.
        ("thing-1.0.tar.gz", (3, 12), False),
        ("thing-1.0-cp39-cp39-win_amd64.whl", (3, 12), False),
    ],
)
def test_wheel_supported_for(filename, version, expected):
    assert bs.wheel_supported_for(filename, version) is expected


def test_pygame_has_no_wheel_for_python_314_but_does_for_313():
    """The case that motivated the check, using its real filename pattern."""
    wheels = [
        "pygame-2.6.1-cp310-cp310-manylinux_2_17_x86_64.whl",
        "pygame-2.6.1-cp312-cp312-manylinux_2_17_x86_64.whl",
        "pygame-2.6.1-cp313-cp313-manylinux_2_17_x86_64.whl",
    ]

    assert any(bs.wheel_supported_for(name, (3, 13)) for name in wheels) is True
    assert any(bs.wheel_supported_for(name, (3, 14)) for name in wheels) is False


def test_wheel_status_maps_each_package():
    def fetch(package):
        return {
            "good": ["good-1.0-py3-none-any.whl"],
            "old": ["old-1.0-cp38-cp38-linux_x86_64.whl"],
            "unknown": [],
        }[package]

    status = bs.wheel_status(["good", "old", "unknown"], (3, 12), fetch=fetch)

    assert status == {"good": True, "old": False, "unknown": None}


def test_blocked_packages_treats_unknown_as_not_blocked(monkeypatch):
    """An unreachable PyPI must not stop the install from being attempted."""
    monkeypatch.setattr(
        bs, "wheel_status", lambda packages, version: {name: None for name in packages}
    )

    assert bs._blocked_packages((3, 14)) == []


def test_blocked_packages_lists_only_absent_ones(monkeypatch):
    monkeypatch.setattr(
        bs,
        "wheel_status",
        lambda packages, version: {"mediapipe": True, "pygame": False, "torch": None},
    )

    assert bs._blocked_packages((3, 14)) == ["pygame"]


# ── Environment detection ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "env,system,expected",
    [
        ({}, "Linux", False),
        ({"DISPLAY": ":0"}, "Linux", True),
        ({"WAYLAND_DISPLAY": "wayland-0"}, "Linux", True),
        ({}, "Darwin", True),
        ({}, "Windows", True),
    ],
)
def test_display_available(env, system, expected):
    assert bs.display_available(env=env, system=system) is expected


def test_opencv_variants_are_mutually_exclusive():
    assert bs.opencv_package(True) == "opencv-python"
    assert bs.opencv_package(False) == "opencv-python-headless"
    assert bs.conflicting_opencv_package(True) == "opencv-python-headless"
    assert bs.conflicting_opencv_package(False) == "opencv-python"


def test_torch_index_url():
    assert bs.torch_index_url("cpu") == bs.CPU_TORCH_INDEX
    assert bs.torch_index_url("rocm") == bs.ROCM_TORCH_INDEX_TEMPLATE.format(
        version=bs.DEFAULT_ROCM_VERSION
    )
    assert bs.torch_index_url("rocm", "6.4").endswith("rocm6.4")
    # Plain PyPI: on Linux/Windows those wheels already bundle CUDA.
    assert bs.torch_index_url("auto") is None
    assert bs.torch_index_url("none") is None


def test_venv_python_layouts():
    root = Path("/tmp/example")

    assert bs.venv_python(root, windows=False) == root / "bin" / "python"
    assert bs.venv_python(root, windows=True) == root / "Scripts" / "python.exe"


# ── Install plan ──────────────────────────────────────────────────────────────


def _commands(steps):
    return [" ".join(step.command) for step in steps]


def test_plan_starts_with_pip_and_ends_with_the_project():
    commands = _commands(bs.build_plan("/usr/bin/python3.12"))

    assert "install --upgrade pip" in commands[0]
    assert commands[-1].endswith("install -e .")


def test_plan_removes_the_conflicting_opencv_before_installing():
    """Both builds provide cv2, so leaving the other installed is a trap."""
    commands = _commands(bs.build_plan("/py", opencv_display=False))

    assert "uninstall -y opencv-python" in commands[1]
    assert commands[2].endswith("install opencv-python-headless")


def test_plan_installs_the_display_build_when_a_display_exists():
    commands = _commands(bs.build_plan("/py", opencv_display=True))

    assert "uninstall -y opencv-python-headless" in commands[1]
    assert commands[2].endswith("install opencv-python")


def test_plan_uses_plain_pypi_torch_by_default():
    commands = _commands(bs.build_plan("/py", torch_mode="auto"))

    torch_step = next(command for command in commands if command.endswith("install torch"))
    assert "--index-url" not in torch_step


def test_plan_adds_the_cpu_index_for_cpu_torch():
    commands = _commands(bs.build_plan("/py", torch_mode="cpu"))

    torch_step = next(command for command in commands if " install torch" in command)
    assert f"--index-url {bs.CPU_TORCH_INDEX}" in torch_step


def test_plan_for_rocm_adds_the_rocm_index_and_onnxruntime_rocm():
    steps = bs.build_plan("/py", torch_mode="rocm", rocm_version="6.4")
    commands = _commands(steps)

    torch_step = next(command for command in commands if " install torch" in command)
    assert "rocm6.4" in torch_step
    assert any(command.endswith("install onnxruntime-rocm") for command in commands)


def test_plan_can_leave_torch_alone():
    commands = _commands(bs.build_plan("/py", torch_mode="none"))

    assert not any(" install torch" in command for command in commands)


def test_plan_includes_dev_extras_when_asked():
    commands = _commands(bs.build_plan("/py", dev=True))

    assert commands[-1].endswith("install -e .[dev]")


def test_plan_without_deps_passes_no_deps():
    commands = _commands(bs.build_plan("/py", with_project_deps=False))

    assert commands[-1].endswith("install -e . --no-deps")


def test_plan_targets_the_given_interpreter():
    commands = _commands(bs.build_plan("/opt/py/bin/python"))

    assert all(command.startswith("/opt/py/bin/python -m pip") for command in commands)


def test_plan_accepts_an_explicit_pip_executable():
    commands = _commands(bs.build_plan("/py", pip_exe="/opt/py/bin/pip"))

    assert commands[0].startswith("/opt/py/bin/pip install")


def test_format_plan_shows_every_command_with_numbers():
    steps = bs.build_plan("/py")

    text = bs.format_plan(steps, "/py")

    assert "Environment plan (using /py):" in text
    for index in range(1, len(steps) + 1):
        assert f"  {index}. " in text
    assert "/py -m pip install -e ." in text


# ── Report formatting ─────────────────────────────────────────────────────────


def _report(**overrides):
    base = {
        "python": "3.12.4",
        "packages": {"cv2": "5.0.0", "torch": "2.14.0"},
        "errors": {},
        "extra": {"device": "CPU", "screen": "1920x1080"},
    }
    base.update(overrides)
    return base


def test_report_is_ok_when_nothing_is_missing():
    text, ok = bs.format_report(_report(), required=("cv2", "torch"))

    assert ok is True
    assert "Everything needed is importable." in text
    assert "PyTorch device" in text
    assert "3.12.4" in text


def test_report_fails_when_a_required_package_is_missing():
    report = _report(errors={"torch": "ModuleNotFoundError: No module named 'torch'"})

    text, ok = bs.format_report(report, required=("cv2", "torch"))

    assert ok is False
    assert "Missing required packages: torch" in text
    assert "Problems" in text


def test_report_stays_ok_when_only_optional_packages_are_missing():
    """tensorrt and pygame are optional; their absence must not fail the check."""
    report = _report(errors={"tensorrt": "ModuleNotFoundError: No module named 'tensorrt'"})

    text, ok = bs.format_report(report, required=("cv2", "torch"))

    assert ok is True
    assert "Optional components unavailable (tensorrt)" in text


def test_report_survives_an_unrunnable_interpreter():
    report = {"errors": {"interpreter": "FileNotFoundError: nope"}, "packages": {}, "extra": {}}

    text, ok = bs.format_report(report, required=("cv2",))

    assert ok is False
    assert "FileNotFoundError" in text
    assert "Could not inspect the environment" in text


# ── The generated check snippet ───────────────────────────────────────────────


def test_snippet_is_valid_python():
    """The snippet is a string, so nothing else would catch a syntax error."""
    compile(bs.build_check_snippet(), "<check-snippet>", "exec")


def test_snippet_imports_every_declared_package():
    snippet = bs.build_check_snippet()

    for name in (*bs.REQUIRED_PACKAGES, *bs.OPTIONAL_PACKAGES):
        assert f"'{name}'" in snippet


def test_snippet_is_generated_from_the_required_list():
    """Drift guard: the snippet and format_report must agree on what is required."""
    snippet = bs.build_check_snippet()

    assert repr(list(bs.REQUIRED_PACKAGES)) in snippet
    assert repr(list(bs.OPTIONAL_PACKAGES)) in snippet
    assert "__REQUIRED__" not in snippet and "__OPTIONAL__" not in snippet


def test_snippet_executes_and_reports_the_interpreter():
    """Run it for real against the current interpreter."""
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", bs.build_check_snippet()],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0
    assert "<<<BOOTSTRAP_JSON>>>" in completed.stdout


def test_snippet_accepts_the_camera_probe_flag():
    assert "--probe-camera" in bs.build_check_snippet()

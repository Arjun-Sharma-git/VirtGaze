#!/usr/bin/env python3
"""One-command environment setup and health check for VirtGaze.

A bare ``pip install -e .`` fails on a fresh machine in several confusing ways.
This script avoids all of them and then verifies the result:

* **PEP 668** — Debian/Ubuntu 24.04+ refuse a system-wide ``pip install`` with
  "externally managed environment".  A virtualenv sidesteps it entirely.
* **No wheel for your Python** — MediaPipe and PyTorch stop publishing wheels
  above 3.12, so a 3.13+ interpreter produces an inexplicable resolver error.
  The version is checked up front with an actionable message.
* **Headless OpenCV** — ``opencv-python`` installs fine on a server but
  ``import cv2`` then fails on ``libGL.so.1``.  The headless build is selected
  automatically when no display is detected.
* **Silent CPU PyTorch** — on Linux/Windows the PyPI wheel already bundles CUDA;
  ``--torch cpu`` (or ``rocm``) selects an explicit build instead.

Usage::

    python scripts/bootstrap.py                     # .venv, default torch, OpenCV auto
    python scripts/bootstrap.py --torch cpu         # CPU-only PyTorch
    python scripts/bootstrap.py --opencv headless   # no-display install
    python scripts/bootstrap.py --dev               # + test/lint tooling
    python scripts/bootstrap.py --check             # verify only, change nothing
    python scripts/bootstrap.py --dry-run           # print the plan and stop

Only the standard library is imported, so it runs before anything is installed.
Exit codes: 0 success, 1 checks failed, 2 unsupported environment or usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

# ── Supported interpreters ────────────────────────────────────────────────────

# The range this project is tested on.  Above it the check does not guess: it asks
# PyPI whether the required packages actually publish a wheel for the interpreter,
# because upstream availability changes (MediaPipe and PyTorch have shipped wheels
# for interpreters newer than this before now, and pygame has lagged).
MAX_PYTHON: Tuple[int, int] = (3, 12)

# Packages whose wheel availability decides whether a newer interpreter works.
PROBE_PACKAGES: Tuple[str, ...] = ("mediapipe", "torch", "pygame", "opencv-python")

# Used only if pyproject.toml cannot be read (e.g. the script was copied out).
FALLBACK_REQUIRES_PYTHON = ">=3.10"

_REPO_ROOT = Path(__file__).resolve().parent.parent

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"

# ── PyTorch build selection ───────────────────────────────────────────────────

# None means "let pip use PyPI": on Linux and Windows those wheels already bundle
# CUDA, so a GPU user needs no extra flag.
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
ROCM_TORCH_INDEX_TEMPLATE = "https://download.pytorch.org/whl/rocm{version}"
DEFAULT_ROCM_VERSION = "6.2"


class Step(NamedTuple):
    """One shell command in the install plan."""

    description: str
    command: List[str]


class VersionVerdict(NamedTuple):
    """Outcome of the interpreter version check."""

    ok: bool
    message: str
    outside_tested_range: bool = False


# ── Version helpers ───────────────────────────────────────────────────────────


def parse_version(text: str) -> Tuple[int, int]:
    """Extract the first ``major.minor`` from a version string.

    Accepts ``"3.11.9"``, ``"3.10"``, ``"Python 3.12.4"``, and the ``3.11`` in
    ``"3.11.9+something"``.  Raises ``ValueError`` when no version is present.
    """
    match = re.search(r"(\d+)\.(\d+)", str(text))
    if not match:
        raise ValueError(f"no version found in {text!r}")
    return int(match.group(1)), int(match.group(2))


def parse_requires_python(spec: str) -> Tuple[int, int]:
    """Return the lower bound of a ``requires-python`` specifier.

    Only the ``>=`` form is understood, which is what this project uses; an
    unparseable spec falls back to :data:`FALLBACK_REQUIRES_PYTHON`'s value.
    """
    try:
        return parse_version(spec)
    except ValueError:
        return parse_version(FALLBACK_REQUIRES_PYTHON)


def read_requires_python(pyproject: Optional[Path] = None) -> str:
    """Read ``requires-python`` from pyproject.toml without a TOML library.

    The value is a single quoted string, so a regex is enough and avoids a
    dependency on ``tomllib`` (3.11+) for interpreters this script must support.
    """
    path = pyproject if pyproject is not None else _REPO_ROOT / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return FALLBACK_REQUIRES_PYTHON
    match = re.search(r'^\s*requires-python\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else FALLBACK_REQUIRES_PYTHON


def check_python_version(
    version_info: Sequence[int], requires_python: Optional[str] = None
) -> "VersionVerdict":
    """Judge an interpreter version against the project's metadata floor.

    *ok* is False only below the floor (a hard failure).  Above
    :data:`MAX_PYTHON` the verdict is still ok but flagged as outside the tested
    range, so the caller can consult PyPI before deciding.
    """
    spec = requires_python if requires_python is not None else read_requires_python()
    minimum = parse_requires_python(spec)
    version = (int(version_info[0]), int(version_info[1]))
    rendered = f"{version[0]}.{version[1]}"

    if version < minimum:
        return VersionVerdict(
            False,
            f"Python {rendered} is older than the required {spec}. "
            f"Install Python {minimum[0]}.{minimum[1]} or newer and re-run.",
        )
    if version > MAX_PYTHON:
        return VersionVerdict(
            True,
            f"Python {rendered} is newer than the tested range "
            f"({minimum[0]}.{minimum[1]}–{MAX_PYTHON[0]}.{MAX_PYTHON[1]}); checking "
            f"whether the required packages still publish wheels for it.",
            outside_tested_range=True,
        )
    return VersionVerdict(True, f"Python {rendered} is supported and tested.")


# ── Wheel availability (asked, not assumed) ───────────────────────────────────


def wheel_supported_for(filename: str, version_info: Tuple[int, int]) -> bool:
    """Whether a wheel filename can be installed on the given interpreter.

    Three cases count as compatible:

    * an exact CPython tag (``cp312``);
    * a wheel whose *Python tag* is just ``py3`` with a ``none`` ABI tag — e.g.
      ``mediapipe-1.0.1-py3-none-manylinux_2_28_x86_64.whl``.  The platform tag
      still has to match the machine, but that is pip's job; treating these as
      incompatible would wrongly block a working install;
    * a stable-ABI wheel (``abi3``) built for an equal or older CPython.
    """
    major, minor = version_info
    lowered = filename.lower()
    if f"cp{major}{minor}" in lowered:
        return True
    if re.search(r"-py3-none-", lowered) or re.search(r"-py2\.py3-none-", lowered):
        return True
    abi3 = re.search(r"cp3(\d+)-abi3", lowered)
    if abi3 and (major, int(abi3.group(1))) <= (major, minor):
        return True
    return False


def fetch_wheel_filenames(package: str, timeout: float = 20.0) -> List[str]:
    """Wheel filenames published for the newest release of *package*.

    Returns an empty list when the query fails, which callers must treat as
    "unknown" rather than "absent".
    """
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed https PyPI endpoint
            PYPI_JSON_URL.format(package=package), timeout=timeout
        ) as response:
            payload = json.load(response)
    except Exception:
        return []
    return [
        entry["filename"]
        for entry in payload.get("urls", [])
        if str(entry.get("filename", "")).endswith(".whl")
    ]


def wheel_status(
    packages: Sequence[str],
    version_info: Tuple[int, int],
    fetch=fetch_wheel_filenames,
) -> Dict[str, Optional[bool]]:
    """Map each package to whether a usable wheel exists.

    ``None`` means the probe could not tell (offline, or no files published).
    """
    status: Dict[str, Optional[bool]] = {}
    for package in packages:
        filenames = fetch(package)
        if not filenames:
            status[package] = None
        else:
            status[package] = any(wheel_supported_for(name, version_info) for name in filenames)
    return status


def _blocked_packages(version_info: Tuple[int, int]) -> List[str]:
    """Packages with no wheel for this interpreter.

    A failed probe (offline, or a package that publishes no wheels at all) counts
    as unknown rather than blocked, so the script attempts the install instead of
    refusing to start on a bad network.
    """
    status = wheel_status(PROBE_PACKAGES, version_info)
    return sorted(name for name, available in status.items() if available is False)


# ── Platform / display detection ──────────────────────────────────────────────


def display_available(env: Optional[Dict[str, str]] = None, system: Optional[str] = None) -> bool:
    """Whether a display is likely present, i.e. the OpenCV *display* build works.

    Linux servers routinely have neither variable set; macOS and Windows always
    have a window server.
    """
    environ = os.environ if env is None else env
    sysname = platform.system() if system is None else system
    if sysname in ("Darwin", "Windows"):
        return True
    return bool(environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"))


def opencv_package(for_display: bool) -> str:
    """The OpenCV distribution to install."""
    return "opencv-python" if for_display else "opencv-python-headless"


def conflicting_opencv_package(for_display: bool) -> str:
    """The OpenCV distribution that must be removed first.

    The two builds provide the same ``cv2`` module, so leaving the other one
    installed yields a confusing mixture.
    """
    return "opencv-python-headless" if for_display else "opencv-python"


def torch_index_url(mode: str, rocm_version: str = DEFAULT_ROCM_VERSION) -> Optional[str]:
    """Extra index URL for the requested PyTorch build, or None for plain PyPI."""
    if mode == "cpu":
        return CPU_TORCH_INDEX
    if mode == "rocm":
        return ROCM_TORCH_INDEX_TEMPLATE.format(version=rocm_version)
    return None


# ── Install plan ──────────────────────────────────────────────────────────────


def build_plan(
    python_exe: str,
    torch_mode: str = "auto",
    opencv_display: bool = True,
    dev: bool = False,
    editable: bool = True,
    with_project_deps: bool = True,
    rocm_version: str = DEFAULT_ROCM_VERSION,
    pip_exe: Optional[str] = None,
) -> List[Step]:
    """Return the ordered commands that set up the environment.

    Building the plan separately from running it keeps the decisions (which
    OpenCV, which torch index, which extras) testable without touching a
    package index.
    """
    pip = pip_exe if pip_exe is not None else f"{python_exe} -m pip"
    pip_cmd = pip.split()

    steps: List[Step] = []
    steps.append(
        Step("Upgrade pip and the build backend", [*pip_cmd, "install", "--upgrade", "pip"])
    )

    # The two OpenCV distributions conflict, so remove the other one first.
    steps.append(
        Step(
            "Remove the conflicting OpenCV build, if present",
            [*pip_cmd, "uninstall", "-y", conflicting_opencv_package(opencv_display)],
        )
    )
    steps.append(
        Step(
            f"Install {opencv_package(opencv_display)}",
            [*pip_cmd, "install", opencv_package(opencv_display)],
        )
    )

    if torch_mode != "none":
        index = torch_index_url(torch_mode, rocm_version)
        command = [*pip_cmd, "install", "torch"]
        if index:
            command += ["--index-url", index]
        label = (
            f"torch ({torch_mode})" if index else "torch from PyPI (CUDA-enabled on Linux/Windows)"
        )
        steps.append(Step(f"Install {label}", command))

    extras = "[dev]" if dev else ""
    target = f".{extras}"
    install = [*pip_cmd, "install"]
    if editable:
        install.append("-e")
    install.append(target)
    if not with_project_deps:
        # Used by the ROCm path, where the wheels are installed separately.
        install.append("--no-deps")
    steps.append(Step("Install the project (and its dependencies)", install))

    if torch_mode == "rocm":
        steps.append(
            Step(
                "Install onnxruntime-rocm (not on PyPI)",
                [*pip_cmd, "install", "onnxruntime-rocm"],
            )
        )

    return steps


def format_plan(steps: Sequence[Step], python_exe: str) -> str:
    """Render the plan for the user, one numbered step per line."""
    lines = [f"Environment plan (using {python_exe}):", ""]
    for index, step in enumerate(steps, start=1):
        lines.append(f"  {index}. {step.description}")
        lines.append(f"       {' '.join(step.command)}")
    return "\n".join(lines)


# The packages without which the pipeline cannot run, and the optional ones that
# only degrade a feature.  Declared once: the check snippet below is generated
# from these, so it cannot drift from what format_report treats as required.
REQUIRED_PACKAGES: Tuple[str, ...] = (
    "cv2",
    "numpy",
    "scipy",
    "yaml",
    "pydantic",
    "mediapipe",
    "torch",
    "onnxruntime",
)
OPTIONAL_PACKAGES: Tuple[str, ...] = ("screeninfo", "pygame", "onnx", "tensorrt")


# ── Health check ──────────────────────────────────────────────────────────────

# Run inside the target interpreter.  Emitting JSON keeps the parsing trivial and
# avoids depending on the wording of any library's output.
_CHECK_TEMPLATE = r"""
import json, sys

REQUIRED = __REQUIRED__
OPTIONAL = __OPTIONAL__

report = {"python": sys.version.split()[0], "packages": {}, "errors": {}, "extra": {}}
ALIASES = {"yaml": "PyYAML"}
for name in REQUIRED + OPTIONAL:
    try:
        module = __import__(name)
        value = getattr(module, "__version__", None)
        if value is None:  # e.g. screeninfo exposes no __version__
            from importlib import metadata as _metadata

            value = _metadata.version(ALIASES.get(name, name))
        report["packages"][name] = str(value)
    except Exception as exc:
        report["errors"][name] = f"{type(exc).__name__}: {exc}"

try:
    # Detected from distribution metadata, not by probing cv2: the headless wheel
    # still defines cv2.imshow (it only fails when called).
    from importlib import metadata

    for _dist in ("opencv-python", "opencv-python-headless", "opencv-contrib-python"):
        try:
            report["extra"]["opencv_build"] = f"{_dist} {metadata.version(_dist)}"
            break
        except metadata.PackageNotFoundError:
            continue
except Exception:
    pass

try:
    from gaze_estimation.utils.device import describe_device
    report["extra"]["device"] = describe_device()
except Exception as exc:
    report["extra"]["device"] = f"unavailable ({type(exc).__name__})"

try:
    import screeninfo
    monitors = screeninfo.get_monitors()
    if monitors:
        report["extra"]["screen"] = f"{monitors[0].width}x{monitors[0].height}"
except Exception:
    report["extra"]["screen"] = "not detected"

if "--probe-camera" in sys.argv:
    try:
        import cv2
        capture = cv2.VideoCapture(0)
        ok = capture.isOpened()
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0
        capture.release()
        report["extra"]["camera"] = f"index 0 OK ({width}x{height})" if ok else "index 0 not available"
    except Exception as exc:
        report["extra"]["camera"] = f"probe failed ({type(exc).__name__})"

print("<<<BOOTSTRAP_JSON>>>" + json.dumps(report))
"""


def build_check_snippet() -> str:
    """The verification snippet, with the package lists interpolated.

    Generated from :data:`REQUIRED_PACKAGES` / :data:`OPTIONAL_PACKAGES` so the
    names the snippet imports cannot drift from the ones
    :func:`format_report` classifies as required — a mismatch would let the check
    report "everything is importable" while a required package was missing.
    """
    return _CHECK_TEMPLATE.replace("__REQUIRED__", repr(list(REQUIRED_PACKAGES))).replace(
        "__OPTIONAL__", repr(list(OPTIONAL_PACKAGES))
    )


def run_checks(python_exe: str, probe_camera: bool = False) -> Dict:
    """Run :func:`build_check_snippet` in *python_exe* and return the parsed report.

    Returns a dict with an ``errors`` entry when the interpreter cannot be run at
    all, so callers never have to handle an exception.
    """
    command = [python_exe, "-c", build_check_snippet()]
    if probe_camera:
        command.append("--probe-camera")
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "errors": {"interpreter": f"{type(exc).__name__}: {exc}"},
            "packages": {},
            "extra": {},
        }

    marker = "<<<BOOTSTRAP_JSON>>>"
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    # The interpreter ran but the snippet died before printing.
    return {
        "errors": {"snippet": (completed.stderr.strip().splitlines() or ["no output"])[-1]},
        "packages": {},
        "extra": {},
    }


def format_report(report: Dict, required: Sequence[str] = ()) -> Tuple[str, bool]:
    """Render the check report; return ``(text, ok)``.

    *required* names the packages whose absence makes the install unusable; the
    optional ones are reported as informational.
    """
    errors = report.get("errors", {})
    packages = report.get("packages", {})
    extra = report.get("extra", {})

    lines = ["Environment check", "=================", ""]
    lines.append(f"  Python        : {report.get('python', 'unknown')}")
    for label, key in (
        ("PyTorch device", "device"),
        ("Screen", "screen"),
        ("Camera", "camera"),
        ("OpenCV build", "opencv_build"),
    ):
        if key in extra:
            lines.append(f"  {label:<14}: {extra[key]}")

    lines.append("")
    lines.append("  Packages")
    for name in sorted(packages):
        lines.append(f"    {name:<12} {packages[name]}")

    missing_required = [name for name in required if name in errors]
    # "interpreter"/"snippet" mean the check itself could not run; that must fail,
    # not silently pass because no *package* was individually reported missing.
    broken_tooling = [name for name in ("interpreter", "snippet") if name in errors]
    optional_errors = sorted(
        name for name in errors if name not in missing_required and name not in broken_tooling
    )
    if errors:
        lines.append("")
        lines.append("  Problems")
        for name, message in sorted(errors.items()):
            lines.append(f"    {name:<12} {message}")

    ok = not missing_required and not broken_tooling
    lines.append("")
    if not errors:
        lines.append("  Everything needed is importable.")
    elif broken_tooling:
        lines.append("  Could not inspect the environment — see Problems above.")
    elif ok:
        lines.append(
            f"  Optional components unavailable ({', '.join(optional_errors)}); "
            "the core pipeline works without them."
        )
    else:
        lines.append(f"  Missing required packages: {', '.join(sorted(missing_required))}")
        lines.append("  Re-run this script without --check to install them.")
    return "\n".join(lines), ok


# The packages without which the pipeline cannot run.
REQUIRED_PACKAGES: Tuple[str, ...] = (
    "cv2",
    "numpy",
    "scipy",
    "yaml",
    "pydantic",
    "mediapipe",
    "torch",
    "onnxruntime",
)


# ── Environment construction ──────────────────────────────────────────────────


def interpreter_version(python_exe: str) -> Optional[Tuple[int, int]]:
    """Return ``(major, minor)`` for *python_exe*, or None if it cannot be run."""
    try:
        completed = subprocess.run(
            [python_exe, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        return parse_version(completed.stdout.strip())
    except ValueError:
        return None


def venv_python(venv_path: Path, windows: Optional[bool] = None) -> Path:
    """Path of the interpreter inside a virtualenv (POSIX or Windows layout).

    *windows* defaults to the running platform and exists so both layouts can be
    asserted from one machine.
    """
    is_windows = (os.name == "nt") if windows is None else windows
    if is_windows:
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def externally_managed(python_exe: str) -> bool:
    """Whether the interpreter refuses system-wide installs (PEP 668)."""
    snippet = (
        "import os, sysconfig;"
        "print(os.path.exists(os.path.join(sysconfig.get_path('stdlib'), 'EXTERNALLY-MANAGED')))"
    )
    try:
        completed = subprocess.run(
            [python_exe, "-c", snippet], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.stdout.strip() == "True"


def create_venv(python_exe: str, venv_path: Path, verbose: bool = True) -> bool:
    """Create *venv_path*, bootstrapping pip if the interpreter ships without ensurepip.

    Debian/Ubuntu split ``ensurepip`` into a separate package, so ``python -m venv``
    fails with "ensurepip is not available" on a stock install.  Falling back to
    ``--without-pip`` plus the official ``get-pip.py`` keeps the one-command
    promise on those systems.
    """
    if venv_path.exists() and venv_python(venv_path).exists():
        if verbose:
            print(f"Reusing the existing virtualenv at {venv_path}")
        return True

    if verbose:
        print(f"Creating a virtualenv at {venv_path} …")
    if subprocess.run([python_exe, "-m", "venv", str(venv_path)]).returncode == 0:
        return True

    print("`python -m venv` failed (ensurepip is often a separate package) — retrying.")
    if subprocess.run([python_exe, "-m", "venv", "--without-pip", str(venv_path)]).returncode != 0:
        print("Could not create a virtualenv. Install the venv module, e.g. on Ubuntu:")
        print("  sudo apt install python3-venv")
        return False

    return _bootstrap_pip(venv_python(venv_path))


def _bootstrap_pip(python_path: Path) -> bool:
    """Install pip into a venv created with ``--without-pip``."""
    import tempfile

    url = "https://bootstrap.pypa.io/get-pip.py"
    print(f"Downloading pip from {url} …")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "get-pip.py"
            urllib.request.urlretrieve(url, script)  # noqa: S310 - fixed https URL
            result = subprocess.run([str(python_path), str(script)])
        return result.returncode == 0
    except Exception as exc:  # network, permissions, …
        print(f"Could not bootstrap pip: {type(exc).__name__}: {exc}")
        return False


def run_steps(steps: Sequence[Step], verbose: bool = True) -> bool:
    """Run every step in order, stopping at the first failure."""
    for index, step in enumerate(steps, start=1):
        if verbose:
            print(f"\n[{index}/{len(steps)}] {step.description}")
            print(f"    {' '.join(step.command)}")
        try:
            completed = subprocess.run(step.command)
        except OSError as exc:
            print(f"    Failed to start: {exc}")
            return False
        if completed.returncode != 0:
            print(f"    Command failed with exit code {completed.returncode}.")
            return False
    return True


def next_steps(python_exe: str, venv_path: Optional[Path]) -> str:
    """How to activate the environment and run the pipeline."""
    lines = ["", "Next steps", "==========", ""]
    if venv_path is not None:
        activate = (
            f"{venv_path}\\Scripts\\activate"
            if os.name == "nt"
            else f"source {venv_path}/bin/activate"
        )
        lines.append(f"  1. Activate the environment:  {activate}")
        prefix = "python"
    else:
        prefix = python_exe
    lines.append(f"  2. Check everything is ready: {prefix} scripts/bootstrap.py --check")
    lines.append(f"  3. Run the tracker:           {prefix} scripts/run_tracker.py")
    lines.append(f"  4. Calibrate (first run):     {prefix} scripts/run_calibration.py")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bootstrap.py",
        description="Set up and verify a VirtGaze environment in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/bootstrap.py                    # .venv, default torch, OpenCV auto\n"
            "  python scripts/bootstrap.py --torch cpu --dev  # CPU-only, with test tooling\n"
            "  python scripts/bootstrap.py --opencv headless  # server / no display\n"
            "  python scripts/bootstrap.py --check            # verify an existing install\n"
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter used to create the environment (default: the one running this script).",
    )
    parser.add_argument(
        "--venv",
        default=str(_REPO_ROOT / ".venv"),
        help="Virtualenv location (default: .venv in the repository root).",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Install into the chosen interpreter instead of a virtualenv.",
    )
    parser.add_argument(
        "--torch",
        choices=["auto", "cpu", "rocm", "none"],
        default="auto",
        help="PyTorch build: auto (PyPI — CUDA-enabled on Linux/Windows), cpu, rocm, or none.",
    )
    parser.add_argument(
        "--rocm-version",
        default=DEFAULT_ROCM_VERSION,
        help=f"ROCm wheel series for --torch rocm (default: {DEFAULT_ROCM_VERSION}).",
    )
    parser.add_argument(
        "--opencv",
        choices=["auto", "display", "headless"],
        default="auto",
        help="OpenCV flavour: auto picks headless when no display is detected.",
    )
    parser.add_argument("--dev", action="store_true", help="Also install the test/lint tooling.")
    parser.add_argument(
        "--allow-unsupported-python",
        action="store_true",
        help="Try anyway when the interpreter is newer than the tested range and wheels are missing.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Install the project without pulling its dependencies (wheels already present).",
    )
    parser.add_argument(
        "--check", action="store_true", help="Only verify the environment; change nothing."
    )
    parser.add_argument(
        "--probe-camera",
        action="store_true",
        help="Also try to open camera index 0 (adds a second and may need permissions).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.  Returns a process exit code (0 ok, 1 checks failed, 2 usage)."""
    args = build_parser().parse_args(argv)

    venv_path = None if args.no_venv else Path(args.venv).expanduser().resolve()
    target_python = str(venv_python(venv_path)) if venv_path is not None else args.python

    # --check: verify whatever is already installed and stop.
    if args.check:
        report = run_checks(target_python, probe_camera=args.probe_camera)
        text, ok = format_report(report, REQUIRED_PACKAGES)
        print(text)
        return 0 if ok else 1

    version = interpreter_version(args.python)
    if version is None:
        print(
            f"Could not run {args.python!r}. Pass a working interpreter, e.g. --python python3.12."
        )
        return 2
    verdict = check_python_version(version)
    print(verdict.message)
    if not verdict.ok:
        return 2
    if verdict.outside_tested_range:
        blocked = _blocked_packages(version)
        if blocked and not args.allow_unsupported_python:
            print()
            print(
                f"No wheels are published for Python {version[0]}.{version[1]}: {', '.join(blocked)}."
            )
            minimum = parse_requires_python(read_requires_python())
            print(
                f"Install Python {minimum[0]}.{minimum[1]}–{MAX_PYTHON[0]}.{MAX_PYTHON[1]} for a "
                f"guaranteed setup, e.g. `python3.12 scripts/bootstrap.py` "
                f"(Ubuntu: sudo apt install python3.12 python3.12-venv)."
            )
            print("Or pass --allow-unsupported-python to try anyway.")
            return 2
        if blocked:
            print(
                f"Wheels are missing for {', '.join(blocked)}; continuing because "
                "--allow-unsupported-python was given."
            )
        else:
            print("Every required package publishes a wheel for this interpreter — continuing.")

    display = display_available() if args.opencv == "auto" else args.opencv == "display"
    steps = build_plan(
        python_exe=target_python,
        torch_mode=args.torch,
        opencv_display=display,
        dev=args.dev,
        editable=True,
        with_project_deps=not args.no_deps,
        rocm_version=args.rocm_version,
    )
    print()
    print(format_plan(steps, target_python))
    if args.dry_run:
        print("\n--dry-run: nothing was installed.")
        return 0

    if args.no_venv and externally_managed(args.python):
        print(
            "\nThis interpreter is externally managed (PEP 668), so pip will refuse to "
            "install.\nDrop --no-venv to use a virtualenv, or install with "
            "`pip install --break-system-packages` if you are certain."
        )
        return 2

    if venv_path is not None and not create_venv(args.python, venv_path):
        return 2

    if not run_steps(steps):
        print(
            "\nSetup failed. Fix the problem above and re-run this command; it is safe to repeat."
        )
        return 2

    print("\nVerifying the new environment …")
    report = run_checks(target_python, probe_camera=args.probe_camera)
    text, ok = format_report(report, REQUIRED_PACKAGES)
    print()
    print(text)
    if not ok:
        print("\nSome required packages are not importable — see Problems above.")
        return 1

    print(next_steps(target_python, venv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

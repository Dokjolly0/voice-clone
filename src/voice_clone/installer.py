"""`voice-clone setup`: installs everything that does not need root, and verifies it.

- speech.cpp in ~/.local/bin, with Drama 3 (voice cloning) and Qwen3-ASR (verification);
- Piper's training code, pinned, in third_party/piper1-gpl with its own Python 3.12 environment (PyTorch).

Two modes, saved in settings.json: "gpu" (NVIDIA driver, speech.cpp's Vulkan build, PyTorch with CUDA) and "cpu"
(no GPU needed, CPU-only PyTorch; much slower). Every part is skipped when it is already in place, so running setup
again is cheap.
"""

import json
import subprocess
import time

from voice_clone import host, settings, system
from voice_clone.paths import PIPER_DIR, PIPER_PYTHON, PIPER_REPO, PIPER_REV, THIRD_PARTY_DIR
from voice_clone.util import UserError, info, run, step, warn, which

SPEECH_INSTALLER = "https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh"
SYNTH_MODEL = "drama-3"
ASR_MODEL = "qwen3-asr-0.6b"
PIPER_MARKER = THIRD_PARTY_DIR / ".piper-ready"

PIPER_CHECK = """
import sys
import torch
from piper import espeakbridge
from piper.train.vits import monotonic_align
if sys.argv[1] == "gpu":
    assert torch.cuda.is_available(), "PyTorch does not see a CUDA GPU"
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}")
else:
    print(f"torch {torch.__version__}, CPU with {torch.get_num_threads()} threads")
"""


def speech_bin() -> str:
    path = which("speech")
    if not path:
        raise UserError("speech.cpp is not installed: run `voice-clone setup` first")
    return path


def speech_gpus() -> list[dict]:
    out = subprocess.run([speech_bin(), "devices", "--json"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", check=True).stdout
    return [d for d in json.loads(out).get("devices", []) if d.get("kind") == "gpu"]


def device_args() -> list[str]:
    """speech.cpp options for the current mode: in CPU mode, never touch a GPU it may find."""
    return ["--device", "cpu"] if settings.is_cpu() else []


def choose_device(requested: str) -> str:
    """'auto' picks gpu when an NVIDIA driver works, else cpu."""
    gpu = system.query_gpu()
    if requested == "gpu" and not gpu:
        pm = system.detect_package_manager() or ""
        raise UserError("GPU mode needs a working NVIDIA driver (nvidia-smi is missing or fails).\n"
                        + system.DRIVER_HELP.get(pm, "Install the proprietary NVIDIA driver.")
                        + "\nOr use the CPU instead: voice-clone setup --device cpu")
    if requested == "auto":
        return "gpu" if gpu else "cpu"
    return requested


def check_system(device: str) -> system.SystemReport:
    step("Checking the system")
    report = system.check(device)
    info(f"Distribution: {report.distro} (package manager: {report.package_manager or 'unknown'})")
    info(f"RAM: {report.ram_gib:.0f} GB")
    if device == "gpu":
        info(f"Mode: GPU — {report.gpu.name}, {report.gpu.memory_mib / 1024:.0f} GB, driver {report.gpu.driver}")
    else:
        info("Mode: CPU only. Everything works without a GPU but is much slower: on a 16-core desktop CPU the "
             "dataset takes ~4 h and 2000 training epochs ~3 days (vs ~1 h and ~5 h on an RTX 4090); on a "
             "4-core CPU expect weeks. Fewer epochs (500-1000) shorten it.")
    if report.missing:
        cmd = report.install_command()
        hint = f"Run:\n  {cmd}\nthen run `voice-clone setup` again." if cmd else "Install them and try again."
        raise UserError(f"missing system tools: {', '.join(report.missing)}.\n{hint}")
    info("System packages: all present")
    if report.ram_gib and report.ram_gib < system.MIN_RAM_GIB[device]:
        warn(f"{report.ram_gib:.0f} GB of RAM: Drama 3 needs about {system.MIN_RAM_GIB[device]} GB in {device} "
             "mode; generating the dataset may fail or swap heavily. Close other programs, or add RAM.")
    return report


def install_speech(device: str) -> None:
    step("speech.cpp (Drama 3 and Qwen3-ASR runtime)")
    if not which("speech"):
        run(["sh", "-c", f"curl -fsSL {SPEECH_INSTALLER} | sh"])
    if device == "cpu":
        info("speech.cpp installed; it will run on the CPU")
        return
    gpus = speech_gpus()
    if not gpus:
        raise UserError("speech.cpp does not see the GPU (the CPU build was installed?). Check that the NVIDIA "
                        "driver and the Vulkan loader are installed, delete ~/.local/bin/speech and run setup again, "
                        "or use the CPU: voice-clone setup --device cpu")
    info(f"speech.cpp sees: {', '.join(d['name'] + ' (' + d['description'] + ')' for d in gpus)}")


def pull_models() -> None:
    step(f"Models: {SYNTH_MODEL} (20.8 GB) and {ASR_MODEL} (0.8 GB)")
    for model in (SYNTH_MODEL, ASR_MODEL):
        for attempt in range(1, 6):
            # `speech pull` resumes an interrupted download.
            if subprocess.run([speech_bin(), "pull", model]).returncode == 0:
                break
            warn(f"download of {model} interrupted, retrying ({attempt}/5)")
            time.sleep(10)
        else:
            raise UserError(f"could not download {model}")


def _piper_check(device: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(PIPER_PYTHON), "-c", PIPER_CHECK, device], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def piper_ready(device: str | None = None) -> bool:
    device = device or settings.device()
    marker = PIPER_MARKER.read_text(encoding="utf-8").split() if PIPER_MARKER.exists() else []
    if not (marker[:1] == [PIPER_REV] and PIPER_PYTHON.exists()):
        return False
    return _piper_check(device).returncode == 0


def install_piper(device: str) -> None:
    step(f"Piper training code ({PIPER_REV[:7]}, PyTorch for {'CUDA' if device == 'gpu' else 'CPU'})")
    if piper_ready(device):
        info("already installed and working")
        return
    uv = which("uv")
    if not uv:
        raise UserError("uv not found: start the program with ./voice-clone, which installs it")
    THIRD_PARTY_DIR.mkdir(exist_ok=True)
    if not (PIPER_DIR / ".git").exists():
        run(["git", "clone", "--quiet", PIPER_REPO, PIPER_DIR])
    run(["git", "-C", PIPER_DIR, "fetch", "--quiet", "origin"])
    run(["git", "-C", PIPER_DIR, "checkout", "--quiet", PIPER_REV])

    # Its own environment: PyTorch, Lightning and CUDA stay out of the app's environment.
    if not PIPER_PYTHON.exists():
        run([uv, "venv", "--seed", "--python", "3.12", PIPER_DIR / ".venv"])
    # GPU: PyPI's PyTorch, which is built with CUDA. CPU: the CPU-only build, without ~3 GB of CUDA libraries.
    # When the mode changed since the last install, PyTorch is reinstalled in place.
    previous = PIPER_MARKER.read_text(encoding="utf-8").split()[1:2] if PIPER_MARKER.exists() else []
    previous_device = previous[0] if previous else ("gpu" if PIPER_MARKER.exists() else None)
    backend = ["--torch-backend", "cpu"] if device == "cpu" else []
    if previous_device and previous_device != device:
        backend += ["--reinstall-package", "torch"]
    run([uv, "pip", "install", "--python", PIPER_PYTHON, *backend, "-e", f"{PIPER_DIR}[train]",
         "scikit-build", "cmake>=3.26,<4", "ninja", "cython"])

    # Two compiled pieces Piper's training needs and its package does not build when installed for development.
    align_dir = PIPER_DIR / "src/piper/train/vits/monotonic_align"
    (align_dir / "monotonic_align").mkdir(exist_ok=True)
    run([host.venv_script(PIPER_DIR / ".venv", "cythonize"), "-i", "core.pyx"], cwd=align_dir)
    for suffix in host.ext_suffixes():
        for ext in align_dir.glob(f"core*{suffix}"):
            ext.replace(align_dir / "monotonic_align" / ext.name)
    run([PIPER_PYTHON, "setup.py", "build_ext", "--inplace"], cwd=PIPER_DIR, capture=True)

    result = _piper_check(device)
    if result.returncode != 0:
        raise UserError("Piper was installed but does not work:\n" + result.stderr.strip()[-2000:])
    info(result.stdout.strip())
    PIPER_MARKER.write_text(f"{PIPER_REV} {device}\n", encoding="utf-8")


def setup(skip_models: bool = False, device: str = "auto") -> None:
    device = choose_device(device)
    check_system(device)
    settings.set_device(device)
    install_speech(device)
    install_piper(device)
    if skip_models:
        info("Skipping model downloads (--skip-models)")
    else:
        pull_models()
    step(f"Setup complete ({device.upper()} mode)")
    info("Next: voice-clone new <name> --lang <language>")


def doctor() -> None:
    """Report what is installed, without installing anything."""
    device = settings.device()
    report = system.check(device)
    ok = True

    def line(good: bool, text: str) -> None:
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {text}")

    print(f"System: {report.distro}, {report.ram_gib:.0f} GB RAM")
    print(f"Mode: {device.upper()}" + ("" if settings.SETTINGS_FILE.exists() else " (detected; not set up yet)"))
    if device == "gpu":
        line(report.gpu is not None, f"NVIDIA GPU: {report.gpu.name}, driver {report.gpu.driver}" if report.gpu
             else "NVIDIA driver: not working")
    line(not report.missing, "system packages" + (f": missing {', '.join(report.missing)}" if report.missing else ""))
    if report.missing and report.install_command():
        print(f"       fix: {report.install_command()}")
    if report.ram_gib and report.ram_gib < system.MIN_RAM_GIB[device]:
        line(False, f"RAM: {report.ram_gib:.0f} GB, Drama 3 needs about {system.MIN_RAM_GIB[device]} GB")
    line(bool(which("uv")), "uv")
    has_speech = bool(which("speech"))
    line(has_speech, "speech.cpp" + ("" if has_speech else ": not installed"))
    if has_speech and device == "gpu":
        try:
            gpus = speech_gpus()
            line(bool(gpus), "speech.cpp sees the GPU" if gpus else "speech.cpp does not see a GPU")
        except (subprocess.SubprocessError, OSError, ValueError):
            line(False, "speech.cpp devices failed")
    line(piper_ready(device), f"Piper training environment ({device.upper()})")
    print("Everything is ready." if ok else "Run `voice-clone setup` to install what is missing.")

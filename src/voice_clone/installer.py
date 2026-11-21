"""`voice-clone setup`: installs everything that does not need root, and verifies it.

- speech.cpp (Vulkan build) in ~/.local/bin, with Drama 3 (voice cloning) and Qwen3-ASR (verification);
- Piper's training code, pinned, in third_party/piper1-gpl with its own Python 3.12 environment (PyTorch + CUDA).

Every part is skipped when it is already in place, so running setup again is cheap.
"""

import json
import subprocess
import time

from voice_clone import system
from voice_clone.paths import PIPER_DIR, PIPER_PYTHON, PIPER_REPO, PIPER_REV, THIRD_PARTY_DIR
from voice_clone.util import UserError, info, run, step, warn, which

SPEECH_INSTALLER = "https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh"
SYNTH_MODEL = "drama-3"
ASR_MODEL = "qwen3-asr-0.6b"
PIPER_MARKER = THIRD_PARTY_DIR / ".piper-ready"

PIPER_CHECK = """
import torch
assert torch.cuda.is_available(), "PyTorch does not see a CUDA GPU"
from piper import espeakbridge
from piper.train.vits import monotonic_align
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}")
"""


def speech_bin() -> str:
    path = which("speech")
    if not path:
        raise UserError("speech.cpp is not installed: run `voice-clone setup` first")
    return path


def speech_gpus() -> list[dict]:
    out = subprocess.run([speech_bin(), "devices", "--json"], capture_output=True, text=True, check=True).stdout
    return [d for d in json.loads(out).get("devices", []) if d.get("kind") == "gpu"]


def check_system() -> system.SystemReport:
    step("Checking the system")
    report = system.check()
    info(f"Distribution: {report.distro} (package manager: {report.package_manager or 'unknown'})")
    if report.gpu:
        info(f"GPU: {report.gpu.name}, {report.gpu.memory_mib / 1024:.0f} GB, driver {report.gpu.driver}")
    else:
        help_text = system.DRIVER_HELP.get(report.package_manager or "", "Install the proprietary NVIDIA driver.")
        raise UserError("no working NVIDIA driver (nvidia-smi is missing or fails).\n" + help_text)
    if report.missing:
        cmd = report.install_command()
        hint = f"Run:\n  {cmd}\nthen run `voice-clone setup` again." if cmd else "Install them and try again."
        raise UserError(f"missing system tools: {', '.join(report.missing)}.\n{hint}")
    info("System packages: all present")
    return report


def install_speech() -> None:
    step("speech.cpp (Drama 3 and Qwen3-ASR runtime)")
    if not which("speech"):
        run(["sh", "-c", f"curl -fsSL {SPEECH_INSTALLER} | sh"])
    gpus = speech_gpus()
    if not gpus:
        raise UserError("speech.cpp does not see the GPU (the CPU build was installed?). Check that the NVIDIA "
                        "driver and the Vulkan loader are installed, delete ~/.local/bin/speech and run setup again.")
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


def piper_ready() -> bool:
    if not (PIPER_MARKER.exists() and PIPER_MARKER.read_text().strip() == PIPER_REV and PIPER_PYTHON.exists()):
        return False
    return subprocess.run([str(PIPER_PYTHON), "-c", PIPER_CHECK], capture_output=True).returncode == 0


def install_piper() -> None:
    step(f"Piper training code ({PIPER_REV[:7]})")
    if piper_ready():
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
    run([uv, "pip", "install", "--python", PIPER_PYTHON, "-e", f"{PIPER_DIR}[train]",
         "scikit-build", "cmake>=3.26,<4", "ninja", "cython"])

    # Two compiled pieces Piper's training needs and its package does not build when installed for development.
    align_dir = PIPER_DIR / "src/piper/train/vits/monotonic_align"
    (align_dir / "monotonic_align").mkdir(exist_ok=True)
    run([PIPER_PYTHON.parent / "cythonize", "-i", "core.pyx"], cwd=align_dir)
    for so in align_dir.glob("core*.so"):
        so.replace(align_dir / "monotonic_align" / so.name)
    run([PIPER_PYTHON, "setup.py", "build_ext", "--inplace"], cwd=PIPER_DIR, capture=True)

    result = subprocess.run([str(PIPER_PYTHON), "-c", PIPER_CHECK], capture_output=True, text=True)
    if result.returncode != 0:
        raise UserError("Piper was installed but does not work:\n" + result.stderr.strip()[-2000:])
    info(result.stdout.strip())
    PIPER_MARKER.write_text(PIPER_REV + "\n")


def setup(skip_models: bool = False) -> None:
    check_system()
    install_speech()
    install_piper()
    if skip_models:
        info("Skipping model downloads (--skip-models)")
    else:
        pull_models()
    step("Setup complete")
    info("Next: voice-clone new <name> --lang <language>")


def doctor() -> None:
    """Report what is installed, without installing anything."""
    report = system.check()
    ok = True

    def line(good: bool, text: str) -> None:
        nonlocal ok
        ok &= good
        print(f"  [{'ok' if good else '!!'}] {text}")

    print(f"System: {report.distro}")
    line(report.gpu is not None,
         f"NVIDIA GPU: {report.gpu.name}, driver {report.gpu.driver}" if report.gpu else "NVIDIA driver: not working")
    line(not report.missing, "system packages" + (f": missing {', '.join(report.missing)}" if report.missing else ""))
    if report.missing and report.install_command():
        print(f"       fix: {report.install_command()}")
    line(bool(which("uv")), "uv")
    has_speech = bool(which("speech"))
    line(has_speech, "speech.cpp" + ("" if has_speech else ": not installed"))
    if has_speech:
        try:
            gpus = speech_gpus()
            line(bool(gpus), "speech.cpp sees the GPU" if gpus else "speech.cpp does not see a GPU")
        except (subprocess.SubprocessError, OSError, ValueError):
            line(False, "speech.cpp devices failed")
    line(piper_ready(), "Piper training environment")
    print("Everything is ready." if ok else "Run `voice-clone setup` to install what is missing.")

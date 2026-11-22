"""Checks the machine: operating system, system packages, NVIDIA driver, Vulkan.

Nothing here needs root or administrator rights. When system packages are missing, the user gets the commands to
run: one sudo command on Linux (dnf or apt), winget commands on Windows.
"""

import ctypes.util
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from voice_clone import host
from voice_clone.util import which

# Command (or "lib:" shared library, or "msvc") -> package name, per package manager. A requirement without an entry
# for a package manager is not needed on that system: on Windows cmake and ninja come from pip in Piper's
# environment, and the C/C++ compilers are Visual Studio's Build Tools.
REQUIREMENTS = {
    "git": {"dnf": "git", "apt": "git", "winget": "Git.Git"},
    "curl": {"dnf": "curl", "apt": "curl"},
    "gcc": {"dnf": "gcc", "apt": "build-essential"},
    "g++": {"dnf": "gcc-c++", "apt": "build-essential"},
    "make": {"dnf": "make", "apt": "build-essential"},
    "cmake": {"dnf": "cmake", "apt": "cmake"},
    "ninja": {"dnf": "ninja-build", "apt": "ninja-build"},
    "ffmpeg": {"dnf": "ffmpeg-free", "apt": "ffmpeg", "winget": "Gyan.FFmpeg"},
    "ffprobe": {"dnf": "ffmpeg-free", "apt": "ffmpeg", "winget": "Gyan.FFmpeg"},
    "lib:vulkan": {"dnf": "vulkan-loader", "apt": "libvulkan1", "winget": "NVIDIA driver"},
    "msvc": {"winget": "Microsoft.VisualStudio.2022.BuildTools"},
}
GPU_ONLY = {"lib:vulkan"}  # speech.cpp's GPU build; the CPU mode does without
VULKAN_LIB = {"win": "vulkan-1", "linux": "vulkan"}

# winget options per package: the Build Tools need the C++ workload, which Piper's compiled parts are built with.
WINGET_OPTIONS = {
    "Microsoft.VisualStudio.2022.BuildTools":
        '--override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"',
}

# Drama 3 keeps about 21 GB in RAM when it runs on the CPU (and ~15 GB alongside a GPU with little VRAM).
MIN_RAM_GIB = {"gpu": 16, "cpu": 30}

DRIVER_HELP = {
    "dnf": (
        "Install the NVIDIA driver from RPM Fusion, wait a few minutes for the kernel module to build, then reboot:\n"
        "  sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \\\n"
        "    https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm\n"
        "  sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda\n"
        "With Secure Boot enabled the module must be signed: see RPM Fusion's NVIDIA guide."
    ),
    "apt": "Install the NVIDIA driver, then reboot:\n  sudo ubuntu-drivers install",
    "winget": ("Install the NVIDIA driver (Game Ready or Studio) from https://www.nvidia.com/drivers or the NVIDIA "
               "app, then restart Windows. It also provides nvidia-smi and Vulkan."),
}


@dataclass
class GpuInfo:
    name: str
    memory_mib: int
    driver: str


@dataclass
class SystemReport:
    distro: str
    package_manager: str | None
    missing: list[str] = field(default_factory=list)
    gpu: GpuInfo | None = None
    ram_gib: float = 0.0

    @property
    def missing_packages(self) -> list[str]:
        if not self.package_manager:
            return []
        return sorted({REQUIREMENTS[m][self.package_manager] for m in self.missing
                       if self.package_manager in REQUIREMENTS[m]})

    def install_command(self) -> str | None:
        """The command(s) that install the missing packages; on Windows one winget command per line."""
        pkgs = self.missing_packages
        if not pkgs:
            return None
        if self.package_manager == "winget":
            lines = []
            for pkg in pkgs:
                if pkg == "NVIDIA driver":
                    lines.append("install the NVIDIA driver from https://www.nvidia.com/drivers")
                    continue
                extra = f" {WINGET_OPTIONS[pkg]}" if pkg in WINGET_OPTIONS else ""
                lines.append(f"winget install -e --id {pkg}{extra}")
            return "\n  ".join(lines) + "\n  (then open a new terminal, so PATH includes the new tools)"
        if self.package_manager == "apt":
            return "sudo apt update && sudo apt install -y " + " ".join(pkgs)
        return "sudo dnf install -y " + " ".join(pkgs)


def read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    data = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value.strip().strip('"')
    return data


def os_name() -> str:
    if host.IS_WINDOWS:
        return f"Windows {platform.release()} ({platform.version()})"
    return read_os_release().get("PRETTY_NAME", "unknown Linux")


def detect_package_manager() -> str | None:
    if host.IS_WINDOWS:
        return "winget"
    for pm in ("dnf", "apt"):
        if which(pm):
            return pm
    return None


def find_msvc() -> str | None:
    """The folder of a Visual Studio (or Build Tools) installation with the C++ compilers, found with vswhere."""
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    try:
        out = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def needed_here(packages: dict) -> bool:
    """Whether a requirement applies to this operating system."""
    return "winget" in packages if host.IS_WINDOWS else bool({"dnf", "apt"} & packages.keys())


def _present(req: str) -> bool:
    if req == "msvc":
        return find_msvc() is not None
    if req.startswith("lib:"):
        name = VULKAN_LIB["win" if host.IS_WINDOWS else "linux"] if req == "lib:vulkan" else req[4:]
        return ctypes.util.find_library(name) is not None
    return which(req) is not None


def query_gpu() -> GpuInfo | None:
    """The first NVIDIA GPU as nvidia-smi reports it, or None when there is no working driver."""
    if not which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
            timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return None
    name, memory, driver = [p.strip() for p in lines[0].split(",")]
    return GpuInfo(name=name, memory_mib=int(float(memory)), driver=driver)


def ram_gib() -> float:
    try:
        return psutil.virtual_memory().total / 1024 ** 3
    except (OSError, RuntimeError):
        return 0.0


def check(device: str = "gpu") -> SystemReport:
    """System packages needed in the given mode ("gpu" or "cpu"), the GPU if any, and the RAM."""
    report = SystemReport(distro=os_name(), package_manager=detect_package_manager())
    for req, packages in REQUIREMENTS.items():
        if device == "cpu" and req in GPU_ONLY:
            continue
        if not needed_here(packages):
            continue
        if not _present(req):
            report.missing.append(req)
    report.gpu = query_gpu()
    report.ram_gib = ram_gib()
    return report

"""Checks the machine: distribution, system packages, NVIDIA driver, Vulkan.

Nothing here needs root. When system packages are missing, the user gets the single sudo command to run.
"""

import ctypes.util
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from voice_clone.util import which

# Command (or shared library) -> package name, per package manager.
REQUIREMENTS = {
    "git": {"dnf": "git", "apt": "git"},
    "curl": {"dnf": "curl", "apt": "curl"},
    "gcc": {"dnf": "gcc", "apt": "build-essential"},
    "g++": {"dnf": "gcc-c++", "apt": "build-essential"},
    "make": {"dnf": "make", "apt": "build-essential"},
    "cmake": {"dnf": "cmake", "apt": "cmake"},
    "ninja": {"dnf": "ninja-build", "apt": "ninja-build"},
    "ffmpeg": {"dnf": "ffmpeg-free", "apt": "ffmpeg"},
    "ffprobe": {"dnf": "ffmpeg-free", "apt": "ffmpeg"},
    "lib:vulkan": {"dnf": "vulkan-loader", "apt": "libvulkan1"},
}
GPU_ONLY = {"lib:vulkan"}  # speech.cpp's GPU build; the CPU mode does without

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
        return sorted({REQUIREMENTS[m][self.package_manager] for m in self.missing})

    def install_command(self) -> str | None:
        pkgs = self.missing_packages
        if not pkgs:
            return None
        if self.package_manager == "apt":
            return "sudo apt update && sudo apt install -y " + " ".join(pkgs)
        return "sudo dnf install -y " + " ".join(pkgs)


def read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    data = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value.strip().strip('"')
    return data


def detect_package_manager() -> str | None:
    for pm in ("dnf", "apt"):
        if which(pm):
            return pm
    return None


def query_gpu() -> GpuInfo | None:
    """The first NVIDIA GPU as nvidia-smi reports it, or None when there is no working driver."""
    if not which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
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
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def check(device: str = "gpu") -> SystemReport:
    """System packages needed in the given mode ("gpu" or "cpu"), the GPU if any, and the RAM."""
    osr = read_os_release()
    report = SystemReport(distro=osr.get("PRETTY_NAME", "unknown Linux"), package_manager=detect_package_manager())
    for req in REQUIREMENTS:
        if device == "cpu" and req in GPU_ONLY:
            continue
        if req.startswith("lib:"):
            if not ctypes.util.find_library(req[4:]):
                report.missing.append(req)
        elif not which(req):
            report.missing.append(req)
    report.gpu = query_gpu()
    report.ram_gib = ram_gib()
    return report

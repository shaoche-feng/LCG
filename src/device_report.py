"""Compute and rendering devices for the exploration startup banner.

Physical GPU numbers are nvidia-smi's indexes, matched by UUID; a process-local device such as
cuda:0 is never taken as a physical index (CUDA_VISIBLE_DEVICES renumbers devices). Rendering
runs in the env worker processes; for EGL a subprocess with this process's environment asks the
EGL device that dm_control selects (MUJOCO_EGL_DEVICE_ID) for its CUDA device and that device
for its UUID, without creating a GL or CUDA context. Whatever cannot be verified is reported as
configured."""
import json
import os
import subprocess
import sys
from typing import Dict, List, Mapping, Optional

import torch

# Run in a subprocess (same environment as the env workers): EGL device -> CUDA ordinal
# (EGL_NV_device_cuda, process-local numbering) -> UUID (CUDA driver API, no context).
EGL_PROBE = r"""
import ctypes, json, os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from dm_control._render.pyopengl import egl_ext as EGL
out = {}
devices = EGL.eglQueryDevicesEXT()
out["num_devices"] = len(devices)
index = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
attrib = ctypes.CFUNCTYPE(EGL.EGLBoolean, EGL.EGLDeviceEXT, EGL.EGLint, ctypes.POINTER(ctypes.c_ssize_t))(
    EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
ordinal = ctypes.c_ssize_t(-1)
if 0 <= index < len(devices) and attrib(devices[index], 0x323A, ctypes.byref(ordinal)) and ordinal.value >= 0:  # EGL_CUDA_DEVICE_NV
    cuda = ctypes.CDLL("libcuda.so.1")
    dev, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    if cuda.cuInit(0) == 0 and cuda.cuDeviceGet(ctypes.byref(dev), int(ordinal.value)) == 0 and cuda.cuDeviceGetUuid(raw, dev) == 0:
        h = bytes(raw).hex()
        out["uuid"] = f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
print(json.dumps(out))
"""


def nvidia_smi_gpus() -> Optional[List[Dict[str, str]]]:
    """Physical GPUs (nvidia-smi index, UUID, name), or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        index, uuid, name = [part.strip() for part in line.split(",", 2)]
        gpus.append(dict(index=index, uuid=uuid, name=name))
    return gpus


def cuda_uuid(device: torch.device) -> Optional[str]:
    uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if uuid is None:
        return None
    uuid = str(uuid)
    return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"


def physical_index(uuid: Optional[str], gpus: Optional[List[Dict[str, str]]]) -> Optional[str]:
    matches = [g["index"] for g in gpus or [] if uuid is not None and g["uuid"].lower() == uuid.lower()]
    return matches[0] if len(matches) == 1 else None


def physical_gpu_index(device: torch.device) -> Optional[int]:
    """nvidia-smi index of this process's CUDA device (matched by UUID); None on CPU or if unverified."""
    if device.type != "cuda":
        return None
    index = physical_index(cuda_uuid(device), nvidia_smi_gpus())
    return None if index is None else int(index)


def compute_device_line(device: torch.device, name: Optional[str], uuid: Optional[str],
                        gpus: Optional[List[Dict[str, str]]]) -> str:
    if device.type != "cuda":
        return "Compute device: CPU"
    index = physical_index(uuid, gpus)
    where = f"physical GPU {index}" if index is not None else \
        f"UUID {uuid or 'unknown'} (physical index not verified)"
    return f"Compute device: {name} | {where} | process {device}"


def probe_egl(env: Mapping[str, str]) -> Optional[Dict]:
    """Run EGL_PROBE with this environment; None if it fails."""
    try:
        out = subprocess.run([sys.executable, "-c", EGL_PROBE], capture_output=True, text=True, timeout=60,
                             env=dict(env), check=True).stdout
        return json.loads(out.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def rendering_device_line(env: Mapping[str, str], probe: Optional[Dict],
                          gpus: Optional[List[Dict[str, str]]]) -> str:
    backend = env.get("MUJOCO_GL")
    if backend is None:
        return "Rendering device: MuJoCo default backend (MUJOCO_GL not set; device not verified)"
    backend = backend.lower()
    if backend == "osmesa":
        return "Rendering device: OSMesa | CPU (software rendering)"
    if backend != "egl":
        return f"Rendering device: {backend.upper()} | default display device (configured; physical GPU not verified)"
    device_id = env.get("MUJOCO_EGL_DEVICE_ID")
    if device_id is None:
        return "Rendering device: EGL | first usable EGL device (configured; physical GPU not verified)"
    index = physical_index((probe or {}).get("uuid"), gpus)
    if index is not None:
        return f"Rendering device: EGL | physical GPU {index}"
    if probe and probe.get("uuid"):
        return f"Rendering device: EGL | EGL device {device_id} = UUID {probe['uuid']} (physical index not verified)"
    return f"Rendering device: EGL | EGL device {device_id} (configured; physical GPU not verified)"


def device_lines(device: torch.device, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """The two banner lines for this process (rank 0) and its env workers."""
    env = os.environ if env is None else env
    gpus = nvidia_smi_gpus() if device.type == "cuda" or env.get("MUJOCO_GL", "").lower() == "egl" else None
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    uuid = cuda_uuid(device) if device.type == "cuda" else None
    egl = env.get("MUJOCO_GL", "").lower() == "egl" and env.get("MUJOCO_EGL_DEVICE_ID") is not None
    probe = probe_egl(env) if egl else None
    return [compute_device_line(device, name, uuid, gpus), rendering_device_line(env, probe, gpus)]

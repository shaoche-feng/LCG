"""Startup-banner device lines (physical vs process-local numbering, verified vs configured
rendering, CPU) and the progress-bar rate text."""
import subprocess

import pytest
import torch

import device_report
from device_report import compute_device_line, device_lines, rendering_device_line
from utils import rate_text

GPUS = [dict(index="0", uuid="GPU-c92ce703-bfdc-65db-41a9-592f149e455f", name="NVIDIA GeForce RTX 5090"),
        dict(index="1", uuid="GPU-923e4182-b64b-82be-7913-2bcf8993a3dd", name="NVIDIA GeForce RTX 5090")]
GPU1 = GPUS[1]["uuid"]
CUDA0 = torch.device("cuda", 0)


def test_compute_device_maps_process_local_cuda_to_the_physical_gpu_by_uuid():
    # cuda:0 in this process is physical GPU 1 (CUDA_VISIBLE_DEVICES hides GPU 0).
    assert compute_device_line(CUDA0, "NVIDIA GeForce RTX 5090", GPU1, GPUS) == \
        "Compute device: NVIDIA GeForce RTX 5090 | physical GPU 1 | process cuda:0"
    # Never inferred from the process index: unknown UUID or no nvidia-smi -> the UUID, unverified.
    for gpus in (None, [], GPUS[:1]):
        assert compute_device_line(CUDA0, "RTX", GPU1, gpus) == \
            f"Compute device: RTX | UUID {GPU1} (physical index not verified) | process cuda:0"
    assert compute_device_line(CUDA0, "RTX", None, GPUS) == \
        "Compute device: RTX | UUID unknown (physical index not verified) | process cuda:0"
    assert compute_device_line(torch.device("cpu"), None, None, None) == "Compute device: CPU"


@pytest.mark.parametrize("env,probe,expected", [
    (dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="1"), dict(uuid=GPU1), "EGL | physical GPU 1"),
    (dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="1"), None, "EGL | EGL device 1 (configured; physical GPU not verified)"),
    (dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0"), dict(num_devices=5),  # hidden from CUDA: no mapping
     "EGL | EGL device 0 (configured; physical GPU not verified)"),
    (dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="1"), dict(uuid="GPU-other"),
     "EGL | EGL device 1 = UUID GPU-other (physical index not verified)"),
    (dict(MUJOCO_GL="egl"), None, "EGL | first usable EGL device (configured; physical GPU not verified)"),
    (dict(MUJOCO_GL="osmesa"), None, "OSMesa | CPU (software rendering)"),
    (dict(MUJOCO_GL="glfw"), None, "GLFW | default display device (configured; physical GPU not verified)"),
    ({}, None, "MuJoCo default backend (MUJOCO_GL not set; device not verified)"),
])
def test_rendering_device_is_verified_only_through_the_egl_device_uuid(env, probe, expected):
    assert rendering_device_line(env, probe, GPUS) == f"Rendering device: {expected}"


def test_device_lines_cpu_run_queries_nothing(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("no nvidia-smi or probe subprocess on a CPU run without EGL")
    monkeypatch.setattr(subprocess, "run", fail)
    assert device_lines(torch.device("cpu"), {}) == [
        "Compute device: CPU", "Rendering device: MuJoCo default backend (MUJOCO_GL not set; device not verified)"]


def test_device_lines_gpu_run_wiring(monkeypatch):
    env = dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="1", CUDA_VISIBLE_DEVICES=GPU1)
    probed = []
    monkeypatch.setattr(device_report, "nvidia_smi_gpus", lambda: GPUS)
    monkeypatch.setattr(device_report, "cuda_uuid", lambda device: GPU1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(device_report, "probe_egl", lambda e: probed.append(dict(e)) or dict(uuid=GPU1))
    assert device_lines(CUDA0, env) == ["Compute device: NVIDIA GeForce RTX 5090 | physical GPU 1 | process cuda:0",
                                        "Rendering device: EGL | physical GPU 1"]
    assert probed == [env]  # the probe sees the env workers' environment


def test_failed_probe_or_nvidia_smi_means_unverified(monkeypatch):
    def broken(*args, **kwargs):
        raise subprocess.TimeoutExpired("x", 1)
    monkeypatch.setattr(subprocess, "run", broken)
    assert device_report.nvidia_smi_gpus() is None
    assert device_report.probe_egl(dict(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="1")) is None


def test_rate_text_has_a_space_and_inverts_below_one_per_second():
    assert rate_text(dict(rate=295.552, n=300, elapsed=1.0), "steps") == "295.55 steps/s"
    assert rate_text(dict(rate=1 / 3.85, n=2, elapsed=7.7), "updates") == "3.85 s/update"
    assert rate_text(dict(rate=45.02, n=320, elapsed=7.1), "trans") == "45.02 trans/s"
    assert rate_text(dict(rate=1 / 2.5, n=1, elapsed=2.5), "ep") == "2.50 s/ep"
    assert rate_text(dict(rate=None, n=0, elapsed=0), "ep") == "? ep/s"
    assert rate_text(dict(rate=None, n=4, elapsed=2.0, initial=0), "steps") == "2.00 steps/s"  # tqdm's fallback

#!/usr/bin/env python3
"""Read-only check of the active Linux training environment."""
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from m1_locomotion.launch import checked_checkout

result = {"platform": platform.platform(), "python": sys.version, "errors": []}
if platform.system() != "Linux":
    result["errors"].append("Training target must be Linux with NVIDIA GPU; use CPU audit on this host")
try:
    result["robot_lab"] = str(checked_checkout())
except Exception as exc:
    result["errors"].append(str(exc))
# Isaac Lab v2.3.2 uses Python extension package version 0.54.2.
for package, expected in (("isaaclab", "0.54.2"), ("isaacsim", "5.1.0"), ("rsl-rl-lib", "3.1.2")):
    try:
        found = metadata.version(package)
        result[package] = found
        if found != expected:
            result["errors"].append(f"Expected {package}=={expected}, found {found}")
    except metadata.PackageNotFoundError:
        result["errors"].append(f"Missing {package}=={expected}")
try:
    result["gpus"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    import torch
    result["torch"] = torch.__version__
    result["cuda"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        result["errors"].append("PyTorch CUDA is unavailable")
    else:
        # Actual GPU kernel, not just device enumeration (relevant for newer GPU architectures).
        matrix = torch.ones((32, 32), device="cuda")
        assert (matrix @ matrix).sum().item() == 32768.0
        torch.cuda.synchronize()
except Exception as exc:
    result["errors"].append(str(exc))
print(json.dumps(result, indent=2))
sys.exit(bool(result["errors"]))

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
PIP_CMD=("$PYTHON_BIN" -m pip)
WHEEL_PAGE="https://data.pyg.org/whl/torch-2.10.0+cu128.html"
REQUIRED_PACKAGES=(torch_scatter torch_sparse torch_cluster torch_spline_conv)
OPTIONAL_PACKAGES=(pyg_lib)

printf '== Python / torch environment ==\n'
"$PYTHON_BIN" - <<'PY'
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
expected_torch = "2.10.0"
expected_cuda = "12.8"
actual_torch = torch.__version__.split("+", 1)[0]
if actual_torch != expected_torch or torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"Expected torch {expected_torch} with CUDA {expected_cuda}, got torch={torch.__version__} cuda={torch.version.cuda}. Activate the target environment before running this script."
    )
PY

printf '\n== Removing stale PyG native extensions ==\n'
"${PIP_CMD[@]}" uninstall -y pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv || true

printf '\n== Installing torch_geometric and matching extension wheels ==\n'
"${PIP_CMD[@]}" install --no-cache-dir --upgrade torch_geometric
"${PIP_CMD[@]}" install --no-cache-dir --force-reinstall -f "$WHEEL_PAGE" "${REQUIRED_PACKAGES[@]}"

printf '\n== Checking whether pyg_lib is published for this torch/CUDA combo ==\n'
if "$PYTHON_BIN" - <<'PY'
import urllib.request
url = "https://data.pyg.org/whl/torch-2.10.0+cu128.html"
with urllib.request.urlopen(url, timeout=30) as response:
    html = response.read().decode("utf-8", errors="replace")
raise SystemExit(0 if "pyg_lib" in html else 1)
PY
then
  printf 'pyg_lib wheel found on %s; installing it.\n' "$WHEEL_PAGE"
  "${PIP_CMD[@]}" install --no-cache-dir --force-reinstall -f "$WHEEL_PAGE" "${OPTIONAL_PACKAGES[@]}"
else
  printf 'pyg_lib wheel not published on %s; skipping it (repo does not require it).\n' "$WHEEL_PAGE"
fi

printf '\n== Import smoke test ==\n'
"$PYTHON_BIN" - <<'PY'
import importlib
import sys
import torch
modules = ["torch_scatter", "torch_cluster", "torch_sparse", "torch_spline_conv"]
for name in modules:
    module = importlib.import_module(name)
    print(f"{name}: OK version={getattr(module, '__version__', 'n/a')} file={getattr(module, '__file__', 'n/a')}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
PY

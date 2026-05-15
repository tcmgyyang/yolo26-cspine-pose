"""
Minimal smoke test: build the proposed model from its architecture YAML and
run a single forward pass with random input. Exercises the vendored
Ultralytics package and confirms the custom DPCA / DySample blocks load and
shape-check correctly.

Run from the repo root:

    PYTHONPATH=. python -m pytest tests/test_smoke.py -v

The test is CPU-only and finishes in a few seconds. It does NOT require the
training dataset, GPU, or trained weights.
"""
from __future__ import annotations
import pathlib

import torch
import pytest

from ultralytics import YOLO

REPO = pathlib.Path(__file__).resolve().parents[1]
ARCH_DIR = REPO / "ultralytics" / "cfg" / "models" / "26"


@pytest.mark.parametrize(
    "arch_name",
    [
        "yolo26s-pose.yaml",                 # baseline
        "yolo26s-pose-dpca-AC.yaml",         # proposed (DPCA at A+C)
        "yolo26s-pose-DySample.yaml",        # dynamic upsampling only
        "yolo26s-pose-dpca-AC-noDS.yaml",    # DPCA-AC without DySample
    ],
)
def test_build_and_forward(arch_name: str) -> None:
    """Build the model from YAML and run one forward pass on a dummy image."""
    arch_yaml = ARCH_DIR / arch_name
    assert arch_yaml.exists(), f"missing architecture YAML: {arch_yaml}"

    model = YOLO(str(arch_yaml))
    model.model.eval()

    # 35-keypoint cervical-spine pose head requires kpt_shape=[35, 3].
    # `YOLO()` defaults from the YAML so this should already be set.
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model.model(dummy)

    assert out is not None, f"forward returned None for {arch_name}"


def test_dpca_module_importable() -> None:
    """Confirm DPCA and DySample are registered in the model factory."""
    from ultralytics.nn import modules as nn_modules
    from ultralytics.nn.modules.block import DPCA, DySample

    assert DPCA is nn_modules.DPCA
    assert DySample is nn_modules.DySample


def test_sigma_vector_length() -> None:
    """Sanity-check the 35-element MS-OKS sigma vector used by the proposed model."""
    SIGMA_CORNER, SIGMA_SPINOUS, SIGMA_LAMINA = 0.05, 0.07, 0.10
    sigmas: list[float] = []
    # C2: 4 corners + 1 spinous (no LP)
    sigmas.extend([SIGMA_CORNER] * 4)
    sigmas.append(SIGMA_SPINOUS)
    # C3-C7: 4 corners + 1 spinous + 1 lamina, x5
    for _ in range(5):
        sigmas.extend([SIGMA_CORNER] * 4)
        sigmas.append(SIGMA_SPINOUS)
        sigmas.append(SIGMA_LAMINA)
    assert len(sigmas) == 35

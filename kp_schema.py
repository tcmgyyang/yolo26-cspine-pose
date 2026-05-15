"""
Keypoint schema for the 35-landmark cervical spine YOLO-pose model.

Order (user manuscript §2.3 + table; matches YOLO-pose flat index 0..34):
    C2: AS, AI, PS, PI, SP                (5 points, NO LP)
    C3: AS, AI, PS, PI, SP, LP
    C4: AS, AI, PS, PI, SP, LP
    C5: AS, AI, PS, PI, SP, LP
    C6: AS, AI, PS, PI, SP, LP
    C7: AS, AI, PS, PI, SP, LP
Total = 5 + 5 * 6 = 35.

AS = Anterior-Superior  (anterosuperior corner of the vertebral body)
AI = Anterior-Inferior  (anteroinferior corner of the vertebral body)
PS = Posterior-Superior (posterosuperior corner of the vertebral body)
PI = Posterior-Inferior (posteroinferior corner of the vertebral body)
SP = Spinous Process tip
LP = Lamina Point       (anterior edge of the lamina / spinolaminar line)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

VERTEBRAE: List[str] = ["C2", "C3", "C4", "C5", "C6", "C7"]

# display order within each vertebra (also the x-axis order inside a group)
PER_VERTEBRA: Dict[str, List[str]] = {
    "C2": ["AS", "AI", "PS", "PI", "SP"],              # LP absent
    "C3": ["AS", "AI", "PS", "PI", "SP", "LP"],
    "C4": ["AS", "AI", "PS", "PI", "SP", "LP"],
    "C5": ["AS", "AI", "PS", "PI", "SP", "LP"],
    "C6": ["AS", "AI", "PS", "PI", "SP", "LP"],
    "C7": ["AS", "AI", "PS", "PI", "SP", "LP"],
}

# flat list in YOLO-pose index order (0..34)
FLAT_KP: List[Tuple[str, str]] = [(v, p) for v in VERTEBRAE for p in PER_VERTEBRA[v]]
assert len(FLAT_KP) == 35, f"Expected 35 keypoints, got {len(FLAT_KP)}"
FLAT_IDX: Dict[Tuple[str, str], int] = {vp: i for i, vp in enumerate(FLAT_KP)}

# reference-style pastel palette, one hue per vertebra
VERTEBRA_COLORS: Dict[str, Dict[str, str]] = {
    "C2": {"fill": "#F6C5C5", "edge": "#C62828"},      # salmon / red
    "C3": {"fill": "#C5E1F4", "edge": "#1976D2"},      # sky / blue
    "C4": {"fill": "#CBEBC9", "edge": "#388E3C"},      # mint / green
    "C5": {"fill": "#E1C5EB", "edge": "#7B1FA2"},      # lavender / purple
    "C6": {"fill": "#FFF0B3", "edge": "#F9A825"},      # pale yellow / amber
    "C7": {"fill": "#DCECB8", "edge": "#689F38"},      # chartreuse / lime
}


@dataclass
class KPRecord:
    image_id: str
    vertebra: str
    keypoint: str
    err_px: float
    err_mm: float
    pred_x: float = float("nan")
    pred_y: float = float("nan")
    gt_x:   float = float("nan")
    gt_y:   float = float("nan")
    mm_per_px: float = float("nan")

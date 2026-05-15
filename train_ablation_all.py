"""
YOLO26s-pose ablation experiments — progressive-addition design.

Three novel components: DPCA, DySample (dynamic upsampling), MS-OKS loss.
Already covered elsewhere: Baseline / +DPCA+DySample (no MS-OKS) / Full model.
This script trains the remaining 5 combinations.
"""
from ultralytics import YOLO
import torch

# ==================== MS-OKS sigma configuration ====================
SIGMA_CORNER  = 0.05   # vertebra corners (AS/AI/PS/PI)
SIGMA_SPINOUS = 0.07   # spinous-process tip (SP)
SIGMA_LAMINA  = 0.10   # lamina point (LP)

def get_cervical_sigmas(device='cpu'):
    """Build the 35-element sigma vector for the cervical-spine schema."""
    sigmas = []
    # C2: AS, AI, PS, PI, SP (5 points, no LP)
    sigmas.extend([SIGMA_CORNER] * 4)
    sigmas.append(SIGMA_SPINOUS)
    # C3-C7: AS, AI, PS, PI, SP, LP (6 points each)
    for _ in range(5):
        sigmas.extend([SIGMA_CORNER] * 4)
        sigmas.append(SIGMA_SPINOUS)
        sigmas.append(SIGMA_LAMINA)
    return torch.tensor(sigmas, dtype=torch.float32, device=device)

def on_train_start(trainer):
    """Trainer callback that swaps in the custom sigma vector at train start."""
    if hasattr(trainer, 'loss') and hasattr(trainer.loss, 'keypoint_loss'):
        device = trainer.loss.keypoint_loss.sigmas.device
        trainer.loss.keypoint_loss.sigmas = get_cervical_sigmas(device)
        print(f"\n[MS-OKS] sigmas injected  "
              f"corner={SIGMA_CORNER}  spinous={SIGMA_SPINOUS}  lamina={SIGMA_LAMINA}\n")

# ==================== Architecture YAMLs ====================
YAML_DIR = 'ultralytics/cfg/models/26'
YAMLS = {
    'baseline':       f'{YAML_DIR}/yolo26s-pose.yaml',                # no DPCA, no DySample
    'dpca':           f'{YAML_DIR}/yolo26s-pose-dpca-AC-noDS.yaml',   # +DPCA, no DySample
    'dysample':       f'{YAML_DIR}/yolo26s-pose-DySample.yaml',       # no DPCA, +DySample
    'dpca_dysample':  f'{YAML_DIR}/yolo26s-pose-dpca-AC.yaml',        # +DPCA, +DySample
}

# ==================== Ablation matrix ====================
# Already covered: Baseline(x/x/x) / +DPCA+DySample(v/v/x) / Full(v/v/v)
# Remaining 5 combinations:
EXPERIMENTS = [
    # ---- Single component ----
    ('dpca',          False, 'ablation_DPCA_only'),
    ('dysample',      False, 'ablation_DySample_only'),
    ('baseline',      True,  'ablation_MSOKS_only'),

    # ---- Pairwise combinations ----
    ('dpca',           True,  'ablation_DPCA_MSOKS'),
    ('dysample',       True,  'ablation_DySample_MSOKS'),
]

# ==================== Common training parameters ====================
COMMON_PARAMS = dict(
    data='dataset/data.yaml',
    imgsz=1280,
    epochs=600,
    batch=16,
    device='0,1,2,3',
    workers=12,
    optimizer='AdamW',
    lr0=0.0005,
    lrf=0.005,
    weight_decay=0.0005,
    pose=30.0,
    kobj=5.0,
    box=2.0,
    degrees=5.0,
    scale=0.3,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=1.0,
    mixup=0.05,
    close_mosaic=150,
    val=True,
    amp=False,
    label_smoothing=0.0,
    patience=0,
    project='spine_extreme',
)


# ==================== Main ====================
if __name__ == '__main__':
    total = len(EXPERIMENTS)

    print("=" * 70)
    print("YOLO26s-pose ablation experiments (progressive addition)")
    print("=" * 70)
    print(f"Novel components: DPCA / DySample / MS-OKS Loss")
    print(f"Experiments to run: {total}")
    print(f"Already covered: Baseline / +DPCA+DySample / Full model")
    print("-" * 70)

    print(f"{'#':<4} {'Name':<35} {'DPCA':<6} {'DySample':<10} {'MS-OKS':<8}")
    print("-" * 70)
    for i, (yaml_key, use_msoks, name) in enumerate(EXPERIMENTS, 1):
        has_dpca = 'dpca' in yaml_key
        has_ds = 'dysample' in yaml_key
        print(f"{i:<4} {name:<35} {'Y' if has_dpca else 'N':<6} {'Y' if has_ds else 'N':<10} {'Y' if use_msoks else 'N':<8}")
    print("-" * 70)

    for i, (yaml_key, use_msoks, name) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{total}] {name}")
        print(f"  YAML:   {YAMLS[yaml_key]}")
        print(f"  MS-OKS: {'enabled' if use_msoks else 'disabled'}")
        print(f"{'='*70}")

        model = YOLO(YAMLS[yaml_key])

        if use_msoks:
            model.add_callback('on_train_start', on_train_start)

        model.train(
            **COMMON_PARAMS,
            name=name,
        )

        print(f"\n[OK] [{i}/{total}] {name} training complete")

    print(f"\n{'='*70}")
    print(f"All ablation experiments complete! ({total}/{total})")
    print(f"{'='*70}")

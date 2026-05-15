"""
YOLO26s-pose generalization experiments.

Experiment 1: clinical-cohort train/val  ->  public-dataset test (data2.yaml)
Experiment 2: public-dataset  train/val  ->  clinical-cohort test (data3.yaml)
"""
from ultralytics import YOLO
import torch

# ==================== MS-OKS sigma configuration ====================
SIGMA_CORNER  = 0.05
SIGMA_SPINOUS = 0.07
SIGMA_LAMINA  = 0.10

def get_cervical_sigmas(device='cpu'):
    sigmas = []
    sigmas.extend([SIGMA_CORNER] * 4)
    sigmas.append(SIGMA_SPINOUS)
    for _ in range(5):
        sigmas.extend([SIGMA_CORNER] * 4)
        sigmas.append(SIGMA_SPINOUS)
        sigmas.append(SIGMA_LAMINA)
    return torch.tensor(sigmas, dtype=torch.float32, device=device)

def on_train_start(trainer):
    if hasattr(trainer, 'loss') and hasattr(trainer.loss, 'keypoint_loss'):
        device = trainer.loss.keypoint_loss.sigmas.device
        trainer.loss.keypoint_loss.sigmas = get_cervical_sigmas(device)
        print(f"\n[MS-OKS] sigmas injected  "
              f"corner={SIGMA_CORNER}  spinous={SIGMA_SPINOUS}  lamina={SIGMA_LAMINA}\n")

# ==================== Experiment configuration ====================
MODEL_YAML = 'ultralytics/cfg/models/26/yolo26s-pose.yaml'

EXPERIMENTS = [
    {
        'data': 'dataset/data2.yaml',
        'name': 'generalization_clinical_train_public_test',
        'desc': 'clinical train/val  ->  public test',
    },
    {
        'data': 'dataset/data3.yaml',
        'name': 'generalization_public_train_clinical_test',
        'desc': 'public train/val  ->  clinical test',
    },
]

# ==================== Common training parameters ====================
COMMON_PARAMS = dict(
    imgsz=1280,
    epochs=600,
    batch=16,
    device='0',
    workers=18,
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
    print("YOLO26s-pose generalization experiments (full model)")
    print("=" * 70)
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['desc']}")
        print(f"     data: {exp['data']}")
    print("=" * 70)

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{total}] {exp['desc']}")
        print(f"  YAML: {MODEL_YAML}")
        print(f"  Data: {exp['data']}")
        print(f"{'='*70}")

        model = YOLO(MODEL_YAML)
        model.add_callback('on_train_start', on_train_start)

        model.train(
            **COMMON_PARAMS,
            data=exp['data'],
            name=exp['name'],
        )

        print(f"\n[OK] [{i}/{total}] {exp['name']} training complete")

    print(f"\n{'='*70}")
    print(f"All generalization experiments complete! ({total}/{total})")
    print(f"{'='*70}")

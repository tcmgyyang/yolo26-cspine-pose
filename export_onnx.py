import torch
import torch.nn as nn
from ultralytics import YOLO

class AdaptiveAvgPool2dPatch(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        oh, ow = self.output_size
        kh = max(1, h // oh)
        kw = max(1, w // ow)
        return nn.functional.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))

def patch_adaptive_pool(model):
    for name, module in model.named_children():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            out = module.output_size
            if isinstance(out, int):
                out = (out, out)
            print(f"  ✅ Patching [{name}]: AdaptiveAvgPool2d(output_size={out})")
            setattr(model, name, AdaptiveAvgPool2dPatch(out))
        else:
            patch_adaptive_pool(module)

print("Loading model...")
model = YOLO('runs/pose/spine_extreme/DPCA/DPCA_AC_sigma/weights/best.pt')

print("Patching AdaptiveAvgPool2d layers...")
patch_adaptive_pool(model.model)

print("Exporting to ONNX (opset=17)...")
model.export(
    format='onnx',
    opset=17,
    simplify=True,
    imgsz=640,
)
print("🎉 Export complete! Saved as best.onnx")

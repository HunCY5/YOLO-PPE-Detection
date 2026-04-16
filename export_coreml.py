"""
PPE Detection - CoreML Export Script
Converts best.pt (PyTorch/YOLO) to CoreML (.mlpackage) for iOS deployment.

Requirements:
    pip install ultralytics coremltools onnx

Usage:
    python export_coreml.py --weights runs/ppe_yolo11n_832/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def export(weights: str, imgsz: int = 640, nms: bool = True, half: bool = False):
    """
    Export YOLO model to CoreML format.

    Args:
        weights: Path to best.pt
        imgsz:   Input size for CoreML model (fixed at export time).
                 Training used 832, but 640 is recommended for mobile inference speed.
        nms:     Include NMS inside the model.
                 True  → Vision/CoreML handles suppression, simpler app code.
                 False → App must implement NMS manually.
        half:    FP16 quantization. Reduces model size and speeds up inference.
                 Verify on real device before shipping — may cause precision issues
                 on older hardware.
    """
    model = YOLO(weights)

    model.export(
        format="coreml",
        imgsz=imgsz,
        nms=nms,
        half=half,
    )
    print(f"\nExport complete.")
    print(f"Output: {weights.replace('.pt', '.mlpackage')}")
    print("\nNext steps:")
    print("  1. Drag .mlpackage into your Xcode project")
    print("  2. Verify input size matches: imgsz={imgsz}")
    print("  3. Check class order matches data.yaml")
    print("  4. Handle Vision coordinate transform (Y-axis flip for UIKit)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/ppe_yolo11n_832/weights/best.pt",
        help="Path to best.pt"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--nms", action="store_true", default=True)
    parser.add_argument("--half", action="store_true", default=False)
    args = parser.parse_args()

    export(
        weights=args.weights,
        imgsz=args.imgsz,
        nms=args.nms,
        half=args.half,
    )

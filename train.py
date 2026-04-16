"""
PPE Detection - YOLOv11n Training Script
Model: yolo11n.pt (YOLOv11 nano)
Target: iOS real-time inference via CoreML
"""

from ultralytics import YOLO

def train():
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=100,
        batch=10,
        imgsz=832,
        # Scheduler
        optimizer="auto",
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Regularization
        weight_decay=0.0005,
        # Mosaic
        mosaic=1.0,
        close_mosaic=10,      # disable mosaic last 10 epochs for stable convergence
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.05,
        scale=0.5,
        shear=2.0,
        erasing=0.4,
        auto_augment="randaugment",
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Validation
        iou=0.7,
        patience=100,
        # Misc
        cache=True,
        device="0",           # GPU
        workers=6,
        amp=True,
        plots=True,
        project="runs",
        name="ppe_yolo11n_832",
    )

if __name__ == "__main__":
    train()

# Weights

Trained model weights are not stored directly in this repository due to file size.

## Download

| File | Description | Link |
|---|---|---|
| `best.pt` | YOLOv11n trained weights (PyTorch) | [GitHub Releases](../../releases) |
| `DetectionYolov11.mlpackage` | CoreML export for iOS | [GitHub Releases](../../releases) |

## Training info

- Model: YOLOv11n
- Epochs: 100
- imgsz: 832
- mAP50: 0.706
- See `results/` for training curves and confusion matrix.

## Re-export from best.pt

```bash
python export_coreml.py --weights best.pt --imgsz 640 --nms
```

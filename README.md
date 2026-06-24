# Benchmarking Object Detection Architectures for Road Anomaly Perception Under Adverse Conditions

**A Luminance-Adaptive Preprocessing Framework for Edge-Deployed Autonomous Vehicles**


---

## Overview

This repository contains all code, evaluation scripts, and results for a systematic benchmarking study evaluating four object detection architectures under physically grounded adverse lighting and weather conditions, with a proposed luminance-adaptive SCI preprocessing pipeline validated on NVIDIA Jetson Nano edge hardware.

**Models benchmarked:** YOLOv8n, YOLOv12n, RT-DETR, Faster R-CNN ResNet-50 FPN

**Conditions tested:** 3 lighting levels (dusk, night, severe) + 3 weather conditions (light rain, heavy rain, Harmattan fog)

**Key finding:** Uniform SCI enhancement is harmful at moderate lighting conditions (up to -30.7% mAP@50). A luminance-adaptive router recovers most of this penalty by selectively applying the appropriate enhancement branch per image.

---

## Repository Structure

```
road-anomaly-detection-benchmark/
├── data/                        # Dataset (gitignored — download separately)
│   └── README_data_sources.md   # Download instructions
├── degradation/                 # Physically grounded degradation pipeline
│   ├── lighting_pipeline.py     # Gamma + signal-dependent noise simulation
│   ├── weather_pipeline.py      # Rain streaks + Koschmieder fog
│   └── sanity_check.py          # Visual preview before full generation
├── training/                    # Training notebooks (Kaggle)
│   ├── train_yolov8n.ipynb
│   ├── train_yolov12n.ipynb
│   ├── train_rtdetr.ipynb
│   └── train_fasterrcnn.ipynb
├── evaluation/                  # Stress test evaluation scripts
│   ├── eval_yolo_rtdetr.py      # Ultralytics-based evaluation loop
│   ├── eval_fasterrcnn.py       # torchvision-based evaluation loop
│   └── run_all_evals.sh         # Convenience script to run all models
├── sci/                         # SCI enhancement module
│   ├── sci_model.py             # Inline architecture (no repo dependency)
│   ├── enhance_lighting.py      # Batch enhancement of lighting datasets
│   ├── adaptive_router.py       # Luminance-adaptive preprocessing system
│   └── weights/                 # SCI pretrained weights (download separately)
├── jetson/                      # Edge deployment
│   ├── latency_benchmark.py     # Per-branch FPS benchmark for Jetson Nano
│   └── setup_jetson.sh          # Environment setup script for Jetson
├── results/                     # All CSV results (committed)
│   ├── baseline_lighting/       # No-preprocessing lighting results
│   ├── baseline_weather/        # No-preprocessing weather results
│   ├── sci_lighting/            # Post-SCI lighting results
│   ├── adaptive/                # Adaptive router results
│   └── jetson_latency/          # Jetson Nano FPS benchmark
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
git clone https://github.com/obedhonoureje/road-anomaly-detection-benchmark.git
cd road-anomaly-detection-benchmark
pip install -r requirements.txt
```

### Dataset

Download and place under `data/`:
- [Pothole Detection Dataset](https://www.kaggle.com/datasets/rajdalsaniya/pothole-detection-dataset) (Rajdalsaniya, 2022)
- [speed-unmarked-bumb v3](https://universe.roboflow.com/pothole-detection-1nczj/speed-unmarked-bumb) (Roboflow Universe, 2023)

See `data/README_data_sources.md` for full instructions.

### SCI Weights

Download from the [official SCI repository](https://github.com/vis-opt-group/SCI):
```bash
# Place medium.pt and difficult.pt into sci/weights/
```

---

## Reproducing Results

### Step 1: Generate degraded datasets

```bash
# Lighting degradation
python degradation/lighting_pipeline.py

# Weather degradation
python degradation/weather_pipeline.py

# Visual sanity check (run before full generation)
python degradation/sanity_check.py
```

### Step 2: Run baseline evaluations

```bash
# YOLO and RT-DETR models
python evaluation/eval_yolo_rtdetr.py --model yolov8n --condition lighting
python evaluation/eval_yolo_rtdetr.py --model yolov12n --condition lighting
python evaluation/eval_yolo_rtdetr.py --model rtdetr --condition lighting

# Faster R-CNN
python evaluation/eval_fasterrcnn.py --condition lighting

# Or run everything at once
bash evaluation/run_all_evals.sh
```

### Step 3: Apply SCI and run adaptive router

```bash
# Generate SCI-enhanced lighting datasets
python sci/enhance_lighting.py

# Generate adaptive-routed datasets
python sci/adaptive_router.py

# Re-run evaluations on enhanced datasets
python evaluation/eval_yolo_rtdetr.py --model yolov8n --condition sci_lighting
```

### Step 4: Jetson Nano benchmark

Copy the `jetson/` folder and model weights to the Jetson, then:

```bash
python jetson/latency_benchmark.py
```

---

## Results Summary

### Baseline Lighting (mAP@50)

| Model       | Clean | Dusk  | Night | Severe |
|-------------|-------|-------|-------|--------|
| RT-DETR     | 0.883 | 0.861 | 0.612 | 0.324  |
| YOLOv12n    | 0.894 | 0.867 | 0.570 | 0.305  |
| YOLOv8n     | 0.881 | 0.855 | 0.590 | 0.268  |
| Faster R-CNN| 0.779 | 0.755 | 0.548 | 0.215  |

### Adaptive Router vs Baseline (mAP@50 delta)

| Model       | Dusk delta | Night delta | Severe delta |
|-------------|-----------|------------|-------------|
| RT-DETR     | -0.014    | +0.045     | +0.014      |
| YOLOv12n    | -0.038    | +0.007     | -0.011      |
| YOLOv8n     | -0.019    | -0.036     | -0.050      |
| Faster R-CNN| -0.068    | -0.141     | -0.052      |

Plain SCI (uniform) dusk delta for comparison: -0.139 to -0.232 across models.

### Jetson Nano FPS (adaptive pipeline)

| Model       | Branch A (Clean) | Branch B (Dusk) | Branch C (Severe) |
|-------------|-----------------|----------------|------------------|
| YOLOv8n     | 25.83           | 18.18          | 13.52            |
| YOLOv12n    | 18.54           | 14.97          | 11.44            |
| RT-DETR     | 8.23            | 7.65           | 6.69             |
| Faster R-CNN| 4.42            | 4.28           | 3.81             |

---

## Citation

If you use this code or results, please cite:

```
Eje, O. H. (2026). Benchmarking Object Detection Architectures for Road Anomaly
Perception Under Adverse Conditions: A Luminance-Adaptive Preprocessing Framework
for Edge-Deployed Autonomous Vehicles. FUTMinna.
```

SCI module from:
```
Ma, L., Ma, T., Liu, R., Fan, X., and Luo, Z. (2022). Toward Fast, Flexible,
and Robust Low-Light Image Enhancement. CVPR 2022.
https://github.com/vis-opt-group/SCI
```

# Multi-Modality Image Processing

Low-light enhancement system built with multi-modality fusion (Illumination + Structure), including GUI interaction, video enhancement, real-time webcam demos, YOLO detection comparison, and PSNR/SSIM evaluation.

## Features

- Interactive GUI sliders for contrast, detail, color, and denoising.
- Adaptive single-image enhancement pipeline.
- Video enhancement with side-by-side output.
- Real-time webcam enhancement.
- YOLO object detection comparison (original vs enhanced).
- Quantitative evaluation against a reference image (PSNR / SSIM).
- Experiment mode: fixed HE baseline vs adaptive method.

## Project Structure

```text
.
|-- main.py
|-- illumination.py
|-- structure.py
|-- fusion.py
|-- evaluation.py
|-- utils.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- data/
|   |-- samples/
|   |   |-- images/
|   |   |   |-- 1_2.png
|   |   |   |-- 5_teddybear.png
|   |   |   |-- input.jpg
|   |   |   |-- normal_light.jpg
|   |   |-- videos/
|   |   |   |-- 3.mp4
|   |-- low/
|       |-- *.png
|-- models/
|   |-- yolov8n.pt
|-- outputs/
|   |-- images/
|   |-- videos/
```

## Requirements

- Python 3.10+
- Windows / macOS / Linux (GUI modes require Tkinter + OpenCV window support)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the launcher:

```bash
python main.py
```

Available modes in the launcher:

1. Interactive GUI Adjustment
2. Single Image Enhancement (Adaptive)
3. Video Enhancement (File)
4. Quantitative Evaluation (PSNR/SSIM)
5. YOLO Object Detection Test
6. Webcam Enhancement Only
7. Webcam + YOLO
8. Adaptive vs Fixed Experiment

## Default Paths (after reorganization)

- Default test image: `data/samples/images/1_2.png`
- YOLO model: `models/yolov8n.pt`
- Evaluation reference image: `data/samples/images/normal_light.jpg`
- Generated images: `outputs/images/`
- Generated videos: `outputs/videos/`

## Standalone Evaluation

```bash
python evaluation.py
```

This will:

- Generate a synthetic low-light input from `normal_light.jpg`.
- Compare HE baseline vs fusion result using PSNR/SSIM.
- Save `eval_input.jpg`, `eval_ours.jpg`, `eval_he.jpg`, and `eval_gt.jpg` to `outputs/images/`.

## Notes

- YOLO-related modes require `ultralytics` and `models/yolov8n.pt`.
- Generated output files are covered by `.gitignore` and can be adjusted to match your release policy.
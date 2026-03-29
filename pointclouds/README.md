# 3D Point Cloud Processing and Privacy Protection System

## Overview

This project is part of my **Master's Thesis** on privacy-preserving 3D point cloud processing using Intel RealSense cameras. The system combines real-time point cloud capture, 2D/3D object detection, semantic segmentation, and privacy-enhancing effects to protect sensitive information in 3D spatial data.

### Core Capabilities
- **Real-time Point Cloud Capture**: Intel RealSense D435I camera integration(supports all cameras that realsense SDK supports)
- **2D Object Detection**: Face and object detection using OpenCV DNN (MobileNet-SSD)
- **3D Object Detection**: PointPillars from Open3D-ML pretrained on KITTI dataset for outdoor scenes
- **Semantic Segmentation**: RandLANet and KPFCNN from Open3D-ML on SemanticKITTI for point cloud labeling
- **Privacy Protection**: Automated face blurring in 2D and 3D point clouds

## Installation

### Prerequisites
- Python 3.10-3.12 (Open3D requires ≤3.12)
- Intel RealSense SDK 2.0 (for live camera capture)
- CUDA-capable GPU (optional, for faster 3D detection)

### Using uv (Recommended)
```bash
uv run main.py --app <application-to-run>
```
uv automatically installs dependencies no need to manually add them.


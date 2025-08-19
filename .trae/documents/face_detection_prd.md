# Face Detection & Recognition System - Product Requirements Document

## 1. Product Overview
A Python-based face detection and recognition system using template matching for detection and eigenfaces (PCA) with cosine similarity for recognition. The system processes video files to extract, train, and identify faces without relying on machine learning libraries.

- Core purpose: Detect faces from training videos, build PCA-based recognition models, and identify faces in test videos with visual annotations.
- Target users: Developers and researchers needing custom face recognition solutions with manual PCA implementation.
- Market value: Educational tool for understanding computer vision fundamentals and custom implementation of face recognition algorithms.

## 2. Core Features

### 2.1 User Roles
Single user system - no role distinction required.

### 2.2 Feature Module
Our face detection and recognition system consists of the following main components:
1. **Face Detection Module**: Video processing, face detection using template matching, face extraction and storage.
2. **Training Module**: PCA model training, eigenface generation, feature extraction without ML libraries.
3. **Recognition Module**: Face identification, similarity calculation, visual annotation with bounding boxes and labels.

### 2.3 Page Details
| Module Name | Component | Feature Description |
|-------------|-----------|--------------------|
| Face Detection | Video Processing | Load video files, extract frames, detect faces using template matching |
| Face Detection | Face Extraction | Crop detected faces, save to /faces directory, generate JSON metadata |
| Training | PCA Implementation | Manual PCA calculation, eigenface generation, feature vector extraction |
| Training | Model Storage | Save PCA model, eigenfaces, and training data for recognition |
| Recognition | Face Matching | Load test video, detect faces, compare with trained model using cosine similarity |
| Recognition | Visual Output | Draw red bounding boxes, add name labels, save annotated video |

## 3. Core Process

### Main Workflow
1. **Training Phase**: Process Joseph_Lai.mp4 → Extract faces → Train PCA model → Save model data
2. **Recognition Phase**: Process test.mp4 → Detect faces → Compare with trained model → Output annotated video

```mermaid
graph TD
    A[Joseph_Lai.mp4] --> B[detection.py]
    B --> C[/faces directory]
    B --> D[metadata.json]
    C --> E[train.py]
    E --> F[PCA Model]
    G[test.mp4] --> H[scan.py]
    F --> H
    H --> I[Annotated Video]
```

## 4. User Interface Design

### 4.1 Design Style
- **Primary colors**: Red (#FF0000) for bounding boxes, White (#FFFFFF) for text labels
- **Visual elements**: Rectangular bounding boxes with 2-pixel thickness
- **Font**: OpenCV default font (FONT_HERSHEY_SIMPLEX), size 0.7
- **Layout style**: Video overlay with top-left positioned labels
- **Output format**: MP4 video files with visual annotations

### 4.2 Page Design Overview
| Component | Visual Element | UI Details |
|-----------|----------------|------------|
| Face Detection | Bounding Box | Red rectangular border, 2px thickness, positioned around detected faces |
| Name Label | Text Overlay | White text on red background, positioned above bounding box |
| Video Output | Annotated Frame | Original video with overlaid detection results |

### 4.3 Responsiveness
Desktop-first application with video file processing capabilities. No touch interaction required - command-line interface for processing.
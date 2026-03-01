# 🎯 Yifer Face Classification — Colab GPU

## 📋 Overview

This project is a robust **Face Classification System** designed to run on **Google Colab** with GPU acceleration. It automates the process of identifying a specific person (referred to as "Yifer") in a collection of group photos.

The system leverages state-of-the-art Deep Learning models, specifically **ArcFace** for face recognition and a dynamic fallback mechanism for face detection (including **YOLOv8n**, **RetinaFace**, and **MTCNN**), ensuring high accuracy even in challenging conditions.

## 🚀 Key Features

-   **High Accuracy**: Uses **ArcFace**, one of the most accurate face recognition models available.
-   **Robust Detection**: Implements a **Multi-Detector Fallback** system (YOLOv8n -> RetinaFace -> MTCNN -> OpenCV -> SSD) to detect faces that single detectors might miss.
-   **Smart Matching**: Utilizes **L2-Normalized Matching** and a **Mean Centroid** approach to compare faces against a reference set, reducing false positives.
-   **GPU Accelerated**: Optimized for Google Colab's T4 GPU for fast processing.
-   **Automated Sorting**: Automatically sorts images into "Class 1" (Target Present) and "Class 2" (Target Absent) folders on Google Drive.
-   **Visual Reporting**: Generates comprehensive charts and visual grids to analyze classification performance.

## 🛠️ Prerequisites

To run this notebook, you need:

1.  **Google Account**: To access Google Colab and Google Drive.
2.  **Google Drive Structure**:
    Create the following folders in your Google Drive:
    -   `MyDrive/practice/yife/`: Upload reference photos of the target person here.
    -   `MyDrive/practice/group_photos/`: Upload the group photos you want to classify here.

## 📦 Dependencies

The project installs the following key libraries automatically:

-   `deepface`: For face recognition and analysis.
-   `tf-keras`: TensorFlow Keras interface.
-   `retina-face`: High-performance face detector.
-   `mtcnn`: Multi-task Cascaded Convolutional Networks for face detection.
-   `ultralytics`: For YOLOv8 face detection.
-   `scikit-learn`, `tqdm`, `matplotlib`, `Pillow`: For data processing and visualization.

## 📖 Usage Guide

1.  **Open in Colab**: Click the "Open in Colab" badge at the top of the notebook.
2.  **Select GPU Runtime**: Go to `Runtime` -> `Change runtime type` -> Select `T4 GPU`.
3.  **Run Steps 1-3**:
    -   Verify GPU availability.
    -   Install necessary libraries.
    -   Mount your Google Drive.
4.  **Configure (Step 4)**:
    -   Adjust paths if your Drive folder structure is different.
    -   Set the `THRESHOLD` (default is 0.65 for ArcFace). Lower is stricter, higher is more lenient.
5.  **Run the Pipeline**: Execute the remaining cells sequentially.
    -   **Preprocessing**: Images are resized and cached to the local Colab SSD for speed.
    -   **Diagnosis**: The system auto-selects the best face detector for your specific images.
    -   **Embedding Generation**: Creates a mathematical representation of the target face.
    -   **Classification**: Scans group photos and classifies them.
6.  **View Results**:
    -   Sorted images will be saved to `MyDrive/practice/class_1` and `MyDrive/practice/class_2`.
    -   A visual report `classification_report.png` will be saved to your Drive.

## 🧠 Methodology

### Face Detection
The system employs a fallback strategy. It attempts detection in the following order:
1.  **YOLOv8n**: Fast and accurate.
2.  **RetinaFace**: Highly accurate for small faces.
3.  **MTCNN**: Reliable standard.
4.  **OpenCV / SSD**: Fallback options.

### Face Recognition
-   **Model**: **ArcFace** (Additive Angular Margin Loss) is used to extract 512-dimensional face embeddings.
-   **Normalization**: All embeddings are L2-normalized to ensure consistent distance measurements.
-   **Matching Logic**: A face is considered a match if its Cosine Distance to the **Mean Centroid** of the reference faces OR to **any individual reference face** is below the set threshold.

## 📊 Outputs

-   **Class 1 Folder**: Contains images where the target person was found.
-   **Class 2 Folder**: Contains images where the target person was NOT found.
-   **PDF/PNG Report**: A distribution chart showing cosine distances and classification split.

## 📝 License

This project is open-source and available for educational and personal use.

---
*Created for efficient photo sorting using the power of Deep Learning.*

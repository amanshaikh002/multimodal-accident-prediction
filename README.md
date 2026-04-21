# Multimodal Vision Audio Framework for Workplace Accident Prediction

An advanced, AI-powered industrial safety system designed to monitor workplaces in real-time, enforce safety compliance, and proactively predict accidents. The framework unites computer vision and audio analysis through a modular pipeline, separating a high-performance Python (FastAPI) backend from a unified React dashboard.

---

## ⚙️ Architecture Overview

The system is built on a **Microservice-oriented** architecture to ensure scalability and real-time processing capabilities.

*   **Frontend Controls:** React (Vite/CRA) *(Integration Planned)*
*   **Backend Server:** FastAPI, Uvicorn
*   **Computer Vision Engine:** Ultralytics YOLOv8, OpenCV
*   **Machine Learning (Ergonomics):** Scikit-Learn, LightGBM/XGBoost
*   **Hardware Efficiency:** Automatic fallback architecture capable of deploying on CPUs or leveraging CUDA GPUs (`device=0`) for maximum framerates.

---

## 🚀 Implemented Modules

### 1. Personal Protective Equipment (PPE) Detection (`backend/`)
*Status: Fully Operational (Backend API)*

A high-speed vision pipeline ensuring workers are adhering to mandatory safety-gear protocols.

*   **Custom YOLOv8 Model:** Trained specifically to detect 5 localized classes: `human`, `helmet`, `vest`, `gloves`, and `boots`.
*   **Performance:** Achieves double inference speed by dynamically downscaling to 640x480 and applying a stride-processing algorithm (`FRAME_STRIDE=2`, processing 15fps effectively).
*   **Core Safety Logic:** Validates that if a `human` is present, both a `helmet` AND `vest` must be detected on the worker.
*   **Outputs:** 
    *   **JSON Compliance Report:** Returns a React-parseable summary detailing safe/unsafe frame counts, total compliance percentages, and an incident timeline mapping exact missing gear to frame numbers.
    *   **Visual Generation:** Outputs `ppe_annotated.mp4` rendering color-coded bounding boxes and high-visibility "SAFE" / "UNSAFE" banners.

### 2. Worker Pose Ergonomics & Safety Pipeline (Root Directory)
*Status: Established, preparing for FastAPI migration*

Analyzes skeletal mechanics to identify unsafe lifting postures and prevent musculoskeletal injuries.

*   **Keypoint Extraction:** Uses `yolov8n-pose.pt` / `yolov8s-pose.pt` to extract 17 key human skeletal markers, having fully migrated away from legacy MediaPipe implementations.
*   **Feature Engineering Engine:** Converts raw coordinate data into advanced vector features such as joint angles, normalized coordinates, keypoint velocities, and accelerations.
*   **Machine Learning Backend:** Employs an ML classifier (`model.pkl` - RandomForest/XGBoost) trained on fuzzy-labeled datasets (`dataset.csv`) to classify safe vs. unsafe postures.
*   **Real-time Alerts:** Features temporal stability algorithms for smooth inference and triggers context-aware Text-to-Speech (TTS) alerts when violations occur.

---

## 🔮 Planned Modules

### 3. Anomaly Sound Detection (Phase 3)
*Status: Stubs prepared in Backend*

An upcoming acoustic monitoring module designed to identify catastrophic industrial events (e.g., machinery crashes, high-pressure gas leaks, or emergency distress alarms) that may occur completely out of the camera's line of sight.

---

## 📂 Project Structure

```text
├── backend/                       # The FastAPI Backend System
│   ├── main.py                    # Root entry point, routers, CORS, and health checks
│   ├── models/
│   │   └── ppe_model.pt           # Custom trained YOLOv8 PPE weights
│   ├── routes/
│   │   └── ppe.py                 # POST /detect/ppe file upload and API handlers
│   ├── services/
│   │   └── ppe_service.py         # Hardware-accelerated YOLO inference and frame skipping
│   └── utils/
│       ├── ppe_utils.py           # Class mapping and SAFE/UNSAFE rule engine
│       └── video_utils.py         # OpenCV video chunking and file I/O
│
├── app_yolo.py                    # Core Pose Detection visualizer
├── auto_label.py                  # Fuzzy labeler for skeletal training data
├── dataset_generator_yolo.py      # Skeletal vector math and feature extraction
├── train_model.py                 # XGBoost/RF ML classifier training pipeline
└── yolov8n-pose.pt                # Base model for pose keypoints
```

---

## 💻 Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   *(Optional but Recommended)* NVIDIA GPU with CUDA drivers installed.

### 2. Installing Backend Dependencies
Navigate to the root project directory and install the necessary requirements:
```bash
cd backend
pip install -r requirements.txt
```

*(Note: Provide your custom trained `ppe_model.pt` inside the `backend/models/` folder prior to startup).*

### 3. Running the Server
Launch the FastAPI development server:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The server will boot and natively watch for hot-reload file changes. 

### 4. API Testing
Navigate to the interactive Swagger UI to test the endpoints in your browser:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

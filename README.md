# Multimodal Vision Audio Framework for Workplace Accident Prediction

An advanced, AI-powered industrial safety system designed to monitor workplaces in real-time, enforce safety compliance, and proactively predict accidents. The framework unites computer vision and audio analysis through a modular pipeline, separating a high-performance Python (FastAPI) backend from a unified React dashboard.

---

## ⚙️ Architecture Overview

The system is built on a **Microservice-oriented** architecture to ensure scalability and real-time processing capabilities.

*   **Frontend Controls:** React (Vite) Dashboard - Fully Integrated
*   **Backend Server:** FastAPI, Uvicorn (Modular Service Architecture)
*   **Computer Vision Engine:** Ultralytics YOLOv8, OpenCV
*   **Machine Learning (Ergonomics):** Scikit-Learn, XGBoost/RandomForest
*   **Hardware Efficiency:** Automatic fallback architecture capable of deploying on CPUs or leveraging CUDA GPUs (`device=0`) for maximum framerates.

---

## 🚀 Implemented Modules

### 1. Personal Protective Equipment (PPE) Detection (`backend/`)
*Status: Fully Operational (Backend API)*

A high-speed vision pipeline ensuring workers are adhering to mandatory safety-gear protocols.

*   **Custom YOLOv8 Model:** Trained specifically to detect 5 localized classes: `human`, `helmet`, `vest`, `gloves`, and `boots`.
*   **Performance:** Achieves double inference speed by dynamically downscaling to 640x480 and applying a stride-processing algorithm (`FRAME_STRIDE=2`, processing 15fps effectively).
*   **Core Safety Logic:** Person-centric safety compliance checking using precise center-point containment logic (ensuring gear belongs to the specific worker) rather than inaccurate area-based overlaps. Validates that if a `human` is present, both a `helmet` AND `vest` must be detected on that worker.
*   **Outputs:** 
    *   **JSON Compliance Report:** Returns a React-parseable summary detailing safe/unsafe frame counts, total compliance percentages, and an incident timeline.
    *   **Visual Generation:** Outputs video rendering with color-coded bounding boxes and professional, clean, non-cluttered visualizations.

### 2. Worker Pose Ergonomics & Safety Pipeline (`backend/services/pose_service.py`)
*Status: Fully Operational (Backend API & Unified Pipeline)*

Analyzes skeletal mechanics to identify unsafe lifting postures and prevent musculoskeletal injuries.

*   **Keypoint Extraction:** Uses `yolov8n-pose.pt` / `yolov8s-pose.pt` to extract 17 key human skeletal markers, having fully migrated away from legacy MediaPipe implementations.
*   **Feature Engineering Engine:** Converts raw coordinate data into advanced vector features such as joint angles, normalized coordinates, keypoint velocities, and accelerations.
*   **Machine Learning Backend:** Employs an ML classifier (`model.pkl` - RandomForest/XGBoost) trained on balanced, fuzzy-labeled datasets (`dataset.csv`) to classify safe vs. unsafe postures.
*   **Hybrid Classification Logic:** Uses ML-inference logic coupled with strict geometric rule-based overrides and confidence-based filtering to eliminate "UNSAFE" prediction bias and ensure reliable assessments.
*   **Real-time Alerts:** Features temporal stability algorithms for smooth inference and seamless React dashboard integration.

### 3. Combined Multimodal Safety Pipeline (`backend/services/combined_service.py`)
*Status: Fully Operational*

A unified analysis module running both PPE and Pose Ergonomics simultaneously.
*   **Synchronized Processing:** Runs object detection and pose estimation on the same frames.
*   **Unified Dashboard Visuals:** Delivers a singular video stream with combined safety metrics, color-coded bounding boxes, and joint-mapped skeletons without visual clutter.

---

## 🔮 Planned Modules

### 4. Anomaly Sound Detection (Phase 3)
*Status: Stubs prepared in Backend*

An upcoming acoustic monitoring module designed to identify catastrophic industrial events (e.g., machinery crashes, high-pressure gas leaks, or emergency distress alarms) that may occur completely out of the camera's line of sight.

---

## 📂 Project Structure

```text
├── frontend/                      # React (Vite) Professional Dashboard
│   ├── src/                       # React components (VideoPanel, Dashboard, etc.)
│   └── package.json               # Node dependencies
│
├── backend/                       # The Modular FastAPI Backend System
│   ├── main.py                    # Root entry point, routers, CORS, and health checks
│   ├── models/                    # Custom trained YOLOv8 weights and ML models
│   ├── routes/
│   │   ├── ppe.py                 # API handlers for PPE detection
│   │   ├── pose.py                # API handlers for Pose safety
│   │   └── detect.py              # Combined pipeline API handlers
│   ├── services/
│   │   ├── ppe_service.py         # Hardware-accelerated YOLO PPE inference
│   │   ├── pose_service.py        # ML-based pose feature extraction and classification
│   │   └── combined_service.py    # Unified multimodal safety engine
│   └── utils/                     # Video I/O, center-point geometric logic, and rules
│
├── ml/                            # Machine Learning & Dataset Pipeline
│   └── pose/                      # Pose training logic
│       ├── auto_label.py          # Fuzzy labeler for skeletal training data
│       ├── dataset_generator_yolo.py # Vector math and feature extraction
│       └── train_model.py         # XGBoost/RF ML classifier training pipeline
└── app_yolo.py                    # Legacy standalone Pose visualizer
```

---

## 💻 Setup & Installation

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

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

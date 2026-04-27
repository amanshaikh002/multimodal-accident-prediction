# System Architecture

This document is a complete architectural reference for the **Multimodal Vision Audio Framework for Workplace Accident Prediction**. It uses Mermaid diagrams that render natively on GitHub, GitLab, VS Code, and any Markdown previewer with Mermaid support. A plain ASCII overview is included up front for environments without Mermaid.

---

## 1. ASCII Top-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BROWSER (Client)                           │
│                                                                         │
│   React 19 + Vite Dashboard                                             │
│   ┌─────────────┬─────────────┬──────────────┬────────────────────┐     │
│   │ App.jsx     │ Per-mode    │ Voice alerts │ Annotated video    │     │
│   │ + Controls  │ Panels      │ (SpeechSynth)│ (HTML5 <video>)    │     │
│   └─────────────┴─────────────┴──────────────┴────────────────────┘     │
│             │                                                           │
│       fetch /detect?mode=...     GET /output/<file>.mp4                 │
└─────────────┼───────────────────────────────────────┼───────────────────┘
              │                                       │
              ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FASTAPI BACKEND (Uvicorn)                         │
│                                                                         │
│   ┌─────────────────────────────────────┐    ┌────────────────────────┐ │
│   │  routes/detect.py (unified router)  │    │  StaticFiles /output   │ │
│   │  POST /detect?mode=<...>            │    │  serves *_annotated.mp4│ │
│   └─────────────────────────────────────┘    └────────────────────────┘ │
│                       │                                                 │
│       ┌───────────────┼───────────────┬────────────────┬─────────────┐  │
│       ▼               ▼               ▼                ▼             ▼  │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ ┌───────┐ │
│  │ PPE      │  │ Pose       │  │ Fire       │  │ Sound      │ │Combined│ │
│  │ Service  │  │ Service    │  │ Service    │  │ Service    │ │Service │ │
│  └──────────┘  └────────────┘  └────────────┘  └────────────┘ └───────┘ │
│       │              │               │                │             │   │
│       ▼              ▼               ▼                ▼             ▼   │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ ┌───────┐ │
│  │ ppe_     │  │ pose_      │  │ HF Hub +   │  │ librosa +  │ │ all   │ │
│  │ utils    │  │ tracker +  │  │ YOLO-World │  │ ffmpeg     │ │ mode  │ │
│  │          │  │ accident_  │  │ fallback   │  │ +RandomFor.│ │ (chain│ │
│  │          │  │ detector   │  │            │  │            │ │  ed)  │ │
│  └──────────┘  └────────────┘  └────────────┘  └────────────┘ └───────┘ │
└─────────────────────────────────────────────────────────────────────────┘
              │              │              │              │
              ▼              ▼              ▼              ▼
        ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐
        │ ppe_model │  │yolov8s-  │  │ HF Cache  │  │ audio_model  │
        │   .pt     │  │ pose.pt  │  │ + YOLO-W  │  │   .pkl       │
        │           │  │ model.pkl│  │ weights   │  │              │
        └───────────┘  └──────────┘  └───────────┘  └──────────────┘
```

---


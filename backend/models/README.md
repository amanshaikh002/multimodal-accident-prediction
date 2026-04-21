# PPE Model Placeholder

Place your pretrained YOLOv8 PPE detection model here:

    backend/models/ppe_model.pt

## Recommended free models

| Model | Labels include | Download |
|-------|---------------|---------|
| yolov8n-ppe (Roboflow Universe) | helmet, vest, person | https://universe.roboflow.com/roboflow-universe-projects/ppe-detection-using-yolov8 |
| keremberke/yolov8n-PPE-detection | Hardhat, Safety Vest, Person | https://huggingface.co/keremberke/yolov8n-PPE-detection |

## Quick download (Python)

```python
# Option A — Hugging Face Hub
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="keremberke/yolov8n-PPE-detection",
    filename="best.pt",
)
import shutil; shutil.copy(path, "backend/models/ppe_model.pt")

# Option B — Roboflow
# Follow the Roboflow export guide and save as backend/models/ppe_model.pt
```

## Label mapping

The service auto-normalises these labels (case-insensitive):

| Raw model label | Canonical | PPE item |
|----------------|-----------|----------|
| helmet, hardhat, hard hat | helmet | ✅ |
| vest, safety_vest, safety vest, hi-vis vest | vest | ✅ |
| person | person | ℹ️ (not PPE) |

# ctOS — Real-Time Anomaly Detection System
### *Watch Dogs–Themed Anomaly Detection on NVIDIA Jetson Orin*

<p align="center">
  <img src="docs/banner.png" alt="ctOS Banner" width="800"/>
</p>

---

## About The Project

A **real-time anomaly detection system** running on **NVIDIA Jetson Orin (JetPack 6)** using **YOLOv8** object detection. The UI is inspired by **Watch Dogs' ctOS surveillance system** — featuring dark hacker aesthetics, cyan grid overlays, corner-tick bounding boxes, threat meters, and a live browser dashboard connected via WebSocket.

The system continuously analyzes a USB camera feed, applies rule-based anomaly logic, logs all events to CSV, saves annotated frames, and streams live data to a Watch Dogs themed browser HUD.

> **Person = Normal (Green box + ID:VERIFIED)**
> **Knife = CRITICAL Anomaly (Red pulsing box + ALERT)**

---

## Demo Screenshot

<p align="center">
  <img src="docs/jetson_photo.jpeg" alt="Jetson Output" width="600"/>
</p>

<p align="center">
  <img src="docs/ui_screenshot.png" alt="ctOS UI" width="600"/>
</p>

---

## Features

| Feature | Description |
|---|---|
| **Knife / weapon detection** | CRITICAL alert when knife detected — red pulsing box |
| **Person = normal** | Person is safe — green box with `ID:VERIFIED` badge |
| **Overcrowd detection** | Warning when >3 persons in frame simultaneously |
| **Proximity alert** | Warning when persons stand too close together |
| **Watch Dogs HUD overlay** | ctOS-themed OpenCV overlay on live camera feed |
| **Browser dashboard** | Live WebSocket-connected Watch Dogs UI in browser |
| **Active rules panel** | Shows Personnel Count, Forbidden Object, Proximity status |
| **Event log** | Timestamped log of every anomaly event in browser |
| **Last anomaly tracker** | Shows type, time, and seconds since last anomaly |
| **CSV anomaly log** | Every event saved to `output/anomaly_log.csv` |
| **Auto frame capture** | Annotated frames auto-saved on every detection |
| **Cooldown debounce** | Prevents duplicate log entries (2s cooldown per rule) |
| **Threat level meter** | Dynamic threat circle — CONDITION GREEN / AMBER / RED |

---

## Built With

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — YOLOv8n object detection
- Python 3.10 + asyncio + websockets + OpenCV
- NVIDIA Jetson Orin — JetPack 6 (R36.5)
- Conda environment `dev_38`
- Watch Dogs / ctOS aesthetic

---

## Getting Started

### Prerequisites

```bash
# Jetson Orin — JetPack 6 — inside dev_38 conda env
conda activate dev_38

pip install ultralytics --index-url https://pypi.org/simple/
pip install "numpy>=1.26,<2.0" --index-url https://pypi.org/simple/
pip install lap websockets --index-url https://pypi.org/simple/
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ctos-anomaly-detection.git
cd ctos-anomaly-detection
```

### Running

**Step 1 — Start the backend on Jetson:**
```bash
conda activate dev_38
python3 anomaly_detector.py --camera 0
```

**Step 2 — Open the browser UI:**
```bash
# On Jetson — open in Firefox/Chromium
xdg-open watchdogs_ui.html

# From another machine on same network:
# Edit watchdogs_ui.html line: const WS='ws://JETSON_IP:8765';
```

**Step 3 — Point USB camera at scene**
- Person in frame → **green box, ID:VERIFIED, no alert**
- Knife in frame → **red pulsing box, CRITICAL ALERT, logged to CSV**

---

## Anomaly Rules

Edit the `RULES` dict in `anomaly_detector.py`:

```python
RULES = {
    "forbidden_classes":  ["knife", "scissors", "baseball bat"],  # CRITICAL
    "max_persons":         3,        # Overcrowd warning
    "proximity_thresh":    0.25,     # 25% of frame width apart
    "min_confidence":      0.45,     # Ignore weak detections
    "cooldown_seconds":    2.0,      # Debounce between same-type logs
}
```

| Rule | Trigger | Severity |
|---|---|---|
| Forbidden object | Knife / weapon detected | CRITICAL 🔴 |
| Overcrowd | >3 persons in frame | WARNING 🟡 |
| Proximity | Persons too close | WARNING 🟡 |

---

## Output Files

```
output/
├── anomaly_log.csv         ← timestamped anomaly events
└── anomaly_images/         ← annotated frames on detection
```

**Sample `anomaly_log.csv`:**

| timestamp | frame | rule | severity | detail | objects_in_frame |
|---|---|---|---|---|---|
| 2026-04-22T14:32:01 | 000342 | forbidden_object | critical | WEAPON DETECTED: 'knife' (88% conf) | person, knife |
| 2026-04-22T14:35:10 | 001540 | overcrowd | warning | 4 persons in frame (limit 3) | person |
| 2026-04-22T14:36:44 | 002103 | proximity | warning | Persons too close: 18% frame apart | person |

---

## Project Structure

```
ctos-anomaly-detection/
├── anomaly_detector.py     ← YOLOv8 backend + WebSocket server
├── watchdogs_ui.html       ← Watch Dogs ctOS browser dashboard
├── output/
│   ├── anomaly_log.csv
│   └── anomaly_images/
├── docs/
│   ├── banner.png
│   ├── jetson_photo.jpeg
│   └── ui_screenshot.png
└── README.md
```

---

## Keyboard Shortcuts (OpenCV window)

| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Save screenshot manually to `output/anomaly_images/` |

---

## Observations

| Metric | Value |
|---|---|
| Model | YOLOv8n (nano) |
| Device | Jetson Orin — GPU (CUDA) |
| FPS | ~30–55 FPS with GPU |
| Knife detection confidence | ~75–92% |
| Person detection confidence | ~88–97% |
| False positives | Low — cooldown reduces duplicates |

### Improvement Suggestions
- Use YOLOv8s (small) for better knife accuracy at slight FPS cost
- Lower confidence threshold to 0.35 to catch partial knife views
- Add email/SMS alert on CRITICAL detections
- Train custom model on knife dataset for higher accuracy

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [dusty-nv jetson-inference](https://github.com/dusty-nv/jetson-inference)
- [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- Watch Dogs / ctOS for the aesthetic inspiration

---

*"I can see everything." — Aiden Pearce*

# Real-Time Face Overlay System (Hybrid AR Pipeline)

A real-time facial overlay and augmentation system built using MediaPipe, OpenCV, and NumPy.  
This project implements a hybrid warping pipeline with adaptive landmark stabilization, runtime benchmarking, and comparative evaluation tools.

---

## 🚀 Features

- 🎥 Real-time webcam-based face overlay
- 🔁 Hybrid warping:
  - Affine transformation (baseline)
  - Triangle mesh warping (high fidelity)
  - Automatic fallback system
- 🧠 Adaptive landmark stabilization (motion-aware smoothing)
- 🎯 Face-region masking and blending refinement
- 🎨 Color matching for realistic overlay integration
- 🔀 Multi-overlay switching at runtime
- 🆚 A/B comparison mode:
  - Baseline (affine)
  - Stabilized hybrid pipeline
- 📊 Benchmark logging:
  - FPS
  - tracking success rate
  - smoothing parameters
  - overlay capability
- 📸 Screenshot and 🎬 recording support
- ⚙️ Interactive controls via keyboard + sliders

---

## 🧠 System Architecture

Pipeline:
Webcam → Face Detection → Landmark Extraction → Stabilization → Warping → Mask Refinement → Blending → Output

### Core Components

- **Face Tracking:** MediaPipe Face Landmarker
- **Warping:**
  - Affine (fast, stable)
  - Triangle mesh (realistic)
- **Stabilization:**
  - Adaptive smoothing
  - Pose-aware correction
- **Masking:**
  - Face region restriction
  - Boundary-based occlusion suppression
- **Blending:**
  - Alpha blending
  - Seamless cloning (fallback)
- **Evaluation:**
  - Runtime CSV logging
  - A/B comparison pipeline

---

## 📦 Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

---

## ⚙️ Setup

### 1. Clone repo

```bash
git clone https://github.com/tejaskrishna-work/deep-fake-face.git
cd deep-fake-face
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Download model

Download MediaPipe face landmarker model and place it here:

assets/models/face_landmarker.task
🎮 Controls
Key	Action
n / p	Next / previous overlay
c	Toggle A/B comparison
m	Change warp mode
g	Toggle landmarks
r	Toggle anchor points
o	Toggle overlay
+ / -	Resize overlay
s	Save screenshot
v	Start/stop recording
q / esc	Quit
📊 Benchmarking

Logs are automatically saved in:

reports/benchmark_*.csv
Example metrics tracked:
FPS
face tracking success
smoothing alpha
motion magnitude
overlay capability (triangle vs affine)
Example session output:
Average FPS: 12.60
Tracking success rate: 65.38%
Triangle-capable frame rate: 100.00%
🧪 Evaluation Approach

The system supports runtime comparison between:

Baseline affine pipeline
Stabilized hybrid pipeline

This enables:

qualitative comparison (visual)
quantitative analysis (CSV logs)
⚠️ Limitations
Performance depends on hardware (CPU-bound)
Strong occlusions (hands/hair) are partially handled only
Extreme head rotations reduce overlay accuracy
Triangle warping requires face-detectable overlay images
🔮 Future Improvements
Face segmentation-based occlusion handling
Kalman filter for stronger temporal stabilization
3D head pose estimation
GPU acceleration
Web / mobile deployment

Project Highlights:
Real-time CV system with hybrid geometric warping
Adaptive stabilization with motion-aware smoothing
Modular overlay system with capability detection
Built-in benchmarking and evaluation pipeline
A/B testing framework for pipeline comparison

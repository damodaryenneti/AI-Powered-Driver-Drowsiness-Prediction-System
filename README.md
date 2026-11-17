# AI-Powered Driver Drowsiness Prediction System

A computer-vision system that watches a driver's face through a webcam and raises an alert when it detects signs of fatigue: **prolonged eye closure** and **yawning**.

The project has two parts:

| Part | File | What it is |
|---|---|---|
| **1. Notebook prototype** | `AI-Powered_Driver_Drowsiness_Prediction_System.ipynb` | Business understanding, plus a simple OpenCV + MediaPipe detector that runs in a desktop window |
| **2. Streamlit dashboard** | `app.py` | **VisionDrive AI Coach**, a browser dashboard with live metrics, an event log, adjustable sensitivity and audio alerts |

> **Prototype only.** This project is for learning and demonstration. It is not a certified safety system and must not be relied on to prevent accidents.

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [How Detection Works](#how-detection-works)
3. [Part 1: Notebook Prototype](#part-1-notebook-prototype)
4. [Part 2: Streamlit Dashboard (VisionDrive AI Coach)](#part-2-streamlit-dashboard-visiondrive-ai-coach)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Business Applications](#business-applications)
9. [Limitations](#limitations)
10. [Future Improvements](#future-improvements)

---

## Business Problem

Driver drowsiness is a major contributor to road accidents, especially in long-distance transport, logistics fleets and late-night shifts. Fatigue causes slower reaction time, reduced awareness and microsleep events. These lead to loss of life, vehicle damage, legal liability, downtime and higher insurance costs.

**Objective:** detect early signs of driver fatigue in real time and alert the driver before a critical failure occurs.

**Business constraints considered**

| Constraint | Description |
|---|---|
| Cost | Cameras, IR sensors, embedded hardware and cloud analytics add cost |
| Latency | Alerts must be generated in real time |
| Environment | Must work in low light, night driving, rain and glare |
| Privacy | Facial data must follow data protection and consent rules |
| Accuracy | Both false alarms and missed detections must be kept low |
| Hardware | Should run on low-power edge devices (for example Raspberry Pi) |

---

## How Detection Works

Both parts use the **MediaPipe Face Landmarker** to get facial landmarks from each video frame, then compute two ratios.

**Eye Aspect Ratio (EAR)** measures how open the eye is. It drops towards zero when the eye closes.

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```

**Mouth Aspect Ratio (MAR)** measures how wide the mouth is open. It rises during a yawn.

```
MAR = vertical lip distance / horizontal lip distance
```

A drowsiness alert is raised when **EAR stays low** (eyes closed) **or MAR stays high** (yawning) for a sustained period. Short dips are ignored, so normal blinks do not trigger an alert.

---

## Part 1: Notebook Prototype

The notebook walks through the business case and then builds a working detector.

**Notebook contents**

1. Business understanding (problem, objective, constraints)
2. Technical summary of drowsiness detection
3. Business applications and value
4. Roadmap and market opportunity
5. Code: EAR and MAR functions, model loading, and the live webcam loop

**Detection logic**

| Signal | Threshold | Trigger |
|---|---|---|
| Eyes closed | EAR < `0.2` | More than **30 consecutive frames** |
| Yawning | MAR > `0.6` | More than **30 consecutive frames** |

When either condition is met, the frame shows **"Drowsiness detected"** and a 2000 Hz beep plays for 500 ms. Press **`q`** to quit.

**Notes on the notebook version**

- It uses `winsound`, so the audio alert works on **Windows only**.
- The model path is hard-coded to a local folder. Change `model_path` to wherever you saved `face_landmarker.task`.
- Detection is frame-based, so the alert time depends on the camera's frame rate.

---

## Part 2: Streamlit Dashboard (VisionDrive AI Coach)

The dashboard turns the same idea into a browser app using `streamlit-webrtc`, with time-based rules instead of frame counts.

**Features**

- Live webcam monitoring in the browser
- EAR, MAR, **attention score** and **blink rate** (rolling 60 seconds)
- Drowsiness alert with an on-video warning, a red status badge and a browser beep
- Event log with timestamps
- Adjustable EAR threshold slider (0.16 to 0.30)
- Custom UI with a background image and glass-style panels

**Rules**

| Signal | Rule |
|---|---|
| Eye closure | EAR below the threshold (default `0.22`) for **1.5 s or more** |
| Blink | Closure lasting **0.08 to 0.8 s** |
| Yawn | MAR above `0.65` for **2.0 s or more** |
| Drowsiness | Eye closure **or** yawning; the alert fires once per new episode |

**Attention score** (approximate, for display):

```
eye_score   = clip((EAR - 0.10) / 0.20, 0, 1)
mouth_score = 1.0 if MAR < 0.65 else 0.5
attention   = (0.8 * eye_score + 0.2 * mouth_score) * 100
```

Frames are processed in a background thread, and the metrics are shared with the dashboard through a lock-protected dictionary. The dashboard refreshes every 500 ms using `st.fragment`.

---

## Project Structure

```
Driver-Drowsiness/
├── AI-Powered_Driver_Drowsiness_Prediction_System.ipynb   # Part 1
├── app.py                 # Part 2: Streamlit app
├── face_landmarker.task   # MediaPipe Face Landmarker model
├── drowsiness.jpeg        # Dashboard background image
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/damodaryenneti/<your-repo-name>.git
cd <your-repo-name>
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
streamlit
streamlit-webrtc
mediapipe
opencv-python
av
numpy
jupyter
```

**4. Download the MediaPipe model**

Download `face_landmarker.task` from the [MediaPipe Face Landmarker page](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) and place it next to `app.py`.

**5. Add the background image (Streamlit app only)**

Place `drowsiness.jpeg` in the same folder. If it is missing, the app shows a warning and uses the default background.

---

## Usage

**Run the notebook version**

```bash
jupyter notebook
```

Open the notebook, update `model_path`, and run the cells in order. A webcam window opens; press `q` to close it.

**Run the Streamlit dashboard**

```bash
streamlit run app.py
```

1. Open `http://localhost:8501` in your browser.
2. Click **START** and allow camera access.
3. Keep the face visible and well lit.
4. Adjust the **Drowsiness Threshold** slider if alerts trigger too easily or too rarely.
5. Use **Enable browser audio alert** to switch the beep on or off.

---

## Business Applications

| Sector | Use |
|---|---|
| Automotive | Driver monitoring inside ADAS |
| Fleet and logistics | Monitoring long-haul and highway drivers |
| Public transport | Buses, trains and metro operators |
| Mining and heavy equipment | Crane, excavator and drilling operators |
| Ride-hailing and taxis | Driver alertness for Uber, Ola and corporate fleets |
| Defense and airports | Military and airport ground vehicles |

---

## Limitations

- Accuracy depends on lighting, camera angle and image quality.
- Glasses, sunglasses, masks or heavy head rotation can reduce detection accuracy.
- EAR and MAR use fixed thresholds that may need tuning for different people.
- No machine-learning classifier is trained; detection is rule-based.
- Browsers require **HTTPS** for camera access when the dashboard is deployed (`localhost` works without it).
- The WebRTC connection uses a public STUN server; some networks need a TURN server.
- Head pose, gaze direction and phone use are not detected.

---

## Future Improvements

- CNN + LSTM models for fatigue behaviour prediction
- Personalised fatigue profiles and per-driver calibration
- Head pose estimation (pitch, yaw, roll) for nodding detection
- PERCLOS (percentage of eye closure over time)
- Multi-sensor fusion: heart rate, steering behaviour and lane deviation
- Seat vibration and dashboard warnings alongside audio alerts
- Cloud fleet analytics: fatigue trends and high-risk route detection
- Session history, reports and insurance or compliance integration
- Deployment on edge devices such as Raspberry Pi


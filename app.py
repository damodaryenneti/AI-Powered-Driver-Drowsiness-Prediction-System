import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import av
import numpy as np
import time
import math
import wave
import io
import threading
import base64
from pathlib import Path
from collections import deque
from datetime import datetime


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VisionDrive AI Coach",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "face_landmarker.task"
BG_IMAGE_PATH = "drowsiness.jpeg"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f2f3f7;
    }

    header[data-testid="stHeader"] {
        background-color: #373b91;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .topbar {
        background: linear-gradient(100deg, #35398c, #4549a8);
        padding: 17px 25px;
        border-radius: 8px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .brand {
        font-size: 23px;
        font-weight: 800;
    }

    .nav {
        font-size: 15px;
        word-spacing: 24px;
    }

    .panel {
        background: white;
        padding: 17px;
        border-radius: 13px;
        border: 1px solid #e4e5eb;
        box-shadow: 0 2px 7px rgba(0,0,0,0.07);
        margin-bottom: 17px;
    }

    .panel-title {
        color: #252b65;
        font-size: 21px;
        font-weight: 750;
        border-bottom: 1px solid #eeeeee;
        padding-bottom: 12px;
        margin-bottom: 13px;
    }

    .metric-label {
        color: #333;
        font-size: 15px;
    }

    .metric-value {
        color: #252b65;
        font-size: 20px;
        font-weight: 800;
        float: right;
    }

    .metric-track {
        height: 9px;
        background: #e5e5e5;
        border-radius: 20px;
        margin-top: 8px;
        overflow: hidden;
    }

    .metric-fill {
        height: 9px;
        background: #ed7d16;
        border-radius: 20px;
    }

    .event-row {
        padding: 11px 2px;
        border-bottom: 1px solid #eeeeee;
        font-size: 14px;
        color: #333;
    }

    .event-time {
        color: #777;
        margin-right: 10px;
    }

    .status-ready {
        background: #13864c;
        color: white;
        padding: 7px 18px;
        border-radius: 8px;
        font-weight: 600;
    }

    .status-alert {
        background: #c62828;
        color: white;
        padding: 7px 18px;
        border-radius: 8px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #eeeeee;
        border-radius: 9px;
        padding: 10px;
    }

    .small-note {
        color: #777;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

@st.cache_data
def get_base64_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


if Path(BG_IMAGE_PATH).exists():
    BG_IMAGE = get_base64_image(BG_IMAGE_PATH)

    st.markdown(f"""
    <style>
        .stApp,
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(10, 12, 30, 0.10),
                                rgba(10, 12, 30, 0.10)),
                url("data:image/jpeg;base64,{BG_IMAGE}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}

        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        .main {{
            background: transparent !important;
        }}

        /* Glass-style columns so text stays readable over a clear photo */
        div[data-testid="stColumn"] > div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stColumn"] > div {{
            border-radius: 14px;
        }}

        div[data-testid="stColumn"]:has(.panel-title) {{
            background: rgba(12, 15, 38, 0.55);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            padding: 18px;
        }}

        /* Remove the empty white bars created by the open/close panel divs */
        .panel {{
            background: transparent;
            border: none;
            box-shadow: none;
            padding: 0;
            margin: 0;
        }}

        /* Light text on the darker glass */
        .panel-title {{
            color: #ffffff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.25);
        }}
        .metric-label, .metric-value, .event-row {{
            color: #ffffff;
        }}
        .event-time, .small-note {{
            color: #d0d3e6;
        }}
        .metric-track {{
            background: rgba(255, 255, 255, 0.25);
        }}
        .event-row {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.warning(
        f"Background image '{BG_IMAGE_PATH}' not found. "
        "Place it in the same folder as this app file."
    )


# =========================================================
# SHARED STATE BETWEEN STREAMLIT AND CAMERA THREAD
# =========================================================

if "driver_shared" not in st.session_state:
    st.session_state.driver_shared = {
        "lock": threading.Lock(),
        "ear": 0.0,
        "mar": 0.0,
        "attention": 0,
        "blink_rate": 0,
        "drowsy": False,
        "face_detected": False,
        "alert_id": 0,
        "events": [],
        "threshold": 0.22,
        "show_landmarks": True,
        "last_update": 0
    }

shared = st.session_state.driver_shared


def add_event(message, event_type="info"):
    with shared["lock"]:
        shared["events"].insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "type": event_type
        })
        shared["events"] = shared["events"][:12]


# =========================================================
# EAR AND MAR CALCULATIONS
# =========================================================

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_ear(landmarks, eye_indices, width, height):
    points = []

    for index in eye_indices:
        landmark = landmarks[index]
        points.append([
            landmark.x * width,
            landmark.y * height
        ])

    vertical_1 = euclidean_distance(points[1], points[5])
    vertical_2 = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks, width, height):
    upper = landmarks[13]
    lower = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    upper_point = [upper.x * width, upper.y * height]
    lower_point = [lower.x * width, lower.y * height]
    left_point = [left.x * width, left.y * height]
    right_point = [right.x * width, right.y * height]

    vertical = euclidean_distance(upper_point, lower_point)
    horizontal = euclidean_distance(left_point, right_point)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


# =========================================================
# FACIAL LANDMARK DRAWING
# =========================================================

# =========================================================
# MEDIAPIPE VIDEO PROCESSOR
# =========================================================

class DriverProcessor:

    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

        self.start_time = time.monotonic()
        self.last_timestamp = 0

        self.closed_start = None
        self.yawn_start = None
        self.was_drowsy = False

        self.blink_timestamps = deque()

        self.right_eye_indices = [
            33, 160, 158, 133, 153, 144
        ]

        self.left_eye_indices = [
            362, 385, 387, 263, 373, 380
        ]

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        height, width, _ = image.shape

        timestamp = int(
            (time.monotonic() - self.start_time) * 1000
        )

        timestamp = max(timestamp, self.last_timestamp + 1)
        self.last_timestamp = timestamp

        rgb_image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.landmarker.detect_for_video(
            mp_image, timestamp
        )

        ear = 0.0
        mar = 0.0
        attention = 0
        drowsy = False

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            right_ear = calculate_ear(
                landmarks,
                self.right_eye_indices,
                width,
                height
            )

            left_ear = calculate_ear(
                landmarks,
                self.left_eye_indices,
                width,
                height
            )

            ear = (right_ear + left_ear) / 2

            mar = calculate_mar(
                landmarks, width, height
            )

            current_time = time.monotonic()
            threshold = shared["threshold"]

            # Blink detection
            if ear < threshold:
                if self.closed_start is None:
                    self.closed_start = current_time
            else:
                if self.closed_start is not None:
                    closed_duration = (
                        current_time - self.closed_start
                    )

                    if 0.08 <= closed_duration <= 0.8:
                        self.blink_timestamps.append(
                            current_time
                        )

                    self.closed_start = None

            # Keep only the last 60 seconds of blinks
            while (
                self.blink_timestamps
                and current_time - self.blink_timestamps[0] > 60
            ):
                self.blink_timestamps.popleft()

            blink_rate = len(self.blink_timestamps)

            # Eye closure duration
            eye_closed = (
                self.closed_start is not None
                and current_time - self.closed_start >= 1.5
            )

            # Yawning detection
            if mar > 0.65:
                if self.yawn_start is None:
                    self.yawn_start = current_time
            else:
                self.yawn_start = None

            yawning = (
                self.yawn_start is not None
                and current_time - self.yawn_start >= 2.0
            )

            drowsy = eye_closed or yawning

            # Approximate attention score for dashboard display
            eye_score = np.clip(
                (ear - 0.10) / 0.20, 0, 1
            )

            mouth_score = 1.0 if mar < 0.65 else 0.5

            attention = int(
                np.clip(
                    (eye_score * 0.8 + mouth_score * 0.2) * 100,
                    0, 100
                )
            )

        

            # Camera overlay
            cv2.rectangle(
                image,
                (0, height - 48),
                (width, height),
                (15, 15, 15),
                -1
            )

            cv2.putText(
                image,
                f"EAR: {ear:.2f}",
                (15, height - 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )

            cv2.putText(
                image,
                f"MAR: {mar:.2f}",
                (150, height - 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )

            cv2.putText(
                image,
                f"Attention: {attention}%",
                (285, height - 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 220, 100), 2
            )

            if drowsy:
                cv2.putText(
                    image,
                    "DROWSINESS ALERT!",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 3
                )

            # Log only the transition into drowsiness
            if drowsy and not self.was_drowsy:
                with shared["lock"]:
                    shared["alert_id"] += 1

                add_event(
                    "Drowsiness detected",
                    "alert"
                )

            self.was_drowsy = drowsy

            with shared["lock"]:
                shared["ear"] = float(ear)
                shared["mar"] = float(mar)
                shared["attention"] = int(attention)
                shared["blink_rate"] = int(blink_rate)
                shared["drowsy"] = bool(drowsy)
                shared["face_detected"] = True
                shared["last_update"] = current_time

        else:
            with shared["lock"]:
                shared["ear"] = 0.0
                shared["mar"] = 0.0
                shared["attention"] = 0
                shared["face_detected"] = False
                shared["drowsy"] = False

            cv2.putText(
                image,
                "NO FACE DETECTED",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2
            )

        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="topbar">
    <div class="brand">V VisionDrive AI Coach</div>
    <div class="nav">Dashboard Analytics Settings Help</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MAIN DASHBOARD LAYOUT
# =========================================================

left_column, right_column = st.columns(
    [2.05, 1],
    gap="large"
)


# =========================================================
# LEFT: LIVE DRIVER MONITORING
# =========================================================

with left_column:

    st.markdown("""
    <div class="panel-title">
        Live Driver Monitoring
    </div>
    """, unsafe_allow_html=True)

    camera_context = webrtc_streamer(
        key="driver-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DriverProcessor,
        media_stream_constraints={
            "video": {
                "facingMode": "user"
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "style": {
                "width": "100%",
                "height": "auto",
                "objectFit": "cover"
            }
        }
    )

    st.markdown(
        '<div class="small-note">Position the camera in front of the driver. '
        'Keep the face visible and ensure adequate lighting.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-title">Camera Controls</div>',
        unsafe_allow_html=True
    )

    control_col1, control_col2 = st.columns(2)

    with control_col1:
        st.markdown("**Camera Source**")
        st.selectbox(
            "Camera",
            ["Default front-facing camera"],
            label_visibility="collapsed"
        )

    

    with control_col2:
        st.markdown("**Drowsiness Threshold**")

        sensitivity = st.slider(
            "More sensitive  ← →  Less sensitive",
            min_value=0.16,
            max_value=0.30,
            value=0.22,
            step=0.01
        )

    shared["threshold"] = sensitivity


    st.caption(
        f"EAR threshold: {sensitivity:.2f} | "
        "Lower threshold = fewer eye-closure triggers."
    )

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# RIGHT: DRIVER METRICS AND EVENT LOG
# =========================================================

with right_column:

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-title">Driver Metrics</div>',
        unsafe_allow_html=True
    )

    @st.fragment(run_every="500ms")
    def update_dashboard():

        with shared["lock"]:
            ear = shared["ear"]
            mar = shared["mar"]
            attention = shared["attention"]
            blink_rate = shared["blink_rate"]
            drowsy = shared["drowsy"]
            face_detected = shared["face_detected"]
            alert_id = shared["alert_id"]
            events = list(shared["events"])

        if drowsy:
            st.markdown(
                '<span class="status-alert">DROWSINESS ALERT</span>',
                unsafe_allow_html=True
            )
        elif face_detected:
            st.markdown(
                '<span class="status-ready">Driver Monitoring Active</span>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Waiting for driver face...")

        st.write("")

        def metric_card(label, value, percentage):
            percentage = max(0, min(100, percentage))

            st.markdown(f"""
            <div style="padding:10px 0; border-bottom:1px solid #eee;">
                <span class="metric-label">{label}</span>
                <span class="metric-value">{value}</span>
                <div class="metric-track">
                    <div class="metric-fill"
                         style="width:{percentage}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        metric_card(
            "Attention Score",
            f"{attention}%",
            attention
        )

        metric_card(
            "Eye Openness (EAR)",
            f"{ear:.2f}",
            ear / 0.40 * 100
        )

        metric_card(
            "Mouth Openness (MAR)",
            f"{mar:.2f}",
            mar / 1.0 * 100
        )

        metric_card(
            "Blink Rate",
            f"{blink_rate} bpm",
            blink_rate / 40 * 100
        )

        # Browser-played alert sound, once per new alert event
        if "last_played_alert" not in st.session_state:
            st.session_state.last_played_alert = 0

        if (
            alert_id > st.session_state.last_played_alert
            and st.session_state.get("sound_enabled", True)
        ):
            # Generate a short WAV beep
            sample_rate = 44100
            duration = 0.7
            frequency = 900

            audio_buffer = io.BytesIO()

            with wave.open(audio_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)

                samples = bytearray()

                for i in range(int(sample_rate * duration)):
                    value = int(
                        12000 * math.sin(
                            2 * math.pi * frequency * i / sample_rate
                        )
                    )
                    samples.extend(value.to_bytes(
                        2, byteorder="little", signed=True
                    ))

                wav_file.writeframes(samples)

            st.audio(
                audio_buffer.getvalue(),
                format="audio/wav",
                autoplay=True
            )

            st.session_state.last_played_alert = alert_id

        st.markdown("---")
        st.markdown("**Event Log**")

        if not events:
            st.caption("No events recorded yet.")
        else:
            for event in events[:8]:

                if event["type"] == "alert":
                    symbol = "🔴"
                else:
                    symbol = "🟢"

                st.markdown(
                    f"""
                    <div class="event-row">
                        <span class="event-time">{event['time']}</span>
                        {symbol} {event['message']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    update_dashboard()

    st.markdown('</div>', unsafe_allow_html=True)

    sound_enabled = st.toggle(
        "Enable browser audio alert",
        value=True,
        key="sound_enabled"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div style="
    text-align:center;
    color:#777;
    font-size:12px;
    padding:12px;
">
    VisionDrive AI Coach | MediaPipe Face Landmarker |
    Driver Drowsiness Monitoring Prototype
</div>
""", unsafe_allow_html=True)
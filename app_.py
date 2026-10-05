"""Streamlit app: webcam -> MediaPipe Hands -> engineered features -> FNN -> 2 s stability -> flower.

Run locally:  streamlit run app.py
The browser captures the webcam (streamlit-webrtc), so the same code works locally and when deployed.
"""
from __future__ import annotations

import base64
import io
import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

import config
from src.flowers import render_flower
from src.inference import ArtifactsMissing, artifacts_ready, get_classifier
from src.stability import StabilityTracker

st.set_page_config(page_title="AI Gesture-to-Flower", page_icon="🌸", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
header[data-testid="stHeader"]{display:none}
.block-container{padding:.6rem 1.2rem 0 1.2rem;max-width:100%}
.hero{font-size:1.5rem;font-weight:800;margin:0 0 .3rem 0;background:linear-gradient(90deg,#ff5c8a,#ffb86c,#8be9fd);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.card{background:#1a1d27;border:1px solid #2b3040;border-radius:12px;padding:8px 14px;margin-bottom:6px}
.lbl{color:#9aa4bf;font-size:.7rem;text-transform:uppercase;letter-spacing:.08em}
.big{font-size:1.15rem;font-weight:700}
.panel{height:78vh;min-height:430px;border-radius:16px;border:1px solid #2b3040;background:#141722;
       display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:10px}
.panel img{max-height:58%;max-width:92%;border-radius:14px;box-shadow:0 8px 30px rgba(0,0,0,.45)}
.panel .fname{font-size:1.9rem;font-weight:800;margin:.5rem 0 0 0}
.panel .fdesc{color:#cfd6ea;max-width:90%;margin:.2rem 0}
.panel .empty{color:#6b7391;font-size:1.1rem}
.fade{animation:fade .7s ease-in}
@keyframes fade{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
div[data-testid="stVideo"], video{max-height:52vh}
</style>""", unsafe_allow_html=True)

RTC = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
DISPLAY = {g: i["display"] for g, i in config.GESTURE_INFO.items()}


class GestureProcessor(VideoProcessorBase):
    """Runs in a worker thread: detect hand, classify, update stability, draw overlay."""

    def __init__(self):
        from src.hand_detector import HandDetector
        self._lock = threading.Lock()
        self.detector = HandDetector()
        self.classifier = get_classifier()
        a = config.APP
        self.tracker = StabilityTracker(self.classifier.labels, a["hold_seconds"], a["min_confidence"],
                                        a["grace_seconds"], a["smooth_window"], a.get("release_seconds", 0.8))
        self.state = self.tracker.update(None).as_dict()
        self.error, self.fps, self._t = None, 0.0, time.monotonic()

    def update_params(self, hold: float, min_conf: float) -> None:
        with self._lock:
            self.tracker.hold_seconds, self.tracker.min_confidence = hold, min_conf

    def snapshot(self) -> dict:
        with self._lock:
            return {**self.state, "fps": self.fps, "error": self.error}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)          # selfie view (needed for handedness)
        try:
            res = self.detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            probs = None
            if res is not None:
                self.detector.draw(img, res)
                probs = self.classifier.predict_proba(res.raw, res.is_left)
            with self._lock:
                self.state = self.tracker.update(probs).as_dict()
                now = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 / max(now - self._t, 1e-3)
                self._t, self.error = now, None
                s = dict(self.state)
            self._overlay(img, s)
        except Exception as exc:                                       # never kill the video thread
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    @staticmethod
    def _overlay(img, s: dict) -> None:
        h, w = img.shape[:2]
        text = (f"{DISPLAY.get(s['predicted'], s['predicted'])}  {s['confidence'] * 100:.0f}%"
                if s["predicted"] else "Show your hand")
        cv2.rectangle(img, (0, 0), (w, 38), (20, 20, 20), -1)
        cv2.putText(img, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 14), (w, h), (40, 40, 40), -1)
        done = s["progress"] >= 1.0
        cv2.rectangle(img, (0, h - 14), (int(w * s["progress"]), h), (90, 220, 120) if done else (80, 160, 255), -1)



# ------------------------------------------------------------------ sidebar (settings only)
with st.sidebar:
    st.header("How to use")
    st.markdown("1. Click **START**, allow the camera.\n2. Show **one hand** in good light.\n"
                "3. **Hold the gesture 2 seconds** - the bar fills, the flower blooms.\n"
                "4. Take your hand away - the flower disappears.")
    st.subheader("Gesture → Flower")
    for g, f in config.FLOWERS.items():
        st.caption(f"**{DISPLAY[g]}** → {f['name']} - {config.GESTURE_INFO[g]['how_to']}")
    st.subheader("Settings")
    hold = st.slider("Hold time (s)", 1.0, 4.0, float(config.APP["hold_seconds"]), 0.5)
    min_conf = st.slider("Min. confidence", 0.50, 0.99, float(config.APP["min_confidence"]), 0.01)
    st.caption("Defaults (2 s, 0.80) are the evaluated settings.")

missing = artifacts_ready()
if missing:
    st.error("Trained model files are missing: " + ", ".join(missing))
    st.code("python -m src.collect_data\npython -m src.preprocess\npython -m src.train --tune\n"
            "python -m src.evaluate", language="bash")
    st.stop()
try:
    get_classifier()
except (ArtifactsMissing, RuntimeError, OSError, ValueError) as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.markdown('<p class="hero">🌸 AI Gesture-to-Flower</p>', unsafe_allow_html=True)
col_cam, col_flower = st.columns(2, gap="medium")
with col_cam:
    ctx = webrtc_streamer(key="gesture-flower", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC,
                          video_processor_factory=GestureProcessor, async_processing=True,
                          media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False})
    alert = st.empty()
    s1, s2 = st.columns(2)
    with s1:
        pred_ph = st.empty()
    with s2:
        conf_gesture_ph = st.empty()
    conf_ph = st.empty()
    prog_ph = st.empty()
with col_flower:
    flower_ph = st.empty()


@st.cache_data(show_spinner=False)
def flower_b64(gesture: str) -> str:
    buf = io.BytesIO()
    render_flower(gesture, 520).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def show_flower(gesture):
    if gesture is None:                       # no gesture -> NO flower
        flower_ph.markdown('<div class="panel"><div class="empty">Show a gesture and hold it for '
                           '2 seconds…</div></div>', unsafe_allow_html=True)
        return
    f = config.FLOWERS[gesture]
    flower_ph.markdown(
        f'<div class="panel fade"><img src="data:image/png;base64,{flower_b64(gesture)}"/>'
        f'<p class="fname">{f["name"]} <span style="font-size:.9rem;color:#9aa4bf"><i>{f["latin"]}</i></span></p>'
        f'<p class="fdesc">{f["description"]}</p>'
        f'<p class="lbl" style="margin:.3rem 0 0 0">Meaning · <b style="color:#fff">{f["meaning"]}</b></p>'
        f'<p class="lbl" style="margin:.1rem 0 0 0">Gesture · <b style="color:#fff">{DISPLAY[gesture]}</b></p></div>',
        unsafe_allow_html=True)


def render(s: dict):
    name = DISPLAY.get(s["predicted"], "—") if s["hand_present"] else "No hand detected"
    pred_ph.markdown(f'<div class="card"><div class="lbl">Prediction</div><div class="big">{name}</div></div>',
                     unsafe_allow_html=True)
    c = DISPLAY.get(s["confirmed"], "none") if s["confirmed"] else "none"
    conf_gesture_ph.markdown(f'<div class="card"><div class="lbl">Confirmed</div><div class="big">{c}</div></div>',
                             unsafe_allow_html=True)
    conf_ph.progress(min(1.0, max(0.0, s["confidence"])), text=f"Confidence: {s['confidence'] * 100:.0f}%")
    if s["candidate"]:
        done = s["progress"] >= 1.0
        txt = ("Confirmed ✔" if done else f"Hold steady… {s['remaining']:.1f}s left") + f" - {DISPLAY[s['candidate']]}"
        prog_ph.progress(s["progress"], text=txt)
    else:
        prog_ph.progress(0.0, text="2-second stability timer: waiting for a steady gesture")


show_flower(None)
render(GestureProcessor.__dict__.get("_empty", {"hand_present": False, "predicted": None, "confidence": 0.0,
                                                "candidate": None, "progress": 0.0, "remaining": 0.0,
                                                "confirmed": None}))
if ctx.state.playing:
    shown = None
    while ctx.state.playing:
        proc = ctx.video_processor
        if proc is None:
            time.sleep(0.1)
            continue
        proc.update_params(hold, min_conf)
        snap = proc.snapshot()
        if snap["error"]:
            alert.warning(f"Processing error (will keep trying): {snap['error']}")
        else:
            alert.empty()
        render(snap)
        if snap["confirmed"] != shown:
            shown = snap["confirmed"]
            show_flower(shown)
        time.sleep(0.1)

"""Streamlit app: webcam -> MediaPipe Hands -> engineered features -> FNN -> 2 s stability -> flower.

Everything (prediction, 2-second timer, flower) is drawn INSIDE the video frame, so it is also visible
when you press the video's full-screen button.
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import threading
import time

import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

import config
from src.ice import get_ice_servers, has_turn
from src.inference import ArtifactsMissing, artifacts_ready, get_classifier
from src.overlay import draw_overlay
from src.stability import StabilityTracker

st.set_page_config(page_title="AI Gesture-to-Flower", page_icon="🌸", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
header[data-testid="stHeader"]{display:none}
.block-container{padding:.6rem 1.2rem 0 1.2rem;max-width:100%}
.hero{font-size:1.5rem;font-weight:800;margin:0 0 .3rem 0;background:linear-gradient(90deg,#ff5c8a,#ffb86c,#8be9fd);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent}
video{width:100%;max-height:82vh;background:#000;border-radius:10px}
</style>""", unsafe_allow_html=True)



@st.cache_data(ttl=3000, show_spinner=False)      # TURN credentials expire, so refresh regularly
def _ice_servers() -> list:
    try:
        secrets = dict(st.secrets)
    except Exception:                              # no secrets file (local run)
        secrets = {}
    return get_ice_servers(secrets)


_SERVERS = _ice_servers()
RTC = RTCConfiguration({"iceServers": _SERVERS})
DISPLAY = {g: i["display"] for g, i in config.GESTURE_INFO.items()}


class GestureProcessor(VideoProcessorBase):
    """Worker thread: detect hand, classify, update stability, draw the full UI into the frame."""

    def __init__(self):
        from src.hand_detector import HandDetector
        self._lock = threading.Lock()
        self.detector = HandDetector()
        self.classifier = get_classifier()
        a = config.APP
        self.tracker = StabilityTracker(self.classifier.labels, a["hold_seconds"], a["min_confidence"],
                                        a["grace_seconds"], a["smooth_window"], a.get("release_seconds", 0.8))
        self.state = self.tracker.update(None).as_dict()
        self.error = None
        self._last_count, self._confirm_t = 0, 0.0

    def update_params(self, hold: float, min_conf: float) -> None:
        with self._lock:
            self.tracker.hold_seconds, self.tracker.min_confidence = hold, min_conf

    def snapshot(self) -> dict:
        with self._lock:
            return {**self.state, "error": self.error}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = cv2.flip(frame.to_ndarray(format="bgr24"), 1)          # selfie view (needed for handedness)
        try:
            res = self.detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            probs = None
            if res is not None:
                self.detector.draw(img, res)
                probs = self.classifier.predict_proba(res.raw, res.is_left)
            now = time.monotonic()
            with self._lock:
                self.state = self.tracker.update(probs, now).as_dict()
                s = dict(self.state)
                self.error = None
            if s["confirm_count"] != self._last_count:
                self._last_count, self._confirm_t = s["confirm_count"], now
            draw_overlay(img, s, DISPLAY, confirm_age=now - self._confirm_t)
        except Exception as exc:                                       # never kill the video thread
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
        return av.VideoFrame.from_ndarray(img, format="bgr24")


with st.sidebar:
    st.header("How to use")
    st.markdown("1. Click **START**, allow the camera.\n2. Show **one hand** in good light.\n"
                "3. **Hold the gesture 2 seconds** - the flower appears inside the video.\n"
                "4. Use the video's **full-screen** button for a bigger view.\n"
                "5. Take your hand away - the flower disappears.")
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
if not has_turn(_SERVERS):
    st.warning("No TURN server configured: the webcam usually cannot connect on Streamlit Cloud. "
               "Add METERED_DOMAIN + METERED_API_KEY (or Twilio keys) in app Settings → Secrets, then reboot. "
               "Locally this warning can be ignored.")
ctx = webrtc_streamer(key="gesture-flower", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC,
                      video_processor_factory=GestureProcessor, async_processing=True,
                      media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                                                "audio": False})
alert = st.empty()

if ctx.state.playing:                      # keeps sidebar settings in sync + shows processing errors
    while ctx.state.playing:
        proc = ctx.video_processor
        if proc is not None:
            proc.update_params(hold, min_conf)
            err = proc.snapshot()["error"]
            if err:
                alert.warning(f"Processing error (will keep trying): {err}")
            else:
                alert.empty()
        time.sleep(0.3)

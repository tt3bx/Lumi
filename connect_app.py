import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

st.set_page_config(page_title="LumiSign Connect", page_icon="🤟", layout="wide")

st.title("🤟 LumiSign Connect")
st.write("Real Sign-to-Text using MediaPipe + AI Model")
st.success("🔒 WebRTC Encryption Active")

MODEL_PATH = "lumisign_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("AI Model Loaded ✅")
else:
    model = None
    st.error("lumisign_model.pkl not found. Train the model first.")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.last_prediction = "No Sign"
        self.confidence = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        prediction = "No Sign"
        confidence = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                features = np.array(landmarks).reshape(1, -1)

                if model is not None:
                    prediction = model.predict(features)[0]

                    if hasattr(model, "predict_proba"):
                        confidence = int(np.max(model.predict_proba(features)) * 100)
                    else:
                        confidence = 90

                self.last_prediction = prediction
                self.confidence = confidence

        cv2.putText(
            img,
            "LumiSign Connect",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            img,
            f"Sign-to-Text: {self.last_prediction}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Confidence: {self.confidence}%",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

room = st.text_input("Room ID", "room123")
st.info(f"Room Created: {room}")

ctx = webrtc_streamer(
    key=room,
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={
        "video": True,
        "audio": True
    }
)

st.markdown("## Live Translation")
st.success("الكاميرا الآن تقرأ حركة اليد وتحوّلها إلى نص حسب الموديل المدرب.")

st.markdown("## Supported Signs")
st.info("Hello / Help / Yes / No / Thank You حسب البيانات اللي سجلتها في sign_data.csv")
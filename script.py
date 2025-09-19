import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time
try:
    from deepface import DeepFace

    deepface_available = True
except ImportError:
    st.error("DeepFace library not available. Please install it with: pip install deepface")
    deepface_available = False
st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😀",
    layout="centered"
)
st.title("😀 Emotion Detection App")
st.markdown("Analyze emotions in real-time using your webcam or uploaded images")

if deepface_available:
    webcam_tab, upload_tab, about_tab = st.tabs(["Webcam Detection", "Image Upload", "About"])

    with webcam_tab:
        st.subheader("Real-time Emotion Detection")
        run = st.checkbox('Start Webcam')
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        emotion_placeholder = st.empty()

        if run:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not access webcam. Please check your connection.")
                else:
                    emotions = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
                                'sad': 0, 'surprise': 0, 'neutral': 0}

                    while run:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture image")
                            break
                        try:
                            status_placeholder.info("Analyzing...")
                            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                            dominant_emotion = result[0]['dominant_emotion']
                            emotions[dominant_emotion] += 1
                            emotion_scores = result[0]['emotion']
                            confidence = emotion_scores[dominant_emotion]
                            emotion_placeholder.success(
                                f"**Detected: {dominant_emotion.upper()}** (Confidence: {confidence:.1f}%)")
                            cv2.putText(frame, f"{dominant_emotion.upper()}: {confidence:.1f}%",
                                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            status_placeholder.empty()
                        except Exception as e:
                            status_placeholder.warning("No face detected. Position yourself in front of the camera.")
                            emotion_placeholder.empty()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                        time.sleep(0.1)
                    cap.release()
                    if sum(emotions.values()) > 0:
                        st.subheader("Emotion Summary")
                        st.bar_chart(emotions)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Click 'Start Webcam' to begin", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            frame_placeholder.image(img, channels="RGB", use_column_width=True)

    with upload_tab:
        st.subheader("Upload Image for Emotion Analysis")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
            if st.button("Analyze Emotion"):
                with st.spinner("Analyzing..."):
                    try:
                        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
                        dominant_emotion = result[0]['dominant_emotion']
                        emotion_scores = result[0]['emotion']
                        st.success(f"**Dominant Emotion: {dominant_emotion.upper()}**")
                        st.subheader("All Detected Emotions")
                        emotion_data = {k: v for k, v in emotion_scores.items()}
                        st.bar_chart(emotion_data)

                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
                        st.info("Make sure there's a clearly visible face in the image.")

    with about_tab:
        st.subheader("About This App")
        st.markdown("""
        This app uses DeepFace to analyze emotions in images and webcam feeds.

        ### Detectable Emotions:
        - Happy
        - Sad
        - Angry
        - Surprise
        - Fear
        - Disgust
        - Neutral

        ### How it works:
        The app uses deep neural networks to:
        1. Detect faces in the image/video
        2. Analyze facial expressions
        3. Classify emotions based on these expressions

        ### Tips for best results:
        - Ensure good lighting
        - Face the camera directly
        - Keep a neutral background
        - Position your face to fill a good portion of the frame
        """)

else:
    st.warning("This app requires the DeepFace library to function.")
    st.info("Please install required dependencies with: `pip install deepface opencv-python streamlit`")
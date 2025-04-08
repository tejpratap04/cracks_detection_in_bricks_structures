
import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO

# Set page title
st.title("🧱 Crack Detection in Brick Structures (YOLOv8-Seg)")

# Upload the video file
uploaded_file = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])

# Proceed only if a file is uploaded
if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name

    st.video(input_video_path)

    # Proceed button
    if st.button("🚀 Proceed Prediction"):
        with st.spinner("Running predictions..."):

            # Load the trained model
            model = YOLO("D:\\acadamics\\5th Year\\MTP\\final_model\\crack_prediction\\best.pt")  # Replace with your trained model path

            # Setup for output video
            cap = cv2.VideoCapture(input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Temporary output file
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run segmentation prediction
                results = model.predict(source=frame, task='segment', conf=0.25, save=False, stream=False, verbose=False)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            cap.release()
            out.release()

        st.success("✅ Prediction Complete!")

        # Show predicted video
        st.video(output_path)

        # Provide download link
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Predicted Video", f, file_name="predicted_video.mp4")
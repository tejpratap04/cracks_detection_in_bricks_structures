import streamlit as st
import tempfile
import cv2
import os
from ultralytics import YOLO
import numpy as np
from skimage.morphology import skeletonize
from skimage.draw import line
import math

# Set page title
st.title("🧱 Crack Detection in Brick Structures")

with st.spinner("Model loading..."):
    model = YOLO("best.pt")

def get_max_width(mask):
    skeleton = skeletonize(mask > 0)
    coords = np.argwhere(skeleton)
    max_width = 0

    for y, x in coords[::5]:  # Sample every 5th pixel for performance
        if 1 < y < mask.shape[0] - 2 and 1 < x < mask.shape[1] - 2:
            dy = int(int(skeleton[y+1, x]) - int(skeleton[y-1, x]))
            dx = int(int(skeleton[y, x+1]) - int(skeleton[y, x-1]))

            if dx == 0 and dy == 0:
                continue

            length = math.hypot(dx, dy)
            dx, dy = dx / length, dy / length

            pdx, pdy = -dy, dx
            profile_half_length = 20

            x0 = int(x - pdx * profile_half_length)
            y0 = int(y - pdy * profile_half_length)
            x1 = int(x + pdx * profile_half_length)
            y1 = int(y + pdy * profile_half_length)

            rr, cc = line(y0, x0, y1, x1)
            rr = np.clip(rr, 0, mask.shape[0]-1)
            cc = np.clip(cc, 0, mask.shape[1]-1)
            profile = mask[rr, cc]

            width = np.count_nonzero(profile)
            if width > max_width:
                max_width = width

    return max_width

# Upload video
uploaded_file = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])

# Add inputs for real-world dimensions
frame_width_mm = st.number_input("Enter frame width in mm:", min_value=1.0, value=200.0)
frame_height_mm = st.number_input("Enter frame height in mm:", min_value=1.0, value=100.0)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name

    st.video(input_video_path)

    if st.button("🚀 Proceed Prediction"):
        with st.spinner("Running predictions..."):
            cap = cv2.VideoCapture(input_video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # Calculate scale factors
            scale_x = frame_width_mm / width
            scale_y = frame_height_mm / height
            scale_avg = (scale_x + scale_y) / 2  # average scale for length approximation

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, task='segment', conf=0.25, save=False, stream=False, verbose=False)
                annotated_frame = results[0].plot()

                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()

                    for mask in masks:
                        binary_mask = (mask * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for cnt in contours:
                            rect = cv2.minAreaRect(cnt)
                            (x, y), (w, h), angle = rect
                            length_px = max(w, h)
                            min_width_px = min(w, h)

                            max_width_px = get_max_width(binary_mask)
                            skeleton = skeletonize(binary_mask > 0)
                            crack_length_px = np.count_nonzero(skeleton)

                            # Convert to mm
                            crack_length_mm = crack_length_px * scale_avg
                            max_width_mm = max_width_px * scale_avg
                            length_mm = length_px * scale_avg
                            width_mm = min_width_px * scale_avg

                            # Prepare text lines
                            text1 = f"Crack L:{crack_length_px:.1f}px, Max_W:{max_width_px:.1f}px"
                            text2 = f" Box  L:{length_px:.1f}px, W:{min_width_px:.1f}px"
                            text3 = f"Crack L:{crack_length_mm:.1f}mm, Max_W:{max_width_mm:.1f}mm"
                            text4 = f" Box  L:{length_mm:.1f}mm, W:{width_mm:.1f}mm"

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.0
                            thickness = 2

                            # Calculate total height of 4 lines
                            (tw1, th1), bl1 = cv2.getTextSize(text1, font, font_scale, thickness)
                            (tw2, th2), bl2 = cv2.getTextSize(text2, font, font_scale, thickness)
                            (tw3, th3), bl3 = cv2.getTextSize(text3, font, font_scale, thickness)
                            (tw4, th4), bl4 = cv2.getTextSize(text4, font, font_scale, thickness)

                            total_height = th1 + th2 + th3 + th4 + bl1 + bl2 + bl3 + bl4 + 15
                            text_width = max(tw1, tw2, tw3, tw4)

                            top_left = (int(x), int(y) - total_height)
                            bottom_right = (int(x) + text_width, int(y))

                            # Draw background box
                            cv2.rectangle(annotated_frame, top_left, bottom_right, (255, 255, 255), -1)

                            # Draw texts
                            y_pos = int(y) - total_height + th1
                            cv2.putText(annotated_frame, text1, (int(x), y_pos), font, font_scale, (0, 255, 0), thickness)
                            y_pos += th2 + bl2
                            cv2.putText(annotated_frame, text2, (int(x), y_pos), font, font_scale, (0, 255, 0), thickness)
                            y_pos += th3 + bl3
                            cv2.putText(annotated_frame, text3, (int(x), y_pos), font, font_scale, (0, 255, 0), thickness)
                            y_pos += th4 + bl4
                            cv2.putText(annotated_frame, text4, (int(x), y_pos), font, font_scale, (0, 255, 0), thickness)

                out.write(annotated_frame)
                frame_count += 1

            cap.release()
            out.release()

        st.success("✅ Prediction Complete!")
        st.success(f"Total frames processed: {frame_count}")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Predicted Video", f, file_name="predicted_video.mp4")

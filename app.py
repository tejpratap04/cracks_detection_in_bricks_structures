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
st.title("🧱 Crack Detection in Brick Structures (YOLOv8-Seg)")

with st.spinner("Model loading..."):
    model = YOLO("best.pt")

def get_max_width(mask):
    skeleton = skeletonize(mask > 0)
    coords = np.argwhere(skeleton)
    max_width = 0

    for y, x in coords[::5]:  # Sample every 5th pixel for performance
        if 1 < y < mask.shape[0] - 2 and 1 < x < mask.shape[1] - 2:
            #dy = int(skeleton[y+1, x] - skeleton[y-1, x])
            #dx = int(skeleton[y, x+1] - skeleton[y, x-1])

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

uploaded_file = st.file_uploader("Upload a video file (.mp4)", type=["mp4"])

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

                            # Compute max width and skeleton length
                            max_width_px = get_max_width(binary_mask)
                            skeleton = skeletonize(binary_mask > 0)
                            crack_length_px = np.count_nonzero(skeleton)

                            # Draw red box
                            #box = cv2.boxPoints(rect)
                            #box = box.astype(np.int32)
                            #cv2.drawContours(annotated_frame, [box], 0, (0, 0, 255), 2)

                            # Prepare two lines of text
                            text1 = f"Crack L:{crack_length_px:.1f}px, Max_W:{max_width_px:.1f}px"
                            text2 = f" Box  L:{length_px:.1f}px, W:{min_width_px:.1f}px"

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.0
                            thickness = 2

                            (text_width1, text_height1), baseline1 = cv2.getTextSize(text1, font, font_scale, thickness)
                            (text_width2, text_height2), baseline2 = cv2.getTextSize(text2, font, font_scale, thickness)
                            total_height = text_height1 + text_height2 + baseline1 + baseline2 + 10

                            top_left = (int(x), int(y) - total_height)
                            bottom_right = (int(x) + max(text_width1, text_width2), int(y))

                            # Draw background box for better readability
                            cv2.rectangle(annotated_frame, top_left, bottom_right, (255, 255, 255), -1)

                            # Draw both lines
                            cv2.putText(annotated_frame, text1, (int(x), int(y) - text_height2 - baseline2 - 5), font, font_scale, (0, 255, 0), thickness)
                            cv2.putText(annotated_frame, text2, (int(x), int(y) - baseline2), font, font_scale, (0, 255, 0), thickness)

                out.write(annotated_frame)
                frame_count += 1


            cap.release()
            out.release()

        st.success("✅ Prediction Complete!")
        st.success(f"Total frames processed: {frame_count}")

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Predicted Video", f, file_name="predicted_video.mp4")

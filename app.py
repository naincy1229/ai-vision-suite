# app.py
import streamlit as st
import numpy as np
from PIL import Image
import os
from io import BytesIO
from utils import (
    detect_faces, detect_objects, detect_text, detect_emotions,
    estimate_age_gender, extract_image_info, apply_bokeh,
    apply_cartoon, apply_sketch, generate_caption
)

# App Configuration
st.set_page_config(page_title="AI Vision Suite", page_icon="🧠", layout="wide")

# Load Custom CSS
if os.path.exists("assets/style.css"):
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("""
<h1 class='title'>🧠 AI Vision Suite</h1>
<p class='subtitle'>All-in-one image processing: detect faces, objects, text, age, emotions, and more!</p>
""", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    img_array = np.array(image)

    # Feature Selection
    st.markdown("### 🧪 Choose AI Features to Apply")
    col1, col2, col3 = st.columns(3)
    with col1:
        do_face = st.checkbox("🧍 Face Detection")
        do_object = st.checkbox("📦 Object Detection")
        do_text = st.checkbox("🔤 Text Detection")
    with col2:
        do_emotion = st.checkbox("😊 Emotion Detection")
        do_age_gender = st.checkbox("🎂 Age & Gender Estimation")
        do_info = st.checkbox("🖼️ Image Info Extraction")
    with col3:
        do_bokeh = st.checkbox("🎯 Bokeh Effect")
        do_cartoon = st.checkbox("🎨 Cartoon Filter")
        do_sketch = st.checkbox("📝 Sketch Filter")

    do_caption = st.checkbox("🧾 Generate AI Caption")

    if st.button("🚀 Process Image"):
        processed_img = img_array.copy()
        desc = []

        # Process Features
        if do_face:
            processed_img, count = detect_faces(processed_img)
            desc.append(f"Detected {count} face(s)")

        if do_object:
            processed_img, obj_count = detect_objects(processed_img)
            desc.append(f"Detected {obj_count} object(s)")

        if do_text:
            text_result = detect_text(processed_img)
            if text_result:
                st.markdown(f"**📝 Text Found:** `{text_result}`")
            desc.append("Detected text")

        if do_emotion:
            emotion = detect_emotions(processed_img)
            st.markdown(f"**😊 Emotion:** `{emotion}`")
            desc.append(f"Emotion: {emotion}")

        if do_age_gender:
            age, gender = estimate_age_gender(processed_img)
            st.markdown(f"**🎂 Age:** {age}, **Gender:** {gender}")
            desc.append(f"Age: {age}, Gender: {gender}")

        if do_info:
            info = extract_image_info(processed_img)
            st.markdown(f"**📐 Info:** {info}")
            desc.append("Image info extracted")

        if do_bokeh:
            processed_img = apply_bokeh(processed_img)
            desc.append("Applied bokeh")

        if do_cartoon:
            processed_img = apply_cartoon(processed_img)
            desc.append("Applied cartoon filter")

        if do_sketch:
            processed_img = apply_sketch(processed_img)
            desc.append("Applied sketch filter")

        if do_caption:
            caption = generate_caption(processed_img)
            st.markdown(f"**🧠 AI Caption:** {caption}")
            desc.append("AI-generated caption")

        # Show Final Result
        st.image(processed_img, caption="🧪 Processed Image", use_container_width=True)

        # Download Button
        result_pil = Image.fromarray(processed_img)
        buffer = BytesIO()
        result_pil.save(buffer, format="PNG")
        st.download_button("⬇️ Download Result", buffer.getvalue(), file_name="ai_result.png", mime="image/png")

        # Summary
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### ✅ Summary of Applied Features:")
        st.markdown("\n".join([f"- {d}" for d in desc]))

else:
    st.info("📤 Upload an image above to get started!")

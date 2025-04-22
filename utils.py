# utils.py
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 150), 2)
    return image, len(faces)

def detect_objects(image):
    # Dummy implementation
    h, w, _ = image.shape
    cv2.rectangle(image, (w//4, h//4), (3*w//4, 3*h//4), (255, 0, 0), 2)
    return image, 1

def detect_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def detect_emotions(image):
    # Dummy emotion detection
    return "Happy"

def estimate_age_gender(image):
    # Dummy estimation
    return 25, "Male"

def extract_image_info(image):
    h, w, c = image.shape
    return f"Width: {w}, Height: {h}, Channels: {c}"

def apply_bokeh(image):
    # Simple blur effect
    return cv2.GaussianBlur(image, (21, 21), 0)

def apply_cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

def generate_caption(image):
    # Dummy caption
    return "An image containing faces and objects."

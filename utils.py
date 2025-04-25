import cv2
import numpy as np
import pytesseract
from PIL import Image

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect faces
def detect_faces(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 150), 2)  # Draw rectangles around faces
        return image, len(faces)
    except Exception as e:
        return image, 0

# Function to detect objects (dummy implementation, extend as needed)
def detect_objects(image):
    try:
        h, w, _ = image.shape
        cv2.rectangle(image, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 0, 0), 2)  # Dummy object detection
        return image, 1
    except Exception:
        return image, 0

# Function to detect text using pytesseract
def detect_text(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        text = pytesseract.image_to_string(gray)  # Extract text from the image
        return text.strip()
    except Exception:
        return "Error detecting text."

# Function to simulate emotion detection
def detect_emotions(image):
    # Replace with a proper emotion detection model if needed
    return "Happy"  # Placeholder, replace with actual emotion detection logic.

# Function to estimate age and gender (dummy implementation, replace with actual model)
def estimate_age_gender(image):
    # Placeholder values, replace with actual age/gender estimation model
    return 25, "Male"

# Function to extract image info (dimensions and color channels)
def extract_image_info(image):
    try:
        h, w, c = image.shape
        return f"Width: {w}, Height: {h}, Channels: {c}"
    except Exception:
        return "Could not extract image info."

# Function to apply bokeh (blur) effect
def apply_bokeh(image):
    try:
        return cv2.GaussianBlur(image, (21, 21), 0)  # Apply Gaussian blur
    except Exception:
        return image

# Function to apply cartoon effect
def apply_cartoon(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)  # Apply median blur
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)  # Apply edge detection
        color = cv2.bilateralFilter(image, 9, 250, 250)  # Apply bilateral filter
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    except Exception:
        return image

# Function to apply sketch effect
def apply_sketch(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray  # Invert the grayscale image
        blur = cv2.GaussianBlur(inv, (21, 21), 0)  # Apply Gaussian blur
        sketch = cv2.divide(gray, 255 - blur, scale=256)  # Combine grayscale with blurred version
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)  # Convert back to RGB for display
    except Exception:
        return image

# Function to generate a caption for the image (dummy placeholder)
def generate_caption(image):
    try:
        # Placeholder for actual image captioning model
        return "This is a photo containing faces and objects."
    except Exception:
        return "Error generating caption."

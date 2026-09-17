# ============================================================
# FER2013 Emotion Detection Web App
# Gradio + EfficientNetB0
# ============================================================

import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms, models

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "EfficientNetB0_final_best.pth"

IMAGE_SIZE = 224
NUM_CLASSES = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "Angry",
    "Disgusted",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]

# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ============================================================
# MODEL
# ============================================================

class EfficientNetB0FER(nn.Module):

    def __init__(self, num_classes=7):

        super(EfficientNetB0FER, self).__init__()

        self.backbone = models.efficientnet_b0(
            weights=None
        )

        feature_dim = self.backbone.classifier[1].in_features

        # MUST MATCH TRAINED MODEL
        self.backbone.classifier = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(feature_dim, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        return self.backbone(x)
# ============================================================
# LOAD MODEL
# ============================================================

model = EfficientNetB0FER(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:

        return None, image

    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face = image[y:y+h, x:x+w]

    image_with_box = image.copy()

    cv2.rectangle(
        image_with_box,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        2
    )

    return face, image_with_box


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(image):

    if image is None:

        return (
            None,
            "Please upload an image.",
            None
        )

    # FIX FOR WEBCAM INPUT
    image = np.array(image)

    # Detect face
    face, boxed_image = detect_face(image)

    if face is None:

        return (
            image,
            "No face detected.",
            None
        )

    # ========================================================
    # PREPROCESS
    # ========================================================

    face_tensor = transform(face)

    face_tensor = face_tensor.unsqueeze(0)

    face_tensor = face_tensor.to(DEVICE)

    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        outputs = model(face_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    probabilities = probabilities.cpu().numpy()

    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index]

    # ========================================================
    # RESULT TEXT
    # ========================================================

    result_text = (
        f"Predicted Emotion: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )

    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        CLASS_NAMES,
        probabilities
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Probability")

    ax.set_title("Emotion Probabilities")

    for bar, prob in zip(bars, probabilities):

        ax.text(
            bar.get_x() + bar.get_width()/2,
            prob + 0.01,
            f"{prob:.2f}",
            ha='center'
        )

    plt.tight_layout()

    return (
        boxed_image,
        result_text,
        fig
    )
# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="FER2013 Emotion Detection"
) as demo:

    gr.Markdown(
        """
        # Facial Emotion Recognition

        Upload an image or use your webcam.

        The model predicts:

        - Angry
        - Disgusted
        - Fearful
        - Happy
        - Neutral
        - Sad
        - Surprised
        """
    )

    with gr.Row():

        with gr.Column():

            # input_image = gr.Image(
            #     type="numpy",
            #     sources=["webcam"],
            #     streaming=True,
            #     label="Live Webcam"
            # )
            input_image = gr.Image(
                type="numpy",
                sources=["upload", "webcam"],
                label="Upload Image or Webcam"
            )
            # regular upload + webcam input --- IGNORE ---
            predict_button = gr.Button(
                "Predict Emotion"
            )

        with gr.Column():

            output_image = gr.Image(
                label="Detected Face"
            )

            output_text = gr.Textbox(
                label="Prediction"
            )

    # probability_plot = gr.Plot(
    #     label="Emotion Probability Distribution"
    # )

    predict_button.click(
        fn=predict_emotion,
        inputs=input_image,
        outputs=[
            output_image,
            output_text,
            # probability_plot
        ]
    )
    # input_image.stream(
    #     fn=predict_emotion,
    #     inputs=input_image,
    #     outputs=[
    #         output_image,
    #         output_text,
    #         probability_plot
    #     ]
    # )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        share=True
    )
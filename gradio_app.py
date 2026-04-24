import gradio as gr
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import torch
from PIL import Image
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get list of images
image_folder = "images"
images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

# Get list of models
model_folder = "models"
models = [f for f in os.listdir(model_folder) if f.endswith('.pt')]

# Global model cache
model_cache = {}

def load_model(model_path):
    if model_path not in model_cache:
        model_cache[model_path] = YOLO(model_path)
    return model_cache[model_path]

def detectar(image_path, model, labels=[0,1], conf=0.4):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, conf=conf, classes=labels, verbose=False)
    boxes = results[0].boxes
    detections = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf_val = float(box.conf[0])
            cls = int(box.cls.cpu().item())
            detections.append({"rect": (x1, y1, x2, y2), "conf": conf_val, "cls": cls})
    return img_rgb, detections

def plot_detections(img, detections, model):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.imshow(img)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(detections), 1)))
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["rect"]
        conf = det["conf"]
        label = model.names[det["cls"]]
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 6, f"{label.title()} {conf:.2f}", color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
    ax.axis("off")
    plt.tight_layout()
    # Save to buffer and return PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)
    plt.close(fig)
    return pil_img

def process_image(image_path, model_path):
    model = load_model(model_path)
    img, dets = detectar(image_path, model)
    result_img = plot_detections(img, dets, model)
    return result_img

# Gradio functions
def update_model(model_name, current_index):
    model_path = os.path.join(model_folder, model_name)
    image_path = os.path.join(image_folder, images[current_index])
    result = process_image(image_path, model_path)
    image_name = images[current_index]
    return result, image_name

def next_image(current_index, model_name):
    new_index = (current_index + 1) % len(images)
    image_path = os.path.join(image_folder, images[new_index])
    model_path = os.path.join(model_folder, model_name)
    preview = cv2.imread(image_path)
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    result = process_image(image_path, model_path)
    image_name = images[new_index]
    return new_index, preview, result, image_name

def prev_image(current_index, model_name):
    new_index = (current_index - 1) % len(images)
    image_path = os.path.join(image_folder, images[new_index])
    model_path = os.path.join(model_folder, model_name)
    preview = cv2.imread(image_path)
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    result = process_image(image_path, model_path)
    image_name = images[new_index]
    return new_index, preview, result, image_name

# Initial values
initial_index = 0
initial_model = models[0]
initial_image_name = images[initial_index]
initial_image_path = os.path.join(image_folder, images[initial_index])
initial_model_path = os.path.join(model_folder, initial_model)
initial_preview = cv2.imread(initial_image_path)
initial_preview = cv2.cvtColor(initial_preview, cv2.COLOR_BGR2RGB)
initial_result = process_image(initial_image_path, initial_model_path)

# Interface
with gr.Blocks() as demo:
    gr.Markdown("# Aplicação para Detecção de armas")
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(choices=models, value=initial_model, label="Selecione o modelo")
            image_display = gr.Image(value=initial_preview, label="Current Image", height=400, width=600)
            image_name_display = gr.Textbox(label="Nome da Imagem Atual", value=initial_image_name, interactive=False)
            with gr.Row():
                prev_btn = gr.Button("Anterior")
                next_btn = gr.Button("Próxima")
        with gr.Column():
            result_display = gr.Image(value=initial_result, label="Resultado da Detecção", height=600, width=800)
    
    # State
    index_state = gr.State(initial_index)
    
    # Events
    model_dropdown.change(update_model, inputs=[model_dropdown, index_state], outputs=[result_display, image_name_display])
    next_btn.click(next_image, inputs=[index_state, model_dropdown], outputs=[index_state, image_display, result_display, image_name_display])
    prev_btn.click(prev_image, inputs=[index_state, model_dropdown], outputs=[index_state, image_display, result_display, image_name_display])

if __name__ == "__main__":
    demo.launch()
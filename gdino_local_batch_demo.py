import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2





"""
Hyperparameters
"""
IMAGE_FOLDER = "/Dataset/rio10/scene02/seq02/seq02_01/images"
JSON_PATH = "/Dataset/rio10/scene02/seq02/seq02_01/frame_instance_visibility.json"  # Frame-object mapping JSON
MODEL_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
OUTPUT_DIR = "/Dataset/rio10/scene02/seq02/seq02_01/gdino_static_objects"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load frame-wise object mapping
with open(JSON_PATH, "r") as f:
    frame_object_dict = json.load(f)


model = load_model(MODEL_CONFIG, MODEL_CHECKPOINT)

# Process all images in the folder
for img_file in os.listdir(IMAGE_FOLDER):
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, img_file)
    image_source, image = load_image(image_path)

    # Get frame-specific text prompt
    image_name = os.path.basename(image_path)
    # text_prompt = ". ".join(frame_object_dict.get(image_name, ["object"])) + "."
    text_prompt = "wall . floor . ceiling ."
    print(f"Processing {image_name} with prompt: {text_prompt}")

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # print(boxes, logits, phrases)

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{Path(img_file).stem}_segmented.jpg"), annotated_frame)

    # Save detection results as JSON
    detection_data = {
        "image_path": image_path,
        "boxes": boxes.tolist(),
        "logits": logits.tolist(),
        "phrases": phrases
    }
    output_json_path = os.path.join(OUTPUT_DIR, f"{Path(img_file).stem}_results.json")
    with open(output_json_path, "w") as f:
        json.dump(detection_data, f, indent=4)

    print(f"Processed {img_file}")

print("Batch object detection completed.")

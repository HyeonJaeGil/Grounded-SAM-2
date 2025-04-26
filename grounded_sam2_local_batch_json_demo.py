import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

def get_prompt_from_file(file_path):
    """
    Reads instances.txt and returns a period-separated string of unique instance names.
    """
    with open(file_path, "r") as f:
        instances = {line.strip().split(" ", 1)[1] for line in f if " " in line.strip()}
    return ". ".join(sorted(instances)) + " ."

"""
Hyper parameters
"""
TEXT_PROMPT = get_prompt_from_file("/Dataset/rio10/scene02/seq02/seq02_01/instances.txt")
JSON_PATH = "/Dataset/rio10/scene02/seq02/seq02_01/frame_instance_visibility.json"  # Frame-object mapping JSON
IMG_DIR = "/Dataset/rio10/scene02/seq02/seq02_01/images"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/Dataset/rio10/scene02/seq02/seq02_01/gdino_sam2_static_objects")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load frame-visible object mapping
with open(JSON_PATH, "r") as f:
    frame_object_dict = json.load(f)


# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


def process_image(image_path):
    image_source, image = load_image(image_path)
    sam2_predictor.set_image(image_source)

    # Get frame-wise text prompt
    image_key = os.path.basename(image_path)
    if image_key in frame_object_dict:
        text_prompt = ". ".join(frame_object_dict[image_key]) + ". object ."
    else:
        text_prompt = TEXT_PROMPT
    text_prompt = "wall . floor . ceiling ."
    print(f"Processing {image_key} with prompt: {text_prompt}")

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        remove_combined=True,
    )
    
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # skip if no boxes are detected
    if len(input_boxes) == 0:
        output_image_path = OUTPUT_DIR / (Path(image_path).stem + "_segmented.jpg")
        cv2.imwrite(str(output_image_path), image_source)
        results = {
            "image_path": image_path,
            "annotations": [],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        if DUMP_JSON_RESULTS:
            output_json_path = OUTPUT_DIR / (Path(image_path).stem + "_results.json")
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)
        return
    
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    img = cv2.imread(image_path)
    detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=class_ids)
    
    annotated_frame = sv.BoxAnnotator().annotate(scene=img.copy(), detections=detections)
    annotated_frame = sv.LabelAnnotator().annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = sv.MaskAnnotator().annotate(scene=annotated_frame, detections=detections)
    
    output_image_path = OUTPUT_DIR / (Path(image_path).stem + "_segmented.jpg")
    cv2.imwrite(str(output_image_path), annotated_frame)
    
    if DUMP_JSON_RESULTS:
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        mask_rles = [single_mask_to_rle(mask) for mask in masks]
        results = {
            "image_path": image_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes.tolist(), mask_rles, scores.tolist())
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        output_json_path = OUTPUT_DIR / (Path(image_path).stem + "_results.json")
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
    
# Process all images in the directory
for img_file in os.listdir(IMG_DIR):
    if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        process_image(os.path.join(IMG_DIR, img_file))

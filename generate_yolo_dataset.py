import argparse
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
from tqdm import tqdm

def rle_to_polygon(rle, img_width, img_height):
    """
    Convert RLE mask to polygon coordinates normalized to 0-1 range.
    
    Args:
        rle: Run-length encoded mask
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of polygons, each polygon is a list of normalized coordinates [x1, y1, x2, y2, ...]
    """
    # Decode RLE to binary mask
    mask = mask_util.decode(rle)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < 10:
            continue
            
        # Simplify polygon
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Normalize and flatten coordinates
        if len(approx) >= 3:  # Need at least 3 points for a polygon
            poly = approx.squeeze().astype(np.float32)
            poly[:, 0] /= img_width  # Normalize x
            poly[:, 1] /= img_height  # Normalize y
            polygons.append(poly.flatten().tolist())
            
    return polygons

def main(args):
    print("--- Starting Grounded SAM 2 Batch Demo ---")

    # Unpack arguments
    TEXT_PROMPT = args.text_prompt
    IMAGE_FOLDER = args.image_folder
    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config
    GROUNDING_DINO_CONFIG = args.grounding_dino_config
    GROUNDING_DINO_CHECKPOINT = args.grounding_dino_checkpoint
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    DEVICE = args.device
    OUTPUT_DIR = Path(args.output_dir)
    DUMP_JSON_RESULTS = not args.no_dump_json
    YOLO_OUTPUT = args.yolo_output
    SPLIT_RATIO = args.split_ratio

    # For YOLO output, create proper directory structure
    if YOLO_OUTPUT:
        (OUTPUT_DIR / "images/train").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "images/val").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels/train").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels/val").mkdir(parents=True, exist_ok=True)
    
    # Global class registry for YOLO output
    class_registry = {}
    next_class_id = 0

    # create output directory structure
    print(f"Creating output directory structure at: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # build SAM2 image predictor
    print("Loading SAM 2 model...")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM 2 model loaded successfully.")

    # build grounding dino model
    print("Loading Grounding DINO model...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    print("Grounding DINO model loaded successfully.")

    # Get list of images to process
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in image_extensions]
    print(f"Found {len(image_files)} images in the folder.")
    
    # For YOLO output, shuffle and split images
    if YOLO_OUTPUT:
        import random
        random.shuffle(image_files)
        split_idx = int(len(image_files) * SPLIT_RATIO)
        train_files = set(image_files[:split_idx])
        val_files = set(image_files[split_idx:])
        print(f"Splitting dataset: {len(train_files)} train, {len(val_files)} val")

    # Loop over all images in the folder
    for image_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_FOLDER, image_name)
        base_image_name = os.path.splitext(image_name)[0]

        print(f"\nProcessing: {img_path}")
        
        try:
            image_source, image = load_image(img_path)
            print("Image loaded and transformed.")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

        sam2_predictor.set_image(image_source)

        print(f"Running Grounding DINO prediction with prompt: '{TEXT_PROMPT}'")
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )
        print(f"Found {len(boxes)} objects.")

        if len(boxes) == 0:
            print("No objects detected. Skipping to next image.")
            continue

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        print("Running SAM 2 mask prediction...")
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        print("Mask prediction complete.")

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels
        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, confidences)
        ]

        print("Visualizing and saving annotated images...")
        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        output_image_path = os.path.join(OUTPUT_DIR, f"{base_image_name}_annotated.jpg")
        cv2.imwrite(output_image_path, annotated_frame)
        print(f"Saved annotated image with boxes to: {output_image_path}")

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        output_mask_path = os.path.join(OUTPUT_DIR, f"{base_image_name}_masked.jpg")
        cv2.imwrite(output_mask_path, annotated_frame)
        print(f"Saved annotated image with masks to: {output_mask_path}")

        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        # Handle YOLO output format
        if YOLO_OUTPUT:
            print("Generating YOLO segmentation output...")
            
            # Register new classes and get their IDs
            for class_name in class_names:
                if class_name not in class_registry:
                    class_registry[class_name] = next_class_id
                    next_class_id += 1
            
            # Determine if this image goes to train or val
            if image_name in train_files:
                img_dest = "train"
            else:
                img_dest = "val"
            
            # Generate label lines for YOLO format
            label_lines = []
            for i in range(len(masks)):
                class_name = class_names[i]
                class_id = class_registry[class_name]
                
                # Convert mask to RLE then to polygon
                mask_rle = single_mask_to_rle(masks[i])
                polygons = rle_to_polygon(mask_rle, w, h)
                
                # Add each polygon as a separate line in YOLO format
                for poly in polygons:
                    if len(poly) >= 6:  # Need at least 3 points (6 coordinates)
                        line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in poly])
                        label_lines.append(line)
            
            # Write YOLO label file
            if label_lines:  # Only create file if there are detections
                label_path = os.path.join(OUTPUT_DIR, "labels", img_dest, f"{base_image_name}.txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(label_lines))
                print(f"Saved YOLO labels to: {label_path}")
            
            # Copy original image to appropriate images directory
            output_image_path = os.path.join(OUTPUT_DIR, "images", img_dest, image_name)
            img = cv2.imread(img_path)
            cv2.imwrite(output_image_path, img)
            print(f"Saved original image to: {output_image_path}")
        
        # Handle JSON output format (original functionality)
        elif DUMP_JSON_RESULTS:
            print("Dumping results to JSON file...")
            mask_rles = [single_mask_to_rle(mask) for mask in masks]
            input_boxes_list = input_boxes.tolist()
            scores_list = scores.tolist()
            results = {
                "image_path": img_path,
                "annotations": [
                    {
                        "class_name": class_name,
                        "bbox": box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for class_name, box, mask_rle, score in zip(class_names, input_boxes_list, mask_rles, scores_list)
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }
            output_json_path = os.path.join(OUTPUT_DIR, f"{base_image_name}_results.json")
            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved JSON results to: {output_json_path}")

    # Generate data.yaml for YOLO output
    if YOLO_OUTPUT:
        print("Generating data.yaml...")
        # Sort class registry by ID for consistent ordering
        sorted_classes = sorted(class_registry.items(), key=lambda x: x[1])
        
        data_yaml_content = f"""path: .
train: images/train
val: images/val

names:"""
        for class_name, class_id in sorted_classes:
            data_yaml_content += f"\n  {class_id}: {class_name}"
        
        data_yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
        with open(data_yaml_path, "w") as f:
            f.write(data_yaml_content)
        print(f"Saved data.yaml to: {data_yaml_path}")
        
        # Print class registry summary
        print("Class registry:")
        for class_name, class_id in sorted_classes:
            print(f"  {class_id}: {class_name}")

    print("--- Batch demo finished successfully! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded SAM 2 Batch Processing Demo")

    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for object detection.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing input images.")
    
    parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt", help="Path to the SAM 2 checkpoint.")
    parser.add_argument("--sam2_model_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to the SAM 2 model config.")
    
    parser.add_argument("--grounding_dino_config", type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to the Grounding DINO config.")
    parser.add_argument("--grounding_dino_checkpoint", type=str, default="gdino_checkpoints/groundingdino_swint_ogc.pth", help="Path to the Grounding DINO checkpoint.")
    
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box threshold for Grounding DINO.")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for Grounding DINO.")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the models on.")
    parser.add_argument("--output_dir", type=str, default="outputs/batch_run", help="Directory to save the output.")
    
    parser.add_argument("--no_dump_json", action="store_true", help="Do not dump results to a JSON file.")
    parser.add_argument("--yolo_output", action="store_true", help="Output in YOLO segmentation format instead of JSON.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Train/validation split ratio (default: 0.8)")

    args = parser.parse_args()
    main(args)

import argparse
import json
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import os
from datetime import datetime
import numpy as np  # For image validation

LABELS = ['bare_hand', 'gloved_hand']  # Order must match data.yaml names!

def validate_image(img_path):
    """Quick check if image can be loaded (prevents PIL errors)."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        # Basic shape check
        if img.shape[0] == 0 or img.shape[1] == 0:
            return False
        return True
    except Exception:
        return False

def infer_folder(model, input_folder, output_folder, logs_folder, conf_thresh=0.25, iou=0.45, batch=1, device=''):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    logs_folder = Path(logs_folder)

    output_folder.mkdir(parents=True, exist_ok=True)
    logs_folder.mkdir(parents=True, exist_ok=True)

    images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))
    if not images:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(images)} images. Validating and processing with batch size {batch}...")

    processed_count = 0
    skipped_count = 0
    valid_images = []

    # Pre-validate images to filter bad ones early
    for img_path in images:
        if validate_image(img_path):
            valid_images.append(img_path)
        else:
            print(f"Skipped invalid image (pre-check): {img_path.name}")
            skipped_count += 1

    images = valid_images  # Only process valid ones
    if not images:
        print("No valid images to process!")
        return

    print(f"Processing {len(images)} valid images...")

    # Process in small batches or single (to avoid batch-wide failures)
    batch_size = min(batch, len(images))  # Cap batch size
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        try:
            # Predict on batch
            batch_results = model.predict(
                source=[str(img) for img in batch_images], 
                conf=conf_thresh, 
                iou=iou, 
                device=device, 
                batch=batch_size, 
                save=False,
                verbose=False  # Reduce YOLO chatter
            )

            # Process each result in batch
            for idx, (img_path, res) in enumerate(zip(batch_images, batch_results)):
                detections = []
                if hasattr(res, 'boxes') and len(res.boxes) > 0:
                    for box in res.boxes:
                        xyxy = box.xyxy.tolist()[0]
                        conf = float(box.conf.tolist()[0])
                        cls_id = int(box.cls.tolist()[0])
                        if cls_id < len(LABELS):  # Ensure valid class
                            label = LABELS[cls_id]
                            x1, y1, x2, y2 = [float(v) for v in xyxy]
                            detections.append({
                                'label': label,
                                'confidence': round(float(conf), 4),  # Match example format
                                'bbox': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                            })

                # Annotate and save image (once per image)
                try:
                    annotated = res.plot()  # Draws boxes/labels on image
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    out_img_path = output_folder / img_path.name
                    success = cv2.imwrite(str(out_img_path), annotated_bgr)
                    if not success:
                        print(f"Warning: Failed to save {img_path.name}")
                except Exception as save_err:
                    print(f"Error annotating/saving {img_path.name}: {save_err}")
                    continue

                # Log JSON per image
                log = {'filename': img_path.name, 'detections': detections}
                log_file = logs_folder / f"{img_path.stem}.json"
                try:
                    with open(log_file, 'w') as f:
                        json.dump(log, f, indent=2)
                except Exception as log_err:
                    print(f"Error logging {img_path.name}: {log_err}")

                processed_count += 1
                print(f"Processed {img_path.name}: {len(detections)} detections")

        except Exception as batch_err:
            # If batch fails (rare PIL error on a valid image), fallback to single-image
            print(f"Batch failed (error: {batch_err}). Falling back to single-image mode for remaining...")
            for remaining_img in batch_images:
                try:
                    single_res = model.predict(source=str(remaining_img), conf=conf_thresh, iou=iou, device=device, save=False, verbose=False)
                    res = single_res[0]  # Single result

                    detections = []
                    if hasattr(res, 'boxes') and len(res.boxes) > 0:
                        for box in res.boxes:
                            xyxy = box.xyxy.tolist()[0]
                            conf = float(box.conf.tolist()[0])
                            cls_id = int(box.cls.tolist()[0])
                            if cls_id < len(LABELS):
                                label = LABELS[cls_id]
                                x1, y1, x2, y2 = [float(v) for v in xyxy]
                                detections.append({
                                    'label': label,
                                    'confidence': round(float(conf), 4),
                                    'bbox': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                                })

                    # Annotate/save/log 
                    annotated = res.plot()
                    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    out_img_path = output_folder / remaining_img.name
                    cv2.imwrite(str(out_img_path), annotated_bgr)

                    log = {'filename': remaining_img.name, 'detections': detections}
                    log_file = logs_folder / f"{remaining_img.stem}.json"
                    with open(log_file, 'w') as f:
                        json.dump(log, f, indent=2)

                    processed_count += 1
                    print(f"Processed (single) {remaining_img.name}: {len(detections)} detections")
                except Exception as single_err:
                    print(f"Skipped corrupted image (during prediction): {remaining_img.name} - {single_err}")
                    skipped_count += 1

    print(f"Inference complete! Processed: {processed_count}, Skipped: {skipped_count}")

def train_model(weights, data_yaml, epochs, device, project):
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}. Download dataset and create it!")
    
    print('Starting training...')
    model = YOLO(weights)
    # Basic augmentation for robustness (handles lighting/angles in factory cams)
    model.train(
        data=data_yaml, 
        epochs=epochs, 
        device=device, 
        project=project, 
        name='glove_train',
        imgsz=640,
        augment=True,  
        mosaic=1.0,
        mixup=0.1
    )
    print('Training finished! Trained model saved to', project)

def parse_args():
    parser = argparse.ArgumentParser(description="Glove Detection Pipeline")
    parser.add_argument('--mode', default='infer', choices=['train', 'infer'], required=False, help="Mode: train or infer")
    parser.add_argument('--weights', default='yolov8n.pt', help="Path to model weights (e.g., trained best.pt)")
    parser.add_argument('--data', default=r'C:\Folder\New folder\Sharon\'s_files\Sharon-s-Code\Glove_Detection_Submission\Part_1_Glove_Detection\data.yaml', help="Path to data.yaml for training")
    parser.add_argument('--input', default=r"C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Glove_Detection_Submission\Part_1_Glove_Detection\dataset\images\train", help="Input folder of images for inference")
    parser.add_argument('--output', default='./output', help="Output folder for annotated images")
    parser.add_argument('--logs', default='./logs', help="Folder for JSON logs")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")
    parser.add_argument('--iou', type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument('--epochs', type=int, default=20, help="Epochs for training (reduced for quick testing)")
    parser.add_argument('--batch', type=int, default=1, help="Batch size for inference/training (set to 1 for safety with corrupted images)")
    parser.add_argument('--device', default='', help="Device: '' (auto), 'cpu', or '0' (GPU)")
    args = parser.parse_args()
    print(f"Args parsed: mode={args.mode}, input={args.input}, output={args.output}")  # Debug print
    return args

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Execution started at {timestamp} | Mode: {args.mode} | Device: {args.device or 'auto'}")

    # Auto-detect device if not specified (bonus for reliability)
    if not args.device:
        args.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {args.device}")

    if args.mode == 'train':
        train_model(args.weights, args.data, args.epochs, args.device, project='./runs')
    else:
        if not os.path.exists(args.weights):
            print(f"Warning: Weights not found at {args.weights}. Downloading/using default (yolov8n.pt) - detections may be inaccurate without training!")
            args.weights = 'yolov8n.pt'  # Auto-download if missing
        model = YOLO(args.weights)
        infer_folder(model, args.input, args.output, args.logs,
                     conf_thresh=args.conf, iou=args.iou, batch=args.batch, device=args.device)
    print("Done!")

if __name__ == '__main__':
    main()

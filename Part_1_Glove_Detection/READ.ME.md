# 🧤 Glove Detection – Safety Compliance System
This project implements a safety compliance system for detecting gloved vs. bare hands in factory images using YOLOv8 object detection. The script supports training and inference, with annotated outputs and JSON logs as required.

##  Dataset Name and Source
**Dataset Name:** Gloved vs Ungloved Hands Dataset  
**Source:** Combined from  
- [Kaggle Public Dataset](https://www.kaggle.com/datasets)  
- [Roboflow Universe Public Dataset](https://universe.roboflow.com)  

Images were cleaned, labeled, and formatted into YOLO structure with custom `data.yaml`.

---

##  Model Used
**Model:** YOLOv8n (Nano Version)  
**Framework:** Ultralytics YOLOv8  
**Base Weights:** `yolov8n.pt` (Pretrained on COCO dataset)  
**Language:** Python 3.11  
**Device:** CPU  
**Base Model**: YOLOv8n (nano variant: lightweight, fast for video stream deployment on factory cameras; 3.2M params, ~80 FPS on CPU).
**Fine-Tuning**: Trained from `yolov8n.pt` (pre-trained on COCO) for 20 epochs. Custom classes: bare_hand, gloved_hand.
**Inference Settings**: Confidence threshold 0.25, IoU 0.45 for NMS. Batch size 1 (for robustness to corrupted images).

YOLOv8n was chosen for its speed, small size, and reliable accuracy for real-time edge detection.

---

##  Preprocessing and Training

### **Preprocessing Steps**
- All images auto-resized to **640×640**.  
- Verified each image–label pair in YOLO format.  
- Removed corrupted and unlabeled images using validation script.  
- Split dataset automatically:
  - **Train:** 669 images  
  - **Validation:** 129 images  
  - **Test:** 50 images  
- Applied augmentations: random flip, rotation, brightness & contrast adjustment.

### **Training Details**
- Epochs: 50  
- Batch Size: 8  
- Optimizer: Adam  
- Learning Rate: Auto (default YOLOv8)  
- Evaluation Metric: mAP50 and class accuracy  

**How to Run and Training Command Example:**
1. **Setup**:
   - Clone/download this folder.
   - Install dependencies: `pip install -r requirements.txt` (CPU; for GPU, adjust torch).
   - Place `data.yaml` next to script (edit paths if needed).
   - Add test images to `./input/` (3-5 JPG/PNG of hands/gloves) or use full dataset.

2. **Train the Model** (Optional, ~30-60 min):

**Run Inference** (Generates outputs, <5 min for 3-5 images):
bash"
python Detection_script.py --mode infer --input "./input" --output "./output" --logs "./logs" --weights yolov8n.pt --conf 0.25 --batch 1

**Then Train** bash"
python Detection_script.py --mode train --data "C:\Folder\New folder\Sharon's_files\Sharon-s-Code\Glove_Detection_Submission\Part_1_Glove_Detection\data.yaml" --epochs 20 --batch 8

**Re-run inference with trained model:** 
python Detection_script.py --mode infer --input "./input" --weights "./runs/glove_train/weights/best.pt" --output "./output" --logs "./logs" --conf 0.25 --batch 1

**Run Inference** (Generates outputs, <5 min for 3-5 images):

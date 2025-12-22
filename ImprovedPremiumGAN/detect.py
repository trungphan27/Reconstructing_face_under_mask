
from ultralytics import YOLO
import cv2
import numpy as np
import os

class MaskDetector:
    def __init__(self, model_path=None):
        # Default to finding yolov8n.pt in the project root if not provided
        if model_path is None:
            # Assuming this file is in ImprovedPremiumGAN/detect.py
            # Root is two levels up: ../../yolov8n.pt? No, Unmask System/ImprovedPremiumGAN/detect.py
            # Root is Unmask System which contains yolov8n.pt
            base_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(base_dir) 
            model_path = os.path.join(root_dir, 'yolov8n.pt')
        
        self.model = YOLO(model_path)

    def detect_mask(self, image_path):
        """
        Detects mask region in the image.
        Returns: image (numpy BGR), box (x, y, w, h) of the mask/lower-face region.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not open or find the image: {image_path}")

        results = self.model(image_path, verbose=False)
        
        best_box = None
        max_conf = -1
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box format xywh
                b = box.xywh[0].cpu().numpy() # x_center, y_center, w, h
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # COCO classes: 0 is person.
                # We prioritize 'person' detection if we don't have a specific mask detector.
                if cls == 0 and conf > max_conf:
                    max_conf = conf
                    # Convert to Top-Left x,y, w, h
                    x_c, y_c, w, h = b
                    
                    face_x = int(x_c - w/2)
                    face_y = int(y_c - h/2)
                    face_w = int(w)
                    face_h = int(h)
                    
                    # Logic: Approximate mask as the lower half of the detected face (person face area)
                    # Note: YOLO detects full body 'person'. 
                    # If the image is a closeup (selfie), 'person' box is effectively the face/upper body.
                    # If it's a full body shot, this heuristic might fail (masking the legs?). 
                    # Ideally we want a FACE detector.
                    # But assuming the dataset/test images are cropped faces or selfies:
                    
                    mask_x = face_x + int(face_w * 0.15) # Indent 15% (matching dataset.py slightly better)
                    mask_y = face_y + int(face_h * 0.50) # Start from middle
                    mask_w = int(face_w * 0.70) # 70% width (15% margin on both sides ~ 30% total reduction? No, 85-15 = 70)
                    # dataset.py: x_start=0.15, x_end=0.85 -> width = 0.70
                    mask_h = int(face_h * 0.45) # 0.50 to 0.95 -> 0.45 height
                    
                    best_box = (mask_x, mask_y, mask_w, mask_h)
        
        # Fallback: If no person detected, assumes the whole image is a face 
        # and returns the fixed crop relative to image size
        if best_box is None:
            h, w = img.shape[:2]
            mask_y_start = int(h * 0.50)
            mask_h = int(h * 0.45)
            mask_x_start = int(w * 0.15)
            mask_w = int(w * 0.70)
            best_box = (mask_x_start, mask_y_start, mask_w, mask_h)

        return img, best_box

    def apply_blackout(self, img, box):
        """
        Black out the region defined by box.
        box: (x, y, w, h)
        """
        if box is None:
            return img 
            
        x, y, w, h = box
        img_h, img_w, _ = img.shape
        
        # Clip to image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Black out
        img[y:y+h, x:x+w] = 0
        return img

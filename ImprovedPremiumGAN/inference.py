
import torch
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

# Internal imports
# Assuming this file is inside ImprovedPremiumGAN/
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model import Generator
from detect import MaskDetector

def infer(image_path, gen_model=None, detector=None):
    """
    Performs inference on a single image.
    
    Args:
        image_path (str): Path to the input image (Ground Truth).
        gen_model (torch.nn.Module, optional): Loaded Generator model. If None, it will be loaded.
        detector (MaskDetector, optional): Loaded Detector. If None, it will be initialized.
        
    Returns:
        tuple: (detect_viz_img, gt_img, generated_img)
        All images are numpy arrays in BGR format (ready for cv2.imshow/imwrite).
    """
    
    # 1. Initialize models if not provided
    if detector is None:
        detector = MaskDetector()
        
    if gen_model is None:
        gen_model = Generator(in_channels=3).to(config.DEVICE)
        # Load latest checkpoint
        checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith('gen_') and f.endswith('.pth.tar')]
        if checkpoints:
            latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            cp_path = os.path.join(config.CHECKPOINT_DIR, latest_cp)
            checkpoint = torch.load(cp_path, map_location=config.DEVICE)
            gen_model.load_state_dict(checkpoint['state_dict'])
            gen_model.eval()
        else:
            print("Warning: No checkpoints found for Generator. Using random weights.")
            gen_model.eval()

    # 2. Process Input (Ground Truth)
    # Detect Mask Region (simulated or real dependent on detection logic)
    gt_img, box = detector.detect_mask(image_path) # gt_img is BGR (Original High Res)
    
    # Create "Detect Mask Region" Visualization on High Res
    detect_viz_img = gt_img.copy()
    if box:
        x, y, w, h = box
        cv2.rectangle(detect_viz_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Create Masked Input for GAN (High Res)
    masked_input_bgr = gt_img.copy()
    if box:
        masked_input_bgr = detector.apply_blackout(masked_input_bgr, box)
        
    # RESIZE EVERYTHING TO 128x128 (Model Native Resolution)
    # This ensures "sharpness" relative to the training samples and perfect alignment.
    # We use Inter_Area for shrinking which is generally better for quality.
    
    gt_small = cv2.resize(gt_img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    masked_small = cv2.resize(masked_input_bgr, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    detect_small = cv2.resize(detect_viz_img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # 4. Run Generator
    # Preprocess: BGR -> RGB -> PIL -> Tensor -> Normalize
    masked_pil = Image.fromarray(cv2.cvtColor(masked_small, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.ToTensor(), # Already resized to 128x128 above
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(masked_pil).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        output_tensor = gen_model(input_tensor)
        
    # 5. Post-process Output
    # Tensor -> Numpy -> Un-normalize -> BGR
    # Output is (1, 3, 128, 128)
    gen_img_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    gen_img_np = (gen_img_np * 0.5) + 0.5 # [-1, 1] -> [0, 1]
    gen_img_np = np.clip(gen_img_np, 0, 1)
    gen_img_np = (gen_img_np * 255).astype(np.uint8)
    generated_img = cv2.cvtColor(gen_img_np, cv2.COLOR_RGB2BGR)
    
    # Return 128x128 images
    return detect_small, gt_small, generated_img

if __name__ == "__main__":
    # Define directories
    test_dir = os.path.join(config.DATA_ROOT, "dataset", "test_images")
    output_dir = os.path.join(config.RESULT_DIR, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for test images in: {test_dir}")
    
    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not files:
            print("No images found in test_images directory.")
        
        # Initialize models once
        detector = MaskDetector()
        gen_model = Generator(in_channels=3).to(config.DEVICE)
        
        # Load latest checkpoint
        checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith('gen_') and f.endswith('.pth.tar')]
        if checkpoints:
            latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            cp_path = os.path.join(config.CHECKPOINT_DIR, latest_cp)
            checkpoint = torch.load(cp_path, map_location=config.DEVICE)
            gen_model.load_state_dict(checkpoint['state_dict'])
            gen_model.eval()
            print(f"Loaded generator from {latest_cp}")
        else:
            print("Warning: No checkpoints found for Generator. Using random weights.")
            gen_model.eval()
            
        for file_name in files:
            file_path = os.path.join(test_dir, file_name)
            print(f"Processing {file_path}...")
            
            try:
                # Run Inference
                det_img, gt, gen = infer(file_path, gen_model, detector)
                
                # Save Individual Results
                base_name = os.path.splitext(file_name)[0]
                
                # Option 1: Save separately
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_detect.jpg"), det_img)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_groundtruth.jpg"), gt)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_generated.jpg"), gen)
                
                # Option 2: Save combined for easier viewing
                # Resize to same height if needed (usually they are same size from infer)
                combined = np.hstack((det_img, gt, gen))
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_combined.jpg"), combined)
                
                print(f"Saved results for {file_name} in {output_dir}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                
    else:
        print(f"Test directory not found: {test_dir}")

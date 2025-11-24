import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# ============================================================
# 1. SETUP: Define Paths and Parameters
# ============================================================

# --- Path to your saved model ---
MODEL_PATH = "U-Net_Crop_model.h5"

# --- Path to your test data (created by your training script) ---
# NOTE: This assumes you are using the 'test' folder from your 'dataset_split' directory
BASE_TEST_DIR = r"C:\Users\hitan\Music\Project\dataset_split\test"
TEST_IMG_DIR = os.path.join(BASE_TEST_DIR, "images")
TEST_MASK_DIR = os.path.join(BASE_TEST_DIR, "masks")

# --- Path to save visual results ---
RESULTS_DIR = "test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Model Parameters (must match training) ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 8

# ============================================================
# 2. HELPER FUNCTIONS (Copied from your training script)
# ============================================================

def load_image_paths(image_dir, mask_dir):
    """Finds all matching image and mask pairs."""
    image_paths = []
    mask_paths = []
    image_files = sorted(os.listdir(image_dir))
    
    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        
        # This logic MUST match your training script
        base = os.path.splitext(filename)[0]
        mask_filename = base + ".png" # Assumes masks are .png
        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"WARNING: Mask missing for {filename}")
            
    return image_paths, mask_paths

def load_and_preprocess_image(img_path, mask_path, img_size=(256, 256)):
    """Loads and prepares a single image/mask pair for the data generator."""
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, img_size)
    mask = (mask > 128).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask

def data_generator(image_paths, mask_paths, batch_size, img_size=(256, 256)):
    """Yields batches of test images and masks."""
    num_samples = len(image_paths)
    
    # NOTE: No shuffling for evaluation, we process in order
    for i in range(0, num_samples, batch_size):
        batch_indices = range(i, min(i + batch_size, num_samples))
        batch_images, batch_masks = [], []

        for idx in batch_indices:
            img, mask = load_and_preprocess_image(image_paths[idx], mask_paths[idx], img_size)
            batch_images.append(img)
            batch_masks.append(mask)

        yield np.array(batch_images), np.array(batch_masks)

# ============================================================
# 3. MAIN ANALYSIS FUNCTION
# ============================================================
if __name__ == "__main__":
    
    # --- 1. Load the trained model ---
    print(f"Loading model from {MODEL_PATH}...")
    # CRITICAL: We must pass the custom mIoU metric when loading
    custom_objects = {'mIoU': MeanIoU(num_classes=2)}
    model = load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Model loaded successfully.")

    # --- 2. Load Test Data Paths ---
    print(f"Loading test data from {BASE_TEST_DIR}...")
    test_imgs, test_masks = load_image_paths(TEST_IMG_DIR, TEST_MASK_DIR)
    print(f"Found {len(test_imgs)} test images.")

    if len(test_imgs) == 0:
        print("Error: No test images found. Check your paths.")
    else:
        # --- PART 1: QUANTITATIVE ANALYSIS (The "Record") ---
        print("\n--- Starting Quantitative Analysis (Overall Score) ---")
        
        # Create the test generator
        test_gen = data_generator(test_imgs, test_masks, BATCH_SIZE, img_size=IMG_SIZE)
        test_steps = max(1, len(test_imgs) // BATCH_SIZE)

        # Run model.evaluate()
        results = model.evaluate(test_gen, steps=test_steps, verbose=1)
        
        print("\n--- Test Set Results (Recorded) ---")
        print(f"Test Loss:     {results[0]:.4f}")
        print(f"Test mIoU:     {results[1]:.4f}")
        print(f"Test Accuracy: {results[2]:.4f}")
        print("----------------------------------\n")

        # --- PART 2: QUALITATIVE ANALYSIS (Visuals) ---
        print(f"--- Starting Qualitative Analysis (Saving visuals to '{RESULTS_DIR}') ---")
        
        for img_path, mask_path in zip(test_imgs, test_masks):
            # 1. Load original image and true mask for display
            original_img = cv2.imread(img_path)
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 2. Pre-process the image for the model
            img_resized = cv2.resize(original_img, IMG_SIZE)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb / 255.0
            img_batch = np.expand_dims(img_norm, axis=0) # Add batch dim

            # 3. Predict
            pred_mask_norm = model.predict(img_batch)[0] # Get first (and only) mask
            
            # 4. Post-process the predicted mask
            pred_mask_thresh = (pred_mask_norm > 0.5).astype(np.uint8) * 255
            # Resize mask back to original image size
            pred_mask_resized = cv2.resize(pred_mask_thresh, (original_img.shape[1], original_img.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            # 5. Create overlay
            overlay = cv2.addWeighted(original_img_rgb, 0.6, 
                                      cv2.cvtColor(pred_mask_resized, cv2.COLOR_GRAY2RGB), 
                                      0.4, 0)
            
            # 6. Save the comparison plot
            fig, ax = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f"Analysis for: {os.path.basename(img_path)}", fontsize=16)

            ax[0, 0].imshow(original_img_rgb)
            ax[0, 0].set_title("Original Image")
            ax[0, 0].axis("off")

            ax[0, 1].imshow(true_mask, cmap='gray')
            ax[0, 1].set_title("Ground Truth Mask")
            ax[0, 1].axis("off")

            ax[1, 0].imshow(pred_mask_resized, cmap='gray')
            ax[1, 0].set_title("Predicted Mask (mIoU > 0.5)")
            ax[1, 0].axis("off")

            ax[1, 1].imshow(overlay)
            ax[1, 1].set_title("Prediction Overlay")
            ax[1, 1].axis("off")
            
            save_path = os.path.join(RESULTS_DIR, os.path.splitext(os.path.basename(img_path))[0] + "_analysis.png")
            plt.savefig(save_path)
            plt.close(fig) # Close figure to save memory

        print(f"--- Visual analysis complete! ---")
        print(f"All comparison images saved to '{RESULTS_DIR}' folder.")

        
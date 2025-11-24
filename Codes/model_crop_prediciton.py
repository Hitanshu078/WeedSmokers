import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# --- Parameters ---
SAVED_MODEL_PATH = "U-Net_Crop_model.h5"
IMG_WIDTH = 256
IMG_HEIGHT = 256

# --- !! UPDATE THIS PATH !! ---
# Change this to the image you want to test
TEST_IMAGE_PATH = r"C:\Users\hitan\Music\Project\dataset_split\test\images\2_image.jpg" # Example path


def load_and_preprocess_test_image(img_path, img_size):
    """Loads and preprocesses a single image for prediction."""
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    
    # Store original shape for resizing back
    original_shape = image.shape
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image / 255.0 # Normalize to [0, 1]
    
    # Add batch dimension (model expects 4D input: [batch, height, width, channels])
    image = np.expand_dims(image, axis=0)
    
    return image, original_shape

def post_process_mask(pred_mask, original_shape):
    """Converts model output to a displayable mask."""
    # Remove batch and channel dimensions
    pred_mask = np.squeeze(pred_mask)
    
    # Apply threshold (0.5) to get binary mask (0 or 1)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # Convert to 255 (white) for visualization
    pred_mask = pred_mask * 255
    
    # Resize back to original image size
    pred_mask_resized = cv2.resize(
        pred_mask, 
        (original_shape[1], original_shape[0]), # (width, height)
        interpolation=cv2.INTER_NEAREST # Use nearest neighbor for masks
    )
    
    return pred_mask_resized

def predict(model_path, image_path):
    """Loads model, predicts mask, and displays results."""
    
    # 1. Load the trained model
    print(f"Loading model from {model_path}...")
    # CRITICAL: You must pass the custom metric 'mIoU' when loading
    custom_objects = {'mIoU': MeanIoU(num_classes=2)}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # 2. Load and preprocess the image
    print(f"Loading and preprocessing image from {image_path}...")
    image_tensor, original_shape = load_and_preprocess_test_image(
        image_path, 
        (IMG_HEIGHT, IMG_WIDTH)
    )

    # 3. Make prediction
    print("Running prediction...")
    pred_mask_normalized = model.predict(image_tensor)

    # 4. Post-process the mask
    print("Post-processing mask...")
    final_mask = post_process_mask(pred_mask_normalized, original_shape)
    
    # 5. Display results
    print("Displaying results...")
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    mask_rgb = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(original_image_rgb, 0.6, mask_rgb, 0.4, 0)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(final_mask, cmap='gray')
    plt.title("Predicted Crop Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

    # --- Save the mask ---
    # Get the directory of the original image
    output_dir = os.path.dirname(image_path)
    # Get the base name of the original image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Create the new mask filename
    mask_filename = os.path.join(output_dir, f"{base_name}_predicted_mask.png")
    
    cv2.imwrite(mask_filename, final_mask)
    print(f"Predicted mask saved to {mask_filename}")

if __name__ == "__main__":
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: Model file not found at {SAVED_MODEL_PATH}")
    elif not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at {TEST_IMAGE_PATH}")
        print("Please update the 'TEST_IMAGE_PATH' variable in this script.")
    else:
        predict(SAVED_MODEL_PATH, TEST_IMAGE_PATH)
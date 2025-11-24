import cv2
import numpy as np
from PIL import Image
import glob
import os

# ============================================================
# Load dataset images
# ============================================================
def load_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg"))) + \
            sorted(glob.glob(os.path.join(folder, "*.png")))
    
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
            images.append(img)
    return images


# ============================================================
# Create mosaic using user-defined ROWS and COLS
# No interpolation, no resizing — original image detail preserved
# ============================================================
def create_image_mosaic(images, ROWS, COLS):
    h, w, ch = images[0].shape
    total_needed = ROWS * COLS
    total_available = len(images)

    # Pad missing tiles with black
    if total_available < total_needed:
        print(f"Warning: {total_available} images found, expected {total_needed}. Padding with blank tiles.")
        blank = np.zeros_like(images[0])
        for _ in range(total_needed - total_available):
            images.append(blank)

    mosaic = np.zeros((ROWS * h, COLS * w, 3), dtype=np.uint8)

    for idx in range(ROWS * COLS):
        img = images[idx]
        r = idx // COLS
        c = idx % COLS
        mosaic[r*h:(r+1)*h, c*w:(c+1)*w, :] = img

    return mosaic


# ============================================================
# Save final mosaic as TIFF
# ============================================================
def save_tif(mosaic, path):
    img = Image.fromarray(mosaic)
    img.save(path, format="TIFF")
    print(f"Saved farm mosaic TIFF → {path}")


# ============================================================
# ========================= MAIN ==============================
# ============================================================

DATASET_FOLDER = r"C:\Users\hitan\Music\Project\Raw_dataset_images"  # <-- SET YOUR DATASET
ROWS = 40       # <-- SET
COLS = 20        # <-- SET
OUTPUT = "FARM_MOSAIC.tif"

# Load all dataset images
dataset_images = load_images(DATASET_FOLDER)

# Build mosaic
farm_mosaic = create_image_mosaic(dataset_images, ROWS, COLS)

# Save high-resolution TIFF
save_tif(farm_mosaic, OUTPUT)

# Optional: Preview
import matplotlib.pyplot as plt
plt.figure(figsize=(14,14))
plt.imshow(farm_mosaic)
plt.axis("off")
plt.title("Farm Mosaic (Original Images)")
plt.show()

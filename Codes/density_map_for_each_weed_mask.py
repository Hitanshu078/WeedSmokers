import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --------------------------------------------------------
# USER INPUT
# --------------------------------------------------------
weed_mask_folder = r"C:\Users\hitan\Music\Project\WeedMasks"
output_folder = r"C:\Users\hitan\Music\Project\DensityMaps"

os.makedirs(output_folder, exist_ok=True)

# Patch size (adjust as needed)
PATCH_SIZE = 16   # common values: 16, 32, 64


# --------------------------------------------------------
# FUNCTION: CREATE DENSITY MAP
# --------------------------------------------------------
def create_density_map(mask, patch_size):
    h, w = mask.shape
    density = np.zeros((h // patch_size, w // patch_size), dtype=np.float32)

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = mask[i:i+patch_size, j:j+patch_size]
            weed_pixels = np.sum(patch == 255)

            density[i // patch_size, j // patch_size] = weed_pixels

    # Normalize between 0–255
    density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)
    return density_norm.astype(np.uint8)


# --------------------------------------------------------
# PROCESS EACH WEED MASK
# --------------------------------------------------------
for file in os.listdir(weed_mask_folder):
    if not file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
        continue

    mask_path = os.path.join(weed_mask_folder, file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"❌ Error reading {file}")
        continue

    print(f"Processing {file} ...")

    # Generate density map
    density = create_density_map(mask, PATCH_SIZE)

    # Resize back to image size
    density_upscaled = cv2.resize(density, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply heatmap color
    heatmap = cv2.applyColorMap(density_upscaled, cv2.COLORMAP_JET)

    # Save density heatmap
    out_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
    cv2.imwrite(out_path, heatmap)

    print(f"🔥 Saved density map → {out_path}")

print("\n🎉 Done! Density maps created for all images.")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from PIL import Image
import glob
import os

# ============================================================
# Load masks
# ============================================================
def load_masks(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    masks = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            masks.append(img)
    return masks


# ============================================================
# Create mosaic using USER-GIVEN ROWS & COLS
# Supports incomplete last row
# ============================================================
def create_mosaic(masks, ROWS, COLS):
    h, w = masks[0].shape
    total_needed = ROWS * COLS
    total_available = len(masks)

    # Pad missing tiles
    if total_available < total_needed:
        print(f"Warning: {total_available} tiles found, expected {total_needed}. Padding remaining tiles with blank masks.")
        blank = np.zeros_like(masks[0])
        for _ in range(total_needed - total_available):
            masks.append(blank)

    # Build mosaic
    mosaic = np.zeros((ROWS * h, COLS * w), dtype=np.uint8)
    for idx in range(ROWS * COLS):
        mask = masks[idx]
        r = idx // COLS
        c = idx % COLS
        mosaic[r*h:(r+1)*h, c*w:(c+1)*w] = mask

    return mosaic, masks   # <--- RETURN PADDED MASKS



# ============================================================
# Compute tile density
# ============================================================
def compute_densities(masks):
    return np.array([m.mean() for m in masks])


# ============================================================
# Generate tile coordinates
# ============================================================
def tile_coords(ROWS, COLS):
    coords = []
    for r in range(ROWS):
        for c in range(COLS):
            coords.append([r, c])
    return np.array(coords)


# ============================================================
# Interpolate density into continuous field
# ============================================================
def interpolate_density(coords, values, ROWS, COLS, upscale=20):
    x = coords[:, 1]
    y = coords[:, 0]

    rbf = Rbf(x, y, values, function='gaussian')

    xi = np.linspace(0, COLS - 1, COLS * upscale)
    yi = np.linspace(0, ROWS - 1, ROWS * upscale)
    XI, YI = np.meshgrid(xi, yi)

    ZI = rbf(XI, YI)
    ZI = (ZI - ZI.min()) / (ZI.max() - ZI.min() + 1e-9)
    return ZI


# ============================================================
# Resize density to mosaic resolution
# ============================================================
def resize_to_mosaic(ZI, mosaic):
    H, W = mosaic.shape
    ZI_resized = cv2.resize(ZI, (W, H), interpolation=cv2.INTER_CUBIC)
    return ZI_resized


# ============================================================
# Blend crop mosaic + density
# ============================================================
def blend_maps(crop_mosaic, density, alpha=0.55):
    heatmap = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    crop_rgb = cv2.cvtColor(crop_mosaic, cv2.COLOR_GRAY2BGR)
    final = cv2.addWeighted(heatmap, alpha, crop_rgb, 1-alpha, 0)
    return final


# ============================================================
# Save TIFF
# ============================================================
def save_tif(final_map, path):
    Image.fromarray(final_map).save(path, format="TIFF")
    print(f"Saved TIFF → {path}")


# ============================================================
# MAIN
# ============================================================
CROP_FOLDER = r"C:\Users\hitan\Music\Project\Crop_Mask"
WEED_FOLDER = r"C:\Users\hitan\Music\Project\Weed_Masks"

ROWS = 40     # <── YOU SET ROWS HERE
COLS = 20    # <── YOU SET COLS HERE

OUTPUT = "FIELD_FINAL.tif"

# Load masks
crop_masks = load_masks(CROP_FOLDER)
weed_masks = load_masks(WEED_FOLDER)

# Create mosaic
crop_mosaic, crop_masks = create_mosaic(crop_masks, ROWS, COLS)
_, weed_masks = create_mosaic(weed_masks, ROWS, COLS)


# Density per tile
densities = compute_densities(weed_masks)


# Tile coordinates
coords = tile_coords(ROWS, COLS)


# Interpolation
ZI = interpolate_density(coords, densities, ROWS, COLS)

# Match resolution
ZI_resized = resize_to_mosaic(ZI, crop_mosaic)

# Blend
final_map = blend_maps(crop_mosaic, ZI_resized)

# Save TIFF
save_tif(final_map, OUTPUT)

# Display
plt.figure(figsize=(12,12))
plt.imshow(final_map)
plt.axis('off')
plt.title("Final Field Map — Continuous + Plant-level Detail")
plt.show()

import cv2
import numpy as np
import os

# --------------------------------------------------------
# USER INPUT
# --------------------------------------------------------
density_folder = r"C:\Users\hitan\Music\Project\DensityMaps"   # DEFAULT FOLDER
output_path = r"C:\Users\hitan\Music\Project\FarmHeatMap\FarmHeatmap.png"  # DEFAULT OUTPUT

rows = int(input("Enter number of rows in field: "))
cols = int(input("Enter number of plants per row: "))

# Validate folders
if not os.path.exists(density_folder):
    raise ValueError("❌ Density map folder does not exist!")

# Read all density maps (expects names like 1_image_density.png, 2_image_density.png, etc.)
files = sorted(os.listdir(density_folder))

# Filter valid images only
valid_ext = [".png", ".jpg", ".jpeg"]
files = [f for f in files if any(f.endswith(ext) for ext in valid_ext)]

if len(files) < rows * cols:
    print(f"⚠️ Warning: Expected {rows*cols} files, found only {len(files)}.")

# --------------------------------------------------------
# LOAD IMAGES IN ORDER
# --------------------------------------------------------
images = []
for file in files:
    img = cv2.imread(os.path.join(density_folder, file))
    if img is None:
        continue
    images.append(img)

if len(images) == 0:
    raise ValueError("❌ No valid images found in folder!")

# Get each image size
h, w, c = images[0].shape

# --------------------------------------------------------
# BUILD THE FINAL FARM MAP AS A GRID
# --------------------------------------------------------
farm_map = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

idx = 0
for r in range(rows):
    for c in range(cols):
        if idx < len(images):
            farm_map[r*h:(r+1)*h, c*w:(c+1)*w] = images[idx]
            idx += 1

# --------------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------------
cv2.imwrite(output_path, farm_map)
print(f"\n🌾 Full Farm Weed Density Map saved at:\n{output_path}")

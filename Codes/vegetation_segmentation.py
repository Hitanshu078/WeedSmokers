# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # ---------- Step 1: Read the image ----------
# # ✅ Use a raw string for Windows path
# img = cv2.imread(r'C:\Users\hitan\Music\Project\dataset_split\test\images\1_image.jpg')

# if img is None:
#     raise ValueError("Image not found. Check the path.")

# img = img.astype(np.float32)

# # ---------- Step 2: Split color channels ----------
# B, G, R = cv2.split(img)

# # ---------- Step 3: Compute color indices ----------

# # Excess Green Index (ExG)
# ExG = (2 * G - R - B) / (G + R + B + 1e-6)

# # Color Index of Vegetation Extraction (CIVE)
# CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745

# # Modified Excess Green Index (MExG)
# MExG = 1.262 * G - 0.884 * R - 0.311 * B

# # Excess Green minus Excess Red Index (ExGR)
# ExR = (1.4 * R - G) / (G + R + B + 1e-6)
# ExGR = ExG - ExR


# # ---------- Step 4: Normalize for display ----------
# def normalize(img):
#     img_min, img_max = np.min(img), np.max(img)
#     return ((img - img_min) / (img_max - img_min + 1e-6) * 255).astype(np.uint8)

# ExG_norm = normalize(ExG)
# CIVE_norm = normalize(CIVE)
# MExG_norm = normalize(MExG)
# ExGR_norm = normalize(ExGR)


# # ---------- Step 5: Apply Otsu Thresholding ----------
# _, mask_ExG = cv2.threshold(ExG_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, mask_CIVE = cv2.threshold(CIVE_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, mask_MExG = cv2.threshold(MExG_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, mask_ExGR = cv2.threshold(ExGR_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# # ---------- Step 6: Clean up noise (optional but recommended) ----------
# kernel = np.ones((3, 3), np.uint8)
# mask_MExG_clean = cv2.morphologyEx(mask_MExG, cv2.MORPH_OPEN, kernel)


# # ---------- Step 7: Create overlay to visualize vegetation ----------
# overlay = cv2.bitwise_and(img.astype(np.uint8), img.astype(np.uint8), mask=mask_MExG_clean)


# # ---------- Step 8: Display results ----------
# plt.figure(figsize=(12, 10))

# plt.subplot(2, 3, 1)
# plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(2, 3, 2)
# plt.imshow(MExG_norm, cmap='gray')
# plt.title('MExG Index (Best)')
# plt.axis('off')

# plt.subplot(2, 3, 3)
# plt.imshow(mask_MExG_clean, cmap='gray')
# plt.title('Binary Mask (MExG + Otsu)')
# plt.axis('off')

# plt.subplot(2, 3, 4)
# plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
# plt.title('Vegetation Overlay')
# plt.axis('off')

# plt.subplot(2, 3, 5)
# plt.imshow(ExG_norm, cmap='gray')
# plt.title('ExG Index')
# plt.axis('off')

# plt.subplot(2, 3, 6)
# plt.imshow(ExGR_norm, cmap='gray')
# plt.title('ExGR Index')
# plt.axis('off')

# plt.tight_layout()
# plt.show()


# # ---------- Step 9: Save best mask ----------
# cv2.imwrite(r'C:\Users\hitan\Music\Project\VegetationMask\vegetation_mask.png', mask_MExG_clean)
# print("✅ Saved vegetation mask as 'vegetation_mask.png'")

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# --------------------------------------------------------
#               USER INPUT
# --------------------------------------------------------
input_folder = r"C:\Users\hitan\Music\Project\images"
output_folder = r"C:\Users\hitan\Music\Project\VegetationMask"

# Create parent output folder
os.makedirs(output_folder, exist_ok=True)

# Create subfolders for masks and overlays
mask_folder = os.path.join(output_folder, "Masks")
overlay_folder = os.path.join(output_folder, "Overlays")

os.makedirs(mask_folder, exist_ok=True)
os.makedirs(overlay_folder, exist_ok=True)


# --------------------------------------------------------
# NORMALIZATION FUNCTION
# --------------------------------------------------------
def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    return ((img - img_min) / (img_max - img_min + 1e-6) * 255).astype(np.uint8)


# --------------------------------------------------------
# PROCESS EACH IMAGE
# --------------------------------------------------------
valid_ext = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

for file in os.listdir(input_folder):
    if not any(file.lower().endswith(ext) for ext in valid_ext):
        continue  # Skip non-image files

    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Error reading: {file}")
        continue

    print(f"Processing: {file}")

    img = img.astype(np.float32)

    # --------- Split channels ---------
    B, G, R = cv2.split(img)

    # --------- Vegetation index (MExG) ---------
    MExG = 1.262 * G - 0.884 * R - 0.311 * B
    MExG_norm = normalize(MExG)

    # --------- Otsu thresholding ---------
    _, mask = cv2.threshold(MExG_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --------- Morphological cleaning ---------
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --------- Vegetation overlay ---------
    overlay = cv2.bitwise_and(img.astype(np.uint8), img.astype(np.uint8), mask=mask_clean)

    # --------- Save outputs ---------
    base = os.path.splitext(file)[0]

    mask_path = os.path.join(mask_folder, f"{base}.png")
    overlay_path = os.path.join(overlay_folder, f"{base}.png")

    cv2.imwrite(mask_path, mask_clean)
    cv2.imwrite(overlay_path, overlay)

    print(f"✅ Saved mask → {mask_path}")
    print(f"✅ Saved overlay → {overlay_path}")

print("\n🎉 DONE! All vegetation masks and overlays saved.")

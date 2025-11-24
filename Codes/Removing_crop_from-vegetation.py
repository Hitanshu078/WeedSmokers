import cv2
import os
import numpy as np

# --------------------------------------------------------
# USER INPUT
# --------------------------------------------------------
crop_mask_folder = r"C:\Users\hitan\Music\Project\crop mask"          # U-Net model output
veg_mask_folder = r"C:\Users\hitan\Music\Project\VegetationMask\Masks" # Your MExG masks
output_folder = r"C:\Users\hitan\Music\Project\WeedMasks"

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------------
# PROCESS FILES
# --------------------------------------------------------
for file in os.listdir(veg_mask_folder):
    if not file.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        continue

    veg_path = os.path.join(veg_mask_folder, file)
    crop_path = os.path.join(crop_mask_folder, file)

    # Skip if corresponding crop mask doesn't exist
    if not os.path.exists(crop_path):
        print(f"⚠️ Crop mask missing for {file}, skipping...")
        continue

    # Load the two masks
    veg = cv2.imread(veg_path, cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)

    if veg is None or crop is None:
        print(f"❌ Error reading: {file}")
        continue

    # Ensure same size
    if veg.shape != crop.shape:
        crop = cv2.resize(crop, (veg.shape[1], veg.shape[0]))

    # --------------------------------------------------------
    # COMPUTE WEED MASK: vegetation - crop
    # --------------------------------------------------------
    # veg = 255 where vegetation exists
    # crop = 255 where crop exists

    crop_inv = cv2.bitwise_not(crop)      # invert crop mask
    weed_mask = cv2.bitwise_and(veg, crop_inv)

    # Save output
    out_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
    cv2.imwrite(out_path, weed_mask)

    print(f"🌱 Saved weed mask → {out_path}")

print("\n🎉 Done! Weed masks generated successfully.")

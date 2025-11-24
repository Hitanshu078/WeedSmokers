import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                                     concatenate, BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

import numpy as np
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split

# ============================================================
# 1. U-NET MODEL
# ============================================================

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2))(inputs)
    x = concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net_Crop")
    return model


# ============================================================
# 2. DATA LOADING + PREPROCESSING
# ============================================================

def load_image_paths(image_dir, mask_dir):
    image_paths = []
    mask_paths = []

    image_files = sorted(os.listdir(image_dir))

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)

        base = os.path.splitext(filename)[0]
        mask_filename = base + ".png"

        mask_path = os.path.join(mask_dir, mask_filename)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"WARNING: Mask missing for {filename}")

    return image_paths, mask_paths


def load_and_preprocess_image(img_path, mask_path, img_size=(256, 256)):
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
    num_samples = len(image_paths)

    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images, batch_masks = [], []

            for idx in batch_indices:
                img, mask = load_and_preprocess_image(image_paths[idx], mask_paths[idx], img_size)
                batch_images.append(img)
                batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)


# ============================================================
# 3. CREATE TRAIN / VAL / TEST SPLIT (80/10/10)
# ============================================================

def create_dataset_split(image_paths, mask_paths, output_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    total = len(indices)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def copy_files(idxs, split):
        for i in idxs:
            shutil.copy(image_paths[i], os.path.join(output_dir, split, "images"))
            shutil.copy(mask_paths[i], os.path.join(output_dir, split, "masks"))

    print("Copying Train...")
    copy_files(train_idx, "train")
    print("Copying Validation...")
    copy_files(val_idx, "val")
    print("Copying Test...")
    copy_files(test_idx, "test")

    print("\nDataset Split Completed")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # --- Original dataset (all images together) ---
    IMAGE_DIR = r"C:\Users\hitan\Music\Project\images"
    MASK_DIR = r"C:\Users\hitan\Music\Project\crop mask"

    # --- Output organized dataset ---
    OUTPUT_DATASET = r"C:\Users\hitan\Music\Project\dataset_split"

    print("Loading paths...")
    image_paths, mask_paths = load_image_paths(IMAGE_DIR, MASK_DIR)
    print("Total images:", len(image_paths))

    # --- Create train/val/test split (run once) ---
    create_dataset_split(image_paths, mask_paths, OUTPUT_DATASET)

    # --- Now load from split folders ---
    train_imgs, train_masks = load_image_paths(
        os.path.join(OUTPUT_DATASET, "train", "images"),
        os.path.join(OUTPUT_DATASET, "train", "masks")
    )

    val_imgs, val_masks = load_image_paths(
        os.path.join(OUTPUT_DATASET, "val", "images"),
        os.path.join(OUTPUT_DATASET, "val", "masks")
    )

    test_imgs, test_masks = load_image_paths(
        os.path.join(OUTPUT_DATASET, "test", "images"),
        os.path.join(OUTPUT_DATASET, "test", "masks")
    )

    print(f"Train={len(train_imgs)}  Val={len(val_imgs)}  Test={len(test_imgs)}")

    # ============================================================
    # Create Data Generators
    # ============================================================

    IMG_SIZE = (256, 256)
    BATCH_SIZE = 8

    train_gen = data_generator(train_imgs, train_masks, BATCH_SIZE, img_size=IMG_SIZE)
    val_gen = data_generator(val_imgs, val_masks, BATCH_SIZE, img_size=IMG_SIZE)

    steps = len(train_imgs) // BATCH_SIZE
    val_steps = len(val_imgs) // BATCH_SIZE

    # ============================================================
    # Build + Train Model
    # ============================================================

    model = build_unet(input_shape=(256, 256, 3))
    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[MeanIoU(num_classes=2), "accuracy"]
    )

    model.summary()

    print("Training Started...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=50,
        verbose=1
    )

    model.save("U-Net_Crop_model.h5")
    print("\nTraining Completed! Model saved as U-Net_Crop_model.h5")


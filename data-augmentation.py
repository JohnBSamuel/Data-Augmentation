import os
import cv2
import numpy as np
from imgaug import augmenters as iaa
from tqdm import tqdm
import shutil

# Paths
image_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images"
label_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels"
output_image_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images_aug"
output_label_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels_aug"

# Create output directories if they don't exist
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Define augmentations using imgaug
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),                       # horizontal flip
    iaa.Affine(rotate=(-15, 15)),         # random rotation
    iaa.GaussianBlur(sigma=(0, 1.0)),     # blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # noise
    iaa.Multiply((0.8, 1.2)),             # brightness
    iaa.LinearContrast((0.75, 1.5)),      # contrast
    iaa.Cutout(nb_iterations=2, size=0.2, squared=False)  # Cutout
])

# Loop through image files
for filename in tqdm(os.listdir(image_dir)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.rsplit('.', 1)[0] + '.txt')

    # Skip if label file does not exist
    if not os.path.exists(label_path):
        print(f"[!] Skipping {filename} (label not found)")
        continue

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Skipping {filename} (failed to load image)")
        continue

    # Convert BGR to RGB for augmentation (optional but consistent with most DL pipelines)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform augmentation
    for i in range(2):  # Number of augmented versions per image
        augmented_image = augmenters(image=image)
        aug_filename = filename.rsplit('.', 1)[0] + f"_aug{i}.jpg"
        aug_image_path = os.path.join(output_image_dir, aug_filename)
        aug_label_path = os.path.join(output_label_dir, aug_filename.rsplit('.', 1)[0] + '.txt')

        # Save image (convert back to BGR for OpenCV write)
        cv2.imwrite(aug_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

        # Copy label file as-is (assuming bounding boxes are YOLO format and unaffected)
        shutil.copy(label_path, aug_label_path)

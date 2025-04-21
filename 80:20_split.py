import os
import shutil
import random
from tqdm import tqdm

# Set paths
image_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images_aug2"
label_dir = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels_aug2"

output_image_train = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images_split/train"
output_image_val = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images_split/val"
output_label_train = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels_split/train"
output_label_val = "/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels_split/val"

# Create output directories
for path in [output_image_train, output_image_val, output_label_train, output_label_val]:
    os.makedirs(path, exist_ok=True)

# List and shuffle image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# Split ratio
split_ratio = 0.8
split_idx = int(len(image_files) * split_ratio)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Function to copy files
def copy_data(files, split_type):
    for file in tqdm(files, desc=f"Copying {split_type} files"):
        name_wo_ext = os.path.splitext(file)[0]
        image_src = os.path.join(image_dir, file)
        label_src = os.path.join(label_dir, f"{name_wo_ext}.txt")

        image_dst = os.path.join(output_image_train if split_type == "train" else output_image_val, file)
        label_dst = os.path.join(output_label_train if split_type == "train" else output_label_val, f"{name_wo_ext}.txt")

        if os.path.exists(label_src):
            shutil.copy(image_src, image_dst)
            shutil.copy(label_src, label_dst)
        else:
            print(f"⚠️ Label missing for image {file}, skipping.")

# Copy files
copy_data(train_files, "train")
copy_data(val_files, "val")

print("✅ Dataset split complete.")

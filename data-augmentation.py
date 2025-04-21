import os
import cv2
import numpy as np
from tqdm import tqdm
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Input paths
image_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images'
label_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels'

# Output paths
aug_img_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/augmented_images'
aug_label_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/augmented_labels'
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Define augmentation pipeline
augmenter = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flip
    iaa.Flipud(0.3),  # vertical flip
    iaa.Multiply((0.8, 1.2)),  # brightness
    iaa.Affine(rotate=(-20, 20)),  # rotation
    iaa.GaussianBlur(sigma=(0, 1.5)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.CropToFixedSize(width=256, height=256),
    iaa.CoarseDropout(0.05, size_percent=0.2),  # Cutout-like
])

# Process each image
image_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(image_list):
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')

    # Read image
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    # Read YOLO bboxes
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                # Convert YOLO -> pixel format
                x1 = int((x - w/2) * width)
                y1 = int((y - h/2) * height)
                x2 = int((x + w/2) * width)
                y2 = int((y + h/2) * height)
                bboxes.append((cls, x1, y1, x2, y2))

    if not bboxes:
        continue

    for i in range(20):  # 20 augmentations per image
        try:
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4], label=int(box[0]))
                for box in bboxes
            ], shape=image.shape)

            aug_img, aug_bbs = augmenter(image=image, bounding_boxes=bbs)
            aug_bbs = aug_bbs.remove_out_of_image().clip_out_of_image()

            # Save image
            new_img_name = f"{os.path.splitext(img_name)[0]}_aug{i+1}.jpg"
            cv2.imwrite(os.path.join(aug_img_dir, new_img_name), aug_img)

            # Save labels in YOLO format
            new_label_path = os.path.join(aug_label_dir, new_img_name.replace('.jpg', '.txt'))
            with open(new_label_path, 'w') as f:
                for bbox in aug_bbs:
                    cls = bbox.label
                    x_center = ((bbox.x1 + bbox.x2) / 2) / aug_img.shape[1]
                    y_center = ((bbox.y1 + bbox.y2) / 2) / aug_img.shape[0]
                    w = (bbox.x2 - bbox.x1) / aug_img.shape[1]
                    h = (bbox.y2 - bbox.y1) / aug_img.shape[0]
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        except Exception as e:
            print(f"Skipping {img_name} aug {i+1} due to error: {e}")

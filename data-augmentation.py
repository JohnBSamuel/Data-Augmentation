import os
import cv2
import albumentations as A
from tqdm import tqdm

# Input paths
image_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/images'
label_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/labels'

# Output paths
aug_img_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/augmented_images'
aug_label_dir = '/Users/johnsamuel/Desktop/CiviLens/CiviLens-backend/dataset/augmented_labels'
os.makedirs(aug_img_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# Define augmentations
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    A.RandomRotate90(p=0.2),
    A.VerticalFlip(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.RandomFog(p=0.2),  # Fixed: Removed invalid arguments
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.2),  # Fixed: Removed invalid argument
    A.cutout(num_holes=5, max_h_size=30, max_w_size=30, fill_value=0, p=0.4),  # Fixed: lowercase cutout
])


# Process each image
image_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for img_name in tqdm(image_list):
    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, img_name.rsplit('.', 1)[0] + '.txt')

    # Read image
    image = cv2.imread(img_path)
    height, width, _ = image.shape

    # Read YOLO bboxes
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, x, y, w, h = parts
                bboxes.append([float(x), float(y), float(w), float(h)])
                class_labels.append(int(class_id))

    # Skip if no bboxes
    if not bboxes:
        continue

    for i in range(20):  # 20 augmentations per image
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            # Save new image
            new_img_name = f"{os.path.splitext(img_name)[0]}_aug{i+1}.jpg"
            cv2.imwrite(os.path.join(aug_img_dir, new_img_name), aug_img)

            # Save new label
            new_label_path = os.path.join(aug_label_dir, new_img_name.replace('.jpg', '.txt'))
            with open(new_label_path, 'w') as f:
                for bbox, cls in zip(aug_bboxes, aug_labels):
                    x, y, w, h = bbox
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            print(f"Skipping {img_name} aug {i+1} due to error: {e}")

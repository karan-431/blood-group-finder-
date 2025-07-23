import os
import shutil
from sklearn.model_selection import train_test_split

source_folder = r"C:\Users\karan\OneDrive\Desktop\bdtask\dataset"
output_folder = r"C:\Users\karan\OneDrive\Desktop\bdtask\spitted"
splits = ['train', 'val', 'test']
split_ratios = [0.7, 0.2, 0.1]

for blood_group in os.listdir(source_folder):
    group_path = os.path.join(source_folder, blood_group)
    all_images = [img for img in os.listdir(group_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

    train_val, test = train_test_split(all_images, test_size=split_ratios[2], random_state=42)
    train, val = train_test_split(train_val, test_size=split_ratios[1]/(1 - split_ratios[2]), random_state=42)

    for split_name, split_images in zip(splits, [train, val, test]):
        split_dir = os.path.join(output_folder, split_name, blood_group)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_images:
            shutil.copy2(os.path.join(group_path, img), os.path.join(split_dir, img))

print("✅ Dataset split into train/val/test.")

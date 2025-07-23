import os
import shutil

# Root of dataset with subfolders: train/, test/, valid/
source_root = r"C:\Users\karan\Downloads\Blood Group Detection.v4i.yolov7pytorch"  # <- update this
new_dataset_root = r"C:\Users\karan\OneDrive\Desktop\bdtask\dataset"  # the final grouped folder

# We'll process all 3 splits
splits = ['train', 'test', 'valid']

# Valid blood group types (excluding Rh)
blood_groups = ['A', 'B', 'AB', 'O']

for split in splits:
    split_path = os.path.join(source_root, split)

    for group_folder in os.listdir(split_path):  # A, B, AB, O
        group_path = os.path.join(split_path, group_folder)
        if not os.path.isdir(group_path):
            continue

        for filename in os.listdir(group_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Extract main blood group from filename
            name_part = filename.split('_')[0].upper()
            main_group = None

            # Match A, B, AB, O (AB must come first to avoid misclassification as A)
            if name_part.startswith('AB'):
                main_group = 'AB'
            elif name_part.startswith('A'):
                main_group = 'A'
            elif name_part.startswith('B'):
                main_group = 'B'
            elif name_part.startswith('O'):
                main_group = 'O'

            if main_group and main_group in blood_groups:
                dest_folder = os.path.join(new_dataset_root, main_group)
                os.makedirs(dest_folder, exist_ok=True)

                src_path = os.path.join(group_path, filename)
                dest_path = os.path.join(dest_folder, filename)
                shutil.copy2(src_path, dest_path)

print("✅ Images grouped into 4 folders: A, B, AB, O inside 'grouped_dataset_4classes'")

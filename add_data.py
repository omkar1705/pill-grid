import os
import shutil
import random
from pathlib import Path

def move_data(src_img_dir, src_label_dir, dest_root, val_split=0.2):
    """
    Moves matched image/label pairs from source to destination with a train/val split.
    """
    # Define destination paths
    train_img_dir = os.path.join(dest_root, 'images', 'train')
    val_img_dir = os.path.join(dest_root, 'images', 'val')
    train_label_dir = os.path.join(dest_root, 'labels', 'train')
    val_label_dir = os.path.join(dest_root, 'labels', 'val')

    # Get list of all label files
    label_files = [f for f in os.listdir(src_label_dir) if f.endswith('.txt') and f != 'classes.txt']
    
    if not label_files:
        print("No label files found in", src_label_dir)
        return

    print(f"Found {len(label_files)} annotated images.")
    
    moved_count = 0
    for label_file in label_files:
        # Determine base name and corresponding image
        base_name = os.path.splitext(label_file)[0]
        
        # Look for image with common extensions
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if os.path.exists(os.path.join(src_img_dir, base_name + ext)):
                image_file = base_name + ext
                break
        
        if image_file:
            # Decide split
            is_val = random.random() < val_split
            
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_label_dir = val_label_dir if is_val else train_label_dir
            
            # Move files
            shutil.move(os.path.join(src_img_dir, image_file), os.path.join(target_img_dir, image_file))
            shutil.move(os.path.join(src_label_dir, label_file), os.path.join(target_label_dir, label_file))
            
            print(f"Moved {base_name} to {'val' if is_val else 'train'}")
            moved_count += 1
        else:
            print(f"Warning: Image for label {label_file} not found in {src_img_dir}")

    print(f"Successfully moved {moved_count} pairs.")

if __name__ == "__main__":
    BASE_DIR = Path("datasets/medical-pills")
    RAW_IMAGES = BASE_DIR / "raw_images"
    RAW_LABELS = BASE_DIR / "raw_labels"
    
    if not RAW_IMAGES.exists() or not RAW_LABELS.exists():
        print("Error: Raw directories not found.")
    else:
        move_data(RAW_IMAGES, RAW_LABELS, BASE_DIR)


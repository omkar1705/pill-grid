"""
make_synthetic_dataset_augmented.py

Creates a synthetic augmented dataset from input blister images.

Outputs:
- augmented images in output_dataset/images/

Requirements:
- Python 3.8+
- numpy
- opencv-python

Usage:
    python make_synthetic_dataset_augmented.py
"""

import os
import cv2
import numpy as np
import random

# ------------------------------
# CONFIG
# ------------------------------
INPUT_DIR = "input_images"   # folder with source blister images
OUTPUT_DIR = "output_dataset"
IMAGES_OUT = os.path.join(OUTPUT_DIR, "images")

AUG_PER_IMAGE = 10   # how many augmented variants per input image
IMAGE_SIZE = (640, 480)  # resized output canvas (w,h)

# Random augmentation ranges
ROTATE_RANGE = (-20, 20)       # degrees
SCALE_RANGE = (0.9, 1.1)
BRIGHTNESS_RANGE = (0.25, 1.3) # broadened to include low-light (dark) images
CONTRAST_RANGE = (0.6, 1.4)
GAUSSIAN_BLUR_RANGE = (0, 2)   # kernel sizes (0 to disable)
NOISE_STD = 8                  # gaussian noise stdev
PERSPECTIVE_PROB = 0.3
PERSPECTIVE_MAX_RATIO = 0.05   # how large perspective displacement may be (fraction of dims)

# Ensure output directories
os.makedirs(IMAGES_OUT, exist_ok=True)

# ------------------------------
# Geometric transforms helpers
# ------------------------------
def random_affine_transform(img):
    """Return transformed image."""
    h, w = img.shape[:2]

    # random rotate & scale & translate
    angle = random.uniform(*ROTATE_RANGE)
    scale = random.uniform(*SCALE_RANGE)
    tx = random.uniform(-0.05*w, 0.05*w)
    ty = random.uniform(-0.05*h, 0.05*h)

    # construct affine matrix
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0,2] += tx
    M[1,2] += ty

    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # maybe perspective
    if random.random() < PERSPECTIVE_PROB:
        max_dx = PERSPECTIVE_MAX_RATIO * w
        max_dy = PERSPECTIVE_MAX_RATIO * h
        pts1 = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        pts2 = np.float32([
            [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [w-1+random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)],
            [w-1+random.uniform(-max_dx, max_dx), h-1+random.uniform(-max_dy, max_dy)],
            [random.uniform(-max_dx, max_dx), h-1+random.uniform(-max_dy, max_dy)]
        ])
        P = cv2.getPerspectiveTransform(pts1, pts2)
        out = cv2.warpPerspective(out, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return out

def apply_color_jitter(img):
    out = img.astype(np.float32)
    alpha = random.uniform(*CONTRAST_RANGE)
    beta = random.uniform(*BRIGHTNESS_RANGE)
    out = out * alpha
    out = out * beta
    out = np.clip(out, 0, 255).astype(np.uint8)
    # blur
    k = int(random.uniform(*GAUSSIAN_BLUR_RANGE))
    if k % 2 == 0:
        k += 1
    if k > 1:
        out = cv2.GaussianBlur(out, (k,k), 0)
    # noise
    noise = np.random.normal(0, NOISE_STD * random.random(), out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

# ------------------------------
# Main augmentation loop
# ------------------------------
def generate_dataset():
    # find candidate images in INPUT_DIR
    candidates = []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory '{INPUT_DIR}' not found.")
        return

    for f in os.listdir(INPUT_DIR):
        if f.lower().endswith(exts):
            candidates.append(os.path.join(INPUT_DIR, f))
    if not candidates:
        print("No images found in", INPUT_DIR)
        return

    out_count = 0

    for src_path in candidates:
        img_orig = cv2.imread(src_path)
        if img_orig is None:
            continue

        # resize preserving aspect to fit IMAGE_SIZE
        h0, w0 = img_orig.shape[:2]
        target_w, target_h = IMAGE_SIZE
        scale = min(target_w / w0, target_h / h0)
        new_w, new_h = int(w0*scale), int(h0*scale)
        img = cv2.resize(img_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # pad to target size centered
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8)*255
        ox = (target_w - new_w)//2
        oy = (target_h - new_h)//2
        canvas[oy:oy+new_h, ox:ox+new_w] = img

        base_name = os.path.splitext(os.path.basename(src_path))[0]

        # create AUG_PER_IMAGE variants
        for ai in range(AUG_PER_IMAGE):
            aug_img = random_affine_transform(canvas)
            aug_img = apply_color_jitter(aug_img)

            out_name = f"{base_name}_aug_{out_count:05d}.jpg"
            out_img_path = os.path.join(IMAGES_OUT, out_name)
            
            # save image
            cv2.imwrite(out_img_path, aug_img)
            out_count += 1

        print(f"[INFO] processed {os.path.basename(src_path)} -> generated {AUG_PER_IMAGE} variants")

    print(f"[DONE] Generated ~{out_count} images in {IMAGES_OUT}")

if __name__ == "__main__":
    generate_dataset()

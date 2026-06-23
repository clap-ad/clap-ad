import os
import cv2
import glob
from tqdm import tqdm

# =========================
# Configs
# =========================
DIR_ROOT = "/path/to/evaluation/results"
DIR_SUB = "rgb_front"

FPS = 10

# ⭐ Output resolutions（480p，16:9）
TARGET_HEIGHT = 480
TARGET_WIDTH = 854

# =========================
# Tools function
# =========================

def collect_dir_names(root_dir, sub_dir):
    """
    Scan all subdirectories including sub_dir under root_dir
    """
    dir_names = []
    for d in os.listdir(root_dir):
        full_dir = os.path.join(root_dir, d)
        if not os.path.isdir(full_dir):
            continue
        if os.path.isdir(os.path.join(full_dir, sub_dir)):
            dir_names.append(d)
    return sorted(dir_names)


# =========================
# Min logic
# =========================

def make_video(dir_name, fps=10):
    image_dir = os.path.join(DIR_ROOT, dir_name, DIR_SUB)
    # ⭐ Videos are saved under DIR_ROOT
    output_path = os.path.join(DIR_ROOT, f"vid_{dir_name}_480p.mp4")

    images = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )

    if not images:
        print(f"[skipped] no image: {image_dir}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (TARGET_WIDTH, TARGET_HEIGHT)
    )

    if not writer.isOpened():
        print(f"[error] VideoWriter failed at opening: {output_path}")
        return

    for img_path in tqdm(images, desc=dir_name, unit="frame", leave=False):
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.resize(
            frame,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        writer.write(frame)

    writer.release()
    print(f"[finished] {output_path}")


if __name__ == "__main__":
    DIR_NAME_LIST = collect_dir_names(DIR_ROOT, DIR_SUB)

    print(f"[info] Found {len(DIR_NAME_LIST)} directories to process")

    for dir_name in tqdm(DIR_NAME_LIST, desc="Processing videos", unit="video"):
        make_video(dir_name, FPS)

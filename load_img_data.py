import os
import cv2
import numpy as np



#  Parse Age from IMDB Filename
def parse_age_from_filename(filename):
    """
    IMDB filename format example:
        nm0000104_rm1026201600_1960-8-10_2003.jpg

    Format pattern:
        [personID]_[random]_[birthYear-month-day]_[photoYear].jpg

    Age = photoYear − birthYear
    """
    try:
        parts = filename.split("_")
        birth = parts[-2]  # '1960-8-10'
        photo_year_str = parts[-1].split(".")[0]

        birth_year = int(birth.split("-")[0])     # 1960
        photo_year = int(photo_year_str)          # 2003

        return photo_year - birth_year
    except:
        return None


#  Load IMDB Folders
def _load_from_folders(base_dir):
    """
    Loads grayscale images from folders IMDB/00, IMDB/01, ..., IMDB/10.
    Each folder = user identity.
    """
    images = []
    labels = []
    ages = []

    subfolders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

    for sf in subfolders:
        folder_path = os.path.join(base_dir, sf)

        print(f"Loading images in {sf}")

        files = [x for x in os.listdir(folder_path)
                 if x.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Safety: limit to 40 images per identity (matches project)
        if len(files) > 40:
            print(f"  → Loading 40 images from {sf}")
            files = files[:40]
        else:
            print(f"  → Loading {len(files)} images from {sf}")

        for filename in files:
            full_path = os.path.join(folder_path, filename)

            # Load grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                print(f"    Very small or empty image: {filename}")
                continue

            # Store image + label + age
            images.append(img)
            labels.append(sf)                  # User ID = folder name
            ages.append(parse_age_from_filename(filename))

    print("All images are loaded (folder mode).")
    return images, labels, ages


#  Public API
def get_images(base_dir):
    """
    Returns:
        images: list of numpy arrays
        labels: list of user IDs (string folder names)
        ages:   list of ages (or None)
    """
    return _load_from_folders(base_dir)

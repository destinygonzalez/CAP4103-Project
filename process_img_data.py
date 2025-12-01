import mediapipe as mp
import numpy as np
import cv2
import os

mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh


def save_face_crops(images, labels, save_directory, margin=0.05):
    """
    Detects faces and saves cropped face images using Mediapipe.
    """

    os.makedirs(save_directory, exist_ok=True)
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    count = 0

    for img, label in zip(images, labels):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        results = detector.process(img_rgb)

        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box

                x = int(box.xmin * w)
                y = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)

                mx = int(bw * margin)
                my = int(bh * margin)

                x0 = max(0, x - mx)
                y0 = max(0, y - my)
                x1 = min(w, x + bw + mx)
                y1 = min(h, y + bh + my)

                crop = img[y0:y1, x0:x1]

                filename = f"{label}_{count}.png"
                cv2.imwrite(os.path.join(save_directory, filename), crop)
                count += 1


def get_landmark_points(images, labels, num_coords=5):
    """
    Returns face landmarks using Mediapipe Face Mesh.
    num_coords ignored — returns all 468 points.
    """

    print("Extracting Mediapipe face mesh landmarks...")

    mesh = mp_mesh.FaceMesh(static_image_mode=True)

    landmarks = []
    new_labels = []

    for img, label in zip(images, labels):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        results = mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                pts = [(p.x, p.y) for p in lm.landmark]
                landmarks.append(pts)
                new_labels.append(label)

    return np.array(landmarks), np.array(new_labels)

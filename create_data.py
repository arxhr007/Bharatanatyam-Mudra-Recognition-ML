import os
import csv
import math
import cv2
import mediapipe as mp

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = r"Bharatanatyam-Mudra-Dataset"
# Extract up to two hands per image so mixed one-/two-hand mudras are supported
MAX_NUM_HANDS = 2

# Always write the 2-hand CSV (single-hand samples will have zeros for the missing hand)
OUTPUT_CSV = r"mudra_keypoints_2hands.csv"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.5
)

# -----------------------------
# FUNCTION: Extract Keypoints
# -----------------------------
FEATURES_PER_HAND = 21 * 3


def _hand_features_from_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]

    scale = math.sqrt(
        (middle_mcp.x - wrist.x) ** 2 +
        (middle_mcp.y - wrist.y) ** 2 +
        (middle_mcp.z - wrist.z) ** 2
    )

    if scale == 0:
        return None

    feats = []
    for lm in hand_landmarks.landmark:
        x = (lm.x - wrist.x) / scale
        y = (lm.y - wrist.y) / scale
        z = (lm.z - wrist.z) / scale
        feats.extend([x, y, z])

    return feats


def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        # no hands detected -> return zero vector and count 0
        return [0.0] * (FEATURES_PER_HAND * MAX_NUM_HANDS), 0

    # Build mapping by handedness when available so order is stable (Left then Right)
    hands_feats = {"Left": None, "Right": None}

    # `multi_handedness` aligns with `multi_hand_landmarks` by index in MediaPipe
    try:
        handedness_list = [h.classification[0].label for h in results.multi_handedness]
    except Exception:
        handedness_list = [None] * len(results.multi_hand_landmarks)

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # try to get label, fallback to position-based ordering
        label = None
        if idx < len(handedness_list) and handedness_list[idx] is not None:
            label = handedness_list[idx]

        feats = _hand_features_from_landmarks(hand_landmarks)
        if feats is None:
            continue

        if label in ("Left", "Right"):
            hands_feats[label] = feats
        else:
            # fallback: decide by wrist x (smaller x is left on the image)
            wrist_x = hand_landmarks.landmark[0].x
            if "_temp" not in hands_feats:
                hands_feats["_temp"] = []
            hands_feats["_temp"].append((wrist_x, feats))

    # If fallback entries exist, sort them by x and assign Left then Right
    if "_temp" in hands_feats:
        sorted_temp = sorted(hands_feats["_temp"], key=lambda t: t[0])
        if len(sorted_temp) >= 1 and hands_feats["Left"] is None:
            hands_feats["Left"] = sorted_temp[0][1]
        if len(sorted_temp) >= 2 and hands_feats["Right"] is None:
            hands_feats["Right"] = sorted_temp[1][1]

    # Build final feature vector in a consistent order: Left then Right
    final_feats = []
    detected = 0
    for side in ("Left", "Right")[:MAX_NUM_HANDS]:
        if hands_feats.get(side) is not None:
            final_feats.extend(hands_feats[side])
            detected += 1
        else:
            final_feats.extend([0.0] * FEATURES_PER_HAND)

    return final_feats, detected

# -----------------------------
# BUILD DATASET
# -----------------------------
rows = []
processed = 0
skipped = 0
class_names = set()

for folder_name in sorted(os.listdir(DATASET_DIR)):
    folder_path = os.path.join(DATASET_DIR, folder_name)

    # skip non-folders (like .py, .md, venv)
    if not os.path.isdir(folder_path):
        continue

    # skip hidden/system folders
    if folder_name.startswith(".") or folder_name in ("mp-env", "venv", ".venv"):
        continue

    label = folder_name
    class_names.add(label)

    print(f"\nProcessing class: {label}")

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(IMAGE_EXTS):
            continue

        img_path = os.path.join(folder_path, filename)

        res = extract_keypoints(img_path)

        if res is None:
            skipped += 1
            continue

        features, num_hands = res
        # skip images with no hands detected
        if num_hands == 0:
            skipped += 1
            continue

        rows.append(features + [num_hands] + [label])
        processed += 1

# -----------------------------
# SAVE CSV
# -----------------------------
if len(rows) > 0:
    num_features = len(rows[0]) - 2
    headers = [f"f{i}" for i in range(num_features)] + ["num_hands", "label"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print("\n==============================")
    print("✅ DATASET CREATED SUCCESSFULLY")
    print("==============================")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Total samples: {processed}")
    print(f"Skipped images: {skipped}")
    print(f"Total classes: {len(class_names)}")
    print(f"Classes: {sorted(class_names)}")
else:
    print("❌ No data extracted. Check images or paths.")

hands.close()
import check_python
import joblib
import cv2
import mediapipe as mp
import math

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("mudra_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# CONFIG / MEDIAPIPE SETUP
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Set to 1 or 2 depending on the model / CSV used for training
# Default to 2 to support mixed one-/two-hand mudras
MAX_NUM_HANDS = 2

hands = mp_hands.Hands(
    static_image_mode=False,   # IMPORTANT for video
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# KEYPOINT EXTRACTION
# -----------------------------
FEATURES_PER_HAND = 21 * 3


def _hand_features_from_landmarks(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle = hand_landmarks.landmark[9]

    scale = math.sqrt(
        (middle.x - wrist.x) ** 2 +
        (middle.y - wrist.y) ** 2 +
        (middle.z - wrist.z) ** 2
    )

    if scale == 0:
        return None

    features = []
    for lm in hand_landmarks.landmark:
        x = (lm.x - wrist.x) / scale
        y = (lm.y - wrist.y) / scale
        z = (lm.z - wrist.z) / scale
        features.extend([x, y, z])

    return features

# -----------------------------
# START CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand(s)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        # Build mapping by handedness when available
        hands_feats = {"Left": None, "Right": None}
        try:
            handedness_list = [h.classification[0].label for h in results.multi_handedness]
        except Exception:
            handedness_list = [None] * len(results.multi_hand_landmarks)

        # Draw and compute per-hand features
        temp_unspecified = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = None
            if idx < len(handedness_list):
                label = handedness_list[idx]

            feats = _hand_features_from_landmarks(hand_landmarks)
            if feats is None:
                continue

            if label in ("Left", "Right"):
                hands_feats[label] = feats
            else:
                wrist_x = hand_landmarks.landmark[0].x
                temp_unspecified.append((wrist_x, feats))

        # fallback: sort unspecified by x (left first)
        temp_unspecified.sort(key=lambda t: t[0])
        assign_order = [f[1] for f in temp_unspecified]
        if hands_feats["Left"] is None and len(assign_order) >= 1:
            hands_feats["Left"] = assign_order.pop(0)
        if hands_feats["Right"] is None and len(assign_order) >= 1:
            hands_feats["Right"] = assign_order.pop(0)

        # Build final vector in consistent order: Left then Right
        final_feats = []
        for side in ("Left", "Right")[:MAX_NUM_HANDS]:
            if hands_feats.get(side) is not None:
                final_feats.extend(hands_feats[side])
            else:
                final_feats.extend([0.0] * FEATURES_PER_HAND)

        # Validate feature length against model if possible
        if hasattr(model, "n_features_in_") and len(final_feats) != model.n_features_in_:
            cv2.putText(frame, f"Feature mismatch: expected {model.n_features_in_}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Skip prediction if all zeros
            if any(abs(x) > 1e-6 for x in final_feats):
                pred = model.predict([final_feats])
                mudra = le.inverse_transform(pred)[0]
                cv2.putText(frame, f"Predicted: {mudra}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Mudra Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

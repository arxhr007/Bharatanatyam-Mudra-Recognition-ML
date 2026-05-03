import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# CONFIG
# -----------------------------
# Set to 1 or 2 depending on whether you generated single- or two-hand CSV
# Default to 2 to support mixed one-/two-hand mudras (zeros used for missing hands)
MAX_NUM_HANDS = 2


# -----------------------------
# LOAD DATA
# -----------------------------
# Read CSV from repository root (no external path required)
csv_name = "mudra_keypoints.csv" if MAX_NUM_HANDS == 1 else "mudra_keypoints_2hands.csv"
df = pd.read_csv(csv_name)

# If the CSV includes a `num_hands` column, drop it from features (it's not a label)
if "num_hands" in df.columns:
    X = df.drop(["label", "num_hands"], axis=1)
else:
    X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# ENCODE LABELS
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# -----------------------------
# EVALUATE
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "mudra_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("\nModel saved!")
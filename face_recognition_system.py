import cv2
import os
import json
import numpy as np
import tensorflow as tf

from face_detection import FaceDetector, preprocess_face


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "shared_cnn.h5"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "threshold.json"
)

REGISTERED_USERS_PATH = os.path.join(
    BASE_DIR,
    "registered_users"
)


# ============================================================
# Rebuild exact CNN and load trained weights
# ============================================================

shared_cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(112, 92, 1)),
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        128,
        activation="sigmoid"
    )
])

shared_cnn.load_weights(MODEL_PATH)


# ============================================================
# Load recognition threshold
# ============================================================

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

THRESHOLD = float(threshold_data["threshold"])

print("Recognition threshold:", THRESHOLD)


# ============================================================
# Load registered users
# ============================================================

def load_registered_users():

    users = []

    if not os.path.exists(REGISTERED_USERS_PATH):
        return users

    for filename in os.listdir(REGISTERED_USERS_PATH):

        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(
            REGISTERED_USERS_PATH,
            filename
        )

        try:

            with open(filepath, "r") as file:
                data = json.load(file)

            embedding = np.array(
                data["embedding"],
                dtype=np.float32
            )

            users.append({
                "user_id": data["user_id"],
                "name": data["name"],
                "embedding": embedding
            })

        except Exception as error:

            print(
                f"Could not load {filename}: {error}"
            )

    return users


# ============================================================
# Generate face embedding
# ============================================================

def generate_embedding(face):

    processed_face = preprocess_face(face)

    embedding = shared_cnn.predict(
        processed_face,
        verbose=0
    )

    return embedding[0]


# ============================================================
# Euclidean distance
# ============================================================

def euclidean_distance(embedding1, embedding2):

    return float(
        np.linalg.norm(
            embedding1 - embedding2
        )
    )


# ============================================================
# Recognize one face
# ============================================================

def recognize_face(face, users):

    if len(users) == 0:

        return "Unknown", None, None


    query_embedding = generate_embedding(face)

    best_user = None
    best_distance = float("inf")


    for user in users:

        distance = euclidean_distance(
            query_embedding,
            user["embedding"]
        )

        if distance < best_distance:

            best_distance = distance
            best_user = user


    if best_distance <= THRESHOLD:

        return (
            best_user["name"],
            best_user["user_id"],
            best_distance
        )

    return (
        "Unknown",
        None,
        best_distance
    )


# ============================================================
# Start real-time recognition
# ============================================================

def start_recognition():

    detector = FaceDetector()

    users = load_registered_users()


    print("\n========================================")
    print("      FACE RECOGNITION SYSTEM")
    print("========================================")

    print(
        "Registered users:",
        len(users)
    )

    for user in users:

        print(
            f" - {user['user_id']}: {user['name']}"
        )

    print("----------------------------------------")
    print("Press Q to exit.")
    print("========================================\n")


    if len(users) == 0:

        print(
            "No registered users found."
        )

        return


    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )


    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        return


    while True:

        ret, frame = camera.read()


        if not ret:

            print(
                "ERROR: Could not read webcam frame."
            )

            break


        faces = detector.detect_faces(frame)


        # ----------------------------------------------------
        # No face
        # ----------------------------------------------------

        if len(faces) == 0:

            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # ----------------------------------------------------
        # Detect and recognize faces
        # ----------------------------------------------------

        for (x, y, w, h) in faces:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            face = gray[
                y:y + h,
                x:x + w
            ]


            name, user_id, distance = recognize_face(
                face,
                users
            )


            if name == "Unknown":

                label = "Unknown"

                box_color = (0, 0, 255)

            else:

                label = f"{name} ({user_id})"

                box_color = (0, 255, 0)


            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )


            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )


            if distance is not None:

                cv2.putText(
                    frame,
                    f"Distance: {distance:.4f}",
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2
                )


        cv2.putText(
            frame,
            "Press Q to exit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "Real-Time Face Recognition",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    start_recognition()

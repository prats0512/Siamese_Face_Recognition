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

REGISTERED_USERS_PATH = os.path.join(
    BASE_DIR,
    "registered_users"
)

os.makedirs(REGISTERED_USERS_PATH, exist_ok=True)


# ============================================================
# Rebuild exact CNN architecture and load trained weights
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

print("Trained shared CNN loaded successfully.")
print("Embedding dimension:", shared_cnn.output_shape[-1])


# ============================================================
# Generate 128-D face embedding
# ============================================================

def generate_embedding(face):

    processed_face = preprocess_face(face)

    embedding = shared_cnn.predict(
        processed_face,
        verbose=0
    )

    return embedding[0]


# ============================================================
# Save registered user
# ============================================================

def save_user(user_id, name, embeddings):

    final_embedding = np.mean(
        np.array(embeddings),
        axis=0
    )

    user_data = {
        "user_id": str(user_id),
        "name": str(name),
        "embedding": final_embedding.tolist()
    }

    user_file = os.path.join(
        REGISTERED_USERS_PATH,
        f"{user_id}.json"
    )

    with open(user_file, "w") as file:

        json.dump(
            user_data,
            file,
            indent=4
        )

    return user_file


# ============================================================
# User enrollment
# ============================================================

def enroll_user(
    user_id,
    name,
    required_samples=5
):

    detector = FaceDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")

        return False


    embeddings = []


    print("\n========================================")
    print("        FACE ENROLLMENT")
    print("========================================")
    print(f"User ID : {user_id}")
    print(f"Name    : {name}")
    print(f"Samples : {required_samples}")
    print("----------------------------------------")
    print("Look at the camera.")
    print("Press SPACE to capture a sample.")
    print("Press Q to cancel.")
    print("========================================\n")


    while len(embeddings) < required_samples:

        ret, frame = camera.read()

        if not ret:

            print(
                "ERROR: Could not read webcam frame."
            )

            break


        faces = detector.detect_faces(frame)


        # Draw detected faces
        for (x, y, w, h) in faces:

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


        cv2.putText(
            frame,
            f"Samples: {len(embeddings)}/{required_samples}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "SPACE = Capture | Q = Quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "Face Enrollment",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # Capture sample
        # ----------------------------------------------------

        if key == ord(" "):

            if len(faces) == 1:

                x, y, w, h = faces[0]

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                face = gray[
                    y:y + h,
                    x:x + w
                ]


                embedding = generate_embedding(face)

                embeddings.append(embedding)


                print(
                    f"Captured sample "
                    f"{len(embeddings)}/{required_samples}"
                )


            elif len(faces) == 0:

                print(
                    "No face detected. "
                    "Please position your face."
                )


            else:

                print(
                    "Multiple faces detected. "
                    "Only one person should be visible."
                )


        # ----------------------------------------------------
        # Cancel enrollment
        # ----------------------------------------------------

        elif key == ord("q"):

            print("Enrollment cancelled.")

            camera.release()

            cv2.destroyAllWindows()

            return False


    camera.release()

    cv2.destroyAllWindows()


    # --------------------------------------------------------
    # Check enrollment
    # --------------------------------------------------------

    if len(embeddings) < required_samples:

        print("Enrollment failed.")

        return False


    # --------------------------------------------------------
    # Save user
    # --------------------------------------------------------

    user_file = save_user(
        user_id,
        name,
        embeddings
    )


    print("\n========================================")
    print("       ENROLLMENT SUCCESSFUL")
    print("========================================")
    print("User ID:", user_id)
    print("Name:", name)
    print(
        "Embedding dimension:",
        len(embeddings[0])
    )
    print("Saved to:", user_file)
    print("========================================")


    return True


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("========================================")
    print("       USER ENROLLMENT SYSTEM")
    print("========================================")

    user_id = input(
        "Enter User ID: "
    ).strip()

    name = input(
        "Enter Name: "
    ).strip()


    if not user_id or not name:

        print(
            "User ID and Name cannot be empty."
        )

    else:

        enroll_user(
            user_id=user_id,
            name=name,
            required_samples=5
        )

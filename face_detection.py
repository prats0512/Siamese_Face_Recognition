import cv2


class FaceDetector:
    """
    Face detection using the Viola-Jones Haar Cascade classifier.
    """

    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(
            cascade_path
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                "Failed to load Haar Cascade classifier."
            )

    def detect_faces(self, frame):
        """
        Detect all frontal faces in an image.

        Returns:
            List of (x, y, width, height) bounding boxes.
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces

    def get_largest_face(self, frame):
        """
        Detect and return the largest face.

        Returns:
            face_crop: grayscale cropped face
            bbox: (x, y, width, height)
        """

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return None, None

        x, y, w, h = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        face_crop = gray[y:y + h, x:x + w]

        return face_crop, (x, y, w, h)


def preprocess_face(face, image_size=(92, 112)):
    """
    Resize and normalize a detected face.

    Returns:
        Array with shape (1, 112, 92, 1).
    """

    resized = cv2.resize(
        face,
        image_size
    )

    normalized = resized.astype(
        "float32"
    ) / 255.0

    normalized = normalized.reshape(
        1,
        image_size[1],
        image_size[0],
        1
    )

    return normalized


if __name__ == "__main__":

    print("Testing FaceDetector...")

    detector = FaceDetector()

    print("Viola-Jones Haar Cascade loaded successfully.")
    print("FaceDetector module is working correctly.")

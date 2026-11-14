from keras_facenet import FaceNet
from mtcnn import MTCNN
import numpy as np
import cv2

embedder = FaceNet()

print("Input shape:", embedder.model.input_shape)
print("Output shape:", embedder.model.output_shape)

embedder.model.summary()

detector = MTCNN()

# Function to get face embeddings from an image
def get_face_embedding(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face(s) in the image
    faces = detector.detect_faces(image_rgb)
    if not faces:
        return None  # No face detected

    # Take the first detected face and crop it
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)  # avoid negative coordinates
    face = image_rgb[y:y+h, x:x+w]

    # Resize to the size FaceNet expects
    face = cv2.resize(face, (160, 160))

    # Get the embedding from the cropped face
    embeddings = embedder.embeddings([face])
    return np.array(embeddings[0])
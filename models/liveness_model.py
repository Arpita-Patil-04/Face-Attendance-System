import dlib
import cv2
import numpy as np

SHAPE_PREDICTOR_PATH = 'data/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

EAR_CLOSED_THRESHOLD = 0.21   # below this = eye closed
EAR_OPEN_THRESHOLD = 0.25     # at or above this = eye open
MIN_CLOSED_FRAMES = 2         # frames the eyes must stay closed to count as a blink


def point(landmarks, i):
    p = landmarks.part(i)
    return np.array([p.x, p.y], dtype=np.float32)


def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [point(landmarks, i) for i in eye_indices]
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_average_ear(frame, scale=0.5):
    """EAR of the largest face in the frame, or None if no face is found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)  # smaller image = faster
    faces = detector(gray)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda r: r.width() * r.height())
    shape = predictor(gray, face)
    return (eye_aspect_ratio(shape, LEFT_EYE) + eye_aspect_ratio(shape, RIGHT_EYE)) / 2.0


class BlinkDetector:
    """Counts blinks: eyes open -> closed for a few frames -> open again."""

    def __init__(self, closed_threshold=EAR_CLOSED_THRESHOLD,
                 open_threshold=EAR_OPEN_THRESHOLD,
                 min_closed_frames=MIN_CLOSED_FRAMES):
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self.min_closed_frames = min_closed_frames
        self.blink_count = 0
        self.closed_frames = 0
        self.seen_open = False

    def update(self, ear):
        completed = False
        if ear >= self.open_threshold:
            if self.seen_open and self.closed_frames >= self.min_closed_frames:
                self.blink_count += 1
                completed = True
            self.seen_open = True
            self.closed_frames = 0
        elif ear < self.closed_threshold:
            self.closed_frames += 1
        # between the two thresholds: ambiguous, ignore
        return completed
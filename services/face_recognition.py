import cv2
from models.facenet_model import get_face_embedding
from models.liveness_model import check_liveness
from database.db_operations import add_employee, get_all_employee_embeddings, mark_attendance
import numpy as np

# Function to register a new employee
def register_employee(name, image):
    # Check liveness
    if not check_liveness(image):
        print("Liveness check failed. Registration not allowed.")
        return False

    # Get face embedding
    face_embedding = get_face_embedding(image)

    # Add employee to the database
    if face_embedding is not None:
        employee_id = add_employee(name, face_embedding)
        print(f"Employee {name} registered successfully and Generated Id is : {employee_id}")
        return employee_id
    else:
        return None

# Function to recognize face and mark attendance
def recognize_and_mark_attendance(image):
    # Check liveness
    if not check_liveness(image):
        print("Liveness check failed. Access denied.")
        return False

    # Get the embedding of the current face
    current_face_embedding = get_face_embedding(image)

    # Get all employee embeddings from the database
    employees = get_all_employee_embeddings()

    # Define threshold for cosine similarity to consider a match
    threshold = 0.5  # Adjust based on your tests

    for employee_id, name, stored_embedding in employees:
        similarity = cosine_similarity(current_face_embedding, stored_embedding)
        if similarity > threshold:
            # Mark attendance for the matched employee
            mark_attendance(employee_id)
            print(f"Attendance marked for: Name - {name} , Employee id - {employee_id}")
            return name, employee_id

    print("No match found. Access denied.")
    return False

# Function to calculate cosine similarity between two face embeddings
def cosine_similarity(embedding1, embedding2):
    dot_product = embedding1 @ embedding2.T
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)
import numpy as np
from models.facenet_model import get_face_embedding
from database.db_operations import add_employee, get_all_employee_embeddings, mark_attendance

# Register a new employee (liveness is already verified at capture time)
def register_employee(name, image):
    face_embedding = get_face_embedding(image)

    if face_embedding is not None:
        employee_id = add_employee(name, face_embedding)
        print(f"Employee {name} registered successfully and Generated Id is : {employee_id}")
        return employee_id
    else:
        print("No face detected. Registration failed.")
        return None

# Recognize face and mark attendance (liveness is already verified at capture time)
def recognize_and_mark_attendance(image):
    current_face_embedding = get_face_embedding(image)
    if current_face_embedding is None:
        print("No face detected. Access denied.")
        return False

    employees = get_all_employee_embeddings()

    threshold = 0.8 

    for employee_id, name, stored_embedding in employees:
        similarity = cosine_similarity(current_face_embedding, stored_embedding)
        if similarity > threshold:
            mark_attendance(employee_id)
            print(f"Attendance marked for: Name - {name} , Employee id - {employee_id}")
            return name, employee_id

    print("No match found. Access denied.")
    return False

# Cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    dot_product = embedding1 @ embedding2.T
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)
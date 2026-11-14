import time
import cv2
from services.face_recognition import register_employee, recognize_and_mark_attendance
from database.db_operations import get_attendance_by_emp_id
from models.liveness_model import BlinkDetector, get_average_ear

# Initialize webcam
cap = cv2.VideoCapture(0)

def capture_live_face(cap, required_blinks=2, timeout_seconds=8):
    """Shows the webcam, waits for the required blinks, returns an open-eyes frame or None."""
    blink_detector = BlinkDetector()
    start = time.time()

    while time.time() - start < timeout_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        ear = get_average_ear(frame)
        display = frame.copy()

        if ear is None:
            status = 'No face detected'
        else:
            blink_detector.update(ear)
            status = f'EAR {ear:.2f} | Blinks {blink_detector.blink_count}/{required_blinks}'
            if blink_detector.blink_count >= required_blinks:
                cv2.destroyAllWindows()
                return frame  # eyes just reopened, good frame for FaceNet

        cv2.putText(display, f'Blink {required_blinks} times to verify', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Liveness check - press q to cancel', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return None

# Fetch attendance by Employee ID
def fetch_attendance(emp_id):
    attendance_records = get_attendance_by_emp_id(emp_id)

    if attendance_records:
        print(f"Attendance records for Employee id {emp_id}:")
        for record in attendance_records:
            print(f"Attendance details: {record}")
    else:
        print(f"No attendance records found for Employee id {emp_id}")

# Main menu loop
def main():
    while True:
        print("\n1. Register a new employee")
        print("2. Recognize and mark attendance")
        print("3. Check attendance by Employee ID")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            name = input("Enter employee name: ")
            print("Starting liveness check for registration...")
            image = capture_live_face(cap)
            if image is None:
                print("Liveness check failed or cancelled.")
                continue

            employee_id = register_employee(name, image)
            if employee_id:
                print(f"Successfully registered {name}. Your Employee ID is: {employee_id}")
            else:
                print("Registration failed.")

        elif choice == "2":
            print("Starting liveness check for recognition...")
            image = capture_live_face(cap)
            if image is None:
                print("Liveness check failed or cancelled.")
                continue

            result = recognize_and_mark_attendance(image)
            if result:
                name, employee_id = result
                print(f"Attendance marked for {name}. Employee ID: {employee_id}")
            else:
                print("Face not recognized. Access denied.")

        elif choice == "3":
            emp_id = input("Enter Employee ID: ")
            fetch_attendance(emp_id)

        elif choice == "4":
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
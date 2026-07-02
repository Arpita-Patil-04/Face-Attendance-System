import cv2
from services.face_recognition import register_employee, recognize_and_mark_attendance
from database.db_operations import get_attendance_by_emp_id, get_attendance_by_name

# Initialize webcam
cap = cv2.VideoCapture(0)

def capture_image():
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam - Press "c" to capture or "q" to quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

# Function to fetch attendance by Employee ID
def fetch_attendance(emp_id):
    attendance_records = get_attendance_by_emp_id(emp_id)

    if attendance_records:
        print(f"Attendance records for Employee id {emp_id}:")
        for record in attendance_records:
            print(f"Attendance details: {record}")
    else:
        print(f"No attendance records found for Employee id {emp_id}")

# Main logic to handle options
def main():
    while True:
        print("\n1. Register a new employee")
        print("2. Recognize and mark attendance")
        print("3. Check attendance by Employee ID")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            name = input("Enter employee name: ")
            print("Capturing image for registration...")
            image = capture_image()
            employee_id = register_employee(name, image)
            if employee_id:
                print(f"Successfully registered {name}. Your Employee ID is: {employee_id}")
            else:
                print("Registration failed.")

        elif choice == "2":
            print("Capturing image for recognition...")
            image = capture_image()
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

if __name__ == "__main__":
    main()
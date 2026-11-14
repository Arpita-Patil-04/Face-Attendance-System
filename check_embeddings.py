import sqlite3

conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

cursor.execute('SELECT id, name, face_embedding FROM employees')
for emp_id, name, blob in cursor.fetchall():
    print(f"{name} (id {emp_id}): {len(blob)} bytes -> {len(blob) // 4} float32 numbers")

conn.close()
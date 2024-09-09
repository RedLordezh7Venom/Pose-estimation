import cv2
import mediapipe as mp
import math
import time
import csv

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate the angle between three points.
def calculate_angle(a, b, c):
    angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    angle = angle + 360 if angle < 0 else angle
    return angle

# Function to select the pose.
def choose_pose(choosed_number):
    switcher = {
        1: "warriorII",
        2: "cat",
        3: "chair",
        4: "triangle",
    }
    return switcher.get(choosed_number, "nothing")

# Load video
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to a video file.

# Main menu
print("_____________________________________________________________________________________________")
accurate_angle_list = []
choosed_number = int(input("\tEnter Pose to perform:-\n\t\t1.WarriorII Pose\n\t\t2.Cat Pose\n\t\t3.Chair Pose\n\t\t4.Triangle Pose\n\t"))
selected_pose = choose_pose(choosed_number).replace(" ", "")
print(f"**************** Selected Pose: {selected_pose} Pose ****************")

# Load pose angles from CSV.
with open('fetchSet.csv', 'r') as inputCSV:
    for row in csv.reader(inputCSV):
        if row[8] == selected_pose:
            accurate_angle_list = [int(row[i]) for i in range(8)]
            break
print(f"**************** List: {accurate_angle_list} ****************")
print("_____________________________________________________________________________________________")

# Points for different angles.
angle_coordinates = [
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),  # Rshoulder
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),    # Lshoulder
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),   # Relbow
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),      # Lelbow
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),       # Rhip
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),          # Lhip
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE),       # Rknee
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE)           # Lknee
]

angle_name_list = ["Rshoulder", "Lshoulder", "Relbow", "Lelbow", "Rhip", "Lhip", "Rknee", "Lknee"]
correction_value = 13  # Tolerance for angle correctness
fps_time = 0
pos_on_screen = [(10, 50), (10, 100), (10, 400), (10, 450), (10, 500), (400, 400), (400, 450), (400, 500)]

# Processing video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect pose.
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        correct_angle_count = 0

        # Check angles
        for itr in range(8):
            a, b, c = angle_coordinates[itr]
            angle_obtained = calculate_angle(
                [landmarks[a.value].x, landmarks[a.value].y],
                [landmarks[b.value].x, landmarks[b.value].y],
                [landmarks[c.value].x, landmarks[c.value].y]
            )

            if angle_obtained < accurate_angle_list[itr] - correction_value:
                status = "extend more"
            elif angle_obtained > accurate_angle_list[itr] + correction_value:
                status = "extend less"
            else:
                status = "OK"
                correct_angle_count += 1

            cv2.putText(frame, f"{angle_name_list[itr]}: {status}", pos_on_screen[itr],
                        cv2.FONT_HERSHEY_PLAIN, 1.4, (240,0,255), 1)

        # Display posture correctness
        posture = "CORRECT" if correct_angle_count > 6 else "WRONG"
        cv2.putText(frame, f"Posture: {posture}", (400, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {1.0 / (time.time() - fps_time):.2f}", (400, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fps_time = time.time()

    # Resize the frame to a larger size
    frame = cv2.resize(frame, (1280, 720))  # Resize to 1280x720
    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Break on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

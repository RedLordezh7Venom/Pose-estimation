import cv2
import mediapipe as mp
import math
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Video Capture
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

def calculate_angle(p1, p2, p3):
    """ Calculate angle between three points. """
    angle = math.degrees(
        math.atan2(p3[1] - p2[1], p3[0] - p2[0]) -
        math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    )
    return abs(angle)

def euclidean_distance(p1, p2):
    """ Calculate Euclidean distance between two points. """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def load_pose_data(filename):
    """ Load pose data from a CSV file. """
    try:
        return pd.read_csv(filename, index_col='Pose')
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        exit()

def check_pose(current_angles, current_distances, pose_data, pose_name):
    """ Check if the current angles and distances match the reference pose data. """
    if pose_name not in pose_data.index:
        return "Pose Not Defined", (0, 0, 255)  # Red for undefined pose
    
    ref_data = pose_data.loc[pose_name]
    
    # Define tolerance level
    tolerance = 10  # degrees for angles, 0.05 for distances
    
    angle_diff_right = abs(current_angles[0] - ref_data['Elbow_Angle_Right'])
    angle_diff_left = abs(current_angles[1] - ref_data['Elbow_Angle_Left'])
    distance_diff_right = abs(current_distances['right'] - ref_data['Distance_Right'])
    distance_diff_left = abs(current_distances['left'] - ref_data['Distance_Left'])
    
    if (angle_diff_right < tolerance and angle_diff_left < tolerance and
        distance_diff_right < tolerance and distance_diff_left < tolerance):
        return f"{pose_name} Correct", (0, 255, 0)  # Green for correct posture
    
    return "Incorrect Posture", (0, 0, 255)  # Red for incorrect posture

def main():
    # Load pose data from CSV
    pose_data = load_pose_data('yoga_poses.csv')

    # User selects the yoga pose
    print("Select a yoga pose from the following options:")
    for i, pose in enumerate(pose_data.index, 1):
        print(f"{i}. {pose}")
    
    choice = input("Enter the number corresponding to your choice: ")
    
    pose_dict = {str(i+1): pose for i, pose in enumerate(pose_data.index)}
    
    if choice not in pose_dict:
        print("Invalid choice. Exiting.")
        cam.release()
        cv2.destroyAllWindows()
        return
    
    selected_pose = pose_dict[choice]
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image. Exiting.")
            break
        
        # Process the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints
            WR = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            WL = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            ER = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            EL = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            KR = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            KL = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            SHR = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            SHL = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

            # Calculate angles
            elbow_knee_angle_right = calculate_angle(
                (SHR.x, SHR.y), (ER.x, ER.y), (KR.x, KR.y)
            )
            elbow_knee_angle_left = calculate_angle(
                (SHL.x, SHL.y), (EL.x, EL.y), (KL.x, KL.y)
            )

            # Calculate distances between knees and elbows
            knee_to_elbow_dist_right = euclidean_distance((KR.x, KR.y), (ER.x, ER.y))
            knee_to_elbow_dist_left = euclidean_distance((KL.x, KL.y), (EL.x, EL.y))

            distances = {
                'right': knee_to_elbow_dist_right,
                'left': knee_to_elbow_dist_left
            }

            angles = (elbow_knee_angle_right, elbow_knee_angle_left)

            # Check posture against reference data
            posture_status, color = check_pose(angles, distances, pose_data, selected_pose)

            # Display results
            cv2.putText(frame, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Yoga Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
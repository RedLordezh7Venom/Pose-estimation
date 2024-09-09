import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Video Capture
cam = cv2.VideoCapture(0)

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

# Define functions for different yoga poses
def check_downward_dog(angles, distances):
    elbow_angle_right, elbow_angle_left = angles
    distance_right, distance_left = distances
    
    # Example criteria for Downward Dog (adjust as needed)
    if (elbow_angle_right > 150 and elbow_angle_left > 150 and
        distance_right < 0.2 and distance_left < 0.2):
        return True
    return False

def check_tree_pose(angles, distances):
    # Example criteria for Tree Pose (placeholder, adjust as needed)
    return False

def check_warrior_ii(angles, distances):
    # Example criteria for Warrior II (placeholder, adjust as needed)
    return False

def check_triangle_pose(angles, distances):
    # Example criteria for Triangle Pose (placeholder, adjust as needed)
    return False

def check_plank_pose(angles, distances):
    # Example criteria for Plank Pose (placeholder, adjust as needed)
    return False

# Create a dictionary of pose functions
pose_functions = {
    'Downward Dog': check_downward_dog,
    'Tree Pose': check_tree_pose,
    'Warrior II': check_warrior_ii,
    'Triangle Pose': check_triangle_pose,
    'Plank Pose': check_plank_pose
}

def main():
    # User selects the yoga pose
    print("Select a yoga pose from the following options:")
    print("1. Downward Dog")
    print("2. Tree Pose")
    print("3. Warrior II")
    print("4. Triangle Pose")
    print("5. Plank Pose")
    choice = input("Enter the number corresponding to your choice: ")
    
    pose_dict = {
        '1': 'Downward Dog',
        '2': 'Tree Pose',
        '3': 'Warrior II',
        '4': 'Triangle Pose',
        '5': 'Plank Pose'
    }
    
    if choice not in pose_dict:
        print("Invalid choice. Exiting.")
        return
    
    selected_pose = pose_dict[choice]
    
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
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

            # Determine if the posture is correct for the selected pose
            if selected_pose in pose_functions:
                if pose_functions[selected_pose](angles, distances):
                    posture_status = f"{selected_pose} Correct"
                    color = (0, 255, 0)  # Green for correct posture
                else:
                    posture_status = "Incorrect Posture"
                    color = (0, 0, 255)  # Red for incorrect posture
            else:
                posture_status = "Pose Not Defined"
                color = (0, 0, 255)  # Red for undefined pose

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

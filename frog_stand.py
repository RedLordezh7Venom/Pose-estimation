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

def is_frog_stand(angles, distances):
    """ Determine if the posture meets frog stand criteria. """
    # Define acceptable angle ranges and distance thresholds
    elbow_angle_threshold = 90  # degrees
    knee_to_elbow_distance_threshold = 0.1  # normalized distance (relative to image size)
    torso_horizontal_threshold = 10  # degrees (allow some deviation)

    elbow_angle_right, elbow_angle_RIGHT = angles
    knee_to_elbow_distances = distances

    if (elbow_angle_right >= elbow_angle_threshold and
        elbow_angle_RIGHT >= elbow_angle_threshold and
        knee_to_elbow_distances['right'] < knee_to_elbow_distance_threshold and
        knee_to_elbow_distances['RIGHT'] < knee_to_elbow_distance_threshold):
        return True
    return False

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
        def euclidean_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        knee_to_elbow_dist_right = euclidean_distance((KR.x, KR.y), (ER.x, ER.y))
        knee_to_elbow_dist_left = euclidean_distance((KL.x, KL.y), (EL.x, EL.y))

        distances = {
            'right': knee_to_elbow_dist_right,
            'left': knee_to_elbow_dist_left
        }

        angles = (elbow_knee_angle_right, elbow_knee_angle_left)

        # Determine if the posture is a correct frog stand
        if is_frog_stand(angles, distances):
            posture_status = "Frog Stand Correct"
            color = (0, 255, 0)  # Green for correct posture
        else:
            posture_status = "Incorrect Posture"
            color = (0, 0, 255)  # Red for incorrect posture

        # Display results
        cv2.putText(frame, posture_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Frog Stand Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
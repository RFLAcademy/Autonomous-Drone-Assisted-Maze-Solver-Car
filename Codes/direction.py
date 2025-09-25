#simple code to test yaw degree of apriltag
import cv2
import numpy as np
from pupil_apriltags import Detector

# --- Camera parameters (in pixels) ---
fx = 600  # focal length in x
fy = 600  # focal length in y
cx = 320  # optical center x
cy = 240  # optical center y
tag_size = 0.05  # size of AprilTag in meters

detector = Detector(families='tag36h11')

cap = cv2.VideoCapture(1)  # change to your webcam index

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=tag_size
    )

    for det in detections:
        # Rotation matrix (3x3) and translation vector (3x1)
        R = np.array(det.pose_R)
        t = np.array(det.pose_t).flatten()

        # --- Calculate yaw (rotation around vertical axis) ---
        # Extract yaw from rotation matrix
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

        # Draw the detection
        cv2.polylines(frame, [np.int32(det.corners)], True, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f} deg", 
                    (int(det.center[0]), int(det.center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        print(f"Tag ID: {det.tag_id}, Yaw: {yaw:.1f} degrees, Pos: {t}")

    cv2.imshow("AprilTag Direction", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

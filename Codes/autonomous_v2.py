# auto_drive_pi_step_control_stop_v2.py

#started using apriltag as starting point

import cv2
import numpy as np
import time, math, requests
from pupil_apriltags import Detector

# ---------------- CONFIG ----------------
ESP32_IP = "http://192.168.4.1"
YAW_TOLERANCE = 10         # stop correcting if within ±10°
FORWARD_POWER = 150
TURN_POWER = 110           # enough torque to turn
STEP_DURATION = 0.15       # shorter moves → smaller yaw change
END_RADIUS_FACTOR = 0.05   # 5% of frame width ≈ ~10 cm (auto scaling)
DETECTOR = Detector(families="tag36h11")

# ---------------- HELPERS ----------------
def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def angle_difference(current, target):
    return normalize_angle(target - current)

def path_angle_to_tag_yaw(path_angle):
    return normalize_angle(path_angle + 90.0)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def send_step(endpoint, power=None, duration=STEP_DURATION):
    """Send movement, stop, then wait 2s."""
    url = f"{ESP32_IP}/{endpoint}"
    if power is not None:
        url += f"?power={power}"
    try:
        requests.get(url, timeout=0.5)
        print(f"▶️ Move: {url}")
    except:
        print("⚠️ Command failed")

    time.sleep(duration)

    try:
        requests.get(f"{ESP32_IP}/stop", timeout=0.5)
        print("⏹ Stop")
    except:
        print("⚠️ Stop failed")

    print("⏸ Waiting 2s...")
    time.sleep(0.3)

# ---------------- DETECTION ----------------
def detect_apriltag(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = DETECTOR.detect(gray)
    if len(detections) > 0:
        d = detections[0]
        cx, cy = d.center
        yaw = math.degrees(math.atan2(
            d.corners[1][1] - d.corners[0][1],
            d.corners[1][0] - d.corners[0][0]
        ))
        return (int(cx), int(cy)), yaw
    return None, None

def detect_end(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        return (cx, cy)
    return None

# ---------------- MAIN ----------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    END_RADIUS = frame_width * END_RADIUS_FACTOR   # scale with frame size

    print(f"📏 Using END_RADIUS={END_RADIUS:.1f}px")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start, yaw = detect_apriltag(frame)
        end = detect_end(frame)

        if not start or not end:
            print("⚠️ AprilTag or End not detected")
            time.sleep(0.2)
            continue

        # check if close enough to end point
        dist = distance(start, end)
        if dist < END_RADIUS:
            print(f"🎉 Reached destination! dist={dist:.1f}px → stopping")
            requests.get(f"{ESP32_IP}/stop", timeout=0.5)
            break

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        path_angle = math.degrees(math.atan2(dy, dx))

        target_yaw = path_angle_to_tag_yaw(path_angle)
        diff = angle_difference(yaw, target_yaw)

        print(f"Start={start} End={end} | dist={dist:.1f}px "
              f"path_angle={path_angle:.1f}° target_yaw={target_yaw:.1f}° "
              f"current_yaw={yaw:.1f}° diff={diff:.1f}°")

        if abs(diff) > YAW_TOLERANCE:
            if diff > 0:
                send_step("right", TURN_POWER)
            else:
                send_step("left", TURN_POWER)
        else:
            send_step("forward", FORWARD_POWER)

    cap.release()
    print("✅ Mission complete.")

#simple code to test how much overshoot is built in due to lag
# auto_drive_pi_straight_stop_visual.py

import cv2
import numpy as np
import math, time, requests
from pupil_apriltags import Detector

# ---------------- CONFIG ----------------
ESP32_IP = "http://192.168.4.1"
FORWARD_POWER = 150
END_RADIUS_FACTOR = 0.05   # 5% of frame width ≈ ~10 cm
DETECTOR = Detector(families="tag36h11")

# ---------------- HELPERS ----------------
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def send_forward(power=FORWARD_POWER):
    try:
        requests.get(f"{ESP32_IP}/forward?power={power}", timeout=0.3)
    except:
        print("⚠️ Forward command failed")

def send_stop():
    try:
        requests.get(f"{ESP32_IP}/stop", timeout=0.3)
    except:
        print("⚠️ Stop command failed")

# ---------------- DETECTION ----------------
def detect_apriltag(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = DETECTOR.detect(gray)
    if len(detections) > 0:
        d = detections[0]
        cx, cy = d.center
        return (int(cx), int(cy))
    return None

def detect_end(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([100, 120, 50])
    upper_pink = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

# ---------------- MAIN ----------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    END_RADIUS = frame_width * END_RADIUS_FACTOR

    print(f"📏 Using END_RADIUS={END_RADIUS:.1f}px")

    # Start moving forward
    send_forward()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        start = detect_apriltag(frame)
        end = detect_end(frame)

        # Draw detections
        if start:
            cv2.circle(frame, start, 10, (255, 0, 0), -1)  # blue dot = AprilTag
            cv2.putText(frame, "START", (start[0] + 10, start[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if end:
            cv2.circle(frame, end, 10, (0, 0, 255), -1)  # red dot = end
            cv2.putText(frame, "END", (end[0] + 10, end[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if start and end:
            cv2.line(frame, start, end, (0, 255, 0), 2)  # green path
            dist = distance(start, end)
            cv2.putText(frame, f"dist={dist:.1f}px", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Stop condition
            if dist < END_RADIUS:
                print("🎉 Reached destination → STOP")
                send_stop()
                break

        cv2.imshow("Live Path", frame)

        # Press 'q' to manually quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            send_stop()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Mission complete.")

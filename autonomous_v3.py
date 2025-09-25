# integrated_waypoint_nav_v2.py
# Uses your EXACT path detection algorithm + your step-control driver
# Strategy: start -> next turning point -> ... -> end, then stop.

# first version which implemented obstacles

import cv2
import numpy as np
import time, math, requests
import matplotlib.pyplot as plt
from scipy.ndimage import grey_dilation
from pupil_apriltags import Detector

# ===================== CONFIG =====================
# If you use a single camera, set both to the same index.
PLANNING_CAM_INDEX = 0   # your path-detection code used cam=1
CONTROL_CAM_INDEX  = 0   # your driving code used cam=0

ESP32_IP = "http://192.168.4.1"
DETECTOR = Detector(families="tag36h11")

# Grid/cell params (match your algorithm)
CELL_SIZE = 10           # must match 'factor' below
FACTOR = 10              # downscale factor used in resize_for_pathfinding
INFLATION_RADIUS = 5     # same as your code

# Driving/control (your values)
YAW_TOLERANCE = 10
FORWARD_POWER = 110
TURN_POWER = 130
STEP_DURATION = 0.15
SETTLE_DELAY = 2.0
END_RADIUS_FACTOR = 0.05     # final end tolerance (~5% of frame width)
WAYPOINT_RADIUS_FACTOR = 0.03  # intermediate waypoint tolerance (~3%)

SHOW_PLANNING_FIG = True
SHOW_LIVE_OVERLAY = True

# ===================== HELPERS (CONTROL) =====================
def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def angle_difference(current, target):
    return normalize_angle(target - current)

def path_angle_to_tag_yaw(path_angle):
    # keep your convention
    return normalize_angle(path_angle + 90.0)

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def send_step(endpoint, power=None, duration=STEP_DURATION, settle=SETTLE_DELAY):
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

    if settle > 0:
        print(f"⏸ Waiting {settle:.1f}s...")
        time.sleep(settle)

def detect_apriltag_pose(frame):
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

# ===================== YOUR EXACT PATH DETECTION ALGO =====================
def heuristic(a, b):
    # Euclidean gives smoother paths with diagonals
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def astar(grid, start, end):
    import heapq
    neighbors = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, end)}
    oheap = [(fscore[start], start)]

    while oheap:
        _, current = heapq.heappop(oheap)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        close_set.add(current)

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue

            tentative_g = gscore[current] + heuristic(current, neighbor)

            if neighbor in close_set and tentative_g >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g
                fscore[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return []

def detect_start_end_obstacles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Yellow for start
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))

    # blue for end
    blue_mask = cv2.inRange(hsv, (65, 50, 100), (85, 255, 255))  # end before 160

    # Red obstacles (both ends of HSV range)
    lower_red1 = cv2.inRange(hsv, (0, 100, 50), (12, 255, 255))
    upper_red2 = cv2.inRange(hsv, (170, 100, 50), (179, 255, 255))  # shrink from 180 to 170
    red_mask = cv2.bitwise_or(lower_red1, upper_red2)

    # Green obstacles
    green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))

    # Combine red and green masks for obstacles
    obstacle_mask = cv2.bitwise_or(red_mask, green_mask)

    def find_centroid(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None

    start = find_centroid(yellow_mask)
    end = find_centroid(blue_mask)

    return start, end, obstacle_mask

def pixel_to_grid(pixel, cell_size=CELL_SIZE):
    return (pixel[1] // cell_size, pixel[0] // cell_size)

def grid_to_pixel(grid, cell_size=CELL_SIZE):
    return (grid[1] * cell_size + cell_size // 2, grid[0] * cell_size + cell_size // 2)

def resize_for_pathfinding(mask, factor):
    small = cv2.resize(mask, (mask.shape[1] // factor, mask.shape[0] // factor), interpolation=cv2.INTER_NEAREST)
    return (small > 0).astype(np.uint8)

def inflate_obstacles(grid, inflation_radius=1):
    return grey_dilation(grid, size=(2 * inflation_radius + 1, 2 * inflation_radius + 1))

# (kept from your earlier version so visualize_grid calls don't break)
def visualize_grid(grid, start=None, end=None):
    vis = (grid * 255).astype(np.uint8)
    if start:
        vis[start[0], start[1]] = 100  # gray for start
    if end:
        vis[end[0], end[1]] = 200      # lighter gray for end
    plt.figure(); plt.imshow(vis, cmap='gray'); plt.title("Grid with Obstacles, Start & End"); plt.show()

# Turning points (same logic you used earlier)
def get_direction_changes(path):
    if len(path) < 3:
        return []
    turns = []
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for i in range(2, len(path)):
        curr_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        if curr_dir != prev_dir:
            turns.append(path[i-1])  # Turning point
        prev_dir = curr_dir
    return turns

# ===================== MAIN =====================
def main():
    # ---------- PATHFINDING PHASE (your exact loop, wrapped) ----------
    cap_plan = cv2.VideoCapture(PLANNING_CAM_INDEX)
    path = []
    path_pixels = []
    frame = None
    inflated_grid = None

    start_time = time.time()
    while True:
        ret, frame = cap_plan.read()
        if not ret:
            print("Camera read failed")
            continue

        start_px, end_px, obstacle_mask = detect_start_end_obstacles(frame)
        print("Start:", start_px, "End:", end_px)
        
        factor = FACTOR
        if start_px and end_px:
            # Convert obstacle mask to binary grid
            obstacle_grid = resize_for_pathfinding(obstacle_mask, factor)

            # Convert pixel coordinates to grid
            start_grid = pixel_to_grid(start_px)
            end_grid = pixel_to_grid(end_px)

            # Clear start and end locations before inflation
            obstacle_grid[start_grid[0], start_grid[1]] = 0
            obstacle_grid[end_grid[0], end_grid[1]] = 0

            # Inflate obstacles
            inflated_grid = inflate_obstacles(obstacle_grid, inflation_radius=INFLATION_RADIUS)

            # Run A*
            path = astar(inflated_grid, start_grid, end_grid)
            visualize_grid(inflated_grid, start_grid, end_grid)

            if path:
                print("Path found! Exiting loop.")
                path_pixels = [grid_to_pixel(pt) for pt in path]
                break
            else:
                print("No path found")

        # Optional: timeout
        if time.time() - start_time > 10:
            print("Timeout: Could not find valid path in time.")
            break

    cap_plan.release()

    if not path:
        print("❌ No path computed. Aborting.")
        return

    visualize_grid(inflated_grid, pixel_to_grid(start_px), pixel_to_grid(end_px))

    # Draw (debug)
    output = frame.copy()
    for pt in path_pixels:
        cv2.circle(output, pt, 2, (0, 0, 0), -1)
    if path_pixels:
        cv2.circle(output, path_pixels[0], 5, (0, 255, 255), -1)   # start
        cv2.circle(output, path_pixels[-1], 5, (255, 0, 255), -1)  # end
    if SHOW_PLANNING_FIG:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1); plt.title("Original Image")
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.subplot(1, 2, 2); plt.title("Path Detected")
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.show()

    # Turning points and targets (waypoints)
    turning_points = get_direction_changes(path)
    turning_pixels = [grid_to_pixel(pt) for pt in turning_points]
    targets = list(turning_pixels)
    if path_pixels:
        end_pix = path_pixels[-1]
        if len(targets) == 0 or distance(targets[-1], end_pix) > 1.0:
            targets.append(end_pix)

    print("Direction change points (grid):")
    for pt in turning_points:
        print("→", pt)
    print("Direction change points (pixels):")
    for pt in turning_pixels:
        print("→", pt)

    # ---------- DRIVING PHASE (sequential waypoint loop) ----------
    cap_ctrl = cv2.VideoCapture(CONTROL_CAM_INDEX)
    if not cap_ctrl.isOpened():
        print("❌ Control camera open failed"); return

    W = int(cap_ctrl.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
    WAYPOINT_RADIUS = W * WAYPOINT_RADIUS_FACTOR
    END_RADIUS = W * END_RADIUS_FACTOR
    print(f"📏 Radii: waypoint={WAYPOINT_RADIUS:.1f}px, end={END_RADIUS:.1f}px")

    current_idx = 0
    total_targets = len(targets)
    if total_targets == 0:
        print("ℹ️ No turning points; driving directly to end.")
        targets = [end_pix]; total_targets = 1

    print(f"🏁 Total targets: {total_targets}")
    print("🚗 Starting waypoint navigation...")

    while current_idx < total_targets:
        ret, frame_ctrl = cap_ctrl.read()
        if not ret:
            continue

        robot_pos, yaw = detect_apriltag_pose(frame_ctrl)
        if robot_pos is None or yaw is None:
            print("⚠️ AprilTag not detected")
            time.sleep(0.2)
            continue

        target = targets[current_idx]
        arrive_r = END_RADIUS if (current_idx == total_targets - 1) else WAYPOINT_RADIUS

        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        dist = math.hypot(dx, dy)
        path_angle = math.degrees(math.atan2(dy, dx))
        target_yaw = path_angle_to_tag_yaw(path_angle)
        diff = angle_difference(yaw, target_yaw)

        print(f"[{current_idx+1}/{total_targets}] dist={dist:.1f}px | "
              f"path_angle={path_angle:.1f}° target_yaw={target_yaw:.1f}° "
              f"yaw={yaw:.1f}° diff={diff:.1f}°")

        # Arrived at waypoint?
        if dist < arrive_r:
            print(f"✅ Reached target {current_idx+1}/{total_targets}")
            current_idx += 1
            time.sleep(0.5)
            continue

        # Control: turn until yaw error small, then step forward
        if abs(diff) > YAW_TOLERANCE:
            if diff > 0:
                send_step("right", TURN_POWER)
            else:
                send_step("left", TURN_POWER)
        else:
            send_step("forward", FORWARD_POWER)

        if SHOW_LIVE_OVERLAY:
            disp = frame_ctrl.copy()
            # planned path dots
            for p in path_pixels: cv2.circle(disp, p, 2, (0,0,0), -1)
            # robot + current target + arrival circle
            cv2.circle(disp, robot_pos, 5, (0,0,255), -1)
            cv2.circle(disp, target, 6, (0,255,0), 2)
            cv2.circle(disp, target, int(arrive_r), (0,255,0), 1)
            cv2.putText(disp, f"Yaw {yaw:.1f}  Target {target_yaw:.1f}  Diff {diff:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 2)
            cv2.imshow("Live Nav", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Stop & cleanup
    try:
        requests.get(f"{ESP32_IP}/stop", timeout=0.5)
    except: pass
    cap_ctrl.release()
    cv2.destroyAllWindows()
    print("🎉 Final destination reached. Mission complete.")

if __name__ == "__main__":
    main()
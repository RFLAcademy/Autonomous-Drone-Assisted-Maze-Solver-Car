import requests
import time

ESP32_IP = "192.168.4.1"  # ESP32 in AP mode
BASE_URL = f"http://{ESP32_IP}"

def send_cmd(cmd):
    try:
        requests.get(f"{BASE_URL}/{cmd}", timeout=1)
    except Exception as e:
        print(f"Error sending {cmd}: {e}")

def drive_square():
    forward_time = 4  # seconds to travel ~4 feet (adjust after testing)
    turn_time = 0.5     # seconds for ~90° turn (adjust after testing)

    for i in range(4):
        print(f"Side {i+1}: Forward")
        send_cmd("forward")
        time.sleep(forward_time)
        send_cmd("stop")
        time.sleep(0.3)

        print(f"Turn {i+1}: Right")
        send_cmd("right")
        time.sleep(turn_time)
        send_cmd("stop")
        time.sleep(0.3)

    print("Square complete!")

if __name__ == "__main__":
    drive_square()
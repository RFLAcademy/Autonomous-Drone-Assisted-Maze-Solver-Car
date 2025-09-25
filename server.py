# server.py
from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/start", methods=["GET"])
def start_robot():
    # Run your autonomous script
    subprocess.Popen(["python3", "/home/pi/autonomous_v5.py"])
    return "✅ autonomous_v5.py started"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
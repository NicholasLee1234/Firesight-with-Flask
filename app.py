from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO

# Initialisation Params
app = Flask(__name__)
model = YOLO("yolov8n.pt")

# Check if human is present
person_detected = 0  # Default
def human_presence_check(class_list: list) -> int:
    return 1 if "person" in class_list else 0

# Video 
def generate_frames():
    global person_detected

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

    if not camera.isOpened():
        print("[ERROR] Could not open camera")
        return

    while True:
        success, frame = camera.read()
        if not success:
            continue

        try:
            results = model(frame, conf=0.4)
            result = results[0]

            # Detected classes in list
            if result.boxes is not None and len(result.boxes) > 0:
                class_ids = result.boxes.cls.tolist()
                class_names = [model.names[int(cls)] for cls in class_ids]
            else:
                class_names = []

            person_detected = human_presence_check(class_names)

            annotated_frame = result.plot()
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

        except Exception as e:
            print("[ERROR]", e)
            break

    camera.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health_stats')
def health_stats():
    return render_template('health_stats.html')


@app.route('/slam_map')
def slam_map():
    return render_template('slam_map.html')


@app.route('/thermal')
def thermal():
    return render_template('thermal.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection_status')
def detection_status():
    global person_detected
    return jsonify({"person": person_detected})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
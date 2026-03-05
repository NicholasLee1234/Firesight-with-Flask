from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import threading


app = Flask(__name__)
model = YOLO("yolov8n.pt")  

# Global detection states for both cameras
person_detected_cam0 = 0
person_detected_cam1 = 0
running = True

def human_presence_check(class_list: list) -> int:
    return 1 if "person" in class_list else 0

def generate_frames(camera_index: int):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}")
        return

    global person_detected_cam0, person_detected_cam1

    while running:
        success, frame = camera.read()
        if not success:
            continue

        try:
            results = model(frame, conf=0.4)
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                class_ids = result.boxes.cls.tolist()
                class_names = [model.names[int(cls)] for cls in class_ids]
            else:
                class_names = []

            if camera_index == 0:
                person_detected_cam0 = human_presence_check(class_names)
            elif camera_index == 1:
                person_detected_cam1 = human_presence_check(class_names)

            # Annotate and send the frame
            annotated_frame = result.plot()
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

        except Exception as e:
            print(f"[ERROR - Camera {camera_index}]", e)
            break

    camera.release()

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


# ---- Video Feeds ----
@app.route('/video_feed_cam0')
def video_feed_cam0():
    return Response(generate_frames(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_cam1')
def video_feed_cam1():
    return Response(generate_frames(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection_status_cam0')
def detection_status_cam0():
    global person_detected_cam0
    return jsonify({"person": person_detected_cam0})

@app.route('/detection_status_cam1')
def detection_status_cam1():
    global person_detected_cam1
    return jsonify({"person": person_detected_cam1})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

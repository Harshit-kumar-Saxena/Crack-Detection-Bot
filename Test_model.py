import cv2
import numpy as np
import time  # For timestamp when saving images
import os   # Optional: to create directory for screenshots

# === CONFIGURATION ===
weights_path = "/home/harshitji/vscode/python/main-pr/yolov4-tiny-custom_final.weights"
config_path = "/home/harshitji/vscode/python/main-pr/yolov4-tiny-custom.cfg"
names_path = "/home/harshitji/vscode/python/main-pr/obj.names"

conf_threshold = 0.2
nms_threshold = 0.4

# === Create output directory if it doesn't exist ===
output_dir = "/home/harshitji/vscode/python/main-pr/crack_screenshots"
os.makedirs(output_dir, exist_ok=True)

# === LOAD YOLO MODEL ===
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels
with open(names_path, "r") as f:
    classes = f.read().strip().split("\n")

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# === VIDEO SOURCE (change index if needed) ===
cap = cv2.VideoCapture(2)

# === Preprocessing Function ===
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(gray_eq, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(opened, (3, 3), 0)
    preprocessed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    return preprocessed

# === MAIN LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or failed.")
        break

    height, width = frame.shape[:2]
    processed_frame = preprocess_frame(frame)

    blob = cv2.dnn.blobFromImage(processed_frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    crack_detected = False

    if len(indices) > 0:
        crack_detected = True
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # === Save screenshot if crack detected ===
    if crack_detected:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"crack_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Crack detected and saved: {filename}")

    # Show the frame
    cv2.imshow("Crack Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()



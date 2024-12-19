import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

model_path = "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx"
artifacts_folder = "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/artifacts"


providers = ['TIDLExecutionProvider', 'TIDLCompilationProvider', 'CPUExecutionProvider', {"artifacts_folder": artifacts_folder}]

print("Starting...")
session = ort.InferenceSession(model_path, providers=providers)

print("Naish")
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]


def preprocess_image(image, input_size=(416, 416)):
    image = Image.fromarray(image).convert("RGB")
    image = image.resize(input_size)
    image_data = np.array(image, dtype=np.float32) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    return image_data


print("Opening Cam")

cap = cv2.VideoCapture("usb camera0")


if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    input_data = preprocess_image(frame)

    outputs = session.run(output_names, {input_name: input_data})

    detections, labels = outputs
    print("Detections:", detections)
    print("Labels:", labels)

    for detection in detections[0]:
        x1, y1, x2, y2, confidence = detection[:5]
        if confidence > 0.6:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

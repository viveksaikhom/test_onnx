import onnxruntime as ort
import cv2
import numpy as np
from PIL import Image

model_path = "../../model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx"
camera_input = "/dev/video0"

session = ort.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

def preprocess_image(image, input_size=(416, 416)):
    image = Image.fromarray(image).convert("RGB")
    image = image.resize(input_size)
    image_data = np.array(image, dtype=np.float32) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    return image_data

cap = cv2.VideoCapture(camera_input)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    input_data = preprocess_image(frame)

    outputs = session.run(output_names, {input_name: input_data})

    detections, labels = outputs
    for detection in detections[0]:
        x1, y1, x2, y2 = detection[:4]
        confidence = detection[4]
        if confidence > 0.6:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

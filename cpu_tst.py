import cv2
import onnxruntime
import numpy as np
import time

model_path = "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx"
artifacts_folder = "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/artifacts"
class_names = ['with helmet', 'without_helmet']

ort_session = onnxruntime.InferenceSession(
    model_path,
    providers=[
        {"provider": "CPUExecutionProvider"}
    ]
)

print("Starting...")
input_name = ort_session.get_inputs()[0].name
output_names = [ort_session.get_outputs()[0].name]

# Load the image
image_path = "/opt/edgeai-test-data/images/0000.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not read the image from {image_path}.")
    exit()

# Preprocess the image
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (320, 320))  # Resize to the expected input size
img = img.astype('float32') / 255.0
img = img.transpose(2, 0, 1)  # Change shape to (C, H, W)
img = img[None, ...]  # Add batch dimension

start_time = time.time()
outputs = ort_session.run(output_names, {input_name: img})
end_time = time.time()
inference_time = end_time - start_time

boxes = outputs[0][0][:, :4]
scores = outputs[0][0][:, 4]
classes = outputs[0][0][:, 5]

for i in range(len(boxes)):
    if scores[i] > 0.5:
        box = boxes[i]
        x1, y1, x2, y2 = map(int, box)
        label = class_names[int(classes[i])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"Detected: {label} (Inference time: {inference_time:.4f}s)")

cv2.imshow('YOLOX Nano Lite Inference', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

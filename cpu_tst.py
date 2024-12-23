import cv2
import numpy as np
import onnxruntime as ort

tidl_options = {
    "provider": "TIDLExecutionProvider",
    "options": {
        "artifacts_folder": "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/artifacts",
        "enable_layer_grouping": True,
    }
}

ort_session = ort.InferenceSession(
    "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx",
    providers=[tidl_options]
)

input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
output_names = [output.name for output in ort_session.get_outputs()]

image_path = "/opt/edgeai-test-data/images/0000.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

input_height, input_width = input_shape[2], input_shape[3]
image = cv2.resize(image, (input_width, input_height))

image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

outputs = ort_session.run(output_names, {input_name: image})
print(outputs)

import numpy as np
import onnxruntime as ort

tidl_options = {
    "provider": "TIDLExecutionProvider",
    "options": {
        "artifacts_folder": "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/artifacts",
        "enable_layer_grouping": True,
    }
}

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession(
    "/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx",
    providers=[tidl_options]
)

input_name = ort_session.get_inputs()[0].name
output_names = [output.name for output in ort_session.get_outputs()]

input_shape = ort_session.get_inputs()[0].shape
input_data = np.random.rand(*input_shape).astype(np.float32)

outputs = ort_session.run(output_names, {input_name: input_data})
print(outputs)

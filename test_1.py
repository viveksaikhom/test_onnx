import onnxruntime_tidl as ort_tidl
import numpy as np

model_path = '/opt/model_zoo/20241208-173443_yolox_nano_lite_onnxrt_AM62A/model/model.onnx'

session = ort_tidl.InferenceSession(model_path)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input shape: {input_shape}")

dummy_input = np.random.randn(*input_shape).astype(np.float32)

output = session.run(None, {input_name: dummy_input})
print("Output:", output)

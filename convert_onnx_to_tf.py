import onnx
from onnx_tf.backend import prepare

model_path="peekingduck_weights/yolox/yolox-tiny.onnx"
output_path="yolox_tiny_tf"

onnx_model = onnx.load(model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(output_path)


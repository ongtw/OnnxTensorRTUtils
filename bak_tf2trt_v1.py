#from helper import ModelOptimizer
import tensorrt as trt
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert
from time import perf_counter

#
# dotw: 2021-01-12
# - FPS increased greatly
# - accuracy untested
# - model file size larger
# - very slow startup time
# - lots of TF/TRT warnings
#

print(f"tensorflow version={tf.__version__}")
print(f"tensorrt version={trt.__version__}")

PRECISION = "FP16"
GPU_RAM_4G = 4000000000
GPU_RAM_6G = 6000000000
GPU_RAM_8G = 8000000000
MPL = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/multipose_lightning"
SPL = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/singlepose_lightning"
SPT = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/singlepose_thunder"

model_dir = SPL
model_out_dir = model_dir + "_fp16"

# dotw: uses helper but error, helper not found...
#opt_model = ModelOptimizer(model_dir)
#model_fp16 = opt_model.convert(model_dir + "_fp16", precision=PRECISION)

# dotw: error, create_inference_graph() missing 2 required positional arguments:
#               'input_graph_def' and 'outputs'
#trt_convert.create_inference_graph(
#    input_saved_model_dir = model_dir,
#    output_saved_model_dir = model_out_dir
#)

conv_parms = trt_convert.TrtConversionParams(
    precision_mode = trt_convert.TrtPrecisionMode.FP16,
    max_workspace_size_bytes = GPU_RAM_4G,
)
converter = trt_convert.TrtGraphConverterV2(
    input_saved_model_dir = model_dir,
    conversion_params = conv_parms
)

print(f"generating {model_out_dir}")
print("converting original model...")
st0 = perf_counter()
converter.convert()
#converter.build(input_fn = self.my_input_fn)
et0 = perf_counter()
print(f"conversion time = {et0 - st0:.2f} sec")

print("saving generated model...")
st1 = perf_counter()
converter.save(model_out_dir)
et1 = perf_counter()
print(f"save time = {et1 - st1:.2f} sec")

print(f"Total conversion time = {et1 - st0:.2f} sec")


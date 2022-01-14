#from helper import ModelOptimizer
import numpy as np
import tensorrt as trt
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert
from time import perf_counter

print("TF to TRT Converter v2")
print(f"tensorflow version={tf.__version__}")
print(f"tensorrt version={trt.__version__}")

PRECISION = "FP16"
GPU_RAM_2G = 2000000000
GPU_RAM_4G = 4000000000
GPU_RAM_6G = 6000000000
GPU_RAM_8G = 8000000000
MPL = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/multipose_lightning"
SPL = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/singlepose_lightning"
SPT = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/singlepose_thunder"

DIMENSIONS = {
    "SPL": (1, 192, 192, 3),
    "SPT": (1, 256, 256, 3),
    "MPL": (1, 256, 256, 3),
}

model_dir = SPL
model_out_dir = model_dir + "_fp16"

def my_input_fn_zz():
    # Generation function that yields input data used to build TRT engines,
    # to reduce runtime delay caused by building the engines during inference.
    # Must return data with same input shape as that during runtime.
    dims = DIMENSIONS["SPL"]
    inp1 = np.random.normal(size=dims).astype(np.int32)
#    inp2 = np.random.normal(size=(1, 192, 192, 3)).astype(np.float32)
#    inp3 = np.random.normal(size=(1, 192, 192, 3)).astype(np.float32)
    yield (inp1,)
#    input_shape = (1, 192, 192, 3)
#    batch_input = np.zeros(input_shape, dtype=np.int32)
#    batch_input_tf = tf.constant(batch_input)
#    yield (batch_input_tf,)


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
print("building engines...")
converter.build(input_fn = my_input_fn_zz)
et0 = perf_counter()
print(f"conversion time = {et0 - st0:.2f} sec")

print("saving generated model...")
st1= perf_counter()
converter.save(model_out_dir)
et1= perf_counter()
print(f"save time = {et1 - st1:.2f} sec")

print(f"Total conversion time = {et1 - st0:.2f} sec")


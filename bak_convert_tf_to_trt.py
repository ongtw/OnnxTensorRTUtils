#from helper import ModelOptimizer
#import tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert
from time import perf_counter

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
    max_batch_size = 1
)
converter = trt_convert.TrtGraphConverterV2(
    input_saved_model_dir = model_dir,
    conversion_params = conv_parms
)

print("converting model...")
st = perf_counter()
converter.convert()
#converter.build(input_fn = self.my_input_fn)
et = perf_counter()
print(f"conversion time = {et - st:.2f} sec")

print("saving  model...")
st = perf_counter()
converter.save(model_out_dir)
et = perf_counter()
print(f"model saving time = {et - st:.2f} sec")


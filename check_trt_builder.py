import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
attr_list = [
    "platform_has_fast_fp16",
    "platform_has_fast_int8",
    "platform_has_tf32",
    "max_DLA_batch_size",
    "num_DLA_cores",
    "max_batch_size",
]

print(f"Checking Builder Attributes for TensorRT {trt.__version__}:")
for i, attr in enumerate(attr_list):
    builder_fn = eval(f"builder.{attr}")
    print(f"{i:2}  {attr:24}  {builder_fn}")

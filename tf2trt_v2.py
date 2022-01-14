import numpy as np
#import tensorrt as trt
#import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert
from time import perf_counter

print("TF to TRT Converter v2")
#print(f"tensorflow version={tf.__version__}")
#print(f"tensorrt version={trt.__version__}")

#
# Generator functions
# Since cannot pass params into generators (Python Error: 'generator' object is not callable),
# construct different generator types
#
def my_input_gen_192():
    inp = np.zeros((1, 192, 192, 3)).astype(np.int32)
    yield (inp,)

def my_input_gen_256():
    inp = np.zeros((1, 256, 256, 3)).astype(np.int32)
    yield (inp,)

PRECISION = "FP16"
GPU_RAM_2G = 2000000000
GPU_RAM_4G = 4000000000
GPU_RAM_6G = 6000000000
GPU_RAM_8G = 8000000000
GPU_RAM = "4G"
MODEL_PRECISION = {
    "INT8": trt_convert.TrtPrecisionMode.INT8,
    "FP16": trt_convert.TrtPrecisionMode.FP16,
    "FP32": trt_convert.TrtPrecisionMode.FP32,
}
MODEL_RAM = {
    2: 2000000000,
    4: 4000000000,
    6: 6000000000,
    8: 8000000000,
}
# Main data structure to store model code and model info
MODEL_MAP = {
    "SPL": {
        "dir": "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/singlepose_lightning",
        "gen": my_input_gen_192,
    },
    "SPT": {
        "dir": "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/singlepose_thunder",
        "gen": my_input_gen_256,
    },
    "MPL": {
        "dir": "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/movenet/multipose_lightning",
        "gen": my_input_gen_256,
    },
}


#
# Model attribute queries
#
def get_model_dir(model_code: str) -> str:
    the_dir = MODEL_MAP[model_code]["dir"]
    return the_dir

def get_model_gen(model_code: str):
    the_gen = MODEL_MAP[model_code]["gen"]
    return the_gen

def get_model_save_filepath(model_code: str, prec: str, gpu_ram: int) -> str:
    the_dir = get_model_dir(model_code)
    the_path = f"{the_dir}_v2_{prec.lower()}_{gpu_ram}GB"
    return the_path


#
# Main program
#
def convert_model(model_code: str, prec: str, gpu_ram: int):
    model_dir = get_model_dir(model_code)
    model_save_path = get_model_save_filepath(model_code, prec, gpu_ram)
    print(f"generating {model_save_path}")
    # setup converter params
    conv_parms = trt_convert.TrtConversionParams(
        precision_mode = MODEL_PRECISION[prec],
        max_workspace_size_bytes = MODEL_RAM[gpu_ram],
    )
    converter = trt_convert.TrtGraphConverterV2(
        input_saved_model_dir = get_model_dir(model_code),
        conversion_params = conv_parms
    )
    # convert original base model to TF-TRT model
    print("converting original model...")
    pc1 = perf_counter()
    converter.convert()
    conv_dur = perf_counter() - pc1
    print(f"conversion time = {conv_dur:.2f} sec")
    # build runtime engine
    print("building engines...")
    pc2 = perf_counter()
    converter.build(input_fn = get_model_gen(model_code))
    build_dur = perf_counter() - pc2
    print(f"build time = {build_dur:.2f} sec")
    # save model
    print("saving generated model...")
    pc3 = perf_counter()
    converter.save(model_save_path)
    save_dur = perf_counter() - pc3
    print(f"save time = {save_dur:.2f} sec")
    # print time stats
    total_dur = perf_counter() - pc1
    print(f"{model_save_path}:")
    print(f"Conversion time = {conv_dur:.2f} sec")
    print(f"Build time      = {build_dur:.2f} sec")
    print(f"Save time       = {save_dur:.2f} sec")
    print(f"Total time      = {total_dur:.2f} sec")


if __name__ == "__main__":
    for model_code in MODEL_MAP.keys():
        convert_model(model_code, "FP16", 4)



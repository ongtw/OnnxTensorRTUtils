import cv2
import numpy as np
import yaml
from pathlib import Path
from peekingduck.pipeline.nodes.model.yoloxv1.yolox_model import YOLOXModel
from time import perf_counter

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

print(f"TensorRT version={trt.__version__}")

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
img_path = "images/testing/t1.jpg"
cfg_path = "peekingduck/configs/model/yolox.yml"
# trt model created using:
# NB: "chw" == "channel, height, width"
# trtexec --onnx=yolox-tiny.onnx --saveEngine=yolox-tiny-3.trt --inputIOFormats=fp32:chw
#         --outputIOFormats=fp16:chw --fp16 --useCudaGraph
trt_model_path = "peekingduck_weights/yolox/yolox-tiny-3.trt"
trt_logger = trt.Logger(trt.Logger.WARNING)

img = cv2.imread(img_path)
# print(img.shape)
height, width = img.shape[:2]
print(f"Image width={width}, height={height}")

#
# Using pytorch->TRT YOLOX
#
trt.init_libnvinfer_plugins(None, "")

input_batch = np.array(
    np.repeat(np.expand_dims(np.array(img, dtype=np.float32), axis=0), 1, axis=0),
    np.float32,
)

with open(trt_model_path, "rb") as f:
    serialized_engine = f.read()

    print("loading TRT model...")
    st = perf_counter()
    runtime = trt.Runtime(trt_logger)
    et = perf_counter()
    print(f"TRT model load time = {et - st:.2f} sec")

    print("deserializing CUDA engine...")
    st = perf_counter()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    et = perf_counter()
    print(f"deserializing CUDA engine time = {et - st:.2f} sec")

    device_memory_required = engine.get_device_memory_size()
    print(f"device_memory_required={device_memory_required}")

    # What are `input_name` and `output_name`?
    # input_idx = engine[input_name]
    # output_idx = engine[output_name]

    print("creating execution context...")
    st = perf_counter()
    context = engine.create_execution_context()
    et = perf_counter()
    print(f"create exec context time = {et - st:.2f} sec")

    print("creating TRT pointers bindings...")
    output = np.empty([1, 1000], dtype=np.float16)
    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)
    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()

    def predict(batch):
        cuda.memcpy_htod_async(d_input, batch, stream)
        context.execute_async_v2(bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
        return output

    pred = predict(img)
    print("pred=", pred)

exit(0)


#
# Using PKD YOLOX
#
# with open(cfg_path) as cfg_file:
#    cfg = yaml.safe_load(cfg_file)
# path = Path(__file__)
# cfg["root"] = path.parent
#
# mod_yolox = YOLOXModel(cfg)
# bboxes, class_names, scores = mod_yolox.predict(img)
# print(bboxes)
# print(class_names)
# print(scores)

for i, bbox in enumerate(bboxes):
    class_name = class_names[i]
    score = scores[i]
    top_left = bbox[:2]
    bottom_right = bbox[2:]
    top_left = (int(top_left[0] * width), int(top_left[1] * height))
    bottom_right = (int(bottom_right[0] * width), int(bottom_right[1] * height))
    print(f"{i}: {class_name} {score:.2f} {top_left}, {bottom_right}")
    if score < 0.40:
        color = GREEN
    elif score < 0.80:
        color = BLUE
    else:
        color = RED
    color_int = tuple([int(x) for x in color])
    # print(f"color_int={color_int}")
    cv2.rectangle(img, top_left, bottom_right, color_int, 2)

cv2.imshow("image_win", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

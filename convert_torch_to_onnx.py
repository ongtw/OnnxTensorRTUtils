import numpy as np
import torch
from peekingduck.pipeline.nodes.model.yoloxv1.yolox_files.model import YOLOX

# technotes:
# YOLOX input  shape = (1, 3, 416, 416)
#       output shape = (N, 85)  Qn: what is N?
#                      where cols[0:79] are the 80 classes,
#                                [80:83] is the bbox, [84] is confidence

INPUT_SIZE = (416, 416)
YOLOX_DIR = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox"
MODEL_MAP = {
    "yolox-tiny": {
        "path": "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox/yolox-tiny.pth",
        "size": {"depth": 0.33, "width": 0.375},
    },
    "yolox-s": {
        "path": "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox/yolox-s.pth",
        "size": {"depth": 0.33, "width": 0.5},
    },
    "yolox-m": {
        "path": "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox/yolox-m.pth",
        "size": {"depth": 0.67, "width": 0.75},
    },
    "yolox-l": {
        "path": "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox/yolox-l.pth",
        "size": {"depth": 1.0, "width": 1.0},
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "yolox-tiny"


def get_model_path(model_code: str) -> str:
    the_path = MODEL_MAP[model_code]["path"]
    return the_path


def get_model_size(model_code: str) -> str:
    the_size = MODEL_MAP[model_code]["size"]
    return the_size


def get_model_save_path(model_code: str) -> str:
    the_path = f"{YOLOX_DIR}/{model_code}.onnx"
    return the_path


def convert_model(model_code: str):
    model_path = get_model_path(model_code)
    model_save_path = get_model_save_path(model_code)
    model_size = get_model_size(model_code)
    print(f"generating {model_save_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    model = YOLOX(80, model_size["depth"], model_size["width"])
    # model.half()
    model.eval()
    model.load_state_dict(ckpt["model"])
    # model = fuse_model(model) # what is this?

    batch_size = 1
    x = torch.randn(batch_size, 3, 416, 416, requires_grad=True)
    torch_out = model(x)

    torch.onnx.export(
        model,
        x,
        "yolox.onnx",
        verbose=True,
        export_params=True,
        opset_version=10,
        do_constant_folding=False,
        input_names=["images"],
        output_names=["pred_output"],
        #                      output_names=["classes", "bboxes"],
        #                      dynamic_axes={"input":{0:"batch_size"},
        #                                    "output":{0:"batch_size"}}
    )


if __name__ == "__main__":
    for model_code in MODEL_MAP.keys():
        convert_model(model_code)
        break

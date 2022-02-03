from typing import Any, Dict, List, Tuple
import cv2
import json
import numpy as np
import os
import torch
import torchvision

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# config from yolo.yml
NUM_CHANNELS = 3
NUM_CLASSES = 80
SCORE_THRESHOLD = 0.25
MAX_OUTPUT_SIZE_PER_CLASS = 50
MAX_TOTAL_SIZE = 50
INPUT_SIZE = 416
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
# global constants
RED, GREEN, BLUE = (0, 0, 255), (255, 0, 0), (0, 255, 0)
# global vars
the_device = "cpu"
class_names_path = (
    "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox/coco.names"
)
model_path = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox_tiny_tf"
img_path = "/home/aisg/src/ongtw/PeekingDuck/images/testing/t1.jpg"


def read_class_names(class_names_path: str):
    with open(class_names_path) as infile:
        class_names = [line.strip() for line in infile.readlines()]
    return class_names


def read_image(img_path: str):
    img = cv2.imread(img_path)
    return img


def get_last_2d(a: np.array) -> np.array:
    m, n = a.shape[-2:]
    a_2d = a.flat[: m * n].reshape(m, n)
    return a_2d


def xywh2xyxy(inputs: torch.Tensor) -> torch.Tensor:
    # converts [x, y, w, h] to [x1, y1, x2, y2] format
    outputs = torch.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] - inputs[:, 2] / 2
    outputs[:, 1] = inputs[:, 1] - inputs[:, 3] / 2
    outputs[:, 2] = inputs[:, 0] + inputs[:, 2] / 2
    outputs[:, 3] = inputs[:, 1] + inputs[:, 3] / 2
    return outputs


def xyxy2xyxyn(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    # converts [x1, y1, x2, y2] to normalised [x1, y1, x2, y2]
    outputs = np.empty_like(inputs)
    outputs[:, [0, 2]] = inputs[:, [0, 2]] / width
    outputs[:, [1, 3]] = inputs[:, [1, 3]] / height
    return outputs


def preprocess(image: np.ndarray) -> Tuple[np.ndarray, float]:
    # setup "self" vars
    input_size = (INPUT_SIZE, INPUT_SIZE)
    # Initialize canvas for padded image as gray
    padded_img = np.full(
        (input_size[0], input_size[1], NUM_CHANNELS), 114, dtype=np.uint8
    )
    scale = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    scaled_height = int(image.shape[0] * scale)
    scaled_width = int(image.shape[1] * scale)
    resized_img = cv2.resize(
        image,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[:scaled_height, :scaled_width] = resized_img

    # Rearrange from (H, W, C) to (C, H, W)
    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, scale


def postprocess(
    prediction: torch.Tensor,
    scale: float,
    image_shape: Tuple[int, int],
    class_names: List[str],
) -> Tuple[List[np.ndarray], List[str], List[float]]:
    print(f"scale={scale:.4f}, image_shape={image_shape}")

    # from PKD YOLOX detector.py
    prediction[:, :4] = xywh2xyxy(prediction[:, :4])
    # Get score and class with highest confidence
    pred_class = prediction[:, 5 : 5 + NUM_CLASSES]
    print(f"pred_class.shape={pred_class.shape}")
    class_score, class_pred = torch.max(
        prediction[:, 5 : 5 + NUM_CLASSES], 1, keepdim=True
    )
    print(f"class_score.shape={class_score.shape}, class_pred.shape={class_pred.shape}")
    # Filter by score_threshold
    print("score_threshold:", SCORE_THRESHOLD)
    conf_mask = (prediction[:, 4] * class_score.squeeze() >= SCORE_THRESHOLD).squeeze()
    print(f"conf_mask.shape={conf_mask.shape}")
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((prediction[:, :5], class_score, class_pred.float()), 1)
    print(f"detections.shape={detections.shape}")
    detections = detections[conf_mask]
    print(f"detections.shape after mask={detections.shape}")
    print(detections[:5, :])

    # torch.save(detections, "/tmp/detections.pt")
    # np.savetxt("/tmp/detections_tensor.txt", detections.numpy())
    # detections = torch.load("/tmp/detections_m1_mac.pt", map_location="cpu")
    # print(f"loaded detections.shape after mask={detections.shape}")

    # Early return if all are below score_threshold
    if not detections.size(0):
        return np.empty((0, 4)), np.empty(0), np.empty(0)

    # Class agnostic NMS
    print("iou_threshold:", IOU_THRESHOLD)
    nms_out_index = torchvision.ops.nms(
        detections[:, :4],
        detections[:, 4] * detections[:, 5],
        IOU_THRESHOLD,
    )
    output = detections[nms_out_index]
    print(f"output.shape={output.shape}")
    print(output)
    # torch.save(output, "/tmp/output_jetson.pt")

    # Filter by detect ids
    detect_ids = torch.Tensor(
        [
            0,
        ]
    )
    if detect_ids.size(0):
        output = output[torch.isin(output[:, 6], detect_ids)]
    output_np = output.cpu().detach().numpy()
    bboxes = xyxy2xyxyn(output_np[:, :4] / scale, *image_shape)
    scores = output_np[:, 4] * output_np[:, 5]
    classes = np.array([class_names[int(i)] for i in output_np[:, 6]])

    return bboxes, classes, scores


def show_image_with_bboxes(img, bboxes, scores, classes):
    height, width = img.shape[:2]
    # print(f"Image width={width}, height={height}")
    for i, bbox in enumerate(bboxes):
        class_name = classes[i]
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
        cv2.rectangle(img, top_left, bottom_right, color_int, 2)
    cv2.imshow("image_win", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    class_names = read_class_names(class_names_path)
    img = read_image(img_path)
    print(f"img.shape={img.shape}")
    img_rs = cv2.resize(img, dsize=(416, 416), interpolation=cv2.INTER_LINEAR).astype(
        np.uint8
    )
    print(f"img_rs.shape={img_rs.shape}")

    image_size = img.shape[:2]
    image, scale = preprocess(img)
    image = torch.from_numpy(image).unsqueeze(0).to(the_device)
    print(f"image.shape={image.shape}")
    data = json.dumps({"data": image.tolist()})
    data = np.array(json.loads(data)["data"]).astype("float32")
    print(f"data.shape={data.shape}")

    # todo: load TF model here
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(input_name, output_name)

    result = session.run([output_name], {input_name: data})
    # print(f"result type={type(result)}, len={len(result)}")
    # print(result)
    res_arr = np.array(result)
    print(f"res_arr.shape={res_arr.shape}")
    pred = get_last_2d(res_arr)  # raw predictions from model
    prediction = torch.from_numpy(pred)
    print(f"prediction.shape={prediction.shape}")
    # print(pred)
    # torch.save(prediction, "/tmp/prediction.pt")
    # np.savetxt("/tmp/detections_tensor.txt", detections.numpy())

    bboxes, classes, scores = postprocess(prediction, scale, image_size, class_names)

    print(bboxes)
    print(classes)
    print(scores)

    show_image_with_bboxes(img, bboxes, scores, classes)


if __name__ == "__main__":
    main()

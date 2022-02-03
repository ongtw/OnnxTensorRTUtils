import tensorflow as tf

model_dir = "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox_tiny_tf"
model_path = (
    "/home/aisg/src/ongtw/PeekingDuck/peekingduck_weights/yolox_tiny_tf/saved_model.pb"
)

model = tf.saved_model.load(model_dir)
print(model.summary())

from models import backbone_darknet
import tensorflow as tf
from utils import  *
import numpy as np
import pandas as pd

yolov1 = tf.keras.models.load_model("yolov1.keras", compile=False, safe_mode=False)

image_path, boxes, class_identity = parse_xml("data/annotations/Cats_Test0.xml")
image, boxes, class_identity = prepare_data(image_path=image_path, boxes=boxes, class_identity=class_identity)
image, label = convert_onepart(image, boxes, class_identity)
label = np.expand_dims(label,0)
image = np.expand_dims(image,0)

result = yolov1.predict(image)
xywh, c, classes = outputyolo(result)

index = np.where(c > 0.7)[0]
index = index.reshape((-1,1))[0]


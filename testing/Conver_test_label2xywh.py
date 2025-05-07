import numpy as np

from utils import *
from losses import YOLOLoss
from models import build_yolo_v1_vgg16
from metrix import YoloV1Metric

BATCH_SIZE = 1
# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join("../data/annotations/", file_name)
        for file_name in os.listdir("../data/annotations/")
        if file_name.endswith(".xml")
    ]
)[:1]

labels = []
images = []
for xml in xml_files:
    image_path, boxes, class_identity = parse_xml(xml)
    image, boxes, class_identity = prepare_data(image_path=image_path, boxes=boxes, class_identity=class_identity)
    image, label = convert_onepart(image, boxes, class_identity)
    labels.append(label)
    images.append(image)

labels = np.array(labels)
zeros = np.zeros(shape=(1,7,7,5))
labels = np.concatenate([labels, zeros], axis=-1)
labels = tf.convert_to_tensor(labels, dtype=tf.float32)

boxes= delta2org_v1(labels[...,0:4])
boxes = np.reshape(boxes,(-1,4))
boxes =box_center_to_corner(boxes)
boxes = np.array(boxes)
plot_anchors_xyxy(images[0],boxes,0 )



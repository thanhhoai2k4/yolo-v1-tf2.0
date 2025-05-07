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
)[:2]

images = []
labels = []
for xml in xml_files:
    image_path, boxes, class_identity = parse_xml(xml)
    image, boxes, class_identity = prepare_data(image_path=image_path, boxes=boxes, class_identity=class_identity)
    image, label = convert_onepart(image, boxes, class_identity)
    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# load backbone
model_yolo_v1 = build_yolo_v1_vgg16(input_shape=(448,448,3),grid_size=7,num_boxes=2,num_classes=2, train_backbone=False)
# compile model
optimizer = tf.keras.optimizers.SGD(0.0001, momentum=0.9, clipnorm=10.0)
model_yolo_v1.compile(optimizer=optimizer, loss=YOLOLoss(), run_eagerly=False)
model_yolo_v1.load_weights("my_model.weights.h5")
model_yolo_v1.fit(images, labels,epochs=100)

# save model
model_yolo_v1.save_weights(filepath="../my_model.weights.h5")
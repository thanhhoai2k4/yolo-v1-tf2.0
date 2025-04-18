from models import backbone_darknet
from utils import *

# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join("data/annotations/", file_name)
        for file_name in os.listdir("data/annotations/")
        if file_name.endswith(".xml")
    ]
)


def get_aLL_data(xmls: list[str]):
    images, labels = [], []
    for xml in xmls:
        image_path, boxes, class_identity = parse_xml(xml)
        image, boxes, class_identity = prepare_data(image_path=image_path, boxes=boxes, class_identity=class_identity)
        image, label = convert_onepart(image, boxes, class_identity)
        images.append(image)
        labels.append(label)
    return images, labels

# load dữ liệu lên random of memory
images, labels = get_aLL_data(xml_files)
images = tf.convert_to_tensor(images, dtype=tf.float32)
labels = tf.convert_to_tensor(labels, dtype=tf.float32)

# set learning rate
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler) # call batch để setting learning rate
# load backbone
model_yolo_v1 = backbone_darknet(input_shape=(448,448,3))
# try:
#     model_yolo_v1.load_weights("my_model.weights.h5")
# except:
#     pass

# fit model
history = model_yolo_v1.fit(images, labels, epochs=50,verbose=1)

# save model
model_yolo_v1.save_weights(filepath="my_model.weights.h5")
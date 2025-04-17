from models import backbone_darknet
from utils import *

BATCH_SIZE = 20
# Get all XML file paths in path_annot and sort them
xml_files = sorted(
    [
        os.path.join("data/annotations/", file_name)
        for file_name in os.listdir("data/annotations/")
        if file_name.endswith(".xml")
    ]
)

# chia ra thành 3 tập train validation và test
xml_files_train = xml_files[:int(len(xml_files)* 0.8)]
xml_files_validation = xml_files[int(len(xml_files)* 0.8):int(len(xml_files)* 0.9)]
xml_files_test = xml_files[int(len(xml_files)* 0.9):]


def tf_data(xmls: list[str]):
    for xml in xmls:
        image_path, boxes, class_identity = parse_xml(xml)
        image, boxes, class_identity = prepare_data(image_path=image_path, boxes=boxes, class_identity=class_identity)
        image, label = convert_onepart(image, boxes, class_identity)
        yield image, label

# load dataset
dataset_train = tf.data.Dataset.from_generator(
    lambda: tf_data(xml_files_train),
    output_signature=(
        tf.TensorSpec(shape=(448,448,3), dtype=tf.float32),  # Đầu vào X_batch image
        tf.TensorSpec(shape=(7,7,7), dtype=tf.float32))
)

dataset_val = tf.data.Dataset.from_generator(
    lambda: tf_data(xml_files_validation),
    output_signature=(
        tf.TensorSpec(shape=(448,448,3), dtype=tf.float32),  # Đầu vào X_batch image
        tf.TensorSpec(shape=(7,7,7), dtype=tf.float32))
)

# data for training
dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)
dataset_train = dataset_train.repeat()

# data for validation
dataset_val = dataset_val.batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)


# set learning rate
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler) # call batch để setting learning rate

# load backbone
model_yolo_v1 = backbone_darknet(input_shape=(448,448,3))
try:
    model_yolo_v1.load_weights("my_model.weights.h5")
except:
    pass

# fit model
history = model_yolo_v1.fit(dataset_train, epochs=50,verbose=1,steps_per_epoch=len(xml_files_train) // BATCH_SIZE, callbacks=lr_callback, validation_data=dataset_val ,validation_steps=len(xml_files_validation)//BATCH_SIZE)

# save model
model_yolo_v1.save_weights(filepath="my_model.weights.h5")
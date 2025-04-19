from pickletools import optimize

from models import  pre_traing
import tensorflow as tf


# Cac sieu tham so
PATH = "data-classfication/training"
BATCH_SIZE = 4
IMG_SIZE = (448,448)

# load model
model = pre_traing(input_shape=(448,448,3))

# tải dữ liệu từ disk
DataTraining, DataValidation = tf.keras.preprocessing.image_dataset_from_directory(
    directory = PATH,
    labels = "inferred", # inferred = return tf.data.dataset
    label_mode='int',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed = 1000,
    validation_split = 0.2,
    subset = "both"
)

opt = tf.keras.optimizers.SGD (lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(DataTraining)
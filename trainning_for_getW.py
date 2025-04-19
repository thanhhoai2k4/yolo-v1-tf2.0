from pickletools import optimize

from models import  pre_traing
import tensorflow as tf


# Cac sieu tham so
PATH = "train"
BATCH_SIZE = 64
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

IMG_SIZE = 180

rescale = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255)
])

DataTraining = DataTraining.map(
  lambda x, y: (rescale(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
DataTraining = DataTraining.shuffle(1000)
DataTraining = DataTraining.prefetch(tf.data.AUTOTUNE)


DataValidation = DataValidation.map(
  lambda x, y: (rescale(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
DataValidation = DataValidation.shuffle(1000)
DataValidation = DataValidation.prefetch(tf.data.AUTOTUNE)



opt = tf.keras.optimizers.SGD (learning_rate=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(DataTraining, epochs=50 , validation_data=DataValidation)
model.save_weights(filepath="my_model_getW.weights.h5")
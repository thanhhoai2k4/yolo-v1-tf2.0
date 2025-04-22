import tensorflow as tf
from keras.src.metrics.accuracy_metrics import accuracy

from losses.loss import YOLOLoss
from metrix.metrix import *

leak_rl = tf.keras.layers.LeakyReLU(0.1)


def backbone_darknet(input_shape=(448, 448, 3)):
    # backbone darknet for model.
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(128, (7, 7), strides=2,kernel_initializer='he_normal', padding='same', use_bias=False ,activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',kernel_initializer='he_normal',use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (1, 1),kernel_initializer='he_normal',use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3),kernel_initializer='he_normal', padding='same', use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (1, 1),kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3),kernel_initializer='he_normal', padding='same', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 4 (4x convolutional layers)
    for _ in range(4):
        x = tf.keras.layers.Conv2D(128, (1, 1),kernel_initializer='he_normal',use_bias=False, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3),kernel_initializer='he_normal', padding='same',use_bias=False, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2D(128, (1, 1),kernel_initializer='he_normal', use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',kernel_initializer='he_normal', use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 5 (2x convolutional layers)
    for _ in range(2):
        x = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal',activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',kernel_initializer='he_normal', use_bias=False,activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)


    x = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (3,3), strides=2, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    # head
    x = tf.keras.layers.Flatten()(x) # flatten layers
    x = tf.keras.layers.Dense(588, activation="linear")(x)
    x = tf.keras.layers.Reshape(target_shape=(7,7,12))(x)
    # Output Feature Map (7x7x12 cho YOLO head)
    outputs = x

    model_yolo_v1 = tf.keras.Model(inputs=inputs, outputs=outputs, name='yolo_v1_model') # model cần trả về

    yolo_map_metric = YoloV1Metric(iou_threshold=0.5)

    #compile model
    optimizer = tf.keras.optimizers.SGD(0.001, momentum=0.9, clipnorm=1.0)
    model_yolo_v1.compile(
        optimizer=optimizer, loss=YOLOLoss(), metrics=[yolo_map_metric]
    )

    return model_yolo_v1

def pre_traing(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, kernel_initializer='he_normal', padding='same',
                               activation=leak_rl)(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', kernel_initializer='he_normal', activation=leak_rl)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (1, 1), kernel_initializer='he_normal', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same', activation=leak_rl)(x)
    x = tf.keras.layers.Conv2D(256, (1, 1), kernel_initializer='he_normal', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation=leak_rl)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 4 (4x convolutional layers)
    for _ in range(4):
        x = tf.keras.layers.Conv2D(256, (1, 1), kernel_initializer='he_normal', activation=leak_rl)(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same', activation=leak_rl)(x)

    x = tf.keras.layers.Conv2D(512, (1, 1), kernel_initializer='he_normal', activation=leak_rl)(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal', activation=leak_rl)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 5 (2x convolutional layers)
    for _ in range(2):
        x = tf.keras.layers.Conv2D(512, (1, 1),kernel_initializer='he_normal', activation=leak_rl)(x)
        x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal', activation=leak_rl)(x)

    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=1, padding="same", kernel_initializer='he_normal',
                               activation=leak_rl)(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=2, padding="same", kernel_initializer='he_normal',
                               activation=leak_rl)(x)

    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=1, padding="same", kernel_initializer='he_normal',
                               activation=leak_rl)(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=1, padding="same", kernel_initializer='he_normal',
                               activation=leak_rl)(x)

    x = tf.keras.layers.Flatten()(x) # flatten layers
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(400, activation=None)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name='backbone_pred')
    return model
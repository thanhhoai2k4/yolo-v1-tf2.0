import tensorflow as tf
from losses.loss import YOLOLoss
from metrix.metrix import *

def backbone_darknet(input_shape=(448, 448, 3)):
    # backbone darknet for model.
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, kernel_regularizer=tf.keras.regularizers.L1(0.0005), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.L1(0.0005), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (1, 1), activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, (1, 1), activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 4 (4x convolutional layers)
    for _ in range(4):
        x = tf.keras.layers.Conv2D(256, (1, 1), activation='relu')(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)

    x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu')(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 5 (2x convolutional layers)
    for _ in range(2):
        x = tf.keras.layers.Conv2D(512, (1, 1), activation='relu')(x)
        x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)

    x = tf.keras.layers.Conv2D(1024, (3,3), strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(1024, (3,3), strides=2, padding="same", activation="relu")(x)

    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=1, padding="same", activation="relu")(x)

    # head
    x = tf.keras.layers.Flatten()(x) # flatten layers
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(588, activation="linear")(x)
    x = tf.keras.layers.Reshape(target_shape=(7,7,12))(x)
    # Output Feature Map (7x7x12 cho YOLO head)
    outputs = x

    model_yolo_v1 = tf.keras.Model(inputs, outputs, name='yolo_v1_model') # model cần trả về

    yolo_map_metric = YoloV1Metric(iou_threshold=0.5)

    #compile model
    optimizer = tf.keras.optimizers.SGD(0.001, momentum=0.9)
    model_yolo_v1.compile(
        optimizer=optimizer, loss=YOLOLoss(), metrics=[yolo_map_metric]
    )

    return model_yolo_v1
import tensorflow as tf

leak_rl = tf.keras.layers.LeakyReLU(0.1)


def backbone_darknet(input_shape=(448, 448, 3)):
    """
        Lớp cơ sở cho yolo v1
        inputs
    """
    inputs = tf.keras.Input(shape=input_shape)
    # Block 1
    x = tf.keras.layers.Conv2D(31, (7, 7), strides=2,kernel_initializer='he_normal', padding='same', use_bias=False ,activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 2
    x = tf.keras.layers.Conv2D(192, (3, 3), padding='same',kernel_initializer='he_normal',use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, (1, 1),kernel_initializer='he_normal',use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(256, (3, 3),kernel_initializer='he_normal', padding='same', use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(256, (1, 1),kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(512, (3, 3),kernel_initializer='he_normal', padding='same', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 4 (4x convolutional layers)
    for _ in range(4):
        x = tf.keras.layers.Conv2D(256, (1, 1),kernel_initializer='he_normal',use_bias=False, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        x = tf.keras.layers.Conv2D(512, (3, 3),kernel_initializer='he_normal', padding='same',use_bias=False, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2D(512, (1, 1),kernel_initializer='he_normal', use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same',kernel_initializer='he_normal', use_bias=False, activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 5 (2x convolutional layers)
    for _ in range(2):
        x = tf.keras.layers.Conv2D(512, (1, 1), kernel_initializer='he_normal',activation=None, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)
        x = tf.keras.layers.Conv2D(1024, (3, 3), padding='same',kernel_initializer='he_normal', use_bias=False,activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.1)(x)


    x = tf.keras.layers.Conv2D(1024, (3,3), strides=1, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(1024, (3,3), strides=2, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=1, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=1, padding="same",kernel_initializer='he_normal', use_bias=False,activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)

    model_yolo_v1 = tf.keras.Model(inputs=inputs, outputs=x, name='yolo_v1_model') # model cần trả về
    return model_yolo_v1

def yolo(input_shape=(448,448,3)):
    """
        Lớp dense cho dự đoán dầu ra.
        inputs
    """
    # Đầu vào của model
    model_yolo_v1 = backbone_darknet(input_shape)
    x = tf.keras.layers.Flatten()(model_yolo_v1.output)  # flatten layers
    x = tf.keras.layers.Dense(1000, activation=leak_rl)(x)
    x = tf.keras.layers.Dense(588, activation="linear")(x)
    x = tf.keras.layers.Reshape(target_shape=(7, 7, 12))(x)
    yolo_head = tf.keras.Model(inputs=model_yolo_v1.input, outputs=x, name='yolo_head')
    return yolo_head
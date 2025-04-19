from models import  pre_traing
import tensorflow as tf


# Các siêu tham số
PATH = "train"
BATCH_SIZE = 40
IMG_SIZE = (448,448)

# Tải mẩu
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

rescale = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255)
])


# DATA TRAINING -----------------------------------------------------------------
DataTraining = DataTraining.map(
  lambda x, y: (rescale(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
DataTraining = DataTraining.prefetch(tf.data.AUTOTUNE)

# DATA VALIDATION ----------------------------------------------------------------
DataValidation = DataValidation.map(
  lambda x, y: (rescale(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
DataValidation = DataValidation.prefetch(tf.data.AUTOTUNE)



checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_model_getW.keras',# Tên file để lưu
    monitor='val_accuracy',             # Theo dõi metric nào (ở đây là val_accuracy)
    save_best_only=True,                # Chỉ lưu nếu tốt hơn model trước đó
    mode='max',                         # mode='max' vì accuracy càng cao càng tốt
    verbose=1  ,                        # In log khi có model được lưu
)

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(DataTraining, epochs=50 , validation_data=DataValidation, callbacks=[checkpoint_callback])
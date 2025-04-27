from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model


def backboneVGG16(input_shape=(448,448,3),train_backbone=False):
    backbone = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    for layer in backbone.layers:
        layer.trainable = train_backbone
    return backbone
def build_yolo_v1_vgg16(
    input_shape=(448, 448, 3),
    grid_size=7,
    num_boxes=2,
    num_classes=2,
    train_backbone=False
):
    """
    Xây dựng kiến trúc YOLO v1 kế thừa từ VGG16.

    Tham số:
      - input_shape: tuple, kích thước đầu vào của ảnh.
      - grid_size: S, kích thước lưới (S x S).
      - num_boxes: B, số bounding boxes mỗi ô.
      - num_classes: C, số lớp cần dự đoán.
      - train_backbone: bool, có cho phép fine-tune VGG16 hay không.

    Trả về:
      - tf.keras.Model: model YOLO v1.
    """

    # 1. Backbone: VGG16 không bao gồm fully-connected layers
    backbone = backboneVGG16(input_shape=input_shape, train_backbone=train_backbone)

    x = backbone.output  # Output shape ví dụ (None, 14,14,512) với input 448x448

    # 2. Thêm các convolutional layers theo YOLO v1 (tùy biến)
    # Conv 1
    x = layers.Conv2D(1024, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # Conv 2 với stride 2 để giảm độ phân giải
    x = layers.Conv2D(1024, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    # 3. Flatten và FC layers như YOLO v1
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)

    # 4. Output layer dự đoán S x S x (B*5 + C)
    output_units = grid_size * grid_size * (num_boxes * 5 + num_classes)
    x = layers.Dense(output_units, activation='linear')(x)
    output = layers.Reshape((grid_size, grid_size, num_boxes * 5 + num_classes))(x)

    # 5. Build model
    model = Model(inputs=backbone.input, outputs=output, name='YOLOv1_VGG16')
    return model
from utils import  *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.backbone import *


def plot_detections(image, boxes, confidences, classes, conf_threshold=0.5):
    """
    Hiển thị ảnh và vẽ bounding box trên đó đối với các dự đoán có confidence > conf_threshold.

    Parameters:
      image: np.array, ảnh đầu vào (có thể là định dạng HxWx3).
      boxes: np.array với shape (N, 4), các box có định dạng [x_center, y_center, width, height].
      confidences: np.array với shape (N, 1), giá trị confidence cho từng box.
      classes: np.array, chứa thông tin lớp (ở đây có shape (1, 7, 7, 2), có thể dùng để xác định nhãn lớp).
      conf_threshold: Ngưỡng confidence để lọc box (mặc định 0.7).

    Lưu ý: Nếu muốn sử dụng thông tin từ lớp (classes) để hiển thị nhãn, bạn cần logic mapping giữa box và grid cell.
    """

    # Nếu confidences có shape (N,1) chuyển về mảng 1 chiều
    confidences = confidences.squeeze(-1)

    # Tạo figure cho matplotlib
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    plt.imshow(image)

    # Duyệt qua từng box
    for i, box in enumerate(boxes):
        conf = confidences[i]
        if conf > conf_threshold:
            # Với định dạng (x_center, y_center, width, height)
            x_center, y_center, w, h = box

            # Chuyển sang tọa độ góc trên bên trái (x_min, y_min)
            x_min = x_center - w / 2
            y_min = y_center - h / 2

            # Tạo hình chữ nhật với viền màu đỏ
            rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Hiển thị confidence lên trên box
            ax.text(x_min, y_min, f"{conf:.2f}", color="red", fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5))

    ax.set_title("Detected Boxes (confidence > {:.1f})".format(conf_threshold))
    plt.axis("off")
    plt.show()

# # tải model từ local
yolov1 = backbone_darknet((448,448,3))
yolov1.load_weights("my_model.weights.h5")
#
# ones = np.ones((1,448,448,3), dtype=np.float32)
# zeros = np.zeros((1,448,448,3), dtype=np.float32)
#
# o = yolov1.predict(ones)
# z = yolov1.predict(zeros)
#
# box_1 ,c1 , classes1 = outputyolo(o,7)
#
# box_2 ,c2 , classes2 = outputyolo(z,7)
#
#
# c = np.concatenate([c1,c2], axis=-1)
# print(c)




image1,_,_ = loadimage("01.jpg", (448,448))
image_pre1 = np.expand_dims(image1,0)
image_pre1 = image_pre1/255.0
result1 = yolov1.predict(image_pre1)
box_1 ,c1 , classes1 = outputyolo(result1,7)
c1 = tf.sigmoid(c1)
box_1 = np.array(box_1)
c1 = np.array(c1)
classes1 = np.array(classes1)
plot_detections(image_pre1[0] , box_1 , c1 , classes1 , 0.7)
print(c1)




#
# image_path, boxes, class_identity = parse_xml("data/annotations/Cats_Test100.xml")
# image, boxes, class_identity = prepare_data(image_path=image_path, boxes=boxes, class_identity=class_identity)
# image, label = convert_onepart(image, boxes, class_identity)
# a = label[...,0:5]
# a1 =  label[..., 5:]
# zeros = np.zeros((7,7,5))
#
# kq = np.concatenate([a,a1,zeros], axis=-1)
# kq = np.expand_dims(kq,0)
# kq = tf.convert_to_tensor(kq, dtype=tf.float32)
# box_1 ,c1 , classes1 = outputyolo(kq,7)
# box_1 = np.array(box_1)
# c1 = np.array(c1)
# classes1 = np.array(classes1)
#
# plot_detections(image , box_1 , c1 , classes1 , 0.5)

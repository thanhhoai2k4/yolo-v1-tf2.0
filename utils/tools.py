from tkinter import Image
from xml.etree import  ElementTree as ET
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
# Tắt ký hiệu khoa học và đặt số chữ số thập phân
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.set_printoptions(suppress=True, precision=6)

class_ids = ["cat", "dog"]
class_mapping_decoder = dict(zip( range(len(class_ids)), class_ids ))
class_mapping_encoder = zip( class_ids,  range(len(class_ids)))
target_size = (448,448)
S = 7
B=2
C = 2


def parse_xml(path: str):
    """
    doc file xml

    :param path: duong dan den file xml
    :return: tra ve duong dan cua anh, box, classes
    """
    try:
        # di qua path de cho phep su ly
        tree = ET.parse(path)

        # truy cao vao root
        root = tree.getroot()

        # lay ten cua anh
        image_name = root.find("filename").text

        # duong dan den hinh anh
        image_path = os.path.join("data/images/", image_name) # return [0]

        boxes = [] # return [1]
        classes = []

        # chay qua tung object trong xml
        for obj in root.iter("object"):
            # lop cua doi tuong
            cls = obj.find("name").text
            classes.append(cls)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        class_identity = [
        list(class_mapping_decoder.keys())[list(class_mapping_decoder.values()).index(cls)]
        for cls in classes ]

        return image_path, np.array(boxes), np.array(class_identity)

    except :
        print("path khong doc duoc: {0}".format(path))
        return None

def loadimage(imagepath: str, target_size: tuple[int, int]):
    im = Image.open(imagepath, mode="r")
    width_org, height_org = im.size
    im = im.resize(size=target_size)
    im = im.convert("RGB")
    return im, width_org, height_org # image, width_org ,height_org


def prepare_data(image_path:str, boxes: np.array, class_identity:np.array):
    """

    :param image_path:
    :param boxes:
    :param class_identity:
    :return:
    """

    # load image
    image, width_org, height_org = loadimage(image_path, target_size=target_size)

    scaleX = width_org / image.size[0]
    scaleY = height_org/ image.size[1]

    image = np.array(image)/255.0 # chuyen image tu  image object sang  array

    # chuyen toa do ve dung cai ti le
    boxes[...,0] = boxes[...,0] / scaleX
    boxes[...,1] = boxes[...,1] / scaleY
    boxes[...,2] = boxes[...,2] / scaleX
    boxes[...,3] = boxes[...,3] / scaleY

    labels = np.zeros(shape=(len(class_identity), len(class_ids)), dtype=np.float32)
    labels[range(len(class_identity)),class_identity] = 1

    return image, boxes, labels

def box_corner_to_center(boxes):
    """
    Convert box corners[xmin, ymin, xmax, ymax] to center coordinates[x_center, y_center, width, height].
    boxes : shape[number box, 4]
    return : shape[number box, 4]
    """

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # center_x
    cx = (x1 + x2) / 2
    # center_y
    cy = (y1 + y2) / 2
    # width
    w = (x2 - x1)
    # height
    h = (y2 - y1)
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """
    apllay for  npArray
    Convert box center coordinates[x_center, y_center, width, height] to corners[xmin, ymin, xmax, ymax].
    boxes: shape[number box, 4]
    return: shape[number box, 4]
    """

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def convert_onepart(image, boxes, labels):
    """
    :param image:
    :param boxes:
    :param labels:
    :return:
    """
    label = np.zeros(shape=(S,S,5+C), dtype=np.float32)

    boxes = box_corner_to_center(boxes) # xywh
    # chuyen doi toa boxes do 0-1 sang

    x_center = boxes[...,0] / target_size[0]
    y_center = boxes[...,1] / target_size[1]
    width    = boxes[...,2] / target_size[0]
    height   = boxes[...,3] / target_size[1]

    i = np.asarray(x_center * S, dtype="int")
    j = np.asarray(y_center * S, dtype="int")


    x_rel = (x_center * S) - i
    y_rel = (y_center * S) - j

    confident = np.ones(shape=(len(boxes)),dtype=np.float32)
    cell = np.stack([x_rel, y_rel, width, height,confident], axis=-1)
    cell = np.concatenate([cell, labels],axis=1)
    label[j,i,:] = cell
    return image, label

def plot_anchors_xyxy(image:np.array, all_anchors: np.array, labels: np.array)->None:
    """
         show image with boxes and labels
    :param image: 448,448,3
    :param all_anchors: N,4
    :param labels: N,1
    :return:
    """
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    plt.imshow(image)
    for i in range(len(all_anchors)):
        x = all_anchors[i][0]
        y = all_anchors[i][1]
        w = all_anchors[i][2] - all_anchors[i][0]
        h = all_anchors[i][3] - all_anchors[i][1]

        rect = Rectangle((x,y),w,h, facecolor='none',edgecolor="red", lw=2 )
        ax.add_patch(rect)
        # plt.text(x+w/3,y, class_mapping_decoder[labels[i]], fontsize =12)
    plt.show()


def lr_scheduler(epoch):
    """
    :param epoch: số lượng training hiện hành để thực hiện điều chỉnh learning rate
    :return: learning rate . Lúc khởi đầu thì lớn càng về sau thì càng nhỏ

    congthuc: tuyen tinh.
        start_lr + (end_lr - start_lr) * (epoch / total_epochs)
        epoch (int): Epoch hiện tại.
        total_epochs (int): Tổng số epoch.
        start_lr (float): Learning rate ban đầu (mặc định là 0.0001).
        end_lr (float): Learning rate cuối cùng (mặc định là 0.01)
    """
    if (epoch < 20):
        # Công thức tuyến tính để tính learning rate
        return 0.00001 + (0.0001 - 0.00001) * (epoch / 20)
    elif epoch < 40:
        return 0.0001
    else:
        return 0.00007
def delta2org_v1(input, imagesize=(448.0, 448.0), S=7):
    """
    input: (..., 4) tensor, channels = [x_offset, y_offset, w_pred, h_pred]
    imagesize: (W_img, H_img)
    S: grid size
    predict_sqrt_wh: True nếu network output sqrt(w_norm), sqrt(h_norm)
    returns: (..., 4) tensor of absolute [x_center, y_center, width, height] in pixels
    """
    W_img, H_img = imagesize
    cell_size_x = W_img / S
    cell_size_y = H_img / S

    # grid indices
    cell_indices = tf.range(S, dtype=tf.float32)
    cx, cy = tf.meshgrid(cell_indices, cell_indices)  # cx: x index, cy: y index
    # expand dims to broadcast với batch và box dims
    cell_x = tf.expand_dims(cx, axis=-1)  # shape (S, S, 1)
    cell_y = tf.expand_dims(cy, axis=-1)

    # offsets within cell (0–1)
    x_offset = input[..., 0:1]
    y_offset = input[..., 1:2]
    w_norm = input[..., 2:3]
    h_norm = input[..., 3:4]

    # absolute centers
    x_center = (cell_x + x_offset) * cell_size_x
    y_center = (cell_y + y_offset) * cell_size_y

    # absolute sizes
    width  = w_norm * W_img
    height = h_norm * H_img

    return tf.concat([x_center, y_center, width, height], axis=-1)

def outputyolo(label, S=7):

    c1 = label[...,4:5] # 1,S,S,1
    c2 = label[..., 9:10] # 1,7,7,1

    confident = tf.concat([c1, c2], axis=-1)

    box1 = label[...,0:4]
    box2 = label[...,5:9]
    box_pred = tf.stack([box1, box2], axis=-2)

    index_max = tf.math.argmax(confident, axis=-1)

    box_max = tf.gather(box_pred, index_max, batch_dims=3, axis=-2)
    confident_max = tf.gather(confident, index_max, batch_dims=3,axis=-1)


    # bước chuyển giá trị tương đối sang tuyệt đối
    xywh = delta2org_v1(box_max) # 7,7,4

    xywh = tf.reshape(xywh, shape=(S*S,4))
    c = tf.reshape(confident_max, shape=(-1,1))
    classes = label[...,10:]

    return np.array(xywh), np.array(c), np.array(classes)


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
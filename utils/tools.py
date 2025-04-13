from tkinter import Image
from xml.etree import  ElementTree as ET
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tensorflow as tf
# Tắt ký hiệu khoa học và đặt số chữ số thập phân
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
    zeros = np.zeros(shape=(len(boxes)), dtype=np.float32)
    cell = np.stack([x_rel, y_rel, width, height,confident], axis=-1)
    cell = np.concatenate([cell, labels],axis=1)
    label[i,j,:] = cell
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



def deta2org(input, imagesize = (448.0,448.0), S=7):
    """

    :param input: 7 x 7 x 4
    :param imagesize: img size
    :param S: number grird
    :return: box x y w h
    """

    W_img, H_img = imagesize

    # Tạo chỉ số của cell trong lưới
    cell_indices = tf.range(S, dtype=tf.float32)
    cx, cy = tf.meshgrid(cell_indices, cell_indices)  # (S, S)

    # Nhập cx và cy vào một tensor có dạng (S, S, 2)
    cell_grid = tf.stack([cy, cx], axis=-1)  * 64.0 # (S, S, 2)
    xy = input[...,0:2] * 64 + cell_grid
    wh = input[...,2:4]*W_img
    kq = tf.concat([xy,wh], axis=-1) # xywh
    return kq

def lr_scheduler(epoch):
    """
    :param epoch: số lượng training hiện hành để thực hiện điều chỉnh learning rate
    :return: learning rate . Lúc khởi đầu thì lớn càng về sau thì càng nhỏ
    """
    if epoch < 5:
        return 0.0001
    elif epoch <10:
        return 0.001
    elif epoch < 30:
        return 0.001
    else:
        return 0.0001
def outputyolo(label, S=7):

    c1 = label[...,4:5] # confident cua box 1
    c2 = label[..., 9:10] # confident cua box 2

    confident = tf.concat([c1, c2], axis=-1)

    box1 = label[...,0:4]
    box2 = label[...,5:9]
    box_pred = tf.stack([box1, box2], axis=-2)

    index_max = tf.math.argmax(confident, axis=-1)

    box_max = tf.gather(box_pred, index_max, batch_dims=3, axis=-2)
    confident_max = tf.gather(confident, index_max, batch_dims=3,axis=-1)


    # bước chuyển giá trị tương đối sang tuyệt đối
    xywh = deta2org(box_max, (448.0, 448.0), 7) # 7,7,4

    xywh = tf.reshape(xywh, shape=(S*S,4))
    c = tf.reshape(confident_max, shape=(-1,1))
    classes = label[...,10:]

    return xywh, c, classes
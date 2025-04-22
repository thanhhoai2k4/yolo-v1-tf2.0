import tensorflow as tf

class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=2, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.S = grid_size  # Kích thước lưới (S x S), mặc định 7x7
        self.B = num_boxes  # Số hộp dự đoán mỗi ô lưới, mặc định 2
        self.C = num_classes  # Số lớp, mặc định 20 (theo VOC dataset)
        self.lambda_coord = lambda_coord  # Hệ số phạt cho tọa độ
        self.lambda_noobj = lambda_noobj  # Hệ số phạt cho ô không có đối tượng

    def call(self, y_true, y_pred):
        """

        :param y_true:  đầu vào là batch, 7,7,7: x y w h c p1 p2 .. pn : Ở đây số class là 2
        :param y_pred:  đầu vào là batch, 7,7,12: x y w h c x y w h cc p1 ... pn ở đây số class là 2
        :return: loss của tất cả batch. Không tính trung binh cộng. nếu muốn tính thế tf.math.reduce_sum = tf.math.reduce_mean
        """
        y_true = tf.reshape(y_true, shape=(-1,self.S, self.S, 5+self.C)) # chuyển về chiều mẫu
        y_pred = tf.reshape(y_pred, shape=(-1, self.S, self.S, 5*self.B + self.C)) # chuyển về chiều mẩu


        # giua box 1 va box true
        iou1 = self.calculate_iou(y_true[...,0:4], y_pred[..., 0:4]) # tinh iou của box 1 vs ground box
        # giua box 2 va box true
        iou2 = self.calculate_iou(y_true[...,0:4], y_pred[..., 5:9]) # tinh iou của box 1 vs ground box

        ious = tf.stack([iou1, iou2], axis=-1)
        index_max = tf.math.argmax(ious, axis=-1)
        iou_max = tf.gather(ious, index_max, batch_dims=3, axis=-1)

        box_pred = tf.stack([y_pred[...,0:5], y_pred[...,5:10]], axis=-2)
        box_max = tf.gather(box_pred, index_max, batch_dims=3, axis=-2)

        # xy
        loss_xy = tf.reduce_sum(
            y_true[...,4:5] * tf.square(
                y_true[...,0:2] - box_max[...,0:2]
            )
        )

        # wh
        w = y_true[...,4:5] * tf.math.square(tf.math.sqrt(tf.maximum(box_max[...,2:3], 1e-6)) - tf.math.sqrt(tf.maximum(y_true[...,2:3], 1e-6)))
        h = y_true[...,4:5] * tf.math.square(tf.math.sqrt(tf.maximum(box_max[...,3:4], 1e-6)) - tf.math.sqrt(tf.maximum(y_true[...,3:4], 1e-6)))
        loss_wh = tf.math.reduce_sum(w) + tf.math.reduce_sum(h)

        # confident
        confident_loss = tf.math.reduce_sum(y_true[...,4:5] * tf.math.square(tf.reshape(iou_max, (-1,self.S,self.S,1)) - box_max[...,4:5]))


        # classification loss
        # class_loss = tf.math.reduce_sum(y_true[...,4:5] * tf.math.square(y_true[...,5:5+self.C] - y_pred[...,-self.C:]))

        class_loss = tf.math.reduce_sum(tf.math.square(
            y_true[..., 4:5] * (y_true[...,5:5+self.C] - y_pred[...,-self.C:])
        ))

        # confiden no object
        no_confident_loss = tf.math.reduce_sum((1 - y_true[..., 4:5]) * tf.math.square(0 - y_pred[..., 4:5])) + \
                            tf.math.reduce_sum((1 - y_true[..., 4:5]) * tf.math.square(0 - y_pred[..., 9:10]))

        total_loss = self.lambda_coord * (loss_xy + loss_wh)  +  confident_loss +  class_loss + self.lambda_noobj * no_confident_loss
        return total_loss

    def calculate_iou(self, true_boxes, pred_boxes):
        """
        Tính Intersection over Union giữa hộp thật và hộp dự đoán
        """
        true_xy = true_boxes[..., :2]
        true_wh = true_boxes[..., 2:4]
        pred_xy = pred_boxes[..., :2]
        pred_wh = pred_boxes[..., 2:4]

        # Tính tọa độ góc hộp
        true_half_wh = true_wh / 2.0
        true_min = true_xy - true_half_wh
        true_max = true_xy + true_half_wh
        pred_half_wh = pred_wh / 2.0
        pred_min = pred_xy - pred_half_wh
        pred_max = pred_xy + pred_half_wh

        # Tính giao và hợp
        intersect_min = tf.maximum(true_min, pred_min)
        intersect_max = tf.minimum(true_max, pred_max)
        intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_area = true_wh[..., 0] * true_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        union_area = true_area + pred_area - intersect_area

        iou = intersect_area / (union_area + tf.keras.backend.epsilon())
        return iou


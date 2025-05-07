import tensorflow as tf

class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=2, lambda_coord=5, lambda_noobj=0.5):
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
        y_true = tf.reshape(y_true, shape=(-1,self.S, self.S, 5+self.C)) # batch,S,S,7
        y_pred = tf.reshape(y_pred, shape=(-1, self.S, self.S, 5*self.B + self.C)) # batch,S,S,12

        xywhc_true = tf.reshape(y_true[..., :-self.C], shape=(-1,self.S, self.S, 1, 5)) # batch,S,S,1,5
        xywhc_pred = tf.reshape(y_pred[...,:-self.C], shape=(-1,self.S, self.S, self.B, 5)) # batch,S,S,B,5

        iou_scores = self.calculate_iou(xywhc_true, xywhc_pred) #batch,S,S,B

        # tinh ra vi tri lon nhat giua 2 cai box dua tren ious
        arg_max = tf.math.argmax(input=iou_scores, axis=-1, name="arg_max_x") # batch,S,S

        # tao ra cai one hot de tim co object va ko object
        onehot = tf.one_hot(indices=arg_max, depth=self.B) # batch,7,7,2

        # lay cac doi tuong co doi tuong
        has_obj = xywhc_true[...,4] # batch,7,7,1
        index_has_object = has_obj * onehot # batch,7,7,2
        # them 1 chieu vao de phu hop voi phep nhan ma tran o hang xx
        index_has_object_exp = tf.expand_dims(index_has_object,axis=-2)
        index_no_object = 1 - index_has_object_exp # batch,7,7,2

        # xy loss
        xy_true = xywhc_true[...,:2] # batch,7,7,1,2
        xy_pred = xywhc_pred[...,:2] # batch,7,7,2,2
        loss_xy = tf.math.reduce_sum(index_has_object_exp*tf.math.square(xy_true - xy_pred))

        # wh loss
        wh_true = xywhc_true[...,2:4] # batch,7,7,1,2
        wh_pred = xywhc_pred[...,2:4] # batch,7,7,2,2

        # sy ly gia tri am trong wh vi trong qua trinh su dung can thi wh am tra ve Nan va x + Nan  = Nan
        wh_true = tf.math.maximum(wh_true, 1e-6)
        wh_pred = tf.math.maximum(wh_pred, 1e-6)
        loss_wh = tf.math.reduce_sum(index_has_object_exp*tf.math.square(wh_true - wh_pred))

        # loss cho confident co object va ko co obj
        c_true = xywhc_true[...,4]
        c_pred = xywhc_pred[...,4]
        loss_has_object_c = tf.math.reduce_sum(index_has_object*tf.math.square((c_true - c_pred)))
        loss_no_object_c = tf.math.reduce_sum((1-index_has_object) * tf.math.square((c_true - c_pred)))


        p_true = y_true[...,-self.C:] # batch,S,S,C
        y_pred = y_pred[...,-self.C:] # batch,S,S,C
        loss_p = tf.math.reduce_sum(has_obj*tf.math.square(p_true - y_pred))


        total_loss = self.lambda_coord*loss_xy + self.lambda_coord*loss_wh + loss_has_object_c + self.lambda_noobj*loss_no_object_c + loss_p

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
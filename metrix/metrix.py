import tensorflow as tf

def convert_xywh_to_xyxy(boxes):
    """Chuyển đổi từ (x, y, w, h) sang (x_min, y_min, x_max, y_max)."""
    x, y, w, h = tf.split(boxes, 4, axis=-1)
    x_min = x - w / 2.0
    y_min = y - h / 2.0
    x_max = x + w / 2.0
    y_max = y + h / 2.0
    return tf.concat([x_min, y_min, x_max, y_max], axis=-1)

def compute_iou(boxes1, boxes2):
    """Tính IoU giữa hai tập bounding boxes."""
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxes1, 4, axis=-1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxes2, 4, axis=-1)

    inter_xmin = tf.maximum(x_min1, x_min2)
    inter_ymin = tf.maximum(y_min1, y_min2)
    inter_xmax = tf.minimum(x_max1, x_max2)
    inter_ymax = tf.minimum(y_max1, y_max2)

    inter_area = tf.maximum(inter_xmax - inter_xmin, 0) * tf.maximum(inter_ymax - inter_ymin, 0)
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = area1 + area2 - inter_area
    iou = inter_area / tf.maximum(union_area, 1e-6)
    return tf.squeeze(iou, axis=-1)

class YoloV1Metric(tf.keras.metrics.Metric):
    def __init__(self, iou_threshold=0.5, name="yolo_v1_metric", **kwargs):
        super(YoloV1Metric, self).__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Cập nhật TP, FP, FN cho YOLOv1.
        y_true: (batch, 7, 7, 7) - (x, y, w, h, confidence, p1, p2)
        y_pred: (batch, 7, 7, 12) - (x1, y1, w1, h1, c1, x2, y2, w2, h2, c2, p1, p2)
        """
        # Ground truth
        true_boxes = convert_xywh_to_xyxy(y_true[..., :4])  # (batch, 7, 7, 4)
        true_conf = y_true[..., 4]  # (batch, 7, 7)
        true_classes = tf.argmax(y_true[..., 5:], axis=-1)  # (batch, 7, 7)

        # Predicted boxes (2 boxes per cell)
        pred_box1 = convert_xywh_to_xyxy(y_pred[..., :4])  # (batch, 7, 7, 4)
        pred_box2 = convert_xywh_to_xyxy(y_pred[..., 5:9])  # (batch, 7, 7, 4)
        pred_conf1 = y_pred[..., 4]  # (batch, 7, 7)
        pred_conf2 = y_pred[..., 9]  # (batch, 7, 7)
        pred_classes = tf.argmax(y_pred[..., 10:], axis=-1)  # (batch, 7, 7)

        # Tính IoU cho cả 2 box dự đoán
        iou1 = compute_iou(pred_box1, true_boxes)  # (batch, 7, 7)
        iou2 = compute_iou(pred_box2, true_boxes)  # (batch, 7, 7)

        # Chọn box có IoU cao hơn
        iou_max = tf.maximum(iou1, iou2)  # (batch, 7, 7)
        best_conf = tf.where(iou1 > iou2, pred_conf1, pred_conf2)  # (batch, 7, 7)

        # Xác định TP, FP, FN
        class_match = true_classes == pred_classes  # (batch, 7, 7)
        iou_condition = iou_max >= self.iou_threshold  # (batch, 7, 7)
        obj_mask = true_conf > 0  # Chỉ xét các cell có object thật

        # True Positives: IoU >= threshold và class đúng
        tp_mask = obj_mask & iou_condition & class_match
        tp_count = tf.reduce_sum(tf.cast(tp_mask, tf.float32))

        # False Positives: Dự đoán có object nhưng sai class hoặc IoU thấp
        pred_obj = best_conf > 0.5  # Giả sử ngưỡng confidence là 0.5
        fp_mask = pred_obj & (~iou_condition | ~class_match)
        fp_count = tf.reduce_sum(tf.cast(fp_mask, tf.float32))

        # False Negatives: Có object thật nhưng không được dự đoán đúng
        fn_mask = obj_mask & ~tp_mask
        fn_count = tf.reduce_sum(tf.cast(fn_mask, tf.float32))

        # Cập nhật giá trị
        self.tp.assign_add(tp_count)
        self.fp.assign_add(fp_count)
        self.fn.assign_add(fn_count)

    def result(self):
        """Tính F1 score."""
        precision = self.tp / tf.maximum(self.tp + self.fp, 1e-6)
        recall = self.tp / tf.maximum(self.tp + self.fn, 1e-6)
        f1 = 2 * (precision * recall) / tf.maximum(precision + recall, 1e-6)
        return f1

    def reset_state(self):
        """Reset các giá trị về 0."""
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
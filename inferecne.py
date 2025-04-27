from utils import  *
import numpy as np
from models.backbone import *

# # tải model từ local
yoloV1 = build_yolo_v1_vgg16((448,448,3))
yoloV1.load_weights("my_model.weights.h5")


image,_,_ = loadimage("data\images\Cats_Test3.png", (448,448))
image_pre = np.expand_dims(image,0)
image_pre = image_pre/255.0
result = yoloV1.predict(image_pre)

# get box max , confident, label
boxes, c, classes = outputyolo(result)
plot_detections(image_pre[0],boxes,c,classes,0.8)
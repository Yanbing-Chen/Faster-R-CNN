from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import ImageDraw , Image

from utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         show_config)
import nets.frcnn as frcnn

def generate_anchors(sizes = [128, 256, 512], ratios = [[1, 1], [1, 2], [2, 1]]):
    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    
    for i in range(len(ratios)):
        anchors[3 * i: 3 * i + 3, 2] = anchors[3 * i: 3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i: 3 * i + 3, 3] = anchors[3 * i: 3 * i + 3, 3] * ratios[i][1]
    
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, anchors, stride=16):
    #---------------------------------------------------#
    #   [0,1,2,3,4,5……37]
    #   [0.5,1.5,2.5……37.5]
    #   [8,24,……]
    #---------------------------------------------------#
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    # print(shift_x,shift_y)
    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])

    #---------------------------------------------------#
    #   进行图像的绘制
    #---------------------------------------------------#
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    # plt.ylim(0,600)
    # plt.xlim(0,600)
    plt.scatter(shift_x,shift_y)
    box_widths  = shifted_anchors[:, 2] - shifted_anchors[:, 0]
    box_heights = shifted_anchors[:, 3] - shifted_anchors[:, 1]
    initial = 0
    for i in [initial + 0, initial + 1, initial + 2, initial + 3, initial + 4, initial + 5, initial + 6, initial + 7, initial + 8]:
        rect = plt.Rectangle([shifted_anchors[i, 0], shifted_anchors[i, 1]], box_widths[i], box_heights[i], color="r", fill=False)
        ax.add_patch(rect)
    return shifted_anchors

#---------------------------------------------------#
#   获得resnet50对应的baselayer大小
#---------------------------------------------------#
def get_resnet50_output_length(height, width):
    def get_output_length(input_length):
        filter_sizes    = [7, 3, 1, 1]
        padding         = [3, 1, 0, 0]
        stride          = 2
        for i in range(4):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(height), get_output_length(width)

#---------------------------------------------------#
#   获得vgg对应的baselayer大小
#---------------------------------------------------#
def get_vgg_output_length(height, width):
    def get_output_length(input_length):
        filter_sizes    = [2, 2, 2, 2]
        padding         = [0, 0, 0, 0]
        stride          = 2
        for i in range(4):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(height), get_output_length(width)

def get_anchors(input_shape, backbone, sizes = [128, 256, 512], ratios = [[1, 1], [1, 2], [2, 1]], stride=16):
    if backbone == 'vgg':
        feature_shape = get_vgg_output_length(input_shape[0], input_shape[1])
        print(feature_shape)
    else:
        feature_shape = get_resnet50_output_length(input_shape[0], input_shape[1])
        
    anchors = generate_anchors(sizes = sizes, ratios = ratios)
    anchors = shift(feature_shape, anchors, stride = stride)
    anchors[:, ::2]  /= input_shape[1]
    anchors[:, 1::2] /= input_shape[0]
    anchors = np.clip(anchors, 0, 1)
    return anchors

def decode_boxes(mbox_loc, anchors, variances):
        # 获得先验框的宽与高
        anchor_width     = anchors[:, 2] - anchors[:, 0]
        anchor_height    = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x  = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y  = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        detections_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        detections_center_x += anchor_center_x
        detections_center_y = mbox_loc[:, 1] * anchor_height * variances[1]
        detections_center_y += anchor_center_y
        
        # 真实框的宽与高的求取
        detections_width   = np.exp(mbox_loc[:, 2] * variances[2])
        detections_width   *= anchor_width
        detections_height  = np.exp(mbox_loc[:, 3] * variances[3])
        detections_height  *= anchor_height

        # 获取真实框的左上角与右下角
        detections_xmin = detections_center_x - 0.5 * detections_width
        detections_ymin = detections_center_y - 0.5 * detections_height
        detections_xmax = detections_center_x + 0.5 * detections_width
        detections_ymax = detections_center_y + 0.5 * detections_height

        # 真实框的左上角与右下角进行堆叠
        detections = np.concatenate((detections_xmin[:, None],
                                      detections_ymin[:, None],
                                      detections_xmax[:, None],
                                      detections_ymax[:, None]), axis=-1)
        # 防止超出0与1
        detections = np.minimum(np.maximum(detections, 0.0), 1.0)
        return detections

def detection_out_rpn(rpn_pre_boxes, predictions, anchors, variances = [0.25, 0.25, 0.25, 0.25]):
    #---------------------------------------------------#
    #   获得种类的置信度
    #---------------------------------------------------#
    mbox_conf   = predictions[0]
    #---------------------------------------------------#
    #   mbox_loc是回归预测结果
    #---------------------------------------------------#
    mbox_loc    = predictions[1]

    results = []
    # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
    for i in range(len(mbox_loc)):
        #--------------------------------#
        #   利用回归结果对先验框进行解码
        #--------------------------------#
        detections     = decode_boxes(mbox_loc[i], anchors, variances)
        #--------------------------------#
        #   取出先验框内包含物体的概率
        #--------------------------------#
        c_confs         = mbox_conf[i, :, 0]
        c_confs_argsort = np.argsort(c_confs)[::-1][:rpn_pre_boxes]

        #------------------------------------#
        #   原始的预测框较多，先选一些高分框
        #------------------------------------#
        confs_to_process = c_confs[c_confs_argsort]
        boxes_to_process = detections[c_confs_argsort, :]
        #--------------------------------#
        #   进行iou的非极大抑制
        #--------------------------------#
        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, 150, iou_threshold = 0.7).numpy()
        
        #--------------------------------#
        #   取出在非极大抑制中效果较好的内容
        #--------------------------------#
        good_boxes  = boxes_to_process[idx]
        results.append(good_boxes)
    return np.array(results)


def get_proposal(image,model_rpn,anchors):
    image_shape = np.array(np.shape(image)[0:2])
    input_shape = get_new_img_size(image_shape[0], image_shape[1])
    image       = cvtColor(image)
    image_data  = resize_image(image, [600,600])
    image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
    rpn_pred        =  model_rpn(image_data)
    rpn_pred        = [x.numpy() for x in rpn_pred]
    
    
   
    
    results = detection_out_rpn(12000,rpn_pred,anchors)
    
    top_boxes   = results[0][:, :4]
    
    draw = ImageDraw.Draw(image)
    for bbox in top_boxes:
        
        xmin, ymin, xmax, ymax = bbox[0] * image_shape[1],bbox[1] *  image_shape[0],bbox[2]* image_shape[1],bbox[3]* image_shape[0]
        draw.rectangle([xmin, ymin, xmax, ymax])
                
    del draw
    
    return image
    
if __name__ == "__main__":
    anchors = get_anchors([600, 600], 'vgg')
    img_paths = glob.glob(os.path.join('img','*.*'))

    model_rpn,_ = frcnn.get_predict_model(20, 'vgg')
    model_rpn.load_weights('model_data/voc_weights_vgg.h5', by_name=True)
    for path in img_paths:
        image = Image.open(path)
        basename = os.path.basename(path)
        w_image = get_proposal(image,model_rpn,anchors)
        w_image.save(os.path.join('first_stage_out',basename))
    
    

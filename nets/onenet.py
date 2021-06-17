import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input, Lambda,
                                     MaxPooling2D, Reshape, ZeroPadding2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from nets.onenet_loss import loss, cls, loc, giou
from nets.resnet import ResNet50, onenet_head
from nets.resnet18 import ResNet18


def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='SAME')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(cls_pred, max_objects=100):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 20
    #   Hot map热力图 -> b, 128, 128, 20
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    cls_pred = nms(cls_pred)
    b, h, w, c = tf.shape(cls_pred)[0], tf.shape(cls_pred)[1], tf.shape(cls_pred)[2], tf.shape(cls_pred)[3]
    #-------------------------------------------#
    #   将所有结果平铺，获得(b, 128 * 128 * 20)
    #-------------------------------------------#
    cls_pred = tf.reshape(cls_pred, (b, -1))
    #-----------------------------#
    #   scores.shape = (b, k), indices.shape = (b, k)
    #-----------------------------#
    scores, indices = tf.math.top_k(cls_pred, k=max_objects, sorted=True)

    #--------------------------------------#
    #   计算求出种类、网格点以及索引。
    #--------------------------------------#
    #   這裡的 indices 包含了種類，值的range 在[0, 128 * 128 * 20)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    #   這裡的 indices 已不包含種類，只計算位置，值的range 在[0, 128 * 128)
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(cls_pred, loc_pred, max_objects=100, num_classes=20):
    #-----------------------------------------------------#
    #   cls_pred          b, 128, 128, num_classes
    #   loc_pred          b, 128, 128, 4
    #   reg         b, 128, 128, 2
    #   scores      b, max_objects
    #   indices     b, max_objects
    #   class_ids   b, max_objects
    #   xs          b, max_objects
    #   ys          b, max_objects
    #-----------------------------------------------------#

    scores, indices, class_ids, xs, ys = topk(cls_pred, max_objects=max_objects)
    b = tf.shape(cls_pred)[0]

    #-----------------------------------------------------#
    #   loc_pred          b, 128 * 128, 4
    #-----------------------------------------------------#
    loc_pred = tf.reshape(loc_pred, [b, -1, 4])
    length = tf.shape(loc_pred)[1]

    #-----------------------------------------------------#
    #   找到其在1维上的索引
    #   batch_idx   b, max_objects
    #   length = 128*128
    #-----------------------------------------------------#
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) + tf.reshape(indices, [-1])
    #-----------------------------------------------------#
    #   取出top_k个框对应的参数
    #-----------------------------------------------------#
    topk_loc = tf.gather(tf.reshape(loc_pred, [-1, 4]), full_indices)
    topk_loc = tf.reshape(topk_loc, [b, -1, 4])


    #-----------------------------------------------------#
    #   计算预测框左上角和右下角
    #   topk_x1     b,k,1       预测框左上角x轴坐标 
    #   topk_y1     b,k,1       预测框左上角y轴坐标
    #   topk_x2     b,k,1       预测框右下角x轴坐标
    #   topk_y2     b,k,1       预测框右下角y轴坐标
    #-----------------------------------------------------#
    topk_x1, topk_y1 = topk_loc[..., 0:1], topk_loc[..., 1:2]
    topk_x2, topk_y2 = topk_loc[..., 2:3], topk_loc[..., 3:4]
    #-----------------------------------------------------#
    #   scores      b,k,1       预测框得分
    #   class_ids   b,k,1       预测框种类
    #-----------------------------------------------------#
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)

    #-----------------------------------------------------#
    #   detections  预测框所有参数的堆叠
    #   前四个是预测框的坐标，后两个是预测框的得分与种类
    #-----------------------------------------------------#
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections


def onenet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", prior_prob=0.01,num_stacks=2):
    assert backbone in ['resnet18', 'resnet50']
    output_size = input_shape[0] // 4
    image_input = Input(shape=input_shape)
    cls_input = Input(shape=(max_objects, num_classes))
    # hm_input = Input(shape=(output_size, output_size, num_classes))
    # wh_input = Input(shape=(max_objects, 2))
    # reg_input = Input(shape=(max_objects, 2))
    loc_input = Input(shape=(max_objects, 4))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    if backbone=='resnet18':
        # -----------------------------------#
        #   对输入图片进行特征提取
        #   512, 512, 3 -> 16, 16, 512
        # -----------------------------------#
        C5 = ResNet18(image_input)
    elif backbone=='resnet50':
        # -----------------------------------#
        #   对输入图片进行特征提取
        #   512, 512, 3 -> 16, 16, 2048
        # -----------------------------------#
        C5 = ResNet50(image_input)
    else:
        try:
            raise EOFError
        except:
            print('EOFError')

    #--------------------------------------------------------------------------------------------------------#
    #   对获取到的特征进行上采样，进行分类预测和回归预测
    #   16, 16, 1024 -> 32, 32, 256 -> 64, 64, 128 -> 128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
    #        or  512                                               -> 128, 128, 64 -> 128, 128, 2
    #                                                              -> 128, 128, 64 -> 128, 128, 2
    #--------------------------------------------------------------------------------------------------------#
    y1, y2 = onenet_head(C5, num_classes, prior_prob)
    if mode=="train":
        l1, l2, l3 = Lambda(loss, name='loss')([y1, y2, cls_input, loc_input, reg_mask_input])
        cls_loss_ = Lambda(cls, name='cls')([l1])
        loc_loss_ = Lambda(loc, name='loc')([l2])
        giou_loss_ = Lambda(giou, name='giou')([l3])
        model = Model(inputs=[image_input, cls_input, loc_input, reg_mask_input], outputs=[cls_loss_, loc_loss_, giou_loss_])
        return model
    else:
        detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
                                            num_classes=num_classes))([y1, y2])
        prediction_model = Model(inputs=image_input, outputs=detections)
        return prediction_model


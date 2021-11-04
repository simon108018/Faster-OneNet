# -------------------------------------------------------------#
#   ResNet的网络部分
# -------------------------------------------------------------#
from typing import Optional, Dict, Any, Union, Tuple

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (InputSpec, Layer, Activation, BatchNormalization, Conv2D, GlobalAvgPool2D,
                                     Dense, Conv2DTranspose, Add, MaxPooling2D, ZeroPadding2D)

from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers
from tensorflow_addons.layers import GroupNormalization

class apply_ltrb(Layer):
    def __init__(self, name=None, **kwargs):
        super(apply_ltrb, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        b, w, h, c= self.input_spec.shape
        self.ct = tf.cast(tf.transpose(tf.meshgrid(tf.range(0, w), tf.range(0, h))), tf.float32)
    def get_config(self):
        config = super(apply_ltrb, self).get_config()
        return config

    def call(self, pred_ltrb):
        '''
        pred_ltrb 上的4個value分別是(x1, y1, x2, y2)表示以每個cell為中心，預測出來的框架左上角與右下角的相對距離
        ltrb(left-up-right-bottom)
        此函數將預測出來的相對位置換算成絕對位置

        下面是一個框，在cell(cx,cy)取得相對距離(x1,y1,x2,y2)後，換算成絕對位置(cx-x1,cy-y1,cx+x2,cy+y2)
        (cx-x1,cy-y1)
          ----------------------------------
          |          ↑                     |
          |          |                     |
          |          |y1                   |
          |          |                     |
          |←------(cx,cy)-----------------→|
          |   x1     |          x2         |
          |          |                     |
          |          |                     |
          |          |y2                   |
          |          |                     |
          |          |                     |
          |          ↓                     |
          ----------------------------------(cx+x2,cy+y2)
        '''
        # b, w, h, c = tf.shape(pred_ltrb)[0], tf.shape(pred_ltrb)[1], tf.shape(pred_ltrb)[2], tf.shape(pred_ltrb)[3]
        # ct = tf.cast(tf.transpose(tf.meshgrid(tf.range(0, w), tf.range(0, h))), tf.float32)
        # locations : w*h*2 這2個 value包含 cx=ct[0], cy=ct[1]
        locations = tf.concat((self.ct - pred_ltrb[:, :, :, :2], self.ct + pred_ltrb[:, :, :, 2:]), axis=-1)
        return locations

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def ResNet50(image_input=tf.keras.Input(shape=(512, 512, 3))):
    model = tf.keras.applications.ResNet50(include_top=False, input_tensor=image_input)

    #  64, 64, 256  (38, 38, 256)
    o1 = model.get_layer('conv3_block4_out').output
    # o1 = model.get_layer('conv2_block3_1_relu').output
    #  32, 32, 1024  (19, 19, 1024)
    o2 = model.get_layer('conv4_block6_out').output
    # o2 = model.get_layer('conv4_block6_1_relu').output
    #  16, 16, 2048  (10, 10, 2048)
    o3 = model.get_layer('conv5_block3_out').output
    # o = model.get_layer('post_relu').output

    #  8, 8, 256 (5, 5, 256)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu',
               padding='same',
               name='o4_conv1x1')(o3)
    o4 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2),
                                   activation='relu', padding='same',
                                   name='o4_conv3x3')(x)
    # o4 = Conv2D(256, kernel_size=(3,3), strides=(2, 2),
    #            activation='relu', padding='same',
    #            name='stage5_b')(x)

    #  4, 4, 256 (3, 3, 256)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu',
                                   padding='same',
                                   name='o5_conv1x1')(o4)
    o5= Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='o5_conv3x3')(x)

    #  2, 2, 256 (1, 1, 256)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu',
                                   padding='same',
                                   name='o6_conv1x1')(o5)
    o6= Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                                   activation='relu', padding='valid',
                                   name='o6_conv3x3')(x)
    return [o1, o2, o3, o4, o5, o6]



def SSD_OneNet(x, num_classes, prior_prob, shortcut=True, mode=None):
    o1, o2, o3, o4, o5, o6 = x
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    output_list = []
    ## o1

    o1 = BatchNormalization(name='o1_bn')(o1)
    # cls1 header (38*38*20)
    cls1 = Conv2D(num_classes, 3, padding='same',
                  kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                  bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid',
                  name='pred_cls1')(o1)

    # loc1 header (38*38*4)
    loc1 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal',
                  kernel_regularizer=l2(5e-4), activation='relu', name='loc1_output')(o1)
    loc_dir1 = apply_ltrb(name='pred_location1')(loc1)


    ## o2

    # cls2 header (19*19*20)
    cls2 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                  bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls2')(o2)

    # loc2 header (19*19*4)
    loc2 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu',
                  name='loc2_output')(o2)
    loc_dir2 = apply_ltrb(name='pred_location2')(loc2)

    ## o3

    # cls3 header (10*10*20)
    cls3 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                  bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls3')(o3)

    # loc3 header (10*10*4)
    loc3 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu',
                  name='loc3_output')(o3)
    loc_dir3 = apply_ltrb(name='pred_location3')(loc3)

    ## o4

    # cls4 header (5*5*20)
    cls4 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                  bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls4')(o4)

    # loc4 header (5*5*4)
    loc4 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu',
                  name='loc4_output')(o4)
    loc_dir4 = apply_ltrb(name='pred_location4')(loc4)

    ## o5

    # cls5 header (3*3*20)
    cls5 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                  bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls5')(o5)

    # loc5 header (3*3*4)
    loc5 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu',
                  name='loc5_output')(o5)
    loc_dir5 = apply_ltrb(name='pred_location5')(loc5)

    ## o6

    # cls6 header (1*1*20)
    cls6 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                  bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls6')(o6)

    # loc6 header (1*1*4)
    loc6 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu',
                  name='loc6_output')(o6)
    loc_dir6 = apply_ltrb(name='pred_location6')(loc6)




    if '1' in mode or 'all' in mode:
        output_list.extend([cls1, loc1, loc_dir1])

    if '2' in mode or 'all' in mode:
        output_list.extend([cls2, loc2, loc_dir2])

    if '3' in mode or 'all' in mode:
        output_list.extend([cls3, loc3, loc_dir3])

    if '4' in mode or 'all' in mode:
        output_list.extend([cls4, loc4, loc_dir4])

    if '5' in mode or 'all' in mode:
        output_list.extend([cls5, loc5, loc_dir5])

    if '6' in mode:
        output_list.extend([cls6, loc6, loc_dir6])

    return output_list

# def SSD_OneNet(x, num_classes, prior_prob, shortcut=True, mode=None):
#     o1, o2, o3, o4, o5, o6 = x
#     bias_value = -np.log((1 - prior_prob) / prior_prob)
#     output_list = []
#     ## o1
#     o1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
#                use_bias=False, padding='same',
#                kernel_initializer='he_normal',
#                kernel_regularizer=l2(5e-4), name='o1_conv')(o1)
#     o1 = BatchNormalization(name='o1_bn')(o1)
#
#     o1 = Activation('relu', name='o1_relu')(o1)
#
#
#     # loc header
#     cls1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name='cls1_conv')(o1)
#     cls1 = GroupNormalization(groups=32, name='cls1_bn')(cls1)
#     cls1 = Activation('relu', name='cls1_relu')(cls1)
#     cls1 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
#                 bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls1')(cls1)
#
#     # loc header (128*128*4)
#     loc1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name='loc1_conv')(o1)
#     loc1 = GroupNormalization(groups=32, name='loc1_bn')(loc1)
#     loc1 = Activation('relu', name='loc1_relu')(loc1)
#     loc1 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu', name='loc1_output')(loc1)
#     loc_dir1 = apply_ltrb(name='pred_location1')(loc1)
#
#
#     ## o2
#     o2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
#                use_bias=False, padding='same',
#                kernel_initializer='he_normal',
#                kernel_regularizer=l2(5e-4), name='o2_conv')(o2)
#     o2 = BatchNormalization(name='o2_bn')(o2)
#     o2 = Activation('relu', name='o2_relu')(o2)
#
#     # cls header (32*32*4)
#     cls2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
#                 name='cls2_conv')(o2)
#     cls2 = GroupNormalization(groups=32, name='cls2_bn')(cls2)
#     cls2 = Activation('relu', name='cls2_relu')(cls2)
#     if shortcut:
#         cls_shortcut = Conv2D(32, 3, 2, padding='same', use_bias=False, name='cls_shortcut1_conv1')(cls1)
#         cls_shortcut = BatchNormalization(name='cls_shortcut1_bn1')(cls_shortcut)
#         cls_shortcut = Activation('relu', name='cls_shortcut1_relu1')(cls_shortcut)
#         cls_shortcut = Conv2D(64, 3, 2, padding='same', use_bias=False, name='cls_shortcut1_conv2')(cls_shortcut)
#         cls_shortcut = BatchNormalization(name='cls_shortcut1_bn2')(cls_shortcut)
#         cls_shortcut = Activation('relu', name='cls_shortcut1_relu2')(cls_shortcut)
#         cls2 = Add()([cls2, cls_shortcut])
#     cls2 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
#                   bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls2')(cls2)
#
#     # loc header (32*32*4)
#     loc2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
#                 name='loc2_conv')(o2)
#     loc2 = GroupNormalization(groups=32, name='loc2_bn')(loc2)
#     loc2 = Activation('relu', name='loc2_relu')(loc2)
#     if shortcut:
#         loc_shortcut = Conv2D(32, 3, 2, padding='same', use_bias=False, name='loc_shortcut1_conv1')(loc1)
#         loc_shortcut = BatchNormalization(name='loc_shortcut1_bn1')(loc_shortcut)
#         loc_shortcut = Activation('relu', name='loc_shortcut1_relu1')(loc_shortcut)
#         loc_shortcut = Conv2D(64, 3, 2, padding='same', use_bias=False, name='loc_shortcut1_conv2')(loc_shortcut)
#         loc_shortcut = BatchNormalization(name='loc_shortcut1_bn2')(loc_shortcut)
#         loc_shortcut = Activation('relu', name='loc_shortcut1_relu2')(loc_shortcut)
#         loc2 = Add()([loc2, loc_shortcut])
#     loc2 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu',
#                   name='loc2_output')(loc2)
#     loc_dir2 = apply_ltrb(name='pred_location2')(loc2)
#
#
#     ## o3
#     o3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
#                use_bias=False, padding='same',
#                kernel_initializer='he_normal',
#                kernel_regularizer=l2(5e-4), name='o3_conv')(o3)
#     o3 = BatchNormalization(name='o3_bn')(o3)
#     o3 = Activation('relu', name='o3_relu' )(o3)
#
#     # cls header (8*8*4)
#     cls3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name='cls3_conv')(o3)
#     cls3 = GroupNormalization(groups=32, name='cls3_bn')(cls3)
#     cls3 = Activation('relu', name='cls3_relu')(cls3)
#     if shortcut:
#         cls_shortcut = Conv2D(32, 3, 2, padding='same', use_bias=False, name='cls_shortcut2_conv1')(cls2)
#         cls_shortcut = BatchNormalization(name='cls_shortcut2_bn1')(cls_shortcut)
#         cls_shortcut = Activation('relu', name='cls_shortcut2_relu1')(cls_shortcut)
#         cls_shortcut = Conv2D(64, 3, 2, padding='same', use_bias=False, name='cls_shortcut2_conv2')(cls_shortcut)
#         cls_shortcut = BatchNormalization(name='cls_shortcut2_bn2')(cls_shortcut)
#         cls_shortcut = Activation('relu', name='cls_shortcut2_relu2')(cls_shortcut)
#         cls3 = Add()([cls3, cls_shortcut])
#     cls3 = Conv2D(num_classes, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
#                 bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid', name='pred_cls3')(cls3)
#
#     # loc header (8*8*4)
#     loc3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),name='loc3_conv')(o3)
#     loc3 = GroupNormalization(groups=32, name='loc3_bn')(loc3)
#     loc3 = Activation('relu', name='loc3_relu')(loc3)
#     if shortcut:
#         loc_shortcut = Conv2D(32, 3, 2, padding='same', use_bias=False, name='loc_shortcut2_conv1')(loc2)
#         loc_shortcut = BatchNormalization(name='loc_shortcut2_bn1')(loc_shortcut)
#         loc_shortcut = Activation('relu', name='loc_shortcut2_relu1')(loc_shortcut)
#         loc_shortcut = Conv2D(64, 3, 2, padding='same', use_bias=False, name='loc_shortcut2_conv2')(loc_shortcut)
#         loc_shortcut = BatchNormalization(name='loc_shortcut2_bn2')(loc_shortcut)
#         loc_shortcut = Activation('relu', name='loc_shortcut2_relu2')(loc_shortcut)
#         loc3 = Add()([loc3, loc_shortcut])
#     loc3 = Conv2D(4, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu', name='loc3_output')(loc3)
#     loc_dir3 = apply_ltrb(name='pred_location3')(loc3)
#
#
#     if '1' in mode:
#         output_list.extend([cls1, loc1, loc_dir1])
#
#     if '2' in mode:
#         output_list.extend([cls2, loc2, loc_dir2])
#
#     if '3' in mode:
#         output_list.extend([cls3, loc3, loc_dir3])
#
#     return output_list




def BasicBlock(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):

    filters1, filters2 = filters

    conv_name_base = 'conv' + str(stage) + '_' + block
    bn_name_base = 'bn' + str(stage) + '_' + block

    x = Conv2D(filters1, kernel_size, strides=strides, padding='same',
               name=conv_name_base + '_0', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '_0', momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '_1', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '_1', momentum=0.9, epsilon=1e-5)(x)

    if strides != (1, 1):
        shortcut = Conv2D(filters2, (1, 1), strides=strides, padding='same',
                          name=conv_name_base + '_shortcut', use_bias=False)(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + '_shortcut', momentum=0.9, epsilon=1e-5)(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation('relu', name='stage{}_{}'.format(stage, block))(x)
    return x

def ResNet18_model(image_input=tf.keras.Input(shape=(512, 512, 3))) -> tf.keras.Model:
    # 256,256,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False)(image_input)
    x = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 128,128,64 -> 128,128,64
    x = BasicBlock(x, 3, [64, 64], stage=1, block='a', strides=(1, 1))
    x = BasicBlock(x, 3, [64, 64], stage=1, block='b', strides=(1, 1))

    # 128,128,64 -> 64,64,128
    x = BasicBlock(x, 3, [128, 128], stage=2, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [128, 128], stage=2, block='b', strides=(1, 1))

    # 64,64,128 -> 32,32,256
    x = BasicBlock(x, 3, [256, 256], stage=3, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [256, 256], stage=3, block='b', strides=(1, 1))

    # 32,32,256 -> 16,16,512
    x = BasicBlock(x, 3, [512, 512], stage=4, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [512, 512], stage=4, block='b', strides=(1, 1))
    x = GlobalAvgPool2D()(x)
    x = Dense(1000, name='fully_connected', activation='softmax', use_bias=False)(x)

    return tf.keras.models.Model(inputs=image_input, outputs=x)


def ResNet18(image_input=tf.keras.Input(shape=(512,512,3))):
    model = ResNet18_model(image_input)
    model.load_weights('./my_ResNet_18.h5')
    o1 = model.get_layer('stage1_b').output
    o2 = model.get_layer('stage2_b').output
    o3 = model.get_layer('stage3_b').output
    o4 = model.get_layer('stage4_b').output

    return [o1, o2, o3, o4]


def onenet_head(x, num_classes, prior_prob):
    o1, o2, o3, o4 = x
    x = o4
    # -------------------------------#
    #   解码器
    # -------------------------------#
    num_filters = 256
    #   Deconvolution
    # 16, 16, 2048  ->  32, 32, 256  -> 64, 64, 128 -> 128, 128, 64
    #   conv_output
    # o             ||  o3           || o2          ||  o1
    # 16, 16, 2048  ||  32, 32, 1024 || 64, 64, 512 || 128, 128, 256
    # 將conv_output 再經過一次conv_layer 使 channel 相同，再經過Add_layer加起來
    o = [o3, o2, o1]
    for i in range(3):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, Conv2D(num_filters // pow(2, i), (1, 1), strides=1, padding='same')(o[i])])
    # 最终获得128,128,64的特征层
    # cls header
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    cls = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                bias_initializer=initializers.Constant(value=bias_value), activation='sigmoid')(y1)

    # loc header (128*128*4)
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    loc = Conv2D(4, 3, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu')(y2)
    loc_dir = apply_ltrb(name='pred_location')(loc)

    return cls, loc, loc_dir
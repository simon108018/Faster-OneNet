#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (Activation, AveragePooling2D,
                                     BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Flatten, Lambda,
                                     Input, MaxPooling2D, ZeroPadding2D, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2
from tensorflow.keras import initializers


def apply_ltrb(pred_ltrb):

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
    b, w, h, c = tf.shape(pred_ltrb)[0], tf.shape(pred_ltrb)[1], tf.shape(pred_ltrb)[2], tf.shape(pred_ltrb)[3]
    ct = tf.cast(tf.transpose(tf.meshgrid(tf.range(0, w), tf.range(0, h))), tf.float32)
    # locations : w*h*2 這2個 value包含 cx=ct[0], cy=ct[1]
    locations = tf.concat((ct - pred_ltrb[:, :, :, :2], ct + pred_ltrb[:, :, :, 2:]), axis=-1)
    return locations

# def identity_block(input_tensor, kernel_size, filters, stage, block):
#
#     filters1, filters2, filters3 = filters
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
#     x = BatchNormalization(name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b', use_bias=False)(x)
#     x = BatchNormalization(name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
#     x = BatchNormalization(name=bn_name_base + '2c')(x)
#
#     x = layers.add([x, input_tensor])
#     x = Activation('relu')(x)
#     return x
#
#
# def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#
#     filters1, filters2, filters3 = filters
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), strides=strides,
#                name=conv_name_base + '2a', use_bias=False)(input_tensor)
#     x = BatchNormalization(name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, padding='same',
#                name=conv_name_base + '2b', use_bias=False)(x)
#     x = BatchNormalization(name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
#     x = BatchNormalization(name=bn_name_base + '2c')(x)
#
#     shortcut = Conv2D(filters3, (1, 1), strides=strides,
#                       name=conv_name_base + '1', use_bias=False)(input_tensor)
#     shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
#
#     x = layers.add([x, shortcut])
#     x = Activation('relu')(x)
#     return x


# def ResNet50(inputs):
#     # 512x512x3
#     x = ZeroPadding2D((3, 3))(inputs)
#     # 256,256,64
#     x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
#     x = BatchNormalization(name='bn_conv1')(x)
#     x = Activation('relu')(x)
#
#     # 256,256,64 -> 128,128,64
#     x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
#
#     # 128,128,64 -> 128,128,256
#     x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
#     o1 = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
#
#     # 128,128,256 -> 64,64,512
#     x = conv_block(o1, 3, [128, 128, 512], stage=3, block='a')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
#     x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
#     o2 = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
#
#     # 64,64,512 -> 32,32,1024
#     x = conv_block(o2, 3, [256, 256, 1024], stage=4, block='a')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
#     x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
#     o3 = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
#
#     # 32,32,1024 -> 16,16,2048
#     x = conv_block(o3, 3, [512, 512, 2048], stage=5, block='a')
#     x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
#     o = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
#     x = [o, o1, o2, o3]
#     return x

def ResNet50(image_input=tf.keras.Input(shape=(512, 512, 3))):
    model = tf.keras.applications.ResNet50(include_top=False, input_tensor=image_input)

    # 128,128,  256
    o1 = model.get_layer('conv2_block3_out').output
    #  64, 64,  512
    o2 = model.get_layer('conv3_block4_out').output
    #  32, 32, 1024
    o3 = model.get_layer('conv4_block6_out').output
    #  16, 16,  2048
    o = model.get_layer('conv5_block3_out').output
    x = [o, o1, o2, o3]
    return x

def ResNet50V2(image_input=tf.keras.Input(shape=(512, 512, 3))):
    model = tf.keras.applications.ResNet50V2(include_top=False, input_tensor=image_input)

    # 128,128,  256
    o1 = model.get_layer('conv2_block2_preact_relu').output
    #  64, 64,  512
    o2 = model.get_layer('conv3_block2_preact_relu').output
    #  32, 32, 1024
    o3 = model.get_layer('conv4_block6_preact_relu').output
    #  16, 16,  512
    o = model.get_layer('post_relu').output
    x = [o, o1, o2, o3]
    return x

def onenet_head(x, num_classes, prior_prob):
    # x = Dropout(rate=0.5)(x)
    x, o1, o2, o3 = x
    #-------------------------------#
    #   解码器
    #-------------------------------#
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
    bias_value = -tf.math.log((1 - prior_prob) / prior_prob)
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                bias_initializer=initializers.Constant(value=bias_value),activation='sigmoid')(y1)

    # loc header (128*128*4)
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv2D(4, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='relu')(y2)
    y2 = Lambda(apply_ltrb, name='pred_location')(y2)


    return y1, y2

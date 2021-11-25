import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, InputSpec, Layer,
                                     Activation, BatchNormalization,
                                     Conv2D, Conv2DTranspose, Add,
                                     Concatenate, Flatten, Reshape)
from tensorflow.keras.regularizers import l2
from nets.resnet import Backbone
from tensorflow.keras import initializers
class relative_to_abslolue(Layer):
    def __init__(self, name=None, **kwargs):
        super(relative_to_abslolue, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        self.b, self.h, self.w, self.c= self.input_spec.shape
        self.ct = tf.cast(tf.transpose(tf.meshgrid(tf.range(0, self.h), tf.range(0, self.w))), tf.float32)+0.5
    def get_config(self):
        config = super(relative_to_abslolue, self).get_config()
        return config

    def call(self, pred_ltrb):
        '''
        pred_ltrb 上的4個value分別是(y1, x1, y2, x2)表示以每個cell為中心，預測出來的框架左上角與右下角的相對距離
        ltrb(left-up-right-bottom)
        此函數將預測出來的相對位置換算成絕對位置

        下面是一個框，在cell(cy,cx)取得相對距離(y1,x1,y2,x2)後，換算成絕對位置(cy-y1,cx-x1,cy+y2,cx+x2)
        (cy-y1,cx-x1)
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

        # locations : w*h*2 這2個 value包含 cy=ct[0], cx=ct[1]
        locations = tf.concat((self.ct - pred_ltrb[:, :, :, :2], self.ct + pred_ltrb[:, :, :, 2:]), axis=-1)
        locations = tf.divide(locations, [self.h, self.w, self.h, self.w])
        return locations

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ssd_onenet_head(input_tensor = Input(shape=(300, 300, 3)), num_classes=20, prior_prob=0.01, backbone='resnet50', output_layers=6):
    # ---------------------------------#
    #   典型的输入大小为[300,300,3]
    # ---------------------------------#
    # net变量里面包含了整个SSD的结构，通过层名可以找到对应的特征层
    net = Backbone(input_tensor, backbone_name=backbone)
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    # loc_bias_value = -5.

    ## o1

    # o1 = BatchNormalization(name='o1_bn')(o1)
    # cls1 header (38*38*20)
    if output_layers >= 1:
        net['cls1_conv'] = Conv2D(num_classes, 3, padding='same',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(5e-4),
                                  bias_initializer=initializers.Constant(value=bias_value),
                                  name='cls1_conv')(net['o1'])
        net['cls1_flatten'] = Flatten(name='cls1_flatten')(net['cls1_conv'])
        # loc1 header (38*38*4)
        net['relative_loc1'] = Conv2D(4, 3, padding='same',
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=l2(5e-4),
                                      # bias_initializer=initializers.Constant(value=loc_bias_value),
                                      activation='relu',
                                      name='relative_loc1')(net['o1'])
        net['absolute_loc1'] = relative_to_abslolue(name='absolute_loc1')(net['relative_loc1'])
        net['loc1_flatten'] = Flatten(name='loc1_flatten')(net['absolute_loc1'])
        # net['loc1_flatten'] = Flatten(name='loc1_flatten')(net['relative_loc1'])

    ## o2
    if output_layers >= 2:
        # cls2 header (19*19*20)
        net['cls2_conv'] = Conv2D(num_classes, 3, padding='same',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(5e-4),
                                  bias_initializer=initializers.Constant(value=bias_value),
                                  name='cls2_conv')(net['o2'])
        net['cls2_flatten'] = Flatten(name='cls2_flatten')(net['cls2_conv'])
        # loc2 header (19*19*4)
        net['relative_loc2'] = Conv2D(4, 3, padding='same',
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=l2(5e-4),
                                      # bias_initializer=initializers.Constant(value=loc_bias_value),
                                      activation='relu',
                                      name='relative_loc2')(net['o2'])
        net['absolute_loc2'] = relative_to_abslolue(name='absolute_loc2')(net['relative_loc2'])
        net['loc2_flatten'] = Flatten(name='loc2_flatten')(net['absolute_loc2'])
        # net['loc2_flatten'] = Flatten(name='loc2_flatten')(net['relative_loc2'])


    ## o3
    if output_layers >= 3:
        # cls3 header (10*10*20)
        net['cls3_conv'] = Conv2D(num_classes, 3, padding='same',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(5e-4),
                                  bias_initializer=initializers.Constant(value=bias_value),
                                  name='cls3_conv')(net['o3'])
        net['cls3_flatten'] = Flatten(name='cls3_flatten')(net['cls3_conv'])
        # loc3 header (10*10*4)
        net['relative_loc3'] = Conv2D(4, 3, padding='same',
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=l2(5e-4),
                                      # bias_initializer=initializers.Constant(value=loc_bias_value),
                                      activation='relu',
                                      name='relative_loc3')(net['o3'])
        net['absolute_loc3'] = relative_to_abslolue(name='absolute_loc3')(net['relative_loc3'])
        net['loc3_flatten'] = Flatten(name='loc3_flatten')(net['absolute_loc3'])
        # net['loc3_flatten'] = Flatten(name='loc3_flatten')(net['relative_loc3'])

    ## o4
    if output_layers >= 4:
        # cls4 header (5*5*20)
        net['cls4_conv'] = Conv2D(num_classes, 3, padding='same',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(5e-4),
                                  bias_initializer=initializers.Constant(value=bias_value),
                                  name='cls4_conv')(net['o4'])
        net['cls4_flatten'] = Flatten(name='cls4_flatten')(net['cls4_conv'])
        # loc4 header (5*5*4)
        net['relative_loc4'] = Conv2D(4, 3, padding='same',
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=l2(5e-4),
                                      # bias_initializer=initializers.Constant(value=loc_bias_value),
                                      activation='relu',
                                      name='relative_loc4')(net['o4'])
        net['absolute_loc4'] = relative_to_abslolue(name='absolute_loc4')(net['relative_loc4'])
        net['loc4_flatten'] = Flatten(name='loc4_flatten')(net['absolute_loc4'])
        # net['loc4_flatten'] = Flatten(name='loc4_flatten')(net['relative_loc4'])
    ## o5
    if output_layers >= 5:
        # cls5 header (3*3*20)
        net['cls5_conv'] = Conv2D(num_classes, 3, padding='same',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(5e-4),
                                  bias_initializer=initializers.Constant(value=bias_value),
                                  name='cls5_conv')(net['o5'])
        net['cls5_flatten'] = Flatten(name='cls5_flatten')(net['cls5_conv'])
        # loc5 header (3*3*4)
        net['relative_loc5'] = Conv2D(4, 3, padding='same',
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=l2(5e-4),
                                      # bias_initializer=initializers.Constant(value=loc_bias_value),
                                      activation='relu',
                                      name='relative_loc5')(net['o5'])
        net['absolute_loc5'] = relative_to_abslolue(name='absolute_loc5')(net['relative_loc5'])
        net['loc5_flatten'] = Flatten(name='loc5_flatten')(net['absolute_loc5'])
        # net['loc5_flatten'] = Flatten(name='loc5_flatten')(net['relative_loc5'])

    ## o6
    if output_layers >= 6:
        # cls6 header (1*1*20)
        net['cls6_conv'] = Conv2D(num_classes, 3, padding='same',
                                  bias_initializer=initializers.Constant(value=bias_value),
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(5e-4),
                                  name='cls6_conv')(net['o6'])
        net['cls6_flatten'] = Flatten(name='cls6_flatten')(net['cls6_conv'])
        # loc6 header (1*1*4)
        net['relative_loc6'] = Conv2D(4, 3, padding='same',
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=l2(5e-4),
                                      # bias_initializer=initializers.Constant(value=loc_bias_value),
                                      activation='relu',
                                      name='relative_loc6')(net['o6'])
        net['absolute_loc6'] = relative_to_abslolue(name='absolute_loc6')(net['relative_loc6'])
        net['loc6_flatten'] = Flatten(name='loc6_flatten')(net['absolute_loc6'])
        # net['loc6_flatten'] = Flatten(name='loc6_flatten')(net['relative_loc6'])

    cls_concate_list = []
    loc_concate_list = []
    for i in range(1,output_layers+1):
        cls_concate_list.append(net['cls{}_flatten'.format(i)])
        loc_concate_list.append(net['loc{}_flatten'.format(i)])
    net['cls_concate'] = Concatenate(axis=1, name='cls_concate')(cls_concate_list)
    net['loc_concate'] = Concatenate(axis=1, name='loc_concate')(loc_concate_list)
    net['cls_pred'] = Reshape((-1, num_classes), name='cls_pred')(net['cls_concate'])
    net['cls_pred'] = Activation('sigmoid', name='cls_pred_final')(net['cls_pred'])
    net['loc_pred'] = Reshape((-1, 4), name='loc_pred')(net['loc_concate'])
    return net


def onenet_head(input_tensor = Input(shape=(512, 512, 3)), num_classes=20, prior_prob=0.01, backbone='resnet50'):
    net = Backbone(input_tensor, backbone_name=backbone)
    x = net['o4']
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
    o = ['o3','o2','o1']
    for i in range(len(o)):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4),
                            name='Dconv_{}'.format(o[i]))(x)
        x = BatchNormalization(name='Dconv_{}_bn'.format(o[i]))(x)
        x = Activation('relu', name='Dconv_{}_relu'.format(o[i]))(x)
        x = Add(name='{}_adding'.format(o[i]))([x, Conv2D(num_filters // pow(2, i), (1, 1), strides=1,
                             padding='same', name='for_{}_adding'.format(o[i]))(net[o[i]])])
    # 最终获得128,128,64的特征层
    # cls header
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    y1 = Conv2D(64, 3, padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(5e-4),
                name='cls_conv1')(x)
    y1 = BatchNormalization(name='cls_bn')(y1)
    y1 = Activation('relu', name='cls_relu')(y1)
    y1 = Conv2D(num_classes, 3,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(5e-4),
                             bias_initializer=initializers.Constant(value=bias_value),
                             activation='sigmoid',
                             name='cls_pred')(y1)
    net['cls_pred'] = Reshape((-1, num_classes), name='cls_pred_final')(y1)
    # loc header (128*128*4)
    y2 = Conv2D(64, 3, padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(5e-4),
                name='loc_conv1')(x)
    y2 = BatchNormalization(name='loc_bn')(y2)
    y2 = Activation('relu', name='loc_relu')(y2)
    loc = Conv2D(4, 3,
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(5e-4),
                 activation='relu',
                 name='loc_conv2')(y2)
    absolute_loc = relative_to_abslolue(name='absolute_loc')(loc)
    net['loc_pred'] = Reshape((-1, 4), name='loc_pred')(absolute_loc)
    # net['loc_pred'] = Reshape((-1, 4), name='loc_pred')(loc)

    return net
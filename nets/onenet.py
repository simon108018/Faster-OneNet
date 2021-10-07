import tensorflow as tf
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input, Lambda,
                                     MaxPooling2D, Reshape, ZeroPadding2D, Concatenate)
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from nets.onenet_loss import MinCostMatcher, Focal_loss, Giou_loss, Loc_loss
from nets.resnet import ResNet18, ResNet50, onenet_head, SSD_OneNet


def onenet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", prior_prob=0.01, alpha=0.25, gamma=2.0, num_stacks=2):
    assert backbone in ['resnet18', 'resnet50']
    image_input = Input(shape=input_shape, name="image_input")
    cls_input = Input(shape=(max_objects, num_classes), name='cls_input')
    loc_input = Input(shape=(max_objects, 4), name='loc_input')
    reg_mask_input = Input(shape=(max_objects,), name='res_mask_input')

    if backbone == 'resnet18':
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
    cls1, loc1, loc_dir1, cls2, loc2, loc_dir2, cls3, loc3, loc_dir3 = SSD_OneNet(C5, num_classes, prior_prob)

    if mode == "train":
        # label assignment
        matcher1 = MinCostMatcher(alpha, gamma, name='min_cost_matcher1')([cls1, loc_dir1, cls_input, loc_input, reg_mask_input])
        matcher2 = MinCostMatcher(alpha, gamma, name='min_cost_matcher2')([cls2, loc_dir2, cls_input, loc_input, reg_mask_input])
        matcher3 = MinCostMatcher(alpha, gamma, name='min_cost_matcher3')([cls3, loc_dir3, cls_input, loc_input, reg_mask_input])
        # training loss
        cls_cost1 = Focal_loss(alpha, gamma, name='cls1')([cls1, cls_input, reg_mask_input, matcher1])
        reg_cost1 = Loc_loss(name='loc1')([loc_dir1, loc_input, reg_mask_input, matcher1])
        giou_cost1 = Giou_loss(name='giou1')([loc_dir1, loc_input, reg_mask_input, matcher1])
        cls_cost2 = Focal_loss(alpha, gamma, name='cls2')([cls2, cls_input, reg_mask_input, matcher2])
        reg_cost2= Loc_loss(name='loc2')([loc_dir2, loc_input, reg_mask_input, matcher2])
        giou_cost2 = Giou_loss(name='giou2')([loc_dir2, loc_input, reg_mask_input, matcher2])
        cls_cost3 = Focal_loss(alpha, gamma, name='cls3')([cls3, cls_input, reg_mask_input, matcher3])
        reg_cost3 = Loc_loss(name='loc3')([loc_dir3, loc_input, reg_mask_input, matcher3])
        giou_cost3 = Giou_loss(name='giou3')([loc_dir3, loc_input, reg_mask_input, matcher3])
        model = Model(inputs=[image_input, cls_input, loc_input, reg_mask_input],
                      outputs=[cls_cost1, reg_cost1, giou_cost1, cls_cost2, reg_cost2, giou_cost2, cls_cost3, reg_cost3, giou_cost3])
        return model
    else:
        if "1" in mode and "2" not in mode and "3" not in mode:
            prediction_model = Model(inputs=image_input, outputs=[cls1, loc1])
        elif "2" in mode and "1" not in mode and "3" not in mode:
            prediction_model = Model(inputs=image_input, outputs=[cls2, loc2])
        elif "3" in mode and "1" not in mode and "2" not in mode:
            prediction_model = Model(inputs=image_input, outputs=[cls3, loc3])
        elif "1" in mode and "2" in mode and "3" not in mode:
            prediction_model = Model(inputs=image_input, outputs=[cls1, loc1, cls2, loc2])
        elif "1" in mode and "3" in mode and "2" not in mode:
            prediction_model = Model(inputs=image_input, outputs=[cls1, loc1, cls3, loc3])
        elif "2" in mode and "3" in mode and "1" not in mode:
            prediction_model = Model(inputs=image_input, outputs=[cls2, loc2, cls3, loc3])
        else:
            prediction_model = Model(inputs=image_input, outputs=[cls1, loc1, cls2, loc2, cls3, loc3])

        return prediction_model
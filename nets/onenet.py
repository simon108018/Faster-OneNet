import tensorflow as tf
from tensorflow.keras.layers import (InputSpec, Layer, Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input, Lambda,
                                     MaxPooling2D, Reshape, ZeroPadding2D, Concatenate)
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from nets.onenet_loss import MinCostMatcher, Focal_loss, Giou_loss, Loc_loss
from nets.resnet import ResNet18, ResNet50, onenet_head, SSD_OneNet


class decode(Layer):
    def __init__(self, in_shape, max_objects=100, scale=1., name=None, **kwargs):
        super(decode, self).__init__(name=name, **kwargs)
        self.in_shape = in_shape
        self.max_objects = max_objects
        self.scale = scale
        length = in_shape[0]/self.scale * in_shape[1]/self.scale
        batch_idx = tf.expand_dims(tf.range(0, 1), 1)
        batch_idx = tf.tile(batch_idx, (1, self.max_objects))
        self.batch_idx = tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32)
    def get_config(self):
        config = super(decode, self).get_config()
        return config

    def call(self, cls_pred, loc_pred, **kwargs):
        scores, indices, class_ids, xs, ys = self.topk(cls_pred)
        b = tf.shape(cls_pred)[0]

        # -----------------------------------------------------#
        #   loc         b, 128 * 128, 4
        # -----------------------------------------------------#
        loc_pred = tf.reshape(loc_pred, [b, -1, 4]) * self.scale
        # length = tf.shape(loc_pred)[1]

        # -----------------------------------------------------#
        #   找到其在1维上的索引
        #   batch_idx   b, max_objects
        # -----------------------------------------------------#
        full_indices = self.batch_idx + tf.reshape(indices, [-1])
        # -----------------------------------------------------#
        #   取出top_k个框对应的参数
        # -----------------------------------------------------#
        topk_loc = tf.gather(tf.reshape(loc_pred, [-1, 4]), full_indices)
        topk_loc = tf.reshape(topk_loc, [b, -1, 4])

        # -----------------------------------------------------#
        #   计算预测框左上角和右下角
        #   topk_x1     b,k,1       预测框左上角x轴坐标
        #   topk_y1     b,k,1       预测框左上角y轴坐标
        #   topk_x2     b,k,1       预测框右下角x轴坐标
        #   topk_y2     b,k,1       预测框右下角y轴坐标
        # -----------------------------------------------------#
        topk_x1, topk_y1 = topk_loc[..., 0:1], topk_loc[..., 1:2]
        topk_x2, topk_y2 = topk_loc[..., 2:3], topk_loc[..., 3:4]
        # -----------------------------------------------------#
        #   scores      b,k,1       预测框得分
        #   class_ids   b,k,1       预测框种类
        # -----------------------------------------------------#
        scores = tf.expand_dims(scores, axis=-1)
        class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)

        # -----------------------------------------------------#
        #   detections  预测框所有参数的堆叠
        #   前四个是预测框的坐标，后两个是预测框的得分与种类
        # -----------------------------------------------------#
        detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

        return detections

    def topk(self, cls_pred):
        # -------------------------------------------------------------------------#
        #   利用512x512x3圖片進行coco數據集預測的時候
        #   h , w = 輸出的長寬 , num_classes = 20
        #   找出得分最大的特徵點
        # -------------------------------------------------------------------------#
        b, h, w, c = tf.shape(cls_pred)[0], tf.shape(cls_pred)[1], tf.shape(cls_pred)[2], tf.shape(cls_pred)[3]
        # -------------------------------------------#
        #   将所有结果平铺，获得(b, w * h * c)
        # -------------------------------------------#
        cls_pred = tf.reshape(cls_pred, (1, -1))
        # -----------------------------#
        #   (b, k), (b, k)
        # -----------------------------#
        scores, indices = tf.math.top_k(cls_pred, k=self.max_objects, sorted=False)

        # --------------------------------------#
        #   計算求出種類、網格點以及索引。
        # --------------------------------------#
        class_ids = indices % c
        xs = indices // c % w
        ys = indices // c // w
        indices = ys * w + xs
        return scores, indices, class_ids, xs, ys

    def get_config(self):
        config = super(decode, self).get_config()
        config.update({'alpha': self.alpha,
                       'gamma': self.gamma,
                       'scale': self.scale,
                       'batch_idx': self.batch_idx})
        return config





def onenet(input_shape, num_classes, structure='onenet', backbone='resnet50',
           max_objects=100, mode="train", shortcut=True, prior_prob=0.01, alpha=0.25, gamma=2.0):
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
    if structure=='onenet':
        X = onenet_head(C5, num_classes, prior_prob, mode=mode)
    elif structure=='SSD_OneNet':
        X = SSD_OneNet(C5, num_classes, prior_prob, shortcut=shortcut, mode=mode)
    if "train" in mode:
        output_list = []
        if '1' in mode or 'all' in mode:
            cls1, loc1, loc_dir1 = X[:3]
            del X[:3]
            # label assignment
            matcher1 = MinCostMatcher(alpha, gamma, name='min_cost_matcher1')(
                [cls1, loc_dir1, cls_input, loc_input, reg_mask_input])
            # training loss
            cls_cost1 = Focal_loss(alpha, gamma, name='cls1')([cls1, cls_input, reg_mask_input, matcher1])
            reg_cost1 = Loc_loss(name='loc1')([loc_dir1, loc_input, reg_mask_input, matcher1])
            giou_cost1 = Giou_loss(name='giou1')([loc_dir1, loc_input, reg_mask_input, matcher1])
            output_list.extend([cls_cost1, reg_cost1, giou_cost1])
        if '2' in mode or 'all' in mode:
            cls2, loc2, loc_dir2 = X[:3]
            del X[:3]
            # label assignment
            matcher2 = MinCostMatcher(alpha, gamma, name='min_cost_matcher2')(
                [cls2, loc_dir2, cls_input, loc_input, reg_mask_input])
            # training loss
            cls_cost2 = Focal_loss(alpha, gamma, name='cls2')([cls2, cls_input, reg_mask_input, matcher2])
            reg_cost2 = Loc_loss(name='loc2')([loc_dir2, loc_input, reg_mask_input, matcher2])
            giou_cost2 = Giou_loss(name='giou2')([loc_dir2, loc_input, reg_mask_input, matcher2])
            output_list.extend([cls_cost2, reg_cost2, giou_cost2])
        if '3' in mode or 'all' in mode:
            cls3, loc3, loc_dir3 = X[:3]
            del X[:3]
            # label assignment
            matcher3 = MinCostMatcher(alpha, gamma, name='min_cost_matcher3')(
                [cls3, loc_dir3, cls_input, loc_input, reg_mask_input])
            # training loss
            cls_cost3 = Focal_loss(alpha, gamma, name='cls3')([cls3, cls_input, reg_mask_input, matcher3])
            reg_cost3 = Loc_loss(name='loc3')([loc_dir3, loc_input, reg_mask_input, matcher3])
            giou_cost3 = Giou_loss(name='giou3')([loc_dir3, loc_input, reg_mask_input, matcher3])
            output_list.extend([cls_cost3, reg_cost3, giou_cost3])
        if '4' in mode or 'all' in mode:
            cls4, loc4, loc_dir4 = X[:3]
            del X[:3]
            # label assignment
            matcher4 = MinCostMatcher(alpha, gamma, name='min_cost_matcher4')(
                [cls4, loc_dir4, cls_input, loc_input, reg_mask_input])
            # training loss
            cls_cost4 = Focal_loss(alpha, gamma, name='cls4')([cls4, cls_input, reg_mask_input, matcher4])
            reg_cost4 = Loc_loss(name='loc4')([loc_dir4, loc_input, reg_mask_input, matcher4])
            giou_cost4 = Giou_loss(name='giou4')([loc_dir4, loc_input, reg_mask_input, matcher4])
            output_list.extend([cls_cost4, reg_cost4, giou_cost4])
        if '5' in mode or 'all' in mode:
            cls5, loc5, loc_dir5 = X[:3]
            del X
            # label assignment
            matcher5 = MinCostMatcher(alpha, gamma, name='min_cost_matcher5')(
                [cls5, loc_dir5, cls_input, loc_input, reg_mask_input])
            # training loss
            cls_cost5 = Focal_loss(alpha, gamma, name='cls5')([cls5, cls_input, reg_mask_input, matcher5])
            reg_cost5 = Loc_loss(name='loc5')([loc_dir5, loc_input, reg_mask_input, matcher5])
            giou_cost5 = Giou_loss(name='giou5')([loc_dir5, loc_input, reg_mask_input, matcher5])
            output_list.extend([cls_cost5, reg_cost5, giou_cost5])
        model = Model(inputs=[image_input, cls_input, loc_input, reg_mask_input],
                      outputs=output_list)
        return model

    else:
        detections_list = []
        if "1" in mode or 'all' in mode:
            cls1, loc1, loc_dir1= X[:3]
            del X[:3]
            detections1 = decode(in_shape=input_shape, max_objects=max_objects, scale=300./38., name='detections1')(cls1, loc_dir1)
            detections_list.append(detections1)
        if "2" in mode or 'all' in mode:
            cls2, loc2, loc_dir2= X[:3]
            del X[:3]
            detections2 = decode(in_shape=input_shape, max_objects=max_objects, scale=300./19., name='detections2')(cls2, loc_dir2)
            detections_list.append(detections2)
        if "3" in mode or 'all' in mode:
            cls3, loc3, loc_dir3= X[:3]
            del X[:3]
            detections3 = decode(in_shape=input_shape, max_objects=max_objects, scale=300./10., name='detections3')(cls3, loc_dir3)
            detections_list.append(detections3)
        if "4" in mode or 'all' in mode:
            cls4, loc4, loc_dir4= X[:3]
            del X[:3]
            detections4 = decode(in_shape=input_shape, max_objects=max_objects, scale=300./5., name='detections4')(cls4, loc_dir4)
            detections_list.append(detections4)
        if "5" in mode or 'all' in mode:
            cls5, loc5, loc_dir5= X
            del X
            detections5 = decode(in_shape=input_shape, max_objects=max_objects, scale=300./3., name='detections5')(cls5, loc_dir5)
            detections_list.append(detections5)


        detections = Concatenate(axis=1, name='detections')(detections_list)
        prediction_model = Model(inputs=image_input, outputs=detections)
        return prediction_model
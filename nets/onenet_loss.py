import tensorflow as tf
import tensorflow_addons as tfa

def focal_loss(cls_pred, cls_true, alpha = 0.25, gamma = 2.0):
    #   cls_true：類別真實值          (batch_size, max_objects, num_classes)
    #   cls_pred：類別預測值          (batch_size, 128*128, num_classes)
    #   將兩個值expand_dims到         (batch_size, max_objects, 128*128, num_classes)

    max_objects = tf.shape(cls_true)[1]
    w_x_h = tf.shape(cls_pred)[1]
    cls_pred = tf.expand_dims(cls_pred, 1)
    cls_pred = tf.tile(cls_pred, (1, max_objects, 1, 1))
    cls_true = tf.expand_dims(cls_true, 2)
    cls_true = tf.tile(cls_true, (1, 1, w_x_h, 1))
    # (batch_size, max_objects, 128*128, num_classes)
    cls_true = tf.cast(tf.equal(cls_true, 1), tf.float32)
    # 為了節省記憶體，neg_mask 使用1-pos_mask
    # neg_mask = tf.cast(tf.equal(cls_true, 0), tf.float32)

    # (batch_size, max_objects, 128*128, num_classes)
    cls_loss = tfa.losses.sigmoid_focal_crossentropy(cls_true, cls_pred, alpha=alpha, gamma=gamma, from_logits=False)
    cls_loss_for_matcher = tf.reduce_mean(
        (- alpha * tf.pow(1 - cls_pred, gamma) * tf.math.log(tf.clip_by_value(cls_pred, 1e-6, 1.)) * cls_true \
         -(1 - alpha) * tf.pow(cls_pred, gamma) * tf.math.log(tf.clip_by_value(1 - cls_pred, 1e-6, 1.)) * (1.-cls_true)),
        axis=-1
    )
    return [cls_loss_for_matcher, cls_loss]


def reg_l1_loss(y_pred, y_true):
    #   y_pred：位置預測值          (batch_size, 128*128, 4)
    #   y_true：位置真實值          (batch_size, max_objects, 4)
    #-------------------------------------------------------------------------#
    #   將兩個值expand_dims到         (batch_size, max_objects, 128*128, 4)
    #   計算後                       (batch_size, max_objects, 128*128)
    #-------------------------------------------------------------------------#
    w_x_h = tf.shape(y_pred)[1]
    max_objects = tf.shape(y_true)[1]
    y_pred = tf.expand_dims(y_pred, 1)
    y_pred = tf.tile(y_pred, (1, max_objects, 1, 1))
    y_true = tf.expand_dims(y_true, 2)
    y_true = tf.tile(y_true, (1, 1, w_x_h, 1))
    reg_loss = tf.reduce_sum(tf.abs(y_true - y_pred), axis=-1)

    return reg_loss


def GIOU(y_pred, y_true):
    #   y_pred：位置預測值          (batch_size, 128*128, 4)
    #   y_true：位置真實值          (batch_size, max_objects, 4)
    #-------------------------------------------------------------------------#
    #   將兩個值expand_dims到         (batch_size, max_objects, 128*128, 4)
    #   計算後                       (batch_size, max_objects, 128*128)
    #-------------------------------------------------------------------------#
    w_x_h = tf.shape(y_pred)[1]
    max_objects = tf.shape(y_true)[1]
    y_pred = tf.expand_dims(y_pred, 1)
    y_pred = tf.tile(y_pred, (1, max_objects, 1, 1))
    y_true = tf.expand_dims(y_true, 2)
    y_true = tf.tile(y_true, (1, 1, w_x_h, 1))

    giou_loss = tfa.losses.giou_loss(y_pred, y_true)

    return giou_loss


def MinCostMatcher(total_loss):
    # -----------------------------------------------------------------------------------------------------------------#
    #   total_loss：每個對應位置的loss(batch_size, max_objects, 128*128)
    #   reg_mask：真实值的mask        (batch_size, max_objects)
    # -----------------------------------------------------------------------------------------------------------------#
    b, k = tf.shape(total_loss)[0], tf.shape(total_loss)[1]
    #  利用tf.argmin找出最match的框的位置
    argmin_total = tf.expand_dims(tf.cast(tf.argmin(total_loss, axis=-1), tf.int32), -1)
    grid = tf.meshgrid(tf.range(0, b), tf.range(0, k))
    grid = tf.transpose(grid)
    indices = tf.reshape(tf.concat((grid, argmin_total), -1), (b, k, 3))

    return indices


def loss(args):
    #-----------------------------------------------------------------------------------------------------------------#
    #   cls_pred：類別預測值          (batch_size, 128, 128, num_classes) --> (batch_size, 128*128, num_classes)
    #   loc_pred：位置預測值          (batch_size, 128, 128, 4)-------------> (batch_size, 128*128, 4)
    #   cls_true：類別真實值          (batch_size, max_objects, num_classes)
    #   loc_true：位置真實值          (batch_size, max_objects, 4)
    #   reg_mask：真实值的mask        (batch_size, max_objects)
    #   indices：真实值对应的坐标     (batch_size, max_objects) 回傳值 [0, 128*128)
    #   total_loss：每個對應位置的loss (batch_size, max_objects, 128*128)
    #-----------------------------------------------------------------------------------------------------------------#
    cls_pred, loc_pred, cls_true, loc_true, reg_mask = args
    b, w, h, c = tf.shape(cls_pred)[0], tf.shape(cls_pred)[1], tf.shape(cls_pred)[2], tf.shape(cls_pred)[3]
    cls_pred = tf.reshape(cls_pred, (b, w*h, c))
    loc_pred = tf.reshape(loc_pred, (b, w*h, 4))
    loc_pred = tf.divide(loc_pred, [w, h, w, h])
    # loc_true = tf.divide(loc_true, [w, h, w, h])

    # 各種loss計算
    num_box = tf.cast(tf.reduce_sum(reg_mask), tf.float32)
    cls_losses = focal_loss(cls_pred, cls_true) #cls_losses : [cls_loss_for_matcher, cls_loss]
    loc_loss = reg_l1_loss(loc_pred, loc_true)
    giou_loss = GIOU(loc_pred, loc_true)
    # cls_loss[0] == cls_loss_for_matcher
    total_loss = 2. * cls_losses[0] + 5. * loc_loss + 2. * giou_loss

    indices = MinCostMatcher(total_loss)

    # 從位置計算各個loss
    # cls_losses[1] : cls_loss
    min_cls_loss = tf.gather_nd(cls_losses[1], indices) * reg_mask
    min_loc_loss = tf.gather_nd(loc_loss, indices) * reg_mask
    min_giou_loss = tf.gather_nd(giou_loss, indices) * reg_mask

    # reg_mask的用途在於有些objects 不存在，會有多算的loss

    mean_min_cls = tf.cond(tf.equal(num_box, 0.), lambda: 0., lambda: tf.reduce_sum(min_cls_loss) / num_box)
    mean_min_loc = tf.cond(tf.equal(num_box, 0.), lambda: 0., lambda: tf.reduce_sum(min_loc_loss) / num_box)
    mean_min_giou = tf.cond(tf.equal(num_box, 0.), lambda: 0., lambda: tf.reduce_sum(min_giou_loss) / num_box)

    return mean_min_cls, mean_min_loc, mean_min_giou

@tf.function
def cls(mean_min_cls):
    return mean_min_cls

@tf.function
def loc(mean_min_loc):
    return mean_min_loc

@tf.function
def giou(mean_min_giou):
    return mean_min_giou




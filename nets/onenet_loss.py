import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class MinCostMatcher(Layer):
    #     # -----------------------------------------------------------------------------------------------------------------#
    #     #   total_loss：每個對應位置的loss(batch_size, max_objects, 128*128)
    #     #   reg_mask：真实值的mask        (batch_size, max_objects)
    #     # -----------------------------------------------------------------------------------------------------------------#
    def __init__(self, alpha=0.25, gamma=2.0, name=None, **kwargs):
        super(MinCostMatcher, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, args, **kwargs):
        cls_pred, loc_pred, cls_true, loc_true, reg_mask = args
        b, w, h, c = tf.shape(cls_pred)[0], tf.shape(cls_pred)[1], tf.shape(cls_pred)[2], tf.shape(cls_pred)[3]
        m = tf.shape(cls_true)[1]
        cls_true = tf.cast(tf.equal(cls_true, 1), tf.float32)
        cls_pred = tf.reshape(cls_pred, (b, w * h, c))
        loc_pred = tf.reshape(loc_pred, (b, w * h, 4))
        loc_pred = tf.divide(loc_pred, [w, h, w, h])

        # cls

        cls_prob = tf.expand_dims(cls_pred, 1)
        cls_true_ = tf.expand_dims(cls_true, 2)
        neg_cost_class = (1 - self.alpha) * (cls_prob ** self.gamma) * (-tf.math.log(1 - cls_prob + 1e-8))
        pos_cost_class = self.alpha * ((1 - cls_prob) ** self.gamma) * (-tf.math.log(cls_prob + 1e-8))
        cls_loss = tf.reduce_sum((pos_cost_class - neg_cost_class) * cls_true_, axis=-1)
        # loc
        loc_pred_ = tf.expand_dims(loc_pred, 1)
        loc_true_ = tf.expand_dims(loc_true, 2)
        reg_loss = tf.reduce_sum(tf.abs(tf.subtract(loc_true_, loc_pred_)), axis=-1)
        giou_loss = tfa.losses.giou_loss(loc_pred_, loc_true_)
        total_loss = 2. * cls_loss + 5. * reg_loss + 2. * giou_loss

        #  利用tf.argmin找出最match的框的位置
        argmin_total = tf.expand_dims(tf.cast(tf.argmin(total_loss, axis=-1), tf.int32), -1)
        batch = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, b), 1), (1, m)), -1)
        cls_id = tf.expand_dims(tf.cast(tf.argmax(cls_true, axis=-1), tf.int32), -1)
        indices = tf.concat((batch, argmin_total, cls_id), -1)

        return indices

    def get_config(self):
        config = super(MinCostMatcher, self).get_config()
        config.update({'alpha': self.alpha,
                       'gamma': self.gamma})
        return config


class Focal_loss(Layer):
    def __init__(self, alpha=0.25, gamma=2.0, name=None, **kwargs):
        super(Focal_loss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, args, **kwargs):
        cls_pred, cls_true, reg_mask, indices = args
        b, w, h, c = tf.shape(cls_pred)[0], tf.shape(cls_pred)[1], tf.shape(cls_pred)[2], tf.shape(cls_pred)[3]
        cls_pred = tf.reshape(cls_pred, (b, w * h, c))
        num_box = tf.cast(tf.reduce_sum(reg_mask), tf.float32)
        scatter = tf.scatter_nd(indices=indices, updates=reg_mask, shape=[b, w * h, c])
        labels = tf.cast(tf.greater(scatter, 0), tf.float32)
        cls_loss = self.sigmoid_focal_loss(cls_pred, labels, alpha=self.alpha, gamma=self.gamma,
                                           reduction='sum') / num_box

        return cls_loss

    def sigmoid_focal_loss(self,
                           inputs: tf.Tensor,
                           targets: tf.Tensor,
                           alpha: float = -1,
                           gamma: float = 2,
                           reduction: str = "none",
                           ) -> tf.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        # ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        ce_loss = K.binary_crossentropy(target=targets, output=inputs, from_logits=False)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = tf.reduce_mean(loss)
        elif reduction == "sum":
            loss = tf.reduce_sum(loss)

        return loss

    def get_config(self):
        config = super(Focal_loss, self).get_config()
        config.update({'alpha': self.alpha,
                       'gamma': self.gamma})
        return config


class Giou_loss(Layer):
    def __init__(self, name=None, **kwargs):
        super(Giou_loss, self).__init__(name=name, **kwargs)

    def call(self, args, **kwargs):
        loc_pred, loc_true, reg_mask, indices = args
        b, w, h = tf.shape(loc_pred)[0], tf.shape(loc_pred)[1], tf.shape(loc_pred)[2]
        loc_pred = tf.reshape(loc_pred, (b, w * h, 4))
        loc_pred = tf.divide(loc_pred, [w, h, w, h])
        num_box = tf.cast(tf.reduce_sum(reg_mask), tf.float32)
        loc_pred_ = tf.gather_nd(params=loc_pred, indices=indices[:, :, :-1])
        giou_loss = tf.reduce_sum(tfa.losses.giou_loss(loc_pred_, loc_true) * reg_mask) / num_box
        return giou_loss

    def get_config(self):
        config = super(Giou_loss, self).get_config()
        return config


class Loc_loss(Layer):
    def __init__(self, name=None, **kwargs):
        super(Loc_loss, self).__init__(name=name, **kwargs)

    def call(self, args, **kwargs):
        loc_pred, loc_true, reg_mask, indices = args
        b, w, h = tf.shape(loc_pred)[0], tf.shape(loc_pred)[1], tf.shape(loc_pred)[2]
        loc_pred = tf.reshape(loc_pred, (b, w * h, 4))
        loc_pred = tf.divide(loc_pred, [w, h, w, h])
        num_box = tf.cast(tf.reduce_sum(reg_mask), tf.float32)
        loc_pred_ = tf.gather_nd(params=loc_pred, indices=indices[:, :, :-1])
        reg_loss = tf.reduce_sum(tf.abs(tf.subtract(loc_true, loc_pred_)) * tf.expand_dims(reg_mask, -1)) / num_box

        return reg_loss

    def get_config(self):
        config = super(Loc_loss, self).get_config()
        return config
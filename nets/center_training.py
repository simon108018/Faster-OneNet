import math
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from utils.utils import draw_gaussian, gaussian_radius


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


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
    pos_mask = tf.cast(tf.equal(cls_true, 1), tf.float32)

    # (batch_size, max_objects, 128*128, num_classes)
    neg_loss = -(1 - alpha) * tf.pow(cls_pred, gamma) * tf.math.log(tf.clip_by_value(1 - cls_pred, 1e-6, 1.)) * pos_mask
    pos_loss = -alpha * tf.pow(1 - cls_pred, gamma) * tf.math.log(tf.clip_by_value(cls_pred, 1e-6, 1.)) * pos_mask
    cls_loss = pos_loss + neg_loss

    # (batch_size, max_objects, 128*128)
    # 用reduce_sum是因為最後一個維度上的值，只有一個值不為0，此值即為我們想計算的物種的focal loss
    # 而其餘的值因為皆為0，所以可以直接使用tf.reduce_sum計算降維度並計算focal loss
    cls_loss = tf.reduce_sum(cls_loss, axis=-1)

    return cls_loss


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


def MinCost(total_loss, cls_loss, loc_loss, giou_loss, reg_mask):
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

    # 從位置計算各個loss
    min_cls_loss = tf.gather_nd(cls_loss, indices) * reg_mask
    min_loc_loss = tf.gather_nd(loc_loss, indices) * reg_mask
    min_giou_loss = tf.gather_nd(giou_loss, indices) * reg_mask

    # reg_mask的用途在於有些objects 不存在，會有多算的loss

    return [min_cls_loss, min_loc_loss, min_giou_loss]


def loss(args): # minicost
    #-----------------------------------------------------------------------------------------------------------------#
    #   cls_pred：類別預測值          (batch_size, 128, 128, num_classes) --> (batch_size, 128*128, num_classes)
    #   loc_pred：位置預測值          (batch_size, 128, 128, 4)-------------> (batch_size, 128*128, 4)
    #   cls_true：類別真實值          (batch_size, max_objects, num_classes)
    #   loc_true：位置真實值          (batch_size, max_objects, 4)
    #   reg_mask：真实值的mask        (batch_size, max_objects)
    #   indices：真实值对应的坐标     (batch_size, max_objects) 回傳值 [0, 128*128)
    #   total_loss：每個對應位置的loss (batch_size, max_objects, 128*128)
    #-----------------------------------------------------------------------------------------------------------------#
    cls_pred, loc_pred, cls_true, loc_true, reg_mask, indices = args
    b, w, h, c = tf.shape(cls_pred)[0], tf.shape(cls_pred)[1], tf.shape(cls_pred)[2], tf.shape(cls_pred)[3]
    cls_pred = tf.reshape(cls_pred, (b, w*h, c))
    loc_pred = tf.reshape(loc_pred, (b, w*h, 4))
    loc_pred = tf.divide(loc_pred, [w, h, w, h])
    loc_true = tf.divide(loc_true, [w, h, w, h])

    # 各種loss計算
    num_box = tf.cast(tf.reduce_sum(reg_mask), tf.float32)
    cls_loss = focal_loss(cls_pred, cls_true)
    loc_loss = reg_l1_loss(loc_pred, loc_true)
    giou_loss = GIOU(loc_pred, loc_true)
    total_loss = 2. * cls_loss + 5. * loc_loss + 2. * giou_loss

    mincost = MinCost(total_loss, cls_loss, loc_loss, giou_loss, reg_mask)

    mean_min_cls = tf.cond(tf.equal(num_box, 0.), lambda: 0., lambda: tf.reduce_sum(mincost[0]) / num_box)
    mean_min_loc = tf.cond(tf.equal(num_box, 0.), lambda: 0., lambda: tf.reduce_sum(mincost[1]) / num_box)
    mean_min_giou = tf.cond(tf.equal(num_box, 0.), lambda: 0., lambda: tf.reduce_sum(mincost[2]) / num_box)

    return mean_min_cls, mean_min_loc, mean_min_giou

def cls(mean_min_cls):
    return mean_min_cls

def loc(mean_min_loc):
    return mean_min_loc

def giou(mean_min_giou):
    return mean_min_giou

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class Generator(object):
    def __init__(self, batch_size, train_lines, val_lines,
                input_size, num_classes, max_objects=100):
        
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.input_size = input_size
        self.output_size = (int(input_size[0]/4) , int(input_size[1]/4))
        self.num_classes = num_classes
        self.max_objects = max_objects

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255


        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    def generate(self, train=True, eager=False):
        while True:
            if train:
                # 打乱
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
                
            batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]), dtype=np.float32)
            # batch_hms = np.zeros((self.batch_size, self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
            batch_cls = np.zeros((self.batch_size, self.max_objects, self.num_classes), dtype=np.float32)
            # batch_whs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
            # batch_regs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
            batch_loc = np.zeros((self.batch_size, self.max_objects, 4), dtype=np.float32)
            batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)
            batch_indices = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)


            b = 0
            for annotation_line in lines:  
                img,y = self.get_random_data(annotation_line,self.input_size[0:2],random=train)

                if len(y)!=0:
                    boxes = np.array(y[:,:4],dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.input_size[1]*self.output_size[1]
                    boxes[:,1] = boxes[:,1]/self.input_size[0]*self.output_size[0]
                    boxes[:,2] = boxes[:,2]/self.input_size[1]*self.output_size[1]
                    boxes[:,3] = boxes[:,3]/self.input_size[0]*self.output_size[0]

                for i in range(len(y)):
                    bbox = boxes[i].copy()
                    bbox = np.array(bbox)
                    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size[1] - 1)
                    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size[0] - 1)
                    cls_id = int(y[i,-1])
                    
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if h > 0 and w > 0:
                        ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)


                        # 獲得類別
                        batch_cls[b, i, cls_id] = 1.
                        # # 計算真實框的寬高
                        # batch_whs[b, i] = 1. * w, 1. * h
                        # # 计算中心偏移量
                        # batch_regs[b, i] = ct - ct_int
                        # 計算box 4個位置與中心點(ct_int)的相對位置
                        # batch_loc[b, i] = bbox - np.array([ct_int[0], ct_int[1], ct_int[0], ct_int[1]], dtype=np.float32)
                        batch_loc[b, i] = bbox
                        # 将对应的mask设置为1，用于排除多余的0
                        batch_reg_masks[b, i] = 1
                        # 表示第ct_int[1]行的第ct_int[0]个。
                        batch_indices[b, i] = ct_int[1] * self.output_size[0] + ct_int[0]

                # 将RGB转化成BGR
                img = np.array(img,dtype = np.float32)[:,:,::-1]
                batch_images[b] = preprocess_image(img)
                b = b + 1
                if b == self.batch_size:
                    b = 0
                    # if eager:
                    #     yield batch_images, batch_hms,  batch_whs, batch_regs, batch_reg_masks, batch_indices
                    # else:
                    #     yield [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], np.zeros((self.batch_size,))
                    if eager:
                        yield batch_images, batch_cls, batch_loc, batch_reg_masks, batch_indices
                    else:
                        yield [batch_images, batch_cls, batch_loc, batch_reg_masks, batch_indices], np.zeros((self.batch_size,))

                    batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], 3), dtype=np.float32)
                    # batch_hms = np.zeros((self.batch_size, self.output_size[0], self.output_size[1], self.num_classes), dtype=np.float32)
                    batch_cls = np.zeros((self.batch_size, self.max_objects, self.num_classes), dtype=np.float32)
                    # batch_whs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
                    # batch_regs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
                    batch_loc = np.zeros((self.batch_size, self.max_objects, 4), dtype=np.float32)
                    batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)
                    batch_indices = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)


import math
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from utils.utils import draw_gaussian, gaussian_radius


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std
    
# def focal_loss(hm_pred, hm_true):
#     #-------------------------------------------------------------------------#
#     #   找到每张图片的正样本和负样本
#     #   一个真实框对应一个正样本
#     #   除去正样本的特征点，其余为负样本
#     #-------------------------------------------------------------------------#
#     pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
#     #-------------------------------------------------------------------------#
#     #   正样本特征点附近的负样本的权值更小一些
#     #-------------------------------------------------------------------------#
#     neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
#     neg_weights = tf.pow(1 - hm_true, 4)
#
#     #-------------------------------------------------------------------------#
#     #   计算focal loss。难分类样本权重大，易分类样本权重小。
#     #-------------------------------------------------------------------------#
#     pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-6, 1.)) * tf.pow(1 - hm_pred, 2) * pos_mask
#     neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-6, 1.)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask
#
#     num_pos = tf.reduce_sum(pos_mask)
#     pos_loss = tf.reduce_sum(pos_loss)
#     neg_loss = tf.reduce_sum(neg_loss)
#
#     #-------------------------------------------------------------------------#
#     #   进行损失的归一化
#     #-------------------------------------------------------------------------#
#     cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
#     return cls_loss

def focal_loss(cls_pred, cls_true):
    #   cls_true：類別真實值          (batch_size, max_bojects, num_classes)
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    alpha = 0.25
    gamma = 2.0

    cls_true = tf.cast(cls_true, tf.float32)
    # num_pos = tf.reduce_sum(tgt_ids)
    tgt_ids = tf.where(cls_true==1)[:,1]

    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = -alpha * tf.math.log(tf.clip_by_value(cls_pred, 1e-6, 1.)) * tf.pow(1 - cls_pred, gamma)
    neg_loss = -(1 - alpha) * tf.math.log(tf.clip_by_value(1 - cls_pred, 1e-6, 1.)) * tf.pow(cls_pred, gamma)
    pos_loss = tf.gather(pos_loss, axis=-1, indices=tgt_ids)
    neg_loss = tf.gather(neg_loss, axis=-1, indices=tgt_ids)
    cls_loss = pos_loss + neg_loss
    # tf.print(cls_loss)

    # #-------------------------------------------------------------------------#
    # #   进行损失的归一化
    # #-------------------------------------------------------------------------#
    # cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss




# def reg_l1_loss(y_pred, y_true, indices, mask):
#     #-------------------------------------------------------------------------#
#     #   获得batch_size和num_classes
#     #-------------------------------------------------------------------------#
#     b, c = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
#     k = tf.shape(indices)[1]
#
#     y_pred = tf.reshape(y_pred, (b, -1, c))
#     length = tf.shape(y_pred)[1]
#     indices = tf.cast(indices, tf.int32)
#
#     #-------------------------------------------------------------------------#
#     #   利用序号取出预测结果中，和真实框相同的特征点的部分
#     #-------------------------------------------------------------------------#
#     batch_idx = tf.expand_dims(tf.range(0, b), 1)
#     batch_idx = tf.tile(batch_idx, (1, k))
#     full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) +
#                     tf.reshape(indices, [-1]))
#
#     y_pred = tf.gather(tf.reshape(y_pred, [-1,c]),full_indices)
#     y_pred = tf.reshape(y_pred, [b, -1, c])
#
#     mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
#     #-------------------------------------------------------------------------#
#     #   求取l1损失值
#     #-------------------------------------------------------------------------#
#     total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
#     reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
#     return reg_loss

def reg_l1_loss(y_pred, y_true, mask):
    #-------------------------------------------------------------------------#
    #   获得batch_size和num_classes
    #-------------------------------------------------------------------------#
    b, c = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
    h, w = tf.shape(y_pred)[1], tf.shape(y_pred)[2]

    #-------------------------------------------------------------------------#
    #   利用序号取出预测结果中，和真实框相同的特征点的部分
    #-------------------------------------------------------------------------#
    batch


    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, k))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) +
                    tf.reshape(indices, [-1]))

    y_pred = tf.gather(tf.reshape(y_pred, [-1,c]),full_indices)
    y_pred = tf.reshape(y_pred, [b, -1, c])

    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    #-------------------------------------------------------------------------#
    #   求取l1损失值
    #-------------------------------------------------------------------------#
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

def loss(args):
    #-----------------------------------------------------------------------------------------------------------------#
    # #   hm_pred：热力图的预测值       (batch_size, 128, 128, num_classes)
    #   cls_pred：類別預測值          (batch_size, 128, 128, num_classes)
    # #   wh_pred：宽高的预测值         (batch_size, 128, 128, 2)
    # #   reg_pred：中心坐标偏移预测值  (batch_size, 128, 128, 2)
    #   loc_pred：位置預測值          (batch_size, 128, 128, 4)
    # #   hm_true：热力图的真实值       (batch_size, 128, 128, num_classes)
    #   cls_true：類別真實值          (batch_size, max_bojects, num_classes)
    # #   wh_true：宽高的真实值         (batch_size, max_objects, 2)
    # #   reg_true：中心坐标偏移真实值  (batch_size, max_objects, 2)
    #   loc_true：位置真實值          (batch_size, max_bojects, 4)
    #   reg_mask：真实值的mask        (batch_size, max_objects)
    #   indices：真实值对应的坐标     (batch_size, max_objects)
    #-----------------------------------------------------------------------------------------------------------------#
    # hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices = args
    # hm_loss = focal_loss(hm_pred, hm_true)
    # wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    # reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    # total_loss = hm_loss + wh_loss + reg_loss

    cls_pred, loc_pred, cls_true, loc_true, reg_mask, indices = args

    cls_loss = focal_loss(cls_pred, cls_true)
    total_loss = cls_loss
    # wh_loss = 0.1 * reg_l1_loss(loc_pred, loc_true, indices, reg_mask)
    # reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    # total_loss = cls_loss + wh_loss + reg_loss

    # total_loss = tf.Print(total_loss,[hm_loss,wh_loss,reg_loss])
    return total_loss

def loss_sum(args):
    loss, cls_true = args
    tf.print(tf.reduce_sum(cls_true))
    return tf.reduce_sum(loss)/tf.cast(tf.reduce_sum(cls_true),tf.float32)

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class Generator(object):
    def __init__(self,batch_size,train_lines,val_lines,
                input_size,num_classes,max_objects=100):
        
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
        image = image.resize((nw,nh), Image.BICUBIC)

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

                        # # 获得热力图
                        # radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                        # radius = max(0, int(radius))
                        # batch_hms[b, :, :, cls_id] = draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius)
                        # 獲得類別
                        batch_cls[b, i, cls_id] = 1.
                        # # 計算真實框的寬高
                        # batch_whs[b, i] = 1. * w, 1. * h
                        # # 计算中心偏移量
                        # batch_regs[b, i] = ct - ct_int
                        # 計算box位置
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




if __name__=="__main__":

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    def get_classes(classes_path):
        '''loads the classes'''
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    input_shape = [512, 512, 3]
    classes_path = '../model_data/voc_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    backbone = "resnet50"
    annotation_path = '../2007_train.txt'
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    Batch_size = 4
    gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)


    for data in gen.generate(True):
        break

    y_true = np.array([[[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                       [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [0, 0, 0, 0], [0, 0, 0, 0]],
                       [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                       [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype='float32')
    mask = np.array([[1, 1, 0, 0, 0],[1, 1, 1, 0, 0],[1, 1, 0, 0, 0],[1, 1, 0, 0, 0]], dtype='float32')
    y_pred = np.array([[[[3, 6.5, 8, 12.7],[13, 16.5, 28, 32.7],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100]],
                       [[3, 6.5, 8, 12.7],[13, 16.5, 28, 32.7],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100]],
                       [[3, 6.5, 8, 12.7],[13, 16.5, 28, 32.7],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100]],
                       [[3, 6.5, 8, 12.7],[13, 16.5, 28, 32.7],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100]],
                       [[3, 6.5, 8, 12.7],[13, 16.5, 28, 32.7],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100]],
                       [[3, 6.5, 8, 12.7],[13, 16.5, 28, 32.7],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100],[2, 90, 70, 100]]],
                       [[[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]]],
                       [[[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]]],
                       [[[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]],
                        [[3, 6.5, 8, 12.7], [13, 16.5, 28, 32.7], [2, 90, 70, 100], [2, 90, 70, 100], [2, 90, 70, 100],
                         [2, 90, 70, 100]]]]
                      )
import colorsys
import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
import time


from nets.build_model import build_model
from utils.utils import onenet_correct_boxes, letterbox_image, nms

# import tensorflow_model_optimization as tfmot
# quantize_model = tfmot.quantization.keras.quantize_model
# quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
# quantize_apply = tfmot.quantization.keras.quantize_apply
# quantize_scope = tfmot.quantization.keras.quantize_scope


def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class OneNet(object):
    _defaults = {
        "model_path": 'model_data/ep1500-loss1.156-val_loss1.448.h5',
        "classes_path": 'model_data/voc_classes.txt',
        "structure": 'onenet',
        "backbone": 'resnet18',
        "input_shape": [320, 320, 3],
        "confidence": 0.2,
        "max_objects": 20,
        # backbone为resnet50时建议设置为True
        # backbone为hourglass时建议设置为False
        # 也可以根据检测效果自行选择
        "nms": True,
        "nms_threhold": 0.4,
        "use_quantization": False,
        "letterbox_image":True
    }


    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化onenet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        self.num_classes = len(self.class_names)
        self.model = build_model(self.input_shape,
                                 num_classes=self.num_classes,
                                 structure=self.structure,
                                 backbone=self.backbone,
                                 max_objects=self.max_objects,
                                 mode='predict')
        # self.model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
        # print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
    # def generate(self):
    #     if '.tflite' in self.model_path:
    #         self.use_quantization = True
    #
    #     elif '.h5' in self.model_path:
    #         self.use_quantization = False
    #
    #     model_path = os.path.expanduser(self.model_path)
    #     if self.use_quantization:
    #         assert model_path.endswith('.tflite'), 'tflite model or weights must be a .tflite file.'
    #     else:
    #         assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
    #     # ----------------------------------------#
    #     #   计算种类数量
    #     # ----------------------------------------#
    #     self.num_classes = len(self.class_names)
    #     # ----------------------------------------#
    #     #   创建onenet模型
    #     # ----------------------------------------#
    #     if self.use_quantization:
    #         self.interpreter = tf.lite.Interpreter(model_path=model_path)
    #         self.interpreter.allocate_tensors()
    #         self.input_details = self.interpreter.get_input_details()[0]
    #         self.output_details = self.interpreter.get_output_details()
    #         print('{} model, anchors, and classes loaded.'.format(self.model_path))
    #
    #     else:
    #         self.model = onenet(self.input_shape, num_classes=self.num_classes, structure=self.structure, backbone=self.backbone,
    #                              shortcut=self.shortcut, max_objects=self.max_objects, mode='only_output_'+self.mode)
    #         self.model.load_weights(self.model_path, by_name=True, skip_mismatch=True)
    #         print('{} model, anchors, and classes loaded.'.format(self.model_path))
    #         if self.use_quantization:
    #             self.model = quantize_model(self.model)
    #
    #     # 画框设置不同的颜色
    #     hsv_tuples = [(x / len(self.class_names), 1., 1.)
    #                   for x in range(len(self.class_names))]
    #     self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    #     self.colors = list(
    #         map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
    #             self.colors))

    def topk(self, cls_pred, max_objects=100):
        # -------------------------------------------------------------------------#
        #   当利用512x512x3图片进行coco数据集预测的时候
        #   h = w = 128 num_classes = 80
        #   Hot map热力图 -> b, 128, 128, 20
        #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
        #   找出一定区域内，得分最大的特征点。
        # -------------------------------------------------------------------------#
        b, length, c = cls_pred.shape
        # cls_pred = nms(cls_pred)
        # -------------------------------------------#
        #   将所有结果平铺，获得(b, 128 * 128 * 20)
        # -------------------------------------------#
        cls_pred = np.reshape(cls_pred, (cls_pred.shape[0], -1))

        # -----------------------------#
        #   scores.shape = (b, k), indices.shape = (b, k)
        # -----------------------------#
        indices = cls_pred.argsort()[:, -max_objects:][:, ::-1]
        scores = np.take_along_axis(cls_pred, indices, axis=-1)
        # --------------------------------------#
        #   计算求出种类、网格点以及索引。
        # --------------------------------------#
        #   這裡的 indices 包含了種類，值的range 在[0, 128 * 128 * 20)
        class_ids = indices % c
        indices = indices // c
        #   這裡的 indices 已不包含種類，只計算位置，值的range 在[0, 128 * 128)
        return scores, indices, class_ids



    def decode(self, predictions):
        cls_pred = predictions[:, :, :-4]
        loc_pred = predictions[:, :, -4:]
        scores, indices, class_ids= self.topk(cls_pred, max_objects=self.max_objects)
        # b = cls_pred.shape[0]
        # loc_pred = loc_pred.reshape([b, -1, 4])
        topk_loc = np.take_along_axis(loc_pred, np.expand_dims(indices, axis=-1), axis=1)
        topk_x1, topk_y1 = topk_loc[..., 0:1], topk_loc[..., 1:2]
        topk_x2, topk_y2 = topk_loc[..., 2:3], topk_loc[..., 3:4]
        scores = np.expand_dims(scores, axis=-1)
        class_ids = np.expand_dims(class_ids, axis=-1).astype('float32')
        # -----------------------------------------------------#
        #   detections  预测框所有参数的堆叠
        #   前四个是预测框的坐标，后两个是预测框的得分与种类
        # -----------------------------------------------------#
        detections = np.concatenate([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
        return detections

    @tf.function
    def get_pred(self, photo):
        preds = self.model(photo, training=False)
        return preds

    # def get_pred(self, photo):
    #     # mode
    #     # 123 --> predict all result
    #     # 1 --> only output first prediction
    #     # 12 --> output first & second prediction
    #     # 2 --> only output second prediction
    #     if self.use_quantization:
    #         if self.input_details['dtype'] == np.uint8:
    #             input_scale, input_zero_point = self.input_details["quantization"]
    #             photo = photo / input_scale + input_zero_point
    #         photo = photo.astype(self.input_details["dtype"])
    #         self.interpreter.set_tensor(self.input_details["index"], photo)
    #         self.interpreter.invoke()
    #         output_cls = self.interpreter.get_tensor(self.output_details[0]["index"])
    #         output_loc = self.interpreter.get_tensor(self.output_details[1]["index"])
    #         output_loc = self.get_directly_loc(output_loc)
    #         # print(output_cls)
    #         preds = self.decode(output_cls, output_loc, max_objects=100)
    #         # print(preds)
    #         return preds
    #     else:
    #         # start = time.time()
    #         model_preds = self.model(photo, training=False)
    #         for i in range(len(model_preds)):
    #             if (i % 2 == 1):
    #                 model_preds[i] = self.get_directly_loc(model_preds[i].numpy())
    #
    #
    #         # end = time.time()
    #
    #         preds = self.decode(model_preds,
    #                             max_objects=100)
    #         # print('預測時花費了{:.2f}秒'.format(end - start))
    #         return preds

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        image = image.convert('RGB')
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = letterbox_image(image, [self.input_shape[0], self.input_shape[1]])
        else:
            crop_img = image.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
        # ----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的onenet_hourglass权值是使用BGR通道的图片训练的
        # ----------------------------------------------------------------------------------#
        photo = np.array(crop_img, dtype=np.float32)[:,:,::-1]
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        photo = preprocess_image(np.reshape(photo, [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]]))

        preds = self.get_pred(photo).numpy()
        # -------------------------------------------------------#
        #   对于onenet网络来讲，确立中心非常重要。
        #   对于大目标而言，会存在许多的局部信息。
        #   此时对于同一个大目标，中心点比较难以确定。
        #   使用最大池化的非极大抑制方法无法去除局部框
        #   所以我还是写了另外一段对框进行非极大抑制的代码
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        # -------------------------------------------------------#
        # preds = self.decode(preds)

        if self.nms:
            preds = np.array(nms(preds, self.nms_threhold))

        if len(preds[0]) <= 0:
            return image
        # -----------------------------------------------------------#
        #   将预测结果转换成小数的形式
        # -----------------------------------------------------------#
        # preds[0][:, 0:4] = preds[0][:, 0:4] / self.input_shape[0]

        det_label = preds[0][:, -1]
        det_conf = preds[0][:, -2]
        det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

        # -----------------------------------------------------------#
        #   筛选出其中得分高于confidence的框
        # -----------------------------------------------------------#
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), \
                                                 np.expand_dims(det_ymin[top_indices], -1), \
                                                 np.expand_dims(det_xmax[top_indices], -1), \
                                                 np.expand_dims(det_ymax[top_indices], -1)

        # -----------------------------------------------------------#
        #   去掉灰条部分
        # -----------------------------------------------------------#
        if self.letterbox_image:
            boxes = onenet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.input_shape[0],self.input_shape[1]]),image_shape)
        else:
            top_xmin = top_xmin * image_shape[1]
            top_ymin = top_ymin * image_shape[0]
            top_xmax = top_xmax * image_shape[1]
            top_ymax = top_ymax * image_shape[0]
            boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)


        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    def get_FPS(self, image, test_interval):
        image = image.convert('RGB')
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = letterbox_image(image, [self.input_shape[0], self.input_shape[1]])
        else:
            crop_img = image.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
        # ----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        # ----------------------------------------------------------------------------------#
        photo = np.array(crop_img, dtype=np.float32)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        photo = preprocess_image(np.reshape(photo, [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]]))

        preds = self.get_pred(photo).numpy()
        # preds = self.decode(preds)
        if self.nms:
            preds = np.array(nms(preds, self.nms_threhold))
            # preds = np.array(nms(preds, self.nms_threhold))

        if len(preds[0]) > 0:
            # preds[0][:, 0:4] = preds[0][:, 0:4] / self.input_shape[0]

            det_label = preds[0][:, -1]
            det_conf = preds[0][:, -2]
            det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
                det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(
                det_ymax[top_indices], -1)
            if self.letterbox_image:
                boxes = onenet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                             np.array([self.input_shape[0], self.input_shape[1]]), image_shape)
            else:
                top_xmin = top_xmin * image_shape[1]
                top_ymin = top_ymin * image_shape[0]
                top_xmax = top_xmax * image_shape[1]
                top_ymax = top_ymax * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)


        t1 = time.time()
        # tt = 0
        for _ in range(test_interval):
            # t3 = time.time()
            preds = self.get_pred(photo).numpy()
            preds = self.decode(preds)
            # t4 = time.time()
            # tt += t4-t3
            if self.nms:
                preds = np.array(nms(preds, self.nms_threhold))

            if len(preds[0]) > 0:
                # preds[0][:, 0:4] = preds[0][:, 0:4] / self.input_shape[0]

                det_label = preds[0][:, -1]
                det_conf = preds[0][:, -2]
                det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
                    det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(
                    det_ymax[top_indices], -1)
                if self.letterbox_image:
                    boxes = onenet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                                 np.array([self.input_shape[0], self.input_shape[1]]), image_shape)
                else:
                    top_xmin = top_xmin * image_shape[1]
                    top_ymin = top_ymin * image_shape[0]
                    top_xmax = top_xmax * image_shape[1]
                    top_ymax = top_ymax * image_shape[0]
                    boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        # tact_time_ = tt / test_interval
        return tact_time

    def get_FPS2(self, image, test_interval):
        image = image.convert('RGB')
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        # ---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = letterbox_image(image, [self.input_shape[0], self.input_shape[1]])
        else:
            crop_img = image.resize((self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
        # ----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        # ----------------------------------------------------------------------------------#
        photo = np.array(crop_img, dtype=np.float32)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        photo = preprocess_image(np.reshape(photo, [1, self.input_shape[0], self.input_shape[1], 3]))
        preds = self.get_pred(photo).numpy()
        preds = self.decode(preds)
        if self.nms:
            preds = np.array(nms(preds, self.nms_threhold))
            # preds = np.array(nms(preds, self.nms_threhold))

        if len(preds[0]) > 0:

            det_label = preds[0][:, -1]
            det_conf = preds[0][:, -2]
            det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]

            top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
            top_conf = det_conf[top_indices]
            top_label_indices = det_label[top_indices].tolist()
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
                det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(
                det_ymax[top_indices], -1)
            if self.letterbox_image:
                boxes = onenet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                             np.array([self.input_shape[0], self.input_shape[1]]), image_shape)
            else:
                top_xmin = top_xmin * image_shape[1]
                top_ymin = top_ymin * image_shape[0]
                top_xmax = top_xmax * image_shape[1]
                top_ymax = top_ymax * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)


        t1 = time.time()
        for _ in range(test_interval):
            preds = self.get_pred(photo).numpy()


        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
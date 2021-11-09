import numpy as np
import colorsys
import os

from datetime import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
import tensorflow_model_optimization as tfmot
from utils.utils import ModelCheckpoint
from nets.data_generator import Generator
from nets.build_model import onenet
from nets.resnet import apply_ltrb
from nets.model_loss import MinCostMatcher, Focal_loss, Giou_loss, Loc_loss
from tensorflow.keras import Input, Model
import pathlib
from PIL import Image, ImageDraw, ImageFont


# gpu設定
gpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# 處理quantize functions
quantize_model = tfmot.quantization.keras.quantize_model
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_apply = tfmot.quantization.keras.quantize_apply
quantize_scope = tfmot.quantization.keras.quantize_scope


def new_log(logdir):
    list_ = os.listdir(logdir)
    list_.sort(key=lambda fn: os.path.getmtime(logdir + '/' + fn))
    list_ = [l for l in list_ if '.h5' in l]
    # 获取文件所在目录
    if list_:
        newlog = os.path.join(logdir, list_[-1])
    else:
        print(2)
        newlog = None
    return newlog


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
input_shape = [416, 416, 3]
classes_path = 'model_data/coco_classes.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
backbone = "resnet18"
#----------------------------------------------------#
#   获取onenet模型
#----------------------------------------------------#
model = onenet(input_shape, num_classes=num_classes, backbone=backbone, mode='only_output')


if backbone == "resnet50":
    path = './logs50'
elif backbone == "resnet18":
    path = './logs18'

model_path = new_log(path)
model_path = './ep150-loss2.305-val_loss2.279.h5'
if model_path:
    model.load_weights(model_path, by_name=True, skip_mismatch=False)

model.summary()
# model.save(model_path[:-3]+'_tflite.h5')
# base_model = tf.keras.models.load_model(model_path[:-3] + '_tflite.h5', custom_objects={'apply_ltrb':apply_ltrb})
# with quantize_scope({'apply_ltrb':apply_ltrb}):
#     base_model = tf.keras.models.load_model(model_path[:-3]+'_tflite.h5')
# def apply_quantization_to_apply_ltrb(layer):
#     if layer.name == 'pred_location':
#         return layer
#     return tfmot.quantization.keras.quantize_annotate_layer(layer)
#
# annotated_model = tf.keras.models.clone_model(
#     base_model,
#     clone_function=apply_quantization_to_apply_ltrb,
# )
# with quantize_scope({'apply_ltrb':apply_ltrb}):
#     quant_aware_model = quantize_apply(annotated_model)
# quant_aware_model.summary()

def quantile_onenet(quant_aware_model, num_classes, mode = 'train', max_objects=100, alpha=0.25, gamma=2.0):
    assert backbone in ['resnet18', 'resnet50']
    output_size = input_shape[0] // 4
    image_input = quant_aware_model.inputs
    cls_input = Input(shape=(max_objects, num_classes), name='cls_input')
    loc_input = Input(shape=(max_objects, 4), name='loc_input')
    reg_mask_input = Input(shape=(max_objects,), name='res_mask_input')

    quant_cls = quant_aware_model.get_layer("quant_conv2d_4").output
    quant_loc = quant_aware_model.get_layer("quant_conv2d_6").output
    quant_loc_dir = apply_ltrb(name='pred_location')(quant_loc)
    if mode == 'train':
        # label assignment
        matcher = MinCostMatcher(alpha, gamma, name='min_cost_matcher')([quant_cls, quant_loc_dir, cls_input, loc_input, reg_mask_input])
        # training loss
        cls_cost = Focal_loss(alpha, gamma, name='cls')([quant_cls, cls_input, reg_mask_input, matcher])
        reg_cost = Loc_loss(name='loc')([quant_loc_dir, loc_input, reg_mask_input, matcher])
        giou_cost = Giou_loss(name='giou')([quant_loc_dir, loc_input, reg_mask_input, matcher])
        model = Model(inputs=[image_input, cls_input, loc_input, reg_mask_input], outputs=[cls_cost, reg_cost, giou_cost])
        return model
    elif mode == "only_output":
        prediction_model = Model(inputs=image_input, outputs=[quant_cls, quant_loc])
        return prediction_model

quant_aware_model = quantize_model(model)
quant_aware_model.summary()
quant_aware_model = quantile_onenet(quant_aware_model, num_classes, mode='train', max_objects=100, alpha=0.25, gamma=2.0)

annotation_path = '2007_train.txt'
# ----------------------------------------------------------------------#
#   验证集的划分在train.py代码里面进行
#   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
#   当前划分方式下，验证集和训练集的比例为1:9
# ----------------------------------------------------------------------#
val_split = 0.1
with open(annotation_path) as f:
    lines = f.readlines()
np.random.seed(123)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines) * val_split)
num_train = len(lines) - num_val

if backbone == "resnet50":
    logging = TensorBoard(log_dir="logs50")
    checkpoint = ModelCheckpoint('logs50/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
elif backbone == "resnet18":
    logs = "logs18/tfmot/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logging = TensorBoard(log_dir=logs, profile_batch=2, histogram_freq=1)
    checkpoint = ModelCheckpoint('logs18/tfmot/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
else:
    logging = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

Batch_size = 2
Init_Epoch = 0
Epoch = 150
step_num_per_epoch = num_train // Batch_size
step = tf.Variable(num_train//Batch_size * Init_Epoch, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [115*step_num_per_epoch, 140*step_num_per_epoch], [1e-0, 1e-1, 1e-2])
# lr and wd can be a function or a tensor
lr = 5e-4 * schedule(step)
wd = lambda: 1e-4 * schedule(step)
optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
gen = Generator(Batch_size, lines[:num_train], lines[num_train:], input_shape, num_classes)

quant_aware_model.compile(
    loss={'cls': lambda y_true, y_pred: y_pred, 'loc': lambda y_true, y_pred: y_pred, 'giou': lambda y_true, y_pred: y_pred},
    loss_weights=[2, 5, 2],
    optimizer=optimizer)

quant_aware_model.fit(gen.generate(True),
                    steps_per_epoch=num_train//Batch_size,
                    validation_data=gen.generate(False),
                    validation_steps=num_val//Batch_size,
                    epochs=Epoch,
                    verbose=1,
                    initial_epoch=Init_Epoch,
                    callbacks=[logging, checkpoint])
# converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
# converter.experimental_new_converter = True
# converter.allow_custom_ops = True
# converter.target_spec.supported_types = [tf.float16]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_float16_model = converter.convert()
# tflite_models_dir = pathlib.Path("logs18/tflite")
# tflite_models_dir.mkdir(exist_ok=True, parents=True)
# tflite_model_file = tflite_models_dir/'model.tflite'
# tflite_model_file.write_bytes(tflite_float16_model)






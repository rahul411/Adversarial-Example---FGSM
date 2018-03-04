from __future__ import print_function

import argparse
import os
import sys
import time
import math
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.core.framework import node_def_pb2
import numpy as np
from scipy import misc
from skimage import io
import matplotlib.pyplot as plt

from model import PSPNet101, PSPNet50
from tools import *

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150, 
                'model': PSPNet50}
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101}

SAVE_DIR = './output/'
SNAPSHOT_DIR = './model/'

def get_device_setter(gpu_id):
    def device_setter(op):
        _variable_ops = ["Variable", "VariableV2", "VarHandleOp"]
        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        return  '/device:CPU:0' if node_def.op in _variable_ops else '/device:GPU:%d' % gpu_id
    return device_setter

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes'],
                        required=True)

    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def colorlabel(pred):
    colors=np.array(["#0000FF", "#00FF00", "#FF0000",
            "#6600FF", "#66FF00", "#FF0066",
            "#00FFFF", "#00FF66", "#FFFF66",
            "#1100FF", "#00FF11", "#FF1166",
            "#2200FF", "#22FF00", "#FF2266",
            "#3300FF", "#00FF33", "#FF3366","#FF4466"])
    return colors[pred]

def model(img):
    
    param = cityscapes_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # preprocess images
    # img, filename = load_img(args.img_path)
    img_shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
    img = preprocess(img, h, w)

    # Create network.
    net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
   
    raw_output = net.layers['conv6']
    
   
    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    shape = tf.shape(raw_output_up)
   
    return raw_output_up

def adversarial(img_path,label):
    learning_rate = 0.001
    label = np.where(label<19,label,0)
    label = tf.convert_to_tensor(label)
    y = tf.one_hot(label,19)

    img, filename = load_img(img_path)
    img = tf.cast(img,dtype=tf.float32)

    x_hat = tf.Variable(tf.zeros((1024,2048,3),dtype=tf.float32), name='x_hat')
    epsilon = tf.constant(2.0/255.0,dtype=tf.float32)
    assign = tf.assign(x_hat,img)
    #with tf.device('/cpu:0'):
    logits = model(x_hat)
    with tf.device(get_device_setter(0)):
    
        with tf.variable_scope('adversarial') as scope:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.cast(logits,dtype=tf.float32),labels=tf.cast(y,dtype=tf.float32)))
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(loss,var_list=[x_hat])
            gradients = optimizer.apply_gradients(grads)

            below = img - epsilon
            above = img - epsilon
            projected = tf.clip_by_value(tf.clip_by_value(x_hat,below,above),0,255)
            with tf.control_dependencies([projected]):
                projection = tf.assign(x_hat,projected)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)
    print(tf.global_variables())
    restore_var = [v for v in tf.global_variables() if ('x_hat' not in v.name and 'adversarial' not in v.name)]
    loader = tf.train.Saver(var_list=restore_var)
    loader.restore(sess, 'model/model.ckpt-0')

    sess.run(assign)
    maxIter = 200
    for i in range(maxIter):
        adv_image, steploss,_ = sess.run([projection,loss,gradients])
        print(steploss)
    misc.imsave('advImage', adv_image)
    # plt.imshow(adv_image)
    # plt.show()



    
if __name__ == '__main__':
    img_path = 'input/test_1024x2048.png'
    label = io.imread('targetLabel.png')
    adversarial(img_path,label)

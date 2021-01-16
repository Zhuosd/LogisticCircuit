#!usr/bin/python
# -*- coding: utf-8 -*-
import os
from cleverhans.attacks import FastGradientMethod
import numpy as np
from PIL import Image
from scipy.misc import imread
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim
tensorflow_master = ""
checkpoint_path = "inception_v3.ckpt所在的路径"
input_dir = "合法图像的路径"
output_dir = "对抗样本的输出路径"
max_epsilon = 16.0
image_width = 299
image_height = 299
batch_size = 50
import sys

sys.path.append('cleverhans文件夹所在的路径')
eps = 2.0 * max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
num_classes = 1001


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
        with tf.gfile.Open(filepath, "rb") as f:
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float) * 2.0 / 255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')


class InceptionModel(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        probs = output.op.inputs[0]
        print(output)
        return probs


# 加载图片
image_iterator = load_images(input_dir, batch_shape)

# 得到第一个batch的图片
filenames, images = next(image_iterator)
# 日志
tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    # 实例一个model
    model = InceptionModel(num_classes)
    # 开启一个会话
    ses = tf.Session()
    # 对抗攻击开始
    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, ord=np.inf, clip_min=-1., clip_max=1.)

    # fgsm_a是基于L2范数生成的对抗样本
    # fgsm_a = FastGradientMethod(model)
    # x_adv = fgsm_a.generate(x_input, ord=2, clip_min=-1., clip_max=1.)

    # 恢复inception-v3 模型
    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=checkpoint_path,
        master=tensorflow_master)
    for filenames, images in load_images(input_dir, batch_shape):
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            nontargeted_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(nontargeted_images, filenames, output_dir)
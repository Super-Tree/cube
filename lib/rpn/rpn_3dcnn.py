import os
import numpy as np
import tensorflow as tf
from tools.data_visualize import pcd_vispy
from network.config import cfg
from easydict import EasyDict as edict
from dataset.dataset import dataset_train

DEBUG = False

shape = lambda i: int(np.ceil(np.round(cfg.ANCHOR[i] / cfg.CUBIC_RES[i], 3)))  # Be careful about python number  decimal
cubic_size = [shape(0), shape(1), shape(2), 4]


def cubic_rpn_grid_pyfc(lidarPoints, rpnBoxes):
    x_points = lidarPoints[:, 0]
    y_points = lidarPoints[:, 1]
    z_points = lidarPoints[:, 2]
    reflectance = lidarPoints[:, 3]

    res = []
    for box in rpnBoxes:
        x_min = box[1] - float(cfg.ANCHOR[0]) / 2
        x_max = box[1] + float(cfg.ANCHOR[0]) / 2
        y_min = box[2] - float(cfg.ANCHOR[1]) / 2
        y_max = box[2] + float(cfg.ANCHOR[1]) / 2
        z_min = box[3] - float(cfg.ANCHOR[2]) / 2
        z_max = box[3] + float(cfg.ANCHOR[2]) / 2

        f_filt = np.logical_and((x_points > x_min), (x_points < x_max))
        s_filt = np.logical_and((y_points > y_min), (y_points < y_max))
        z_filt = np.logical_and((z_points > z_min), (z_points < z_max))
        fliter = np.logical_and(np.logical_and(f_filt, s_filt), z_filt)
        indice = np.flatnonzero(fliter)
        rpn_points = lidarPoints[indice]
        points_mv_min = np.subtract(rpn_points,
                                    np.array([x_min, y_min, z_min, 0.], dtype=np.float32))  # using fot coordinate
        points_mv_ctr = np.subtract(rpn_points,
                                    np.array([box[1], box[2], box[3], 0.], dtype=np.float32))  # using as feature

        xi = points_mv_min[:, 0]
        yi = points_mv_min[:, 1]
        zi = points_mv_min[:, 2]

        x_cub = np.divide(xi, cfg.CUBIC_RES[0]).astype(np.int32)
        y_cub = np.divide(yi, cfg.CUBIC_RES[1]).astype(np.int32)
        z_cub = np.divide(zi, cfg.CUBIC_RES[2]).astype(np.int32)

        cubic_feature = np.zeros(shape=cubic_size, dtype=np.float32)
        cubic_feature[x_cub, y_cub, z_cub] = points_mv_ctr  # using center coordinate system
        res.append(cubic_feature)
        if DEBUG:
            box_mv = [box[0], box[1] - box[1], box[2] - box[2], box[3] - box[3], cfg.ANCHOR[0], cfg.ANCHOR[1],
                      cfg.ANCHOR[2], box[7]]
            pcd_vispy(cubic_feature.reshape(-1, 4), np.array(box_mv))

    stack_size = np.concatenate((np.array([-1]), cubic_size))

    return np.array(res, dtype=np.float32).reshape(stack_size)


class cubic(object):
    def __init__(self, batch_size, channel, name, training=True):
        self.batch_size = batch_size
        with tf.variable_scope('Conv3D_1', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_1 = tf.layers.Conv3D(filters=channel[0], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], name='conv3d_1', padding="same", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
        with tf.variable_scope('Conv3D_2', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_2 = tf.layers.Conv3D(filters=channel[0], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], name='conv3d_2', padding="same", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
        with tf.variable_scope('Conv3D_3', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_3 = tf.layers.Conv3D(filters=channel[0], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], name='conv3d_3', padding="same", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)
        with tf.variable_scope('Conv3D_4', reuse=tf.AUTO_REUSE) as scope:
            self.conv3d_4 = tf.layers.Conv3D(filters=channel[0], kernel_size=[3, 3, 3], activation=tf.nn.relu,
                                             strides=[1, 1, 1], name='conv3d_4', padding="same", _reuse=tf.AUTO_REUSE,
                                             _scope=scope, trainable=training)

    def apply(self, inputs):
        conv3d_input = tf.reshape(inputs, np.concatenate((np.array([-1]), cubic_size)))
        out_conv3d_1 = self.conv3d_1.apply(conv3d_input)
        out_conv3d_2 = self.conv3d_2.apply(out_conv3d_1)
        out_conv3d_3 = self.conv3d_3.apply(out_conv3d_2)
        out_conv3d_4 = self.conv3d_4.apply(out_conv3d_3)
        res_stack = out_conv3d_4

        return tf.convert_to_tensor(res_stack, dtype=tf.float32)

    def net_forward2(self, conv3d_input):
        conv3d_input = tf.reshape(conv3d_input, cubic_size)
        out_conv3d_1 = self.conv3d_1.apply(conv3d_input)
        out_conv3d_2 = self.conv3d_2.apply(out_conv3d_1)
        out_conv3d_3 = self.conv3d_3.apply(out_conv3d_2)
        out_conv3d_4 = self.conv3d_4.apply(out_conv3d_3)

        mid_shape = tf.shape(out_conv3d_4)
        res = out_conv3d_4
        return res

    def apply2(self, inputs):
        input_shape = tf.shape(inputs)
        i = tf.constant(0)
        res_stack = []

        def cond(x, y):
            return tf.less(x, input_shape[0])

        def body(x, cubic_stack):
            res = self.net_forward(cubic_stack[x])
            res_stack.append(res)
            return x + 1, cubic_stack

        cnt, point_stack = tf.while_loop(cond, body, [i, inputs])

        return tf.convert_to_tensor(res_stack, dtype=tf.float32)


if __name__ == '__main__':
    arg = edict()
    arg.imdb_type = 'kitti'
    dataset = dataset_train(arg)

    while True:
        idx = input('Type a new index: ')
        blobs = dataset.get_minibatch(idx)
        boxes = np.hstack((np.zeros([blobs['gt_boxes_3d'].shape[0], 1], dtype=np.float32), blobs['gt_boxes_3d']))
        cubic_rpn_grid_pyfc(blobs['lidar3d_data'], boxes)

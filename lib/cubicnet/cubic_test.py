
import tensorflow as tf
from network.config import cfg
from tensorflow.python.client import timeline
import time
from tools.timer import Timer
import os
import random
import string
import cv2
import numpy as np
from tools.transform import lidar_3d_to_bv

DEBUG = False
random_folder = ''.join(random.sample(string.ascii_letters, 4))
os.makedirs(cfg.TEST_RESULT + '/' + random_folder)

class combinet_test(object):
    def __init__(self, network, data_set, args):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.net = network
        self.dataset = data_set
        self.args = args
        self.epoch = self.dataset.input_num

    def testing(self, sess, test_writer):
        with tf.name_scope('loss_cubic'):
            tf.summary.scalar('total_loss', 1)
        with tf.name_scope('view_cubic_rpn'):
            roi_bv = self.net.get_output('rpn_rois')[0]
            data_bv = self.net.lidar_bv_data
            image_rpn = tf.reshape(show_rpn_tf(data_bv,roi_bv), (1, 601, 601, -1))
            tf.summary.image('lidar_bv_test', image_rpn)
        merged = tf.summary.merge_all()

        weights = self.args.weights
        if weights.endswith('.ckpt'):
            print 'Loading pre-trained model weights from {:s}'.format(self.args.weights)
            self.saver.restore(sess, weights)
        else:
            print "error: Function [combinet_test.testing] can not load weights {:s}!".format(self.args.weights)
            return 0

        timer = Timer()
        for idx in range(self.epoch):
            # get one batch
            blobs = self.dataset.get_minibatch(idx)
            feed_dict = {
                self.net.lidar3d_data: blobs['lidar3d_data'],
                self.net.lidar_bv_data: blobs['lidar_bv_data'],
                self.net.im_info: blobs['im_info'],
                self.net.calib: blobs['calib']}
            run_options = None
            run_metadata = None
            timer.tic()
            RPN_view = sess.run(merged, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            timer.toc()

            if cfg.TEST.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-test-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()
            if idx % cfg.TEST.ITER_DISPLAY == 0:
                print 'Test: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f,' % \
                      (idx+1, self.epoch, 0.1,0.1, 0.1)
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
            if idx % 1 == 0 and cfg.TEST.TENSORBOARD:
                test_writer.add_summary(RPN_view, idx)
                pass

        print 'Test process has done!'


def network_testing(network, data_set, args):
    net = combinet_test(network, data_set, args)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        test_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300)
        net.testing(sess, test_writer)


def show_rpn_tf(img, box_pred=None):
    bv_data = tf.reshape(img[:, :, :, 8],(601, 601, 1))
    bv_data = scales_to_255(bv_data,0,3,tf.float32)
    bv_img = tf.reshape(tf.stack([bv_data,bv_data,bv_data],3),(601,601,3))
    return tf.py_func(show_bbox, [bv_img,box_pred], tf.float32)


CNT = 0

def show_bbox(bv_image, bv_box):
    for i in range(bv_box.shape[0]):
        a = bv_box[i, 0]*255
        color_pre = (a, a, a)
        cv2.rectangle(bv_image, (bv_box[i, 1], bv_box[i, 2]), (bv_box[i, 3], bv_box[i, 4]), color=color_pre)
    #
    # global CNT
    # path = cfg.TEST_RESULT + '/' + random_folder
    # filename = os.path.join(path, str(CNT).zfill(6) + '.png')
    # # print 'Write image to {:s}'.format(filename(i))
    # cv2.imwrite(filename, bv_image)
    # CNT += 1
    return bv_image


def scales_to_255(a, min_, max_, type_):
    pass
    return tf.cast(((a - min_) / float(max_ - min_)) * 255, dtype=type_)

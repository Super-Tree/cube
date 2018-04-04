# coding=utf-8
from tools.data_visualize import show_rpn_tf

import random
import os
import math
import numpy as np
import tensorflow as tf
from tools.timer import Timer
from network.config import cfg
from tools.utils import fast_hist
from tensorflow.python.client import timeline
from tensorflow.python import pywrap_tensorflow
from tools.data_visualize import pcd_vispy,vispy_init,pcd_vispy_client
##================================================
# from multiprocessing import Process,Queue
# MSG_QUEUE = Queue(200)
##================================================
DEBUG = False
class msg_qt(object):
    def __init__(self,scans=None, img=None,queue=None, boxes=None, name=None,
                 index=0, vis_size=(800, 600), save_img=False,visible=True, no_gt=False):
        self.scans=scans,
        self.img=img,
        self.boxes=boxes,
        self.name=name,
        self.index=index,
        self.vis_size=vis_size,
        self.save_img=save_img,
        self.visible=visible,
        self.no_gt=no_gt,
        self.queue=queue

    def check(self):
        pass

class CubicNet_Train(object):
    def __init__(self, network, data_set, args):
        self.saver = tf.train.Saver(max_to_keep=100)
        self.net = network
        self.dataset = data_set
        self.args = args
        self.random_folder = cfg.RANDOM_STR
        self.epoch = self.dataset.training_rois_length
        self.val_epoch = self.dataset.validing_rois_length

    def snapshot(self, sess, iter=None):
        output_dir = os.path.join(cfg.ROOT_DIR, 'output', self.random_folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.join(output_dir, 'CubicNet_iter_{:d}'.format(iter) + '.ckpt')
        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

    @staticmethod
    def modified_smooth_l1(sigma, bbox_pred, bbox_targets):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        diffs = tf.subtract(bbox_pred, bbox_targets)

        smooth_l1_sign = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(diffs, diffs), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(diffs), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        outside_mul = smooth_l1_result

        return outside_mul

    def training(self, sess, train_writer):
        with tf.name_scope('loss_cubic'):
            RNet_rpn_yaw_pred = self.net.get_output('RNet_theta')[1]
            RNet_rpn_yaw_gt_delta = self.net.get_output('cubic_grid')[1]
            RNet_rpn_yaw_gt = self.net.get_output('rpn_rois')[1][:,-1]#rpn_3d_boxes:(x1,y1,z1),(x2,y2,z2),score,rpn_cls_label,yaw
            RNet_rpn_yaw_gt_new = RNet_rpn_yaw_gt-RNet_rpn_yaw_gt_delta
            RNet_rpn_yaw_pred_toshow = RNet_rpn_yaw_pred+RNet_rpn_yaw_gt_delta
            rpn_cls_labels = self.net.get_output('rpn_rois')[1][:,-2]#rpn_3d_boxes:(x1,y1,z1),(x2,y2,z2),score,rpn_cls_label,yaw
            tower_l1_loss = self.modified_smooth_l1(sigma=3, bbox_pred=RNet_rpn_yaw_pred, bbox_targets=RNet_rpn_yaw_gt_new)
            tower_l1_loss_keep_positive = tf.multiply(rpn_cls_labels, tower_l1_loss)
            loss = tf.reduce_sum(tower_l1_loss_keep_positive)/(1e-5+tf.reduce_sum(tf.cast(tf.not_equal(tower_l1_loss_keep_positive, 0.0), dtype=tf.float32)))

        with tf.name_scope('train_op'):
            global_step = tf.Variable(1, trainable=False, name='Global_Step')
            lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 10000, 0.92, name='decay-Lr')
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

        with tf.name_scope('debug_tb'):
            tf.summary.scalar('total_loss', loss)
            glb_var = tf.trainable_variables()
            for i in range(len(glb_var)):
                tf.summary.histogram(glb_var[i].name, glb_var[i])
            tf.summary.image('theta', self.net.get_output('RNet_theta')[0],max_outputs=50)
            merged = tf.summary.merge_all() #hxd: before the next summary ops

        sess.run(tf.global_variables_initializer())
        if self.args.fine_tune:
            if False:
                # #full graph restore
                print 'Loading pre-trained model weights from {:s}'.format(self.args.weights)
                self.net.load(self.args.weights, sess, self.saver, True)
            else:  # #part graph restore
                #  # METHOD one
                # ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=['vgg_feat_fc'])
                # saver1 = tf.train.Saver(ref_vars)
                # saver1.restore(sess, self.args.weights)
                #  # METHOD two
                reader = pywrap_tensorflow.NewCheckpointReader(self.args.weights)
                var_to_shape_map = reader.get_variable_to_shape_map()
                with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
                    for key in var_to_shape_map:
                        try:
                            var = tf.get_variable(key, trainable=False)
                            sess.run(var.assign(reader.get_tensor(key)))
                            print "    Assign pretrain model: " + key
                        except ValueError:
                            print "    Ignore variable:" + key
        trainable_var_for_chk=tf.trainable_variables()#tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
        print 'Variables to train: ',trainable_var_for_chk

        timer = Timer()
        rpn_rois_3d = self.net.get_output('rpn_rois')[1]

        if DEBUG:
            pass # TODO: Essential step(before sess.run) for using vispy beacuse of the bug of opengl or tensorflow
            vispy_init()

        training_series = range(self.epoch)  # self.epoch
        for epo_cnt in range(self.args.epoch_iters):
            for data_idx in training_series:  # DO NOT EDIT the "training_series",for the latter shuffle
                iter = global_step.eval()  # function "minimize()"will increase global_step
                blobs = self.dataset.get_minibatch(data_idx, 'train')  # get one batch
                feed_dict = {
                    self.net.lidar3d_data: blobs['lidar3d_data'],
                    self.net.lidar_bv_data: blobs['lidar_bv_data'],
                    self.net.im_info: blobs['im_info'],
                    self.net.keep_prob: 0.5,
                    self.net.gt_boxes_bv: blobs['gt_boxes_bv'],
                    self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                    self.net.gt_boxes_corners: blobs['gt_boxes_corners'],
                    self.net.calib: blobs['calib'],
                }

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                timer.tic()
                delta_,RNet_rpn_yaw_gt_delta_,rpn_rois_3d_,loss_,RNet_rpn_yaw_pred_toshow_,RNet_rpn_yaw_gt_,merged_,_ = \
                    sess.run([tower_l1_loss_keep_positive,RNet_rpn_yaw_gt_delta,rpn_rois_3d,loss,RNet_rpn_yaw_pred_toshow,RNet_rpn_yaw_gt,merged,train_op,]
                             ,feed_dict=feed_dict,options=run_options, run_metadata=run_metadata)
                timer.toc()

                if iter % cfg.TRAIN.ITER_DISPLAY == 0:
                    print 'Iter: %d/%d, Serial_num: %s, speed: %.3fs/iter, loss: %.3f '%(iter,self.args.epoch_iters * self.epoch, blobs['serial_num'],timer.average_time,loss_)
                    print 'theta_delta: ',
                    for i in range(50):
                        if delta_[i]!=0.0:
                            print '%5.3f' % (delta_[i]),
                    print '\n'
                if iter % 20 == 0 and cfg.TRAIN.TENSORBOARD:
                    train_writer.add_summary(merged_, iter)
                    pass
                if (iter % 4000==0 and cfg.TRAIN.DEBUG_TIMELINE) or (iter == 100):
                    #chrome://tracing
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file = open(cfg.LOG_DIR+'/' +'training-step-'+ str(iter).zfill(7) + '.ctf.json', 'w')
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                    trace_file.close()
                if DEBUG:
                    scan = blobs['lidar3d_data']
                    cubic_cls_value = np.ones([cfg.TRAIN.RPN_POST_NMS_TOP_N],dtype=np.float32)*0
                    boxes = boxary2dic(gt_box3d=blobs['gt_boxes_3d'], pre_box3d=rpn_rois_3d_,pre_theta_value=RNet_rpn_yaw_pred_toshow_,pre_cube_cls=cubic_cls_value)# RNet_rpn_yaw_pred_toshow_
                    pcd_vispy(scan, boxes=boxes,name='CubicNet training')
            if cfg.TRAIN.EPOCH_MODEL_SAVE:
                self.snapshot(sess, iter)
                pass
            if cfg.TRAIN.USE_VALID:#TODO: to complete the valid process
                with tf.name_scope('valid_cubic_' + str(epo_cnt + 1)):
                    print 'Valid the net at the end of epoch_{} ...'.format(epo_cnt + 1)
                    # roi_bv = self.net.get_output('rpn_rois')[0]
                    # cubu_bv = np.hstack((roi_bv,cubic_cls_labels.reshape(-1,1)))
                    # pred_rpn_ = show_rpn_tf(self.net.lidar_bv_data,cubu_bv)
                    # pred_rpn = tf.reshape(pred_rpn_,(1, 601, 601, -1))
                    # predicted_bbox = tf.summary.image('predict_bbox_bv', pred_rpn)
                    # valid_result = tf.summary.merge([predicted_bbox])
                    recalls = self.net.get_output('rpn_rois')[2]
                    pred_tp_cnt, gt_cnt = 0., 0.
                    hist = np.zeros((cfg.NUM_CLASS, cfg.NUM_CLASS), dtype=np.float32)

                    for data_idx in range(self.val_epoch):  # self.val_epoch
                        blobs = self.dataset.get_minibatch(data_idx, 'valid')
                        feed_dict_ = {
                            self.net.lidar3d_data: blobs['lidar3d_data'],
                            self.net.lidar_bv_data: blobs['lidar_bv_data'],
                            self.net.im_info: blobs['im_info'],
                            self.net.keep_prob: 0.5,
                            self.net.gt_boxes_bv: blobs['gt_boxes_bv'],
                            self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                            self.net.gt_boxes_corners: blobs['gt_boxes_corners'],
                            self.net.calib: blobs['calib']}
                        cubic_cls_score_, cubic_cls_labels_, recalls_ = sess.run(
                            [cubic_cls_score, cubic_cls_labels, recalls], feed_dict=feed_dict_)
                        # train_writer.add_summary(valid, data_idx)

                        pred_tp_cnt = pred_tp_cnt + recalls_[1]
                        gt_cnt = gt_cnt + recalls_[2]
                        cubic_class = cubic_cls_score_.argmax(axis=1)
                        one_hist = fast_hist(cubic_cls_labels_, cubic_class)
                        if not math.isnan(one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1])):
                            if not math.isnan(one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0])):
                                hist += one_hist
                        if cfg.TRAIN.VISUAL_VALID:
                            print 'Valid step: {:d}/{:d} , rpn recall = {:.3f}'\
                                  .format(data_idx + 1,self.val_epoch,float(recalls_[1]) / recalls_[2])
                            print('    class bg precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[1, 0]+1e-6)),
                                (one_hist[0, 0] / (one_hist[0, 0] + one_hist[0, 1]+1e-6))))
                            print('    class car precision = {:.3f}  recall = {:.3f}'.format(
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[0, 1]+1e-6)),
                                (one_hist[1, 1] / (one_hist[1, 1] + one_hist[1, 0]+1e-6))))
                        if data_idx % 20 ==0 and cfg.TRAIN.TENSORBOARD:
                            pass
                            # train_writer.add_summary(valid_result_, data_idx/20+epo_cnt*1000)

                precise_total = hist[1, 1] / (hist[1, 1] + hist[0, 1]+1e-6)
                recall_total = hist[1, 1] / (hist[1, 1] + hist[1, 0]+1e-6)
                recall_rpn = pred_tp_cnt / gt_cnt
                valid_summary = tf.summary.merge([rpn_recall_smy_op, cubic_recall_smy_op, cubic_prec_smy_op])
                valid_res = sess.run(valid_summary, feed_dict={epoch_rpn_recall: recall_rpn,
                                                               epoch_cubic_recall: recall_total,
                                                               epoch_cubic_precise: precise_total})
                train_writer.add_summary(valid_res, epo_cnt + 1)
                print 'Validation of epoch_{}: rpn_recall {:.3f} cubic_precision = {:.3f}  cubic_recall = {:.3f}'\
                      .format(epo_cnt + 1,recall_rpn,precise_total,recall_total)
            random.shuffle(training_series)  # shuffle the training series
        print 'Training process has done, enjoy every day !'

def boxary2dic(gt_box3d=None,pre_box3d=None,pre_theta_value=None,pre_cube_cls=None):
    # gt_box3d: (x1,y1,z1),(x2,y2,z2),dt_cls,yaw
    # pre_box3d: (x1,y1,z1),(x2,y2,z2),score,rpn_cls_label
    # cubic_theta_value:pre_box3d's yaw value
    boxes=dict({})
    if gt_box3d is None:
        gt_box3d=np.zeros([1,8],dtype=np.float32)
    if pre_box3d is None:
        pre_box3d=np.zeros([cfg.TRAIN.RPN_POST_NMS_TOP_N,8],dtype=np.float32)
    if pre_theta_value is None:
        pre_theta_value=np.ones([cfg.TRAIN.RPN_POST_NMS_TOP_N,1],dtype=np.float32)*(-1.57)
    if pre_cube_cls is None:
        pre_cube_cls = np.zeros([cfg.TRAIN.RPN_POST_NMS_TOP_N, 1], dtype=np.float32)

    boxes["center"]= np.vstack((gt_box3d[:,0:3],pre_box3d[:,0:3]))
    boxes["size"]  = np.vstack((gt_box3d[:,3:6], pre_box3d[:,3:6]))
    boxes["score"]  = np.vstack((gt_box3d[:, 6:7], pre_box3d[:, 6:7]))
    boxes["cls_rpn"]  = np.vstack((gt_box3d[:, 6:7]*4, pre_box3d[:, 7:8]))#two cls flag  to save more information
    boxes["cls_cube"]  = np.vstack((gt_box3d[:, 6:7]*4, np.reshape(pre_cube_cls,[-1,1])))#todo add cubic cls
    boxes["yaw"]   = np.vstack((gt_box3d[:, 7:8], np.reshape(pre_theta_value,[-1,1])))#pre_box3d[:, 8:9]

    return boxes


def network_training(network, data_set, args):
    net = CubicNet_Train(network, data_set, args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(cfg.LOG_DIR, sess.graph, max_queue=300)
        net.training(sess, train_writer)

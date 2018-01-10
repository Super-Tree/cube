

from network import Network
import tensorflow as tf

anchor_scales = [1.0, 1.0]
anchor_num = 1
_feat_stride = [8, 8]

auto = False  # control the head network whether to be trained in cubic net


class train_net(Network):
    def __init__(self, gpu_use):
        self.inputs = []
        self.lidar3d_data = tf.placeholder(tf.float32, shape=[None, 4])
        self.lidar_bv_data = tf.placeholder(tf.float32, shape=[1, 601, 601, 9])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes_bv = tf.placeholder(tf.float32, shape=[None, 5])
        self.gt_boxes_3d = tf.placeholder(tf.float32, shape=[None, 7])
        self.gt_boxes_corners = tf.placeholder(tf.float32, shape=[None, 25])
        self.calib = tf.placeholder(tf.float32, shape=[None, 12])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'lidar3d_data': self.lidar3d_data,
                            'lidar_bv_data': self.lidar_bv_data,
                            'calib': self.calib,
                            'im_info': self.im_info,
                            'gt_boxes_bv': self.gt_boxes_bv,
                            'gt_boxes_3d': self.gt_boxes_3d,
                            'gt_boxes_corners': self.gt_boxes_corners})

        self.setup(gpu_use)

    def setup(self, gpu_id):
        # for idx, dev in enumerate(gpu_id):
        #     with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
        (self.feed('lidar_bv_data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=auto)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=auto)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=auto)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=auto)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1', trainable=auto)
         .conv(3, 3, 256, 1, 1, name='conv3_2', trainable=auto)
         .conv(3, 3, 256, 1, 1, name='conv3_3', trainable=auto)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1', trainable=auto)
         .conv(3, 3, 512, 1, 1, name='conv4_2', trainable=auto)
         .conv(3, 3, 512, 1, 1, name='conv4_3', trainable=auto)
         .conv(3, 3, 512, 1, 1, name='conv5_1', trainable=auto)
         .conv(3, 3, 512, 1, 1, name='conv5_2', trainable=auto)
         .conv(3, 3, 512, 1, 1, name='conv5_3', trainable=auto))
        # ========= RPN ============
        (self.feed('conv5_3')
         # .deconv(shape=None, c_o=512, stride=2, ksize=3,  name='deconv_2x_1')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3', trainable=auto)
         .conv(1, 1, anchor_num * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score', trainable=auto))
        (self.feed('rpn_conv/3x3')
         .conv(1, 1, anchor_num * 3, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred', trainable=auto))

        (self.feed('rpn_cls_score', 'gt_boxes_bv', 'gt_boxes_3d', 'im_info')
         .anchor_target_layer(_feat_stride, name='rpn_anchors_label'))

        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob')
         .reshape_layer(anchor_num * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info', 'gt_boxes_bv')
         .proposal_layer_3d(_feat_stride, 'TRAIN', name='rpn_rois'))

        (self.feed('lidar3d_data', 'rpn_rois', 'im_info')
         .rpn_extraction(name='rpn_points')
         .rpn_points_classify(name='cubic_cls')
         )

        (self.feed('lidar3d_data','rpn_rois')
         .cubic_cnn(name='cubic_cnn')

         )


        pass
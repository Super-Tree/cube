# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
from network.config import cfg
from generate_anchors import generate_anchors_bv, generate_anchors
from rpn.bbox_transform import clip_boxes, bbox_transform_inv_3d, bbox_transform_inv
from rpn.nms_wrapper import nms
from tools.transform import bv_anchor_to_lidar, lidar_to_bv, lidar_3d_to_bv, lidar_3d_to_corners, lidar_cnr_to_img
import pdb
import yaml
import datetime

DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""


def proposal_layer_3d(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, gt_bv, cfg_key, _feat_stride=[8, 8]):
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)

    # layer_params = yaml.load(self.param_str_)
    beg = datetime.datetime.now()
    _anchors = generate_anchors_bv()
    #  _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    im_info = im_info[0]
    assert rpn_cls_prob_reshape.shape[0] == 1, 'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want

    height, width = rpn_cls_prob_reshape.shape[1:3]
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:, :, :, :, 1],
                        [1, height, width, _num_anchors])  # extract the second kind (fg) scores
    bbox_deltas = rpn_bbox_pred
    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'rpn_bbox_pred shape : {}'.format(rpn_bbox_pred.shape)

    # 1. Generate proposals from bbox deltas and shifted anchors
    if DEBUG:
        print 'score map size: {}'.format(scores.shape)
        pass

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride[0]
    shift_y = np.arange(0, height) * _feat_stride[1]
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # Enumerate all shifted anchors:
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))
    # print "anchors shape: ", anchors.shape
    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    # bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 6))
    bbox_deltas = bbox_deltas.reshape((-1, 3))  # delta x delta y delta z
    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    # scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
    scores = scores.reshape((-1, 1))

    if DEBUG:
        print "anchors before filter"
        print "anchors shape: ", anchors.shape
        print "scores shape: ", scores.shape
    ###
    # only keep anchors inside the image
    inds_inside = _filter_anchors(anchors, im_info, allowed_border=0)
    anchors = anchors[inds_inside, :]
    scores = scores[inds_inside]
    bbox_deltas = bbox_deltas[inds_inside, :]
    ####

    # convert anchors bv to anchors_3d
    anchors_3d = bv_anchor_to_lidar(anchors)
    # Convert anchors into proposals via bbox transformations
    proposals_3d = bbox_transform_inv_3d(anchors_3d, bbox_deltas)
    # convert back to lidar_bv
    proposals_bv = lidar_3d_to_bv(proposals_3d)
    if DEBUG:
        print "after filter"
        print "proposals_bv shape: ", proposals_bv.shape
        print "proposals_3d shape: ", proposals_3d.shape
        print "scores shape: ", scores.shape
    # # 2. clip predicted boxes to image
    # proposals_bv = clip_boxes(proposals_bv, im_info[:2])

    # # 3. remove predicted boxes with either height or width < threshold
    # # (NOTE: convert min_size to input image scale stored in im_info[2])
    # keep = _filter_boxes(proposals_bv, min_size * im_info[2])
    # proposals_bv = proposals_bv[keep, :]
    # proposals_3d = proposals_3d[keep, :]
    # # proposals_img = proposals_img[keep, :]
    # scores = scores[keep]


    # keep = _filter_img_boxes(proposals_img, [375, 1242])
    # proposals_bv = proposals_bv[keep, :]
    # proposals_3d = proposals_3d[keep, :]
    # proposals_img = proposals_img[keep, :]
    # scores = scores[keep]

    # print "proposals_img shape: ", proposals_img.shape
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals_bv = proposals_bv[order, :]
    proposals_3d = proposals_3d[order, :]
    # proposals_img = proposals_img[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    if DEBUG:
        print "proposals before nms"
        print "proposals_bv shape: ", proposals_bv.shape
        print "proposals_3d shape: ", proposals_3d.shape

    keep = nms(np.hstack((proposals_bv, scores)), nms_thresh)
    if DEBUG:
        print keep
        print 'keep.shape',len(keep)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals_bv = proposals_bv[keep, :]
    proposals_3d = proposals_3d[keep, :]
    # proposals_img = proposals_img[keep, :]
    scores = scores[keep]
    if DEBUG:
        num = np.sort(scores.ravel())
        num = num[::-1]
        print num
    if DEBUG:
        print "proposals after nms"
        print "proposals_bv shape: ", proposals_bv.shape
        print "proposals_3d shape: ", proposals_3d.shape

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    length = proposals_bv.shape[0]
    box_labels,recall= valid_pred(proposals_bv,gt_bv,length,cfg.TRAIN.RPN_POSITIVE_OVERLAP)
    blob_bv = np.hstack((scores, proposals_bv.astype(np.float32, copy=False),box_labels.reshape(length,-1)))
    blob_3d = np.hstack((scores, proposals_3d.astype(np.float32, copy=False),box_labels.reshape(length,-1)))
    end2 = datetime.datetime.now()

    if DEBUG:
        print 'NMS & bbox use time:', end2 - beg

    return blob_bv, blob_3d, recall


def _filter_anchors(anchors, im_info, allowed_border):
    """Remove all boxes with any side smaller than min_size."""
    inds_inside = np.where(
        (anchors[:, 0] >= allowed_border) &
        (anchors[:, 1] >= allowed_border) &
        (anchors[:, 2] < im_info[1] + allowed_border) &  # width
        (anchors[:, 3] < im_info[0] + allowed_border)    # height
    )[0]
    return inds_inside


def _filter_img_boxes(boxes, im_info):
    """Remove all boxes with any side smaller than min_size."""
    padding = 50
    w_min = -padding
    w_max = im_info[1] + padding
    h_min = -padding
    h_max = im_info[0] + padding
    keep = np.where((w_min <= boxes[:, 0]) & (boxes[:, 2] <= w_max) & (h_min <= boxes[:, 1]) &
                    (boxes[:, 3] <= h_max))[0]
    return keep

def valid_pred(perd_bv,gt_bv,length,thres):
    gt_bv = gt_bv[:,0:4]
    overlaps = bbox_overlaps(
        np.ascontiguousarray(perd_bv, dtype=np.float),
        np.ascontiguousarray(gt_bv, dtype=np.float))
    max_overlaps = overlaps.max(axis=1)
    positive = np.where(max_overlaps[:]>thres)
    labels = np.zeros(length,dtype=np.float32)
    labels[positive] = 1

    max = overlaps.max(axis=0)
    cnt = np.where(max[:]>thres)[0]
    recall = np.array([float(len(cnt))/len(gt_bv),len(cnt),len(gt_bv)],dtype=np.float32)

    return labels,recall


def bbox_overlaps(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    # ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    ua = float(box_area)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


if __name__ =='__main__':
    pred = np.array([[1,1,3,3],[10,10,12,12],[7,7,9,9],[17,17,19,19]])
    gt = np.array([[1,1,2,2,1],[9,9,11,11,1]])
    label = valid_pred(pred,gt,4,0.5)
    pass
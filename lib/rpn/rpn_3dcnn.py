import os
import numpy as np
import math
from tools.data_visualize import pcd_vispy
from network.config import cfg
from easydict import EasyDict as edict
from dataset.dataset import dataset_train

DEBUG = True

def cubic_rpn_grid(lidarPoints,rpnBoxes):

    x_points = lidarPoints[:, 0]
    y_points = lidarPoints[:, 1]
    z_points = lidarPoints[:, 2]
    reflectance = lidarPoints[:,3]

    shape =lambda i:int(np.ceil(np.round(cfg.ANCHOR[i]/cfg.CUBIC_RES[i],3)))  # Be careful about python number  decimal
    cubic_size=[shape(0),shape(1),shape(2),4]
    res =[]
    for box in rpnBoxes:
        x_min = box[1]-float(cfg.ANCHOR[0])/2
        x_max = box[1]+float(cfg.ANCHOR[0]) / 2
        y_min = box[2]-float(cfg.ANCHOR[1])/2
        y_max = box[2]+float(cfg.ANCHOR[1]) / 2
        z_min = box[3]-float(cfg.ANCHOR[2])/2
        z_max = box[3]+float(cfg.ANCHOR[2]) / 2

        f_filt = np.logical_and((x_points > x_min), (x_points < x_max))
        s_filt = np.logical_and((y_points > y_min), (y_points < y_max))
        z_filt = np.logical_and((z_points > z_min), (z_points < z_max))
        fliter = np.logical_and(np.logical_and(f_filt, s_filt),z_filt)
        indice = np.flatnonzero(fliter)
        rpn_points = lidarPoints[indice]
        points_mv_min=np.subtract(rpn_points,np.array([x_min,y_min,z_min,0.],dtype=np.float32))  # using fot coordinate
        points_mv_ctr=np.subtract(rpn_points,np.array([box[1],box[2],box[3],0.],dtype=np.float32))  # using as feature

        xi = points_mv_min[:, 0]
        yi = points_mv_min[:, 1]
        zi = points_mv_min[:, 2]

        x_cub = np.divide(xi,cfg.CUBIC_RES[0]).astype(np.int32)
        y_cub = np.divide(yi,cfg.CUBIC_RES[1]).astype(np.int32)
        z_cub = np.divide(zi,cfg.CUBIC_RES[2]).astype(np.int32)

        cubic_feature = np.zeros(shape=cubic_size, dtype=np.float32)
        cubic_feature[x_cub,y_cub,z_cub]=points_mv_ctr # using center coordinate system
        res.append(cubic_feature)
        if DEBUG:
            box_mv = [box[0], box[1] - box[1], box[2] - box[2],box[3] - box[3], cfg.ANCHOR[0], cfg.ANCHOR[1], cfg.ANCHOR[2], box[7]]
            pcd_vispy(cubic_feature.reshape(-1,4),np.array(box_mv))
            
    stack_size = np.concatenate((np.array([-1]),cubic_size))
    return np.array(res, dtype=np.float32).reshape(stack_size)


if __name__ == '__main__':
    arg=edict()
    arg.imdb_type ='kitti'
    dataset = dataset_train(arg)

    while True:
        idx = input('Type a new index: ')
        blobs = dataset.get_minibatch(idx)
        boxes = np.hstack((np.zeros([blobs['gt_boxes_3d'].shape[0],1],dtype=np.float32),blobs['gt_boxes_3d']))
        cubic_rpn_grid(blobs['lidar3d_data'],boxes)


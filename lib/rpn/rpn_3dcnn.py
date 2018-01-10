import os
import numpy as np
from tools.data_visualize import pcd_vispy
from network.config import cfg

def cubic_rpn_cnn(lidarPoints,rpnBoxes):

    x_points = lidarPoints[:, 0]
    y_points = lidarPoints[:, 1]
    z_points = lidarPoints[:, 2]
    reflectance = lidarPoints[:,3]

    shape =lambda i:int(np.ceil(cfg.ANCHOR[i]/cfg.CUBIC_RES[i]))
    cubic_size=[shape(0),shape(1),shape(2),4]
    cubic_feature = np.zeros(shape=cubic_size,dtype=np.float32)

    for box in rpnBoxes:
        x_min = box[1]-float(box[4])/2
        x_max = box[1]+float(box[4]) / 2
        y_min = box[2]-float(box[5])/2
        y_max = box[2]+float(box[5]) / 2
        z_min = box[3]-float(box[6])/2
        z_max = box[3]+float(box[6]) / 2

        f_filt = np.logical_and(
            (x_points > x_min), (x_points < x_max))
        s_filt = np.logical_and(
            (y_points > y_min), (y_points < y_max))
        z_filt = np.logical_and(
            (z_points > z_min), (z_points < z_max))
        fliter = np.logical_and(np.logical_and(f_filt, s_filt),z_filt)
        indice = np.flatnonzero(fliter)

        rpn_points = lidarPoints[indice]

        points_mv = np.subtract(rpn_points,np.array([x_min,y_min,z_min,0.],dtype=np.float32))

        xi = points_mv[:, 0]
        yi = points_mv[:, 1]
        zi = points_mv[:, 2]

        x_cub = np.divide(xi,cfg.CUBIC_RES[0]).astype(np.int32)
        y_cub = np.divide(yi,cfg.CUBIC_RES[1]).astype(np.int32)
        z_cub = np.divide(zi,cfg.CUBIC_RES[2]).astype(np.int32)

        cubic_feature[x_cub,y_cub,z_cub]=points_mv
        pcd_vispy(cubic_feature.reshape(-1,4))
    pass


if __name__ == '__main__':
    path_ = '/home/hexindong/ws_dl/pyProj/cubic-local/data/training/velodyne'
    name = os.path.join(path_,str(62).zfill(6)+'.bin')
    scans = np.fromfile(name,dtype=np.float32).reshape(-1,4)

    boxes=np.array([[0.8,19,3.9,-0.8,3.9,3.9,1.56]],dtype=np.float32) #,[0.6,5.,2,-0.2,3.9,3.9,1.8],
    cubic_rpn_cnn(scans,boxes)
    a =[]
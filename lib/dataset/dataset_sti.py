
"""Factory method for easily getting imdbs by name."""
import os
import os.path as osp
from os.path import join as path_add
import numpy as np
import random
import cPickle
from network.config import cfg
import cv2
from easydict import EasyDict as edict
import re

from tools.pcd_py_method.py_pcd import point_cloud as pcd2npScan

class datasetSTI_train(object):  # read txt files one by one
    def __init__(self, arguments):
        self._type = arguments.imdb_type  # kitti or sti
        self.data_path = '/home/hexindong/DATASET/stidataset/'
        self.folder_list = ['170818-1743-LM120', '170825-1708-LM120','170829-1743-LM120']
        self._classes = ['unknown', 'smallMot', 'bigMot', 'nonMot', 'pedestrian']
        self.num_classes = len(self._classes)
        self.total_roidb=[]
        self.filter_roidb=[]
        self.percent_train =0.66
        self.percent_valid = 0.26
        self.train_set,self.valid_set,self.test_set=self.load_dataset()
        print 'Done!'

    def load_dataset(self):
        train_cache_file = path_add(self.data_path, 'train_cache_data.pkl')
        valid_cache_file = path_add(self.data_path, 'valid_cache_data.pkl')
        test_cache_file =  path_add(self.data_path, 'test_cache_data.pkl')
        if os.path.exists(train_cache_file) & os.path.exists(valid_cache_file)& os.path.exists(test_cache_file):
            print 'Loaded the STi dataset from pkl cache files ...'
            with open(train_cache_file, 'rb') as fid:
                train_set = cPickle.load(fid)
                print '  Train gt set loaded from {}'.format(train_cache_file)

            with open(valid_cache_file, 'rb') as fid:
                valid_set = cPickle.load(fid)
                print '  valid gt set loaded from {}'.format(valid_cache_file)

            with open(test_cache_file, 'rb') as fid:
                test_set = cPickle.load(fid)
                print '  test gt set loaded from {}'.format(test_cache_file)

            return train_set, valid_set, test_set

        print 'Prepare the STi dataset for training, please wait ...'
        self.total_roidb=self.load_sti_annotation()
        # self.filter_roidb = self.total_roidb
        self.filter_roidb = self.filter(self.total_roidb)
        train_set,valid_set,test_set=self.assign_dataset(self.filter_roidb)  # train,valid percent

        with open(train_cache_file, 'wb') as fid:
            cPickle.dump(train_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded train gt roidb to {}'.format(train_cache_file)
        with open(valid_cache_file, 'wb') as fid:
            cPickle.dump(valid_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded valid gt roidb to {}'.format(valid_cache_file)
        with open(test_cache_file, 'wb') as fid:
            cPickle.dump(test_set, fid, cPickle.HIGHEST_PROTOCOL)
            print '  Wrote and loaded test gt roidb to {}'.format(test_cache_file)

        return train_set, valid_set, test_set

    def load_sti_annotation(self):
        """
        Load points and bounding boxes info from txt file in the KITTI
        format.
        """
        for index,folder in enumerate(self.folder_list):
            libel_fname = path_add(self.data_path, folder, 'label', 'result.txt')
            label = []
            files_names = []
            with open(libel_fname, 'r') as f:
                lines = f.readlines()
            for line in lines:
                files_names.append(re.findall('\d+-\d+-LM\d+/\d+-\d+-LM\d+_\d+\.pcd', line)[0])
                line = line.replace('unknown', '0.0').replace('smallMot', '1.0').replace('bigMot', '2.0').replace('nonMot', '3.0').replace('pedestrian', '4.0')
                object_str = line.translate(None, '\"').split('position:{')[1:]
                label_in_frame = []
                for obj in object_str:
                    f_str_num = re.findall('[-+]?\d+\.\d+', obj)
                    for j, num in enumerate(f_str_num):
                        pass
                        f_str_num[j] = float(num)
                    if j == 10:  # filter the  wrong type label like   type: position
                        label_in_frame.append(f_str_num)
                selected_label = np.array(label_in_frame, dtype=np.float32)
                label.append(selected_label[:, (0, 1, 2, 6, 7, 8, 3, 9)])  # extract the valuable data:x,y,z,l,w,h,theta,type
            if index == 0:
                total_labels=label
                total_fnames=files_names
            else:
                total_labels.extend(label)
                total_fnames.extend(files_names)

        dataset=[dict({'files_list': total_fnames[i],'labels': total_labels[i]}) for i in range(len(total_fnames))]
        return dataset

    def assign_dataset(self,data):
        cnt = len(data)
        test_index = []
        train_index = []

        temp_index = sorted(random.sample(range(cnt), int(cnt * (self.percent_train + self.percent_valid))))
        for i in range(cnt):
            if i not in temp_index:
                test_index.append(i)
        valid_index = sorted(random.sample(temp_index, int(cnt * self.percent_valid)))
        for k in temp_index:
            if k not in valid_index:
                train_index.append(k)

        train_roidb = [data[k] for k in train_index]
        valid_roidb = [data[k] for k in valid_index]
        test_roidb = [data[k] for k in test_index]

        return train_roidb,valid_roidb,test_roidb

    def filter(self,data):
        """Remove roidb entries that have no usable RoIs."""
        #numpy:->   x,y,z, l, w,h ,theta,type
        def is_valid(dataset):
            boxes = dataset['labels']
            inds_inside = np.where((boxes[:, 0] >= -45.) & (boxes[:, 0] <= 45.) &
                                   (boxes[:, 1] >= -45.) & (boxes[:, 1] <= 45.) &
                                   ((boxes[:, 7] == 1.0) | (boxes[:, 7] == 0.0))
                                   )[0]

            if len(inds_inside) ==0:
                return False,None
            else:
                return True,boxes[inds_inside]

        keep_indice =[]
        num=len(data)
        for index in range(num):
            print index
            keep,result = is_valid(data[index])
            if keep:
                data[index]['labels'] = result
                keep_indice.append(index)

        filter_data = [data[k] for k in keep_indice]

        num_after = len(filter_data)
        print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,num, num_after)
        return filter_data

    def augmentation_of_data(self):
        # Rotation of the image or change the scale
        pass

    def get_minibatch(self, idx=0, name='train'):
        """Given a roidb, construct a minibatch sampled from it."""
        if name == 'train':
            dataset = self.train_set
        elif name =='valid':
            dataset = self.valid_set
        else:
            dataset = self.test_set

        fname = dataset[idx]['files_list']
        lidar_data = pcd2npScan.from_path(path_add(self.data_path,fname.split('/')[0],'pcd',fname.split('/')[1]))
        gt_label = dataset[idx]['labels']
        blobs = dict({'lidar3d_data': lidar_data.pc_data,
                      'gt_boxes_3d': gt_label,
                      })

        return blobs

    @staticmethod
    def stiData2pointcloud(Scan):
        point_cloud = Scan.reshape((16, 2016, 4))
        pointx = point_cloud[:, :, 0].flatten()
        pointy = point_cloud[:, :, 1].flatten()
        pointz = point_cloud[:, :, 2].flatten()
        intensity = point_cloud[:, :, 3].flatten()
        # labels = point_cloud[:, :, 6].flatten()

        seg_point = PointCloud()
        seg_point.header.frame_id = 'rslidar'
        channels1 = ChannelFloat32()
        seg_point.channels.append(channels1)
        seg_point.channels[0].name = "rgb"
        channels2 = ChannelFloat32()
        seg_point.channels.append(channels2)
        seg_point.channels[1].name = "intensity"

        for i in range(32256):
            seg_point.channels[1].values.append(intensity[i])
            if True:# labels[i] == 1:
                seg_point.channels[0].values.append(255)
                geo_point = Point32(pointx[i], pointy[i], pointz[i])
                seg_point.points.append(geo_point)
            else:
                seg_point.channels[0].values.append(255255255)
                geo_point = Point32(pointx[i], pointy[i], pointz[i])
                seg_point.points.append(geo_point)
                # elif result[i] == 2:
                #     seg_point.channels[0].values.append(255255255)
                #     geo_point = Point32(pointx[i], pointy[i], pointz[i])
                #     seg_point.points.append(geo_point)
                # elif result[i] == 3:
                #     seg_point.channels[0].values.append(255000)
                #     geo_point = Point32(pointx[i], pointy[i], pointz[i])
                #     seg_point.points.append(geo_point)

        return seg_point


class dataset_test(object):  # read txt files one by one
    def __init__(self, arguments):
        self._type = arguments.imdb_type  # kitti or sti
        self._classes = ('__background__', 'Car')  # , 'Pedestrian', 'Cyclist')
        self.num_classes = len(self._classes)
        if arguments.use_demo:
            self._data_path = osp.join(osp.dirname(__file__), '../../data', 'drive_0064')  # data path
        else:
            self._data_path = osp.join(osp.dirname(__file__), '../../data', 'testing')  # data path
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self.inputIndex = self.get_fileIndex(self._data_path)
        self.input_num = len(self.inputIndex['test_index'])

        print 'The kitti dataset(cnt:{}) is using for testing ...'.format(self.input_num)
        self.roidb = self.prepare_roidb()
        pass

    def prepare_roidb(self):
        roidb = [dict({}) for _ in range(self.input_num)]
        indice = lambda idx, name: self.inputIndex[name][idx]
        for i in xrange(self.input_num):
            roidb[i]['lidar3d_path'] = self.lidar3d_path_at(indice(i, 'test_index'))
            roidb[i]['lidar_bv_path'] = self.lidar_bv_path_at(indice(i, 'test_index'))
            roidb[i]['image_path'] = self.image_path_at(indice(i, 'test_index'))
            roidb[i]['calib'] = self.get_calib(indice(i, 'test_index'))
        return roidb

    def image_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        prefix = 'image_2'
        # image_path = '$Faster-RCNN_TF/data/KITTI/object/training/image_2/000000.png'
        image_path = os.path.join(self._data_path, prefix, str(index).zfill(6) + '.png')
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def lidar3d_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        prefix = 'velodyne'
        # image_path = '$Faster-RCNN_TF/data/KITTI/object/training/image_2/000000.png'
        lidar3d_path = os.path.join(self._data_path, prefix, str(index).zfill(6) + '.bin')
        assert os.path.exists(lidar3d_path), 'Path does not exist: {}'.format(lidar3d_path)
        return lidar3d_path

    def lidar_bv_path_at(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        index = int(index)
        # set the prefix
        prefix = 'lidar_bv'
        # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
        lidar_bv_path = os.path.join(self._data_path, prefix, str(index).zfill(6) + '.npy')
        assert os.path.exists(lidar_bv_path), \
            'Path does not exist: {}'.format(lidar_bv_path)
        return lidar_bv_path

    def get_calib(self, index):

        calib_dir = os.path.join(self._data_path, 'calib', str(index).zfill(6) + '.txt')

        with open(calib_dir) as fi:
            lines = fi.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        calib_ori = {'P2': P2.reshape(3, 4),
                     'P3': P3.reshape(3, 4),
                     'R0': R0.reshape(3, 3),
                     'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}
        # calib_ori = self.load_kitti_calib(index)
        calib = np.zeros((4, 12))
        calib[0, :] = calib_ori['P2'].reshape(12)
        calib[1, :] = calib_ori['P3'].reshape(12)
        calib[2, :9] = calib_ori['R0'].reshape(9)
        calib[3, :] = calib_ori['Tr_velo2cam'].reshape(12)

        return calib

    def get_fileIndex(self, data_path):
        length = len(os.listdir(osp.join(data_path, 'lidar_bv')))
        test_index = range(length)
        return dict({'test_index': test_index})

    def get_minibatch(self, idx=0):
        """Given a roidb, construct a minibatch sampled from it."""
        dataset = self.roidb
        im_scales = [1]
        lidar_bv = np.load(dataset[idx]['lidar_bv_path'])
        lidar_bv_blob = lidar_bv.reshape((1, lidar_bv.shape[0], lidar_bv.shape[1], lidar_bv.shape[2]))
        lidar3d = np.fromfile(dataset[idx]['lidar3d_path'], dtype=np.float32)
        lidar3d_blob = lidar3d.reshape((-1, 4))
        img = cv2.imread(dataset[idx]['image_path'])

        blobs = dict({'lidar_bv_data': lidar_bv_blob,
                      'lidar3d_data': lidar3d_blob,
                      'calib': dataset[idx]['calib'],
                      'im_info': np.array([[lidar_bv_blob.shape[1], lidar_bv_blob.shape[2], im_scales[0]]],
                                          dtype=np.float32),
                      'image_data': img
                      })

        return blobs


def get_data(arguments):
    """Get an imdb (image database) by name."""
    if arguments.method == 'train':
        return datasetSTI_train(arguments)
    else:
        return dataset_test(arguments)


if __name__ == '__main__':
    arg = edict()
    arg.method = 'train'
    arg.imdb_type = 'sti'
    dad = datasetSTI_train(arg)
    a = dad.get_minibatch(0,name='train')

    import rospy
    from sensor_msgs.msg import PointCloud, ChannelFloat32
    from geometry_msgs.msg import Point32

    rospy.init_node('rostensorflow')
    pub = rospy.Publisher('prediction', PointCloud, queue_size=1000)
    rospy.loginfo("ROS begins ...")

    idx = 0
    while True:
        scans = dad.get_minibatch(idx, name='train')
        pointcloud = dad.stiData2pointcloud(scans['lidar3d_data'])
        pub.publish(pointcloud)
        idx += 1
        if idx >3000:
            idx = 0


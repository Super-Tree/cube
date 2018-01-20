
import tensorflow as tf
from tools.utils import scales_to_255
import cv2
import numpy as np
import vispy
from vispy.scene import visuals
vispy.set_log_level('CRITICAL', match='-.-')

def lidar_3d_to_corners(pts_3D):
    """
    convert pts_3D_lidar (x, y, z, l, w, h) to
    8 corners (x0, ... x7, y0, ...y7, z0, ... z7)
    """

    l = pts_3D[:, 3]
    w = pts_3D[:, 4]
    h = pts_3D[:, 5]

    l = l.reshape(-1, 1)
    w = w.reshape(-1, 1)
    h = h.reshape(-1, 1)

    # clockwise, zero at bottom left
    x_corners = np.hstack((l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.))
    y_corners = np.hstack((w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.))
    z_corners = np.hstack((-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.))

    corners = np.hstack((x_corners, y_corners, z_corners))

    corners[:, 0:8] = corners[:, 0:8] + pts_3D[:, 0].reshape((-1, 1)).repeat(8, axis=1)
    corners[:, 8:16] = corners[:, 8:16] + pts_3D[:, 1].reshape((-1, 1)).repeat(8, axis=1)
    corners[:, 16:24] = corners[:, 16:24] + pts_3D[:, 2].reshape((-1, 1)).repeat(8, axis=1)

    return corners

#  using mayavi
def draw_3dPoints_box(lidar=None, Boxex3D=None, is_grid=True, fig=None, draw_axis=True):
    import mayavi.mlab as mlab  # 3d point

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    prs = lidar[:, 3]

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
        pass

    if lidar is not None:
        mlab.points3d(pxs, pys, pzs, prs,
                      mode='point',  # 'point'  'sphere'
                      colormap='gnuplot',  # 'bone',  #'spectral',  #'copper',
                      scale_factor=1,
                      figure=fig)

    if Boxex3D is not None:
        for i in range(Boxex3D.shape[0]):
            b = lidar_3d_to_corners(Boxex3D[i, 1:7].reshape(-1, 6)).reshape(3, 8).transpose()
            a = round(Boxex3D[i, 0], 2)
            if a == 1.0:
                mycolor = (0., 1., 0.)
            else:
                a = max(a - 0.6, 0.025) * 2.5 + 0.01
                mycolor = (a, a, a)

            for k in range(0, 4):
                # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i, j = k, (k + 1) % 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                            line_width=1, figure=fig)

                i, j = k + 4, (k + 1) % 4 + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                            line_width=1, figure=fig)

                i, j = k, k + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=mycolor, tube_radius=None,
                            line_width=1, figure=fig)

    # draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        for y in np.arange(-40, 40, 5):
            x1, y1, z1 = -40.0, float(y), -1.5
            x2, y2, z2 = 40.0, float(y), -1.5
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.1, 0.1, 0.1), tube_radius=None, line_width=0.1,
                        figure=fig)

        for x in np.arange(-40, 40, 5):
            x1, y1, z1 = float(x), -40.0, -1.5
            x2, y2, z2 = float(x), 40.0, -1.5
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(0.1, 0.1, 0.1), tube_radius=None, line_width=0.1,
                        figure=fig)

    # draw axis
    if draw_axis:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
        axes = np.array([
            [2., 0., 0., 0.],
            [0., 2., 0., 0.],
            [0., 0., 2., 0.],
        ], dtype=np.float64)
        fov = np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [40., 40., 0., 0.],
            [40., -40., 0., 0.],
        ], dtype=np.float64)

        mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)
        mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                    figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=None, distance=50, focalpoint=[12.0909996, -1.04700089, -2.03249991])

    mlab.show()

def show_rpn_tf(img, gt_bv_box, anchors, box_pred=None):
    bv_data = tf.reshape(img[:, :, :, 8], (601, 601, 1))
    bv_data = scales_to_255(bv_data, 0, 3, tf.float32)
    bv_img = tf.reshape(tf.stack([bv_data, bv_data, bv_data], 3), (601, 601, 3))
    if gt_bv_box.shape[0] == 0:
        print 'The gt is empty, please debug the code'
    if box_pred is not None:
        return tf.py_func(show_bbox, [bv_img, gt_bv_box, anchors, box_pred], tf.float32)

    return tf.py_func(show_bbox, [bv_img, gt_bv_box, anchors], tf.float32)

def show_bbox(bv_image, bv_gt, anchors, bv_box_pred=None):
    cnt = anchors.shape[0]
    pos = 0
    neg = 0
    for i in range(cnt):
        if anchors[i, 0] == 0:
            neg = neg + 1
            cv2.rectangle(bv_image, (anchors[i, 1], anchors[i, 2]), (anchors[i, 3], anchors[i, 4]), color=(0, 30, 0))
        else:
            pos = pos + 1
            cv2.rectangle(bv_image, (anchors[i, 1], anchors[i, 2]), (anchors[i, 3], anchors[i, 4]), color=(60, 60, 0))

    # print "positive RPN ", pos, 'negative ', neg
    if bv_box_pred is not None:
        for i in range(bv_box_pred.shape[0]):
            a = bv_box_pred[i, 0] * 255
            color_pre = (a, a, a)
            cv2.rectangle(bv_image, (bv_box_pred[i, 1], bv_box_pred[i, 2]), (bv_box_pred[i, 3], bv_box_pred[i, 4]),
                          color=color_pre)

    for i in range(bv_gt.shape[0]):
        cv2.rectangle(bv_image, (bv_gt[i, 0], bv_gt[i, 1]), (bv_gt[i, 2], bv_gt[i, 3]), color=(255, 0, 255))

    # filePath = "/media/disk4/deeplearningoflidar/he/CombiNet-he/output/"
    # cv2.imwrite(filePath+fileName,bv_image)
    return bv_image

#  using vispy
def pcd_vispy(scans=None, boxes=None, name=None, vis_size=(800, 600), now=True,test = False):
    pos = scans[:, :3]
    canvas = vispy.scene.SceneCanvas(title=name, keys='interactive', size=vis_size, show=True)
    vb = canvas.central_widget.add_view()
    vb.camera = 'turntable'
    vb.camera.elevation = 19.0
    vb.camera.center = (3.9, 3.0, 7.1)
    vb.camera.azimuth = -90.0
    vb.camera.scale_factor = 48

    scatter = visuals.Markers()
    scatter.set_gl_state('translucent', depth_test=False)
    scatter.set_data(pos, edge_width=0, face_color=(1, 1, 1, 1), size=0.01, scaling=True)
    vb.add(scatter)

    if False:
        fName='/home/hexindong/ws_dl/pyProj/cubic-local/data/training/image_2/000000.png'
        img = cv2.imread(fName)
        image = visuals.Image(data=img,method='subdivide')
        image.size()
        vb.add(image)

    # *'auto': Automatically
    # *'impostor' if the
    # *'subdivide': ImageVisual is represented as a
    # *'impostor': ImageVisual is represented as a


    if boxes is not None:
        boxes = boxes.reshape(-1, 9)
        gt_indice = np.where(boxes[:, -1] == 2)[0]
        gt_cnt = len(gt_indice)
        i = 0
        for box in boxes:
            radio = max(box[0] - 0.5, 0.005)*2.0
            color = (0, radio, 0, 1)  # Green

            if box[-1] == 4:  #  gt boxes
                i = i + 1
                vsp_box = visuals.Box(width=box[4],  depth=box[5],height=box[6], color=(0.6, 0.8, 0.0, 0.3))#edge_color='yellow')
                mesh_box = vsp_box.mesh.mesh_data
                mesh_border_box = vsp_box.border.mesh_data
                vertices = mesh_box.get_vertices()
                center = np.array([box[1], box[2], box[3]], dtype=np.float32)
                vtcs = np.add(vertices, center)
                mesh_border_box.set_vertices(vtcs)
                mesh_box.set_vertices(vtcs)
                vb.add(vsp_box)
                if False:
                    text = visuals.Text(text='gt: ({}/{})'.format(i, gt_cnt), color='white', face='OpenSans', font_size=12,
                                        pos=[box[1], box[2], box[3]],anchor_x='left', anchor_y='top', font_manager=None)
                    vb.add(text)

            if (box[-1]+box[-2]) == 0: # True negative cls rpn divided by cube
                vb.add(line_box(box,color=color))
            if (box[-1]+box[-2]) == 1: # False negative cls rpn divided by cube
                vb.add(line_box(box,color='red'))
            if (box[-1]+box[-2]) == 2: # False positive cls rpn divided by cube
                if test:
                    vb.add(line_box(box, color='yellow'))
                else:
                    vb.add(line_box(box, color='blue'))
            if (box[-1]+box[-2]) == 3: # True positive cls rpn divided by cube
                vb.add(line_box(box,color='yellow'))

    if now:
        vispy.app.run()
        vispy.app.quit()

    # import vispy.plot as vp
    # import vispy.io as vpio
    # vpio.write_png('name.png',)

    @canvas.connect
    def on_key_press(ev):
        if ev.key.name in '+=':
            a = vb.camera.get_state()
        print(a)

    return canvas

def pcd_show_now():
    vispy.app.run()
    vispy.app.quit()

def vispy_init():
    import vispy
    vispy.use('pyqt4')
    # vispy.app.use_app()
    v = vispy.app.Canvas()

def line_box(box,color=(0, 1, 0, 0.1)):
    p0 = np.array([box[1] - float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])
    p1 = np.array([box[1] - float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])
    p2 = np.array([box[1] + float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])
    p3 = np.array([box[1] + float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])

    p4 = np.array([box[1] - float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])
    p5 = np.array([box[1] - float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])
    p6 = np.array([box[1] + float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])
    p7 = np.array([box[1] + float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])

    pos = np.vstack((p0,p1,p2,p3,p0,p4,p5,p6,p7,p4,p5,p1,p2,p6,p7,p3))
    lines = visuals.Line(pos=pos, connect='strip', width=1, color=color, method='gl')

    return lines

def test_show_rpn_tf(img, box_pred=None):
    bv_data = tf.reshape(img[:, :, :, 8],(601, 601, 1))
    bv_data = scales_to_255(bv_data,0,3,tf.float32)
    bv_img = tf.reshape(tf.stack([bv_data,bv_data,bv_data],3),(601,601,3))
    return tf.py_func(test_show_bbox, [bv_img,box_pred], tf.float32)

def test_show_bbox(bv_image, bv_box):
    for i in range(bv_box.shape[0]):
        a = bv_box[i, 0]*255
        color_pre = (a, a, a)
        cv2.rectangle(bv_image, (bv_box[i, 1], bv_box[i, 2]), (bv_box[i, 3], bv_box[i, 4]), color=color_pre)

    return bv_image


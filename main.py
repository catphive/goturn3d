import numpy as np
import pykitti
import visualizer
import matplotlib.pyplot as plt
import points3d

from pyntcloud import PyntCloud
import pandas as pd

from moviepy.editor import ImageSequenceClip


plt.matplotlib.rcParams['figure.dpi'] = 200

def main2():
    basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
    date = '2011_09_26'
    drive = '0001'

    kitti_data = pykitti.raw(basedir, date, drive)

    tracklet_rects, tracklet_types = visualizer.load_tracklets_for_frames(len(list(kitti_data.velo)),
                                                                          '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(
                                                                              basedir, date, date, drive))

    img = next(kitti_data.cam2)
    img_size = img.size
    velo = next(kitti_data.velo)

    pts_3d, pts_2d = project_velo(kitti_data, fix_velo(velo), img.size)

    plt.imshow(img)
    plt.scatter(pts_2d[:, 0], pts_2d[:,1], s=2, c=pts_3d[:,2])

    tracklet_list = tracklet_rects[0]

    for tracklet in tracklet_list:

        tracklet_3d, tracklet_2d = project_velo(kitti_data, points3d.get_hpts(tracklet.T), img_size)

        plt.scatter(tracklet_2d[:,0], tracklet_2d[:,1], tracklet_2d[:,2], c='red')

    plt.show()


def fix_velo(velo):
    # velo = velo.T

    # get rid of poits with no reflectance.
    # velo = velo[:, 0 < velo[3, :]]

    velo[:, 3] = 1

    return velo

def get_crop_idxs(img_size, pts_2d):
    """
    Takes pts_2 as column points.
    Returns boolean indicies of columns within crop."""
    idxs1 = np.logical_and(0 <= pts_2d[:, 0], pts_2d[:, 0] < img_size[0])

    idxs2 = np.logical_and(0 <= pts_2d[:, 1], pts_2d[:, 1] < img_size[1])

    return np.logical_and(idxs1, idxs2)

def project_velo(kitti_data: pykitti.raw, velo, img_size):

    # pts_3d = fix_velo(velo)

    T_cam0_velo_unrect = kitti_data.calib.T_cam0_velo_unrect

    R_rect_00 = kitti_data.calib.R_rect_00

    # This is the second camera image.
    P_rect_20 = kitti_data.calib.P_rect_20

    pts_3d_cam = velo.dot(T_cam0_velo_unrect.T.dot(R_rect_00.T))

    # pts_3d_cam = R_rect_00.dot(T_cam0_velo_unrect.dot(pts_3d))

    # drop points not in front of camera
    pts_3d_cam = pts_3d_cam[0 < pts_3d_cam[:, 2], :]

    pts_2d_cam = pts_3d_cam.dot(P_rect_20.T)

    pts_2d_cam = pts_2d_cam / pts_2d_cam[:, 2].reshape(-1, 1)

    idxs = get_crop_idxs(img_size, pts_2d_cam)

    pts_2d_cam = pts_2d_cam[idxs, :]
    pts_3d_cam = pts_3d_cam[idxs, :]

    return pts_3d_cam, pts_2d_cam

def main():
    basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
    date = '2011_09_26'
    drive = '0001'

    data = pykitti.raw(basedir, date, drive)#, frames=range(0, 50, 5))


    tracklet_rects, tracklet_types = visualizer.load_tracklets_for_frames(len(list(data.velo)),
                                                                          '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(
                                                                              basedir, date, date, drive))

    # print(f'rects = {tracklet_rects[10][0]}')
    # print(f'types = {tracklet_types[10][0]}')

    # plt.imshow(next(data.cam2))
    # plt.show()

    img = next(data.cam2)

    print(f'img size = {img.size}')

    velo = next(data.velo)





    velo = velo.T

    velo = velo[:, 0 < velo[3, :]]

    velo[3,:] = 1

    cam2_velo_3d = data.calib.T_cam2_velo.dot(velo)
    print(f'cam2 3d shape = {cam2_velo_3d.shape}')
    cam2_velo_3d = cam2_velo_3d[:, 0 < cam2_velo_3d[2, :]]

    print(f'min = {np.min(cam2_velo_3d, 1)}, max = {np.max(cam2_velo_3d, 1)}')

    cam2_velo_2d = data.calib.K_cam2.dot(cam2_velo_3d[0:3, :])
    cam2_velo_2d /= cam2_velo_2d[2, :]

    cam2_velo_2d =  cam2_velo_2d[:, np.logical_and(0 <= cam2_velo_2d[0, :], cam2_velo_2d[0, :] < img.size[0])]

    cam2_velo_2d = cam2_velo_2d[:, np.logical_and(0 <= cam2_velo_2d[1, :], cam2_velo_2d[1, :] < img.size[1])]

    print(f'K = {data.calib.K_cam2}')
    plt.imshow(img)
    plt.scatter(cam2_velo_2d[0,:], cam2_velo_2d[1,:], s=2)
    plt.show()

    print(cam2_velo_2d)
    print(f'min = {np.min(cam2_velo_2d, 1)}, max = {np.max(cam2_velo_2d, 1)}')




    return

    frame = 10
    visualizer.display_frame_statistics(data, tracklet_rects, tracklet_types, frame, points=1.0)

    # visualizer.make_movie(data, tracklet_rects, tracklet_types)


if __name__ == '__main__':
    main2()


import numpy as np
import pykitti

# class Calibration:
#     """Needed because pykitti.raw can't be pickled for some reason."""


def fix_velo(velo):
    # velo = velo.T

    # get rid of poits with no reflectance.
    # velo = velo[:, 0 < velo[3, :]]

    velo[:, 3] = 1

    return velo

def get_crop_idxs(img_shape, pts_2d):
    """
    Takes pts_2 as column points.
    Returns boolean indicies of columns within crop."""
    idxs1 = np.logical_and(0 <= pts_2d[:, 0], pts_2d[:, 0] < img_shape[1])

    idxs2 = np.logical_and(0 <= pts_2d[:, 1], pts_2d[:, 1] < img_shape[0])

    return np.logical_and(idxs1, idxs2)

def get_crop_idxs_bbox(bbox, pts_2d):
    """
    Takes pts_2 as column points.
    Returns boolean indicies of columns within crop."""

    left, top, right, bottom = bbox

    idxs1 = np.logical_and(left <= pts_2d[:, 0], pts_2d[:, 0] < right)

    idxs2 = np.logical_and(top <= pts_2d[:, 1], pts_2d[:, 1] < bottom)

    return np.logical_and(idxs1, idxs2)

def project_velo(kitti_data: pykitti.raw, velo, img_shape):

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

    idxs = get_crop_idxs(img_shape, pts_2d_cam)

    pts_2d_cam = pts_2d_cam[idxs, :]
    pts_3d_cam = pts_3d_cam[idxs, :]

    return pts_3d_cam, pts_2d_cam

def project_velo_bbox(kitti_data: pykitti.raw, velo, bbox):

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

    idxs = get_crop_idxs_bbox(bbox, pts_2d_cam)

    pts_2d_cam = pts_2d_cam[idxs, :]
    pts_3d_cam = pts_3d_cam[idxs, :]

    return pts_3d_cam, pts_2d_cam
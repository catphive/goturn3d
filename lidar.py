import numpy as np

import pykitti
import kitti_transform


class LidarFrame:

    depth_map: np.ndarray

    def __init__(self, kitti_data: pykitti.raw, velo, img_shape):

        # self.depth_map = np.zeros((img_shape[0], img_shape[1]), dtype=np.float64)
        self.kitti_data = kitti_data
        self.velo = velo

        # TODO: get rid of precomputed depth map?
        # pts_3d, pts_2d = kitti_transform.project_velo(kitti_data, kitti_transform.fix_velo(velo), img_shape)
        # pts_2d_idxs = pts_2d.astype(int)
        # self.depth_map[pts_2d_idxs[:, 1], pts_2d_idxs[:, 0]] = pts_3d[:, 2]

    def get_crop(self, bbox, res=224):

        left, top, right, bottom = bbox

        pts_3d, pts_2d = kitti_transform.project_velo_bbox(self.kitti_data, self.velo, bbox)

        depth = np.zeros((res, res), dtype=np.float64)

        # rescale points
        pts_2d = pts_2d - np.array([left, top, 0]).reshape(1, 3)
        pts_2d = pts_2d * np.array([res/(right - left), res/(bottom - top), 1]).reshape(1, 3)

        pts_2d_idxs = pts_2d.astype(int)
        depth[pts_2d_idxs[:, 1], pts_2d_idxs[:, 0]] = pts_3d[:, 2]

        return depth

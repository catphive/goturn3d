import numpy as np

import pykitti
import kitti_transform


class LidarFrame:

    depth_map: np.ndarray

    def __init__(self, kitti_data: pykitti.raw, velo, img_shape):

        self.depth_map = np.zeros((img_shape[0], img_shape[1]), dtype=np.float64)

        pts_3d, pts_2d = kitti_transform.project_velo(kitti_data, kitti_transform.fix_velo(velo), img_shape)

        pts_2d_idxs = pts_2d.astype(int)

        self.depth_map[pts_2d_idxs[:, 1], pts_2d_idxs[:, 0]] = pts_3d[:, 2]

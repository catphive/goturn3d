import unittest

import numpy as np
import lidar
import pykitti
import matplotlib.pyplot as plt


class LidarTest(unittest.TestCase):

    def test_datasets(self):
        basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
        date = '2011_09_26'
        drive = '0001'

        kitti_data = pykitti.raw(basedir, date, drive)

        img = np.array(next(kitti_data.cam2))
        velo = next(kitti_data.velo)

        lidar_frame = lidar.LidarFrame(kitti_data, velo, img.shape)

        plt.imshow(lidar_frame.depth_map)
        plt.show()


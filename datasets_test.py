import unittest

import datasets

import matplotlib.pyplot as plt


def imshow_rgbd(rgbd):

    plt.subplot(2, 1, 1)

    plt.imshow(rgbd[:,:,0:3])

    plt.subplot(2, 1, 2)

    plt.imshow(rgbd[:,:,3])

class DatasetsTest(unittest.TestCase):

    def test_datasets_iter(self):
        basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
        date = '2011_09_26'
        drive = '0001'

        kitti_dataset = datasets.KittiDataset(basedir, date, drive)

        for sample in kitti_dataset:

            # img = sample.get_current_object()
            # print(f'img size = {img.shape}')
            delta = sample.get_delta()
            print(delta)

        print(f'len sample = {len(kitti_dataset)}')


    def test_datasets(self):
        basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
        date = '2011_09_26'
        drive = '0001'

        kitti_dataset = datasets.KittiDataset(basedir, date, drive)

        sample = kitti_dataset[0]

        print('kd len ', len(kitti_dataset))

        img = sample.get_current_object()

        print(img.dtype)


        imshow_rgbd(img)
        # plt.imshow(img)
        plt.show()

        pass

if __name__ == '__main__':
    unittest.main()
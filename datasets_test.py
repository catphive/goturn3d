import unittest

import datasets

import matplotlib.pyplot as plt

from pathlib import Path


def imshow_rgbd(rgbd):

    # convert to numpy format
    rgbd = rgbd.permute(1, 2, 0).numpy()

    plt.subplot(2, 1, 1)

    plt.imshow(rgbd[:,:,0:3])

    plt.subplot(2, 1, 2)

    plt.imshow(rgbd[:,:,3])

class DatasetsTest(unittest.TestCase):

    def test_get_kitti_datasets(self):
        base_path = Path(r'C:\Users\catph\data\kitti_raw\sync\kitti_raw_data\data')

        dataset_list = datasets.get_kitti_datasets(base_path)

        print(dataset_list)


    def test_iter_kitti(self):

        base = Path(r'C:\Users\catph\data\kitti_raw\sync\kitti_raw_data\data')

        arg_list = []

        for date_path in base.iterdir():

            if date_path.is_dir():
                for drive_path in date_path.iterdir():

                    # only get datasets with tracklets
                    if not (drive_path / 'tracklet_labels.xml').exists():
                        continue

                    arg = {'base': base,
                           'date': date_path.name,
                           'drive': drive_path.name.split('_')[-2]}
                    arg_list.append(arg)


        print(arg_list)
        print(len(arg_list))

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

        data = kitti_dataset[0]

        print('kd len ', len(kitti_dataset))

        img = data['current']
        print(f'shape img = {img.size()}')
        # img = sample.get_current_object()

        print(img.dtype)

        imshow_rgbd(img)
        # plt.imshow(img)
        plt.show()

    def test_datasets_data_sizes(self):
        basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
        date = '2011_09_26'
        drive = '0001'

        kitti_dataset = datasets.KittiDataset(basedir, date, drive)

        data = kitti_dataset[0]

        for key, val in data.items():

            print(val.size())


if __name__ == '__main__':
    unittest.main()
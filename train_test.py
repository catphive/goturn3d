import unittest

import datasets
import train
import models
import torch

class TrainTest(unittest.TestCase):

    def test_trainer(self):
        # basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
        # date = '2011_09_26'
        # drive = '0001'

        base = r'C:\Users\catph\data\kitti_raw\sync\kitti_raw_data\data'

        dataset_list = datasets.get_kitti_datasets(base, 1)

        # kitti_dataset = datasets.KittiDataset(basedir, date, drive)

        goturn_model = models.LidarGoturnModel()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        trainer = train.Trainer(dataset_list, goturn_model, device)

        trainer.train()

if __name__ == '__main__':
    unittest.main()
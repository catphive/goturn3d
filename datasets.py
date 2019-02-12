import typing

import numpy as np

from torch.utils.data import Dataset

import pykitti

import parseTrackletXML as xmlParser

import kitti_transform
import points3d
import lidar
import PIL
from torchvision import transforms
import torch
import math

from pathlib import Path

import matplotlib.pyplot as plt


from multiprocessing import Pool

def make_dataset(arg):
    return KittiDataset(arg['base'], arg['date'], arg['drive'])


def get_kitti_datasets(base_path, range_tuple):
    """range_tuple is None or a tuple (begin,end) that turns into slice begin:end

    There are 37 datasets maximum in kitti.
    """

    base_path = Path(base_path)

    arg_list = []

    for date_path in base_path.iterdir():

        if date_path.is_dir():
            for drive_path in date_path.iterdir():

                # only get datasets with tracklets
                if not (drive_path / 'tracklet_labels.xml').exists():
                    continue

                arg = {'base': base_path,
                       'date': date_path.name,
                       'drive': drive_path.name.split('_')[-2]}
                arg_list.append(arg)

    if isinstance(range_tuple, int):
        arg_list = [arg_list[range_tuple]]
    elif isinstance(range_tuple, tuple):
        arg_list = arg_list[range_tuple[0]:range_tuple[1]]


    # with Pool(5) as pool:
    #     dataset_list = pool.map(make_dataset, arg_list)

    dataset_list = []

    for arg in arg_list:
        dataset = KittiDataset(arg['base'], arg['date'], arg['drive'])
        dataset_list.append(dataset)

    return dataset_list


def get_image(kitti_data, tracklet, frame, next_frame):
    basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
    date = '2011_09_26'
    drive = '0001'

def crop_image(img, bbox_pts):

    min_pts = bbox_pts.min(axis=0)
    max_pts = bbox_pts.max(axis=0)

    x_pad = int((max_pts[0] - min_pts[0]) / 2)
    y_pad = int((max_pts[1] - min_pts[1]) / 2)

    # x are columns for numpy array
    # y are rows

    img = img[int(min_pts[1]) - y_pad : int(max_pts[1]) + y_pad,
          int(min_pts[0]) - y_pad:int(max_pts[0]) + y_pad]

    return img


class KittiSample:

    kitti_data: pykitti.raw
    tracklet: xmlParser.Tracklet
    lidar_frame_list: typing.List[lidar.LidarFrame]
    frame: int
    next_frame: int

    img_shape: typing.Tuple[int, int]

    def __init__(self, kitti_data, tracklet, lidar_frame_list: typing.List[lidar.LidarFrame], frame, next_frame, img_shape):

        self.kitti_data = kitti_data
        self.tracklet = tracklet
        self.lidar_frame_list = lidar_frame_list
        self.frame = frame
        self.next_frame = next_frame
        self.img_shape = img_shape

    def get_bbox_pts(self, frame):

        h, w, l = self.tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])

        translation, rotation, state, occlusion, truncation, _, _ = self.tracklet.get_for_frame(frame)

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]  # other rotations are supposedly 0
        assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]
        ])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T

        return cornerPosInVelo.T

    def project_bbox3d(self):

        bbox_pts = self.get_bbox_pts(self.frame)

        pts_3d, pts_2d = kitti_transform.project_velo(self.kitti_data,
                                                      points3d.get_hpts(bbox_pts),
                                                      self.img_shape)

        return pts_3d, pts_2d

    def project_next_bbox3d(self):

        bbox_pts = self.get_bbox_pts(self.next_frame)

        pts_3d, pts_2d = kitti_transform.project_velo(self.kitti_data,
                                                      points3d.get_hpts(bbox_pts),
                                                      self.img_shape)

        return pts_3d, pts_2d

    def _get_object(self, frame, res = 224):

        img = self.kitti_data.get_cam2(self.frame)

        pts_3d, pts_2d = self.project_bbox3d()

        bbox = self.get_crop_bbox(pts_2d)

        img = img.crop(bbox)
        img = img.resize((res, res), PIL.Image.BICUBIC)

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        rgb = transform(img)

        lidar_frame = self.lidar_frame_list[self.frame]
        depth = lidar_frame.get_crop(bbox, res)

        shape = rgb.shape
        rgbd = torch.zeros((4, shape[1], shape[2]), dtype=torch.float32)

        rgbd[0:3,:,:] = rgb
        rgbd[3,:,:] = torch.from_numpy(depth)

        return rgbd


    def get_crop_bbox(self, bbox_pts, res = 172):
        """Takes bbox points in 2d (x,y,1) and gets crop.

        Note that scaling factor of 2 is used to bring in context

        returnx bbox = (x_min, x_max, y_min, y_max)
        """

        # TODO: return square crop rather than rectangular.

        min_pts = bbox_pts.min(axis=0).astype(int)
        max_pts = bbox_pts.max(axis=0).astype(int)

        x_mid = int((max_pts[0] + min_pts[0]) / 2)
        y_mid = int((max_pts[1] + min_pts[1]) / 2)

        offset = res / 2

        return (int(x_mid - offset), int(y_mid - offset), int(x_mid + offset), int(y_mid + offset))

    #
    # def get_crop_bbox(self, bbox_pts):
    #     """Takes bbox points in 2d (x,y,1) and gets crop.
    #
    #     Note that scaling factor of 2 is used to bring in context
    #
    #     returnx bbox = (x_min, x_max, y_min, y_max)
    #     """
    #
    #     # TODO: return square crop rather than rectangular.
    #
    #     min_pts = bbox_pts.min(axis=0).astype(int)
    #     max_pts = bbox_pts.max(axis=0).astype(int)
    #
    #     x_pad = int((max_pts[0] - min_pts[0]) / 2)
    #     y_pad = int((max_pts[1] - min_pts[1]) / 2)
    #
    #     return (int(min_pts[0] - x_pad), int(min_pts[1] - y_pad), int(max_pts[0] + x_pad), int(max_pts[1] + y_pad))

    def get_current_object(self):

        return self._get_object(self.frame)

    def get_next_object(self):

        return self._get_object(self.next_frame)

    def get_delta(self):
        translation1, rotation1, _, _, _, _, _ = self.tracklet.get_for_frame(self.frame)
        translation2, rotation2, _, _, _, _, _ = self.tracklet.get_for_frame(self.next_frame)

        delta_rot = rotation2[2] - rotation1[2]
        c = math.cos(delta_rot)
        s = math.sin(delta_rot)

        return (torch.from_numpy(translation2 - translation1).float(), torch.tensor([c, s]))

class KittiDataset(Dataset):

    samples: typing.List[KittiSample]
    kitti_data: pykitti.raw
    img_shape: typing.Tuple[int, int]
    lidar_frame_list: typing.List[lidar.LidarFrame]

    def __init__(self, basedir, date, drive):

        kitti_data = pykitti.raw(basedir, date, drive)
        self.kitti_data = kitti_data

        xml_path = '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(basedir, date, date, drive)
        tracklets = xmlParser.parseXML(xml_path)

        print(f'num trackelts = {len(tracklets)}')

        self.samples = []

        # get image size
        img = np.array(kitti_data.get_cam2(0))
        img_shape = img.shape
        self.img_shape = img_shape

        self.gen_lidar_frames()

        kept = 0
        skipped = 0
        car = 0
        noncar = 0

        for tracklet in tracklets:
            if tracklet.objectType != "Car":
                noncar += 1
                continue
            else:
                car += 1

            for frame in range(tracklet.firstFrame, tracklet.firstFrame + tracklet.nFrames - 1):

                sample = KittiSample(kitti_data, tracklet, self.lidar_frame_list, frame, frame + 1, img_shape)
                pts_3d, pts_2d = sample.project_bbox3d()

                if pts_2d.shape[0] != 8:
                    skipped += 1
                    continue
                else:
                    kept += 1

                self.samples.append(KittiSample(kitti_data, tracklet, self.lidar_frame_list, frame, frame + 1,
                                                img_shape))

        print(f'sample stats: kept = {kept}, skipped = {skipped}, car = {car}, noncar = {noncar}')

    def gen_lidar_frames(self):

        self.lidar_frame_list = []

        for velo in self.kitti_data.velo:
            self.lidar_frame_list.append(lidar.LidarFrame(self.kitti_data, velo, self.img_shape))


    def __len__(self):

        return len(self.samples)

    def __getitem__(self, item) -> typing.Dict[str, torch.tensor]:

        sample =  self.samples[item]

        current_obj = sample.get_current_object()

        next_obj = sample.get_next_object()

        delta = sample.get_delta()

        delta = torch.cat(delta)

        return {'current': current_obj,
                'next': next_obj,
                'delta': delta}


def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)

def main():

    base =  r"C:\Users\catph\data\kitti_raw\sync\kitti_raw_data\data"
    all_data = get_kitti_datasets(base, (0, 2))


    bbox_list = []
    for data in all_data:
        for sample in data.samples:
            pts_3d, pts_2d = sample.project_bbox3d()
            bbox = sample.get_crop_bbox(pts_2d)

            bbox_list.append(bbox)

    bbox_array = np.array(bbox_list)

    print(bbox_array)

    delta_xs = (bbox_array[:, 2] - bbox_array[:, 0])
    delta_ys = (bbox_array[:, 3] - bbox_array[:, 1])
    print(f'mean delta x = {delta_xs.mean(axis=0)}, std = {delta_xs.std()}')
    print(f'mean delta y = {delta_ys.mean(axis=0)}, std = {delta_ys.std()}')


if __name__ == '__main__':
    main()
import typing

import numpy as np

from torch.utils.data import Dataset

import pykitti

import parseTrackletXML as xmlParser

import kitti_transform
import points3d
import lidar


def get_image(kitti_data, tracklet, frame, next_frame):
    basedir = r'C:\Users\catph\data\kitti_raw\2011_09_26_drive_0001_sync'
    date = '2011_09_26'
    drive = '0001'

def crop_image(img, bbox_pts):

    min_pts = bbox_pts.min(axis=0)
    max_pts = bbox_pts.max(axis=0)

    # x are columns for numpy array
    # y are rows

    img = img[int(min_pts[1]):int(max_pts[1]), int(min_pts[0]):int(max_pts[0])]

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

    def get_current_object(self):

        img = self.kitti_data.get_cam2(self.frame)

        pts_3d, pts_2d = self.project_bbox3d()

        cropped_img = crop_image(np.array(img), pts_2d)

        # convert to float
        cropped_img = cropped_img.astype(np.float64) / 255

        lidar_frame = self.lidar_frame_list[self.frame]
        cropped_depth = crop_image(lidar_frame.depth_map, pts_2d)

        # make (height, width, 4) shape for rgbd
        shape = cropped_img.shape
        rgbd_img = np.zeros((shape[0], shape[1], 4), dtype=np.float64)

        rgbd_img[:,:,0:3] = cropped_img
        rgbd_img[:,:,3] = cropped_depth

        return rgbd_img

    def get_next_object(self):

        img = self.kitti_data.get_cam2(self.next_frame)

        pts_3d, pts_2d = self.project_bbox3d()

        cropped_img = crop_image(np.array(img), pts_2d)

        return cropped_img

    def get_delta(self):
        translation1, rotation1, _, _, _, _, _ = self.tracklet.get_for_frame(self.frame)
        translation2, rotation2, _, _, _, _, _ = self.tracklet.get_for_frame(self.next_frame)

        # TODO: might be better to use cos and sin of angle
        # to avoid sudden jump from 2pi to 0.
        return (translation2 - translation1, rotation2 - rotation1)



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

    def __getitem__(self, item):
        return self.samples[item]




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
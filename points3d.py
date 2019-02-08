import numpy as np
import time
from scipy.spatial import cKDTree

class PointSet:
    """Keeps two copies, one xyz, one in homogeneous form for convenience.

    Assumes speed is more important than memory usage.

    Immutable.
    """

    hpts: np.ndarray

    def __init__(self, pts, hpts=None):
        self.pts: np.ndarray = pts

        if hpts is None:
            self.hpts: np.ndarray = get_hpts(pts)
        else:
            self.hpts = hpts

    def __len__(self):
        """Returns number of rows"""
        return self.pts.shape[0]

    def transform(self, transform_mat: np.ndarray):
        trans_hpts = transform_mat.dot(self.hpts.T).T
        trans_hpts = standardize_hpts(trans_hpts)
        trans_pts = get_non_hpts(trans_hpts)

        return PointSet(trans_pts, trans_hpts)

    def translate(self, offset):
        """Offset in xyz (non-homogeneous)"""

        pts = self.pts + np.reshape(offset, (1, 3))
        return PointSet(pts)

    def permute(self):
        """Return randomly permuted pointset"""

        pts = np.random.permutation(self.pts)
        return PointSet(pts)

    def random_sample(self, size):

        sample_idxs = np.random.choice(self.pts.shape[0], size=size, replace=False)

        pts = self.pts[sample_idxs, :]
        hpts = self.hpts[sample_idxs, :]

        return PointSet(pts, hpts)

    def concat(self, other: 'PointSet'):
        """Concatenate rows from both pointsets"""

        both_pts = np.vstack((self.pts, other.pts))
        both_hpts = np.vstack((self.hpts, other.hpts))

        return PointSet(both_pts, both_hpts)

    def get_closest(self, model: 'PointSet'):
        """Get closest point in model for each point in self

        Returns (points, mse)
        """

        tree = make_kd_tree(model.pts)
        pts = get_closest_points_kd(self.pts, model.pts, tree)

        mse = get_mse(self.pts, pts)
        return (PointSet(pts), mse)


def standardize_hpts(hpts):
    # assumes no points at infinity.
    return hpts / hpts[:, 3].reshape(-1, 1)

def default_type(pts):
    return pts.astype(np.float64)

def is_h(pts):
    return pts.shape[1] == 4

def get_non_hpts(hpts):
    if is_h(hpts):
        hpts = standardize_hpts(hpts)
        return hpts[:, 0:3]
    else:
        return hpts


def get_h(x):
    xh = np.ones(x.shape[0] + 1)
    xh[0:-1] = x
    return xh

def get_hpts(pts):
    res = np.ones((pts.shape[0], pts.shape[1] + 1))
    res[:, 0:-1] = pts
    return res

def make_rand_hpoints(num_points, scale=1):

    hpoints = np.ones((num_points, 4))

    rand_data = np.random.rand(num_points, 3)
    rand_data = (rand_data - 0.5) * scale

    hpoints[:, 0:3] = rand_data
    return hpoints

def get_mse(hpts1, hpts2):
    pts1 = get_non_hpts(hpts1)
    pts2 = get_non_hpts(hpts2)

    diff = pts2 - pts1
    # print(diff)
    se = np.sum((diff) ** 2)
    return se / pts1.shape[0]

def make_kd_tree(hpts):
    pts = get_non_hpts(hpts)
    kd_tree = cKDTree(pts, balanced_tree=False, compact_nodes=False)#)
    return kd_tree

def get_closest_points_kd(hpts1, hpts2, pts_2_kd_tree):
    pts1 = get_non_hpts(hpts1)

    # t1 = time.perf_counter()
    kd_tree = pts_2_kd_tree

    # t2 = time.perf_counter()
    dists, indices = kd_tree.query(pts1, n_jobs=-1)
    # t3 = time.perf_counter()
    # print('t2 = {}, t3 = {}'.format(t2 - t1, t3 - t2))

    return hpts2[indices, :]


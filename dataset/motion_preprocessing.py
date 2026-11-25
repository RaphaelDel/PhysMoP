import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

from constants import G2Hpose_idx
from utils.utils import batch_rodrigues, rotmat2eulerzyx, rotmat2euleryzx, rotmat2eulerzxy


def smpl_to_generalized_coordinates(pose, translation):
    """Convert a continuous, downsampled SMPL recording to PhysMoP's 63 coordinates."""
    pose = torch.as_tensor(pose[:, :66], dtype=torch.float32)
    rotations = batch_rodrigues(pose.reshape(-1, 1, 3)).reshape(-1, 22, 3, 3)
    euler_groups = []
    for start, end, to_euler in (
        (0, 1, rotmat2euleryzx),
        (1, 16, rotmat2eulerzyx),
        (16, 18, rotmat2eulerzxy),
        (18, 20, rotmat2euleryzx),
        (20, 22, rotmat2eulerzyx),
    ):
        group = to_euler(rotations[:, start:end].reshape(-1, 3, 3))
        euler_groups.append(group.reshape(len(pose), -1))
    euler = torch.cat(euler_groups, dim=1).numpy()[:, G2Hpose_idx]

    for angle in range(euler.shape[1]):
        for frame in range(len(euler) - 1):
            while abs(euler[frame + 1, angle] - euler[frame, angle]) > np.pi:
                if euler[frame + 1, angle] < euler[frame, angle]:
                    euler[frame + 1, angle] += 2 * np.pi
                else:
                    euler[frame + 1, angle] -= 2 * np.pi
    euler = gaussian_filter1d(median_filter(euler, size=(3, 1)), sigma=6, axis=0)

    translation = gaussian_filter1d(
        median_filter(translation, size=(4, 1)), sigma=8, axis=0
    )
    q = np.concatenate([translation, euler], axis=1)[15:-15]
    q[:, 3:] = np.remainder(q[:, 3:], 2 * np.pi)
    angles = q[:, 3:]
    angles[angles > np.pi] -= 2 * np.pi
    return q

import os
import numpy as np
from scipy import linalg
from tqdm.contrib.concurrent import process_map
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('..')
from utils.tools import *


def two_d_alignment(v1, v2, flip):
    r1 = R.from_euler('zyx', v1[-1::-1])
    r2 = R.from_euler('zyx', v2[-1::-1])
    if flip:
        r2 = R.from_euler('zyx', [np.pi, 0, 0]) * r2
    r3 = r2 * r1.inv()
    a = r3.as_matrix()
    
    val, v = linalg.eig(a)
    idx = np.where(np.round(val, 6) == 1)[0]
    if len(idx) == 3:
        ax = [0, 1, 0]
    else:
        idx = int(idx[0])
        ax = np.array(v[:, idx])
    
    return (np.abs(np.pi - np.arccos(np.round((a.trace() - 1) / 2, 6))) / np.pi,
            np.abs(np.dot(ax, [0, 1, 0])))


def pf_processing(d2i, d2, d0i, d0, d1i, d1, bin_rotations):

    v2 = (range_mid(d0), range_mid(d1), range_mid(d2))

    intermed_results = np.empty((4, bin_rotations.shape[0]))
    for index, v1 in enumerate(bin_rotations):
        intermed_results[:2, index] = two_d_alignment(v1, v2, False)
        intermed_results[2:, index] = two_d_alignment(v1, v2, True)
    return d0i, d1i, d2i, np.max(intermed_results, axis=1)


def pf_wrapper(args):
    return pf_processing(*args)


def get_pf(num_cubelets, free_axis, hole):
    
    dim0s, dim1s, dim2s = get_heatmap_cell_ranges(num_cubelets, as_ranges=False)

    A = np.zeros((num_cubelets, num_cubelets, num_cubelets))
    E = np.zeros(A.shape)
    A_flipped = np.zeros(A.shape)
    E_flipped = np.zeros(E.shape)
        
        
    # TODO The naming convention here is wrong. the names of alpha and gamma should be switched but otherwise all code can remain the same
    match free_axis:
        case 2:
            alpha_pts = np.linspace(-0.25, 0.25, 5)
            beta_pts = np.linspace(-0.1, 0.1, 5)
            gamma_pts = [range_mid(r[1]) for r in dim2s]
        case 1:
            alpha_pts = np.linspace(-0.25, 0.25, 5)
            beta_pts = [range_mid(r[1]) for r in dim1s]
            gamma_pts = np.linspace(-0.25, 0.25, 5)
        case 0:
            alpha_pts = [range_mid(r[1]) for r in dim0s]
            beta_pts = np.linspace(-0.1, 0.1, 5)
            gamma_pts = np.linspace(-0.25, 0.25, 5)
    
    if hole:
        hole_alpha_pts = np.linspace(-1.8, -1.3, 5)
        if hole == 1:
            alpha_pts = hole_alpha_pts
        elif hole == 2:
            alpha_pts = np.stack([alpha_pts, hole_alpha_pts]).flatten()
            
    bin_rotations = np.moveaxis(np.stack(np.meshgrid(alpha_pts,
                                                     beta_pts,
                                                     gamma_pts)), 0, 3).reshape(-1, 3)

    iterator = ((d2i, d2, d0i, d0,
                 d1i, d1, bin_rotations) for d2i, d2 in dim2s for d0i, d0 in dim0s for d1i, d1 in dim1s)
    for result in process_map(pf_wrapper, iterator, total=(num_cubelets ** 3)):
        d0i, d1i, d2i, (A_max, E_max, A_flipped_max, E_flipped_max) = result
        A[d0i, d1i, d2i] = A_max
        E[d0i, d1i, d2i] = E_max
        A_flipped[d0i, d1i, d2i] = A_flipped_max
        E_flipped[d0i, d1i, d2i] = E_flipped_max

    return A, E, A_flipped, E_flipped


def sigmoid(arr, a, b):
    return 1 / (1 + np.exp((-arr * a) + b))


def sigmoid_on_pf(pf, free_axis):
    
    match free_axis:
        case 'α':
            c1_pow, c1_x, c1_y, c2_pow, c2_x, c2_y = 2, 14, 12, 6, 13, 14
        case 'β':
            c1_pow, c1_x, c1_y, c2_pow, c2_x, c2_y = 2, 9, 8, 15, 5, 5
        case 'γ':
            c1_pow, c1_x, c1_y, c2_pow, c2_x, c2_y = 1, 11, 12, 15, 3, 5
        case 'hole':
            c1_pow, c1_x, c1_y, c2_pow, c2_x, c2_y = 15, 1, 5, 15, 11, 15
    
    sigmoided_pf = np.zeros(pf.shape)
    sigmoided_pf[[0, 2]] = sigmoid(pf[[0, 2]] ** c1_pow, c1_x, c1_y)
    sigmoided_pf[[1, 3]] = sigmoid(pf[[1, 3]] ** c2_pow, c2_x, c2_y)
    return sigmoided_pf


def masked_pf(pf):
    mask = np.zeros(pf.shape)
    mask[[0, 2]] = pf[[0, 2]] > 0.82
    mask[[1, 3]] = (pf[[1, 3]] ** 5) > 0.85

    return np.any(mask, axis=0)


def pf_path_prefix(free_axis_and_hole, project_path):
    return os.path.join(project_path, 'predictive_function', free_axis_and_hole)


def convert_axis_hole_int_to_str(free_axis_and_hole):
    return ['γ', 'β', 'α', 'hole', 'α_hole'][free_axis_and_hole]


def pf_func_path(free_axis_and_hole: str, project_path):
    return f'{pf_path_prefix(free_axis_and_hole, project_path)}_func.npy'


def pf_mask_path(free_axis_and_hole: str, project_path):
    return f'{pf_path_prefix(free_axis_and_hole, project_path)}_mask.npy'


def get_and_set_all_pfs(num_cubelets, project_path):
    # for axis, hole in [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]:
    for axis, hole in [(0, 0), (1, 0), (2, 1), (2, 2)]:

        str_free_axis_and_hole = convert_axis_hole_int_to_str(axis + hole)

        sigmoided_pf = np.stack(get_pf(num_cubelets, axis, hole))
        # sigmoided_pf = sigmoid_on_pf(np.stack(get_pf(num_cubelets, axis, hole)))

        np.save(pf_func_path(str_free_axis_and_hole, project_path), sigmoided_pf)

        np.save(pf_mask_path(str_free_axis_and_hole, project_path), masked_pf(sigmoided_pf))


if __name__ == '__main__':
    get_and_set_all_pfs(32, '/home/avic/OOD_Orientation_Generalization')
        
        
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import pandas as pd
import os
import tqdm
from multiprocessing import Pool
from PIL import Image
from sklearn.preprocessing import normalize
import istarmap

def my_normalize(arr):
    arr_shape = arr.shape
    return normalize(arr.reshape((1, -1)), norm='l2').reshape(arr_shape)

def my_pierson(x, y):
    x = x.flatten()
    y = y.flatten()
    x_avg = np.average(x)
    y_avg = np.average(y)
    x_std = np.std(x)
    y_std = np.std(y)
    return np.dot((x - x_avg).T,(y - y_avg)) / ((x_std * y_std) * (x.shape[0]))

itoaxis = {0:'object_x', 1:'object_y', 2:'object_z'}

def get_unrestricted_axis(restricted_axes):
    if restricted_axes == '(0, 1)':
        return 2
    if restricted_axes == '(0, 2)':
        return 1
    if restricted_axes == '(1, 2)':
        return 0
    return None

def reorder_axes(axes, order):
    ret = [None, None, None]
    for i in range(3):
        ret[order[i]] = axes[i]
    return ret

def get_heatmap_cell_ranges(num_cubelets):
    assert num_cubelets % 2 == 0
    
    longtitude = num_cubelets + 1
    latitude = num_cubelets // 2
    r = 1

    dim0, delta_theta = np.linspace(-np.pi, np.pi, longtitude, retstep=True)
    delta_S = delta_theta / latitude

    dim1 = 1-np.arange(2*latitude+1) * delta_S / (r**2 * delta_theta)
    dim1 =  np.arccos(dim1)
    dim1 = (dim1 - (np.pi / 2))

    dim2 = np.linspace(-np.pi, np.pi, num_cubelets + 1)


    return list(enumerate(zip(dim0, dim0[1:]))), list(enumerate(zip(dim1, dim1[1:]))), list(enumerate(zip(dim2, dim2[1:])))

def twod_alginment(v1, v2):
    r1 = R.from_euler('zyx', v1[-1::-1])
    r2 = R.from_euler('zyx', v2[-1::-1])
    r3 = r2*r1.inv()
    a = r3.as_matrix()
    
    val, v = linalg.eig(a)
    idx = np.where(np.round(val , 6) == 1)[0]
    if len(idx) == 3:
        ax = [0,1,0]
    else:
        idx = int(idx[0])
        ax = np.array(v[:,idx])
    
    return np.abs(np.pi-np.arccos(np.round((a.trace() - 1) / 2, 6)))/np.pi, np.abs(np.dot(ax, [0,1,0]))

def range_mid(r):
    return r[0] + ((r[1] - r[0]) / 2)

def canonical_processing(d2i, d2, d0i, d0, d1i, d1, bin_rotations):

    v2 = (range_mid(d0), range_mid(d1), range_mid(d2))

    intermed_results = np.empty((2, bin_rotations.shape[0]))
    for index, v1 in enumerate(bin_rotations):
        intermed_results[:2, index] = twod_alginment(v1, v2)
        v1_flipped = (v1[0] - np.pi, v1[1], v1[2])
        # intermed_results[2:, index] = twod_alginment(v1_flipped, v2)
    return d0i, d1i, d2i, np.max(intermed_results, axis=1)


def get_canonical_heatmaps(num_bins, unrestricted_axis, hole):

    dim0s, dim1s, dim2s = get_heatmap_cell_ranges(num_bins, num_bins)

    A = np.zeros((num_bins, num_bins, num_bins))
    E = np.zeros(A.shape)
    E_flipped = np.zeros(E.shape)
    # A_flipped = np.zeros(E.shape)
    
    final_dim = dim2s if unrestricted_axis == 2 else dim1s if unrestricted_axis == 1 else dim0s
    center_points = np.array([[0,0],[-0.2,-0.2],[-0.2,0.2],[0.2,-0.2],[0.2,0.2]])
    hole_points = np.array([[-1.55,0],[-1.75,-0.2],[-1.75,0.2],[-1.25,-0.2],[-1.25,0.2]])
    if hole == 0:
        v_1_combinations = center_points
    elif hole == 1:
        v_1_combinations = hole_points
    elif hole == 2:
        v_1_combinations = np.vstack((center_points, hole_points))
        
    bin_rotations = np.repeat(np.insert(v_1_combinations, unrestricted_axis, None, axis=1), len(final_dim), axis=0).reshape(v_1_combinations.shape[0],num_bins,3)
    bin_rotations[:,:,unrestricted_axis] = [range_mid(r[1]) for r in final_dim]
    bin_rotations = bin_rotations.reshape(-1,3)

    # for d2i, d2 in tqdm.tqdm(dim2s):
    #     for d0i, d0 in dim0s:
    #         for d1i, d1 in dim1s:
    #             v2 = (range_mid(d0), range_mid(d1), range_mid(d2))
    #
    #             intermed_results = np.empty((bin_rotations.shape[0], 2))
    #             for index, v1 in enumerate(bin_rotations):
    #                 intermed_results[index] = twod_alginment(v1, v2)
    #
    #             A[d0i, d1i, d2i] = np.max(intermed_results[:, 0])
    #             E[d0i, d1i, d2i] = np.max(intermed_results[:, 1])

    iterator = [(d2i, d2, d0i, d0, d1i, d1, bin_rotations) for d2i, d2 in dim2s for d0i, d0 in dim0s for d1i, d1 in dim1s]

    with Pool(20) as pool:
        for result in tqdm.tqdm(pool.istarmap(canonical_processing, iterator), total=len(iterator)):
            d0i, d1i, d2i, (A_max, E_max) = result
            A[d0i, d1i, d2i] = A_max #max(A_max, A_flipped_max)
            E[d0i, d1i, d2i] = E_max #max(E_max, E_flipped_max)
            # A_flipped[d0i, d1i, d2i] = A_flipped_max
            # E_flipped[d0i, d1i, d2i] = E_flipped_max

    return A, E

def get_and_set_all_canonical_heatmaps():
    for axis, hole in [(0,0), (1,0), (2,0), (2,1), (2,2)]:
        heatmap = get_canonical_heatmaps(20, axis, hole)
        set_generated_canonical_heatmap(heatmap, axis+hole)

def get_generated_canonical_heatmap(unrestricted_axis):
    return np.load(f'/home/avic/Rotation-Generalization/notebooks/canonical_{["x","y","z","hole1","hole2"][unrestricted_axis]}_unrestricted_heatmap.npy')

def set_generated_canonical_heatmap(arr, i):
    return np.save(f'/home/avic/Rotation-Generalization/notebooks/canonical_{["x","y","z","hole1","hole2"][i]}_unrestricted_heatmap.npy', arr)

def get_all_generated_canonical_heatmaps():
    return np.array([get_generated_canonical_heatmap(i) for i in range(5)])

def overlap(a, b):
    return int(min(a[1], b[1]) - max(a[0], b[0]) > 0)

def get_non_bin_cubelets(bins, num_cubelets, return_array):
    dim0s, dim1s, dim2s = get_heatmap_cell_ranges(num_cubelets, num_cubelets)

    for d2i, d2r in dim2s:
        for d0i, d0r in dim0s:
            for d1i, d1r in dim1s:
                for bin in bins:
                    if overlap(bin[0], d0r) + overlap(bin[1], d1r) + overlap(bin[2], d2r) == 2:
                        return_array[d0i, d1i, d2i] = True

def get_all_non_bin_cubelets(num_cubelets=20):
    xy = [[(-0.25, 0.25), (-0.25, 0.25), (0, 0)]]
    xz = [[(-0.25, 0.25), (0, 0), (-0.25, 0.25)]]
    yz = [[(0, 0), (-0.25, 0.25), (-0.25, 0.25)]]
    hole = [[(-1.8, -1.3), (-0.25, 0.25), (0,0)]]
    hole_center = [[(-0.25, 0.25), (-0.25, 0.25), (0, 0)], [(-1.8, -1.3), (-0.25, 0.25), (0,0)]]
    all_restriction_axes = [yz, xz, xy, hole, hole_center]

    results = np.zeros((5, num_cubelets, num_cubelets, num_cubelets), dtype=bool)
    for i, restriction_axes in enumerate(all_restriction_axes):
        get_non_bin_cubelets(restriction_axes, num_cubelets, results[i])

    return ~results
def get_heatmap(eval_df, results, num_images, num_cubelets, images=None, img_boundary=55):

    dim0s, dim1s, dim2s = get_heatmap_cell_ranges(num_cubelets)
    
    for d2i, (d2s, d2e) in dim2s:
            
            in_d2_range = eval_df[eval_df.object_z.between(d2s, d2e)]
            
            for d0i, (d0s, d0e) in dim0s:
                
                    in_d0_range = in_d2_range[in_d2_range.object_x.between(d0s, d0e)]
                
                    for d1i, (d1s, d1e) in dim1s:
                        
                        in_d1_range = in_d0_range[in_d0_range.object_y.between(d1s, d1e)]
                        num_images[d0i, d1i, d2i] = len(in_d1_range)
                        results[d0i, d1i, d2i] = np.nan if len(in_d1_range) == 0 else np.average(in_d1_range.correct)
                        if images is not None:
                            images[d0i, d1i, d2i] = Image.open(in_d1_range.sample(n=1).iloc[0].image_name_x).convert('RGBA').crop((img_boundary, img_boundary, 224 - img_boundary, 224 - img_boundary)) if len(in_d1_range) > 0 else 0
                            
def get_results(exp, num_cubelets, get_images, img_boundary=55, object_scale=None):
    
    results = np.zeros((num_cubelets, num_cubelets, num_cubelets))
    num_images = np.zeros((results.shape))
    images = np.zeros((num_cubelets, num_cubelets, num_cubelets,  224 - (img_boundary * 2),  224 - (img_boundary * 2), 4)) if get_images else None

    if not os.path.exists(exp.eval):
        return None
    
    testing = pd.read_csv(exp.testing_frame_path)
    classic_eval = pd.read_csv(exp.eval)

    eval_frame = testing.merge(right=classic_eval[classic_eval.epoch == max(classic_eval.epoch)], left_on='Unnamed: 0', right_on='Unnamed: 0')
    # if distribution_status > 0:
    #     in_center_z = eval_frame.object_x.between(-0.25, 0.25) & eval_frame.object_y.between(-0.25, 0.25)
    #     in_center_y = eval_frame.object_x.between(-0.25, 0.25) & eval_frame.object_z.between(-0.25, 0.25)
    #     in_center_x = eval_frame.object_y.between(-0.25, 0.25) & eval_frame.object_z.between(-0.25, 0.25)
    #     in_hole = eval_frame.object_x.between(-1.8, -1.3) & eval_frame.object_y.between(-0.25, 0.25)
    #     if distribution_status == 1:
    #         if exp.hole == 0:
    #             if exp.restriction_axes == '(1, 2)':
    #                 eval_frame = eval_frame[in_center_x]
    #             if exp.restriction_axes == '(0, 2)':
    #                 eval_frame = eval_frame[in_center_y]
    #             if exp.restriction_axes == '(0, 1)':
    #                 eval_frame = eval_frame[in_center_z]
    #         if exp.hole == 1:
    #             eval_frame = eval_frame[in_hole]
    #         if exp.hole == 2:
    #             eval_frame = eval_frame[in_center_z | in_hole]
    #     if distribution_status == 2:
    #         if exp.hole == 0:
    #             if exp.restriction_axes == '(1, 2)':
    #                 eval_frame = eval_frame[~in_center_x]
    #             if exp.restriction_axes == '(0, 2)':
    #                 eval_frame = eval_frame[~in_center_y]
    #             if exp.restriction_axes == '(0, 1)':
    #                 eval_frame = eval_frame[~in_center_z]
    #         if exp.hole == 1:
    #             eval_frame = eval_frame[~in_hole]
    #         if exp.hole == 2:
    #             eval_frame = eval_frame[~in_center_z & ~in_hole]

    # if model_num is not None:
    #     all_models = eval_frame.model_name.unique()
    #     eval_frame = eval_frame[eval_frame.model_name == all_models[model_num]]
        
    # if object_scale is not None:
    #     eval_frame = eval_frame[eval_frame.object_scale.between(object_scale[0], object_scale[1])]
    get_heatmap(eval_frame, results, num_images, num_cubelets, images, img_boundary)
    return exp, results, num_images, images
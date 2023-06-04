import os
import numpy as np
import torch
from scipy import linalg
from tqdm.contrib.concurrent import process_map
from scipy.spatial.transform import Rotation as R
from torch.multiprocessing import Pool
from itertools import chain
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


class FitterModule(torch.nn.Module):

    def __init__(self, num_components):
        super().__init__()
        self.weights = torch.nn.init.xavier_uniform_(torch.nn.parameter.Parameter(torch.rand(3, num_components, 1)))

    def corr_distance(self, vx, y):
        vy = y - y.mean()

        return 1 - (torch.sum(vx * vy, axis=0) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2, axis=0))))

    def forward(self, pm, hm_var):
        step1 = pm ** self.weights[0]
        step2 = step1 * self.weights[1]
        step3 = step2 + self.weights[2]
        return self.corr_distance(hm_var, torch.sum(torch.sigmoid(step3), axis=0))

def corr(arr1, arr2):
    assert np.sum(np.isnan(arr1)) == 0
    assert np.sum(np.isnan(arr2)) == 0
    if np.std(arr1) == 0 or np.std(arr2) == 0:
        return 0
    return float(np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1])

def worker_init(num_epochs_init, train_predictive_model_init, val_predictive_model_init, train_heatmap_var_init, val_heatmap_var_init):
    global num_epochs, train_predictive_model, val_predictive_model, train_heatmap_var, val_heatmap_var
    num_epochs = num_epochs_init
    train_predictive_model = train_predictive_model_init
    val_predictive_model = val_predictive_model_init
    train_heatmap_var = train_heatmap_var_init
    val_heatmap_var = val_heatmap_var_init

def driver(args):
    
    # ipm, (num_components,
    #       component_list,
    #       (num_epochs,
    #        train_predictive_model,
    #        val_predictive_model,
    #        train_heatmap_var,
    #        val_heatmap_var)) = args
    ipm, (num_components,
          component_list) = args

    # global num_epochs, train_predictive_model, val_predictive_model, train_heatmap_var, val_heatmap_var

    correlations = torch.full((2, num_epochs), np.nan)

    min_train_cost = torch.tensor(torch.inf)
    min_val_cost = torch.tensor(torch.inf)
    min_epoch = -1
    saved_weights = None

    tpm = train_predictive_model[component_list]
    vpm = val_predictive_model[component_list]

    model = FitterModule(num_components).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-3, min_lr=1e-7, patience=100)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        cost = model(tpm, train_heatmap_var)
        cost_data = cost.data
        correlations[0, epoch] = cost_data
        with torch.no_grad():
            val_cost = model(vpm, val_heatmap_var)
        correlations[1, epoch] = val_cost.data
        if cost < (min_train_cost - 0.001):
            bad_epoch_counter = 0
            saved_weights = model.weights.detach().clone()
            min_train_cost = cost_data
            min_val_cost = val_cost
            min_epoch = epoch
        elif (epoch - 2000) > min_epoch:
            break
        cost.backward()
        optimizer.step()
        scheduler.step(val_cost)
        
    return ipm, correlations, min_train_cost, min_val_cost, saved_weights

def fit_model(result, predictive_model, pbar=None, num_runs=5, num_epochs=5_000):

    partial_heatmap, base_mask, full_heatmap = result.generate_heatmaps()
    
    # partial_heatmap.arr[partial_heatmap.arr <= 0.101] = 0

    partial_heatmap_wo_base = partial_heatmap.arr.flatten()#[~base_mask.arr]

    train_val_split = np.full(partial_heatmap_wo_base.shape, False, bool)
    train_val_split[np.random.choice(len(train_val_split), len(train_val_split) // 2, replace=False)] = True

    train_heatmap = partial_heatmap_wo_base[train_val_split]
    train_heatmap_var = torch.tensor(train_heatmap - train_heatmap.mean(), dtype=torch.float32).cuda()#.share_memory_()
    val_heatmap = partial_heatmap_wo_base[~train_val_split]
    val_heatmap_var = torch.tensor(val_heatmap - val_heatmap.mean(), dtype=torch.float32).cuda()#.share_memory_()

    predictive_model_wo_base = predictive_model#[:, ~base_mask.arr.flatten()]
    train_predictive_model = predictive_model_wo_base[:, train_val_split]
    val_predictive_model = predictive_model_wo_base[:, ~train_val_split]

    def result_list():
        return [[] for _ in range(5)]

    all_correlations = result_list()
    all_min_train_cost = result_list()
    all_min_val_cost = result_list()
    all_saved_weights = result_list()

        
    with Pool(4, initializer=worker_init, initargs=(num_epochs, train_predictive_model, val_predictive_model,
                                                    train_heatmap_var, val_heatmap_var)) as p:
            for (ipm, correlations, min_train_cost, min_val_cost,
                 saved_weights) in p.imap_unordered(driver,
                                                    chain(*[enumerate(zip([1, 1, 2, 2, 4],
                                                                          [0, 1, [0, 1], [2, 3], [0, 1, 2, 3]]))
                                                            for _ in range(num_runs)])):

                all_correlations[ipm].append(correlations)
                all_min_train_cost[ipm].append(min_train_cost)
                all_min_val_cost[ipm].append(min_val_cost)
                all_saved_weights[ipm].append(saved_weights)
                pbar.update()
            
    # return all_correlations, all_min_train_cost, all_min_val_cost, all_saved_weights

    result.predictive_model_params = [all_saved_weights[i][torch.argmin(torch.stack(all_min_val_cost[i]))].tolist() for i in range(5)]
    filtered_min_val_cost = [1 - torch.min(torch.stack(all_min_val_cost[i])).item() for i in range(5)]

    result.unif_corr = corr(partial_heatmap.arr, np.random.sample(partial_heatmap.arr.shape))
    result.a_corr, result.e_corr, result.ae_corr, result.f_ae_corr, result.all_corr = filtered_min_val_cost

    result.save()


def sigmoid(arr):
    return 1 / (1 + np.exp((-arr)))


def sigmoid_cp(arr, a, b):
    import cupy as cp
    return 1 / (1 + cp.exp((-arr * a) + b))


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
        
        
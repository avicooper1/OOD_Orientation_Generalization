import sys
sys.path.append('/home/avic/OOD_Orientation_Generalization')
from my_dataclasses import *
from tools import *
import csv
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool as torchPool
from scipy.optimize import differential_evolution

GPUS = [5, 6, 7]
def convert_to_gpu_tensor(arr, d):
    return torch.from_numpy(arr).float().to(torch.device(f'cuda:{d}'))

def gpu_pierson(x, y):
    
    assert x.shape == y.shape

    assert not (torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)))
    
    x = torch.nan_to_num(x)
    y = torch.nan_to_num(y)
    
    x_avg = torch.mean(x)
    y_avg = torch.mean(y)
    x_std = torch.std(x)
    y_std = torch.std(y)
    
    return (((x - x_avg) @ (y - y_avg)) / ((x_std * y_std) * (x.shape[-1])))

def get_unrestricted_axis(restricted_axes):
    if restricted_axes[1] == '1':
        return 0
    if restricted_axes[4] == '1':
        return 2
    return 1

def gpu_sig(arr, x, y, z):
    return torch.divide(1, 1 + torch.exp(((-torch.pow(arr, x)) + y) * z))

model_heatmaps = get_all_generated_canonical_heatmaps()
bin_masks = ~get_all_bin_masks()

model_heatmaps_bins_removed = [model_heatmaps[x, np.broadcast_to(np.expand_dims(bin_masks[x], 0), (4, 20, 20, 20))].reshape(4, -1) for x in range(5)]
model_heatmaps_gpu = [[convert_to_gpu_tensor(model_heatmaps_bins_removed[x], g) for x in range(5)] for g in GPUS]
bin_masks_gpu = [convert_to_gpu_tensor(bin_masks, g).bool() for g in GPUS]
uniform_model_gpu = [[convert_to_gpu_tensor(np.random.random_sample(model_heatmaps_bins_removed[i].shape[1]), g) for i in range(5)] for g in GPUS]
del model_heatmaps
del bin_masks
del model_heatmaps_bins_removed

def compute_fit(i, exp, scale_index=None):
    
    gpui = i % len(GPUS)
    gpu = GPUS[gpui]

    unrestricted_axis = get_unrestricted_axis(exp.restriction_axes)
    restriction_index = unrestricted_axis + exp.hole

    if scale_index is not None:
        heatmap_path = exp.eval_heatmap_scaled
    else:
        heatmap_path = exp.eval_heatmap

    if not os.path.exists(heatmap_path):
        return None

    if scale_index is not None:
        heatmap = convert_to_gpu_tensor(
            np.nan_to_num(np.load(heatmap_path)[scale_index])[out_bin_masks[restriction_index]])
    else:
        heatmap = convert_to_gpu_tensor(np.nan_to_num(np.load(heatmap_path)), gpu)
        
    ret = [exp.num,
           exp.data_div,
           unrestricted_axis]
    heatmap = heatmap[bin_masks_gpu[gpui][restriction_index]]
    model_heatmap_for_gpu = model_heatmaps_gpu[gpui][restriction_index]

    def gpu_pred_single_component(args):
        a, b, c = args
        return -1 * gpu_pierson(gpu_sig(uniform_model_gpu[gpui][restriction_index], a, b, c).flatten(), heatmap.flatten()).item()

    result = differential_evolution(gpu_pred_single_component, [(0, 10), (0, 1), (10, 10)], polish=False)

    ret.extend(result.x)
    ret.append(-1 * gpu_pred_single_component(result.x))
        
    def gpu_pred_all_components(args, include_components=[0, 1, 2, 3]):
        a, b, c, d, e, f, g, h, i, j, k, l = args
        A_sig = gpu_sig(model_heatmap_for_gpu[0], a, b, c)
        E_sig = gpu_sig(model_heatmap_for_gpu[1], d, e, f)
        A_flipped_sig = gpu_sig(model_heatmap_for_gpu[2], g, h, i)
        E_flipped_sig = gpu_sig(model_heatmap_for_gpu[3], j, k, l)

        components = torch.stack([A_sig, E_sig, A_flipped_sig, E_flipped_sig])
        
        maxes = torch.sum(components[include_components], axis=0)

        return -1 * gpu_pierson(maxes.flatten(), heatmap.flatten()).item()
    
    result = differential_evolution(gpu_pred_all_components, 
                             [(0, 10), (0, 1), (10, 10),
                              (0, 10), (0, 1), (10, 10),
                              (0, 10), (0, 1), (10, 10),
                              (0, 10), (0, 1), (10, 10)], polish=False)

    ret.append(gpu_pred_all_components(result.x, [0]))
    ret.append(gpu_pred_all_components(result.x, [1]))
    ret.append(gpu_pred_all_components(result.x, [0, 1]))
    ret.append(gpu_pred_all_components(result.x, [2, 3]))
    
    ret.extend(result.x)
    ret.append(gpu_pred_all_components(result.x))
    
    
    
    if scale_index is not None:
        ret.insert(3, scale_index)
    return ret

if __name__ == '__main__':

    for SCALE in [False]:
        BASE_PATH = '/home/avic/OOD_Orientation_Generalization'
        exps_frames_paths = [os.path.join(BASE_PATH, f'exps{exp_iter}.csv') for exp_iter in range(1,4)]
        # if not SCALE:
        #     exps_frames_paths += [os.path.join(EXPS_PATH, f'exps_{exp_type}.csv') for exp_type in ['half', 'cross']]

        fits_path = [f'exp{exps_num}_fits{"" if not SCALE else "_scale"}.csv' for exps_num in range(1, 4)]
        if not SCALE:
            fits_path += [f'exp_{exp_type}_fits.csv' for exp_type in ['half', 'cross']]

        FITS_DIRECTORY = os.path.join(BASE_PATH, 'fits')
        os.makedirs(FITS_DIRECTORY, exist_ok=True)

        exps_and_fits = zip(exps_frames_paths, fits_path)

        if not SCALE:
            fits_cols = 'num,data_div,unrestricted_axis,U0,U1,U2,UC,AC,EC,AEC,AfEfC,CA0,CA1,CA2,CE0,CE1,CE2,CAf0,CAf1,CAf2,CEf0,CEf1,CEf2,CC\n'
        else:
            fits_cols = 'num,data_div,unrestricted_axis,scale_index,A_fit_arg0,A_fit_arg1,A_fit,E_fit_arg0,E_fit_arg1,E_fit,AE_fit_arg0,AE_fit_arg1,AE_fit_arg2,AE_fit_arg3,AE_fit\n'

        for exp_path, fit_path in exps_and_fits:
            FITS_FILE = os.path.join(FITS_DIRECTORY, fit_path)

            exps_frame = pd.read_csv(exp_path, index_col=0)
            exps_frame = exps_frame[exps_frame.model_type.isin(
                ['ModelType.ResNet'])]#, 'ModelType.DenseNet', 'ModelType.Inception', 'ModelType.CorNet'])]
            if SCALE:
                exps_frame = exps_frame[exps_frame.scale]
            else:
                exps_frame = exps_frame[~exps_frame.scale]

            # exps_frame = exps_frame[
            #     np.sum(np.array([(exps_frame.augment).to_numpy(dtype=np.int), (exps_frame.pretrained).to_numpy(dtype=np.int),
            #                      (exps_frame.scale).to_numpy(dtype=np.int)]), axis=0) <= 1]
            
            exps_frame = exps_frame[~exps_frame.augment & ~exps_frame.pretrained & ~exps_frame.scale]
            exps_frame = exps_frame[~(exps_frame.training_category == 'lamp')]
            
            for e in exps_frame.itertuples():
                compute_fit(0, e)
                break
            
            with open(FITS_FILE, 'w') as f:
                f.write(fits_cols)
                writer = csv.writer(f)

                mp.set_start_method('spawn', force=True)
                with torchPool(3) as pool:
                # for result in tqdm.tqdm(map(lambda exp: compute_fit(exp[0], exp[1]), exps_frame.iterrows()), total=len(exps_frame)):
                    for result in tqdm.tqdm(pool.istarmap(compute_fit, exps_frame.iterrows()), total=len(exps_frame)):
                        if result is not None:
                            writer.writerow(result)
                            f.flush()
                            # typically the above line would do. however this is used to ensure that the file is written
                            os.fsync(f.fileno())

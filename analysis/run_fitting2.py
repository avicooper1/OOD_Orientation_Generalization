import sys
sys.path.append('/home/avic/Rotation-Generalization')
from my_dataclasses import *
from tools import *
import csv
import numpy as np
import torch

device = torch.device("cuda:6")
def convert_to_gpu_tensor(arr):
    return torch.from_numpy(arr).float().to(device)

def gpu_pierson(x, y):
    x_avg = torch.unsqueeze(torch.mean(x, axis=-1), axis=-1)
    y_avg = torch.unsqueeze(torch.mean(y, axis=-1), axis=-1)
    x_std = torch.std(x, axis=-1)
    y_std = torch.std(y, axis=-1)

    return (((x - x_avg) @ (y - y_avg)) / ((x_std * y_std) * (x.shape[-1])))

def get_unrestricted_axis(restricted_axes):
    if restricted_axes[1] == '1':
        return 0
    if restricted_axes[4] == '1':
        return 2
    return 1

def get_computed_sig(arr, component):
    computed = np.zeros((15, 30, len(arr)))
    for x in range(15):
            for y in range(30):
                if component == 0:
                    computed[x, y] = argsig(arr, x, y).flatten()
                else:
                    computed[x, y] = argsig(arr ** 20, x, y).flatten()
    return convert_to_gpu_tensor(computed.reshape(450, len(arr)))

def sig(arr, x, y):
    return np.divide(1, 1 + np.exp(((-arr) + x) * y))

def argsig(arr, x, y):
    return sig(arr, (x / 10) + 0.1, y + 1)

def unravel(index):
    return np.unravel_index(index, (15,30))

model_heatmaps = get_all_generated_canonical_heatmaps()
out_bin_masks = get_all_non_bin_cubelets()
computed_model_heatmaps = []
for i in range(5):
    computed_model_heatmaps.append((
        get_computed_sig(model_heatmaps[i, 0][out_bin_masks[i]], 0),
        get_computed_sig(model_heatmaps[i, 1][out_bin_masks[i]], 1),
    ))

def compute_fit(exp, scale_index=None):
    unrestricted_axis = get_unrestricted_axis(exp.restriction_axes)
    restriction_index = unrestricted_axis + exp.hole

    if scale_index is not None:
        heatmap_path = exp.eval_heatmap_scaled
    else:
        heatmap_path = exp.eval_heatmap_ood

    if not os.path.exists(heatmap_path):
        return None

    if scale_index is not None:
        heatmap = convert_to_gpu_tensor(
            np.nan_to_num(np.load(heatmap_path)[scale_index])[out_bin_masks[restriction_index]])
    else:
        heatmap = convert_to_gpu_tensor(np.nan_to_num(np.load(heatmap_path))[out_bin_masks[restriction_index]])
    computed = computed_model_heatmaps[restriction_index]
    combined_results = torch.empty((450, 3))
    torch1 = convert_to_gpu_tensor(np.array([1]))
    for i in range(450):
        AE_argmax = torch.argmax(torch.nan_to_num(gpu_pierson(
            torch.maximum(computed[0], computed[1][i]),
            heatmap)))
        combined_results[i] = torch.tensor([AE_argmax, i,
            torch.min(torch1, torch.nan_to_num(gpu_pierson(torch.maximum(computed[0][AE_argmax], computed[1][i]), heatmap)))])

    A_argmax = torch.argmax(torch.nan_to_num(gpu_pierson(computed[0], heatmap))).data.item()
    E_argmax = torch.argmax(torch.nan_to_num(gpu_pierson(computed[1], heatmap))).data.item()
    AE_combined_argmax = combined_results[torch.argmax(combined_results[:,2])]

    A_unraveled_argmax = unravel(A_argmax)
    E_unraveled_argmax = unravel(E_argmax)
    AE_unraveled_argmax0 = unravel(AE_combined_argmax[0].to(torch.int))
    AE_unraveled_argmax1 = unravel(AE_combined_argmax[1].to(torch.int))

    ret = [exp.num,
           exp.data_div,
           unrestricted_axis,
            A_unraveled_argmax[0],
            A_unraveled_argmax[1],
            torch.nan_to_num(gpu_pierson(computed[0][A_argmax], heatmap)).data.item(),
            E_unraveled_argmax[0],
            E_unraveled_argmax[1],
            torch.nan_to_num(gpu_pierson(computed[1][E_argmax], heatmap)).data.item(),
            AE_unraveled_argmax0[0],
            AE_unraveled_argmax0[1],
            AE_unraveled_argmax1[0],
            AE_unraveled_argmax1[1],
            combined_results[int(AE_combined_argmax[2].data.item()), 2].data.item()]
    if scale_index is not None:
        ret.insert(3, scale_index)
    return ret

for SCALE in [False, True]:
    EXPS_PATH = '/home/avic/Rotation-Generalization'
    exps_frames_paths = [os.path.join(EXPS_PATH, f'exps{exp_iter}.csv') for exp_iter in range(1,4)]
    if not SCALE:
        exps_frames_paths += [os.path.join(EXPS_PATH, f'exps_{exp_type}.csv') for exp_type in ['half', 'cross']]

    fits_path = [f'exp{exps_num}_fits{"" if not SCALE else "_scale"}.csv' for exps_num in range(1, 4)]
    if not SCALE:
        fits_path += [f'exp_{exp_type}_fits.csv' for exp_type in ['half', 'cross']]

    FITS_DIRECTORY = '/home/avic/Rotation-Generalization/fits'
    os.makedirs(FITS_DIRECTORY, exist_ok=True)

    exps_and_fits = zip(exps_frames_paths, fits_path)

    if not SCALE:
        fits_cols = 'num,data_div,unrestricted_axis,A_fit_arg0,A_fit_arg1,A_fit,E_fit_arg0,E_fit_arg1,E_fit,AE_fit_arg0,AE_fit_arg1,AE_fit_arg2,AE_fit_arg3,AE_fit\n'
    else:
        fits_cols = 'num,data_div,unrestricted_axis,scale_index,A_fit_arg0,A_fit_arg1,A_fit,E_fit_arg0,E_fit_arg1,E_fit,AE_fit_arg0,AE_fit_arg1,AE_fit_arg2,AE_fit_arg3,AE_fit\n'

    for exp_path, fit_path in exps_and_fits:
        FITS_FILE = os.path.join(FITS_DIRECTORY, fit_path)

        exps_frame = pd.read_csv(exp_path, index_col=0)
        exps_frame = exps_frame[exps_frame.model_type.isin(
            ['ModelType.ResNet', 'ModelType.DenseNet', 'ModelType.Inception', 'ModelType.CorNet'])]
        if SCALE:
            exps_frame = exps_frame[exps_frame.scale]
        else:
            exps_frame = exps_frame[~exps_frame.scale]

        exps_frame = exps_frame[
            np.sum(np.array([(exps_frame.augment).to_numpy(dtype=np.int), (exps_frame.pretrained).to_numpy(dtype=np.int),
                             (exps_frame.scale).to_numpy(dtype=np.int)]), axis=0) <= 1]
        with open(FITS_FILE, 'w') as f:
            f.write(fits_cols)
            writer = csv.writer(f)

            for _, exp in tqdm.tqdm(exps_frame.iterrows(), total=len(exps_frame)):

                # for scale_index in range(2):
                if SCALE:
                    for i in range(5):
                        result = compute_fit(exp, i)
                        if result is not None:
                            writer.writerow(result)
                            f.flush()
                            # typically the above line would do. however this is used to ensure that the file is written
                            os.fsync(f.fileno())
                else:
                    result = compute_fit(exp)
                    if result is not None:
                        writer.writerow(result)
                        f.flush()
                        # typically the above line would do. however this is used to ensure that the file is written
                        os.fsync(f.fileno())
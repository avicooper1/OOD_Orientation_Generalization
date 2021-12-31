import pandas as pd
import numpy as np
import os
import csv
import tqdm

def compute_average(exp, scale_index=None):
    if scale_index is not None:
        path = exp.eval_heatmap_scaled
        if not os.path.exists(path):
            average_scale = None
        else:
            average_scale = np.nanmean(np.load(path)[scale_index])
        return [exp.num, exp.data_div, scale_index, average_scale]

    else:
        id_path = exp.eval_heatmap_id
        if not os.path.exists(id_path):
            id_average = None
        else:
            id_average = np.nanmean(np.load(id_path))

        ood_path = exp.eval_heatmap_ood
        if not os.path.exists(ood_path):
            ood_average = None
        else:
            ood_average = np.nanmean(np.load(ood_path))

        return [exp.num, exp.data_div, id_average, ood_average]

for SCALE in [False, True]:
    EXPS_PATH = '/home/avic/Rotation-Generalization'
    exps_frames_paths = [os.path.join(EXPS_PATH, f'exps{exp_iter}.csv') for exp_iter in range(1,4)]
    if not SCALE:
        exps_frames_paths += [os.path.join(EXPS_PATH, f'exps_{exp_type}.csv') for exp_type in ['half', 'cross']]

    averages_path = [f'exp{exps_num}_averages{"" if not SCALE else "_scale"}.csv' for exps_num in range(1, 4)]
    if not SCALE:
        averages_path += [f'exp_{exp_type}_averages.csv' for exp_type in ['half', 'cross']]

    FITS_DIRECTORY = '/home/avic/Rotation-Generalization/averages'
    os.makedirs(FITS_DIRECTORY, exist_ok=True)

    exps_and_averages = zip(exps_frames_paths, averages_path)

    if not SCALE:
        fits_cols = 'num,data_div,id_average,ood_average\n'
    else:
        fits_cols = 'num,data_div,scale_index,average\n'

    for exp_path, average_path in exps_and_averages:
        AVERAGES_FILE = os.path.join(FITS_DIRECTORY, average_path)

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
        with open(AVERAGES_FILE, 'w') as f:
            f.write(fits_cols)
            writer = csv.writer(f)

            for _, exp in tqdm.tqdm(exps_frame.iterrows(), total=len(exps_frame)):

                # for scale_index in range(2):
                if SCALE:
                    for i in range(5):
                        result = compute_average(exp, i)
                        if result is not None:
                           writer.writerow(result)
                           f.flush()
                           # typically the above line would do. however this is used to ensure that the file is written
                           os.fsync(f.fileno())
                else:
                    result = compute_average(exp)
                    if result is not None:
                        writer.writerow(result)
                        f.flush()
                        # typically the above line would do. however this is used to ensure that the file is written
                        os.fsync(f.fileno())
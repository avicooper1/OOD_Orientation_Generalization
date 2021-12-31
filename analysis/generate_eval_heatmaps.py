import sys
sys.path.append('/home/avic/Rotation-Generalization')
from tools import *
import istarmap

exps_frames_paths = [f'/home/avic/Rotation-Generalization/exps{exp_iter}.csv' for exp_iter in range(1,4)]
exps_frames_paths += ['/home/avic/Rotation-Generalization/exps_half.csv', '/home/avic/Rotation-Generalization/exps_cross.csv']

def get_scale_interval(i, training_category):
    ret = (0.65 + (0.07 * i), 0.72 + (i * 0.07))
    if training_category == 'SM':
        ret = (ret[0] / 20, ret[1] / 20)
    return ret

for frame_path in exps_frames_paths:
    exps_frame = pd.read_csv(frame_path)
    exps_frame = exps_frame[exps_frame.model_type.isin(['ModelType.ResNet', 'ModelType.DenseNet', 'ModelType.Inception', 'ModelType.CorNet'])]
    exps_frame = exps_frame[
        np.sum(np.array([(exps_frame.augment).to_numpy(dtype=np.int), (exps_frame.pretrained).to_numpy(dtype=np.int),
                         (exps_frame.scale).to_numpy(dtype=np.int)]), axis=0) <= 1]

    for exps_subset, is_scale in [(exps_frame[~exps_frame.scale], False), (exps_frame[exps_frame.scale], True)]:

        def generate_eval(_, exp):
            if is_scale:
                heatmaps = np.zeros((5, 20, 20, 20))
                for s in range(5):
                    result = get_results(exp, 20, 20, False, 0, object_scale=get_scale_interval(s, exp.training_category))
                    if result is None:
                        return
                    exp, heatmap, num_images, image = result
                    heatmaps[s] = heatmap
                np.save(exp.eval_heatmap_scaled, heatmaps)

            else:
                result = get_results(exp, 20, 20, False, 1)
                if result is None:
                    return
                exp, heatmap, _, image = result
                np.save(exp.eval_heatmap_id, heatmap)

                result = get_results(exp, 20, 20, False, 2)
                if result is None:
                    return
                exp, heatmap, _, image = result
                np.save(exp.eval_heatmap_ood, heatmap)

        with Pool(32) as pool:
            for _ in tqdm.tqdm(pool.istarmap(generate_eval, exps_subset.iterrows()), total=len(exps_subset)):
                pass
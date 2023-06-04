import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from itertools import product
from utils.persistent_data_class import PersistentDataClass, ExpData
from utils.persistent_data_object import Arr
from utils.tools import get_base_mask
from predictive_function.tools import pf_func_path, corr
import warnings
from pathlib import Path
from scipy.special import expit


def inv_sel_score(pre_act, pre_mask, post_act, post_mask):
    return 1 - np.abs(np.divide((post_act - pre_act), (post_act + pre_act),
                                out=np.full(np.broadcast(pre_act, post_act).shape, np.nan),
                                where=((post_act + pre_act) != 0) & (pre_mask | post_mask)))


@dataclass
class Result(PersistentDataClass):

    d: str
    store_path: str
    exp_data: ExpData
    project_path: str

    num: int = None
    num_fully_seen: int = None
    run: int = None
    free_axis: str = None
    category: str = None
    full_id_acc: float = None
    full_ood_acc: float = None
    partial_id_acc: float = None
    partial_ood_acc: float = None
    full_base_acc: float = None
    full_generalizable_acc: float = None
    full_non_generalizable_acc: float = None
    partial_base_acc: float = None
    partial_generalizable_acc: float = None
    partial_non_generalizable_acc: float = None
    unif_corr: float = None
    id_corr: float = None
    a_corr: float = None
    e_corr: float = None
    ae_corr: float = None
    f_ae_corr: float = None
    all_corr: float = None

    predictive_model_params: list = field(default_factory=lambda: [])

    selectivity: dict = field(default_factory=lambda: {})
    invariance: dict = field(default_factory=lambda: {})

    partial_heatmap: Arr = None
    base_mask: Arr = None
    full_heatmap: Arr = None

    def full_only_frame(self):
        return self.exp_data.full_validation_frame.df[
            self.exp_data.full_validation_frame.df.instance_name.isin(self.exp_data.full_instances)]

    def generate_heatmaps(self, save=False):

        if self.partial_heatmap and os.path.exists(self.partial_heatmap.file_path):
            return

        partial_heatmap = Arr(self.d, 'partial_heatmap')
        partial_heatmap.arr = np.zeros(self.exp_data.dataset_resolution ** 3)

        base_mask = Arr(self.d, 'base_mask')
        base_mask.arr = np.zeros(self.exp_data.dataset_resolution ** 3).astype(bool)

        for base, frame, correct in ((True, self.exp_data.partial_base_frame,
                                      self.exp_data.eval_data.partial_base_correct),
                                     (False, self.exp_data.partial_ood_frame,
                                      self.exp_data.eval_data.partial_ood_correct)):
            for flat_cubelet_i, indices in frame.df.groupby('image_idx', sort=False).indices.items():

                partial_heatmap.arr[flat_cubelet_i] = correct.arr[indices].mean()

                if base:
                    base_mask.arr[np.where(partial_heatmap.arr)] = True

        partial_heatmap.arr = partial_heatmap.arr.reshape((self.exp_data.dataset_resolution,) * 3)
        base_mask.arr = base_mask.arr.reshape((self.exp_data.dataset_resolution,) * 3)

        full_heatmap = Arr(self.d, 'full_heatmap')
        full_heatmap.arr = np.full(self.exp_data.dataset_resolution ** 3, np.nan)

        for flat_cubelet_i, indices in self.full_only_frame().groupby('image_idx', sort=False).indices.items():
            full_heatmap.arr[flat_cubelet_i] = self.exp_data.eval_data.full_validation_correct.arr[indices].mean()

        full_heatmap.arr = full_heatmap.arr.reshape((self.exp_data.dataset_resolution,) * 3)

        if save:
            self.partial_heatmap = partial_heatmap
            self.partial_heatmap.dump()

            self.base_mask = base_mask
            self.base_mask.dump()

            self.full_heatmap = full_heatmap
            self.full_heatmap.dump()

        return partial_heatmap, base_mask, full_heatmap

    def initial_init(self):

        os.makedirs(self.d)
        self.num = self.exp_data.num
        self.num_fully_seen = self.exp_data.num_fully_seen
        self.run = self.exp_data.run
        self.free_axis = self.exp_data.free_axis
        self.category = (self.exp_data.full_category
                         if self.exp_data.full_category == self.exp_data.partial_category
                         else f'{self.exp_data.full_category} -> {self.exp_data.partial_category}')

    def fitted_model(self):
        assert self.predictive_model_params

        predictive_model = np.load(pf_func_path(self.free_axis, self.project_path))

        model_params = np.array(self.predictive_model_params[4])

        return np.sum(expit(((predictive_model ** model_params[0, :, np.newaxis, np.newaxis]) \
                            * model_params[1, :, np.newaxis, np.newaxis]) + model_params[2, :, np.newaxis, np.newaxis]), axis=0)


#     def get_generalizable_mask(self, fitted_predictive_model_pre_calc):

#         if self.num_fully_seen == 40:
#             fitted_predictive_model = fitted_predictive_model_pre_calc
#         else:
#             parts = list(Path(self.exp_data.eval_dir).parts)
#             parts[-3] = '40_fully_seen'
#             results_path = os.path.join(Path(*parts), 'results', 'results.json')
#             if not os.path.exists(results_path):
#                 return None
#             result40 = Result(None, results_path, None, None)
#             if not result40.predictive_model_params:
#                 return None
#             fitted_predictive_model = result40.fitted_model()

#         return fitted_predictive_model > np.min(fitted_predictive_model) + ((np.max(fitted_predictive_model) - np.min(fitted_predictive_model)) / 10)

    def get_generalizable_mask(self, passed_fitted_predictive_model):

        if self.num_fully_seen == 40:
            fitted_predictive_model = passed_fitted_predictive_model
        else:
            parts = list(Path(self.exp_data.eval_dir).parts)
            parts[-3] = '40_fully_seen'
            results_path = os.path.join(Path(*parts), 'results', 'results.json')
            if not os.path.exists(results_path):
                return None
            result40 = Result(None, results_path, None, None)
            if not result40.predictive_model_params:
                return None
            fitted_predictive_model = result40.fitted_model()

        return fitted_predictive_model > np.min(fitted_predictive_model) + ((np.max(fitted_predictive_model) - np.min(fitted_predictive_model)) / 10)

    def run_analysis(self):

        self.full_id_acc = float(self.exp_data.eval_data.full_validation_correct.arr.mean())
        self.partial_id_acc = float(self.exp_data.eval_data.partial_base_correct.arr.mean())
        self.partial_ood_acc = float(self.exp_data.eval_data.partial_ood_correct.arr.mean())

        fitted_predictive_model = self.fitted_model()

        partial_heatmap, base_mask, full_heatmap = self.generate_heatmaps()

        generalizable_mask = self.get_generalizable_mask(fitted_predictive_model)

        if generalizable_mask is None:
            return

        self.exp_data.partial_ood_frame.df['generalizable'] = np.isin(self.exp_data.partial_ood_frame.df.image_idx.values,
                                                                      np.where(generalizable_mask.flatten())[0])
        self.exp_data.full_validation_frame.df['base'] = get_base_mask(self.exp_data.full_validation_frame.df,
                                                                       self.exp_data.base_orientations)
        self.exp_data.full_validation_frame.df['generalizable'] = np.isin(self.exp_data.full_validation_frame.df.image_idx.values,
                                                                          np.where(generalizable_mask.flatten())[0])

        self.partial_base_acc = float(self.exp_data.eval_data.partial_base_correct.arr.mean())
        self.partial_generalizable_acc = float(self.exp_data.eval_data.partial_ood_correct.arr[
                                                   self.exp_data.partial_ood_frame.df[
                                                       self.exp_data.partial_ood_frame.df.generalizable].index].mean())
        self.partial_non_generalizable_acc = float(self.exp_data.eval_data.partial_ood_correct.arr[
                                                       self.exp_data.partial_ood_frame.df[
                                                           ~self.exp_data.partial_ood_frame.df.generalizable].index].mean())

        full_only_frame = self.full_only_frame()

        self.full_base_acc = float(self.exp_data.eval_data.full_validation_correct.arr[
                                       full_only_frame[full_only_frame.base].index].mean())
        self.full_generalizable_acc = float(self.exp_data.eval_data.full_validation_correct.arr[
                                                full_only_frame[full_only_frame.generalizable &
                                                                ~full_only_frame.base].index].mean())
        self.full_non_generalizable_acc = float(self.exp_data.eval_data.full_validation_correct.arr[
                                                    full_only_frame[~full_only_frame.generalizable &
                                                                    ~full_only_frame.base].index].mean())

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Mean of empty slice')
            downsampled_full_heatmap = np.nanmean(full_heatmap.arr.reshape(2, 16, 2, 16, 2, 16), axis=(0, 2, 4))
        nan_elements = np.stack(np.where(np.isnan(downsampled_full_heatmap))).T
        for nan_element in nan_elements:
            downsampled_full_heatmap[nan_element] = np.nanmean(downsampled_full_heatmap[max(nan_element[0] - 1, 0):
                                                                                        nan_element[0] + 2,
                                                              max(nan_element[1] - 1, 0): nan_element[1] + 2,
                                                              max(nan_element[2] - 1, 0): nan_element[2] + 2])
        self.id_corr = corr(downsampled_full_heatmap,
                            np.mean(partial_heatmap.arr.reshape(2, 16, 2, 16, 2, 16), axis=(0, 2, 4)))

        max_act = np.stack([np.abs(act.arr).max(axis=0) for act in self.exp_data.eval_data.activations]).max(axis=0)
        max_act_nonzero = max_act != 0

        for activation in self.exp_data.eval_data.activations:
            activation.arr = np.divide(activation.arr[:, max_act_nonzero],
                                       max_act[max_act_nonzero],
                                       out=np.zeros((activation.arr.shape[0], np.sum(max_act_nonzero))))

        pg_acts, pn_acts = self.get_ood_activations(self.exp_data.partial_instances,
                                                    self.exp_data.partial_ood_frame,
                                                    self.exp_data.eval_data.partial_ood_activations)

        fg_acts, fn_acts = self.get_ood_activations(self.exp_data.full_instances,
                                                    self.exp_data.full_validation_frame,
                                                    self.exp_data.eval_data.full_validation_activations)

        pb_acts = self.get_base_activations(self.exp_data.partial_instances,
                                            self.exp_data.partial_base_frame,
                                            self.exp_data.eval_data.partial_base_activations)

        fb_acts = self.get_base_activations(self.exp_data.full_instances,
                                            self.exp_data.full_validation_frame,
                                            self.exp_data.eval_data.full_validation_activations)

        acts = {'fb': fb_acts,
                'pb': pb_acts,
                'fg': fg_acts,
                'fn': fn_acts,
                'pg': pg_acts,
                'pn': pn_acts}

        # TODO If we use networks that don't only have ReLU activations, we might have negative normalized values here.
        #  This should be changed such that the absolute value is above some threshold

        sorted_acts = np.sort(np.hstack([v.flatten() for v in acts.values()]))
        threshold = sorted_acts[int(sorted_acts.shape[0] * 0.8)]
        max_act = sorted_acts[-1]

        thresh_mask = {k: v > threshold for k, v in acts.items()}

        self.selectivity = {k: v.mean() for k, v in thresh_mask.items()}

        self.invariance = {f'{k1}_{k2}': np.nanmean(inv_sel_score(acts[k1],
                                                                  thresh_mask[k1],
                                                                  acts[k2],
                                                                  thresh_mask[k2])) for k1, k2 in [('fb', 'fg'),
                                                                                                   ('fb', 'fn'),
                                                                                                   ('pb', 'pg'),
                                                                                                   ('pb', 'pn')]}

        self.save()
        del self.exp_data

    @staticmethod
    def get_ood_activations(instance_list, frame, activations):
        ret = np.full((2, len(instance_list), activations.arr.shape[1]), np.nan)

        df = frame.df[~frame.df.base] if 'base' in frame.df.columns else frame.df

        for (generalizable, instance_name), indices in df.groupby(['generalizable', 'instance_name'],
                                                                  sort=False).groups.items():
            ret[int(~generalizable), instance_list.index(instance_name)] = np.mean(activations.arr[indices], axis=0)

        return ret[0], ret[1]

    @staticmethod
    def get_base_activations(instance_list, frame, activations):
        ret = np.full((len(instance_list), activations.arr.shape[1]), np.nan)

        df = frame.df[frame.df.base] if 'base' in frame.df.columns else frame.df

        for instance_name, indices in df.groupby('instance_name', sort=False).groups.items():
            ret[instance_list.index(instance_name)] = np.mean(activations.arr[indices], axis=0)
        return ret

    def finalize_results_frame(self, data):
        frame = pd.DataFrame(data)
        frame['Instances Fully Seen'] = self.num_fully_seen
        frame['Run'] = self.run
        frame['Num'] = self.num
        frame['Model Type'] = self.exp_data.model_type
        frame['Half Data'] = self.exp_data.half_data
        frame['Pretrained'] = self.exp_data.pretrained
        frame['Augmented'] = self.exp_data.augment
        frame['Loss'] = self.exp_data.loss
        frame['Free Axis'] = self.free_axis if self.free_axis != 'hole' else "α'"
        match self.category:
            case 'plane':
                category = 'Airplane'
            case 'car':
                category = 'Car'
            case 'SM':
                category = 'SM'
            case 'plane -> SM':
                category = 'Airplane\n↓\nSM'
            case 'SM -> plane':
                category = 'SM\n↓\nAirplane'
        frame['Category'] = category
        return frame

    @property
    def accuracy_frame(self):
        return self.finalize_results_frame({'Instance': (['Fully Seen'] * 5) + (['Partially Seen'] * 5),
                                            'Orientation': ['In-Distribution',
                                                            'Out-of-Distribution',
                                                            'Base',
                                                            'Generalizable',
                                                            'Non-Generalizable'] * 2,
                                            'Accuracy': [self.full_id_acc,
                                                         self.full_ood_acc,
                                                         self.full_base_acc,
                                                         self.full_generalizable_acc,
                                                         self.full_non_generalizable_acc,
                                                         self.partial_id_acc,
                                                         self.partial_ood_acc,
                                                         self.partial_base_acc,
                                                         self.partial_generalizable_acc,
                                                         self.partial_non_generalizable_acc]})

    @property
    def correlation_frame(self):
        return self.finalize_results_frame({'Predictive Model Component': ['Random Uniform',
                                                                           'In-Distribution',
                                                                           'Small-Angle',
                                                                           'In-Plane',
                                                                           'Small-Angle + In-Plane',
                                                                           'Silhouette',
                                                                           'All Components'],
                                            'Correlation': [self.unif_corr,
                                                            self.id_corr,
                                                            self.a_corr,
                                                            self.e_corr,
                                                            self.ae_corr,
                                                            self.f_ae_corr,
                                                            self.all_corr]})

    @staticmethod
    def k_to_instance(k):
        match k:
            case 'f':
                return 'Fully Seen'
            case 'p':
                return 'Partially Seen'
        raise Exception(f'{k} does not match any Instance Set')

    @staticmethod
    def k_to_orientation(k):
        match k:
            case 'b':
                return 'Base'
            case 'g':
                return 'Generalizable'
            case 'n':
                return 'Non-Generalizable'
        raise Exception(f'{k} does not match any Orientation Set')

    @property
    def selectivity_frame(self):
        keys = self.selectivity.keys()
        return self.finalize_results_frame({'Instance Set': [self.k_to_instance(k[0]) for k in keys],
                                            'Orientation Set': [self.k_to_orientation(k[1]) for k in keys],
                                            'Selectivity': [self.selectivity[k] for k in keys]})

    @property
    def invariance_frame(self):
        keys = self.invariance.keys()
        return self.finalize_results_frame({'Instance': [self.k_to_instance(k[0]) for k in keys],
                                            'Orientation': [self.k_to_orientation(k[4]) for k in keys],
                                            'Invariance': [self.invariance[k] for k in keys]})

    @classmethod
    def from_job_i(cls, project_path, storage_path, job_i, num_runs=5):
        exp_data = ExpData.from_job_i(project_path, storage_path, job_i, num_runs)
        d, f = cls.results_dir(exp_data)
        r = cls(d, f, exp_data, project_path)
        r.exp_data = exp_data
        return r

    @classmethod
    def from_exp_data(cls, exp_data, project_path):
        d, f = cls.results_dir(exp_data)
        r = cls(d, f, exp_data, project_path)
        r.exp_data = exp_data
        return r

    @classmethod
    def results_dir(cls, exp_data):
        d = os.path.join(exp_data.eval_dir, 'results')
        return d, os.path.join(d, 'results.json')

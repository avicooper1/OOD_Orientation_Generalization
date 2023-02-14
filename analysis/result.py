import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from utils.persistent_data_class import PersistentDataClass, ExpData
from utils.persistent_data_object import Arr
from utils.tools import get_base_mask
from predictive_function.tools import pf_func_path, sigmoid_on_pf


def corr(arr1, arr2):
    try:
        return float(np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1])
    except:
        return None


def inv_sel_score(pre_act, pre_mask, post_act, post_mask):
    
    # return 1 - np.abs(np.divide((post_act - pre_act), (post_act + pre_act),
    #                             out=np.full(np.broadcast(pre_act, post_act).shape, 0.0),
    #                             where=((post_act + pre_act) != 0)))
    
    return 1 - np.abs(np.divide((post_act - pre_act), (post_act + pre_act),
                                out=np.full(np.broadcast(pre_act, post_act).shape, np.nan),
                                where=((post_act + pre_act) != 0) & (pre_mask | post_mask)))

def calc_selectivity(pre_act, pre_mask, post_act, post_mask):
    expanded_pre_act = pre_act[np.newaxis]
    expanded_post_act = post_act[:,np.newaxis]
    score = inv_sel_score(expanded_pre_act, None, expanded_post_act, None)
    return np.average(score, weights=expanded_pre_act + expanded_post_act + 0.0000000000001, axis=1)
   
#     # score = inv_sel_score(pre_act[np.newaxis], pre_mask[np.newaxis], post_act[:, np.newaxis], post_mask[:, np.newaxis])
#     # print(np.nanmax(score, axis=(1, 2)).shape)
#     # return np.nanmean(np.nanmax(score, axis=(1, 2)))

#     # diffs = np.abs(pre_act[np.newaxis] - post_act[:, np.newaxis])
#     # 
#     # return 1 - np.divide(diffs.mean(axis=(0, 1)), diffs.max(axis=(0, 1)), where=diffs.max(axis=(0, 1))!=0)

#     pre_maxes = pre_act.argmax(axis=0, keepdims=True)
#     post_maxes = post_act.argmax(axis=0, keepdims=True)
#     score = inv_sel_score(np.take_along_axis(pre_act, pre_maxes, axis=0),
#                          np.take_along_axis(pre_mask, pre_maxes, axis=0),
#                          np.take_along_axis(post_act, post_maxes, axis=0),
#                          np.take_along_axis(post_mask, post_maxes, axis=0))
#     print(np.mean(np.isnan(score)))
#     return np.nanmean(score)
    return inv_sel_score(pre_act[np.newaxis], pre_mask[np.newaxis], post_act[:, np.newaxis], post_mask[:, np.newaxis], False)
    # return np.mean(pre_mask[np.newaxis] & post_mask[:, np.newaxis]) / np.mean(pre_mask[np.newaxis] | post_mask[:, np.newaxis])
    # return np.mean(np.sqrt(pre_mask[np.newaxis] * post_mask[:, np.newaxis]) / (pre_mask[np.newaxis] + post_mask[:, np.newaxis]))


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
    model_type: str = None
    loss: str = None
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
    all_corr: float = None

    selectivity: dict = field(default_factory=lambda: {})
    invariance: dict = field(default_factory=lambda: {})

    partial_heatmap: Arr = None
    base_mask: Arr = None

    def initial_init(self):

        os.makedirs(self.d)
        self.num = self.exp_data.num
        self.num_fully_seen = self.exp_data.num_fully_seen
        self.run = self.exp_data.run
        self.free_axis = self.exp_data.free_axis
        self.category = self.exp_data.full_category if self.exp_data.full_category == self.exp_data.partial_category else f'{self.exp_data.full_category} -> {self.exp_data.partial_category}'
        self.model_type = self.exp_data.model_type
        self.loss = self.exp_data.loss
        
        try:
            self.full_id_acc = float(self.exp_data.eval_data.full_validation_correct.arr.mean())
            self.partial_id_acc = float(self.exp_data.eval_data.partial_base_correct.arr.mean())
            self.partial_ood_acc = float(self.exp_data.eval_data.partial_ood_correct.arr.mean())
        except:
            exit()
            
        pred_func = np.load(pf_func_path(self.free_axis, self.project_path))
        pred_func_sigmoid = sigmoid_on_pf(pred_func, self.free_axis)

        match self.free_axis:
            case 'α':
                pred_func_thresh = 0.15
            case 'β':
                pred_func_thresh = 0.2
            case 'γ':
                pred_func_thresh = 0.12
            case 'hole':
                pred_func_thresh = 0.01

        pred_func_mask = np.max(pred_func_sigmoid, axis=0) > pred_func_thresh

        self.exp_data.partial_ood_frame.df['generalizable'] = np.isin(self.exp_data.partial_ood_frame.df.image_idx.values,
                                                                      np.where(pred_func_mask.flatten())[0])

        self.exp_data.full_validation_frame.df['base'] = get_base_mask(self.exp_data.full_validation_frame.df, self.exp_data.base_orientations)
        self.exp_data.full_validation_frame.df['generalizable'] = np.isin(self.exp_data.full_validation_frame.df.image_idx.values,
                                                                          np.where(pred_func_mask.flatten())[0])

        self.partial_heatmap = Arr(self.d, 'partial_heatmap')
        self.partial_heatmap.arr = np.zeros(self.exp_data.dataset_resolution ** 3)

        self.base_mask = Arr(self.d, 'base_mask')
        self.base_mask.arr = np.zeros(self.exp_data.dataset_resolution ** 3).astype(bool)

        for base, frame, correct in ((True, self.exp_data.partial_base_frame, self.exp_data.eval_data.partial_base_correct),
                                     (False, self.exp_data.partial_ood_frame, self.exp_data.eval_data.partial_ood_correct)):
            for flat_cubelet_i, indices in frame.df.groupby('image_idx', sort=False).indices.items():
                self.partial_heatmap.arr[flat_cubelet_i] = correct.arr[indices].mean()

            if base:
                self.base_mask.arr[np.where(self.partial_heatmap.arr)] = True

        self.partial_heatmap.arr = self.partial_heatmap.arr.reshape((self.exp_data.dataset_resolution,) * 3)
        self.base_mask.arr = self.base_mask.arr.reshape((self.exp_data.dataset_resolution,) * 3)

        self.full_heatmap = Arr(self.d, 'full_heatmap')
        self.full_heatmap.arr = np.empty(self.exp_data.dataset_resolution ** 3)
        self.full_heatmap.arr[:] = np.nan

        full_only_frame = self.exp_data.full_validation_frame.df[self.exp_data.full_validation_frame.df.instance_name.isin(self.exp_data.full_instances)]

        for flat_cubelet_i, indices in full_only_frame.groupby('image_idx', sort=False).indices.items():
            self.full_heatmap.arr[flat_cubelet_i] = self.exp_data.eval_data.full_validation_correct.arr[indices].mean()
        self.full_heatmap.arr = self.full_heatmap.arr.reshape((self.exp_data.dataset_resolution,) * 3)

        self.partial_base_acc = float(self.exp_data.eval_data.partial_base_correct.arr.mean())
        self.partial_generalizable_acc = float(self.exp_data.eval_data.partial_ood_correct.arr[self.exp_data.partial_ood_frame.df[self.exp_data.partial_ood_frame.df.generalizable].index].mean())
        self.partial_non_generalizable_acc = float(self.exp_data.eval_data.partial_ood_correct.arr[self.exp_data.partial_ood_frame.df[~self.exp_data.partial_ood_frame.df.generalizable].index].mean())

        self.full_base_acc = float(self.exp_data.eval_data.full_validation_correct.arr[full_only_frame[full_only_frame.base].index].mean())
        self.full_generalizable_acc = float(self.exp_data.eval_data.full_validation_correct.arr[full_only_frame[full_only_frame.generalizable & ~full_only_frame.base].index].mean())
        self.full_non_generalizable_acc = float(self.exp_data.eval_data.full_validation_correct.arr[full_only_frame[~full_only_frame.generalizable & ~full_only_frame.base].index].mean())
        
        self.unif_corr, self.a_corr, self.e_corr, self.ae_corr, self.all_corr, self.id_corr = (corr(self.partial_heatmap.arr, np.random.sample(self.partial_heatmap.arr.shape)),
                                                                                 corr(self.partial_heatmap.arr, pred_func_sigmoid[0]),
                                                                                 corr(self.partial_heatmap.arr, pred_func_sigmoid[1]),
                                                                                 corr(self.partial_heatmap.arr,
                                                                                      np.max(pred_func_sigmoid[[0, 1]], axis=0)),
                                                                                 corr(self.partial_heatmap.arr,
                                                                                      np.max(pred_func_sigmoid, axis=0)),
                                                                                corr(np.nanmean(self.full_heatmap.arr.reshape(2, 16, 2, 16, 2, 16), axis=(1,3,5)), np.mean(self.partial_heatmap.arr.reshape(2, 16, 2, 16, 2, 16), axis=(1, 3, 5))))

        max_act = np.stack([np.abs(activation.arr).max(axis=0) for activation in self.exp_data.eval_data.activations]).max(axis=0)

        for i, activation in enumerate(self.exp_data.eval_data.activations):
            activation.arr = np.divide(activation.arr, max_act, out=np.zeros(activation.arr.shape), where=max_act != 0)

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

        # TODO If we use networks that don't only have ReLU activations, we might have negative normalized values here. This should be changed such that the absolute value is above some threshold

        sorted_acts = np.sort(np.hstack([v.flatten() for v in acts.values()]))
        threshold = sorted_acts[int(sorted_acts.shape[0] * 0.95)]

        # thresh_mask = {k: np.sort(v.flatten())[int(v.flatten().shape[0] * 0.99)] for k, v in acts.items()}
        thresh_mask = {k: v > threshold for k, v in acts.items()}

        self.selectivity = {k: v.mean() for k, v in thresh_mask.items()}
        
        self.invariance = {f'{k1}_{k2}': np.nanmean(inv_sel_score(acts[k1],
                                                                  thresh_mask[k1],
                                                                  acts[k2],
                                                                  thresh_mask[k2])) for k1, k2 in [('fb', 'fg'),
                                                                                                   ('fb', 'fn'),
                                                                                                   ('pb', 'pg'),
                                                                                                   ('pb', 'pn')]}
        
        # def only_one(a, b, t):
        #     a_any_inv_mask = np.any(~np.isnan(a), axis=0)
        #     b_any_inv_mask = np.any(~np.isnan(b), axis=0)
        #     if t == 'first':
        #         return a_any_inv_mask & ~b_any_inv_mask
        #     if t == 'second':
        #         return ~a_any_inv_mask & b_any_inv_mask
        #     if t == 'both':
        #         return a_any_inv_mask & b_any_inv_mask
        #     if t == 'neither':
        #         return ~a_any_inv_mask & ~b_any_inv_mask
        
        # self.invariance = {k: np.nanmean(v) for k, v in inv.items()}
        # self.invariance_only_partial = {k: np.nanmean(inv[k][:, only_one(inv['fb_fg'], inv[k], 'second')]) for k in ['pb_pg', 'pb_pn']}
        # self.invariance_only_full = {k: np.nanmean(inv['fb_fg'][:, only_one(inv['fb_fg'], inv[k], 'first')]) for k in ['pb_pg', 'pb_pn']}
        # self.invariance_both = {k: np.nanmean(inv['fb_fg'][:, only_one(inv['fb_fg'], inv[k], 'both')]) for k in ['pb_pg', 'pb_pn']}
        # self.invariance_neither = {k: np.nanmean(inv['fb_fg'][:, only_one(inv['fb_fg'], inv[k], 'neithersl')]) for k in ['pb_pg', 'pb_pn']}
    
        
        # self.mixed_selectivity = {f'{k1}_{k2}':
        #                           np.nanmean(calc_selectivity(acts[k1],
        #                                            thresh_mask[k1],
        #                                            acts[k2],
        #                                            thresh_mask[k2])) for k1, k2 in [('fb', 'pb'),
        #                                                                            ('fg', 'pg'),
        #                                                                            ('fn', 'pn')]}
#         self.acts = acts
#         self.thresh_mask = thresh_mask
        
#         import pickle 
#         with open('/home/avic/result2', 'wb') as f:
#             pickle.dump(self, f)

        self.partial_heatmap.dump()
        self.base_mask.dump()
        self.full_heatmap.dump()

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
        frame['Model Type'] = self.model_type
        frame['Loss'] = self.loss
        frame['Free Axis'] = self.free_axis
        frame['Category'] = self.category
        return frame

    @property
    def accuracy_frame(self):
        return self.finalize_results_frame({'Instance': (['Full'] * 5) + (['Partial'] * 5),
                                            'Orientation': ['In Distribution', 
                                                            'Out of Distribution',
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
                                                                           'Small Angle',
                                                                           'In Plane',
                                                                           'Small Angle + In Plane',
                                                                           'All Components'],
                                            'Correlation': [self.unif_corr,
                                                            self.id_corr,
                                                            self.a_corr,
                                                            self.e_corr,
                                                            self.ae_corr,
                                                            self.all_corr]})

    def k_to_instance(self, k):
        match k:
            case 'f':
                return 'Full'
            case 'p':
                return 'Partial'
        raise Exception(f'{k} does not match any Instance Set')

    def k_to_orientation(self, k):
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
    
    @property
    def invariance_only_partial_frame(self):
        keys = self.invariance_only_partial.keys()
        return self.finalize_results_frame({'Instance': [self.k_to_instance(k[0]) for k in keys],
                                            'Orientation': [self.k_to_orientation(k[4]) for k in keys],
                                            'Invariance': [self.invariance_only_partial[k] for k in keys]})
    
    @property
    def invariance_only_full_frame(self):
        keys = self.invariance_only_full.keys()
        return self.finalize_results_frame({'Instance': [self.k_to_instance(k[0]) for k in keys],
                                            'Orientation': [self.k_to_orientation(k[4]) for k in keys],
                                            'Invariance': [self.invariance_only_full[k] for k in keys]})
    
    @property
    def invariance_both_frame(self):
        keys = self.invariance_both.keys()
        return self.finalize_results_frame({'Instance': [self.k_to_instance(k[0]) for k in keys],
                                            'Orientation': [self.k_to_orientation(k[4]) for k in keys],
                                            'Invariance': [self.invariance_both[k] for k in keys]})
    
    @property
    def invariance_neither_frame(self):
        keys = self.invariance_neither.keys()
        return self.finalize_results_frame({'Instance': [self.k_to_instance(k[0]) for k in keys],
                                            'Orientation': [self.k_to_orientation(k[4]) for k in keys],
                                            'Invariance': [self.invariance_neither[k] for k in keys]})

    @property
    def mixed_selectivity_frame(self):
        keys = self.mixed_selectivity.keys()
        return self.finalize_results_frame({'Orientation': [self.k_to_orientation(k[1])  for k in keys],
                                            'Mixed Selectivity': [self.mixed_selectivity[k] for k in keys]})

    @classmethod
    def from_job_i(cls, project_path, storage_path, job_i, num_runs=5):
        exp_data = ExpData.from_job_i(project_path, storage_path, job_i, num_runs)
        d, f = cls.results_dir(exp_data)
        return cls(d, f, exp_data, project_path)

    @classmethod
    def from_exp_data(cls, exp_data, project_path):
        d, f = cls.results_dir(exp_data)
        return cls(d, f, exp_data, project_path)

    @classmethod
    def results_dir(cls, exp_data):
        d = os.path.join(exp_data.eval_dir, 'results')
        return d, os.path.join(d, 'results.json')

import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import minmax_scale
from utils.persistent_data_class import PersistentDataClass, ExpData
from utils.persistent_data_object import Arr
from utils.tools import get_base_mask
from predictive_function.tools import pf_func_path, sigmoid_on_pf


def corr(arr1, arr2):
    return np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]

def upper_weight(arr):
    return np.mean(~np.isnan(arr) & (arr > 0.8))

def calc_invariance(pre_act, pre_mask, post_act, post_mask):
    
    diff_dim0 = pre_act.shape[0] != post_act.shape[0]
    
    if diff_dim0:
        m_pre_act = pre_act[:, np.newaxis]
        m_pre_mask = pre_mask[:, np.newaxis]
        m_post_act = post_act[np.newaxis, :]
        m_post_mask = post_mask[np.newaxis, :]
    else:
        m_pre_act, m_pre_mask, m_post_act, m_post_mask = pre_act, pre_mask, post_act, post_mask
    
    inv = 1 - np.abs(np.divide((m_post_act - m_pre_act), (m_post_act + m_pre_act),
                               out=np.zeros(np.broadcast(m_pre_act, m_post_act).shape).astype(float),
                               where=(m_post_act + m_pre_act)!=0))
    mask = m_pre_mask | m_post_mask
    inv[mask] = np.nan
    
    if diff_dim0:
        inv = np.nanmax(inv, axis=1)
    
    return inv


@dataclass
class Result(PersistentDataClass):

    d: str
    store_path: str
    exp_data: ExpData
    project_path: str

    num_fully_seen: int = None
    run: int = None
    free_axis: str = None
    category: str = None
    id_acc: float = None
    ood_acc: float = None
    pred_acc: float = None
    xpred_acc: float = None
    unif_corr: float = None
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
        self.num_fully_seen = self.exp_data.num_fully_seen
        self.run = self.exp_data.run
        # The following line can be improved once saved config files for exps can be assumed to have .free_axis
        self.free_axis = pd.read_csv(os.path.join(self.project_path, 'exps.csv')).iloc[self.exp_data.num].free_axis
        self.category = self.exp_data.full_category if self.exp_data.full_category == self.exp_data.partial_category else f'{self.exp_data.full_category} -> {self.exp_data.partial_category}'
        
        self.id_acc = self.exp_data.eval_data.partial_base_correct.arr.mean()
        self.ood_acc = self.exp_data.eval_data.partial_ood_correct.arr.mean()

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

        self.exp_data.partial_ood_frame.df['in_pred_func'] = np.isin(self.exp_data.partial_ood_frame.df.image_idx.values,
                                                                     np.where(pred_func_mask.flatten())[0])
        
        self.exp_data.full_validation_frame.df['in_base'] = get_base_mask(self.exp_data.full_validation_frame.df, self.exp_data.base_orientations)
        self.exp_data.full_validation_frame.df['in_pred_func'] = np.isin(self.exp_data.full_validation_frame.df.image_idx.values,
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

        self.pred_acc, self.xpred_acc = self.partial_heatmap.arr[pred_func_mask].mean(), self.partial_heatmap.arr[~pred_func_mask].mean()

        self.unif_corr, self.a_corr, self.e_corr, self.ae_corr, self.all_corr = (corr(self.partial_heatmap.arr, np.random.sample(self.partial_heatmap.arr.shape)),
                                                                corr(self.partial_heatmap.arr, pred_func_sigmoid[0]),
                                                                 corr(self.partial_heatmap.arr, pred_func_sigmoid[1]),
                                                                 corr(self.partial_heatmap.arr,
                                                                      np.max(pred_func_sigmoid[[0, 1]], axis=0)),
                                                                 corr(self.partial_heatmap.arr,
                                                                      np.max(pred_func_sigmoid, axis=0)))
        
        max_act = np.stack([np.abs(activation.arr).max(axis=0) for activation in self.exp_data.eval_data.activations]).max(axis=0)

        for i, activation in enumerate(self.exp_data.eval_data.activations):
            activation.arr /= max_act
            activation.arr[:, np.where(max_act == 0)] = 0
                
        pp_acts, pn_acts = self.get_ood_activations(self.exp_data.partial_instances,
                                                    self.exp_data.partial_ood_frame,
                                                    self.exp_data.eval_data.partial_ood_activations)
        
        fp_acts, fn_acts = self.get_ood_activations(self.exp_data.full_instances,
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
                'fp': fp_acts, 
                'fn': fn_acts, 
                'pp': pp_acts, 
                'pn': pn_acts}
        
        # thresh_mask = {k: v > activation_threshold for k, v in acts.items()}
        # TODO If we use networks that don't only have ReLU activations, we might have negative normalized values here. This should be changed such that the absolute value is above some threshold
        thresh_mask = {k: v > 0.2 for k, v in acts.items()}
        self.selectivity = {k: v.mean() for k, v in thresh_mask.items()}
        self.invariance = {f'{k1}_{k2}': upper_weight(calc_invariance(acts[k1], thresh_mask[k1], acts[k2], thresh_mask[k2])) for k1, k2 in combinations(acts.keys(), 2)}
        

        self.partial_heatmap.dump()
        self.base_mask.dump()
        
    def get_ood_activations(self, instance_list, frame, activations):
        ret = np.zeros((2, len(instance_list), 512))
        
        df = frame.df[~frame.df.in_base] if 'in_base' in frame.df.columns else frame.df
        
        for (in_pred_func, instance_name), indices in frame.df.groupby(['in_pred_func', 'instance_name'], 
                                                                       sort=False).indices.items():
            ret[int(~in_pred_func), instance_list.index(instance_name)] = np.mean(activations.arr[indices], axis=0)
            
        return ret[0], ret[1]
    
    def get_base_activations(self, instance_list, frame, activations):
        ret = np.zeros((len(instance_list), 512))
        
        df = frame.df[frame.df.in_base] if 'in_base' in frame.df.columns else frame.df
        
        for instance_name, indices in df.groupby('instance_name', sort=False).indices.items():
            ret[instance_list.index(instance_name)] = np.mean(activations.arr[indices], axis=0)
        return ret
        
    def finalize_results_frame(self, data):
        frame = pd.DataFrame(data)
        frame['Instances Fully Seen'] = self.num_fully_seen
        frame['Run'] = self.run
        frame['Free Axis'] = self.free_axis
        frame['Category'] = self.category
        return frame
    
    @property
    def id_ood_accuracy_frame(self):
        return self.finalize_results_frame({'Orientation Set': ['In Distribution', 'Out Of Distribution'],
                                            'Accuracy': [self.id_acc, self.ood_acc]})
    
    @property
    def accuracy_frame(self):
        return self.finalize_results_frame({'Orientation Set': ['Predicted', 'Not-Predicted'],
                                            'Accuracy': [self.pred_acc, self.xpred_acc]})
    
    @property
    def correlation_frame(self):
        return self.finalize_results_frame({'Predictive Model Component': ['Random Uniform', 'Small Angle', 'In Plane', 'Small Angle + In Plane', 'All Components'],
                                            'Correlation': [self.unif_corr,
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
        raise Exception(f'{k} does not match any for any Instance Set')
        
    def k_to_orientation(self, k):
        match k:
            case 'b':
                return 'Base'
            case 'p':
                return 'Predicted'
            case 'n':
                    return 'Not-Predicted'
        raise Exception(f'{k} does not match any for any Orientation Set')
    
    @property
    def selectivity_frame(self):
        keys = self.selectivity.keys()
        return self.finalize_results_frame({'Instance Set': [self.k_to_instance(k[0]) for k in keys],
                                            'Orientation Set': [self.k_to_orientation(k[1]) for k in keys],
                                            'Selectivity': [self.selectivity[k] for k in keys]})
    
    @property
    def invariances_frame(self):
        keys = self.invariance.keys()
        return self.finalize_results_frame({'Instance Transform': [f'{self.k_to_instance(k[0])} -> {self.k_to_instance(k[3])}' for k in keys],
                                            'Orientation Transform': [f'{self.k_to_orientation(k[1])} -> {self.k_to_orientation(k[4])}' for k in keys],
                                            'Invariance': [self.invariance[k] for k in keys]})

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

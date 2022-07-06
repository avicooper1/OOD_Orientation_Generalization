import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import pandas as pd
from tqdm import tqdm

itoaxis = {0:'object_x', 1:'object_y', 2:'object_z'}

def convert_tuple_to_axes(t):
    if t == '(0, 1)':
        axes =  [0, 1, 2]
    if t == '(1, 2)':
        axes = [1, 2, 0]
    if t == '(0, 2)':
        axes = [0, 2, 1]
    return [itoaxis[i] for i in axes]

def get_heatmap_cell_ranges(num_restricted_axis_bins, num_unrestricted_axis_bins, unrestricted_axis):
    longtitude = num_restricted_axis_bins + 1
    latitude = num_unrestricted_axis_bins // 2
    r = 1

    dim0, delta_theta = np.linspace(-np.pi, np.pi, longtitude, retstep=True)
    delta_S = delta_theta / latitude

    dim1 = 1 - np.arange(2 * latitude + 1) * delta_S / (r ** 2 * delta_theta)
    dim1 = np.arccos(dim1)
    dim1 = (dim1 - (np.pi / 2)) if unrestricted_axis == 1 else ((2 * dim1) - np.pi)

    unrestricted_axis_range = np.pi if unrestricted_axis != 1 else np.pi / 2
    dim2 = np.linspace(-unrestricted_axis_range, unrestricted_axis_range, num_unrestricted_axis_bins + 1)
    dim2 = dim2 + ((dim2[1] - dim2[0]) / 2)
    dim2[-1] = 100

    return enumerate(zip(dim0, dim0[1:])), enumerate(zip(dim1, dim1[1:])), enumerate(zip(dim2, dim2[1:]))

def twod_alginemnt(v1, v2):
  r1 = R.from_euler('zyx', v1[-1::-1]) #
  r2 = R.from_euler('zyx', v2[-1::-1])
  #print(r1.as_matrix())
  #print(r2.as_matrix())
  r3 = r2*r1.inv()
  a = r3.as_matrix()
  #print(a)

  val, v = linalg.eig(a)
  #print(int(np.where(np.round(val , 2) == 1)[0][-1]))
  #print(v)
  #print(val)
  idx = np.where(np.round(val , 6) == 1)[0]
  #print(idx)
  if len(idx) == 3:
    ax = [0,1,0]
  else:
    idx = int(idx[0])
    ax = np.array(v[:,idx])
  #print(ax)
  return np.dot(ax, [0,1,0]),  np.arccos((a.trace()-1)/2)

def range_mid(r):
    return r[0] + ((r[1] - r[0]) / 2)

def get_canonical_heatmaps(num_restricted_axis_bins, num_unrestricted_axis_bins, axes_order):
    dim0s, dim1s, dim2s = get_heatmap_cell_ranges(num_restricted_axis_bins, num_unrestricted_axis_bins, axes_order[-1])

    rot = np.zeros((num_unrestricted_axis_bins, num_restricted_axis_bins, num_restricted_axis_bins))
    rot_flipped = np.zeros(rot.shape)
    trace = np.zeros(rot.shape)
    trace_flipped = np.zeros(rot.shape)

    for d2i, d2 in dim2s:
        for d0i, d0 in dim0s:
            for d1i, d1 in dim1s:
                rot[d0i, d1i, d2i], trace[d0i, d1i, d2i] = twod_alginemnt((0,0,0), (range_mid(d0), range_mid(d1), range_mid(d2)))
                rot_flipped[d0i, d1i, d2i], trace_flipped[d0i, d1i, d2i] = twod_alginemnt((-np.pi, 0, 0),
                                                                          (range_mid(d0), range_mid(d1), range_mid(d2)))

    return rot, rot_flipped, trace, trace_flipped

    # return np.max(np.array([np.abs(rot[:, :, 0]), np.abs(rot_flipped[:, :, 0]),
    #        np.abs(np.pi-trace[:, :, 0])/np.pi, np.abs(np.pi-trace_flipped[:, :, 0])/np.pi]), axis=0)
    
    
def get_heatmap_cell_ranges2(num_cubelets):

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

    
    return dim0, dim1, dim2

def div_heatmap(df, activations, num_cubelets=20):
    dim0s, dim1s, dim2s = get_heatmap_cell_ranges2(num_cubelets)

    df['object_x_cat'] = pd.cut(df.object_x, dim0s).cat.codes
    df['object_y_cat'] = pd.cut(df.object_y, dim1s).cat.codes
    df['object_z_cat'] = pd.cut(df.object_z, dim2s).cat.codes
    df['model_cats'] = pd.Categorical(df.model_name, categories=df.model_name.unique(), ordered=True).codes

    groups = df.groupby([df.model_cats, df.object_x_cat, df.object_y_cat, df.object_z_cat])
    groups_count = groups.ngroups
    
    activations_heatmap = np.zeros((512, 50, num_cubelets, num_cubelets, num_cubelets), dtype=np.float32)
    for i, group in tqdm(enumerate(groups), total=groups_count):
        m, x, y, z = group[0][0], group[0][1], group[0][2], group[0][3]
        activations_heatmap[:, m, x, y, z] = np.mean(activations[group[1].index.tolist()], axis=0)

    return activations_heatmap
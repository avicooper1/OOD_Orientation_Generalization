import numpy as np
import itertools


def range_mid(r):
    return r[0] + ((r[1] - r[0]) / 2)

def get_heatmap_cell_ranges2(num_cubelets, as_ranges=True):

    assert num_cubelets % 2 == 0
    
    latitude = num_cubelets // 2

    dim0, delta_theta = np.linspace(-np.pi, np.pi, num_cubelets + 1, retstep=True)
    delta_S = delta_theta / latitude

    dim1 = 1-np.arange(2*latitude+1) * delta_S / (delta_theta)
    dim1 = np.arccos(dim1)
    dim1 = (dim1 - (np.pi / 2))

    dim2 = np.linspace(-np.pi, np.pi, num_cubelets + 1)

    if as_ranges:
        return np.array(list(itertools.product(*[np.stack([vs[:-1], vs[1:]]).T for vs in [dim0, dim1, dim2]]))).reshape(num_cubelets, num_cubelets, num_cubelets ,3, 2)
    else:
        return dim0, dim1, dim2
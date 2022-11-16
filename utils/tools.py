import numpy as np
from itertools import product
from functools import reduce
# It is not advisable to add further imports without some research first. This file is imported in the blender
# environment, which does not contain most python libraries. If a specific library is bundled with blender, it can be
# imported here without any problems. If not, it can be added to the /bin/ of blender, and then subsequently used


def range_mid(r):
    return r[0] + ((r[1] - r[0]) / 2)


def get_heatmap_cell_ranges(num_cubelets: int, as_ranges: bool = True):
    """ Return grid point locations for cubelets in the 3D cube.

    :param int num_cubelets: number of cubelets per dimension
    :param bool as_ranges: whether to return a tuple of points (the top left point of the cubelet), or (default) return
    a 3D np.array where each element is a range with the start and stop for each dimension of the cubelet
    :return: a 3-tuple of points or a np.array grid, as specified by `as_ranges`
    """
    assert num_cubelets % 2 == 0
    
    latitude = num_cubelets // 2

    dim0, delta_theta = np.linspace(-np.pi, np.pi, num_cubelets + 1, retstep=True)

    dim1 = np.arccos(1 - np.arange(2 * latitude + 1) * delta_theta / latitude / delta_theta) - (np.pi / 2)

    dim2 = np.linspace(-np.pi, np.pi, num_cubelets + 1)

    if as_ranges:
        return np.array(list(product(*[np.stack([vs[:-1], vs[1:]]).T for vs in [dim0, dim1, dim2]]))).reshape(num_cubelets, num_cubelets, num_cubelets, 3, 2)
    else:
        return [list(enumerate(zip(dim, dim[1:]))) for dim in [dim0, dim1, dim2]]
    
def get_base_mask(df, base_orientations):
    return reduce(lambda a, b: (df.object_x.between(*b[0]) & df.object_y.between(*b[1]) & df.object_z.between(*b[2])) | a, zip(*base_orientations), False)

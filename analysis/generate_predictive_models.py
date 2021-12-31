import sys
sys.path.append('/home/avic/Rotation-Generalization')
from tools import *

for x in [(2,0), (2,1), (2,2), (1,0), (0,0)]:
    res = np.array(get_canonical_heatmaps(20, x[0], x[1]))
    set_generated_canonical_heatmap(res, x[0] + x[1])
import numpy as np
from algorithm.ahp import ahp
from common.enum_common import weights_type
judgment_matrix = np.array([
    [1, 1/3, 3],
    [3, 1, 5],
    [1/3, 1/5, 1]
])

input_matrix = np.array([
    [0.8, 0.6, 0.9],
    [0.9, 0.8, 0.7],
    [0.7, 0.9, 0.8]
])

# 调用ahp函数
max_score_index = ahp(judgment_matrix, input_matrix, weights_type.GEOMETRY)
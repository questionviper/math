import numpy as np
from socks import method
from twisted.python.util import println

from algorithm import ahp
from common.enum_common import WeightsType
import input_data,judgment_matrix

# 得到各准则层的分数
b1_scores = ahp.ahp(input_data.b1_matrix, judgment_matrix.b1_judgment_matrix, WeightsType.EIGEN)
b2_scores = ahp.ahp(input_data.b2_matrix, judgment_matrix.b2_judgment_matrix, WeightsType.EIGEN)
b3_scores = ahp.ahp(input_data.b3_matrix, judgment_matrix.b3_judgment_matrix, WeightsType.EIGEN)
b4_scores = ahp.ahp(input_data.b4_matrix, judgment_matrix.b4_judgment_matrix, WeightsType.EIGEN)
b5_scores = ahp.ahp(input_data.b5_matrix, judgment_matrix.b5_judgment_matrix, WeightsType.EIGEN)
b6_scores = ahp.ahp(input_data.b6_matrix, judgment_matrix.b6_judgment_matrix, WeightsType.EIGEN)
# println(b1_scores)
# println(b2_scores)
# println(b3_scores)
# println(b4_scores)
# println(b5_scores)
# println(b6_scores)
# 将每个得分数组转置成列向量
b1_scores_col = b1_scores.reshape(-1, 1)
b2_scores_col = b2_scores.reshape(-1, 1)
b3_scores_col = b3_scores.reshape(-1, 1)
b4_scores_col = b4_scores.reshape(-1, 1)
b5_scores_col = b5_scores.reshape(-1, 1)
b6_scores_col = b6_scores.reshape(-1, 1)

# 使用 np.hstack 将所有得分列向量水平堆叠起来
all_scores_col = np.hstack((b1_scores_col, b2_scores_col, b3_scores_col, b4_scores_col, b5_scores_col, b6_scores_col))
final_score = ahp.ahp(all_scores_col,judgment_matrix.a_judgment_matrix,WeightsType.EIGEN)
print(final_score)
print(np.argmax(final_score)+1)
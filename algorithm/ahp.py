import numpy as np

from common.enum_common import normalization_type,weights_type
#一致性检验
def consistency_identity(weight,CR_standard=0.1):
    #求出特征值特征向量
    eigenvalues, eigenvectors = np.linalg.eig(weight)
    #最大特征值
    max_eigenvalues = max(eigenvalues)
    #求评价准则个数
    n = weight.shape[0]
    #求CI
    CI = (max_eigenvalues-n)/(n-1)

    if CI<=CR_standard:return True
    else: return False


#按行归一化
#参数说明：input是需要归一化的矩阵，type枚举类中的一个，axis归一化的维度：0-列，1-行
def normalization(input,axis,type=normalization_type.SUM_NORMALIZATION):
    input_tem = input
    if type == normalization_type.EXP_NORMALIZATION:
        #求出指数和
        input_tem = np.exp(input)
    elif type == normalization_type.SUM_NORMALIZATION:
        pass
    else:
        return input
    # 求和
    sums = np.sum(input_tem, axis=axis)
    # 归一化每一维度
    normalized_array = input_tem / sums[:, np.newaxis] if axis == 1 \
        else input_tem / sums[np.newaxis, :]
    return normalized_array

def ahp(input,judgment_matrix,weight_type):
    #判断矩阵归一化
    judgment_normalization = normalization(judgment_matrix, type=normalization_type.SUM_NORMALIZATION, axis = 0)
    if weight_type == weights_type.AVERAGE:
        # 算术平均法求和
        weights = np.mean(judgment_normalization, axis=1)
    elif weight_type == weights_type.GEOMETRY:
        # 几何平均法
        geometric_mean = np.prod(judgment_normalization, axis=0) ** (1 / judgment_normalization.shape[0])
        weights = geometric_mean / np.sum(geometric_mean)
    elif weight_type == weights_type.EIGEN:
        # 特征向量法
        eigenvalues, eigenvectors = np.linalg.eig(judgment_normalization)
        max_index = np.argmax(eigenvalues)  # 获取最大特征值的索引
        max_eigenvector = eigenvectors[:, max_index].real  # 获取对应的特征向量
        weights = max_eigenvector / np.sum(max_eigenvector)  # 归一化特征向量
    else:
        raise ValueError("Unsupported weight type")

    # 计算得分
    scores = np.dot(input, weights)

    # 确定得分最高的人的序号
    max_score_index = np.argmax(scores)

    print("Judgment Normalization:\n", judgment_normalization)
    print("Weights:", weights)
    print("Scores:", scores)
    print("Person with the highest score:", max_score_index + 1)  # +1 for 1-based index

    return max_score_index
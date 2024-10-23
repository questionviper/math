import numpy as np

from common.enum_common import NormalizationType,WeightsType


def generate_consistent_matrix(n):
    """
    Generate an n-dimensional consistent judgment matrix.

    Parameters:
    n (int): The dimension of the matrix.

    Returns:
    np.ndarray: A consistent judgment matrix.
    """
    np.random.seed()  # 可以设置一个固定的种子以确保可复现性，或者不设置以每次生成不同的矩阵
    while True:
        # 生成一个n*n的随机矩阵
        random_matrix = np.random.rand(n, n)
        # 转换为正互反矩阵
        judgment_matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                judgment_matrix[i, j] = random_matrix[i, j]
                judgment_matrix[j, i] = 1 / judgment_matrix[i, j]

        # 检查一致性
        if check_consistency(judgment_matrix, n) <= 0.1:
            return judgment_matrix


def check_consistency(matrix, n):
    """
    Check the consistency of a judgment matrix.

    Parameters:
    matrix (np.ndarray): The judgment matrix to check.
    n (int): The dimension of the matrix.

    Returns:
    float: The consistency ratio of the matrix.
    """
    if n==2 :return 0.0
    eigenvalues, _ = np.linalg.eig(matrix)
    max_eigenvalue = max(eigenvalues)
    CI = (max_eigenvalue - n) / (n - 1)
    RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45]  # Random Index values
    CR = CI / RI[n - 1]
    return CR


#按行归一化
#参数说明：input是需要归一化的矩阵，type枚举类中的一个，axis归一化的维度：0-列，1-行
def normalization(input, axis, type=NormalizationType.SUM_NORMALIZATION):
    input_tem = input
    if type == NormalizationType.EXP_NORMALIZATION:
        #求出指数和
        input_tem = np.exp(input)
    elif type == NormalizationType.SUM_NORMALIZATION:
        pass
    else:
        return input
    # 求和
    sums = np.sum(input_tem, axis=axis)
    # 归一化每一维度
    normalized_array = input_tem / sums[:, np.newaxis] if axis == 1 \
        else input_tem / sums[np.newaxis, :]
    return normalized_array

#判断矩阵转化为权重向量
def judge_to_weights(judgment,weight_type):
    if weight_type == WeightsType.AVERAGE:
        # 算术平均法求和
        weights = np.mean(judgment, axis=1)
    elif weight_type == WeightsType.GEOMETRY:
        # 几何平均法
        geometric_mean = np.prod(judgment, axis=0) ** (1 / judgment.shape[0])
        weights = geometric_mean / np.sum(geometric_mean)
    elif weight_type == WeightsType.EIGEN:
        # 特征向量法
        eigenvalues, eigenvectors = np.linalg.eig(judgment)
        max_index = np.argmax(eigenvalues)  # 获取最大特征值的索引
        max_eigenvector = eigenvectors[:, max_index].real  # 获取对应的特征向量
        weights = max_eigenvector / np.sum(max_eigenvector)  # 归一化特征向量
    else:
        raise ValueError("Unsupported weight type")
    return weights


def ahp(ahp_input, judgment_matrix, weight_type):
    input_normalization = normalization(ahp_input,0)
    #判断矩阵归一化
    judgment_normalization = normalization(judgment_matrix, axis = 0)
    #获得权重矩阵
    weights = judge_to_weights(judgment_normalization,weight_type)
    # 计算得分
    scores = np.dot(ahp_input, weights)

    # 确定得分最高的人的序号
    max_score_index = np.argmax(scores)

    # print("input_normalization:\n",input_normalization)
    # print("Judgment Normalization:\n", judgment_normalization)
    # print("Weights:", weights)
    # print("Scores:", scores)
    # print("Person with the highest score:", max_score_index + 1)  # +1 for 1-based index

    return scores
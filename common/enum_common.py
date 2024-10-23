from enum import Enum,auto

from statsmodels.iolib.summary import summary_top


# 归一化方法枚举类
class NormalizationType(Enum):
    SUM_NORMALIZATION = auto()
    EXP_NORMALIZATION = auto()

class WeightsType(Enum):
    AVERAGE = auto()#算术平均
    GEOMETRY = auto()#几何平均
    EIGEN = auto()#特征向量法
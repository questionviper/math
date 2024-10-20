from enum import Enum,auto

from statsmodels.iolib.summary import summary_top


# 归一化方法枚举类
class normalization_type(Enum):
    SUM_NORMALIZATION = auto()
    EXP_NORMALIZATION = auto()

class weights_type(Enum):
    AVERAGE = auto()#算术平均
    GEOMETRY = auto()#几何平均
    EIGEN = auto()#特征向量法
#%%
import os

import matplotlib
import matplotlib.pyplot as plt
from numpy import array, arange, ndarray, zeros, empty_like, \
    searchsorted, clip, unique


class Interpolate1D:
    """快速一维线性插值"""
    def __init__(self, x_, y_):
        assert len(x_) == len(y_), '自变量序列x_的长度应等于因变量序列y_的长度'
        x_ = array(x_)
        y_ = array(y_)
        assert x_.ndim==1, '自变量序列x_应为1维'
        assert y_.ndim==1, '因变量序列y_应为1维'
        assert unique(x_).size == x_.size, '自变量序列x_不应包含相同值'
        idx_ = x_.argsort()
        self.x_ = x_[idx_]
        self.y_ = y_[idx_]
        del x_, y_, idx_

    def __call__(self, x_: ndarray) -> ndarray:
        """插值"""
        xbase_ = self.x_
        ybase_ = self.y_
        # 找区间
        idx_ = searchsorted(xbase_, x_)  # shape同x
        # 处理边界
        idx_ = clip(idx_, 1, xbase_.size-1)
        idxLow_ = idx_ - 1
        idxHigh_ = idx_
        # 取点
        xLow_  = xbase_[idxLow_]
        xHigh_ = xbase_[idxHigh_]
        yLow_  = ybase_[idxLow_]
        yHigh_ = ybase_[idxHigh_]
        # 插值
        y_ = yLow_ + (x_ - xLow_) * (yHigh_ - yLow_) / (xHigh_ - xLow_)
        return y_

def triband_to_dense(band__: ndarray) -> ndarray:
    """三角阵的带band__ (3, N)  -> 稠密方阵K__ (N, N)"""
    N = band__.shape[1]
    K__ = zeros((N, N), dtype=band__.dtype)
    idx_ = arange(N)
    K__[idx_, idx_] = band__[1]                # 主对角线
    K__[idx_[:-1], idx_[1:]] = band__[0, 1:]   # 上对角线
    K__[idx_[1:], idx_[:-1]] = band__[2, :-1]  # 下对角线
    return K__

def Thomas_solve(
        band__: ndarray,  # (3, N) 三对角矩阵的带
        RHS__: ndarray,   # (N, M) 右端项
        overwrite=False,     # 是否覆盖band__、RHS__
        ) -> ndarray:
    """Thomas算法三对角方程组"""
    # 统一成2维
    squeeze = False
    if RHS__.ndim==1:
        RHS__ = RHS__[:, None]
        squeeze = True
    if overwrite:
        pass
    else:
        band__ = band__.copy()
        RHS__ = RHS__.copy()
    N = RHS__.shape[0]
    # 提取对角线
    du_ = band__[0, 1:]
    d_ = band__[1]
    dl_ = band__[2, :-1]
    # 前向消元
    for i in range(1, N):
        w = dl_[i-1]/d_[i - 1]
        d_[i] -= w*du_[i - 1]
        RHS__[i] -= w*RHS__[i-1]
    # 回代
    X__ = empty_like(RHS__)
    X__[-1] = RHS__[-1]/d_[-1]
    for i in range(N-2, -1, -1):
        X__[i] = (RHS__[i] - du_[i]*X__[i+1])/d_[i]
    return X__.squeeze() if squeeze else X__

def set_matplotlib():
    """设置matplotlib"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    matplotlib.use('Qt5Agg')  # TkAgg/Qt5Agg
    # plt.close(plt.figure())
    fontname = 'Times New Roman'                  # 字体
    plt.rcParams['font.serif'] = [fontname]       # 衬线字体
    plt.rcParams['font.sans-serif'] = [fontname]  # 无衬线字体
    plt.rcParams['font.size'] = 12  # 字号
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['axes.unicode_minus'] = False               # 正常显示负号
    plt.rcParams['mathtext.default'] = 'regular'             # 默认样式：正体、不加粗
    plt.rcParams['mathtext.rm'] = 'STIXGeneral:regular'      # 正体、不加粗
    plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'       # 斜体、不加粗
    plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'  # 斜体、加粗

from .DFNP2D import DFNP2D
from .LPP2D import LPP2D
from .JTFP2D import JTFP2D
from .LPJTFP2D import LPJTFP2D
from .ELPJTFP2D import ELPJTFP2D
from .tools import LumpedParameters, EnhancedLumpedParameters, ConservativeLumpedParameters

try:
    import mph
    from .COMSOLP2D import COMSOLP2D
except ImportError:
    ...
    # print("mph库未安装")
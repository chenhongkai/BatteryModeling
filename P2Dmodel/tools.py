#%%
import os
from math import log10
from typing import Sequence
from collections.abc import Iterable
from functools import partial
from collections import namedtuple

from numpy import (array, asarray, ndarray, zeros, unique, searchsorted,
                   clip, isscalar, linspace, empty_like)
from numba import njit
import matplotlib
import matplotlib.pyplot as plt


F = 96485.33289  # 法拉第Faraday常数 [C/mol]
R = 8.314472     # 理想气体常数 [J/(mol·K)]
M = 6.941e-3     # 锂摩尔质量 [kg/mol]
ρ = 534.         # 锂金属密度 [kg/m^3]


class LumpedParameters:
    """23(+6+2)集总参数取值"""
    __slots__ = (
        'Qnom',
        'bounds__',)

    def __init__(self,
                 Qnom: float | int = 1,  # 电池标称容量 [Ah]
                 thermalModel: bool = False,   # 是否包含6个活化能E、热容Cth、热阻Rth
                 ):
        self.Qnom = Qnom; assert Qnom>0, f'电池标称容量{Qnom = }，应大于0 [Ah]，'
        Qnom_in_C = Qnom*3600  # 电池标称容量 [C]
        self.bounds__ = {
            'SOC0': (0.01, 0.8),
            'Qcell': array([0.7, 1.1])*Qnom,
            'ξQneg': (1.05, 1.35),
            'ξQpos': (1.3, 2.0),
            'qeneg': array([0.01, 0.76])*Qnom_in_C,
            'qesep': array([0.004, 0.23])*Qnom_in_C,
            'qepos': array([0.01, 0.76])*Qnom_in_C,
            'κneg': array([0.00075, 0.08])*Qnom_in_C,
            'κsep': array([0.004, 0.08])*Qnom_in_C,
            'κpos': array([0.00075, 0.08])*Qnom_in_C,
            'σneg': array([0.01, 170])*Qnom_in_C,
            'σpos': array([0.01, 170])*Qnom_in_C,
            'Dsneg': (5e-5, 0.63),
            'Dspos': (5e-5, 0.63),
            'De': (2.56e-3, 2.22e-1),
            'κD': array([0.5, 9.])*2*R/F,
            'kneg': array([1e-6, 5e-2])*Qnom_in_C,
            'kpos': array([1e-6, 5e-2])*Qnom_in_C,
            'Rfneg': array([0.09, 330])/Qnom_in_C,
            'Rfpos': array([0.09, 330])/Qnom_in_C,
            'CDLneg': array([1e-6, 1e-2])*Qnom_in_C,
            'CDLpos': array([1e-6, 1e-2])*Qnom_in_C,
            'l': array([1e-13, 1e-11])*Qnom_in_C, }

        if thermalModel:
            cBounds_ = array([800, 1300])  # 比热容 [J/(kg·K)]
            qBounds_ = array([60, 130])    # 比容量 [Ah/kg]

            self.bounds__.update({
                # Global sensitivity analysis towards non-invasive parameterization of the electrochemical-thermal model for lithium-ion batteries
                # https://doi.org/10.1016/j.adapen.2025.100221
                # Fast and reliable calibration of thermal-physical model of lithium-ion battery: a sensitivity-based method
                # https://doi.org/10.1016/j.est.2022.106435
                'Ekneg': array([5, 80])*1e3,
                'Ekpos': array([5, 80])*1e3,
                'EDsneg': array([5, 85])*1e3,
                'EDspos': array([5, 85])*1e3,
                'EDe': array([5, 50])*1e3,
                'Eκ': array([4, 50])*1e3,
                'Cth': array([cBounds_.min()/qBounds_.max(), cBounds_.max()/qBounds_.min()])*Qnom,  # [J/K]
                'Rth': array([0.001, 0.025])*Qnom,  # [K/W]
            })

        for name, bound_ in self.bounds__.items():
            self.bounds__[name] = (float(bound_[0]), float(bound_[1]))

    def print(self):
        nominalSet_ = self.nominalSet_
        for n, name in enumerate(nominalSet_, start=1):
            print(f'{n: <5}{name: 10}'
                  f'下限{self.bounds__[name][0]: <15g}'
                  f'上限{self.bounds__[name][1]: <15g}'
                  f'标称值{nominalSet_[name]: 3g}')

    @property
    def Qnom_in_C(self) -> float:
        """电池标称容量 [C]"""
        return self.Qnom * 3600

    @property
    def names_(self) -> ndarray:
        # 参数名序列
        return array(list(self.bounds__.keys()))

    @property
    def nominalSet_(self) -> dict:
        # 标称参数集
        denormalize = self.denormalize
        return {name: denormalize(name, 0.5) for name in self.bounds__}

    def normalize(self,
                  name: str,
                  value: float | int,) -> float:
        # 参数值归一化
        assert name in self.bounds__, f'参数{name}不包含于{self.bounds__.keys()}'
        value = float(value)
        lb, ub = self.bounds__[name]
        # assert lb<=value<=ub, f'参数{name}的取值范围为[{lb}, {ub}]，当前输入取值{value}'
        if abs(ub/lb)>(100 + 1e-6):
            normvalue = (log10(value) - log10(lb))/(log10(ub) - log10(lb))
        else:
            normvalue = (value - lb)/(ub - lb)
        return normvalue

    def denormalize(self,
                    name: str,
                    normvalue: float | int) -> float:
        # 归一化参数值去归一化
        assert name in self.bounds__, f'参数{name}不包含于{self.bounds__.keys()}'
        # normvalue = float(normvalue)
        lb, ub = self.bounds__[name]
        if abs(ub/lb)>(100 + 1e-6):
            value = 10**(log10(lb) + normvalue*(log10(ub) - log10(lb)))
        else:
            value = lb + normvalue*(ub - lb)
        return value

    def Normalize(self, parameterSet_: dict):
        # 批量归一化
        return {name: self.normalize(name, value) for name, value in parameterSet_.items()}

    def Denormalize(self, parameterSet_: dict):
        # 批量去归一化
        return {name: self.denormalize(name, normvalue) for name, normvalue in parameterSet_.items()}

    @staticmethod
    def sign(name):
        match name:
            case 'Qcell' | 'Qneg' | 'Qpos':
                sign = rf'${{\overline {{\it Q}} }}_{{\mathrm {{ {name[1:]} }} }}$'
            case 'ξQneg' | 'ξQpos':
                sign = rf'${{\overline {{\it ξ}} }}_{{ {{\it Q}} \mathrm {{,{name[2:]} }}  }}$'
            case 'θminneg' | 'θmaxneg' | 'θminpos' | 'θmaxpos':
                sign = rf'${{\it θ}}_{{\mathrm{{ {name[1:4]},{name[-3:]} }} }}$'
            case 'σneg' | 'σpos' | 'κneg' | 'κsep' | 'κpos':
                sign = rf'${{\overline {{\it {name[0]} }} }}_{{\mathrm{{ {name[-3:]} }} }}$'
            case 'Dsneg' | 'Dspos' | 'qeneg' | 'qesep' | 'qepos':
                sign = rf'${{\overline {{\it {name[0]} }} }}_{{\mathrm{{ {name[1]},{name[-3:]} }} }}$'
            case 'Kqeneg' | 'Kqepos':
                sign = rf'${{\overline {{\it K}} }}_{{\mathrm{{q_{{e,{name[-3:]} }} }} }}$'
            case 'Kκneg' | 'Kκpos':
                sign = rf'${{\overline {{\it K}} }}_{{\mathrm{{κ,{name[-3:]} }} }}$'
            case 'kneg' | 'kpos':
                sign = rf'${{\overline {{\it k}} }}_{{\mathrm{{{name[-3:]} }} }}$'
            case 'Rfneg' | 'Rfpos':
                sign = rf'${{\overline {{\it R}} }}_{{\mathrm{{f,{name[-3:]} }} }}$'
            case 'CDLneg' | 'CDLpos':
                sign = rf'${{\overline {{\it C}} }}_{{\mathrm{{DL,{name[-3:]} }} }}$'
            case 'κD':
                sign = r'${\overline {\it κ} }_{\mathrm{D}}$'
            case 'De':
                sign = r'${\overline {\it D} }_{\mathrm{e}}$'
            case 'κ' | 'T' | 'l':
                sign = rf'${{\it {name}}}$'
            case 'SOC0':
                sign = r'${\it SOC}_{\mathrm{0}}$'
            case 'I0intneg' | 'I0intpos':
                sign = rf'${{\it I}}_{{\mathrm{{0,int,{name[-3:]} }} }}$'
            case 'T0':
                sign = r'${\it T}_{\mathrm{0}}$'
            case 'Ekneg' | 'Ekpos' | 'EDsneg' | 'EDspos' | 'EDe' | 'Eκ':
                parametersign = LumpedParameters.sign(name[1:])
                sign = rf'${{\it E}}_{{{parametersign.replace('$', '')}}}$'
            case 'Cth':
                sign = r'${\overline {\it C} }_{\mathrm{th}}$'
            case 'Rth':
                sign = r'${\overline {\it R} }_{\mathrm{th}}$'
            case _:
                raise ValueError(f'未定义参数{name}')
        return sign

    @staticmethod
    def unit(name):
        match name:
            case 'Qcell' | 'Qneg' | 'Qpos':
                unit = '$Ah$'
            case 'ξQneg' | 'ξQpos' |\
                 'θminneg' | 'θmaxneg' | 'θminpos' | 'θmaxpos' | 'SOC0' |\
                 'Kqeneg' | 'Kqepos' | 'Kκneg' | 'Kκpos':
                unit = ''
            case 'σneg' | 'σpos' | 'κneg' | 'κsep' | 'κpos':
                unit = '$S$'
            case 'Dsneg' | 'Dspos':
                unit = '$s^{-1}$'
            case 'qeneg' | 'qesep' | 'qepos':
                unit = '$C$'
            case 'kneg' | 'kpos' | 'I0intneg' | 'I0intpos':
                unit = '$A$'
            case 'Rfneg' | 'Rfpos':
                unit = '$Ω$'
            case 'CDLneg' | 'CDLpos':
                unit = '$F$'
            case 'κD':
                unit = '$V/K$'
            case 'De':
                unit = '$V$'
            case 'l':
                unit = '$H$'
            case 'T':
                unit = '$K$'
            case 'Cth':
                unit = '$J/K$'
            case 'Rth':
                unit = '$K/W$'
            case 'Ekneg' | 'Ekpos' | 'Eκ' | 'EDsneg' | 'EDspos' | 'EDe':
                unit = '$J/mol$'
            case _:
                raise ValueError(f'未定义参数{name}')
        return unit

    @staticmethod
    def value(name, value):
        match name:
            case 'Qcell' | 'Qneg' | 'Qpos' | 'T'  | 'ξQneg' | 'ξQpos':
                string = f'{value:.2f}'
            case 'θminneg' | 'θmaxneg' | 'θminpos' | 'θmaxpos' | 'SOC0' |\
                 'Kqeneg' | 'Kqepos' | 'Kκneg' | 'Kκpos':
                string = f'{value:.3f}'
            case 'σneg' | 'σpos' | 'κneg' | 'κsep' | 'κpos' |\
                 'Dsneg' |'Dspos' |\
                 'qeneg' | 'qesep' | 'qepos' |\
                 'kneg' | 'kpos'| 'I0intneg' | 'I0intpos' |\
                 'Rfneg' | 'Rfpos' |\
                 'CDLneg' | 'CDLpos' |\
                 'κD' | 'De' | 'l':
                base, expo = f'{value:.2e}'.split('e')
                string = rf'${float(base):.2f}\;×\;10^{{ {int(expo)} }}$'
            case _:
                raise ValueError(f'未定义参数{name}')
        return string

    @classmethod
    def value_unit(cls, name, value):
        return cls.value(name, value) + r'$\;$' + cls.unit(name)

class EnhancedLumpedParameters(LumpedParameters):
    """增强集总参数取值"""
    __slots__ = ()

    def __init__(self, **kwargs):
        LumpedParameters.__init__(self, **kwargs)
        del self.bounds__['κneg'], self.bounds__['κpos'],\
            self.bounds__['qeneg'], self.bounds__['qepos']
        self.bounds__.update({
            'Kκneg': (0.08, 1),
            'Kκpos': (0.08, 1),
            'Kqeneg': (0.5, 4),
            'Kqepos': (0.5, 4),})

class ConservativeLumpedParameters(LumpedParameters):
    """保守集总参数取值（25参数，含4边界嵌锂状态，不含正负极容量比ξQneg、ξQpos）"""
    __slots__ = ()

    def __init__(self, **kwargs):
        LumpedParameters.__init__(self, **kwargs)
        del self.bounds__['ξQneg'], self.bounds__['ξQpos']
        self.bounds__.update({
            'θminneg': (0.001, 0.44),   # SOC=0%的负极嵌锂状态取值范围
            'θmaxneg': (0.60, 0.99),    # SOC=100%的负极嵌锂状态取值范围
            'θminpos': (0.001, 0.44),   # SOC=100%的正极嵌锂状态取值范围
            'θmaxpos': (0.60, 0.99),})  # SOC=0%的正极嵌锂状态取值范围


def set_matplotlib(fontsize: int | float = 12):
    """设置matplotlib"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    matplotlib.use('Qt5Agg')  # TkAgg/Qt5Agg
    # plt.close(plt.figure())
    fontname = 'Times New Roman'                  # 字体
    plt.rcParams['font.serif'] = [fontname]       # 衬线字体
    plt.rcParams['font.sans-serif'] = [fontname]  # 无衬线字体
    plt.rcParams['font.size'] = fontsize          # 字号
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['axes.unicode_minus'] = False               # 正常显示负号
    plt.rcParams['mathtext.default'] = 'regular'             # 默认样式：正体、不加粗
    plt.rcParams['mathtext.rm'] = 'STIXGeneral:regular'      # 正体、不加粗
    plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'       # 斜体、不加粗
    plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'  # 斜体、加粗


def transform37to23(
        A, Lneg, Lsep, Lpos, εsneg, εspos,
        εeneg, εesep, εepos, Rsneg, Rspos,
        bneg, bsep, bpos,
        Dsneg, Dspos, De,
        σneg, σpos, κ,
        tplus, TDF,
        kneg, kpos, Rfneg, Rfpos,
        csmaxneg, csmaxpos, ce0,
        CDLneg, CDLpos, l,
        θminneg, θmaxneg, θminpos, θmaxpos, SOC0,
        ):
    Qneg = A*Lneg*εsneg*csmaxneg*F/3600
    Qpos = A*Lpos*εspos*csmaxpos*F/3600
    Qcellneg = Qneg*(θmaxneg - θminneg)
    Qcellpos = Qpos*(θmaxpos - θminpos)
    Qcell = min(Qcellneg, Qcellpos)
    print(f'正极可用容量{Qcellpos:.4f}Ah，负极可用容量{Qcellneg:.4f}Ah，全电池理论可用容量使用正负极可用容量的较小值')
    aneg = 3*εsneg/Rsneg
    apos = 3*εspos/Rspos
    return {
        'Qcell' : Qcell,       # 全电池理论可用容量 [Ah]
        'ξQneg': Qneg/Qcell,  # 负极容量与全电池理论可用容量之比 [–]
        'ξQpos': Qpos/Qcell,  # 正极容量与全电池理论可用容量之比 [–]
        'σneg' : A*σneg*εsneg**bneg/Lneg,  # 负极集总固相电导率 [S]
        'σpos' : A*σpos*εspos**bpos/Lpos,  # 正极集总固相电导率 [S]
        'κneg' : A*κ*εeneg**bneg/Lneg,  # 负极集总液相离子电导率 [S]
        'κsep' : A*κ*εesep**bsep/Lsep,  # 隔膜集总液相离子电导率 [S]
        'κpos' : A*κ*εepos**bpos/Lpos,  # 正极集总液相离子电导率 [S]
        'Dsneg' : Dsneg/Rsneg**2,  # 负极集总固相锂离子扩散系数 [1/s]
        'Dspos' : Dspos/Rspos**2,  # 正极集总固相锂离子扩散系数 [1/s]
        'qeneg' : F*εeneg*ce0*A*Lneg/(1 - tplus),  # 负极液相锂离子电荷量 [C]
        'qesep' : F*εesep*ce0*A*Lsep/(1 - tplus),  # 隔膜液相锂离子电荷量 [C]
        'qepos' : F*εepos*ce0*A*Lpos/(1 - tplus),  # 正极液相锂离子电荷量 [C]
        'kneg' : F*aneg*A*Lneg*kneg*ce0**0.5*csmaxneg,  # 负极集总反应速率常数 [A]
        'kpos' : F*apos*A*Lpos*kpos*ce0**0.5*csmaxpos,  # 正极集总反应速率常数 [A]
        'Rfneg' : Rfneg/(aneg*A*Lneg),  # 负极集总SEI膜内阻 [Ω]
        'Rfpos' : Rfpos/(apos*A*Lpos),  # 正极集总SEI膜内阻 [Ω]
        'κD' : 2*R/F*(1 - tplus)*TDF,   # 液相离子电导系数 [V/K]
        'De' : F*De*ce0/κ/(1 - tplus),    # 集总离子扩散率/电导率之比 [V]
        'CDLneg' : aneg*A*Lneg*CDLneg,  # 负极集总双电层电容 [F]
        'CDLpos' : apos*A*Lpos*CDLpos,  # 正极集总双电层电容 [F]
        'l' : l,  # 等效电感 [H]
        'SOC0': SOC0, }

@njit(cache=True, fastmath=True)
def JITinterp1d(xbase_: ndarray,
                ybase_: ndarray,
                x_: ndarray):
    # JIT编译一维线性插值
    x_ = asarray(x_)
    # 找区间
    idx_ = searchsorted(xbase_, x_)  # idx_.shape同x_.shape
    # 处理边界
    idx_ = clip(idx_, 1, xbase_.size - 1)
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

def Interpolate1D(xbase_: ndarray, ybase_: ndarray):
    # 快速一维线性插值
    assert len(xbase_)==len(ybase_), '自变量序列xbase_的长度应等于因变量序列ybase_的长度'
    xbase_ = asarray(xbase_, dtype=float)
    ybase_ = asarray(ybase_, dtype=float)
    assert xbase_.ndim==1, '自变量序列xbase_应为1维'
    assert ybase_.ndim==1, '因变量序列ybase_应为1维'
    assert unique(xbase_).size==xbase_.size, '自变量序列xbase_不应包含相同值'

    idx_ = xbase_.argsort()
    xbase_ = xbase_[idx_]
    ybase_ = ybase_[idx_]
    return partial(JITinterp1d, xbase_, ybase_)

@njit(cache=True, fastmath=True)
def triband_to_dense(band__: ndarray) -> ndarray:
    # 三对角矩阵的带band__ (3, N)  -> 稠密方阵K__ (N, N)
    N = band__.shape[1]
    N1 = N + 1
    K__ = zeros((N, N))
    ravelK_ = K__.ravel()
    ravelK_[1::N1] = band__[0, 1:]   # 上对角线
    ravelK_[::N1]  = band__[1]       # 主对角线
    ravelK_[N::N1] = band__[2, :-1]  # 下对角线
    return K__

@njit(cache=True, fastmath=True)
def tridiagonal_matmul(
        band__: ndarray,  # (3, N) 三对角矩阵的带
        X__: ndarray,     # (N, M) 矩阵
        ) -> ndarray:
    # 三对角矩阵的带band__相应的三对角矩阵与矩阵X__的积
    # 主对角线贡献
    Y__ = band__[1][:, None]*X__
    # 上对角线贡献
    Y__[:-1] += band__[0, 1:, None] * X__[1:]
    # 下对角线贡献
    Y__[1:] += band__[2, :-1, None] * X__[:-1]
    return Y__

def diagonalSliceRavel(
        N: int,           # 矩阵列数
        s_rows: slice,    # 切片索引行
        s_cols: slice,    # 切片索引列
        offset: int = 0,  # 对角线 -1 0 1
        ) -> slice:
    """变换切片索引：A__[s_row, s_col].diagonal(offset) -> A__.ravel()[sr_diagonal]"""
    startR = s_rows.start
    startC = s_cols.start
    Nrows = s_rows.stop - startR
    Ncols = s_cols.stop - startC
    start = startR * N + startC    # 主对角线起始
    if offset==0:
        length = min(Nrows, Ncols)  # 主对角线长度
    elif offset>0:
        start += offset
        length = min(Nrows, Ncols - offset)
    else:
        start -= offset*N
        length = min(Nrows + offset, Ncols)
    step = N + 1
    stop = start + length*step
    return slice(start, stop, step)

SlicesK = namedtuple('SlicesK', (
    's_csnegsurf',
    's_cspossurf',
    's_ce',
    's_φsneg',
    's_φspos',
    's_φe',

    's_ceneg',
    's_cepos',
    's_φeneg',
    's_φepos',
    's_c',
    's_φ',

    'sr_csnegsurf_csnegsurf',
    'sr_csnegsurf_ceneg',
    'sr_csnegsurf_φsneg',
    'sr_csnegsurf_φeneg',
    'sr_cspossurf_cspossurf',
    'sr_cspossurf_cepos',
    'sr_cspossurf_φspos',
    'sr_cspossurf_φepos',

    'sr_ceneg_csnegsurf',
    'sr_cepos_cspossurf',
    'sr_ce_ce',
    'sr_ce_ce_l',
    'sr_ce_ce_u',
    'sr_ceneg_φsneg',
    'sr_cepos_φspos',
    'sr_ceneg_φeneg',
    'sr_cepos_φepos',

    'sr_φsneg_csnegsurf',
    'sr_φspos_cspossurf',
    'sr_φsneg_φsneg',
    'sr_φspos_φspos',
    'sr_φsneg_φeneg',
    'sr_φspos_φepos',

    'sr_φeneg_csnegsurf',
    'sr_φepos_cspossurf',
    'sr_φe_ce',
    'sr_φe_ce_l' ,
    'sr_φe_ce_u',
    'sr_φeneg_φsneg',
    'sr_φepos_φspos',
    'sr_φe_φe',
    'sr_φe_φe_l',
    'sr_φe_φe_u',
    'sr_φeneg_φeneg',
    'sr_φepos_φepos',)
    )

SlicesKf = namedtuple('SlicesKf', (
    's_REcsnegsurf',
    's_IMcsnegsurf',
    's_REcspossurf',
    's_IMcspossurf',
    's_REce',
    's_IMce',
    's_REφsneg',
    's_IMφsneg',
    's_REφspos',
    's_IMφspos',
    's_REφe',
    's_IMφe',

    's_REceneg',
    's_IMceneg',
    's_REcepos',
    's_IMcepos',
    's_REφeneg',
    's_IMφeneg',
    's_REφepos',
    's_IMφepos',

    'sr_REcsnegsurf_REcsnegsurf',
    'sr_REcsnegsurf_IMcsnegsurf',
    'sr_IMcsnegsurf_REcsnegsurf',
    'sr_IMcsnegsurf_IMcsnegsurf',
    'sr_REcsnegsurf_REceneg',
    'sr_IMcsnegsurf_IMceneg',
    'sr_REcsnegsurf_REφsneg',
    'sr_REcsnegsurf_IMφsneg',
    'sr_IMcsnegsurf_REφsneg',
    'sr_IMcsnegsurf_IMφsneg',
    'sr_REcsnegsurf_REφeneg',
    'sr_REcsnegsurf_IMφeneg',
    'sr_IMcsnegsurf_REφeneg',
    'sr_IMcsnegsurf_IMφeneg',

    'sr_REcspossurf_REcspossurf',
    'sr_REcspossurf_IMcspossurf',
    'sr_IMcspossurf_REcspossurf',
    'sr_IMcspossurf_IMcspossurf',
    'sr_REcspossurf_REcepos',
    'sr_IMcspossurf_IMcepos',
    'sr_REcspossurf_REφspos',
    'sr_REcspossurf_IMφspos',
    'sr_IMcspossurf_REφspos',
    'sr_IMcspossurf_IMφspos',
    'sr_REcspossurf_REφepos',
    'sr_REcspossurf_IMφepos',
    'sr_IMcspossurf_REφepos',
    'sr_IMcspossurf_IMφepos',

    'sr_REceneg_REcsnegsurf',
    'sr_REceneg_IMcsnegsurf',
    'sr_IMceneg_REcsnegsurf',
    'sr_IMceneg_IMcsnegsurf',
    'sr_REcepos_REcspossurf',
    'sr_REcepos_IMcspossurf',
    'sr_IMcepos_REcspossurf',
    'sr_IMcepos_IMcspossurf',
    'sr_REce_REce',
    'sr_REce_REce_l',
    'sr_REce_REce_u',
    'sr_REce_IMce',
    'sr_IMce_REce',
    'sr_IMce_IMce',
    'sr_IMce_IMce_l',
    'sr_IMce_IMce_u',
    'sr_REceneg_REφsneg',
    'sr_REceneg_IMφsneg',
    'sr_IMceneg_REφsneg',
    'sr_IMceneg_IMφsneg',
    'sr_REcepos_REφspos',
    'sr_REcepos_IMφspos',
    'sr_IMcepos_REφspos',
    'sr_IMcepos_IMφspos',
    'sr_REceneg_REφeneg',
    'sr_REceneg_IMφeneg',
    'sr_IMceneg_REφeneg',
    'sr_IMceneg_IMφeneg',
    'sr_REcepos_REφepos',
    'sr_REcepos_IMφepos',
    'sr_IMcepos_REφepos',
    'sr_IMcepos_IMφepos',

    'sr_REφsneg_REcsnegsurf',
    'sr_REφsneg_IMcsnegsurf',
    'sr_IMφsneg_REcsnegsurf',
    'sr_IMφsneg_IMcsnegsurf',
    'sr_REφspos_REcspossurf',
    'sr_REφspos_IMcspossurf',
    'sr_IMφspos_REcspossurf',
    'sr_IMφspos_IMcspossurf',
    'sr_REφsneg_REceneg',
    'sr_REφsneg_IMceneg',
    'sr_IMφsneg_REceneg',
    'sr_IMφsneg_IMceneg',
    'sr_REφsneg_REφsneg',
    'sr_REφsneg_REφsneg_l',
    'sr_REφsneg_REφsneg_u',
    'sr_REφsneg_IMφsneg',
    'sr_IMφsneg_REφsneg',
    'sr_IMφsneg_IMφsneg',
    'sr_IMφsneg_IMφsneg_l',
    'sr_IMφsneg_IMφsneg_u',
    'sr_REφspos_REφspos',
    'sr_REφspos_REφspos_l',
    'sr_REφspos_REφspos_u',
    'sr_REφspos_IMφspos',
    'sr_IMφspos_REφspos',
    'sr_IMφspos_IMφspos',
    'sr_IMφspos_IMφspos_l',
    'sr_IMφspos_IMφspos_u',
    'sr_REφsneg_REφeneg',
    'sr_REφsneg_IMφeneg',
    'sr_IMφsneg_REφeneg',
    'sr_IMφsneg_IMφeneg',
    'sr_REφspos_REφepos',
    'sr_REφspos_IMφepos',
    'sr_IMφspos_REφepos',
    'sr_IMφspos_IMφepos',

    'sr_REφeneg_REcsnegsurf',
    'sr_REφeneg_IMcsnegsurf',
    'sr_IMφeneg_REcsnegsurf',
    'sr_IMφeneg_IMcsnegsurf',
    'sr_REφepos_REcspossurf',
    'sr_REφepos_IMcspossurf',
    'sr_IMφepos_REcspossurf',
    'sr_IMφepos_IMcspossurf',
    'sr_REφe_REce',
    'sr_REφe_REce_l',
    'sr_REφe_REce_u',
    'sr_IMφe_IMce',
    'sr_IMφe_IMce_l',
    'sr_IMφe_IMce_u',
    'sr_REφeneg_REφsneg',
    'sr_REφeneg_IMφsneg',
    'sr_IMφeneg_REφsneg',
    'sr_IMφeneg_IMφsneg',
    'sr_REφepos_REφspos',
    'sr_REφepos_IMφspos',
    'sr_IMφepos_REφspos',
    'sr_IMφepos_IMφspos',
    'sr_REφe_REφe',
    'sr_REφe_REφe_l',
    'sr_REφe_REφe_u',
    'sr_REφe_IMφe',
    'sr_IMφe_REφe',
    'sr_IMφe_IMφe',
    'sr_IMφe_IMφe_l',
    'sr_IMφe_IMφe_u',)
    )


def get_color(s_: Sequence | int, n: int, cmap='viridis'):
    """返回指定颜色映射中的颜色，默认viridis"""
    if isinstance(s_, Iterable):
        N = len(s_)
    elif isscalar(s_):
        N = int(s_)
    color_ = plt.get_cmap(cmap)(int(linspace(0, 255, N)[n]))[:3]  # (3,)
    return color_


def stepping_aware_cached_property(method):
    # 装饰器：将@property方法变成缓存感知的@property
    # 若当前时刻self.t与self._cached_t[name]不一致，重算；
    # 否则，直接返回 self._cached_properties[name]
    name = method.__name__  # @property方法名

    @property
    def prop(self):
        t = self.t  # 当前时刻
        _cached_t = self._cached_t  # 缓存时刻字典
        try:
            if _cached_t[name]==t:
                # 时刻匹配 → 直接返回缓存值
                return self._cached_properties[name]
        except KeyError:
            pass
        # 时刻不匹配 → 需重算
        self._cached_properties[name] = value = method(self)
        _cached_t[name] = t
        return value
    return prop

@njit(cache=True, fastmath=True)
def batch_inv_2x2(A__: ndarray) -> ndarray:
    # 批量 (2,2) 矩阵求逆
    # A__.shape: (N, 2, 2), 返回 (N, 2, 2) 的逆矩阵
    # 提取各元素：a,b,c,d 形状均为 (N,)
    a_ = A__[:, 0, 0]
    b_ = A__[:, 0, 1]
    c_ = A__[:, 1, 0]
    d_ = A__[:, 1, 1]
    det_ = a_ * d_ - b_ * c_
    # if any(abs(det_) < 1e-12):
    #     raise ValueError("存在行列式接近零的不可逆矩阵")
    invA__ = empty_like(A__)
    invA__[:, 0, 0] = d_
    invA__[:, 0, 1] = -b_
    invA__[:, 1, 0] = -c_
    invA__[:, 1, 1] = a_
    invA__ /= det_[:, None, None]  # 广播除法
    return invA__

if __name__ == '__main__':
    lp = LumpedParameters(thermalModel=True)

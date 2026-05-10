#%%
from typing import Sequence, Callable
from abc import ABC, abstractmethod
from collections import namedtuple

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from numpy import nan, ndarray, zeros, eye,\
    asarray, full, stack, hstack, \
    empty, meshgrid, logspace,\
    ptp, log10,\
    isscalar

from P2Dmodel.P2Dbase import P2Dbase, DiagonalSliceRavel

get_color = P2Dbase.get_color

JTFbase__slots__ = (
    # 恒定量
    'f_', 'ΔIAC',
    'frequency_dependent_cache',
    'banded_experience_of_Kf__',
    'EISdatanames_',
    # 状态量
    'tEIS', 'Z_', 'Zneg_', 'Zpos_',
    'REφsneg__',   'IMφsneg__',   'REφspos__',   'IMφspos__',   'REφe__', 'IMφe__',
    'REηintneg__', 'IMηintneg__', 'REηintpos__', 'IMηintpos__',
    'REηLP__',  'IMηLP__',
    'ravelKf_', 'bKf_',
    # 索引频域因变量
    'sKf',
    )

class JTFbase(ABC):
    """抽象类：锂离子电池时频联合准二维模型 Joint Time-Frequency Pseudo-two-Dimension model"""

    # __slots__ = JTFbase__slots__

    ## 类型注解 ##
    Nneg: int; Nsep: int; Npos: int; Ne: int  # 负极、隔膜、正极全区域网格数
    lithiumPlating: bool
    complete: bool
    verbose: bool

    _i0intneg: float | None  # 负极主反应交换电流密度
    _i0intpos: float | None  # 正极主反应交换电流密度
    l: float                 # 等效电感 [H]

    data: dict[str, list[float | ndarray]]  # 记录的数据
    Δxneg: float; Δxpos: float      # 负极、正极网格厚度
    x_: ndarray                     # (Ne,) 电极厚度方向控制体中心坐标
    xneg_: ndarray; xpos_: ndarray  # (Nneg,) (Npos,) 负极、正极控制体中心坐标
    Δx_: ndarray      # (Ne,) 电极厚度方向控制体厚度序列
    ΔxWest_: ndarray  # (Ne,) 当前控制体中心到左侧控制体中心的距离
    ΔxEast_: ndarray  # (Ne,) 当前控制体中心到右侧控制体中心的距离

    tSign: str; tUnit: str     # 时间t符号、单位
    xSign: str; xUnit: str     # 电极厚度方向坐标x符号、单位
    rSign: str; rUnit: str     # 径向坐标r符号、单位
    cSign: str; cUnit: str     # 锂离子浓度c符号、单位
    jSign: str; jUnit: str     # 局部体积电流密度j符号、单位
    i0Sign: str; i0Unit: str   # 交换电流密度i0符号、单位
    xPlot_: ndarray            # (Ne,) 电极厚度方向控制体中心坐标，用于作图
    xInterfacesPlot_: ndarray  # (Ne+1,) 电极厚度方向控制体界面坐标，用于作图
    plot_interfaces: Callable  # 函数：画区域界面

    def __init__(self,
            f_: Sequence[float] = logspace(3, -1, 26),  # 频率序列 [Hz]
            ):
        self.f_ = f_ = asarray(f_); assert f_.ndim==1, f'频率序列f_应可转化为ndim==1的ndarray，当前{f_ = }'
        Nf = f_.size
        # 状态量
        self.tEIS: float = None                      # 计算阻抗的时刻 [s]
        self.Z_: ndarray = empty(Nf, dtype=complex)  # 全电池阻抗谱 [Ω]
        if self.complete:
            Nneg, Npos, Ne = self.Nneg, self.Npos, self.Ne  # 读取：网格数
            self.Zneg_, self.Zpos_ = empty(Nf, dtype=complex), empty(Nf, dtype=complex)  # 负极、正极阻抗谱 [Ω]
            self.REφsneg__, self.IMφsneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))        # 负极固相电势实部、虚部
            self.REφspos__, self.IMφspos__ = empty((Nf, Npos)), empty((Nf, Npos))        # 正极固相电势实部、虚部
            self.REφe__,    self.IMφe__    = empty((Nf, Ne)), empty((Nf, Ne))            # 电解液电势实部、虚部
            self.REηintneg__, self.IMηintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极主反应过电位实部、虚部
            self.REηintpos__, self.IMηintpos__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极主反应过电位实部、虚部
            if self.lithiumPlating:
                self.REηLP__, self.IMηLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极析锂反应过电位实部、虚部
        # 恒定量
        self.ΔIAC = 1.  # 交流扰动电流振幅 [A]
        self.frequency_dependent_cache = self.solve_frequency_dependent_variables()  # 频率相关变量缓存
        self.banded_experience_of_Kf__: dict[str, ndarray | int] | None = None  # Kf__矩阵带状化经验
        self.EISdatanames_ = ['tEIS', 'Z_']                          # 需记录的阻抗数据名称
        if self.complete:
            extra_EISdatanames_ =[              # 需记录的阻抗数据名称
                'Zneg_', 'Zpos_',               # 负极、正极复阻抗 [Ω]
                'REφsneg__', 'IMφsneg__',       # 负极固相电势实部、虚部 [V]
                'REφspos__', 'IMφspos__',       # 正极固相电势实部、虚部 [V]
                'REφe__', 'IMφe__',             # 电解液电势实部、虚部 [V]
                'REηintneg__', 'IMηintneg__',   # 负极主反应过电位实部、虚部 [V]
                'REηintpos__', 'IMηintpos__',]  # 正极主反应过电位实部、虚部 [V]
            self.EISdatanames_.extend(extra_EISdatanames_)
        self.data.update({name: [] for name in self.EISdatanames_})  # 字典：存储呈时间序列的阻抗数据
        # 频域因变量线性矩阵Kf__
        self.bKf_ = None      # (NKf,) 常数项向量 Kf__ @ X_ = bKf_
        self.ravelKf_ = None  # (NK*NK,) Kf__展平视图
        self.sKf = None       # 切片索引集

    def _generate_Kf__bKf_and_slices(self):
        """生成频域因变量矩阵Kf__、常数项向量bKf_及切片索引，并对Kf__赋常系数、几何网格相关参数"""
        # 读取网格数
        Nneg, Npos, Ne = self.Nneg, self.Npos, self.Ne
        # 读取模式
        lithiumPlating = self.lithiumPlating

        # 生成频域因变量矩阵Kf__及其频域因变量切片索引
        NKf = 0  # 全局索引游标
        def allocate(n: int) -> slice:
            # 分配切片
            nonlocal NKf
            s = slice(NKf, NKf + n)
            NKf += n
            return s
        Ni0intneg =  Nneg if self._i0intneg is None else 0
        Ni0intpos =  Npos if self._i0intpos is None else 0
        slices_ = {
            's_REcsnegsurf': (s_REcsnegsurf := allocate(Nneg)),  # 索引：负极固相表面浓度实部
            's_IMcsnegsurf': (s_IMcsnegsurf := allocate(Nneg)),  # 索引：负极固相表面浓度虚部
            's_REcspossurf': (s_REcspossurf := allocate(Npos)),  # 索引：正极固相表面浓度实部
            's_IMcspossurf': (s_IMcspossurf := allocate(Npos)),  # 索引：正极固相表面浓度虚部
            's_REce': (s_REce := allocate(Ne)),          # 索引：电解液锂离子浓度实部
            's_IMce': (s_IMce := allocate(Ne)),          # 索引：电解液锂离子浓度虚部
            's_REφsneg': (s_REφsneg := allocate(Nneg)),  # 索引：负极固相电势实部
            's_IMφsneg': (s_IMφsneg := allocate(Nneg)),  # 索引：负极固相电势虚部
            's_REφspos': (s_REφspos := allocate(Npos)),  # 索引：正极固相电势实部
            's_IMφspos': (s_IMφspos := allocate(Npos)),  # 索引：正极固相电势虚部
            's_REφe': (s_REφe := allocate(Ne)),  # 索引：电解液电势实部
            's_IMφe': (s_IMφe := allocate(Ne)),  # 索引：电解液电势虚部
            's_REjintneg': (s_REjintneg := allocate(Nneg)),  # 索引：负极主反应局部体积电流密度实部
            's_IMjintneg': (s_IMjintneg := allocate(Nneg)),  # 索引：负极主反应局部体积电流密度虚部
            's_REjintpos': (s_REjintpos := allocate(Npos)),  # 索引：正极主反应局部体积电流密度实部
            's_IMjintpos': (s_IMjintpos := allocate(Npos)),  # 索引：正极主反应局部体积电流密度虚部
            's_REjDLneg': (s_REjDLneg := allocate(Nneg)),    # 索引：负极双电层局部体积电流密度实部
            's_IMjDLneg': (s_IMjDLneg := allocate(Nneg)),    # 索引：负极双电层局部体积电流密度虚部
            's_REjDLpos': (s_REjDLpos := allocate(Npos)),    # 索引：正极双电层局部体积电流密度实部
            's_IMjDLpos': (s_IMjDLpos := allocate(Npos)),    # 索引：正极双电层局部体积电流密度虚部
            's_REi0intneg': (s_REi0intneg := allocate(Ni0intneg)),  # 索引：负极主反应交换电流密度实部
            's_IMi0intneg': (s_IMi0intneg := allocate(Ni0intneg)),  # 索引：负极主反应交换电流密度虚部
            's_REi0intpos': (s_REi0intpos := allocate(Ni0intpos)),  # 索引：正极主反应交换电流密度实部
            's_IMi0intpos': (s_IMi0intpos := allocate(Ni0intpos)),  # 索引：正极主反应交换电流密度虚部
            's_REηintneg': (s_REηintneg := allocate(Nneg)),   # 索引：负极过电位实部
            's_IMηintneg': (s_IMηintneg := allocate(Nneg)),   # 索引：负极过电位虚部
            's_REηintpos': (s_REηintpos := allocate(Npos)),   # 索引：正极过电位实部
            's_IMηintpos': (s_IMηintpos := allocate(Npos)),   # 索引：正极过电位虚部
            }
        if lithiumPlating:
            slices_.update({
                's_REjLP': (s_REjLP := allocate(Nneg)),  # 索引：析锂反应电流密度实部
                's_IMjLP': (s_IMjLP := allocate(Nneg)),  # 索引：正极交换电流密度虚部
                's_REηLP': (s_REηLP := allocate(Nneg)),  # 索引：析锂反应过电位实部
                's_IMηLP': (s_IMηLP := allocate(Nneg)),  # 索引：析锂反应过电位虚部
                })

        slices_.update({
            's_REceneg': (s_REceneg := slice(start := s_REce.start, start + Nneg)),  # 索引：负极电解液浓度实部
            's_IMceneg': (s_IMceneg := slice(start := s_IMce.start, start + Nneg)),  # 索引：负极电解液浓度虚部
            's_REcepos': (s_REcepos := slice((stop := s_REce.stop) - Npos, stop)),   # 索引：正极电解液浓度实部
            's_IMcepos': (s_IMcepos := slice((stop := s_IMce.stop) - Npos, stop)),   # 索引：正极电解液浓度虚部

            's_REφeneg': (s_REφeneg := slice(start := s_REφe.start, start + Nneg)),  # 索引：负极电解液电势实部
            's_IMφeneg': (s_IMφeneg := slice(start := s_IMφe.start, start + Nneg)),  # 索引：负极电解液电势虚部
            's_REφepos': (s_REφepos := slice((stop := s_REφe.stop) - Npos, stop)),   # 索引：正极电解液电势实部
            's_IMφepos': (s_IMφepos := slice((stop := s_IMφe.stop) - Npos, stop)),   # 索引：正极电解液电势虚部
            })
        # Kf__矩阵子块对角元对应Kf__.ravel()的索引
        dsr = DiagonalSliceRavel(NKf)
        slices_.update({
            'sr_REcsnegsurf_REjintneg' : dsr(s_REcsnegsurf, s_REjintneg),
            'sr_REcsnegsurf_IMjintneg' : dsr(s_REcsnegsurf, s_IMjintneg),
            'sr_IMcsnegsurf_REjintneg' : dsr(s_IMcsnegsurf, s_REjintneg),
            'sr_IMcsnegsurf_IMjintneg' : dsr(s_IMcsnegsurf, s_IMjintneg),
            'sr_REcspossurf_REjintpos' : dsr(s_REcspossurf, s_REjintpos),
            'sr_REcspossurf_IMjintpos' : dsr(s_REcspossurf, s_IMjintpos),
            'sr_IMcspossurf_REjintpos' : dsr(s_IMcspossurf, s_REjintpos),
            'sr_IMcspossurf_IMjintpos' : dsr(s_IMcspossurf, s_IMjintpos),
            'sr_REce_REce' : dsr(s_REce, s_REce),
            'sr_REce_REce_l' : dsr(s_REce, s_REce, -1),
            'sr_REce_REce_u' : dsr(s_REce, s_REce,  1),
            'sr_IMce_IMce' : dsr(s_IMce, s_IMce),
            'sr_IMce_IMce_l' : dsr(s_IMce, s_IMce, -1),
            'sr_IMce_IMce_u' : dsr(s_IMce, s_IMce,  1),
            'sr_REce_IMce' : dsr(s_REce, s_IMce),
            'sr_IMce_REce' : dsr(s_IMce, s_REce),
            'sr_REceneg_REjintneg' : dsr(s_REceneg, s_REjintneg),
            'sr_REceneg_REjDLneg'  : dsr(s_REceneg, s_REjDLneg),
            'sr_IMceneg_IMjintneg' : dsr(s_IMceneg, s_IMjintneg),
            'sr_IMceneg_IMjDLneg'  : dsr(s_IMceneg, s_IMjDLneg),
            'sr_REcepos_REjintpos' : dsr(s_REcepos, s_REjintpos),
            'sr_REcepos_REjDLpos'  : dsr(s_REcepos, s_REjDLpos),
            'sr_IMcepos_IMjintpos' : dsr(s_IMcepos, s_IMjintpos),
            'sr_IMcepos_IMjDLpos'  : dsr(s_IMcepos, s_IMjDLpos),
            'sr_REφsneg_REjintneg' : dsr(s_REφsneg, s_REjintneg),
            'sr_REφsneg_REjDLneg'  : dsr(s_REφsneg, s_REjDLneg),
            'sr_IMφsneg_IMjintneg' : dsr(s_IMφsneg, s_IMjintneg),
            'sr_IMφsneg_IMjDLneg'  : dsr(s_IMφsneg, s_IMjDLneg),
            'sr_REφspos_REjintpos' : dsr(s_REφspos, s_REjintpos),
            'sr_REφspos_REjDLpos'  : dsr(s_REφspos, s_REjDLpos),
            'sr_IMφspos_IMjintpos' : dsr(s_IMφspos, s_IMjintpos),
            'sr_IMφspos_IMjDLpos'  : dsr(s_IMφspos, s_IMjDLpos),
            'sr_REφe_REce'   : dsr(s_REφe, s_REce),
            'sr_REφe_REce_l' : dsr(s_REφe, s_REce, -1),
            'sr_REφe_REce_u' : dsr(s_REφe, s_REce,  1),
            'sr_IMφe_IMce'   : dsr(s_IMφe, s_IMce),
            'sr_IMφe_IMce_l' : dsr(s_IMφe, s_IMce, -1),
            'sr_IMφe_IMce_u' : dsr(s_IMφe, s_IMce,  1),
            'sr_REφe_REφe'   : dsr(s_REφe, s_REφe),
            'sr_REφe_REφe_l' : dsr(s_REφe, s_REφe, -1),
            'sr_REφe_REφe_u' : dsr(s_REφe, s_REφe,  1),
            'sr_IMφe_IMφe'   : dsr(s_IMφe, s_IMφe),
            'sr_IMφe_IMφe_l' : dsr(s_IMφe, s_IMφe, -1),
            'sr_IMφe_IMφe_u' : dsr(s_IMφe, s_IMφe,  1),
            'sr_REjintneg_REi0intneg' : dsr(s_REjintneg, s_REi0intneg),
            'sr_IMjintneg_IMi0intneg' : dsr(s_IMjintneg, s_IMi0intneg),
            'sr_REjintneg_REηintneg'  : dsr(s_REjintneg, s_REηintneg),
            'sr_IMjintneg_IMηintneg'  : dsr(s_IMjintneg, s_IMηintneg),
            'sr_REjintpos_REi0intpos' : dsr(s_REjintpos, s_REi0intpos),
            'sr_IMjintpos_IMi0intpos' : dsr(s_IMjintpos, s_IMi0intpos),
            'sr_REjintpos_REηintpos'  : dsr(s_REjintpos, s_REηintpos),
            'sr_IMjintpos_IMηintpos'  : dsr(s_IMjintpos, s_IMηintpos),
            'sr_REjDLneg_IMφeneg'   : dsr(s_REjDLneg, s_IMφeneg),
            'sr_REjDLneg_IMφsneg'   : dsr(s_REjDLneg, s_IMφsneg),
            'sr_REjDLneg_IMjintneg' : dsr(s_REjDLneg, s_IMjintneg),
            'sr_REjDLneg_IMjDLneg'  : dsr(s_REjDLneg, s_IMjDLneg),
            'sr_IMjDLneg_REφeneg'   : dsr(s_IMjDLneg, s_REφeneg),
            'sr_IMjDLneg_REφsneg'   : dsr(s_IMjDLneg, s_REφsneg),
            'sr_IMjDLneg_REjintneg' : dsr(s_IMjDLneg, s_REjintneg),
            'sr_IMjDLneg_REjDLneg'  : dsr(s_IMjDLneg, s_REjDLneg),
            'sr_REjDLpos_IMφepos'   : dsr(s_REjDLpos, s_IMφepos),
            'sr_REjDLpos_IMφspos'   : dsr(s_REjDLpos, s_IMφspos),
            'sr_REjDLpos_IMjintpos' : dsr(s_REjDLpos, s_IMjintpos),
            'sr_REjDLpos_IMjDLpos'  : dsr(s_REjDLpos, s_IMjDLpos),
            'sr_IMjDLpos_REφepos'   : dsr(s_IMjDLpos, s_REφepos),
            'sr_IMjDLpos_REφspos'   : dsr(s_IMjDLpos, s_REφspos),
            'sr_IMjDLpos_REjintpos' : dsr(s_IMjDLpos, s_REjintpos),
            'sr_IMjDLpos_REjDLpos'  : dsr(s_IMjDLpos, s_REjDLpos),
            'sr_REi0intneg_REcsnegsurf' : dsr(s_REi0intneg, s_REcsnegsurf),
            'sr_IMi0intneg_IMcsnegsurf' : dsr(s_IMi0intneg, s_IMcsnegsurf),
            'sr_REi0intneg_REceneg'     : dsr(s_REi0intneg, s_REceneg),
            'sr_IMi0intneg_IMceneg'     : dsr(s_IMi0intneg, s_IMceneg),
            'sr_REi0intpos_REcspossurf' : dsr(s_REi0intpos, s_REcspossurf),
            'sr_IMi0intpos_IMcspossurf' : dsr(s_IMi0intpos, s_IMcspossurf),
            'sr_REi0intpos_REcepos'     : dsr(s_REi0intpos, s_REcepos),
            'sr_IMi0intpos_IMcepos'     : dsr(s_IMi0intpos, s_IMcepos),
            'sr_REηintneg_REcsnegsurf' : dsr(s_REηintneg, s_REcsnegsurf),
            'sr_IMηintneg_IMcsnegsurf' : dsr(s_IMηintneg, s_IMcsnegsurf),
            'sr_REηintpos_REcspossurf' : dsr(s_REηintpos, s_REcspossurf),
            'sr_IMηintpos_IMcspossurf' : dsr(s_IMηintpos, s_IMcspossurf),
            'sr_REηintneg_REjintneg' : dsr(s_REηintneg, s_REjintneg),
            'sr_REηintneg_REjDLneg'  : dsr(s_REηintneg, s_REjDLneg),
            'sr_IMηintneg_IMjintneg' : dsr(s_IMηintneg, s_IMjintneg),
            'sr_IMηintneg_IMjDLneg'  : dsr(s_IMηintneg, s_IMjDLneg),
            'sr_REηintpos_REjintpos' : dsr(s_REηintpos, s_REjintpos),
            'sr_REηintpos_REjDLpos'  : dsr(s_REηintpos, s_REjDLpos),
            'sr_IMηintpos_IMjintpos' : dsr(s_IMηintpos, s_IMjintpos),
            'sr_IMηintpos_IMjDLpos'  : dsr(s_IMηintpos, s_IMjDLpos),
            })

        if lithiumPlating:
            slices_.update({
                'sr_REceneg_REjLP' : dsr(s_REceneg, s_REjLP),
                'sr_IMceneg_IMjLP' : dsr(s_IMceneg, s_IMjLP),
                'sr_REφsneg_REjLP' : dsr(s_REφsneg, s_REjLP),
                'sr_IMφsneg_IMjLP' : dsr(s_IMφsneg, s_IMjLP),
                'sr_REjDLneg_IMjLP' : dsr(s_REjDLneg, s_IMjLP),
                'sr_IMjDLneg_REjLP' : dsr(s_IMjDLneg, s_REjLP),
                'sr_REηintneg_REjLP' : dsr(s_REηintneg, s_REjLP),
                'sr_IMηintneg_IMjLP' : dsr(s_IMηintneg, s_IMjLP),
                'sr_REjLP_REceneg' : dsr(s_REjLP, s_REceneg),
                'sr_IMjLP_IMceneg' : dsr(s_IMjLP, s_IMceneg),
                'sr_REjLP_REηLP'   : dsr(s_REjLP, s_REηLP),
                'sr_IMjLP_IMηLP'   : dsr(s_IMjLP, s_IMηLP),
                'sr_REηLP_REjintneg' : dsr(s_REηLP, s_REjintneg),
                'sr_REηLP_REjDLneg'  : dsr(s_REηLP, s_REjDLneg),
                'sr_REηLP_REjLP'     : dsr(s_REηLP, s_REjLP),
                'sr_IMηLP_IMjintneg' : dsr(s_IMηLP, s_IMjintneg),
                'sr_IMηLP_IMjDLneg'  : dsr(s_IMηLP, s_IMjDLneg),
                'sr_IMηLP_IMjLP'     : dsr(s_IMηLP, s_IMjLP),
                })

        self.sKf = namedtuple('SlicesKf', slices_.keys())(**slices_)
        del slices_
        self.ravelKf_ = ravelKf_ = eye(NKf).ravel()  # (NKf*NKf,) 频域因变量线性矩阵Kf__展平视图
        self.bKf_ = zeros(NKf)  # Kf__ @ X_ = bKf_

        # 对Kf__矩阵赋恒定值

        # 负极、正极固相电势REφsneg行、IMφsneg行、REφspos行、IMφspos行
        for s_REφs, s_IMφs, Nreg in zip(
                (s_REφsneg, s_REφspos),
                (s_IMφsneg, s_IMφspos),
                (Nneg, Npos),):
            ravelKf_[dsr(s_REφs, s_REφs)] = \
            ravelKf_[dsr(s_IMφs, s_IMφs)] = [-1] + [-2]*(Nreg - 2) + [-1]  # REφs列、IMφs列主对角线
            ravelKf_[dsr(s_REφs, s_REφs, -1)] = \
            ravelKf_[dsr(s_REφs, s_REφs,  1)] = \
            ravelKf_[dsr(s_IMφs, s_IMφs, -1)] = \
            ravelKf_[dsr(s_IMφs, s_IMφs,  1)] = 1  # REφs列、IMφs列上下对角线

        # 电解液电势REφe行、IMφe行
        ravelKf_[dsr(s_REφeneg, s_REjintneg)] = \
        ravelKf_[dsr(s_REφeneg, s_REjDLneg) ] = \
        ravelKf_[dsr(s_IMφeneg, s_IMjintneg)] = \
        ravelKf_[dsr(s_IMφeneg, s_IMjDLneg) ] = Δxneg = self.Δxneg   # REjneg、IMjneg列
        ravelKf_[dsr(s_REφepos, s_REjintpos)] = \
        ravelKf_[dsr(s_REφepos, s_REjDLpos) ] = \
        ravelKf_[dsr(s_IMφepos, s_IMjintpos)] = \
        ravelKf_[dsr(s_IMφepos, s_IMjDLpos) ] = self.Δxpos   # REjpos、IMjpos列

        # 负极、正极过电位REηint行、IMηint行
        ravelKf_[dsr(s_REηintneg, s_REφeneg)] = \
        ravelKf_[dsr(s_IMηintneg, s_IMφeneg)] = \
        ravelKf_[dsr(s_REηintpos, s_REφepos)] = \
        ravelKf_[dsr(s_IMηintpos, s_IMφepos)] = 1   # REφe列、IMφe列
        ravelKf_[dsr(s_REηintneg, s_REφsneg)] = \
        ravelKf_[dsr(s_IMηintneg, s_IMφsneg)] = \
        ravelKf_[dsr(s_REηintpos, s_REφspos)] = \
        ravelKf_[dsr(s_IMηintpos, s_IMφspos)] = -1  # REφs列、IMφs列

        # 析锂补充
        if lithiumPlating:
            ravelKf_[dsr(s_REφeneg, s_REjLP)] = \
            ravelKf_[dsr(s_IMφeneg, s_IMjLP)] = Δxneg  # REφe行REjLP列、IMφe行IMjLP列
            ravelKf_[dsr(s_REηLP, s_REφsneg)] = \
            ravelKf_[dsr(s_IMηLP, s_IMφsneg)] = -1  # REηLP行REφsneg列、IMηLP行IMφsneg列
            ravelKf_[dsr(s_REηLP, s_REφeneg)] = \
            ravelKf_[dsr(s_IMηLP, s_IMφeneg)] = 1   # REηLP行REφeneg列、IMηLP行IMφeneg列

        if self.verbose:
            print(f'频域因变量线性矩阵 {ravelKf_.base.shape = }')

    def update_Kf__REce_REce_and_IMce_IMce_(self,
                                            DeeffWest_, DeeffEast_,
                                            ):
        # 更新Kf__矩阵REce行REce列、IMce行IMce列
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Nneg, Nsep = self.Nneg, self.Nsep
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        dl_ = ravelKf_[sKf.sr_REce_REce_l]
        du_ = ravelKf_[sKf.sr_REce_REce_u]
        d_  = ravelKf_[sr_REce_REce := sKf.sr_REce_REce]
        dl_[:] = -DeeffWest_[1:] /ΔxWest_[1:]   # (Ne-1,) REce列下对角线
        du_[:] = -DeeffEast_[:-1]/ΔxEast_[:-1]  # (Ne-1,) REce列上对角线
        d_[:] = -(hstack([0., dl_]) + hstack([du_, 0.]))  # (Ne,) REce列主对角线
        NKf1 = self.bKf_.size + 1
        start0 = sr_REce_REce.start
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            nrW = start0 + nW*NKf1
            nrE = start0 + nE*NKf1
            a, c = -DeeffWest_[nW]/ΔxWest_[nW], -2*DeeffEast_[nW]*DeeffWest_[nE]/(DeeffWest_[nE]*Δx_[nW] + DeeffEast_[nW]*Δx_[nE])
            ravelKf_[nrW-1:nrW+2] = a, -(a + c), c  # 界面左侧控制体
            a, c = c, -DeeffEast_[nE]/ΔxEast_[nE]
            ravelKf_[nrE-1:nrE+2] = a, -(a + c), c  # 界面右侧控制体
        ravelKf_[sKf.sr_IMce_IMce_l] = dl_  # IMce列下对角线
        ravelKf_[sKf.sr_IMce_IMce_u] = du_  # IMce列上对角线
        ravelKf_[sKf.sr_IMce_IMce]   = d_   # IMce列主对角线

    def update_Kf__REφe_REce_and_IMφe_IMce_(self,
            κDeffWest_, κDeffEast_,
            DeeffWest_, DeeffEast_,
            ce_, ceInterfaces_,
            ):
        # 更新Kf__矩阵REφe行REce列、IMφe行IMce列
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Nneg, Nsep = self.Nneg, self.Nsep
        ceWest_ = ceInterfaces_[:-1]                                  # (Ne,) 各控制体左界面的电解液锂离子浓度
        ceEast_ = ceInterfaces_[1:]                                   # (Ne,) 各控制体右界面的电解液锂离子浓度
        gradceWest_ = hstack([0, (ce_[1:] - ce_[:-1])/ΔxWest_[1:]])   # (Ne,) 各控制体左界面的锂离子浓度梯度
        gradceEast_ = hstack([(ce_[1:] - ce_[:-1])/ΔxEast_[:-1], 0])  # (Ne,) 各控制体右界面的锂离子浓度梯度
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradceEast_[nW] = (ceEast_[nW] - ce_[nW])/(0.5*Δx_[nW])
            gradceWest_[nE] = (ce_[nE] - ceWest_[nE])/(0.5*Δx_[nE])
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        dl_ = ravelKf_[sKf.sr_REφe_REce_l]
        du_ = ravelKf_[sKf.sr_REφe_REce_u]
        d_  = ravelKf_[sr_REφe_REce := sKf.sr_REφe_REce]
        a_  = κDeffWest_[1:]  / ceWest_[1:]  / ΔxWest_[1:]
        c_  = κDeffEast_[:-1] / ceEast_[:-1] / ΔxEast_[:-1]
        aa_ = κDeffWest_[1:]  * gradceWest_[1:]  / ceWest_[1:]**2 * 0.5
        cc_ = κDeffEast_[:-1] * gradceEast_[:-1] / ceEast_[:-1]**2 * 0.5
        dl_[:] = -aa_ - a_
        du_[:] = cc_ - c_
        d_[:]  = hstack([0, a_ - aa_]) + hstack([c_ + cc_, 0])
        NKf1 = self.bKf_.size + 1
        start0 = sr_REφe_REce.start
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            nrW = start0 + nW*NKf1
            nrE = start0 + nE*NKf1

            DeeffWest_nE_Δx_nW = DeeffWest_[nE]*Δx_[nW]
            DeeffEast_nW_Δx_nE = DeeffEast_[nW]*Δx_[nE]
            den = DeeffWest_nE_Δx_nW + DeeffEast_nW_Δx_nE
            pDW = DeeffEast_nW_Δx_nE / den
            pDE = 1 - pDW   # 即：DeeffWest_nE_Δx_nW / den
            # 界面左侧控制体
            a  = κDeffWest_[nW]   / ceWest_[nW]     / ΔxWest_[nW]
            aa = κDeffWest_[nW]   * gradceWest_[nW] / ceWest_[nW]**2 * 0.5
            c  = 2*κDeffEast_[nW] * DeeffWest_[nE]  / ceEast_[nW] / den
            cc = κDeffEast_[nW]   * gradceEast_[nW] / ceEast_[nW]**2
            ravelKf_[nrW-1:nrW+2] = -aa - a, a - aa + c + cc*pDW, cc*pDE - c
            # 界面右侧控制体
            a  = 2*κDeffWest_[nE] * DeeffEast_[nW]  / ceWest_[nE] / den
            aa = κDeffWest_[nE]   * gradceWest_[nE] / ceWest_[nE]**2
            c  = κDeffEast_[nE]   / ceEast_[nE]     / ΔxEast_[nE]
            cc = κDeffEast_[nE]   * gradceEast_[nE] / ceEast_[nE]**2 * 0.5
            ravelKf_[nrE-1:nrE+2] = -a - aa*pDW , a - aa*pDE + c + cc, cc - c
        # 电解液电势虚部IMφe行
        ravelKf_[sKf.sr_IMφe_IMce_l] = dl_  # IMce列下对角线
        ravelKf_[sKf.sr_IMφe_IMce_u] = du_  # IMce列上对角线
        ravelKf_[sKf.sr_IMφe_IMce]   = d_   # IMce列主对角线

    def update_Kf__REφsneg_REjneg_and_IMφsneg_IMjneg_(self, σeffneg):
        # 更新Kf__矩阵REφsneg行REjneg列、IMφsneg行IMjneg列
        Δxneg = self.Δxneg
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        ravelKf_[sKf.sr_REφsneg_REjintneg] = \
        ravelKf_[sKf.sr_REφsneg_REjDLneg ] = \
        ravelKf_[sKf.sr_IMφsneg_IMjintneg] = \
        ravelKf_[sKf.sr_IMφsneg_IMjDLneg ] = a = -Δxneg*Δxneg/σeffneg
        if self.lithiumPlating:
            ravelKf_[sKf.sr_REφsneg_REjLP] = \
            ravelKf_[sKf.sr_IMφsneg_IMjLP] = a

    def update_Kf__REφspos_REjpos_and_IMφspos_IMjpos_(self, σeffpos):
        # 更新Kf__矩阵REφspos行REjpos列、IMφspos行IMjpos列
        Δxpos = self.Δxpos
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        ravelKf_[sKf.sr_REφspos_REjintpos] = \
        ravelKf_[sKf.sr_REφspos_REjDLpos ] = \
        ravelKf_[sKf.sr_IMφspos_IMjintpos] = \
        ravelKf_[sKf.sr_IMφspos_IMjDLpos ] = -Δxpos*Δxpos/σeffpos

    def update_Kf__REφe_REφe_and_IMφe_IMφe_(self, κeffWest_, κeffEast_):
        # 更新Kf__矩阵REφe行REφe列、IMφe行IMφe列
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Nneg, Nsep = self.Nneg, self.Nsep
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        dl_ = ravelKf_[sKf.sr_REφe_REφe_l]
        du_ = ravelKf_[sKf.sr_REφe_REφe_u]
        d_  = ravelKf_[sr_REφe_REφe := sKf.sr_REφe_REφe]
        dl_[:] = κeffWest_[1:]/ΔxWest_[1:]               # (Ne-1,) REφe列下对角线
        du_[:] = κeffEast_[:-1]/ΔxEast_[:-1]             # (Ne-1,) REφe列上对角线
        d_[:]  = -(hstack([0, dl_]) + hstack([du_, 0]))  # (Ne,) REφe列主对角线
        d_[0] -= κeffWest_[0]/(0.5*Δx_[0])               # 首元占优
        NKf1 = self.bKf_.size + 1
        start0 = sr_REφe_REφe.start
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            nrW = start0 + nW*NKf1
            nrE = start0 + nE*NKf1
            a, c = κeffWest_[nW]/ΔxWest_[nW], 2*κeffEast_[nW]*κeffWest_[nE]/(κeffWest_[nE]*Δx_[nW] + κeffEast_[nW]*Δx_[nE])
            ravelKf_[nrW-1:nrW+2] = a, -(a + c), c  # 界面左侧控制体
            a, c = c, κeffEast_[nE]/ΔxEast_[nE]
            ravelKf_[nrE-1:nrE+2] = a, -(a + c), c  # 界面右侧控制体

        ravelKf_[sKf.sr_IMφe_IMφe_l] = dl_  # IMφe列下对角线
        ravelKf_[sKf.sr_IMφe_IMφe_u] = du_  # IMφe列上对角线
        ravelKf_[sKf.sr_IMφe_IMφe]   = d_   # IMφe列主对角线

    def update_Kf__REηintneg_REjneg_and_IMηintneg_IMjneg_(self, RSEIneg, aeffneg):
        # 更新Kf__矩阵REηintneg行REjneg列、IMηintneg行IMjneg列
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        ravelKf_[sKf.sr_REηintneg_REjintneg] = \
        ravelKf_[sKf.sr_REηintneg_REjDLneg ] = \
        ravelKf_[sKf.sr_IMηintneg_IMjintneg] = \
        ravelKf_[sKf.sr_IMηintneg_IMjDLneg ] = a = RSEIneg/aeffneg
        if self.lithiumPlating:
            ravelKf_[sKf.sr_REηintneg_REjLP] = \
            ravelKf_[sKf.sr_IMηintneg_IMjLP] = a

    def update_Kf__REηintpos_REjpos_and_IMηintpos_IMjpos_(self, RSEIpos, aeffpos):
        # 更新Kf__矩阵REηintpos行REjpos列、IMηintpos行IMjpos列
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        ravelKf_[sKf.sr_REηintpos_REjintpos] = \
        ravelKf_[sKf.sr_REηintpos_REjDLpos ] = \
        ravelKf_[sKf.sr_IMηintpos_IMjintpos] = \
        ravelKf_[sKf.sr_IMηintpos_IMjDLpos ] = RSEIpos/aeffpos

    def update_Kf__REηLP_REjneg_and_IMηLP_IMjneg_(self, RSEIneg, aeffneg):
        # 更新Kf__矩阵REηLP行REJneg列、IMηLP行IMJneg列
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        ravelKf_[sKf.sr_REηLP_REjintneg] = \
        ravelKf_[sKf.sr_REηLP_REjDLneg ] = \
        ravelKf_[sKf.sr_REηLP_REjLP    ] = \
        ravelKf_[sKf.sr_IMηLP_IMjintneg] = \
        ravelKf_[sKf.sr_IMηLP_IMjDLneg ] = \
        ravelKf_[sKf.sr_IMηLP_IMjLP    ] = RSEIneg/aeffneg

    def record_EISdata(self):
        """记录阻抗数据"""
        data = self.data
        for name in self.EISdatanames_:
            value = getattr(self, name)
            if isscalar(value):
                pass
            else:
                value = value.copy()
            data[name].append(value)

    @property
    def Zl_(self):
        """全电池感抗 [Ω]"""
        return 1j*self.ωl_

    @property
    def Zsep_(self):
        """隔膜复阻抗 [Ω]"""
        return self.Z_ - self.Zneg_ - self.Zpos_ - self.Zl_   # 隔膜复阻抗 [Ω]

    @property
    def ω_(self):
        """角频率序列 [rad/s]"""
        return 6.283185307179586*self.f_

    @property
    def ωl_(self):
        """感抗序列 [Ω]"""
        return self.ω_*self.l

    def interpolate(self,
            variableName: str,           # 字符串：所需插值的变量名
            t_: Sequence,                # 时刻序列 [s]
            x_: Sequence | None = None,  # 厚度方向坐标序列 [m]
            r_: Sequence | None = None,  # 球形颗粒半径方向坐标序列 [m]
            f_: Sequence | None = None,  # 频率 [Hz]
            ) -> ndarray:
        if variableName in self.EISdatanames_:
            tEIS_ = self.data['tEIS']
            logf_ = log10(self.f_)
            kw = dict(bounds_error=False,  # 超出边界不报错
                      fill_value=None,)    # None表示外推
            if variableName.endswith('__'):
                # 与时间t、频率f、厚度方向坐标x相关的变量
                if ('neg' in variableName) or ('LP' in variableName):
                    location_ = self.xneg_
                elif 'pos' in variableName:
                    location_ = self.xpos_
                else:
                    location_ = self.x_
                interpolator = RegularGridInterpolator([tEIS_, logf_, location_], self.data[variableName], **kw)
                points____ = stack(meshgrid(t_, log10(f_), x_,
                                      indexing='ij'), axis=-1)  # (Nt, Nf, Nx, 3) 待插值点
                variable___ = interpolator(points____)  # 插值
                return variable___
            else:
                # 与时间t、频率f相关的变量
                interpolator = RegularGridInterpolator([tEIS_, logf_], self.data[variableName], **kw)
                points___ = stack(meshgrid(t_, log10(f_), indexing='ij'), axis=-1)  # (Nt, Nf, 2) 待插值点
                variable__ = interpolator(points___)  # 插值
                return variable__
        else:
            return P2Dbase.interpolate(self, variableName, t_, x_, r_)

    def plot_Z(self, f: int | float | None = None,
               t_: Sequence | None = None,
               ):
        """频率f复阻抗实部-时间、频率f复阻抗虚部-时间"""
        if f is None:
            f = self.f_[0]
        if t_ is None:
            t_ = self.data['tEIS']

        Z_ = self.interpolate('Z_', t_=t_, f_=f)
        Zneg_ = self.interpolate('Zneg_', t_=t_, f_=f)
        Zpos_ = self.interpolate('Zpos_', t_=t_, f_=f)

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        ax1.set_position([.1, .68, .85, 0.18])
        ax2.set_position([.1, .48, .85, 0.18])
        ax3.set_position([.1, .28, .85, 0.18])
        ax4.set_position([.1, .08, .85, 0.18])

        ax1.set_title(rf'Impedance at ${{\it f}}\;=\;{f:g}\;Hz$', fontsize=12, pad=36)
        ax1.plot(t_, Z_.real*1000, 'k-o', label=r'$\it Z$')
        ax1.plot(t_, Zneg_.real*1000, 'b-^', label=r'${\it Z}_{neg}$')
        ax1.plot(t_, Zpos_.real*1000, 'r-s', label=r'${\it Z}_{neg}$')
        ax1.set_ylabel(r'${\it Z}′$ [mΩ]')
        ax1.set_xticks([])
        ax1.legend(facecolor='none', edgecolor='none', framealpha=0.8,
                   ncols=4, fontsize=16, loc=[0.2, 1.02], )

        ax2.plot(t_, -Z_.imag*1000, 'k-o', label='Full cell')
        ax2.plot(t_, -Zneg_.imag*1000, 'b-^')
        ax2.plot(t_, -Zpos_.imag*1000, 'r-s')
        ax2.set_ylabel(r'$-{\it Z}″$ [mΩ]')
        ax2.set_xticks([])

        ax3.plot(t_, abs(Z_)*1000, 'k-o')
        ax3.plot(t_, abs(Zneg_)*1000, 'b-^')
        ax3.plot(t_, abs(Zpos_)*1000, 'r-s')
        ax3.set_ylabel(r'$|{\it Z}|$ [mΩ]')
        ax3.set_xticks([])
        from numpy import angle

        ax4.plot(t_, -angle(Z_, deg=True), 'k-o')
        ax4.plot(t_, -angle(Zneg_, deg=True), 'b-^')
        ax4.plot(t_, -angle(Zpos_, deg=True), 'r-s')
        ax4.set_ylabel(r'$-∠{\it Z}$ [°]')
        ax4.set_xlabel(r'Time $\it t$ [s]')

        duration = ptp(t_)
        xlim_ = t_[0]-duration*0.02, t_[-1]+duration*0.02
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(xlim_)
            ax.grid(axis='y', linestyle='--', color=[.5, .5, .5])
            ax.minorticks_on()
        plt.show()

    def plot_Nyquist(self, Z: str = 'Z_',  # 'Z_' 'Zneg_' 'Zpos_' 'Zsep'
                     t_: Sequence | None = None,  # 时刻序列
                     f_: Sequence | None = None,  # 频率序列
                     ):
        """Nyquist图"""
        if t_ is None:
            t_ = self.data['tEIS']
        if f_ is None:
            f_ = self.f_
        Z__ = self.interpolate(Z, t_=t_, f_=f_)*1e3  # 呈时间序列的阻抗谱

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        ax.set_position([.1, .08, .75, 0.8])
        ax.set_title(rf'Nyquist plot of ${{\it Z}}_{{{Z[1:-1]}}}$', fontsize=12)
        for n, (Z_, t) in enumerate(zip(Z__, t_)):
            ax.plot(Z_.real, -Z_.imag, 'o-', color=get_color(t_, n),
                    label=rf'$\it t$ = {t:g} s')
        ax.set_ylabel(rf'Imaginary part of impedance $-{{\it Z″}}_{{{Z[1:-1]}}}\;\;{{[mΩ]}}$')
        ax.set_xlabel(rf'Real part of impedance ${{\it Z′}}_{{{Z[1:-1]}}}\;\;{{[mΩ]}}$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(linestyle='--')

        i = ptp(abs(Z__), axis=1).argmax()
        for x, y, f in zip(Z__[i].real, -Z__[i].imag, f_):
            ax.text(x, y,
                    f'  {f:g} Hz',
                    backgroundcolor='w',
                    va='center', ha='left',
                    fontsize=10,
                    bbox=dict(boxstyle='square,pad=0.4', fc='none', ec='none', lw=0.5, alpha=0.8)
                    )
        plt.show()

    def plot_REcssurf_IMcssurf(self,
                               t_: Sequence | None = None,
                               f_: Sequence | None = None):
        """固相表面锂离子浓度实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        cθ = 'c' if self.xUnit else 'θ'  # 浓度符号
        REcsnegsurf__ = self.interpolate(f'RE{cθ}snegsurf__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相表面浓度实部序列
        IMcsnegsurf__ = self.interpolate(f'IM{cθ}snegsurf__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相表面浓度虚部序列
        REcspossurf__ = self.interpolate(f'RE{cθ}spossurf__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相表面浓度实部序列
        IMcspossurf__ = self.interpolate(f'IM{cθ}spossurf__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相表面浓度虚部序列
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (REcsnegsurf_, REcspossurf_, label) in enumerate(zip(REcsnegsurf__, REcspossurf__, labels_)):
            ax1.plot(self.xPlot_, hstack([REcsnegsurf_, full(self.Nsep, nan), REcspossurf_]), 'o-',
                     color=get_color(labels_, n), label=label)
        ax1.set_ylabel(f'{self.cSign}′$_{{s,AC}}$({self.xSign}, {self.rSign}; ${{\\it f}}$, ${{\\it t}}$)|$_{{ {self.rSign[1:-1]} = {"{\\it R}_{s,reg}" if self.rUnit else 1} }}$ [{self.cUnit or '–'}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMcsnegsurf_, IMcspossurf_, label) in enumerate(zip(IMcsnegsurf__, IMcspossurf__, labels_)):
            y_ = *IMcsnegsurf_, *[nan]*self.Nsep, *IMcspossurf_
            ax2.plot(self.xPlot_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def plot_REce_IMce(self,
                       t_: Sequence | None = None,
                       f_: Sequence | None = None):
        """电解液锂离子浓度实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        cθ = 'c' if self.xUnit else 'θ'  # 浓度符号
        REce__ = self.interpolate(f'RE{cθ}e__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液电势实部序列
        IMce__ = self.interpolate(f'IM{cθ}e__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液电势虚部序列
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (REce_, label) in enumerate(zip(REce__, labels_)):
            ax1.plot(self.xPlot_, REce_, 'o-', color=get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.cSign}′$_{{e,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.cUnit or '–'}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMce_, label) in enumerate(zip(IMce__, labels_)):
            ax2.plot(self.xPlot_, IMce_, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def plot_REφs_IMφs(self,
                       t_: Sequence | None = None,
                       f_: Sequence | None = None):
        """固相电势实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        REφsneg__ = self.interpolate('REφsneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相电势实部序列
        IMφsneg__ = self.interpolate('IMφsneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相电势虚部序列
        REφspos__ = self.interpolate('REφspos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相电势实部序列
        IMφspos__ = self.interpolate('IMφspos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相电势虚部序列
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.set_position([.1, .59, .3, 0.375])
        ax2.set_position([.5, .59, .3, 0.375])
        ax3.set_position([.1, .08, .3, 0.375])
        ax4.set_position([.5, .08, .3, 0.375])

        for n, (REφsneg_, label) in enumerate(zip(REφsneg__, labels_)):
            ax1.plot(self.xPlot_[:self.Nneg], REφsneg_*1e3, 'o-', color=get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'${{\it φ′}}_{{s,neg,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [mV]')
        ax1.set_xlim(0, self.xInterfacesPlot_[self.Nneg])

        for n, (REφspos_, label) in enumerate(zip(REφspos__, labels_)):
            ax2.plot(self.xPlot_[-self.Npos:], REφspos_*1e3, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('neg', 'pos'))
        ax2.set_xlim(self.xInterfacesPlot_[self.Nneg+self.Nsep], self.xInterfacesPlot_[-1])
        ax2.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMφsneg_, label) in enumerate(zip(IMφsneg__, labels_)):
            ax3.plot(self.xPlot_[:self.Nneg], IMφsneg_*1e3, 'o-', color=get_color(labels_, n), label=label)
        ax3.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        ax3.set_xlim(0, self.xInterfacesPlot_[self.Nneg])

        for n, (IMφspos_, label) in enumerate(zip(IMφspos__, labels_)):
            ax4.plot(self.xPlot_[-self.Npos:], IMφspos_*1e3, 'o-', color=get_color(labels_, n), label=label)
        ax4.set_ylabel(ax3.get_ylabel().replace('neg', 'pos'))
        ax4.set_xlim(self.xInterfacesPlot_[self.Nneg+self.Nsep], self.xInterfacesPlot_[-1])

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '—'}]')
            ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_REφe_IMφe(self,
                       t_: Sequence | None = None,
                       f_: Sequence | None = None):
        """电解液电势实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        REφe__ = self.interpolate('REφe__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液电势实部序列
        IMφe__ = self.interpolate('IMφe__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液电势虚部序列
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (REφe_, label) in enumerate(zip(REφe__, labels_)):
            ax1.plot(self.xPlot_, REφe_*1e3, 'o-', color=get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'${{\it φ′}}_{{e,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [mV]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMφe_, label) in enumerate(zip(IMφe__, labels_)):
            ax2.plot(self.xPlot_, IMφe_*1e3, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def plot_REjint_IMjint(self,
                           t_: Sequence | None = None,
                           f_: Sequence | None = None):
        """局部体积电流密度实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        jJ = 'j' if self.xUnit else 'J'
        REjintneg__ = self.interpolate(f'RE{jJ}intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极局部体积电流密度实部序列
        IMjintneg__ = self.interpolate(f'IM{jJ}intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极局部体积电流密度虚部序列
        REjintpos__ = self.interpolate(f'RE{jJ}intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极局部体积电流密度实部序列
        IMjintpos__ = self.interpolate(f'IM{jJ}intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极局部体积电流密度虚部序列
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (REjintneg_, REjintpos_, label) in enumerate(zip(REjintneg__, REjintpos__, labels_)):
            x_ = self.xPlot_
            y_ = *REjintneg_, *([nan]*self.Nsep), *REjintpos_
            ax1.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.jSign}′$_{{int,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.jUnit}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMjintneg_, IMjintpos_, label) in enumerate(zip(IMjintneg__, IMjintpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMjintneg_, *([nan]*self.Nsep), *IMjintpos_
            ax2.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def plot_REjDL_IMjDL(self,
                         t_: Sequence | None = None,
                         f_: Sequence | None = None):
        """双电层效应局部体积电流密度实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        jJ = 'j' if self.xUnit else 'J'
        REjDLneg__ = self.interpolate(f'RE{jJ}DLneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 双电层效应负极局部体积电流密度实部序列
        IMjDLneg__ = self.interpolate(f'IM{jJ}DLneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 双电层效应负极局部体积电流密度虚部序列
        REjDLpos__ = self.interpolate(f'RE{jJ}DLpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 双电层效应正极局部体积电流密度实部序列
        IMjDLpos__ = self.interpolate(f'IM{jJ}DLpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 双电层效应正极局部体积电流密度虚部序列
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (REjDLneg_, REjDLpos_, label) in enumerate(zip(REjDLneg__, REjDLpos__, labels_)):
            x_ = self.xPlot_
            y_ = *REjDLneg_, *([nan]*self.Nsep), *REjDLpos_
            ax1.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.jSign}′$_{{DL,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.jUnit}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMjDLneg_, IMjDLpos_, label) in enumerate(zip(IMjDLneg__, IMjDLpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMjDLneg_, *([nan]*self.Nsep), *IMjDLpos_
            ax2.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def plot_REi0int_IMi0int(self,
                             t_: Sequence | None = None,
                             f_: Sequence | None = None):
        """交换电流密度实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        iI = 'i' if self.xUnit else 'I'
        REi0intneg__ = self.interpolate(f'RE{iI}0intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 呈时间序列的负极交换电流密度实部
        IMi0intneg__ = self.interpolate(f'IM{iI}0intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 呈时间序列的负极交换电流密度虚部
        REi0intpos__ = self.interpolate(f'RE{iI}0intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 呈时间序列的正极交换电流密度实部
        IMi0intpos__ = self.interpolate(f'IM{iI}0intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 呈时间序列的正极交换电流密度虚部
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (REi0intneg_, REi0intpos_, label) in enumerate(zip(REi0intneg__, REi0intpos__, labels_)):
            x_ = self.xPlot_
            y_ = *REi0intneg_, *([nan]*self.Nsep), *REi0intpos_
            ax1.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.i0Sign.replace('}', '′}').replace('0', '{0,AC}')}({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.i0Unit}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMi0intneg_, IMi0intpos_, label) in enumerate(zip(IMi0intneg__, IMi0intpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMi0intneg_, *([nan]*self.Nsep), *IMi0intpos_
            ax2.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def plot_REηint_IMηint(self,
                           t_: Sequence | None = None,
                           f_: Sequence | None = None):
        """固相表面反应过电位实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        REηintneg__ = self.interpolate('REηintneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极反应过电位实部序列 [mV]
        IMηintneg__ = self.interpolate('IMηintneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极反应过电位虚部序列 [mV]
        REηintpos__ = self.interpolate('REηintpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极反应过电位实部序列 [mV]
        IMηintpos__ = self.interpolate('IMηintpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极反应过电位虚部序列 [mV]
        labels_ = [rf'$\it t$ = {t:g} s; $\it f$ = {f:g} Hz' for t in t_ for f in f_]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (REηintneg_, REηintpos_, label) in enumerate(zip(REηintneg__, REηintpos__, labels_)):
            x_ = self.xPlot_
            y_ = *REηintneg_, *([nan]*self.Nsep), *REηintpos_
            ax1.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'${{\it η′}}_{{int,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [mV]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMηintneg_, IMηintpos_, t) in enumerate(zip(IMηintneg__, IMηintpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMηintneg_, *([nan]*self.Nsep), *IMηintpos_
            ax2.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    @abstractmethod
    def EIS(self):
        """计算电化学阻抗谱"""
        pass

    @abstractmethod
    def solve_frequency_dependent_variables(self) -> dict:
        """求解频率相关变量"""
        pass


if __name__=='__main__':
    pass

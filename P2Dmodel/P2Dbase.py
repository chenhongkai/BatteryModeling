#%%
import time, pathlib
from math import exp
from typing import Sequence, Callable
from functools import partial
from collections import deque
from abc import ABC, abstractmethod


import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.linalg.lapack import dgbsv
from scipy.optimize import root, brentq
from numpy import ndarray, nan, inf,\
    array, asarray,arange, zeros, eye, full, empty, unique, isfinite, \
    linspace, logspace, stack, hstack, concatenate, meshgrid, \
    cbrt, log10, ptp, ix_, isclose, savez, minimum, maximum
from numba import njit

from P2Dmodel.OCP import NMC111, Graphite
from P2Dmodel.tools import (Interpolate1D, set_matplotlib, get_color,
                            diagonalSliceRavel, stepping_aware_cached_property,
                            F, R, JITinterp1d, SlicesK, SlicesKf)

εOCP = 1e-4  # OCP及其导数计算中的无量纲固相浓度保护阈值 [–]


class P2Dbase(ABC):
    """抽象类：锂离子电池时频联合准二维模型 Joint Time-Frequency Pseudo-two-Dimensional model"""

    ## 常数 ##
    F = F  # 法拉第Faraday常数 [C/mol]
    R = R  # 理想气体常数 [J/(mol·K)]

    __slots__ = (
        # 通用参数名
        'θminneg', 'θmaxneg', 'θminpos', 'θmaxpos',
        'UOCPneg', 'UOCPpos', 'solve_dUOCPdθsneg_', 'solve_dUOCPdθspos_',
        'dUOCPdTneg', 'dUOCPdTpos',
        '_kneg', '_kpos', '_Dsneg', '_Dspos',
        'σneg', 'σpos', 'Rfneg', 'Rfpos',
        'CDLneg', 'CDLpos', 'l',
        '_i0intneg', '_i0intpos',
        # 网格参数
        'Δt', 'Nneg', 'Nsep', 'Npos', 'Ne', 'Nr',
        'x_', 'Δx_', 'ΔxWest_', 'ΔxEast_', 'Δxneg', 'Δxpos',
        'rneg_', 'rpos_', 'Δrneg_', 'Δrpos_', 'Vr_',
        # 热参数
        'EDsneg', 'EDspos', 'Ekneg', 'Ekpos', 'Eκ', 'EDe',
        'Tref', 'Rth', 'Cth',
        # 模式
        'timeDiscretization', 'radialDiscretization', 'timeDL',
        'complete', 'verbose',
        # 通用时域状态量
        'T', 'I', 't',
        'φsneg_', 'φspos_', 'φe_',
        'ηintneg_', 'ηintpos_',
        'nNewton',
        # 通用频域状态量
        'tEIS', 'Z_', 'Zneg_', 'Zpos_',
        'REφsneg__',   'IMφsneg__',   'REφspos__',   'IMφspos__',  'REφe__', 'IMφe__',
        'REηintneg__', 'IMηintneg__', 'REηintpos__', 'IMηintpos__',
        # 恒定量
        'datanames_', 'EISdatanames_',
        'banded_experience_of_J__',
        'banded_experience_of_Kf__',
        'bandKcsneg__', 'bandKcspos__',
        'e__', 'coeffsExpl_',
        'f_', 'ω_', 'ΔIAC',
        # 因变量矩阵
        'ravelK_', 'bK_',
        'ravelKf_', 'bKf_',
        'sK', 'sKf', # 索引时域、频域因变量
        # 数据记录
        'data', 'ΔφsenegHistory__', 'ΔφseposHistory__', 'Ihistory_',
        # 作图恒定量
        'tSign', 'tUnit',
        'xSign', 'xUnit',
        'rSign', 'rUnit',
        'cSign', 'cUnit',
        'jSign', 'jUnit',
        'i0Sign', 'i0Unit',
        # 缓存@property相关
        '_cached_t', '_cached_properties',
        )

    # 类型注解 ##
    Qcell: float; Qneg: float; Qpos: float  # 理论可用容量、负极、正极容量 [Ah]
    _kneg: float; _kpos: float    # 负极、正极主反应速率常数
    _Dsneg: float; _Dspos: float  # 负极、正极固相扩散系数

    def __init__(self,
            Lneg: float = 1.,   # 负极厚度 [m]/[–]
            Lsep: float = 1.,   # 隔膜厚度 [m]/[–]
            Lpos: float = 1.,   # 正极厚度 [m]/[–]
            Rsneg: float = 1.,  # 负极固相颗粒半径 [m]/[–]
            Rspos: float = 1.,  # 正极固相颗粒半径 [m]/[–]
            T0: float = 298.15,         # 初始温度 [K]
            SOC0: float = 0.2,          # 初始荷电状态 [–]
            θminneg: float = .0370744,  # SOC=0%的负极嵌锂状态 [–]
            θmaxneg: float = .8775600,  # SOC=100%的负极嵌锂状态 [–]
            θminpos: float = .0746557,  # SOC=100%的正极嵌锂状态 [–]
            θmaxpos: float = .9589741,  # SOC=0%的正极嵌锂状态 [–]
            Δt: float | int = 10.,      # 时间步长 [s]
            Nneg: int = 10,             # 负极区域网格数
            Nsep: int = 10,             # 隔膜区域网格数
            Npos: int = 10,             # 正极区域网格数
            Nr: int = 10,               # 球形固相颗粒径向网格数
            UOCPneg: Callable = Graphite().Graphite_COMSOL,  # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位UOCPneg_ [V]
            UOCPpos: Callable = NMC111().NMC111_COMSOL,      # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位UOCPpos_ [V]
            dUOCPdθsneg: Callable | None = None,  # None或函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位对嵌锂状态的导数 [V/–]
            dUOCPdθspos: Callable | None = None,  # None或函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位对嵌锂状态的导数 [V/–]
            dUOCPdTneg: Callable | float = 0.,    # 标量或函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位的熵热系数 [V/K]
            dUOCPdTpos: Callable | float = 0.,    # 标量或函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位的熵热系数 [V/K]
            f_: Sequence[float] = logspace(3, -1, 21),  # 频率序列 [Hz]
            EDsneg: float = 30e3,  # 负极固相锂离子扩散系数的比活化能 [J/mol]
            EDspos: float = 30e3,  # 正极固相锂离子扩散系数的比活化能 [J/mol]
            Ekneg: float = 68e3,   # 负极主反应速率常数的比活化能 [J/mol]
            Ekpos: float = 50e3,   # 正极主反应速率常数的比活化能 [J/mol]
            Eκ: float = 4e3,       # 电解液锂离子电导率的比活化能 [J/mol]
            EDe: float = 16.5e3,   # 电解液锂离子扩散系数的比活化能 [J/mol]
            Tref: float = 298.15,  # 参考温度 [K]
            Rth: float = 1.,       # 传热热阻 [K/W]
            Cth: float = 50.,      # 热容 [J/K]
            timeDiscretization: str = 'CN',    # 时间离散格式 'CN'/'backward'
            radialDiscretization: str = 'EV',  # 球形颗粒径向离散方法 等体积'EV'/等间隔'EI'
            complete: bool = True,  # 是否确保功能完备，设置complete==False可加速，但省略不必要的计算和存储
            verbose: bool = True,   # 是否显示初始化、运行进度
            timeDL: bool = True,    # 时域是否考虑双电层电流（频域始终保留）
            ):
        # 通用参数
        assert T0>0, f'初始温度{T0 = }，应大于0 [K]'
        assert 0<=SOC0<=1, f'初始荷电状态{SOC0 = }，取值范围应为[0, 1]'
        # 4边界嵌锂状态参数
        assert 0<θminneg<θmaxneg<1, f'负极最小、最大嵌锂状态{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1'
        assert 0<θminpos<θmaxpos<1, f'正极最小、最大嵌锂状态{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1'
        self.θminneg = θminneg  # SOC=0%的负极嵌锂状态
        self.θmaxneg = θmaxneg  # SOC=100%的负极嵌锂状态
        self.θminpos = θminpos  # SOC=100%的正极嵌锂状态
        self.θmaxpos = θmaxpos  # SOC=0%的正极嵌锂状态
        # 网格参数
        self.Δt = Δt; assert Δt>0, f'时间步长{Δt = }，应大于0 [s]'
        self.Nneg = Nneg; assert isinstance(Nneg, int) and Nneg>=3, f'负极区域网格数{Nneg = }，应为不小于3的正整数'
        self.Nsep = Nsep; assert isinstance(Nsep, int) and Nsep>=3, f'隔膜区域网格数{Nsep = }，应为不小于3的正整数'
        self.Npos = Npos; assert isinstance(Npos, int) and Npos>=3, f'正极区域网格数{Npos = }，应为不小于3的正整数'
        self.Nr = Nr;     assert isinstance(Nr, int) and Nr>=3, f'球形固相颗粒半径方向网格数{Nr = }，应为不小于3的正整数'
        self.Ne = Ne = Nneg + Nsep + Npos  # 电解液网格总数
        # 频率
        self.f_ = f_ = asarray(f_); assert f_.ndim==1, f'频率序列f_应可转化为ndim==1的ndarray，当前{f_ = }'
        assert (f_>0).all(), f'频率序列f_中的所有频率均应大于0 [Hz]，当前{f_ = }'
        self.ω_ = 6.283185307179586*f_  # 角频率序列 [rad/s]
        Nf = f_.size
        # 函数
        self.UOCPneg = UOCPneg; assert callable(UOCPneg), '函数UOCPneg，输入负极嵌锂状态θsneg_ [–]，输出负极开路电位UOCPneg_ [V]'
        self.UOCPpos = UOCPpos; assert callable(UOCPpos), '函数UOCPpos，输入正极嵌锂状态θspos_ [–]，输出正极开路电位UOCPpos_ [V]'
        assert callable(dUOCPdθsneg) or (dUOCPdθsneg is None), '负极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入负极嵌锂状态θsneg_ [–]）'
        assert callable(dUOCPdθspos) or (dUOCPdθspos is None), '正极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入正极嵌锂状态θspos_ [–]）'
        solve_dUOCPdθsneg_ = P2Dbase.generate_solve_dUOCPdθs_(UOCPneg) if (dUOCPdθsneg is None) else dUOCPdθsneg
        solve_dUOCPdθspos_ = P2Dbase.generate_solve_dUOCPdθs_(UOCPpos) if (dUOCPdθspos is None) else dUOCPdθspos
        self.solve_dUOCPdθsneg_ = partial(P2Dbase.solve_dUOCPdθs_, solve_dUOCPdθsneg_)
        self.solve_dUOCPdθspos_ = partial(P2Dbase.solve_dUOCPdθs_, solve_dUOCPdθspos_)
        self.dUOCPdTneg = dUOCPdTneg; assert callable(dUOCPdTneg) or isinstance(dUOCPdTneg, (int, float)), '负极开路电位的熵热系数dUOCPdTneg [V/K]，标量或函数（输入负极嵌锂状态θsneg_ [–]）'
        self.dUOCPdTpos = dUOCPdTpos; assert callable(dUOCPdTpos) or isinstance(dUOCPdTpos, (int, float)), '正极开路电位的熵热系数dUOCPdTpos [V/K]，标量或函数（输入正极嵌锂状态θspos_ [–]）'
        # 比活化能
        self.EDsneg = EDsneg; assert EDsneg>=0, f'负极固相锂离子扩散系数的比活化能{EDsneg = }，应大于或等于0 [J/mol]'
        self.EDspos = EDspos; assert EDspos>=0, f'正极固相锂离子扩散系数的比活化能{EDspos = }，应大于或等于0 [J/mol]'
        self.Ekneg = Ekneg; assert Ekneg>=0,    f'负极主反应速率常数的比活化能{Ekneg = }，应大于或等于0 [J/mol]'
        self.Ekpos = Ekpos; assert Ekpos>=0,    f'正极主反应速率常数的比活化能{Ekpos = }，应大于或等于0 [J/mol]'
        self.Eκ = Eκ;     assert Eκ>=0,   f'电解液锂离子电导率的比活化能{Eκ = }，应大于或等于0 [J/mol]'
        self.EDe = EDe;   assert EDe>=0,  f'电解液锂离子扩散系数的比活化能{EDe = }，应大于或等于0 [J/mol]'
        self.Tref = Tref; assert Tref>0, f'参考温度{Tref = }，应大于0 [K]'
        # 热参数
        self.Rth = Rth; assert Rth>0, f'传热热阻{Rth = }，应大于0 [K/W]'
        self.Cth = Cth; assert Cth>0, f'电池热容{Cth = }，应大于0 [J/K]'
        # 模式
        self.timeDL = timeDL; assert isinstance(timeDL, bool), f'时域双电层开关{timeDL = }，应为bool类型'
        self.timeDiscretization = timeDiscretization;     assert timeDiscretization in {'backward', 'CN'}, f'时间离散格式{timeDiscretization = }，应为 "CN"（Crank-Nicolson） 或 "backward"（后向差分格式）'
        self.radialDiscretization = radialDiscretization; assert radialDiscretization in {'EV', 'EI'}, f'球形颗粒径向离散方法{radialDiscretization = }，应为 "EV"（等体积）/"EI"（等间隔）'
        self.complete = complete  # 是否确保功能完备
        self.verbose = verbose    # 是否显示初始化、运行进度
        # P2D通用状态量
        self.T = T0    # 温度 [K]
        self.I = 0.    # 电流 [A]
        self.t = 0.    # 时刻 [s]
        self.tEIS: float = None             # 计算阻抗的时刻 [s]
        self.Z_ = empty(Nf, dtype=complex)  # 全电池阻抗谱 [Ω]
        θsneg = θminneg + SOC0*(θmaxneg - θminneg)  # 初始负极嵌锂状态
        θspos = θmaxpos + SOC0*(θminpos - θmaxpos)  # 初始正极嵌锂状态
        self.φsneg_ = full(Nneg, self.solve_UOCPneg_(θsneg))  # 初始化：负极固相电势场 [V]
        self.φspos_ = full(Npos, self.solve_UOCPpos_(θspos))  # 初始化：正极固相电势场 [V]
        self.φe_ = zeros(Ne)         # 初始化：电解液电势场 [V]
        self.ηintneg_ = zeros(Nneg)  # 初始化：负极主反应过电位场 [V]
        self.ηintpos_ = zeros(Npos)  # 初始化：正极主反应过电位场 [V]
        self.nNewton = 0
        if complete:
            self.Zneg_, self.Zpos_ = empty(Nf, dtype=complex), empty(Nf, dtype=complex)  # 负极、正极阻抗谱 [Ω]
            self.REφsneg__, self.IMφsneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))        # 负极固相电势实部、虚部 [V]
            self.REφspos__, self.IMφspos__ = empty((Nf, Npos)), empty((Nf, Npos))        # 正极固相电势实部、虚部 [V]
            self.REφe__,    self.IMφe__    = empty((Nf, Ne)), empty((Nf, Ne))            # 电解液电势实部、虚部 [V]
            self.REηintneg__, self.IMηintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))    # 负极主反应过电位实部、虚部 [V]
            self.REηintpos__, self.IMηintpos__ = empty((Nf, Npos)), empty((Nf, Npos))    # 正极主反应过电位实部、虚部 [V]
        # 恒定量
        self.banded_experience_of_J__ = None   # Jacobi矩阵带状化经验
        self.banded_experience_of_Kf__ = None  # Kf__矩阵带状化经验
        self.ΔIAC = 1.                         # 交流扰动电流振幅 [A]
        self.Δxneg = Δxneg = Lneg/Nneg  # 负极网格厚度
        Δxsep      = Lsep/Nsep          # 隔膜网格厚度
        self.Δxpos = Δxpos = Lpos/Npos  # 正极网格厚度
        self.x_ = concatenate([
            linspace(0, Lneg, Nneg + 1)[:-1] + Δxneg*0.5,
            linspace(Lneg, Lneg + Lsep, Nsep + 1)[:-1] + Δxsep*0.5,
            linspace(Lneg + Lsep, Lneg + Lsep + Lpos, Npos + 1)[:-1] + Δxpos*0.5,])        # (Ne,) 全区域各控制体中心坐标
        self.Δx_ = concatenate([full(Nneg, Δxneg), full(Nsep, Δxsep), full(Npos, Δxpos)])  # (Ne,) 全区域控制体厚度
        self.ΔxWest_ = hstack([
            full(Nneg, Δxneg),
            (Δxneg + Δxsep)*0.5, full(Nsep - 1, Δxsep),
            (Δxsep + Δxpos)*0.5, full(Npos - 1, Δxpos)])  # (Ne,) 当前控制体中心到左侧控制体中心的距离
        self.ΔxEast_ = hstack([
            full(Nneg - 1, Δxneg), (Δxneg + Δxsep)*0.5,
            full(Nsep - 1, Δxsep), (Δxsep + Δxpos)*0.5,
            full(Npos, Δxpos)])  # (Ne,) 当前控制体中心到右侧控制体中心的距离
        (self.rneg_,         # (Nr,) 负极球壳控制体中心坐标 [m]/[–]
         self.Δrneg_,        # (Nr,) 负极球壳控制体厚度 [m]/[–]
         self.bandKcsneg__,  # (3, Nr) 负极固相浓度三对角矩阵的带 [m^-2]/[–]
         self.Vr_,           # 从中心到表面球壳体积分数序列 [–]
         ) = P2Dbase.generate_r_Δr_bandKcs__(Nr, Rsneg, radialDiscretization)
        (self.rpos_,
         self.Δrpos_,
         self.bandKcspos__
         ) = (self.rneg_, self.Δrneg_, self.bandKcsneg__) if Rsneg==Rspos else \
             P2Dbase.generate_r_Δr_bandKcs__(Nr, Rspos, radialDiscretization)[:3]
        rneg_ = self.rneg_
        r_3, r_2, r_1 = rneg_[-3:]
        Δr_3, Δr_2, Δr_1 = Rsneg - rneg_[-3:]
        self.coeffsExpl_ = array([
            Δr_1*Δr_2/((r_3 - r_1)*(r_3 - r_2)),
            Δr_1*Δr_3/((r_2 - r_1)*(r_2 - r_3)),
            Δr_2*Δr_3/((r_1 - r_2)*(r_1 - r_3))
            ])  # (3,) 用于由3个颗粒内部节点浓度外推表面浓度的无量纲系数 [–]，计算结果不依赖正负极，仅取决于Nr和radialDiscretization
        self.e__ = zeros((Nr, 1))  # (Nr, 1) 非零末元为1的向量
        self.e__[-1] = 1
        # 需记录时域、频域状态量的名称
        self.datanames_ = datanames_ = ['t', 'U']       # 时刻 [s]、端电压 [V]
        self.EISdatanames_ = EISdatanames_ = ['tEIS', 'Z_']  # 阻抗计算时刻 [s]、全电池阻抗谱 [Ω]
        self.ΔφsenegHistory__ = deque(maxlen=3)  # 历史最近3时刻负极SEI修正后的固液相电势差场 [V]
        self.ΔφseposHistory__ = deque(maxlen=3)  # 历史最近3时刻正极SEI修正后的固液相电势差场 [V]
        self.Ihistory_ = deque(maxlen=4)  # 历史最近4时刻电流 [A]
        # 缓存@property相关
        self._cached_t = {}           # 缓存时刻（作为版本号）
        self._cached_properties = {}  # 缓存属性值
        # 计算结果、大矩阵缓存及索引
        self.ravelK_: ndarray = None   # (NK*NK,) 时域因变量线性矩阵K__展平视图
        self.bK_: ndarray = None       # (NK,) 常数项向量 K__ @ X_ = bK_
        self.sK = None                 # K__切片索引集
        self.ravelKf_: ndarray = None  # (NKf*NKf,) 频域因变量线性矩阵Kf__展平视图
        self.bKf_: ndarray = None      # (NKf,) 常数项向量 Kf__ @ X_ = bKf_
        self.sKf = None                # Kf__切片索引集
        if complete:
            datanames_.extend([      # 需记录的时域状态量名称
                'I',                        # 电流 [A]
                'φsneg_', 'φspos_', 'φe_',  # 负极、正极固相电势场 [V]、电解液电势场 [V]
                'ηintneg_', 'ηintpos_',     # 负极、正极主反应过电位场 [V]
                'ηLPneg_',                  # 负极析锂反应过电位场 [V]
                'θsneg', 'θspos', 'SOC',    # 负极、正极嵌锂状态、全电池荷电状态 [–]
                'T', 'Qgen',                # 温度 [K]、产热功率 [W]
                'nNewton', ])               # Newton迭代次数
            EISdatanames_.extend([   # 需记录的频域状态量名称
                'Zneg_', 'Zpos_',                # 负极、正极复阻抗 [Ω]
                'REφsneg__', 'IMφsneg__',        # 负极固相电势实部、虚部 [V]
                'REφspos__', 'IMφspos__',        # 正极固相电势实部、虚部 [V]
                'REφe__', 'IMφe__',              # 电解液电势实部、虚部 [V]
                'REηintneg__', 'IMηintneg__',    # 负极主反应过电位实部、虚部 [V]
                'REηintpos__', 'IMηintpos__',])  # 正极主反应过电位实部、虚部 [V]
            # matplotlib作图变量单位
            set_matplotlib()  # 设置
            self.tSign, self.tUnit = r'${\it t}$', 's'            # 时间t符号、单位
            self.xSign, self.xUnit = r'${\it x}$', 'μm'           # 电极厚度方向坐标x符号、单位
            self.rSign, self.rUnit = r'${\it r}$', 'μm'           # 径向坐标r符号、单位
            self.cSign, self.cUnit = r'${\it c}$', 'mol/m$^3$'    # 锂离子浓度c符号、单位
            self.jSign, self.jUnit = r'${\it j}$', 'A/m$^3$'      # 局部体积电流密度j符号、单位
            self.i0Sign, self.i0Unit = r'${\it i}_0$', 'A/m$^2$'  # 交换电流密度i0符号、单位
        self.data = {name: [] for name in datanames_ + EISdatanames_}  # 字典：存储时域和频域运行数据

    def CC(self,
           I: float | int = 0,             # 电流 [A]，放电为正，充电为负
           duration: float | int = 1,      # 充放电运行时间 [s]
           thermalModel: bool = False,     # 是否启用热模型更新温度
           Umax: float | None = None,      # 充电电压上限 [V]
           Umin: float | None = None,      # 放电电压下限 [V]
           SOCmax: float | None = None,    # 充电最大SOC [–]
           SOCmin: float | None = None,    # 放电最小SOC [–]
           minΔt: float = 0.1,             # 最小时间步长 [s]
           tEIS_: Sequence | None = None,  # 计算EIS的时刻序列 [s]
           Ts: Callable | float = 298.15,  # 表面温度Ts [K]，插值函数：表面温度Ts [K]-时间t [s]，或标量
           ):
        # 恒流充放电

        # 若运行时间duration非正，直接返回
        if duration<=0:
            return self

        # 读取方法
        _stepping = self._stepping
        record_data = self.record_data
        EIS = self.EIS

        t = self.t  # 当前时刻 [s]
        tStart = t  # 开始时刻 [s]
        tStop = tStart + duration  # 终止时刻 [s]

        callableTs = callable(Ts)  # 是否给定 表面温度-时间 插值函数

        # 电流可能在同一时刻发生切换，清除旧电流对应的Qgen缓存
        if I!=self.I:
            self.I = I
            self._cached_t.pop('Qgen', None)
            self._cached_properties.pop('Qgen', None)

        # 记录初始时刻t=0数据
        if t==0:
            record_data()

        ΔtDefault = self.Δt  # 默认时间步长 [s]
        assert minΔt<=ΔtDefault, f'最小时间步长 {minΔt = }s，应小于或等于{ΔtDefault = }s'

        if verbose := self.verbose:
            startTime = time.time()  # 开始时间戳 [s]
            info = f"电流{I = :.2f}A{'放电' if I>0 else ('充电' if I<0 else '静置')}"

        # 预处理EIS时刻
        if tEIS_ is None:
            tEIS_ = empty(0)
        else:
            tEIS_ = unique(tEIS_)
            tEIS_ = tEIS_[(tStart<=tEIS_) & (tEIS_<=tStop)]
        iEIS = 0  # 当前待处理的EIS时刻索引
        NtEIS = tEIS_.size  # EIS计算时刻数目
        rtolEIS = 1e-8
        atolEIS = 1e-7

        ΔtThreshold = ΔtDefault + minΔt  # 时间步长阈值 [s]

        # 生成K__矩阵、bK_向量及其切片索引
        if self.ravelK_ is None:
            self._generate_K__bK_and_slices()

        earlyStop = any(v is not None for v in (Umax, Umin, SOCmax, SOCmin))  # 提前终止条件

        while t<tStop:
            ## 持续时间步进...

            # 防止因浮点误差或异常步进导致iEIS停留在过去的EIS时刻
            while iEIS<NtEIS and tEIS_[iEIS]<(t - atolEIS):
                iEIS += 1

            # 如果当前时刻正好是EIS时刻，则先计算EIS
            while iEIS<NtEIS and isclose(t, tEIS_[iEIS], rtol=rtolEIS, atol=atolEIS):
                EIS()
                iEIS += 1

            # 下一个EIS时刻
            if iEIS<NtEIS:
                tNextEIS = tEIS_[iEIS]
            else:
                tNextEIS = inf

            if earlyStop:
                if I<0 and (Umax is not None) and self.U>=Umax:
                    if verbose:
                        print(f'\n电压U达{Umax = }V，停止{info}')
                    break
                if I<0 and (SOCmax is not None) and self.SOC>=SOCmax:
                    if verbose:
                        print(f'\nSOC达{SOCmax = }，停止{info}')
                    break
                if I>0 and (Umin is not None) and self.U<=Umin:
                    if verbose:
                        print(f'\n电压U达{Umin = }V，停止{info}')
                    break
                if I>0 and (SOCmin is not None) and self.SOC<=SOCmin:
                    if verbose:
                        print(f'\nSOC达{SOCmin = }，停止{info}')
                    break


            # 选择时间步长
            remainingTime = tStop - t  # 剩余时长 [s]
            if remainingTime < ΔtThreshold:
                Δt = remainingTime  # 使用剩余时长作为最后时间步长
            else:
                Δt = ΔtDefault  # 使用默认时间步长 [s]

            # 如果步长会跨过下一个EIS时刻，则缩短步长，刚好踩中tNextEIS
            if t < tNextEIS < (t + Δt):
                Δt = tNextEIS - t

            # 更新产热量Qgen [W]、传热温差ΔT [K]
            if thermalModel:
                Qgen = self.Qgen
                if callableTs:
                    ΔT = Ts(t) - self.T
                else:
                    ΔT = Ts - self.T

            while True:
                # 试探步进
                Δt = float(Δt)
                nNewton, success, message = _stepping(Δt)
                if success:
                    break  # 步进成功，无报错，跳出

                # 步进失败，缩小Δt
                if Δt<=minΔt:
                    raise P2Dbase.Error(
                        f'异常：时刻{t = }s，时间步长{Δt = }s，第{nNewton}次Newton迭代出现{message}')
                ΔtNew = max(minΔt, Δt*0.5)
                if verbose:
                    print(f'时刻{t = }s，时间步长{Δt = }s，第{nNewton}次Newton迭代出现{message}，缩小Δt -> {ΔtNew}s')
                Δt = ΔtNew

            # 更新温度T [K]
            if thermalModel:
                dTdt = (Qgen + ΔT/self.Rth)/self.Cth
                if abs(dTdt)>5 or (not isfinite(dTdt)):
                    raise P2Dbase.Error(f'异常：时刻{t = }s，时间步长{Δt = }s，温度变化率{dTdt = } K/s，超出合理范围')
                self.T += dTdt*Δt

            # 更新时刻t [s]
            t += Δt
            self.t = t
            self.nNewton = nNewton  # 更新：Newton迭代次数

            record_data()  # 记录运行数据

            # 到达EIS时刻后计算阻抗
            while iEIS<NtEIS and isclose(t, tEIS_[iEIS], rtol=rtolEIS, atol=atolEIS):
                EIS()
                iEIS += 1

            if verbose:
                # 显示进度
                finishedProportion = (t - tStart)/duration  # 已完成的比例
                finishedProgresses = int(25*finishedProportion)  # 已完成的进度条长度
                unfinishedProgresses = 25 - finishedProgresses   # 未完成的进度条长度
                finishedBar = '▓'*finishedProgresses        # 已完成的进度条
                unfinishedBar = '-'*unfinishedProgresses    # 未完成的进度条
                percentage = finishedProportion*100         # 已完成进度的百分比
                timeStamp = time.time() - startTime         # 累计耗时
                U = self.U
                SOC = self.SOC
                print(f'|{finishedBar}{unfinishedBar}|已完成{percentage:.0f}%，耗时{timeStamp:.1f}s，'
                      f'{t = :g}s-->{tStop:g}s，{info}，'
                      f'电压{U = :.3f}V, {SOC = :.3f}，温度{self.T - 273.15:.1f}°C，'
                      f'负极-隔膜界面析锂反应过电位{self.ηLPneg_[-1]*1000:.0f}mV，'
                      f'{nNewton}次Newton迭代'
                      f'\r', end='')
        else:
            if verbose:
                print(f'\n达到运行时长{duration}s，停止{info}')

        return self

    def _generate_K__bK_and_slices(self):
        """生成时域因变量矩阵K__、常数项向量bK_及切片索引，并对K__赋常系数"""
        # 读取网格数
        Nneg, Npos, Ne = self.Nneg, self.Npos, self.Ne
        NK = 0  # 全局索引游标：K__.shape = (NK, NK)
        def allocate(n: int) -> slice:
            # 分配切片
            nonlocal NK
            s = slice(NK, NK + n)
            NK += n
            return s

        # K__矩阵行的切片索引；时域Newton迭代固相表面浓度、电解液浓度、固相电势、电解液电势
        slices_ = {
            's_csnegsurf': (s_csnegsurf := allocate(Nneg)),  # 索引：负极固相表面浓度
            's_cspossurf': (s_cspossurf := allocate(Npos)),  # 索引：正极固相表面浓度
            's_ce':    (s_ce :=  allocate(Ne)),      # 索引：电解液浓度
            's_φsneg': (s_φsneg := allocate(Nneg)),  # 索引：负极固相电势
            's_φspos': (s_φspos := allocate(Npos)),  # 索引：正极固相电势
            's_φe':    (s_φe := allocate(Ne)),       # 索引：电解液电势

            's_ceneg': (s_ceneg := slice(start := s_ce.start, start + Nneg)),  # 索引：负极电解液浓度
            's_cepos': (s_cepos := slice((stop := s_ce.stop) - Npos, stop)),   # 索引：正极电解液浓度
            's_φeneg': (s_φeneg := slice(start := s_φe.start, start + Nneg)),  # 索引：负极电解液电势
            's_φepos': (s_φepos := slice((stop := s_φe.stop) - Npos, stop)),   # 索引：正极电解液电势
            's_c': slice(s_csnegsurf.start, s_ce.stop),  # 索引：所有浓度量
            's_φ': slice(s_φsneg.start, s_φe.stop),      # 索引：所有电势量
            }

        # K__矩阵子块对角元对应K__.ravel()的切片索引
        dsr = partial(diagonalSliceRavel, NK)
        slices_.update({
            'sr_csnegsurf_csnegsurf': dsr(s_csnegsurf, s_csnegsurf),
            'sr_csnegsurf_ceneg':     dsr(s_csnegsurf, s_ceneg),
            'sr_csnegsurf_φsneg':     dsr(s_csnegsurf, s_φsneg),
            'sr_csnegsurf_φeneg':     dsr(s_csnegsurf, s_φeneg),
            'sr_cspossurf_cspossurf': dsr(s_cspossurf, s_cspossurf),
            'sr_cspossurf_cepos':     dsr(s_cspossurf, s_cepos),
            'sr_cspossurf_φspos':     dsr(s_cspossurf, s_φspos),
            'sr_cspossurf_φepos':     dsr(s_cspossurf, s_φepos),

            'sr_ceneg_csnegsurf': dsr(s_ceneg, s_csnegsurf),
            'sr_cepos_cspossurf': dsr(s_cepos, s_cspossurf),
            'sr_ce_ce'   : dsr(s_ce, s_ce),
            'sr_ce_ce_l' : dsr(s_ce, s_ce, -1),
            'sr_ce_ce_u' : dsr(s_ce, s_ce,  1),
            'sr_ceneg_φsneg': dsr(s_ceneg, s_φsneg),
            'sr_cepos_φspos': dsr(s_cepos, s_φspos),
            'sr_ceneg_φeneg': dsr(s_ceneg, s_φeneg),
            'sr_cepos_φepos': dsr(s_cepos, s_φepos),

            'sr_φsneg_csnegsurf': dsr(s_φsneg, s_csnegsurf),
            'sr_φspos_cspossurf': dsr(s_φspos, s_cspossurf),
            'sr_φsneg_φsneg': dsr(s_φsneg, s_φsneg),
            'sr_φspos_φspos': dsr(s_φspos, s_φspos),
            'sr_φsneg_φeneg': dsr(s_φsneg, s_φeneg),
            'sr_φspos_φepos': dsr(s_φspos, s_φepos),

            'sr_φeneg_csnegsurf': dsr(s_φeneg, s_csnegsurf),
            'sr_φepos_cspossurf': dsr(s_φepos, s_cspossurf),
            'sr_φe_ce'   : dsr(s_φe, s_ce),
            'sr_φe_ce_l' : dsr(s_φe, s_ce, -1),
            'sr_φe_ce_u' : dsr(s_φe, s_ce,  1),
            'sr_φeneg_φsneg': dsr(s_φeneg, s_φsneg),
            'sr_φepos_φspos': dsr(s_φepos, s_φspos),
            'sr_φe_φe'   : dsr(s_φe, s_φe),
            'sr_φe_φe_l' : dsr(s_φe, s_φe, -1),
            'sr_φe_φe_u' : dsr(s_φe, s_φe,  1),
            'sr_φeneg_φeneg': dsr(s_φeneg, s_φeneg),
            'sr_φepos_φepos': dsr(s_φepos, s_φepos),
        })
        self.sK = SlicesK(**slices_)
        del slices_
        self.ravelK_ = ravelK_ = eye(NK).ravel()  # (NK*NK,) 因变量线性矩阵K__展平视图
        self.bK_ = zeros(NK)                      # (NK,) 常数项向量 K__ @ X_ = bK_

        # 对K__矩阵赋恒定值
        # 负极、正极固相电势φs行φs列
        for s_φs in (s_φsneg, s_φspos):
            ravelK_[dsr(s_φs, s_φs, -1)] = \
            ravelK_[dsr(s_φs, s_φs,  1)] = 1  # 上下对角线

        if self.verbose:
            print(f'时域因变量线性矩阵 {ravelK_.base.shape = }')

    def _generate_Kf__bKf_and_slices(self):
        """生成频域因变量矩阵Kf__、常数项向量bKf_及切片索引，并对Kf__赋常系数"""
        # 读取网格数
        Nneg, Npos, Ne = self.Nneg, self.Npos, self.Ne

        NKf = 0  # 全局索引游标
        def allocate(n: int) -> slice:
            # 分配切片
            nonlocal NKf
            s = slice(NKf, NKf + n)
            NKf += n
            return s

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

            's_REceneg': (s_REceneg := slice(start := s_REce.start, start + Nneg)),  # 索引：负极电解液浓度实部
            's_IMceneg': (s_IMceneg := slice(start := s_IMce.start, start + Nneg)),  # 索引：负极电解液浓度虚部
            's_REcepos': (s_REcepos := slice((stop := s_REce.stop) - Npos, stop)),   # 索引：正极电解液浓度实部
            's_IMcepos': (s_IMcepos := slice((stop := s_IMce.stop) - Npos, stop)),   # 索引：正极电解液浓度虚部
            's_REφeneg': (s_REφeneg := slice(start := s_REφe.start, start + Nneg)),  # 索引：负极电解液电势实部
            's_IMφeneg': (s_IMφeneg := slice(start := s_IMφe.start, start + Nneg)),  # 索引：负极电解液电势虚部
            's_REφepos': (s_REφepos := slice((stop := s_REφe.stop) - Npos, stop)),   # 索引：正极电解液电势实部
            's_IMφepos': (s_IMφepos := slice((stop := s_IMφe.stop) - Npos, stop)),   # 索引：正极电解液电势虚部
            }
        # Kf__矩阵子块对角元对应Kf__.ravel()的索引
        dsr = partial(diagonalSliceRavel, NKf)
        slices_.update({
            'sr_REcsnegsurf_REcsnegsurf': dsr(s_REcsnegsurf, s_REcsnegsurf),
            'sr_REcsnegsurf_IMcsnegsurf': dsr(s_REcsnegsurf, s_IMcsnegsurf),
            'sr_IMcsnegsurf_REcsnegsurf': dsr(s_IMcsnegsurf, s_REcsnegsurf),
            'sr_IMcsnegsurf_IMcsnegsurf': dsr(s_IMcsnegsurf, s_IMcsnegsurf),
            'sr_REcsnegsurf_REceneg': dsr(s_REcsnegsurf, s_REceneg),
            'sr_IMcsnegsurf_IMceneg': dsr(s_IMcsnegsurf, s_IMceneg),
            'sr_REcsnegsurf_REφsneg': dsr(s_REcsnegsurf, s_REφsneg),
            'sr_REcsnegsurf_IMφsneg': dsr(s_REcsnegsurf, s_IMφsneg),
            'sr_IMcsnegsurf_REφsneg': dsr(s_IMcsnegsurf, s_REφsneg),
            'sr_IMcsnegsurf_IMφsneg': dsr(s_IMcsnegsurf, s_IMφsneg),
            'sr_REcsnegsurf_REφeneg': dsr(s_REcsnegsurf, s_REφeneg),
            'sr_REcsnegsurf_IMφeneg': dsr(s_REcsnegsurf, s_IMφeneg),
            'sr_IMcsnegsurf_REφeneg': dsr(s_IMcsnegsurf, s_REφeneg),
            'sr_IMcsnegsurf_IMφeneg': dsr(s_IMcsnegsurf, s_IMφeneg),

            'sr_REcspossurf_REcspossurf': dsr(s_REcspossurf, s_REcspossurf),
            'sr_REcspossurf_IMcspossurf': dsr(s_REcspossurf, s_IMcspossurf),
            'sr_IMcspossurf_REcspossurf': dsr(s_IMcspossurf, s_REcspossurf),
            'sr_IMcspossurf_IMcspossurf': dsr(s_IMcspossurf, s_IMcspossurf),
            'sr_REcspossurf_REcepos': dsr(s_REcspossurf, s_REcepos),
            'sr_IMcspossurf_IMcepos': dsr(s_IMcspossurf, s_IMcepos),
            'sr_REcspossurf_REφspos': dsr(s_REcspossurf, s_REφspos),
            'sr_REcspossurf_IMφspos': dsr(s_REcspossurf, s_IMφspos),
            'sr_IMcspossurf_REφspos': dsr(s_IMcspossurf, s_REφspos),
            'sr_IMcspossurf_IMφspos': dsr(s_IMcspossurf, s_IMφspos),
            'sr_REcspossurf_REφepos': dsr(s_REcspossurf, s_REφepos),
            'sr_REcspossurf_IMφepos': dsr(s_REcspossurf, s_IMφepos),
            'sr_IMcspossurf_REφepos': dsr(s_IMcspossurf, s_REφepos),
            'sr_IMcspossurf_IMφepos': dsr(s_IMcspossurf, s_IMφepos),

            'sr_REceneg_REcsnegsurf': dsr(s_REceneg, s_REcsnegsurf),
            'sr_REceneg_IMcsnegsurf': dsr(s_REceneg, s_IMcsnegsurf),
            'sr_IMceneg_REcsnegsurf': dsr(s_IMceneg, s_REcsnegsurf),
            'sr_IMceneg_IMcsnegsurf': dsr(s_IMceneg, s_IMcsnegsurf),
            'sr_REcepos_REcspossurf': dsr(s_REcepos, s_REcspossurf),
            'sr_REcepos_IMcspossurf': dsr(s_REcepos, s_IMcspossurf),
            'sr_IMcepos_REcspossurf': dsr(s_IMcepos, s_REcspossurf),
            'sr_IMcepos_IMcspossurf': dsr(s_IMcepos, s_IMcspossurf),
            'sr_REce_REce':   dsr(s_REce, s_REce),
            'sr_REce_REce_l': dsr(s_REce, s_REce, -1),
            'sr_REce_REce_u': dsr(s_REce, s_REce,  1),
            'sr_REce_IMce':   dsr(s_REce, s_IMce),
            'sr_IMce_REce':   dsr(s_IMce, s_REce),
            'sr_IMce_IMce':   dsr(s_IMce, s_IMce),
            'sr_IMce_IMce_l': dsr(s_IMce, s_IMce, -1),
            'sr_IMce_IMce_u': dsr(s_IMce, s_IMce,  1),
            'sr_REceneg_REφsneg': dsr(s_REceneg, s_REφsneg),
            'sr_REceneg_IMφsneg': dsr(s_REceneg, s_IMφsneg),
            'sr_IMceneg_REφsneg': dsr(s_IMceneg, s_REφsneg),
            'sr_IMceneg_IMφsneg': dsr(s_IMceneg, s_IMφsneg),
            'sr_REcepos_REφspos': dsr(s_REcepos, s_REφspos),
            'sr_REcepos_IMφspos': dsr(s_REcepos, s_IMφspos),
            'sr_IMcepos_REφspos': dsr(s_IMcepos, s_REφspos),
            'sr_IMcepos_IMφspos': dsr(s_IMcepos, s_IMφspos),
            'sr_REceneg_REφeneg': dsr(s_REceneg, s_REφeneg),
            'sr_REceneg_IMφeneg': dsr(s_REceneg, s_IMφeneg),
            'sr_IMceneg_REφeneg': dsr(s_IMceneg, s_REφeneg),
            'sr_IMceneg_IMφeneg': dsr(s_IMceneg, s_IMφeneg),
            'sr_REcepos_REφepos': dsr(s_REcepos, s_REφepos),
            'sr_REcepos_IMφepos': dsr(s_REcepos, s_IMφepos),
            'sr_IMcepos_REφepos': dsr(s_IMcepos, s_REφepos),
            'sr_IMcepos_IMφepos': dsr(s_IMcepos, s_IMφepos),

            'sr_REφsneg_REcsnegsurf': dsr(s_REφsneg, s_REcsnegsurf),
            'sr_REφsneg_IMcsnegsurf': dsr(s_REφsneg, s_IMcsnegsurf),
            'sr_IMφsneg_REcsnegsurf': dsr(s_IMφsneg, s_REcsnegsurf),
            'sr_IMφsneg_IMcsnegsurf': dsr(s_IMφsneg, s_IMcsnegsurf),
            'sr_REφspos_REcspossurf': dsr(s_REφspos, s_REcspossurf),
            'sr_REφspos_IMcspossurf': dsr(s_REφspos, s_IMcspossurf),
            'sr_IMφspos_REcspossurf': dsr(s_IMφspos, s_REcspossurf),
            'sr_IMφspos_IMcspossurf': dsr(s_IMφspos, s_IMcspossurf),
            'sr_REφsneg_REceneg': dsr(s_REφsneg, s_REceneg),
            'sr_REφsneg_IMceneg': dsr(s_REφsneg, s_IMceneg),
            'sr_IMφsneg_REceneg': dsr(s_IMφsneg, s_REceneg),
            'sr_IMφsneg_IMceneg': dsr(s_IMφsneg, s_IMceneg),
            'sr_REφsneg_REφsneg': dsr(s_REφsneg, s_REφsneg),
            'sr_REφsneg_REφsneg_l': dsr(s_REφsneg, s_REφsneg, -1),
            'sr_REφsneg_REφsneg_u': dsr(s_REφsneg, s_REφsneg, 1),
            'sr_REφsneg_IMφsneg': dsr(s_REφsneg, s_IMφsneg),
            'sr_IMφsneg_REφsneg': dsr(s_IMφsneg, s_REφsneg),
            'sr_IMφsneg_IMφsneg': dsr(s_IMφsneg, s_IMφsneg),
            'sr_IMφsneg_IMφsneg_l': dsr(s_IMφsneg, s_IMφsneg, -1),
            'sr_IMφsneg_IMφsneg_u': dsr(s_IMφsneg, s_IMφsneg,  1),
            'sr_REφspos_REφspos': dsr(s_REφspos, s_REφspos),
            'sr_REφspos_REφspos_l': dsr(s_REφspos, s_REφspos, -1),
            'sr_REφspos_REφspos_u': dsr(s_REφspos, s_REφspos, 1),
            'sr_REφspos_IMφspos': dsr(s_REφspos, s_IMφspos),
            'sr_IMφspos_REφspos': dsr(s_IMφspos, s_REφspos),
            'sr_IMφspos_IMφspos': dsr(s_IMφspos, s_IMφspos),
            'sr_IMφspos_IMφspos_l': dsr(s_IMφspos, s_IMφspos, -1),
            'sr_IMφspos_IMφspos_u': dsr(s_IMφspos, s_IMφspos, 1),
            'sr_REφsneg_REφeneg': dsr(s_REφsneg, s_REφeneg),
            'sr_REφsneg_IMφeneg': dsr(s_REφsneg, s_IMφeneg),
            'sr_IMφsneg_REφeneg': dsr(s_IMφsneg, s_REφeneg),
            'sr_IMφsneg_IMφeneg': dsr(s_IMφsneg, s_IMφeneg),
            'sr_REφspos_REφepos': dsr(s_REφspos, s_REφepos),
            'sr_REφspos_IMφepos': dsr(s_REφspos, s_IMφepos),
            'sr_IMφspos_REφepos': dsr(s_IMφspos, s_REφepos),
            'sr_IMφspos_IMφepos': dsr(s_IMφspos, s_IMφepos),

            'sr_REφeneg_REcsnegsurf': dsr(s_REφeneg, s_REcsnegsurf),
            'sr_REφeneg_IMcsnegsurf': dsr(s_REφeneg, s_IMcsnegsurf),
            'sr_IMφeneg_REcsnegsurf': dsr(s_IMφeneg, s_REcsnegsurf),
            'sr_IMφeneg_IMcsnegsurf': dsr(s_IMφeneg, s_IMcsnegsurf),
            'sr_REφepos_REcspossurf': dsr(s_REφepos, s_REcspossurf),
            'sr_REφepos_IMcspossurf': dsr(s_REφepos, s_IMcspossurf),
            'sr_IMφepos_REcspossurf': dsr(s_IMφepos, s_REcspossurf),
            'sr_IMφepos_IMcspossurf': dsr(s_IMφepos, s_IMcspossurf),
            'sr_REφe_REce':   dsr(s_REφe, s_REce),
            'sr_REφe_REce_l': dsr(s_REφe, s_REce, -1),
            'sr_REφe_REce_u': dsr(s_REφe, s_REce, 1),
            'sr_IMφe_IMce':   dsr(s_IMφe, s_IMce),
            'sr_IMφe_IMce_l': dsr(s_IMφe, s_IMce, -1),
            'sr_IMφe_IMce_u': dsr(s_IMφe, s_IMce, 1),
            'sr_REφeneg_REφsneg': dsr(s_REφeneg, s_REφsneg),
            'sr_REφeneg_IMφsneg': dsr(s_REφeneg, s_IMφsneg),
            'sr_IMφeneg_REφsneg': dsr(s_IMφeneg, s_REφsneg),
            'sr_IMφeneg_IMφsneg': dsr(s_IMφeneg, s_IMφsneg),
            'sr_REφepos_REφspos': dsr(s_REφepos, s_REφspos),
            'sr_REφepos_IMφspos': dsr(s_REφepos, s_IMφspos),
            'sr_IMφepos_REφspos': dsr(s_IMφepos, s_REφspos),
            'sr_IMφepos_IMφspos': dsr(s_IMφepos, s_IMφspos),
            'sr_REφe_REφe':   dsr(s_REφe, s_REφe),
            'sr_REφe_REφe_l': dsr(s_REφe, s_REφe, -1),
            'sr_REφe_REφe_u': dsr(s_REφe, s_REφe, 1),
            'sr_REφe_IMφe': dsr(s_REφe, s_IMφe),
            'sr_IMφe_REφe': dsr(s_IMφe, s_REφe),
            'sr_IMφe_IMφe': dsr(s_IMφe, s_IMφe),
            'sr_IMφe_IMφe_l': dsr(s_IMφe, s_IMφe, -1),
            'sr_IMφe_IMφe_u': dsr(s_IMφe, s_IMφe, 1),
            })

        self.sKf = SlicesKf(**slices_)
        del slices_
        self.ravelKf_ = ravelKf_ = eye(NKf).ravel()  # (NKf*NKf,) 频域因变量线性矩阵Kf__展平视图
        self.bKf_ = zeros(NKf)  # Kf__ @ X_ = bKf_

        # 对Kf__矩阵赋恒定值

        # REφs行REφs列、IMφs行IMφs列
        for s_REφs, s_IMφs in zip((s_REφsneg, s_REφspos),(s_IMφsneg, s_IMφspos)):
            ravelKf_[dsr(s_REφs, s_REφs, -1)] = \
            ravelKf_[dsr(s_REφs, s_REφs,  1)] = \
            ravelKf_[dsr(s_IMφs, s_IMφs, -1)] = \
            ravelKf_[dsr(s_IMφs, s_IMφs,  1)] = 1  # 上下对角线

        if self.verbose:
            print(f'频域因变量线性矩阵 {ravelKf_.base.shape = }')

    @property
    def kneg(self):
        """负极主反应速率常数"""
        return self.Arrhenius(self._kneg, self.Ekneg)
    @kneg.setter
    def kneg(self, kneg):
        self._kneg = kneg

    @property
    def kpos(self):
        """正极主反应速率常数"""
        return self.Arrhenius(self._kpos, self.Ekpos,)
    @kpos.setter
    def kpos(self, kpos):
        self._kpos = kpos

    @property
    def Dsneg(self):
        """负极固相扩散系数"""
        return self.Arrhenius(self._Dsneg, self.EDsneg)
    @Dsneg.setter
    def Dsneg(self, Dsneg):
        self._Dsneg = Dsneg

    @property
    def Dspos(self):
        """正极固相扩散系数"""
        return self.Arrhenius(self._Dspos, self.EDspos)
    @Dspos.setter
    def Dspos(self, Dspos):
        self._Dspos = Dspos

    @property
    def xneg_(self):
        """负极区域控制体中心坐标"""
        return self.x_[:self.Nneg]

    @property
    def xpos_(self):
        """正极区域控制体中心坐标"""
        return self.x_[-self.Npos:]

    @property
    def xInterfaces_(self):
        """(Ne+1,) 各控制体界面的坐标（包括负极-集流体界面、正极-集流体界面）"""
        return hstack([0, self.Δx_.cumsum()])

    @property
    def Ul(self) -> float:
        """等效电感l两端感应电压 [V]"""
        l = self.l
        if l==0:
            return 0
        data = self.data
        data_t_ = data['t']
        Nt = len(data_t_)  # 存储数据时刻数
        if Nt<=1:
            return 0
        Ihistory_ = self.Ihistory_
        t_1 = data_t_[-1]    # 当前时刻 [s]
        t_2 = data_t_[-2]    # 上时刻 [s]
        I_1 = Ihistory_[-1]  # 当前电流 [A]
        I_2 = Ihistory_[-2]  # 上时刻电流 [A]
        Δt = t_1 - t_2       # 时间步长 [s]
        A = 1/Δt
        if Nt>2:
            t_3 = data_t_[-3]  # 上上时刻
            I_3 = Ihistory_[-3]  # 上上时刻电流 [A]
            A += 1/(t_1 - t_3)
        if Nt>3:
            t_4 = data_t_[-4]  # 上上上时刻
            I_4 = Ihistory_[-4]  # 上上上时刻电流 [A]
            A += 1/(t_1 - t_4)
        if Nt>3:  # 4点向后差分
            B = (t_1 - t_3)*(t_1 - t_4)/-Δt/(t_2 - t_3)/(t_2 - t_4)  # [1/s]
            C = Δt*(t_1 - t_4)/(t_3 - t_1)/(t_3 - t_2)/(t_3 - t_4)
            D = Δt*(t_1 - t_3)/(t_4 - t_1)/(t_4 - t_2)/(t_4 - t_3)
            dIdt = A*I_1 +  B*I_2 +  C*I_3 + D*I_4
        elif Nt==3:  # 3点向后差分
            B = (t_1 - t_3)/(Δt*(t_3 - t_2))
            C = Δt/((t_3 - t_1)*(t_3 - t_2))
            dIdt = A*I_1 + B*I_2 + C*I_3
        elif Nt==2:    # 2点向后差分
            dIdt = A*(I_1 - I_2)
        return dIdt * l

    @property
    def OCV(self) -> float:
        """开路电压 [V]"""
        OCPpos = self.solve_UOCPpos_(self.θspos)
        OCPneg = self.solve_UOCPneg_(self.θsneg)
        return OCPpos - OCPneg

    @property
    def SOC(self) -> float:
        """全电池荷电状态 [–]"""
        return (self.θsneg - self.θminneg)/(self.θmaxneg - self.θminneg)

    @property
    def φeneg_(self) -> ndarray:
        """(Nneg,) 负极区域电解液电势 [V]"""
        return self.φe_[:self.Nneg]

    @property
    def φepos_(self) -> ndarray:
        """(Npos,) 正极区域电解液电势 [V]"""
        return self.φe_[-self.Npos:]

    @property
    def Zl_(self) -> ndarray:
        """等效电感复阻抗 [Ω]"""
        return 1j*self.ωl_

    @property
    def Zsep_(self) -> ndarray:
        """隔膜复阻抗 [Ω]"""
        return self.Z_ - self.Zneg_ - self.Zpos_ - self.Zl_   # 隔膜复阻抗 [Ω]

    @property
    def ωl_(self) -> ndarray:
        """电感感抗序列 [Ω]"""
        return self.ω_*self.l

    Qe: float
    Qohmneg: float; Qohmpos: float
    Qrxnneg: float; Qrxnpos: float
    QSEIneg: float; QSEIpos: float
    Qrevneg: float; Qrevpos: float
    @stepping_aware_cached_property
    def Qgen(self) -> float:
        """总产热功率 [W]"""
        return (  self.Qe + self.Qohmneg + self.Qohmpos  # 总欧姆热功率 [W]
                + self.Qrxnneg + self.Qrxnpos   # 总反应热功率 [W]
                + self.QSEIneg + self.QSEIpos   # 总SEI膜电阻热功率 [W]
                + self.Qrevneg + self.Qrevpos)  # 总可逆热功率 [W]

    def solve_UOCPneg_(self, θsneg_: float | ndarray) -> ndarray:
        """求解负极开路电位场 [V]"""
        θsneg_ = maximum(minimum(θsneg_, 1 - εOCP), εOCP)
        UOCPneg_ = self.UOCPneg(θsneg_)
        if ΔT := (self.T - self.Tref):
            if callable(self.dUOCPdTneg):
                UOCPneg_ += ΔT*self.dUOCPdTneg(θsneg_)
            else:
                UOCPneg_ += ΔT*self.dUOCPdTneg
        return UOCPneg_

    def solve_UOCPpos_(self, θspos_: float | ndarray) -> ndarray:
        """求解正极开路电位场 [V]"""
        θspos_ = maximum(minimum(θspos_, 1 - εOCP), εOCP)
        UOCPpos_ = self.UOCPpos(θspos_)
        if ΔT := (self.T - self.Tref):
            if callable(self.dUOCPdTpos):
                UOCPpos_ += ΔT*self.dUOCPdTpos(θspos_)
            else:
                UOCPpos_ += ΔT*self.dUOCPdTpos
        return UOCPpos_

    @staticmethod
    def solve_dUOCPdθs_(solve_dUOCPdθs_: Callable,
                        θs_: float | ndarray) -> ndarray:
        """使用安全嵌锂状态计算OCP导数，并保持与截断后OCP表达式一致。"""
        θs_ = maximum(minimum(θs_, 1 - εOCP), εOCP)
        return solve_dUOCPdθs_(θs_) * ((θs_ > εOCP) & (θs_ < 1 - εOCP))

    @staticmethod
    def generate_r_Δr_bandKcs__(Nr, Rs, radialDiscretization='EV'):
        """生成颗粒径向r相关坐标、(Nr, Nr)矩阵Kcs__的带bandKcs__"""
        Rs3 = 1 if Rs==1 else (Rs*Rs*Rs)
        if radialDiscretization=='EV':
            # 等体积划分球壳网格
            a = 4.1887902047863905  # 4π/3
            V = a*Rs3  # 颗粒体积 [m^3]/[–]
            ΔV = V/Nr  # 球壳控制体体积 [m^3]/[–]
            rO_ = cbrt(ΔV*arange(1, Nr + 1)/a)  # (Nr,) 球壳外界面坐标 [m]/[–]
            rI_ = hstack([0, rO_[:-1]])         # (Nr,) 球壳内界面坐标 [m]/[–]
            r_ = (rI_ + rO_)*0.5  # (Nr,) 球壳控制体中心坐标 [m]/[–]
            Δr_ = rO_ - rI_       # (Nr,) 球壳网格厚度序列 [m]/[–]
            Vr_ = full(Nr, 1/Nr)  # (Nr,) 球壳体积分数序列 [–]
        elif radialDiscretization=='EI':
            # 等间隔划分球壳网格
            Δr_ = full(Nr, Δr := Rs/Nr) # (Nr,) 球壳网格厚度序列 [m]/[–]
            hΔr = Δr*0.5
            r_ = arange(Nr)*Δr + hΔr    # (Nr,) 球壳控制体中心坐标 [m]/[–]
            rO_ = r_ + hΔr
            rI_ = r_ - hΔr
            Vr_ = (rO_*rO_*rO_ - rI_*rI_*rI_)/Rs3  # (Nr,) 球壳体积分数序列 [–]

        Nr1 = Nr + 1
        dr_ = r_[1:] - r_[:-1]  # (Nr-1,)
        rO2_ = rO_*rO_          # (Nr,)
        rI2_ = rI_*rI_          # (Nr,)
        Kcs__ = zeros((Nr, Nr))
        Kcs__ravel_ = Kcs__.ravel()
        Kcs__ravel_[Nr::Nr1] = a_ = -rI2_[1:] /dr_  # (Nr-1,) 下对角线
        Kcs__ravel_[1::Nr1]  = c_ = -rO2_[:-1]/dr_  # (Nr-1,) 上对角线
        Kcs__ravel_[::Nr1] = -(hstack([0., a_]) + hstack([c_, 0.]))  # (Nr, ) 主对角线
        Kcs__ /= ((rO2_*rO_ - rI2_*rI_)/3).reshape(-1, 1)
        bandKcs__ = zeros((3, Nr))
        bandKcs__[0, 1:]  = Kcs__ravel_[1::Nr1]   # (Nr-1,) 上对角线
        bandKcs__[1, :]   = Kcs__ravel_[::Nr1]    # (Nr,) 主对角线
        bandKcs__[2, :-1] = Kcs__ravel_[Nr::Nr1]  # (Nr-1,) 下对角线

        return r_, Δr_, bandKcs__, Vr_

    @staticmethod
    def solve_banded_matrix(
            A__: ndarray,                  # (N, N) 矩阵
            b_: ndarray,                   # (N,) A__ @ X_ = b_
            idxReorder_: Sequence[int],    # (N,) 索引：重排A__，使带状化
            idxRecover_: Sequence[int],    # (N,) 索引：恢复解向量的原始顺序
            l: int,  # 下带宽
            u: int,  # 上带宽
            idxBandFrom__: tuple[ndarray, ...],  # 各带对应A__.ravel()的索引
            idxBandStart_: ndarray,  # (l + u + 1,) 各带在band__中的起始列
            idxBandStop_: ndarray,   # (l + u + 1,) 各带在band__中的终止列
            ab__: ndarray,           # (2*l + u + 1, N) dgbsv工作矩阵
            bReorder_: ndarray,      # 重排后的右端向量工作区
            ) -> ndarray:
        """解带状化矩阵"""
        # 直接从原矩阵A__.ravel()读取各带元素，避免重排整个dense矩阵
        P2Dbase._pack_banded_matrix_JIT(
            A__, b_, ab__, bReorder_,
            idxBandFrom__, idxBandStart_, idxBandStop_, idxReorder_,
            l)
        X_ = dgbsv(l, u, ab__, bReorder_, True, True)[2]
        return X_[idxRecover_]

    @staticmethod
    @njit(cache=True)
    def _pack_banded_matrix_JIT(
            A__, b_, ab__, bReorder_,
            idxBandFrom__, idxBandStart_, idxBandStop_, idxReorder_,
            l):
        # 直接从原矩阵A__.ravel()读取各带元素，并重排右端向量
        ab__[:] = 0.
        band__ = ab__[l:, :]
        ravelA_ = A__.ravel()
        for row in range(idxBandStart_.size):
            start = idxBandStart_[row]
            stop = idxBandStop_[row]
            band__[row, start:stop] = ravelA_[idxBandFrom__[row]]

        bReorder_[:] = b_[idxReorder_]

    @staticmethod
    def prepare_banded_experience(
            A__: ndarray,          # (N, N) 稠密矩阵
            idxReorder_: ndarray,  # (N,) 带状化重排索引
            ) -> dict:
        # 已知重排索引idxReorder_，计算带宽，并预生成直接填充dgbsv工作矩阵所需的索引
        N = A__.shape[0]
        Areorder__ = A__[ix_(idxReorder_, idxReorder_)]
        idxRecover_ = idxReorder_.argsort()  # 索引：恢复重排后解向量的原始顺序

        # 辨识Areorder__的上下带宽
        diag = Areorder__.diagonal
        for offset in range(-N + 1, 0):
            # 从矩阵左下角开始遍历对角线
            if any(diag(offset)!=0):
                l = abs(offset)  # 下带宽
                break
        for offset in range(N - 1, 0, -1):
            # 从矩阵右上角开始遍历对角线
            if any(diag(offset)!=0):
                u = offset       # 上带宽
                break

        lu1 = l + u + 1
        idxBandFrom__ = []
        idxBandStart_ = empty(lu1, int)
        idxBandStop_ = empty(lu1, int)
        for row, offset in enumerate(range(u, -l - 1, -1)):
            start = max(0, offset)
            stop = min(N, N + offset)
            cols_ = arange(start, stop)
            rows_ = cols_ - offset
            idxBandFrom__.append(idxReorder_[rows_]*N + idxReorder_[cols_])
            idxBandStart_[row] = start
            idxBandStop_[row] = stop

        ab__ = zeros((2*l + u + 1, N), float, order='F')
        bReorder_ = empty(N, float, order='F')
        return {
            'idxReorder_': idxReorder_,
            'idxRecover_': idxRecover_,
            'l': l,
            'u': u,
            'idxBandFrom__': tuple(idxBandFrom__),
            'idxBandStart_': idxBandStart_,
            'idxBandStop_': idxBandStop_,
            'ab__': ab__,
            'bReorder_': bReorder_,}

    @staticmethod
    def banded_by_RCM(A__: ndarray) -> dict:
        # RCM算法创造带状化经验
        idxReorder_ = reverse_cuthill_mckee(csr_matrix((A__!=0) | (A__.T!=0)))  # 索引：重排A__，使之带状化
        return P2Dbase.prepare_banded_experience(A__, idxReorder_)

    def banded_J_by_space(self, J__: ndarray) -> dict:
        # 按空间节点打包时域因变量，创造J__矩阵带状化经验
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos
        sK = self.sK
        idxReorder_ = []  # 索引：重排J__，使J__带状化
        for n in range(Nneg):
            idxReorder_.extend([
                sK.s_csnegsurf.start + n,
                sK.s_ce.start + n,
                sK.s_φsneg.start + n,
                sK.s_φe.start + n])
        for n in range(Nneg, Nneg + Nsep):
            idxReorder_.extend([
                sK.s_ce.start + n,
                sK.s_φe.start + n])
        for n in range(Npos):
            ne = Nneg + Nsep + n
            idxReorder_.extend([
                sK.s_cspossurf.start + n,
                sK.s_ce.start + ne,
                sK.s_φspos.start + n,
                sK.s_φe.start + ne])

        idxReorder_ = array(idxReorder_, dtype=int)
        N = J__.shape[0]
        assert idxReorder_.size == unique(idxReorder_).size == N, \
            f'时域因变量空间打包重排索引异常，{idxReorder_.size = }，{unique(idxReorder_).size = }，{N = }'
        return P2Dbase.prepare_banded_experience(J__, idxReorder_)

    def banded_J__(self, J__: ndarray) -> dict:
        # 同时尝试RCM算法和空间节点打包，选择总带宽(l+u)更小的J__带状化经验
        expeRCM = P2Dbase.banded_by_RCM(J__)
        expeSpace = self.banded_J_by_space(J__)
        widthRCM = expeRCM['l'] + expeRCM['u']
        widthSpace = expeSpace['l'] + expeSpace['u']
        if widthRCM < widthSpace:
            expe = expeRCM
            choice = 'RCM'
        else:
            expe = expeSpace
            choice = '按空间节点打包'
        if self.verbose:
            print(
                f"Jacobi矩阵J__带状化经验："
                f"RCM下带宽 l = {expeRCM['l']}，上带宽 u = {expeRCM['u']}，总带宽 = {widthRCM}；"
                f"按空间节点打包下带宽 l = {expeSpace['l']}，上带宽 u = {expeSpace['u']}，总带宽 = {widthSpace}；"
                f"最终选择{choice}")
        return expe

    def banded_Kf__by_space(self, Kf__: ndarray) -> dict:
        # 按空间节点打包频域因变量，使Kf__形成窄带矩阵
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos
        sKf = self.sKf
        idxReorder_ = []  # 索引：重排Kf__
        for n in range(Nneg):
            idxReorder_.extend([
                sKf.s_REcsnegsurf.start + n,
                sKf.s_IMcsnegsurf.start + n,
                sKf.s_REce.start + n,
                sKf.s_IMce.start + n,
                sKf.s_REφsneg.start + n,
                sKf.s_IMφsneg.start + n,
                sKf.s_REφe.start + n,
                sKf.s_IMφe.start + n,])
        for n in range(Nneg, Nneg + Nsep):
            idxReorder_.extend([
                sKf.s_REce.start + n,
                sKf.s_IMce.start + n,
                sKf.s_REφe.start + n,
                sKf.s_IMφe.start + n,])
        for n in range(Npos):
            ne = Nneg + Nsep + n
            idxReorder_.extend([
                sKf.s_REcspossurf.start + n,
                sKf.s_IMcspossurf.start + n,
                sKf.s_REce.start + ne,
                sKf.s_IMce.start + ne,
                sKf.s_REφspos.start + n,
                sKf.s_IMφspos.start + n,
                sKf.s_REφe.start + ne,
                sKf.s_IMφe.start + ne,])

        idxReorder_ = array(idxReorder_)
        N = Kf__.shape[0]
        assert idxReorder_.size == unique(idxReorder_).size == N, \
            f'频域因变量空间打包重排索引异常，{idxReorder_.size = }，{unique(idxReorder_).size = }，{N = }'
        return P2Dbase.prepare_banded_experience(Kf__, idxReorder_, )

    def Arrhenius(self,
                  X: float | None,  # 参考温度下的参数值
                  E: float,         # 活化能 [J/mol]
                  ):
        # Arrhenius温度修正
        if (self.T==self.Tref) or E==0 or (X is None) :
            return X
        return X * exp(E/R*(1/self.Tref - 1/self.T))

    @staticmethod
    def generate_solve_dUOCPdθs_(
            UOCP: Callable,  # 开路电位函数，输入嵌锂状态θs [–]，输出开路电位UOCP [V]
            ) -> Callable:
        """生成开路电位UOCP对嵌锂状态θs的导数插值函数dUOCPdθs"""
        if isinstance(UOCP, interp1d):
            # 若开路电位函数本身UOCP为scipy插值函数
            θs_ = UOCP.x
            UOCP_ = UOCP.y
        elif isinstance(UOCP, partial) and (UOCP.func is JITinterp1d):
            # 若开路电位函数本身UOCP为Interpolate1D输出的partial包装的interp1d_jit
            θs_, UOCP_ = UOCP.args
        else:
            θs_ = hstack([0.001, arange(0.01, 1, 0.01), 0.999])  # 嵌锂状态序列 [–]
            UOCP_ = UOCP(θs_)                                    # 开路电位序列 [V]
        dUOCPdθs_ = (
              UOCP_[:-2]  * (θs_[1:-1] - θs_[2:]) / (θs_[:-2] - θs_[1:-1]) / (θs_[:-2] - θs_[2:])
            + UOCP_[1:-1] * (1/(θs_[1:-1] - θs_[:-2]) + 1/(θs_[1:-1] - θs_[2:]))
            + UOCP_[2:]   * (θs_[1:-1] - θs_[:-2]) / (θs_[2:] - θs_[:-2]) / (θs_[2:] - θs_[1:-1]) )  # 内部点开路电位导数
        dUOCPdθs0 = (
              UOCP_[0] * (1/(θs_[0] - θs_[1]) + 1/(θs_[0] - θs_[2]))
            + UOCP_[1] * (θs_[0] - θs_[2]) / (θs_[1] - θs_[0]) / (θs_[1] - θs_[2])
            + UOCP_[2] * (θs_[0] - θs_[1]) / (θs_[2] - θs_[0]) / (θs_[2] - θs_[1]) )  # 左界点开路电位导数
        dUOCPdθsEnd = (
              UOCP_[-3] * (θs_[-1] - θs_[-2]) / (θs_[-3] - θs_[-2]) / (θs_[-3] - θs_[-1])
            + UOCP_[-2] * (θs_[-1] - θs_[-3]) / (θs_[-2] - θs_[-3]) / (θs_[-2] - θs_[-1])
            + UOCP_[-1] * (1/(θs_[-1] - θs_[-3]) + 1/(θs_[-1] - θs_[-2]))  )    # 右界点开路电位导数
        return Interpolate1D(θs_, hstack([dUOCPdθs0, dUOCPdθs_, dUOCPdθsEnd]))  # 插值函数
        # return interp1d(θs_, hstack([dUOCPdθs0, dUOCPdθs_, dUOCPdθsEnd]), bounds_error=False, fill_value='extrapolate')

    @staticmethod
    def solve_4θ(UOCPneg: Callable,
                 UOCPpos: Callable,
                 Qcell: float | int,
                 Qneg: float | int,
                 Qpos: float | int,
                 Umin: float  | int,
                 Umax: float | int, ):
        """已知Qcell、Qneg、Qpos，求4个边界嵌锂状态θminneg、θmaxneg、θminpos、θmaxpos"""
        # Qcell, Qneg, Qpos = Qcell, Qneg, Qpos
        # def function(X_: ndarray) -> ndarray:
        #     θminneg, θmaxneg, θminpos, θmaxpos = X_
        #     F_ = array([
        #         UOCPpos(θmaxpos) - UOCPneg(θminneg) - Umin,  # [V]
        #         UOCPpos(θminpos) - UOCPneg(θmaxneg) - Umax,  # [V]
        #         θmaxpos - θminpos - Qcell/Qpos,        # [–]
        #         θmaxneg - θminneg - Qcell/Qneg,])      # [–]
        #     return F_
        # x0_ = array([0.025, 0.85, 0.15, 0.92])  # 迭代初值
        # result = root(function, x0=x0_, method='hybr', tol=1e-8)
        # ΔFmax = abs(function(result.x)).max()
        # θminneg, θmaxneg, θminpos, θmaxpos = result.x

        Ninterval = 20  # θminneg物理可行区间的划分区间数，可手动调整

        # 提取负极开路电位的嵌锂状态、开路电位数据
        if isinstance(UOCPneg, partial):
            θneg_, Uneg_ = UOCPneg.args
        elif isinstance(UOCPneg, interp1d):
            θneg_, Uneg_ = UOCPneg.x, UOCPneg.y
        else:
            θneg_ = arange(0, 1.0001, 0.01)
            Uneg_ = UOCPneg(θneg_)
            UOCPneg = interp1d(θneg_, Uneg_, bounds_error=True)

        # 提取正极开路电位的嵌锂状态、开路电位数据
        if isinstance(UOCPpos, partial) and (
                (UOCPpos.func is JITinterp1d) or
                (UOCPpos.func.__module__ == JITinterp1d.__module__ and
                 UOCPpos.func.__qualname__ == JITinterp1d.__qualname__)):
            θpos_, Upos_ = UOCPpos.args
        elif isinstance(UOCPpos, interp1d):
            θpos_, Upos_ = UOCPpos.x, UOCPpos.y
        else:
            θpos_ = arange(0, 1.0001, 0.01)
            Upos_ = UOCPpos(θpos_)
            UOCPpos = interp1d(θpos_, Upos_, bounds_error=False, fill_value='extrapolate')

        # 构造正负极开路电位反函数
        invUOCPneg = interp1d(Uneg_, θneg_, bounds_error=False, fill_value='extrapolate')
        invUOCPpos = interp1d(Upos_, θpos_, bounds_error=False, fill_value='extrapolate')
        # 正负极嵌锂状态最大范围
        Δθneg = Qcell/Qneg
        Δθpos = Qcell/Qpos
        εθ = 1e-12

        def g(θminneg: float) -> float:
            # 由SOC=0%的开路电压方程反算θmaxpos，再检验SOC=100%的开路电压方程
            θmaxpos = float(invUOCPpos(Umin + UOCPneg(θminneg)))
            θminpos = θmaxpos - Δθpos
            θmaxneg = θminneg + Δθneg
            return float(UOCPpos(θminpos) - UOCPneg(θmaxneg) - Umax)

        # 联立正负极嵌锂状态边界，确定θminneg的物理可行区间
        θminnegPhysicalMin = εθ
        θminnegPhysicalMax = 1 - Δθneg - εθ
        UnegPhysicalBounds_ = UOCPneg(array([θminnegPhysicalMin, θminnegPhysicalMax]))
        UnegRequiredBounds_ = UOCPpos(array([Δθpos + εθ, 1 - εθ])) - Umin
        UnegFeasibleMin = max(UnegPhysicalBounds_.min(), UnegRequiredBounds_.min())
        UnegFeasibleMax = min(UnegPhysicalBounds_.max(), UnegRequiredBounds_.max())
        if not UnegFeasibleMin < UnegFeasibleMax:
            information = {
                'UnegFeasibleMin': UnegFeasibleMin,
                'UnegFeasibleMax': UnegFeasibleMax,
                'constraintViolation': (UnegFeasibleMin - UnegFeasibleMax)/(Umax - Umin),}
            raise P2Dbase.Error('求4个边界嵌锂状态，θminneg的物理可行区间不存在', information)
        θminnegFeasibleBounds_ = invUOCPneg(
            array([UnegFeasibleMin, UnegFeasibleMax]))
        θminnegFeasibleMin = max(θminnegPhysicalMin, θminnegFeasibleBounds_.min())
        θminnegFeasibleMax = min(θminnegPhysicalMax, θminnegFeasibleBounds_.max())
        if not θminnegFeasibleMin < θminnegFeasibleMax:
            information = {
                'θminnegFeasibleMin': θminnegFeasibleMin,
                'θminnegFeasibleMax': θminnegFeasibleMax,
                'constraintViolation': θminnegFeasibleMin - θminnegFeasibleMax,}
            raise P2Dbase.Error('求4个边界嵌锂状态，θminneg的物理可行区间不存在', information)

        # 从小到大遍历θminneg的物理可行区间，找到最小的物理解后停止
        θminnegSearch_ = linspace(θminnegFeasibleMin, θminnegFeasibleMax, Ninterval + 1)
        θmaxposSearch_ = invUOCPpos(Umin + UOCPneg(θminnegSearch_))
        θminposSearch_ = θmaxposSearch_ - Δθpos
        θmaxnegSearch_ = θminnegSearch_ + Δθneg
        g_ = UOCPpos(θminposSearch_) - UOCPneg(θmaxnegSearch_) - Umax  # Ninterval + 1 分隔点的g函数值

        θminneg = None
        for n in range(Ninterval):
            if abs(g_[n]) < 1e-12:
                θminneg = θminnegSearch_[n]
                break
            if g_[n]*g_[n+1] < 0:
                θminneg = brentq(g, θminnegSearch_[n], θminnegSearch_[n+1], xtol=1e-12)
                break
        if θminneg is None and abs(g_[-1]) < 1e-12:
            θminneg = θminnegSearch_[-1]
        if θminneg is None:
            gAbsMin = abs(g_).min()
            information = {
                'θminnegFeasibleMin': θminnegFeasibleMin,
                'θminnegFeasibleMax': θminnegFeasibleMax,
                'gAbsMin': gAbsMin,
                'constraintViolation': gAbsMin/(Umax - Umin),}
            raise P2Dbase.Error('求4个边界嵌锂状态，在物理可行区间内无解', information)

        θmaxneg = θminneg + Δθneg
        θmaxpos = float(invUOCPpos(Umin + UOCPneg(θminneg)))
        θminpos = θmaxpos - Δθpos
        return θminneg, θmaxneg, θminpos, θmaxpos

    def __str__(self):
        Qcell, Qneg, Qpos = self.Qcell, self.Qneg, self.Qpos
        θminneg, θmaxneg, θminpos, θmaxpos = self.θminneg, self.θmaxneg, self.θminpos, self.θmaxpos
        timeDiscretization = self.timeDiscretization
        radialDiscretization = self.radialDiscretization
        θsneg, θspos = self.θsneg, self.θspos
        OCV, U, tC, SOC = self.OCV, self.U, self.T - 273.15, self.SOC
        return (
            f'全电池理论可用容量：{Qcell = :.3f} Ah\n'
            f'电极容量：{Qneg = :.3f} Ah, {Qpos = :.3f} Ah\n'
            f'电极可用容量：{Qneg*(θmaxneg - θminneg) = :.3f} Ah, {Qpos*(θmaxpos - θminpos) = :.3f} Ah\n'
            f'{θminneg = :.4f}, {θmaxneg = :.4f}\n'
            f'{θminpos = :.4f}, {θmaxpos = :.4f}\n'
            f'时间离散：{timeDiscretization = }\n'
            f'球形固相颗粒径向离散：{radialDiscretization = }\n'
            f'当前电极嵌锂状态：{θsneg = :.3f}, {θspos = :.3f}\n'
            f'当前开路电压{OCV = :.3f} V, 端电压{U = :.3f} V, 温度{tC: .1f} °C, {SOC = :.3f}\n'
            + '='*50
            )

    class Error(Exception):
        """P2D模型专属异常类"""
        def __init__(self, information: str, *args):
            super().__init__(information, *args)

    def __call__(self,
            name: str,     # 变量名
            t_: Sequence,  # 时刻序列 [s]
            x_: Sequence | None = None,  # 厚度方向坐标序列 [m]/[–]
            r_: Sequence | None = None,  # 颗粒半径方向坐标序列 [m]/[–]
            f_: Sequence | None = None,  # 频率序列 [Hz]
            ) -> ndarray:
        """因变量插值"""
        data = self.data
        assert name in data.keys(), f"无法识别所输入的变量名'{name}'，变量名name应属于：{data.keys()}"
        kw = {'bounds_error': False,  # 超出边界不报错
              'fill_value': None, }   # None表示外推
        if name in self.EISdatanames_:
            tEIS_ = data['tEIS']
            logf_ = log10(self.f_)
            if name.endswith('__'):
                # 与时间t、频率f、厚度方向坐标x相关的变量
                if ('neg' in name) or ('LP' in name):
                    location_ = self.xneg_
                elif 'pos' in name:
                    location_ = self.xpos_
                else:
                    location_ = self.x_
                interp = RegularGridInterpolator([tEIS_, logf_, location_], data[name], **kw)
                p____ = stack(meshgrid(t_, log10(f_), x_, indexing='ij'), axis=-1)  # (Nt, Nf, Nx, 3) 待插值点
                v___ = interp(p____)  # 插值 (Nt, Nf, Nx)
                return v___
            else:
                # 与时间t、频率f相关的变量
                interp = RegularGridInterpolator([tEIS_, logf_], data[name], **kw)
                p___ = stack(meshgrid(t_, log10(f_), indexing='ij'), axis=-1)  # (Nt, Nf, 2) 待插值点
                v__ = interp(p___)  # 插值 (Nt, Nf)
                return v__
        elif name.endswith('__'):
            # 与厚度方向坐标x、球形颗粒半径方向坐标r、时间t相关的变量，即：'csneg__' 'cspos__' 'θsneg__' 'θspos__'
            reg = name[2:5]  # 提取'neg'或'pos'
            interp = RegularGridInterpolator([data['t'], getattr(self, f'r{reg}_'), getattr(self, f'x{reg}_')], data[name], **kw)
            p____ = stack(meshgrid(t_, r_, x_, indexing='ij'), axis=-1)  # (Nt, Nr, Nx, 3) 待插值点
            v___ = interp(p____)  # 插值 (Nt, Nr, Nx)
            return v___
        elif name.endswith('_'):
            # 与厚度方向坐标x、时间t相关的变量
            if ('neg' in name) or ('LP' in name):
                location_ = self.xneg_
            elif 'pos' in name:
                location_ = self.xpos_
            else:
                location_ = self.x_
            interp = RegularGridInterpolator([data['t'], location_], data[name], **kw)
            p___ = stack(meshgrid(t_, x_, indexing='ij'), axis=-1)  # (Nt, Nx, 2) 待插值点
            v__ = interp(p___)  # 插值 (Nt, Nx)
            return v__
        else:
            # 仅与时间t相关的变量
            v_ = interp1d(data['t'], data[name], bounds_error=False, fill_value='extrapolate')(t_)  # 插值 (Nt,)
            return v_

    def plot_UI(self,
                t_: Sequence | None = None,  # 时刻序列
                ):
        """端电压、电流-时间"""
        if t_ is None:
            t_ = self.data['t']
        U_ = self('U', t_)  # 呈时间序列的电压

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .85, .375])
        ax2.set_position([.1, .08, .85, .375])

        ax1.plot(t_, U_, 'b-')
        ax1.set_ylabel(r'Terminal voltage ${\it U}({\it t})$ [V]')
        ax1.set_ylim([2.4, 4.5])
        ax1.set_yticks(arange(2.4, 4.5 + 1e-6, 0.3))

        if 'I' in self.datanames_:
            I_ = self('I', t_)  # 呈时间序列的电流
            ax2.plot(t_, I_, 'r-')
        ax2.set_ylabel(r'Current ${\it I}({\it t})$ [A]')

        duration = t_[-1] - t_[0]
        xlim_ = [t_[0] - duration*0.02, t_[-1] + duration*0.02]
        for ax in (ax1, ax2):
            ax.set_xlabel(r'Time $\it t$ [s]')
            ax.set_xlim(xlim_)
            ax.grid(axis='y', linestyle='--', color=[.5, .5, .5])
            ax.minorticks_on()
        plt.show()

    def plot_TQgen(self,
               t_: Sequence | None = None,  # 时刻序列
               ):
        """瞬时产热功率、温度-时间"""
        if t_ is None:
            t_ = self.data['t']
        Qgen_ = self('Qgen', t_)  # 呈时间序列的瞬时产热功率
        T_ = self('T', t_)        # 呈时间序列的温度

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .85, .375])
        ax2.set_position([.1, .08, .85, .375])

        ax1.plot(t_, T_ - 273.15, 'r-')
        ax1.set_ylabel(r'Temperature ${\it T}({\it t})$ [°C]')

        ax2.plot(t_, Qgen_, 'k-')
        ax2.set_ylabel(r'Heat generation rate ${\it Q}_{gen}({\it t})$ [W]')

        duration = t_[-1] - t_[0]
        xlim_ = [t_[0] - duration*0.02, t_[-1] + duration*0.02]
        for ax in (ax1, ax2):
            ax.set_xlabel(r'Time $\it t$ [s]')
            ax.set_xlim(xlim_)
            ax.grid(axis='y', linestyle='--', color=[.5, .5, .5])
            ax.minorticks_on()

        plt.show()

    def plot_SOC(self,
                 t_: Sequence | None = None,  # 时刻序列
                 ):
        """SOC-时间"""
        if t_ is None:
            t_ = self.data['t']
        θsneg_ = self('θsneg', t_)  # 呈时间序列的负极嵌锂状态
        θspos_ = self('θspos', t_)  # 呈时间序列的正极嵌锂状态
        SOC_ = self('SOC', t_)      # 呈时间序列的全电池荷电状态

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)

        ax.plot(t_, θsneg_, '--')
        ax.plot(t_, θspos_, 'r--')
        ax.plot(t_, SOC_, 'k-')
        ax.set_ylim(0, 1)
        ax.set_yticks(arange(0, 1.01, 0.1))
        ax.set_ylabel(r'State-of-charge or degree-of-lithiation [–]')
        ax.set_xlabel(r'Time $\it t$ [s]')
        duration = t_[-1] - t_[0]
        ax.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax.legend([r'Negative electrode degree-of-lithiation ${\it θ}_{s,neg}({\it t})$',
                   r'Positive electrode degree-of-lithiation ${\it θ}_{s,pos}({\it t})$',
                   r'Full cell state-of-charge ${\it SOC}({\it t})$'])
        ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_c(self,
               t_: Sequence | None = None,  # 时刻序列
               ):
        """正负极固相颗粒中心、固相颗粒表面、电解液锂离子浓度-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        cθ = 'c' if self.cUnit else 'θ'
        csnegcent___ = self(f'{cθ}sneg__', t_=t_, x_=self.xneg_, r_=[0])  # 呈时间序列的负极固相颗粒中心锂离子浓度
        csposcent___ = self(f'{cθ}spos__', t_=t_, x_=self.xpos_, r_=[0])  # 呈时间序列的正极固相颗粒中心锂离子浓度
        csnegsurf__ = self(f'{cθ}snegsurf_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极固相颗粒表面锂离子浓度
        cspossurf__ = self(f'{cθ}spossurf_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极固相颗粒表面锂离子浓度
        ce__ = self(f'{cθ}e_', t_=t_, x_=self.x_)  # 呈时间序列的电解液锂离子浓度场

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111)
        ax3 = fig.add_subplot(111)
        ax1.set_position([.1, .74, .75, 0.21])
        ax2.set_position([.1, .41, .75, 0.21])
        ax3.set_position([.1, .08, .75, 0.21])

        ax1.set_title('Lithium concentration at electrode particle center', fontsize=12)
        for n, (csnegcent__, csposcent__, t) in enumerate(zip(csnegcent___, csposcent___, t_)):
            x_ = self.xPlot_
            y_ = *csnegcent__.ravel(), *[nan]*self.Nsep, *csposcent__.ravel()
            ax1.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{{self.rSign[1:-1]}=0}}$ [{self.cUnit or '–'}]')
        ax1.legend(bbox_to_anchor=(1, 1))

        ax2.set_title('Lithium concentration at electrode particle surface', fontsize=12)
        for n, (csnegsurf_, cspossurf_, t) in enumerate(zip(csnegsurf__, cspossurf__, t_)):
            x_ = self.xPlot_
            y_ = *csnegsurf_, *[nan]*self.Nsep, *cspossurf_
            ax2.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.rSign[1:-1]} = {"{\\it R}_{s,reg}" if self.rUnit else 1} }}$ [{self.cUnit or '–'}]')

        ax3.set_title('Lithium-ion concentration in electrolyte', fontsize=12)
        for n, (ce_, t) in enumerate(zip(ce__, t_)):
            x_ = 0, *self.xPlot_, self.xInterfacesPlot_[-1]
            y_ = ce_[0], *ce_, ce_[-1]
            ax3.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(rf'{self.cSign}$_e$({self.xSign}, {self.tSign}) [{self.cUnit or '–'}]')

        self.plot_interfaces(ax1, ax2, ax3)
        plt.show()

    def plot_φ(self,
               t_: Sequence | None = None,  # 时刻序列
               ):
        """固液相电势-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        φsneg__ = self('φsneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极固相电势场 [V]
        φspos__ = self('φspos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极固相电势场 [V]
        φe__    = self('φe_', t_=t_, x_=self.x_)        # 呈时间序列的电解液电势场 [V]
        I_      = self('I', t_=t_)  # 呈时间序列的电流 [A]
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(111)
        ax2 = fig.add_subplot(111)
        ax3 = fig.add_subplot(111)
        ax1.set_position([.1,  .59, .33, .375])
        ax2.set_position([.52, .59, .33, .375])
        ax3.set_position([.1,  .08, .75, .375])

        x_ = [0, *self.xPlot_[:Nneg], self.xInterfacesPlot_[Nneg]]
        for n, (φsneg_, I, t) in enumerate(zip(φsneg__, I_, t_)):
            if self.xUnit:
                A, σeffneg = getattr(self, 'A'), getattr(self, 'σeffneg')
                y_ = 1e3*hstack([φsneg_[0] + I/A/σeffneg*0.5*self.Δxneg, φsneg_, φsneg_[-1]])
            else:
                σneg = getattr(self, 'σneg')
                y_ = 1e3*hstack([φsneg_[0] + I/σneg*0.5*self.Δxneg, φsneg_, φsneg_[-1]])
            ax1.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(r'Electrode potential ${\it φ}_{s,neg}$' + f'({self.xSign}, {self.tSign}) [mV]')
        ax1.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
        ax1.set_xlim(x_[0], x_[-1])
        ax1.grid(axis='y', linestyle='--')

        x_ = [self.xInterfacesPlot_[Nneg+Nsep], *self.xPlot_[-Npos:], self.xInterfacesPlot_[-1]]
        for n, (φspos_, I, t) in enumerate(zip(φspos__, I_, t_)):
            if self.xUnit:
                A, σeffpos = getattr(self, 'A'), getattr(self, 'σeffpos')
                y_ = 1e3*hstack([φspos_[0], φspos_, φspos_[-1] - I/A/σeffpos*0.5*self.Δxpos])
            else:
                σpos = getattr(self, 'σpos')
                y_ = 1e3*hstack([φspos_[0], φspos_, φspos_[-1] - I/σpos*0.5*self.Δxpos])
            ax2.plot(x_, y_, 'o-', color=get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(ax1.get_ylabel().replace('neg', 'pos'))
        ax2.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
        ax2.set_xlim(x_[0], x_[-1])
        ax2.grid(axis='y', linestyle='--')
        ax2.legend(bbox_to_anchor=[1, 1])

        x_ = [0, *self.xPlot_, self.xInterfacesPlot_[-1]]
        for n, (φe_, t) in enumerate(zip(φe__, t_)):
            y_ = 1e3*hstack([φe_[0], φe_, φe_[-1]])
            ax3.plot(x_, y_, 'o-', color=get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(r'Electrolyte potential ${\it φ}_e$' + f'({self.xSign}, {self.tSign}) [mV]')

        self.plot_interfaces(ax3)
        plt.show()

    def plot_jint_i0int_ηint(self,
                             t_: Sequence | None = None,  # 时刻序列
                             ):
        """主反应局部体积电流密度/集总局部体积电流密度、交换电流/集总交换电流、过电位-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        jJ, iI = ['j', 'i'] if self.xUnit else ['J', 'I']
        jintneg__ = self(f'{jJ}intneg_', t_=t_, x_=self.xneg_)    # 呈时间序列的负极局部体积电流密度/集总局部体积电流密度场
        jintpos__ = self(f'{jJ}intpos_', t_=t_, x_=self.xpos_)    # 呈时间序列的正极局部体积电流密度/集总局部体积电流密度场
        i0intneg__ = self(f'{iI}0intneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极交换电流/集总交换电流场
        i0intpos__ = self(f'{iI}0intpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极交换电流/集总交换电流场
        ηintneg__ = self('ηintneg_', t_=t_, x_=self.xneg_)*1e3    # 呈时间序列的负极固相表面过电位场 [mV]
        ηintpos__ = self('ηintpos_', t_=t_, x_=self.xpos_)*1e3    # 呈时间序列的正极固相表面过电位场 [mV]

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_position([.1, .74, .75, 0.21])
        ax2.set_position([.1, .41, .75, 0.21])
        ax3.set_position([.1, .08, .75, 0.21])

        ax1.set_title('Field of lithium (de-)intercalation local volumetric current density', fontsize=12)
        for n, (jintneg_, jintpos_, t) in enumerate(zip(jintneg__, jintpos__, t_)):
            x_ = self.xPlot_
            y_ = *jintneg_, *[nan]*self.Nsep, *jintpos_
            ax1.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(rf'{self.jSign}$_{{int}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        ax1.legend(bbox_to_anchor=[1, 1])

        ax2.set_title('Field of lithium (de-)intercalation exchange current density', fontsize=12)
        for n, (i0intneg_, i0intpos_, t) in enumerate(zip(i0intneg__, i0intpos__, t_)):
            x_ = self.xPlot_
            y_ = *i0intneg_, *[nan]*self.Nsep, *i0intpos_
            ax2.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(rf'{self.i0Sign}({self.xSign}, {self.tSign}) [{self.i0Unit}]')

        ax3.set_title('Field of lithium (de-)intercalation overpotential', fontsize=12)
        for n, (ηintneg_, ηintpos_, t) in enumerate(zip(ηintneg__, ηintpos__, t_)):
            x_ = self.xPlot_
            y_ = *ηintneg_, *[nan]*self.Nsep, *ηintpos_
            ax3.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(rf'${{\it η}}_{{int}}$ ({self.xSign}, {self.tSign}) [mV]')

        self.plot_interfaces(ax1, ax2, ax3)
        plt.show()

    def plot_jDL(self,
                 t_: Sequence | None = None,  # 时刻序列
                 ):
        """双电层效应局部体积电流密度/集总局部体积电流密度"""
        if t_ is None:
            t_ = self.data['t']
        jJ = 'j' if self.xUnit else 'J'
        jDLneg__ = self(f'{jJ}DLneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极双电层效应局部体积电流密度/集总局部体积电流密度场
        jDLpos__ = self(f'{jJ}DLpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极双电层效应局部体积电流密度/集总局部体积电流密度场

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        x_ = self.xPlot_
        for n, (jDLneg_, jDLpos_, t) in enumerate(zip(jDLneg__, jDLpos__, t_)):
            y_ = *jDLneg_, *[nan]*self.Nsep, *jDLpos_
            ax1.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'Double-layer local volumetric\ncurrent density {self.jSign}$_{{DL}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        self.plot_interfaces(ax1)
        ax1.legend(bbox_to_anchor=[1, 1])

        t_ = self.data['t']
        jDLneg__ = self(f'{jJ}DLneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极双电层效应局部体积电流密度/集总局部体积电流密度场
        jDLpos__ = self(f'{jJ}DLpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极双电层效应局部体积电流密度/集总局部体积电流密度场
        if jJ=='j':
            A = getattr(self, 'A')
            IDLneg_ = jDLneg__.sum(axis=1)*(self.Δxneg*A)
            IDLpos_ = jDLpos__.sum(axis=1)*(self.Δxpos*A)
        elif jJ=='J':
            IDLneg_ = jDLneg__.mean(axis=1)
            IDLpos_ = jDLpos__.mean(axis=1)
        ax2.plot(t_, IDLneg_, 'k-', label='Negative electrode')
        ax2.plot(t_, IDLpos_, 'r-', label='Positive electrode')
        ax2.set_ylabel(rf'Double-layer current ${{\it I}}_{{DL}}$({self.tSign}) [A]')
        ax2.set_xlabel(rf'Time {self.tSign} [s]')
        duration = t_[-1] - t_[0]
        ax2.legend()
        ax2.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax2.grid(axis='y', linestyle='--', color=[.5, .5, .5])
        ax2.minorticks_on()
        plt.show()

    def plot_csr(self,
                 t_: Sequence | None = None,  # 时刻序列
                 xR: float = 1.,  # 空间位置的相对坐标，X ∈ [0, 1] 表示 [0, Lneg]，X ∈ [2, 3] 表示 [Lneg+Lsep, Lneg+Lsep+Lpos]
                 ):
        """固相颗粒径向锂离子浓度场"""
        if t_ is None:
            t_ = self.data['t']
        assert 0<=xR<=1 or 2<=xR<=3, ('空间位置的相对坐标x的取值范围应为[0, 1]∪[2, 3]，'
                                     'x ∈ [0, 1] 表示 [0, Lneg]，x ∈ [2, 3] 表示 [Lneg+Lsep, Lneg+Lsep+Lpos]')
        LPmodel = self.xUnit==''
        if 0<=xR<=1:
            reg = 'neg'
            x = xR if LPmodel else (xR*getattr(self, 'Lneg'))
        elif 2<=xR<=3:
            reg = 'pos'
            x = xR if LPmodel else ((xR - 2)*getattr(self, 'Lpos') + getattr(self, 'Lneg') + getattr(self, 'Lsep'))
        r_ = getattr(self, f'r{reg}_')
        cθ = 'c' if self.cUnit else 'θ'
        cs___ = self(f'{cθ}s{reg}__', t_=t_, x_=[x], r_=r_)  # 呈时间序列的x位置指定电极固相颗粒锂离子浓度
        cssurf__ = self(f'{cθ}s{reg}surf_', t_=t_, x_=[x])   # 呈时间序列的x位置指定电极固相颗粒表面锂离子浓度

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        ax.set_position([.1, .08, .75, 0.8])
        ax.set_title(rf'Lithium concentration in electrode particle at {self.xSign} = {x if LPmodel else x*1e6:g} {self.xUnit}', fontsize=12)
        X_ = array([0, *r_, 1 if LPmodel else getattr(self, f'Rs{reg}')])
        if not LPmodel:
            X_ *= 1e6
        for n, (cs__, cssurf_, t) in enumerate(zip(cs___, cssurf__, t_)):
            ax.plot(X_,
                    hstack([cs__.ravel()[0], cs__.ravel(), cssurf_.ravel()]),
                    'o-', color=get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax.legend(bbox_to_anchor=[1, 1])
        ax.set_ylabel(rf'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.xSign[1:-1]}={x if LPmodel else x*1e6:g}\;{self.xUnit} }}$ [{self.cUnit or '–'}]')
        ax.set_xlabel(rf'Radial location {self.rSign} [{self.rUnit or '–'}]')
        ax.set_xlim(X_[0], X_[-1])
        ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_OCV_OCP(self):
        """OCV-SOC曲线"""
        fig = plt.figure(figsize=[10, 7])
        axFC = fig.add_subplot(111)
        axFC.set_position([.1, .1, .88, 0.8])
        axNE = axFC.twiny()
        axNE.set_position([.1, .1, .88, 0.8])
        axPE = axNE.twiny()
        axPE.set_position([.1, .1, .88, 0.8])

        axFC.set_ylim([0, 4.5])
        axFC.set_yticks(arange(0, 4.51, 0.3))
        axFC.set_ylabel('Open-circuit voltage/potential [V]')
        SOC_ = arange(0, 1.001, 0.01)                               # 全电池SOC
        θsneg_ = self.θminneg + SOC_*(self.θmaxneg - self.θminneg)  # 负极嵌锂状态
        θspos_ = self.θmaxpos + SOC_*(self.θminpos - self.θmaxpos)  # 正极嵌锂状态
        UOCPpos_ = self.solve_UOCPpos_(θspos_)
        UOCPneg_ = self.solve_UOCPneg_(θsneg_)
        OCV_ = UOCPpos_ - UOCPneg_  # 全电池开路电压
        axFC.plot(SOC_, OCV_, 'k-', label=r'Full cell open-circuit voltage ${\it OCV}$')
        kwargs = {'backgroundcolor':'none', 'fontsize':12,}
        axFC.plot(0, UOCPneg_[0], 'bo', )
        axFC.plot(1, UOCPneg_[-1], 'bo', )
        axFC.plot(0, UOCPpos_[0], 'ro', )
        axFC.plot(1, UOCPpos_[-1], 'ro',)
        axFC.text(0, UOCPneg_[0],  rf'${{\it θ}}_{{min,neg}}$ = {self.θminneg:.3f}', va='bottom', ha='left',**kwargs)
        axFC.text(1, UOCPneg_[-1], rf'${{\it θ}}_{{max,neg}}$ = {self.θmaxneg:.3f}', va='bottom', ha='right', **kwargs)
        axFC.text(0, UOCPpos_[0],  rf'${{\it θ}}_{{max,pos}}$ = {self.θmaxpos:.3f}', va='top', ha='left', **kwargs)
        axFC.text(1, UOCPpos_[-1], rf'${{\it θ}}_{{min,pos}}$ = {self.θminpos:.3f}', va='bottom', ha='right',**kwargs)

        θsneg_ = array([0.001, *arange(0.01, 1, 0.01), 0.999])     # 负极嵌锂状态
        SOCneg_ = (θsneg_ - self.θminneg)/(self.θmaxneg - self.θminneg)  # 全电池SOC
        UOCPneg_ = self.UOCPneg(θsneg_)                                        # 负极开路电位
        axFC.plot(SOCneg_, UOCPneg_, 'b-', label=r'Negative electrode open-circuit potential ${\it U}_{OCP,neg}$')

        θspos_ = array([0.001, *arange(0.01, 1, 0.01), 0.999])     # 正极嵌锂状态
        SOCpos_ = (θspos_ - self.θmaxpos)/(self.θminpos - self.θmaxpos)  # 全电池SOC
        UOCPpos_ = self.UOCPpos(θspos_)                                        # 正极开路电位
        axFC.plot(SOCpos_, UOCPpos_, 'r-', label=r'Positive electrode open-circuit potential ${\it U}_{OCP,pos}$')

        axFC.set_xlabel(r'Full cell $\it SOC$')
        axNE.set_xlabel(r'Negative electrode degree-of-lithiation ${\it θ}_{s,neg}$', color='b')
        axPE.set_xlabel(r'Positive electrode degree-of-lithiation ${\it θ}_{s,pos}$', color='r',
                        labelpad=10,)  # 标签与x轴的间距

        axFC.spines['bottom'].set_position('center')
        axNE.spines['bottom'].set_position(['data', 0])
        axPE.spines['bottom'].set_position(['data', axFC.get_ylim()[1]])

        axFC.spines['bottom'].set_color('k')
        axNE.spines['bottom'].set_color('b')
        axPE.spines['bottom'].set_color('b')

        axFC.spines['top'].set_color('k')
        axNE.spines['top'].set_color('r')
        axPE.spines['top'].set_color('r')

        axFC.xaxis.set_ticks_position('bottom')
        axNE.xaxis.set_ticks_position('bottom')
        axPE.xaxis.set_ticks_position('top')

        axFC.xaxis.set_label_position('bottom')
        axNE.xaxis.set_label_position('bottom')
        axPE.xaxis.set_label_position('top')

        axFC.tick_params(axis='x', colors='k')
        axNE.tick_params(axis='x', colors='b')
        axPE.tick_params(axis='x', colors='r')

        axFC.set_xticks(arange(0, 1.01, 0.1))
        axNE.set_xticks(linspace(SOCneg_[0], SOCneg_[-1], 11))
        axPE.set_xticks(linspace(SOCpos_[0], SOCpos_[-1], 11))

        xticklabels_ = [f'{n:g}' for n in arange(0, 1.1, 0.1)]
        xlim_ = min(0, min(SOCneg_), min(SOCpos_)) - 0.05, max(1, max(SOCneg_), max(SOCpos_)) + 0.05
        for ax in [axFC, axNE, axPE]:
            ax.set_xticklabels(xticklabels_)
            ax.set_xlim(xlim_)

        axFC.legend(loc=[0.5, 0.15])
        kwargs = {'ymin': axFC.get_ylim()[0], 'ymax': axFC.get_ylim()[1],
                  'linestyles': '--', 'color': [.5, .5, .5], 'alpha': 0.5}
        axFC.vlines(0, **kwargs)
        axFC.vlines(1, **kwargs)
        plt.show()

    def plot_dUOCPdθs(self):
        """开路电位对嵌锂状态的导数dUOCPdθs-θs曲线"""
        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        θ_ = arange(0.01, 1.001, 0.01)     # 嵌锂状态
        ax.plot(θ_, self.solve_dUOCPdθsneg_(θ_), 'b-', label='Negative electrode')
        ax.plot(θ_, self.solve_dUOCPdθspos_(θ_), 'r-', label='Positive electrode')
        ax.set_ylim(-10, 0)
        ax.set_xlim(0, 1)
        ax.set_ylabel(r'$\frac{d{\it U}_{OCP}}{d{\it θ}_{s}}$ [V/–]')
        ax.set_xlabel(r'Degree-of-lithiation ${\it θ}_{s}$ [–]')
        ax.grid(axis='y', linestyle='--')
        ax.legend()
        plt.show()

    def plot_nNewton(self):
        """Newton迭代次数nNewton-时间t曲线"""
        t_ = self.data['t']
        nNewton_ = self.data['nNewton']

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        ax.plot(t_, nNewton_, 'x-k')
        duration = max(t_) - min(t_)
        ax.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax.set_xlabel(f'Time {self.tSign} [s]')
        ax.set_ylabel('Number of Newton iterations')
        ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_Z(self,
               f: int | float | None = None,
               t_: Sequence | None = None,
               ):
        """频率f复阻抗实部-时间、频率f复阻抗虚部-时间"""
        if f is None:
            f = self.f_[-1]
        if t_ is None:
            t_ = self.data['tEIS']

        Z_    = self('Z_',    t_=t_, f_=f)
        Zneg_ = self('Zneg_', t_=t_, f_=f)
        Zpos_ = self('Zpos_', t_=t_, f_=f)

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
        ax1.plot(t_, Zpos_.real*1000, 'r-s', label=r'${\it Z}_{pos}$')
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

        duration = t_[-1] - t_[0]
        xlim_ = t_[0]-duration*0.02, t_[-1]+duration*0.02
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(xlim_)
            ax.grid(axis='y', linestyle='--', color=[.5, .5, .5])
            ax.minorticks_on()
        plt.show()

    def plot_Nyquist(self, Z: str = 'Z_',  # 'Z_' 'Zneg_' 'Zpos_'
                     t_: Sequence | None = None,  # 时刻序列
                     f_: Sequence | None = None,  # 频率序列
                     ):
        """Nyquist图"""
        if t_ is None:
            t_ = self.data['tEIS']
        if f_ is None:
            f_ = self.f_
        Z__ = self(Z, t_=t_, f_=f_)*1e3  # 呈时间序列的阻抗谱

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
        REcsnegsurf__ = self(f'RE{cθ}snegsurf__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相表面浓度实部序列
        IMcsnegsurf__ = self(f'IM{cθ}snegsurf__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相表面浓度虚部序列
        REcspossurf__ = self(f'RE{cθ}spossurf__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相表面浓度实部序列
        IMcspossurf__ = self(f'IM{cθ}spossurf__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相表面浓度虚部序列
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
        REce__ = self(f'RE{cθ}e__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液浓度实部序列
        IMce__ = self(f'IM{cθ}e__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液浓度虚部序列
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
        REφsneg__ = self('REφsneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相电势实部序列
        IMφsneg__ = self('IMφsneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极固相电势虚部序列
        REφspos__ = self('REφspos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相电势实部序列
        IMφspos__ = self('IMφspos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极固相电势虚部序列
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
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
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
        REφe__ = self('REφe__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液电势实部序列
        IMφe__ = self('IMφe__', t_=t_, f_=f_, x_=self.x_).reshape(-1, self.Ne)  # 电解液电势虚部序列
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
        """局部体积电流密度/集总局部体积电流密度实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        jJ = 'j' if self.xUnit else 'J'
        REjintneg__ = self(f'RE{jJ}intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极局部体积电流密度/集总局部体积电流密度实部序列
        IMjintneg__ = self(f'IM{jJ}intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极局部体积电流密度/集总局部体积电流密度虚部序列
        REjintpos__ = self(f'RE{jJ}intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极局部体积电流密度/集总局部体积电流密度实部序列
        IMjintpos__ = self(f'IM{jJ}intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极局部体积电流密度/集总局部体积电流密度虚部序列
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
        """双电层效应局部体积电流密度/集总局部体积电流密度实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        jJ = 'j' if self.xUnit else 'J'
        REjDLneg__ = self(f'RE{jJ}DLneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极双电层效应局部体积电流密度/集总局部体积电流密度实部序列
        IMjDLneg__ = self(f'IM{jJ}DLneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 负极双电层效应局部体积电流密度/集总局部体积电流密度虚部序列
        REjDLpos__ = self(f'RE{jJ}DLpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极双电层效应局部体积电流密度/集总局部体积电流密度实部序列
        IMjDLpos__ = self(f'IM{jJ}DLpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 正极双电层效应局部体积电流密度/集总局部体积电流密度虚部序列
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
        """交换电流/集总交换电流实部、虚部-空间、时间"""
        if t_ is None:
            t_ = [self.data['tEIS'][-1]]
        if f_ is None:
            f_ = self.f_
        iI = 'i' if self.xUnit else 'I'
        REi0intneg__ = self(f'RE{iI}0intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 呈时间序列的负极交换电流/集总交换电流实部
        IMi0intneg__ = self(f'IM{iI}0intneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)  # 呈时间序列的负极交换电流/集总交换电流虚部
        REi0intpos__ = self(f'RE{iI}0intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 呈时间序列的正极交换电流/集总交换电流实部
        IMi0intpos__ = self(f'IM{iI}0intpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)  # 呈时间序列的正极交换电流/集总交换电流虚部
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
        REηintneg__ = self('REηintneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)*1e3  # 负极反应过电位实部序列 [mV]
        IMηintneg__ = self('IMηintneg__', t_=t_, f_=f_, x_=self.xneg_).reshape(-1, self.Nneg)*1e3  # 负极反应过电位虚部序列 [mV]
        REηintpos__ = self('REηintpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)*1e3  # 正极反应过电位实部序列 [mV]
        IMηintpos__ = self('IMηintpos__', t_=t_, f_=f_, x_=self.xpos_).reshape(-1, self.Npos)*1e3  # 正极反应过电位虚部序列 [mV]
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

        for n, (IMηintneg_, IMηintpos_, label) in enumerate(zip(IMηintneg__, IMηintpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMηintneg_, *([nan]*self.Nsep), *IMηintpos_
            ax2.plot(x_, y_, 'o-', color=get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def plot_interfaces(self, *axes_):
        for ax in axes_:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')  # 横坐标标签
            ax.set_ylim(ax.get_ylim())                   # 固定纵坐标上下限
            ax.set_xlim(self.xInterfacesPlot_[[0, -1]])  # 横坐标上下限
            kw = {'ymin': ax.get_ylim()[0], 'ymax': ax.get_ylim()[1],
                  'ls': '--', 'color': [.5]*3, 'alpha': 0.5}
            ax.vlines(self.xInterfacesPlot_[self.Nneg], **kw)              # 负极-隔膜界面
            ax.vlines(self.xInterfacesPlot_[self.Nneg + self.Nsep], **kw)  # 隔膜-正极界面
            ax.grid(axis='y', linestyle='--')  # 纵坐标网格线

    @property
    def xPlot_(self):
        """全区域控制体中心的坐标（用于作图） """
        return self.x_

    @property
    def xInterfacesPlot_(self):
        """各控制体交界面的坐标（用于作图） """
        return self.xInterfaces_
    
    def save(self,
            path: str | None = None,  # 路径\文件名
            datanames_: Sequence[str] = None,  # 仅保存self.data当中指定的运行数据
            **otherData_,  # 其它需保存的数据集
            ):
        """保存数据"""
        assert isinstance(otherData_, dict) or otherData_ is None, 'otherData_应为字典或None'
        if path is None:
            # 默认保存路径
            filepath = pathlib.Path().cwd()  # 获得当前工作目录
            path = filepath.joinpath(f'{time.strftime("%Y%m%d %H%M%S", time.localtime())}保存{self.__class__.__name__}模拟数据.npz')

        # 保存运行数据
        if datanames_:
            # 若给定了特定数据名，则只保存datanames_所列出的运行数据
            savedData_ = {}
            for dataname in self.data:
                if dataname in datanames_:
                    savedData_[dataname] = self.data[dataname]
        else:
            savedData_ = self.data

        savez(path, **(savedData_ | otherData_))  # 保存
        print(f'已保存{path}')
        return path

    @staticmethod
    def s2idx(s: slice) -> ndarray:
        """将切片索引s转化为数组索引idx"""
        return arange(s.start, s.stop, s.step)

    def record_data(self):
        """记录时域数据"""
        self.ΔφsenegHistory__.append(self.ηLPneg_)
        self.ΔφseposHistory__.append(self.ηLPpos_)
        self.Ihistory_.append(self.I)
        data = self.data
        for name in self.datanames_:
            value = getattr(self, name)
            if isinstance(value, ndarray):
                value = value.copy()
            data[name].append(value)

    def record_EISdata(self):
        """记录频域数据"""
        data = self.data
        for name in self.EISdatanames_:
            value = getattr(self, name)
            if isinstance(value, ndarray):
                value = value.copy()
            data[name].append(value)

    @property
    def tC(self):
        """摄氏温度 [°C]"""
        return self.T - 273.15

    @property
    @abstractmethod
    def U(self):
        """全电池端电压 [V]"""
        pass

    @property
    @abstractmethod
    def θsneg(self):
        """负极嵌锂状态 [–]"""
        pass

    @property
    @abstractmethod
    def θspos(self):
        """正极嵌锂状态 [–]"""
        pass

    @property
    @abstractmethod
    def ηLPneg_(self):
        """负极析锂反应过电位场 [V]"""
        pass

    @property
    @abstractmethod
    def ηLPpos_(self):
        """正极析锂反应过电位场 [V]"""
        pass

    @abstractmethod
    def _stepping(self, Δt) -> int:
        """时间步进：Newton迭代时域因变量"""
        pass

    @abstractmethod
    def EIS(self):
        """计算电化学阻抗谱"""
        pass

if __name__=='__main__':
    # cell = P2Dbase()

    ...

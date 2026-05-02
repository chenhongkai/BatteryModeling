#%%
import time, pathlib
from math import exp
from typing import Sequence, Callable
from collections.abc import Iterable
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.linalg.lapack import dgbsv
from scipy.optimize import root
from numpy import ndarray, nan, \
    array, arange, zeros, full, linspace, stack, hstack, concatenate, meshgrid, \
    ix_, asfortranarray, isnan, isscalar, savez

from P2Dmodel.OCP import NMC111, Graphite
from P2Dmodel.tools import Interpolate1D, set_matplotlib, F, R


class P2Dbase(ABC):
    """抽象类：锂离子电池准二维模型 Pseudo-two-Dimensional model"""

    ## 常数 ##
    F = F  # 法拉第Faraday常数 [C/mol]
    R = R  # 理想气体常数 [J/(mol·K)]

    ## 类型注解 ##
    _kneg: float; _kpos: float    # 负极、正极主反应速率常数
    _kLP: float                   # 负极析锂反应速率常数
    _Dsneg: float; _Dspos: float  # 负极、正极固相扩散系数
    CDLneg: float; CDLpos: float  # 负极、正极双电层电容
    _i0intneg: float | None       # 负极交换电流密度
    _i0intpos: float | None       # 正极交换电流密度
    data: dict[str, list[float | ndarray]]  # 数据记录字典
    K__: ndarray       # (N, N) 因变量线性矩阵
    x_: ndarray        # (Ne,) 电极厚度方向控制体中心坐标
    Δx_: ndarray       # (Ne,) 电极厚度方向控制体厚度序列
    ΔxWest_: ndarray   # (Ne,) 当前控制体中心到左侧控制体中心的距离
    ΔxEast_: ndarray   # (Ne,) 当前控制体中心到右侧控制体中心的距离

    def __init__(self,
            T0: float = 298.15,  # 初始温度 [K]
            SOC0: float = 0.2,   # 初始荷电状态 [–]
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
            dUOCPdTneg: Callable | float = 0.,    # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位的熵热系数 [V/K]
            dUOCPdTpos: Callable | float = 0.,    # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位的熵热系数 [V/K]
            dUOCPdθsneg: Callable | None = None,  # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位对嵌锂状态的导数 [V/–]
            dUOCPdθspos: Callable | None = None,  # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位对嵌锂状态的导数 [V/–]
            EDsneg: float = 30e3,  # 负极锂离子扩散系数的比活化能 [J/mol]
            EDspos: float = 30e3,  # 正极锂离子扩散系数的比活化能 [J/mol]
            Ekneg: float = 68e3,   # 负极主反应速率常数的比活化能 [J/mol]
            Ekpos: float = 50e3,   # 正极主反应速率常数的比活化能 [J/mol]
            Eκ: float = 4e3,       # 电解液锂离子电导率的比活化能 [J/mol]
            EDe: float = 16.5e3,   # 电解液锂离子扩散系数的比活化能 [J/mol]
            EkLP: float = 3.53e4,  # 负极析锂反应速率常数的比活化能 [J/mol]
            Tref: float = 298.15,  # 参考温度 [K]
            hA: float = 1.5,       # 表面传热系数与传热面积之积 [W/K]
            Cth: float = 600.,     # 热容 [J/K]
            Tamb: float = 298.15,  # 环境温度 [K]
            lithiumPlating: bool = False,      # 是否考虑析锂反应
            doubleLayerEffect: bool = True,    # 是否考虑电极颗粒双电层效应
            timeDiscretization: str = 'CN',    # 时间离散格式 'CN'/'backward'
            radialDiscretization: str = 'EV',  # 球形颗粒径向离散方法 等体积'EV'/等间隔'EI'
            decouple_cs: bool = True,  # 是否解耦固相锂离子浓度的求解，设置decouple_cs==True可加速，几乎无代价
            constants: bool = False,   # 是否使用常量参数缓存，设置constants==True可加速，但应在考虑参数为恒定值的情况下
            complete: bool = True,     # 是否确保功能完备，设置complete==False可加速，但省略不必要的计算和存储
            verbose: bool = True,      # 是否显示初始化、运行进度
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
        self.Nr = Nr; assert isinstance(Nr, int) and Nr>=3, f'球形固相颗粒半径方向网格数{Nr = }，应为不小于3的正整数'
        self.Ne = Ne = Nneg + Nsep + Npos  # 电解液网格总数
        # 函数
        self.UOCPneg = UOCPneg; assert callable(UOCPneg), '函数UOCPneg，输入负极嵌锂状态θsneg_ [–]，输出正极开路电位UOCPneg_ [V]'
        self.UOCPpos = UOCPpos; assert callable(UOCPpos), '函数UOCPpos，输入正极嵌锂状态θspos_ [–]，输出负极开路电位UOCPpos_ [V]'
        self.dUOCPdTneg = dUOCPdTneg; assert callable(dUOCPdTneg) or isinstance(dUOCPdTneg, (int, float)), '负极开路电位的熵热系数dUOCPdTneg [V/K]，标量或函数（输入负极嵌锂状态θsneg_ [–]）'
        self.dUOCPdTpos = dUOCPdTpos; assert callable(dUOCPdTpos) or isinstance(dUOCPdTpos, (int, float)), '正极开路电位的熵热系数dUOCPdTpos [V/K]，标量或函数（输入正极嵌锂状态θspos_ [–]）'
        assert callable(dUOCPdθsneg) or (dUOCPdθsneg is None), '负极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入负极嵌锂状态θsneg_ [–]）'
        self.solve_dUOCPdθsneg_ = P2Dbase.generate_solve_dUOCPdθs_(UOCPneg) if (dUOCPdθsneg is None) else dUOCPdθsneg
        assert callable(dUOCPdθspos) or (dUOCPdθspos is None), '正极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入正极嵌锂状态θspos_ [–]）'
        self.solve_dUOCPdθspos_ = P2Dbase.generate_solve_dUOCPdθs_(UOCPpos) if (dUOCPdθspos is None) else dUOCPdθspos
        # 比活化能
        self.EDsneg = EDsneg; assert EDsneg>=0, f'负极锂离子扩散系数的比活化能{EDsneg = }，应大于或等于0 [J/mol]'
        self.EDspos = EDspos; assert EDspos>=0, f'正极锂离子扩散系数的比活化能{EDspos = }，应大于或等于0 [J/mol]'
        self.Ekneg = Ekneg; assert Ekneg>=0,    f'负极主反应速率常数的比活化能{Ekneg = }，应大于或等于0 [J/mol]'
        self.Ekpos = Ekpos; assert Ekpos>=0,    f'正极主反应速率常数的比活化能{Ekpos = }，应大于或等于0 [J/mol]'
        self.Eκ = Eκ;     assert Eκ>=0,   f'电解液锂离子电导率的比活化能{Eκ = }，应大于或等于0 [J/mol]'
        self.EDe = EDe;   assert EDe>=0,  f'电解液锂离子扩散系数的比活化能{EDe = }，应大于或等于0 [J/mol]'
        self.EkLP = EkLP; assert EkLP>=0, f'负极析锂反应速率常数的比活化能{EkLP = }，应大于或等于0 [J/mol]'
        self.Tref = Tref; assert Tref>0, f'参考温度{Tref = }，应大于0 [K]'
        # 热参数
        self.hA = hA; assert hA>0, f'表面传热系数与传热面积之积{hA = }，应大于0 [W/K]'
        self.Cth = Cth; assert Cth>0, f'电池热容{Cth = }，应大于0 [J/K]'
        self.Tamb = Tamb; assert Tamb>0, f'环境温度{Tamb = }，应大于0 [K]'
        # 模式
        self.lithiumPlating = lithiumPlating        # 是否考虑析锂反应
        self.doubleLayerEffect = doubleLayerEffect  # 是否考虑双电层效应
        self.timeDiscretization = timeDiscretization;     assert timeDiscretization in {'backward', 'CN'}, f'时间离散格式{timeDiscretization = }，应为 "CN"（Crank-Nicolson） 或 "backward"（后向差分格式）'
        self.radialDiscretization = radialDiscretization; assert radialDiscretization in {'EV', 'EI'}, f'球形颗粒径向离散方法{radialDiscretization = }，应为 "EV"（等体积）/"EI"（等间隔）'
        self.decouple_cs = decouple_cs  # 是否解耦固相锂离子浓度的求解
        self.constants = constants      # 是否使用常量参数缓存
        self.complete = complete        # 是否确保功能完备
        self.verbose = verbose          # 是否显示初始化、运行进度
        # 状态量
        self.T = T0  # 温度
        self.I = 0.    # 电流 [A]
        self.t = 0.    # 时刻 [s]
        self.QLP = 0.  # 累计析锂量 [Ah]
        θsneg = θminneg + SOC0*(θmaxneg - θminneg)  # 初始负极嵌锂状态
        θspos = θmaxpos + SOC0*(θminpos - θmaxpos)  # 初始正极嵌锂状态
        self.φsneg_ = full(Nneg, UOCPneg := self.solve_UOCPneg_(θsneg))  # 初始化：负极固相电势场 [V]
        self.φspos_ = full(Npos, UOCPpos := self.solve_UOCPpos_(θspos))  # 初始化：正极固相电势场 [V]
        self.φe_ = zeros(Ne)                                                   # 初始化：电解液电势场 [V]
        self.ηintneg_, self.ηintpos_ = zeros(Nneg), zeros(Npos)                # 初始化：负极、正极主反应过电位场 [V]
        self.ηLPneg_, self.ηLPpos_ = full(Nneg, UOCPneg), full(Npos, UOCPpos)  # 初始化：负极、正极析锂反应过电位场 [V]
        self.datanames_ = [         # 需记录的数据名称
            'U', 'I', 't',          # 端电压 [V]、电流 [A]、时刻 [s]
            'ηLPneg_', 'ηLPpos_',]  # 负极、正极表面析锂反应过电位场 [V]
        self.bandwidthsJ_: dict[str, int] = None  # Jacobi矩阵带宽
        self.idxJreordered_: ndarray = None       # 索引：重排Jacobi矩阵
        self.idxJrecovered_: ndarray = None       # 索引：恢复排序Jacobi矩阵
        # 恒定量
        if decouple_cs:
            self.e__ = zeros((Nr, 1))  # (Nr, 1) 非零末元为1的向量
            self.e__[-1] = 1.
        if complete:
            # matplotlib作图设置
            set_matplotlib()
            # matplotlib作图变量单位
            self.tSign, self.tUnit = r'${\it t}$', 's'           # 时间t符号、单位
            self.xSign, self.xUnit = r'${\it x}$', 'μm'           # 电极厚度方向坐标x符号、单位
            self.rSign, self.rUnit = r'${\it r}$', 'μm'           # 径向坐标r符号、单位
            self.cSign, self.cUnit = r'${\it c}$', 'mol/m$^3$'    # 锂离子浓度c符号、单位
            self.jSign, self.jUnit = r'${\it j}$', 'A/m$^3$'      # 局部体积电流密度j符号、单位
            self.i0Sign, self.i0Unit = r'${\it i}_0$', 'A/m$^2$'  # 交换电流密度i0符号、单位

    # 因变量索引
    idxcsneg_: ndarray; idxcspos_: ndarray
    idxcsnegsurf_: ndarray; idxcspossurf_: ndarray
    idxce_: ndarray
    idxφsneg_: ndarray; idxφspos_: ndarray
    idxφe_: ndarray
    idxjintneg_: ndarray; idxjintpos_: ndarray
    idxjDLneg_: ndarray; idxjDLpos_: ndarray
    idxi0intneg_: ndarray; idxi0intpos_: ndarray
    idxηintneg_: ndarray; idxηintpos_: ndarray
    idxjLP_: ndarray; idxηLP_: ndarray
    idxc_: ndarray; idxφ_: ndarray; idxj_: ndarray

    def generate_indices_of_dependent_variables(self,):
        """生成矩阵K__的因变量索引"""
        Nneg, Npos, Ne, Nr = self.Nneg, self.Npos, self.Ne, self.Nr
        decouple_cs, lithiumPlating = self.decouple_cs, self.lithiumPlating
        N = 0  # 全局索引游标
        def allocate(n):
            # 分配索引
            nonlocal N
            idx_ = arange(N, N + n)
            N += n
            return idx_
        self.idxcsneg_ = idxcsneg_ = allocate(0 if decouple_cs else Nr*Nneg)  # 索引：负极固相内部浓度 先排颗粒径向r，再排x方向
        self.idxcspos_ = idxcspos_ = allocate(0 if decouple_cs else Nr*Npos)  # 索引：正极固相内部浓度
        self.idxcsnegsurf_ = idxcsnegsurf_ = allocate(Nneg)  # 索引：正极固相表面浓度
        self.idxcspossurf_ = idxcspossurf_ = allocate(Npos)  # 索引：正极固相表面浓度
        self.idxce_ = idxce_ = allocate(Ne)  # 索引：电解液浓度
        self.idxφsneg_ = idxφsneg_ = allocate(Nneg)  # 索引：负极固相电势
        self.idxφspos_ = idxφspos_ = allocate(Npos)  # 索引：正极固相电势
        self.idxφe_ = idxφe_ = allocate(Ne)  # 索引：电解液电势
        self.idxjintneg_ = idxjintneg_ = allocate(Nneg)  # 索引：负极主反应局部体积电流密度
        self.idxjintpos_ = idxjintpos_ = allocate(Npos)  # 索引：正极主反应局部体积电流密度
        self.idxjDLneg_ = idxjDLneg_ = allocate(Nneg if self.CDLneg else 0)  # 索引：负极双电层局部体积电流密度
        self.idxjDLpos_ = idxjDLpos_ = allocate(Npos if self.CDLpos else 0)  # 索引：正极双电层局部体积电流密度
        self.idxi0intneg_ = allocate(0 if self._i0intneg else Nneg)  # 索引：负极主反应交换电流密度
        self.idxi0intpos_ = allocate(0 if self._i0intpos else Npos)  # 索引：正极主反应交换电流密度
        self.idxηintneg_ = idxηintneg_ = allocate(Nneg)  # 索引：负极主反应过电位
        self.idxηintpos_ = idxηintpos_ = allocate(Npos)  # 索引：正极主反应过电位
        self.idxjLP_ = idxjLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：负极析锂反应局部体积电流密度
        self.idxηLP_ = idxηLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：负极析锂反应过电位
        self.idxc_ = concatenate([idxcsneg_, idxcspos_, idxcsnegsurf_, idxcspossurf_, idxce_])       # 索引：所有浓度量
        self.idxφ_ = concatenate([idxφsneg_, idxφspos_, idxφe_, idxηintneg_, idxηintpos_, idxηLP_])  # 索引：所有电势量
        self.idxj_ = concatenate([idxjintneg_, idxjintpos_, idxjDLneg_, idxjDLpos_, idxjLP_])        # 索引：所有局部体积电流量
        return N

    def assign_K__with_constants(self):
        # 对K__矩阵赋恒定值
        Nneg, Npos = self.Nneg, self.Npos
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        K__ = self.K__
        # 负极固相电势φsneg行φsneg列
        idxφsneg_ = self.idxφsneg_
        K__[idxφsneg_[1:], idxφsneg_[:-1]] = 1  # φsneg列下对角线
        K__[idxφsneg_[:-1], idxφsneg_[1:]] = 1  # φsneg列上对角线
        K__[idxφsneg_, idxφsneg_] = [-1] + [-2]*(Nneg - 2) + [-1]  # φsneg列主对角线
        # 正极固相电势φspos行φspos列
        idxφspos_ = self.idxφspos_
        K__[idxφspos_[1:], idxφspos_[:-1]] = 1  # φspos列下对角线
        K__[idxφspos_[:-1], idxφspos_[1:]] = 1  # φspos列上对角线
        K__[idxφspos_, idxφspos_] = [-1] + [-2]*(Npos - 2) + [-1]  # φspos列主对角线
        # 电解液电势φe行j列
        idxφe_ = self.idxφe_
        idxφeneg_, idxφepos_ = idxφe_[:Nneg], idxφe_[-Npos:]
        K__[idxφeneg_, self.idxjintneg_] = Δxneg  # jintneg列
        K__[idxφepos_, self.idxjintpos_] = Δxpos  # jintpos列
        if self.idxjDLneg_.size:
            K__[idxφeneg_, self.idxjDLneg_] = Δxneg  # jDLneg列
        if self.idxjDLpos_.size:
            K__[idxφepos_, self.idxjDLpos_] = Δxpos  # jDLpos列
        # 负极、正极主反应局部体积电流密度jint行jint列
        idxjint_ = concatenate([self.idxjintneg_, self.idxjintpos_])
        K__[idxjint_, idxjint_] = 1
        # 负极、正极主反应交换电流密度i0int行i0int列
        idxi0int_ = concatenate([self.idxi0intneg_, self.idxi0intpos_])
        K__[idxi0int_, idxi0int_] = 1
        # 负极、正极主反应过电位ηint行
        idxηintneg_, idxηintpos_ = self.idxηintneg_, self.idxηintpos_
        idxηint_ = concatenate([idxηintneg_, idxηintpos_])
        K__[idxηint_, idxηint_] = 1  # ηint列
        K__[idxηintneg_, idxφeneg_] = \
        K__[idxηintpos_, idxφepos_] = 1  # φe列
        K__[idxηintneg_, idxφsneg_] = \
        K__[idxηintpos_, idxφspos_] = -1  # φs列
        # 析锂补充
        if self.lithiumPlating:
            idxjLP_ = self.idxjLP_
            idxηLP_ = self.idxηLP_
            K__[idxφeneg_, idxjLP_] = Δxneg  # φeneg行jLP列
            K__[idxjLP_, idxjLP_] = 1     # jLP行jLP列
            K__[idxηLP_, idxφsneg_] = -1  # ηLP行φsneg列
            K__[idxηLP_, idxφeneg_] = 1   # ηLP行φeneg列
            K__[idxηLP_, idxηLP_] = 1     # ηLP行ηLP列

    def update_K__idxcsnegsurf_idxjintneg_(self, aneg, Dsneg):
        # 更新K__矩阵csnegsurf行jintneg列
        self.K__[self.idxcsnegsurf_, self.idxjintneg_] = 1/(aneg*P2Dbase.F*Dsneg)

    def update_K__idxcspossurf_idxjintpos_(self, apos, Dspos):
        # 更新K__矩阵cspossurf行jintpos列
        self.K__[self.idxcspossurf_, self.idxjintpos_] = 1/(apos*P2Dbase.F*Dspos)

    def update_K__idxφsneg_idxjneg_(self, σeffneg):
        # 更新K__矩阵φsneg行jneg列
        K__ = self.K__
        idxφsneg_ = self.idxφsneg_
        K__[idxφsneg_, self.idxjintneg_] = a = -self.Δxneg**2/σeffneg  # jintneg列
        if self.idxjDLneg_.size:
            K__[idxφsneg_, self.idxjDLneg_] = a  # jDLneg列
        if self.idxjLP_.size:
            K__[idxφsneg_, self.idxjLP_] = a     # jLP列

    def update_K__idxφspos_idxjpos_(self, σeffpos):
        # 更新K__矩阵φspos行jpos列
        K__ = self.K__
        K__[self.idxφspos_, self.idxjintpos_] = a = -self.Δxpos**2/σeffpos  # jintpos列
        if self.idxjDLpos_.size:
            K__[self.idxφspos_, self.idxjDLpos_] = a  # jDLpos列

    def update_K__idxφe_idxφe_(self, κeffWest_, κeffEast_):
        # 更新K__矩阵φe行φe列
        ΔxEast_, ΔxWest_, Δx_ = self.ΔxEast_, self.ΔxWest_, self.Δx_
        Nneg, Nsep = self.Nneg, self.Nsep
        K__ = self.K__
        idxφe_ = self.idxφe_
        a = κeffEast_[0]/ΔxEast_[0]
        K__[idxφe_[0], idxφe_[:2]] = [-(κeffWest_[0]/(0.5*Δx_[0]) + a), a]  # φe列首行
        a = κeffWest_[-1]/ΔxWest_[-1]
        K__[idxφe_[-1], idxφe_[-2:]] = [a, -a]  # φe列末行
        K__[idxφe_[1:-1], idxφe_[:-2]] = a_ = κeffWest_[1:-1]/ΔxWest_[1:-1]  # φe列下对角线
        K__[idxφe_[1:-1], idxφe_[2:]] = c_ = κeffEast_[1:-1]/ΔxEast_[1:-1]   # φe列上对角线
        K__[idxφe_[1:-1], idxφe_[1:-1]] = -(a_ + c_)  # φe列主对角线
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, c = κeffWest_[nW]/ΔxWest_[nW], 2*κeffEast_[nW]*κeffWest_[nE]/(κeffEast_[nW]*Δx_[nE] + κeffWest_[nE]*Δx_[nW])
            K__[idxφe_[nW], idxφe_[nW-1:nW+2]] = [a, -(a + c), c]  # 界面左侧控制体
            a, c = c, κeffEast_[nE]/ΔxEast_[nE]
            K__[idxφe_[nE], idxφe_[nE-1:nE+2]] = [a, -(a + c), c]  # 界面右侧控制体

    def update_K__idxηintneg_idxjneg_(self, RSEIneg, aeffneg):
        # 更新K__矩阵ηintneg行jneg列
        K__ = self.K__
        K__[self.idxηintneg_, self.idxjintneg_] = a = RSEIneg/aeffneg  # jintneg列
        if self.idxjDLneg_.size:
            K__[self.idxηintneg_, self.idxjDLneg_] = a  # jDLneg列
        if self.lithiumPlating:
            K__[self.idxηintneg_, self.idxjLP_] = a     # jLP列

    def update_K__idxηintpos_idxjpos_(self, RSEIpos, aeffpos):
        # 更新K__矩阵ηintpos行jpos列
        K__ = self.K__
        K__[self.idxηintpos_, self.idxjintpos_] = a = RSEIpos/aeffpos  # jintpos列
        if self.idxjDLpos_.size:
            K__[self.idxηintpos_, self.idxjDLpos_] = a  # jDLpos列

    def update_K__idxηLP_idxjneg_(self, RSEIneg, aeffneg):
        # 更新K__矩阵ηLP行jneg列
        K__ = self.K__
        idxηLP_ = self.idxηLP_
        K__[idxηLP_, self.idxjintneg_] = \
        K__[idxηLP_, self.idxjLP_] = a = RSEIneg/aeffneg  # jintneg、jLP列
        if self.idxjDLneg_.size:
            K__[idxηLP_, self.idxjDLneg_] = a             # jDLneg列

    def CC(self,
            I: float | int = 0,             # 电流 [A]，放电为正，充电为负
            timeInterval: float | int = 1,  # 充放电运行时间 [s]
            thermalModel: bool = False,     # 是否计算电池产热
            Umax: float | None = None,      # 最高电压 [V]
            Umin: float | None = None,      # 最低电压 [V]
            SOCmax: float | None = None,    # 最大SOC
            SOCmin: float | None = None,    # 最小SOC
            minΔt: float = 0.1,             # 最小时间步长 [s]
            ):
        """恒流充放电"""
        assert self.Δt>=minΔt, f'时间步长Δt = {self.Δt}s，应大于{minΔt = }s'
        if verbose := self.verbose:
            startTime = time.time()  # 开始时间戳 [s]
        lithiumPlating = self.lithiumPlating
        if self.t==0:
            self.record_data()        # 记录初始时刻数据
        data = self.data              # 读取：运行数据字典
        tStart = data['t'][-1]        # 开始时刻 [s]
        tEnd = tStart + timeInterval  # 终止时刻 [s]
        self.I = I                    # 电流 [A]
        info = f"电流{I = :.2f}A{'放电' if I>0 else ('充电' if I<0 else '静置')}"

        while True:
            ### 持续时间步进...
            # 检查终止条件
            if self.t>=tEnd:
                if verbose:
                    print(f'\n达到运行时长{timeInterval}s，停止{info}')
                break
            if I<0 and (Umax is not None) and self.U>=Umax:
                if verbose:
                    print(f'\n充电达到{Umax = }V，停止充电')
                break
            if I<0 and (SOCmax is not None) and self.SOC>=SOCmax:
                if verbose:
                    print(f'\n充电达到{SOCmax = }，停止充电')
                break
            if I>0 and (Umin is not None) and self.U<=Umin:
                if verbose:
                    print(f'\n放电达到{Umin = }V，停止放电')
                break
            if I>0 and (SOCmin is not None) and self.SOC<=SOCmin:
                if verbose:
                    print(f'\n放电达到{SOCmin = }，停止放电')
                break
            # 选择时间步长
            remainingTime = tEnd - self.t  # 剩余时长 [s]
            if remainingTime<(self.Δt + minΔt):
                Δt = remainingTime  # 使用剩余时长作为最后时间步长
            else:
                Δt = self.Δt  # 使用默认时间步长 [s]

            while True:
                # 试探步进
                self.t = self.data['t'][-1] + Δt  # 更新：时刻 [s]
                try:
                    nNewton = self.step(Δt)
                    break  # 试探成功，无报错，跳出
                except P2Dbase.Error as message:
                    if Δt==minΔt:
                        raise
                    ΔtNew = max(minΔt, Δt/2)
                    if verbose:
                        print(f'异常：{message}，缩小时间步长{Δt}s-->{ΔtNew}', )
                    Δt = ΔtNew
                    continue

            if lithiumPlating:
                self.QLP += -self.ILP*Δt/3600  # 更新：累计析锂量 [Ah]

            if thermalModel:
                dTdt = (self.Qgen + self.hA*(self.Tamb - self.T))/self.Cth
                self.T += dTdt*Δt  # 更新温度 [K]

            self.record_data()  # 记录运行数据

            if verbose:
                # 显示进度
                finishedProportion = (self.t - tStart)/timeInterval  # 已完成的比例
                finishedProgresses = int(25*finishedProportion)  # 已完成的进度条长度
                unfinishedProgresses = 25 - finishedProgresses   # 未完成的进度条长度
                finishedBar = '▓'*finishedProgresses        # 已完成的进度条
                unfinishedBar = '-'*unfinishedProgresses    # 未完成的进度条
                percentage = finishedProportion*100         # 已完成进度的百分比
                timeStamp = time.time() - startTime         # 累计耗时
                U = self.U
                SOC = self.SOC
                print(f'|{finishedBar}{unfinishedBar}|已完成{percentage:.0f}%，耗时{timeStamp:.1f}s，'
                      f't={self.t:g}s-->{tEnd:g}s，{info}，'
                      f'电压{U = :.3f}V, {SOC = :.3f}，温度{self.T - 273.15:.1f}°C，'
                      f'析锂过电位{self.ηLPneg_[-1]*1000:.0f}mV，'
                      f'{nNewton}次Newton迭代'
                      f'\r', end='')
        return self

    def record_data(self):
        """记录数据"""
        for dataname in self.datanames_:
            value = getattr(self, dataname)
            if isscalar(value):
                pass
            else:
                value = value.copy()
            self.data[dataname].append(value)

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
    def kLP(self):
        """负极析锂反应速率常数"""
        return self.Arrhenius(self._kLP, self.EkLP)
    @kLP.setter
    def kLP(self, kLP):
        self._kLP = kLP

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
    def Δxneg(self):
        """负极网格厚度"""
        return self.Δx_[0]

    @property
    def Δxpos(self):
        """正极网格厚度"""
        return self.Δx_[-1]

    @property
    def xInterfaces_(self):
        """(Ne+1,) 各控制体界面的坐标（包括负极-集流体界面、正极集流体界面）"""
        return hstack([0, self.Δx_.cumsum()])

    @property
    def OCV(self):
        """开路电压 [V]"""
        OCPpos = self.solve_UOCPpos_(self.θspos)
        OCPneg = self.solve_UOCPneg_(self.θsneg)
        return OCPpos - OCPneg

    @property
    def SOC(self):
        return (self.θsneg - self.θminneg)/(self.θmaxneg - self.θminneg)

    @property
    def φeneg_(self):
        """(Nneg,) 负极区域电解液电势 [V]"""
        return self.φe_[:self.Nneg]

    @property
    def φepos_(self):
        """(Npos,) 正极区域电解液电势 [V]"""
        return self.φe_[-self.Npos:]

    Qohme: float
    Qohmneg: float; Qohmpos: float
    Qrxnneg: float; Qrxnpos: float
    Qrevneg: float; Qrevpos: float
    @property
    def Qgen(self):
        """总产热量 [W]"""
        return (  self.Qohme + self.Qohmneg + self.Qohmpos  # 总欧姆热 [W]
                + self.Qrxnneg + self.Qrxnpos   # 总反应热 [W]
                + self.Qrevneg + self.Qrevpos)  # 总可逆热 [W]

    def solve_UOCPneg_(self, θsneg_: float | ndarray):
        """求解负极开路电位场 [V]"""
        UOCPneg_ = self.UOCPneg(θsneg_)
        if ΔT := (self.T - self.Tref):
            if callable(self.dUOCPdTneg):
                UOCPneg_ += ΔT*self.dUOCPdTneg(θsneg_)
            else:
                UOCPneg_ += ΔT*self.dUOCPdTneg
        return UOCPneg_

    def solve_UOCPpos_(self, θspos_: float | ndarray):
        """求解正极开路电位场 [V]"""
        UOCPpos_ = self.UOCPpos(θspos_)
        if ΔT := (self.T - self.Tref):
            if callable(self.dUOCPdTpos):
                UOCPpos_ += ΔT*self.dUOCPdTpos(θspos_)
            else:
                UOCPpos_ += ΔT*self.dUOCPdTpos
        return UOCPpos_

    @staticmethod
    def generate_x_related_coordinates(
            Nneg: int, Nsep: int, Npos: int,        # 负极、隔膜、正极网格数
            Lneg: float, Lsep: float, Lpos: float,  # 负极、隔膜、正极厚度 [m]/[–]
            ):
        """生成厚度方向x相关坐标"""
        Δxneg = Lneg/Nneg  # 负极网格厚度
        Δxsep = Lsep/Nsep  # 隔膜网格厚度
        Δxpos = Lpos/Npos  # 正极网格厚度
        x_ = hstack([
            linspace(0, Lneg, Nneg + 1)[:-1] + Δxneg/2,
            linspace(Lneg, Lneg + Lsep, Nsep + 1)[:-1] + Δxsep/2,
            linspace(Lneg + Lsep, Lneg + Lsep + Lpos, Npos + 1)[:-1] + Δxpos/2,
            ])  # (Ne,) 全区域各控制体中心坐标
        Δx_ = concatenate([full(Nneg, Δxneg),
                           full(Nsep, Δxsep),
                           full(Npos, Δxpos)])  # (Ne,) 全区域控制体厚度
        ΔxWest_ = hstack([
            full(Nneg, Δxneg),
            (Δxneg + Δxsep)/2, full(Nsep - 1, Δxsep),
            (Δxsep + Δxpos)/2, full(Npos - 1, Δxpos)])  # (Ne,) 当前控制体中心到左侧控制体中心的距离
        ΔxEast_ = hstack([
            full(Nneg - 1, Δxneg), (Δxneg + Δxsep)/2,
            full(Nsep - 1, Δxsep), (Δxsep + Δxpos)/2,
            full(Npos, Δxpos)])                         # (Ne,) 当前控制体中心到右侧控制体中心的距离
        return x_, Δx_, ΔxWest_, ΔxEast_

    @staticmethod
    def generate_r_related_coordinates(Nr, Rs, radialDiscretization='EV'):
        """生成颗粒径向r相关坐标"""
        a = 4/3 * 3.141592653589793
        Rs3 = 1 if Rs==1 else (Rs*Rs*Rs)
        match radialDiscretization:
            case 'EV':
                # 等体积划分球壳网格
                V = a*Rs3  # 颗粒体积 [m^3]/[–]
                ΔV = V/Nr  # 球壳控制体体积 [m^3]/[–]
                rW_ = (ΔV*arange(0, Nr)/a)**(1/3)      # (Nr,) 球壳内界面坐标 [m]/[–]
                rE_ = (ΔV*arange(1, Nr + 1)/a)**(1/3)  # (Nr,) 球壳外界面坐标 [m]/[–]
                r_ = (rW_ + rE_)/2  # (Nr,) 颗粒径向控制体中心的坐标 [m]/[–]
                Δr_ = rE_ - rW_     # (Nr,) 颗粒球壳网格厚度序列 [m]/[–]
            case 'EI':
                # 等间隔划分球壳网格
                Δr = Rs/Nr                                # 颗粒球壳网格厚度 [m]/[–]
                r_ = linspace(0, Rs, Nr + 1)[:-1] + Δr/2  # (Nr,) 颗粒径向控制体中心的坐标 [m]/[–]
                Δr_ = full(Nr, Δr)                        # (Nr,) 颗粒球壳网格厚度序列 [m]/[–]
        Vr_ = ((r_ + Δr_/2)**3
             - (r_ - Δr_/2)**3)/Rs3  # (Nr,) 球壳体积分数序列 [–]
        return r_, Δr_, Vr_

    @staticmethod
    def solve_banded_matrix(
            A__: ndarray,                  # (N, N) 矩阵
            b_: ndarray,                   # (N,) A__ @ X_ = b_
            idxreordered_: Sequence[int],  # (N,) 索引：重排，使矩阵A__带状化
            idxrecovered_: Sequence[int],  # (N,) 索引：恢复排序
            bandwidths_: dict[str, ndarray]):
        """解带状化矩阵"""
        N = A__.shape[0]  # 因变量总数
        u, l = bandwidths_['upper'], bandwidths_['lower']  # 上下带宽
        Areordered__ = A__[ix_(idxreordered_, idxreordered_)]       # 重新排列矩阵A__，使之带状化
        ab__ = zeros((2*l + u + 1, N), dtype=A__.dtype, order='F')  # 适合dgbsv
        band__ = ab__[l:, :]   # (u + l + 1, N) 矩阵A__的带
        diag = Areordered__.diagonal
        for row, offset in enumerate(range(u, -l - 1, -1)):
            d_ = diag(offset)  # 提取矩阵Areordered__的带
            start = max(0, offset)
            end   = min(N, N + offset)
            band__[row, start:end] = d_
        X_ = dgbsv(l, u, ab__, asfortranarray(b_[idxreordered_]), True, True)[2]
        # X_ = solve_banded((l, u), band__, asfortranarray(b_[idxReordered_]), True, True, False)
        return X_[idxrecovered_]

    @staticmethod
    def identify_bandwidths(A__: ndarray):
        """辨识矩阵的上、下带宽"""
        N = A__.shape[0]
        diag = A__.diagonal
        bandwidths_ = {}
        for offset in range(-N + 1, 0):
            # 从矩阵左下角开始遍历对角线
            if any(diag(offset)!=0):
                bandwidths_['lower'] = abs(offset)  # 下带宽
                break
        for offset in range(N - 1, 0, -1):
            # 从矩阵右上角开始遍历对角线
            if any(diag(offset)!=0):
                bandwidths_['upper'] = offset       # 上带宽
                break
        return bandwidths_

    def Arrhenius(self,
                  X: float | None,  # 参考温度下的参数值
                  E: float,          # 活化能 [J/mol]
                  ):
        """Arrhenius温度修正"""
        if self.constants or self.T==self.Tref or X is None:
            return X
        return X * exp(E/P2Dbase.R*(1/self.Tref - 1/self.T))

    @staticmethod
    def generate_solve_dUOCPdθs_(
            UOCP: Callable,  # 开路电位函数，输入嵌锂状态θ [–]，输出开路电位U [V]
            ) -> Callable:
        """生成开路电位UOCP对嵌锂状态θs的导数插值函数dUOCPdθs"""
        if isinstance(UOCP, interp1d):
            # 若开路电位函数本身为scipy插值函数
            θs_ = UOCP.x
            UOCP_ = UOCP.y
        else:
            θs_ = hstack([0.001, arange(0.01, 1, 0.01), 0.999])  # 嵌锂状态序列 [–]
            UOCP_ = UOCP(θs_)                                                # 开路电位序列 [V]
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
    def get_color(s_: Sequence | int, n: int, cmap='viridis'):
        """返回viridis颜色"""
        if isinstance(s_, Iterable):
            N = len(s_)
        elif isscalar(s_):
            N = int(s_)
        color_ = plt.get_cmap(cmap)(int(linspace(0, 255, N)[n]))[:3]  # (3,)
        return color_

    @staticmethod
    def solve_4θ(UOCPneg: Callable,
                 UOCPpos: Callable,
                 Qcell: float | int,
                 Qneg: float | int,
                 Qpos: float | int,
                 Umin: float  | int,
                 Umax: float | int, ):
        """已知Qcell、Qneg、Qpos，求4个边界嵌锂状态θminneg、θmaxneg、θminpos、θmaxpos"""
        Qcell, Qneg, Qpos = Qcell, Qneg, Qpos
        def function(X_: ndarray) -> ndarray:
            θminneg, θmaxneg, θminpos, θmaxpos = X_
            F_ = array([
                UOCPpos(θmaxpos) - UOCPneg(θminneg) - Umin,  # [V]
                UOCPpos(θminpos) - UOCPneg(θmaxneg) - Umax,  # [V]
                θmaxpos - θminpos - Qcell/Qpos,        # [–]
                θmaxneg - θminneg - Qcell/Qneg,])      # [–]
            return F_
        x0_ = array([0.025, 0.85, 0.15, 0.92])  # 迭代初值
        result = root(function, x0=x0_, method='hybr', tol=1e-8)
        ΔFmax = abs(function(result.x)).max()
        θminneg, θmaxneg, θminpos, θmaxpos = result.x
        return θminneg, θmaxneg, θminpos, θmaxpos, ΔFmax

    def __str__(self):
        K__ = self.K__
        Qcell, Qneg, Qpos = self.Qcell, self.Qneg, self.Qpos
        θminneg, θmaxneg, θminpos, θmaxpos = self.θminneg, self.θmaxneg, self.θminpos, self.θmaxpos
        lithiumPlating = self.lithiumPlating
        doubleLayerEffect = self.doubleLayerEffect
        timeDiscretization = self.timeDiscretization
        radialDiscretization = self.radialDiscretization
        decouple_cs = self.decouple_cs
        θsneg, θspos = self.θsneg, self.θspos
        OCV, U, tC, SOC = self.OCV, self.U, self.T - 273.15, self.SOC
        return (
            f'时域因变量线性矩阵 {K__.shape = }\n'
            f'全电池理论可用容量：{Qcell = :.3f} Ah\n'
            f'电极容量：{Qneg = :.3f} Ah, {Qpos = :.3f} Ah\n'
            f'电极可用容量：{Qneg*(θmaxneg - θminneg) = :.3f} Ah, {Qpos*(θmaxpos - θminpos) = :.3f} Ah\n'
            f'{θminneg = :.4f}, {θmaxneg = :.4f}\n'
            f'{θminpos = :.4f}, {θmaxpos = :.4f}\n'
            f'析锂反应：{lithiumPlating = }\n'
            f'双电层效应：{doubleLayerEffect = }\n'
            f'时间离散：{timeDiscretization = }\n'
            f'球形固相颗粒径向离散：{radialDiscretization = }\n'
            f'固相锂离子浓度求解：{decouple_cs = }\n'
            f'当前电极嵌锂状态：{θsneg = :.3f}, {θspos = :.3f}\n'
            f'当前开路电压{OCV = :.3f} V, 端电压{U = :.3f} V, 温度{tC: .1f} °C, {SOC = :.3f}'
            )

    class Error(Exception):
        """P2D模型专属异常类"""
        def __init__(self, information: str, *args):
            super().__init__(information, *args)

    def interpolate(self,
            variableName: str,           # 字符串：所需插值的变量名
            t_: Sequence,                # 时刻序列 [s]
            x_: Sequence | None = None,  # 厚度方向坐标序列 [m]
            r_: Sequence | None = None,  # 球形颗粒半径方向坐标序列 [m]
            ) -> ndarray:
        data = self.data
        assert variableName in data.keys(), f"无法识别所输入的变量名'{variableName}'，变量名variableName应属于：{data.keys()}"
        kw = dict(bounds_error=False,  # 超出边界不报错
                  fill_value=None)     # None表示外推
        if variableName.endswith('__'):
            # 与厚度方向坐标x、球形颗粒半径方向坐标r、时间t相关的变量，即：'csneg__' 'cspos__' 'θsneg__' 'θspos__'
            electrode = variableName[2:5]  # 提取'neg'或'pos'
            interpolator = RegularGridInterpolator([data['t'], getattr(self, f'r{electrode}_'), getattr(self, f'x{electrode}_')],
                                                    data[variableName], **kw)  # 时间序列：固相颗粒锂离子浓度
            points____ = stack(meshgrid(t_, r_, x_, indexing='ij'), axis=-1)  # (len(t_), len(r_), len(x_), 3) 待插值点
            v___ = interpolator(points____)  # 插值
            return v___
        elif variableName.endswith('_'):
            # 与厚度方向坐标x、时间t相关的变量
            if ('neg' in variableName) or ('LP' in variableName):
                location_ = self.xneg_
            elif 'pos' in variableName:
                location_ = self.xpos_
            else:
                location_ = self.x_
            interpolator = RegularGridInterpolator([data['t'], location_], data[variableName], **kw)
            points___ = stack(meshgrid(t_, x_, indexing='ij'), axis=-1)  # (len(t_), len(x_), 2) 待插值点
            v__ = interpolator(points___)  # 呈时间序列(axis0)、厚度方向坐标序列(axis1)的变量
            return v__
        else:
            # 仅与时间t相关的变量
            v_ = interp1d(data['t'], data[variableName],
                          bounds_error=False,
                          fill_value='extrapolate',
                          )(t_)  # 呈时间序列的变量
            return v_

    def plot_UI(self,
                t_: Sequence | None = None,  # 时刻序列
                ):
        """端电压、电流-时间"""
        if t_ is None:
            t_ = self.data['t']
        U_ = self.interpolate('U', t_)  # 呈时间序列的电压
        I_ = self.interpolate('I', t_)  # 呈时间序列的电流

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .85, .375])
        ax2.set_position([.1, .08, .85, .375])

        ax1.plot(t_, U_, 'b-')
        ax1.set_ylabel(r'Terminal voltage ${\it U}({\it t})$ [V]')
        ax1.set_ylim([2.4, 4.5])
        ax1.set_yticks(arange(2.4, 4.5 + 1e-6, 0.3))

        ax2.plot(t_, I_, 'r-')
        ax2.set_ylabel('Current ${\it I}({\it t})$ [A]')

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
        """瞬时产热量、温度-时间"""
        if t_ is None:
            t_ = self.data['t']
        Qgen_ = self.interpolate('Qgen', t_)  # 呈时间序列的瞬时产热量
        T_ = self.interpolate('T', t_)        # 呈时间序列的瞬时产热量

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
        SOC_ = self.interpolate('SOC', t_)      # 呈时间序列的全电池荷电状态
        θsneg_ = self.interpolate('θsneg', t_)  # 呈时间序列的负极嵌锂状态
        θspos_ = self.interpolate('θspos', t_)  # 呈时间序列的正极嵌锂状态

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        ax.plot(t_, θsneg_, '--')
        ax.plot(t_, θspos_, 'r--')
        ax.plot(t_, SOC_, 'k-')
        ax.set_ylim([0, 1])  # SOC范围
        ax.set_yticks(arange(0, 1.01, 0.1))  # SOC范围
        ax.set_ylabel(r'State-of-state or degree-of-lithiation [–]')
        ax.set_xlabel(r'Time $\it t$ [s]')
        duration = t_[-1] - t_[0]
        ax.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax.legend([r'Negative electrode degree-of-lithiation ${\it θ}_{s,neg}({\it t})$',
                   r'Positive electrode degree-of-lithiation ${\it θ}_{s,pos}({\it t})$',
                   r'Full cell state-of-state ${\it SOC}({\it t})$'])
        ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_c(self,
               t_: Sequence | None = None,  # 时刻序列
               ):
        """正负极固相颗粒中心、固相颗粒表面、电解液锂离子浓度-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        cθ = 'c' if self.cUnit else 'θ'
        csnegcent___ = self.interpolate(f'{cθ}sneg__', t_=t_, x_=self.xneg_, r_=[0])  # 呈时间序列的负极固相颗粒中心锂离子浓度
        csposcent___ = self.interpolate(f'{cθ}spos__', t_=t_, x_=self.xpos_, r_=[0])  # 呈时间序列的正极固相颗粒中心锂离子浓度
        csnegsurf__ = self.interpolate(f'{cθ}snegsurf_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极固相颗粒表面锂离子浓度
        cspossurf__ = self.interpolate(f'{cθ}spossurf_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极固相颗粒表面锂离子浓度
        ce__ = self.interpolate(f'{cθ}e_', t_=t_, x_=self.x_)  # 呈时间序列的电解液锂离子浓度场

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
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{{self.rSign[1:-1]}=0}}$ [{self.cUnit or '–'}]')
        ax1.legend(bbox_to_anchor=(1, 1))

        ax2.set_title('Lithium concentration at electrode particle surface', fontsize=12)
        for n, (csnegsurf_, cspossurf_, t) in enumerate(zip(csnegsurf__, cspossurf__, t_)):
            x_ = self.xPlot_
            y_ = *csnegsurf_, *[nan]*self.Nsep, *cspossurf_
            ax2.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.rSign[1:-1]} = {"{\\it R}_{s,reg}" if self.rUnit else 1} }}$ [{self.cUnit or '–'}]')

        ax3.set_title('Lithium-ion concentration in electrolyte', fontsize=12)
        for n, (ce_, t) in enumerate(zip(ce__, t_)):
            x_ = [0, *self.xPlot_, self.xInterfacesPlot_[-1]]
            y_ = [ce_[0], *ce_, ce_[-1]]
            ax3.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax3.set_ylabel(rf'{self.cSign}$_e$({self.xSign}, {self.tSign}) [{self.cUnit or '–'}]')

        self.plot_interfaces(ax1, ax2, ax3)
        plt.show()

    def plot_φ(self,
               t_: Sequence | None = None,  # 时刻序列
               ):
        """固液相电势-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        φsneg__ = self.interpolate('φsneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极固相电势场 [V]
        φspos__ = self.interpolate('φspos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极固相电势场 [V]
        φe__ = self.interpolate('φe_', t_=t_, x_=self.x_)           # 呈时间序列的电解液电势场 [V]
        I_ = self.interpolate('I', t_=t_)      # 呈时间序列的电流 [A]
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
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
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
            ax2.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(ax1.get_ylabel().replace('neg', 'pos'))
        ax2.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
        ax2.set_xlim(x_[0], x_[-1])
        ax2.grid(axis='y', linestyle='--')
        ax2.legend(bbox_to_anchor=[1, 1])

        for n, (φe_, t) in enumerate(zip(φe__, t_)):
            ax3.plot([0, *self.xPlot_, self.xInterfacesPlot_[-1]],
                     hstack([φe_[0], φe_, φe_[-1]]), 'o-', color=P2Dbase.get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(r'Electrolyte potential ${\it φ}_e$' + f'({self.xSign}, {self.tSign}) [mV]')

        self.plot_interfaces(ax3)
        plt.show()

    def plot_jint_i0int_ηint(self,
                             t_: Sequence | None = None,  # 时刻序列
                             ):
        """主反应局部体积电流密度、交换电流密度、过电位-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        jJ, iI = ['j', 'i'] if self.xUnit else ['J', 'I']
        jintneg__ = self.interpolate(f'{jJ}intneg_', t_=t_, x_=self.xneg_)    # 呈时间序列的负极局部体积电流密度场
        jintpos__ = self.interpolate(f'{jJ}intpos_', t_=t_, x_=self.xpos_)    # 呈时间序列的正极局部体积电流密度场
        i0intneg__ = self.interpolate(f'{iI}0intneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极交换电流密度场
        i0intpos__ = self.interpolate(f'{iI}0intpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极交换电流密度场
        ηintneg__ = self.interpolate('ηintneg_', t_=t_, x_=self.xneg_)*1e3    # 呈时间序列的负极固相表面过电位场 [mV]
        ηintpos__ = self.interpolate('ηintpos_', t_=t_, x_=self.xpos_)*1e3    # 呈时间序列的正极固相表面过电位场 [mV]

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
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(rf'{self.jSign}$_{{int}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        ax1.legend(bbox_to_anchor=[1, 1])

        ax2.set_title('Field of lithium (de-)intercalation exchange current density', fontsize=12)
        for n, (i0intneg_, i0intpos_, t) in enumerate(zip(i0intneg__, i0intpos__, t_)):
            x_ = self.xPlot_
            y_ = *i0intneg_, *[nan]*self.Nsep, *i0intpos_
            ax2.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(rf'{self.i0Sign}({self.xSign}, {self.tSign}) [{self.i0Unit}]')

        ax3.set_title('Field of lithium (de-)intercalation overpotential', fontsize=12)
        for n, (ηintneg_, ηintpos_, t) in enumerate(zip(ηintneg__, ηintpos__, t_)):
            x_ = self.xPlot_
            y_ = *ηintneg_, *[nan]*self.Nsep, *ηintpos_
            ax3.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(rf'${{\it η}}_{{int}}$ ({self.xSign}, {self.tSign}) [mV]')

        self.plot_interfaces(ax1, ax2, ax3)
        plt.show()

    def plot_jDL(self,
                 t_: Sequence | None = None,  # 时刻序列
                 ):
        """双电层效应局部体积电流密度"""
        if t_ is None:
            t_ = self.data['t']
        jJ = 'j' if self.xUnit else 'J'
        jDLneg__ = self.interpolate(f'{jJ}DLneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的双电层效应负极局部体积电流密度场
        jDLpos__ = self.interpolate(f'{jJ}DLpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的双电层效应正极局部体积电流密度场

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        x_ = self.xPlot_
        for n, (jDLneg_, jDLpos_, t) in enumerate(zip(jDLneg__, jDLpos__, t_)):
            y_ = *jDLneg_, *[nan]*self.Nsep, *jDLpos_
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'Double-layer local volumetric\ncurrent density {self.jSign}$_{{DL}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        self.plot_interfaces(ax1)
        ax1.legend(bbox_to_anchor=[1, 1])

        t_ = self.data['t']
        jDLneg__ = self.interpolate(f'{jJ}DLneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的双电层效应负极局部体积电流密度场
        jDLpos__ = self.interpolate(f'{jJ}DLpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的双电层效应正极局部体积电流密度场
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
        cs___ = self.interpolate(f'{cθ}s{reg}__', t_=t_, x_=[x], r_=r_)  # 呈时间序列的x位置负极固相颗粒锂离子浓度
        cssurf__ = self.interpolate(f'{cθ}s{reg}surf_', t_=t_, x_=[x])   # 呈时间序列的x位置负极固相颗粒表面锂离子浓度

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        ax.set_position([.1, .08, .75, 0.8])
        ax.set_title(rf'Lithium concentration in electrode particale at {self.xSign} = {x if LPmodel else x*1e6:g} {self.xUnit}', fontsize=12)
        X_ = array([0, *r_, 1 if LPmodel else getattr(self, f'Rs{reg}')])
        if not LPmodel:
            X_ *= 1e6
        for n, (cs__, cssurf_, t) in enumerate(zip(cs___, cssurf__, t_)):
            ax.plot(X_,
                    hstack([cs__.ravel()[0], cs__.ravel(), cssurf_.ravel()]),
                    'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax.legend(bbox_to_anchor=[1, 1])
        ax.set_ylabel(rf'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.xSign[1:-1]}={x if LPmodel else x*1e6:g}\;{self.xUnit} }}$ [{self.cUnit or '–'}]')
        ax.set_xlabel(rf'Radial location {self.rSign} [{self.rUnit or '–'}]')
        ax.set_xlim(X_[0], X_[-1])
        ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_jLP_ηLP(self,
                     t_: Sequence | None = None,  # 时刻序列
                     ):
        """负极析锂局部体积电流密度-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        jJ = 'j' if self.xUnit else 'J'
        jLP__ = self.interpolate(f'{jJ}LP_', t_=t_, x_=self.xneg_) if self.lithiumPlating else zeros((len(t_), self.Nneg))   # 呈时间序列的负极析锂局部体积电流密度场
        ηLPneg__ = self.interpolate('ηLPneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极析锂反应过电位场

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (jLP_, t) in enumerate(zip(jLP__, t_)):
            ax1.plot(self.xPlot_[:self.Nneg], jLP_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'Lithium plating local volumetric\ncurrent density {self.jSign}$_{{LP}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (ηLPneg_, t) in enumerate(zip(ηLPneg__, t_)):
            ax2.plot(self.xPlot_[:self.Nneg], ηLPneg_*1e3, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(rf'Lithium plating overpotential ${{\it η}}_{{LP}}$({self.xSign}, {self.tSign}) [mV]')
        for ax in [ax1, ax2]:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit}]')
            ax.set_xlim(0, self.xInterfacesPlot_[self.Nneg])
            ax.grid(axis='y', linestyle='--')

        plt.show()

    def plot_LP(self,
                t_: Sequence | None = None,  # 时刻序列
                ):
        """析锂反应过电位-时间"""
        if t_ is None:
            t_ = self.data['t']
        jJ = 'j' if self.xUnit else 'J'
        jLP__ = self.interpolate(f'{jJ}LP_', t_=t_, x_=self.xneg_) if self.lithiumPlating else zeros((len(t_), self.Nneg))  # 呈时间序列的负极析锂局部体积电流密度场
        ηLP_ = self.interpolate('ηLPneg_', t_=t_, x_=self.xneg_[[-1]])  # 呈时间序列的析锂反应过电位
        I_ = self.interpolate('I', t_=t_)  # 呈时间序列的电流
        I_[I_==0] = nan

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .85, .375])
        ax2.set_position([.1, .08, .85, .375])

        ax1.plot(t_, ηLP_*1e3, '-')
        ylabel = 'Lithium plating\noverpotential ${\\it η}_{LP}$'\
               + f'({self.xSign}, {self.tSign})|$_{{ {self.xSign[1:-1]}={  "{\\it L}_{neg}" if self.xUnit else 1} }}$ [mV]'
        ax1.set_ylabel(ylabel)
        if self.xUnit:
            A = getattr(self, 'A')
            ratio_ = jLP__.sum(axis=1)*self.Δxneg*A/I_  # 析锂电流与总电流之比
        else:
            ratio_ = jLP__.mean(axis=1)/I_  # 析锂电流与总电流之比
        ratio_[isnan(ratio_)] = 0
        ax2.plot(t_, ratio_, '-')
        ax2.set_ylabel('Ratio of lithium plating current\n' + r'to total current ${\it I}_{LP}/{\it I} [–]$')

        duration = max(t_) - min(t_)
        for ax in [ax1, ax2]:
            ax.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
            ax.set_xlabel(f'Time {self.tSign} [s]')
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
        SOC_ = arange(0, 1.001, 0.01)                           # 全电池SOC
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

    def plot_interfaces(self, *axes_):
        for ax in axes_:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')  # 横坐标标签
            ax.set_ylim(ax.get_ylim())                   # 固定纵坐标上下限
            ax.set_xlim(self.xInterfacesPlot_[[0, -1]])  # 横坐标上下限
            ax.vlines(self.xInterfacesPlot_[self.Nneg], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                      ls='--', color=[.5, .5, .5],
                      alpha=0.5)  # 负极-隔膜界面
            ax.vlines(self.xInterfacesPlot_[self.Nneg + self.Nsep], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                      ls='--', color=[.5, .5, .5],
                      alpha=0.5)  # 隔膜-正极界面
            ax.grid(axis='y', linestyle='--')  # 纵坐标网格线
    
    def save(self,
            path: str | None = None,  # 路径\文件名
            datanames_: Sequence[str] = None,  # 仅保存self.data当中指定的运行数据
            **otherData_,  # 其它需保存的数据集
            ):
        """保存数据"""
        assert isinstance(otherData_, dict) or otherData_ is None, 'otherData_应为字典或None'
        if path is None:
            # 默认保存路径
            filepath = pathlib.Path().cwd()  # 获得当前模块所在路径
            path = filepath.joinpath(f'{time.strftime("%Y%m%d %H%M%S", time.localtime())}保存{self.__class__.__name__}模拟数据.npz')

        # 保存运行数据
        if datanames_:
            # 若给定了特定数据名，则只保存datanames_所列出的运行数据
            dataSaved_ = {}
            for dataname in self.data:
                if dataname in datanames_:
                    dataSaved_[dataname] = self.data[dataname]
        else:
            dataSaved_ = self.data

        savez(path, **(dataSaved_ | otherData_))  # 保存
        print(f'已保存{path}')
        return path

    @abstractmethod
    def step(self, Δt):
        """时间步进：Newton法迭代因变量"""
        pass

    @property
    @abstractmethod
    def Qcell(self):
        """全电池理论可用容量 [Ah]"""
        pass

    @property
    @abstractmethod
    def Qneg(self):
        """负极容量 [Ah]"""
        pass

    @property
    @abstractmethod
    def Qpos(self):
        """正极容量 [Ah]"""
        pass

    @property
    @abstractmethod
    def U(self):
        """正负极端电压 [V]"""
        pass

    @property
    @abstractmethod
    def θsneg(self):
        """负极嵌锂状态"""
        pass

    @property
    @abstractmethod
    def θspos(self):
        """正极嵌锂状态"""
        pass

    @property
    @abstractmethod
    def ILP(self):
        """析锂反应电流 [A]"""
        pass

    @property
    @abstractmethod
    def xPlot_(self):
        """全区域控制体中心的坐标（用于作图） """
        pass

    @property
    @abstractmethod
    def xInterfacesPlot_(self):
        """各控制体交界面的坐标（用于作图） """
        pass

if __name__=='__main__':
    pass
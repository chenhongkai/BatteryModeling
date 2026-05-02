#%%
from typing import Sequence, Callable
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from numpy import nan, ndarray,\
    asarray, arange, full, stack, hstack, concatenate, \
    empty, meshgrid, logspace,\
    ptp, log10,\
    isscalar

from P2Dmodel.P2Dbase import P2Dbase
get_color = P2Dbase.get_color

class JTFbase(ABC):
    """抽象类：锂离子电池时频联合准二维模型 Joint Time-Frequency Pseudo-two-Dimension model"""

    ## 类型注解 ##
    Nneg: int; Nsep: int; Npos: int; Ne: int  # 负极、隔膜、正极全区域网格数
    lithiumPlating: bool
    complete: bool

    _i0intneg: float | None  # 负极主反应交换电流密度
    _i0intpos: float | None  # 正极主反应交换电流密度
    l: float                 # 等效电感 [H]

    Kf__: ndarray            # (N, N) 频域因变量线性矩阵
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

    # 索引频域因变量
    idxREcsnegsurf_: ndarray; idxIMcsnegsurf_: ndarray
    idxREcspossurf_: ndarray; idxIMcspossurf_: ndarray
    idxREce_: ndarray; idxIMce_: ndarray
    idxREφsneg_: ndarray; idxIMφsneg_: ndarray
    idxREφspos_: ndarray; idxIMφspos_: ndarray
    idxREφe_: ndarray; idxIMφe_: ndarray
    idxREjintneg_: ndarray; idxIMjintneg_: ndarray
    idxREjintpos_: ndarray; idxIMjintpos_: ndarray
    idxREjDLneg_: ndarray; idxIMjDLneg_: ndarray
    idxREjDLpos_: ndarray; idxIMjDLpos_: ndarray
    idxREjLP_: ndarray; idxIMjLP_: ndarray
    idxREi0intneg_: ndarray; idxIMi0intneg_: ndarray
    idxREi0intpos_: ndarray; idxIMi0intpos_: ndarray
    idxREηintneg_: ndarray; idxIMηintneg_: ndarray
    idxREηintpos_: ndarray; idxIMηintpos_: ndarray
    idxREηLP_: ndarray; idxIMηLP_: ndarray

    def __init__(self,
            f_: Sequence[float] = logspace(3, -1, 26),  # 频率序列 [Hz]
            ):
        self.f_ = f_ = asarray(f_); assert f_.ndim==1, f'频率序列f_应可转化为ndim==1的ndarray，当前{f_ = }'
        Nf, Nneg, Npos, Ne = f_.size, self.Nneg, self.Npos, self.Ne  # 读取：网格数
        lithiumPlating, complete = self.lithiumPlating, self.complete  # 读取：模式
        # 状态量
        self.tEIS: float = None                      # 计算阻抗的时刻 [s]
        self.Z_: ndarray = empty(Nf, dtype=complex)  # 全电池阻抗谱 [Ω]
        self.bandwidthsKf_: dict[str, int] = None    # Kf__矩阵上下带宽
        self.idxKfReordered_: ndarray = None  # 索引：重排Kf__矩阵
        self.idxKfRecovered_: ndarray = None  # 索引：恢复排序Kf__矩阵
        if complete:
            self.Zneg_, self.Zpos_ = empty(Nf, dtype=complex), empty(Nf, dtype=complex)  # 负极、正极阻抗谱 [Ω]
            self.REφsneg__, self.IMφsneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))        # 负极固相电势实部、虚部
            self.REφspos__, self.IMφspos__ = empty((Nf, Npos)), empty((Nf, Npos))        # 正极固相电势实部、虚部
            self.REφe__, self.IMφe__ = empty((Nf, Ne)), empty((Nf, Ne))                  # 电解液电势实部、虚部
            self.REηintneg__, self.IMηintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极主反应过电位实部、虚部
            self.REηintpos__, self.IMηintpos__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极主反应过电位实部、虚部
            if lithiumPlating:
                self.REηLP__, self.IMηLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极析锂反应过电位实部、虚部
        # 恒定量
        self.ΔIAC = 1.  # 交流扰动电流振幅 [A]
        self.frequency_dependent_cache = self.solve_frequency_dependent_variables()  # 频率相关变量缓存
        self.EISdatanames_ = ['tEIS', 'Z_']  # 需记录的阻抗数据名称

    def generate_indices_of_frequency_domain_dependent_variables(self) -> int:
        """生成频域因变量索引"""
        Nneg, Npos, Ne = self.Nneg, self.Npos, self.Ne  # 读取：网格数
        N = 0  # 全局索引游标
        def allocate(n):
            # 分配索引
            nonlocal N
            idx_ = arange(N, N + n)
            N += n
            return idx_
        self.idxREcsnegsurf_ = allocate(Nneg)  # 索引：负极固相表面浓度实部
        self.idxIMcsnegsurf_ = allocate(Nneg)  # 索引：负极固相表面浓度虚部
        self.idxREcspossurf_ = allocate(Npos)  # 索引：正极固相表面浓度实部
        self.idxIMcspossurf_ = allocate(Npos)  # 索引：正极固相表面浓度虚部
        self.idxREce_ = allocate(Ne)          # 索引：电解液锂离子浓度实部
        self.idxIMce_ = allocate(Ne)          # 索引：电解液锂离子浓度虚部
        self.idxREφsneg_ = allocate(Nneg)     # 索引：负极固相电势实部
        self.idxIMφsneg_ = allocate(Nneg)     # 索引：负极固相电势虚部
        self.idxREφspos_ = allocate(Npos)     # 索引：正极固相电势实部
        self.idxIMφspos_ = allocate(Npos)     # 索引：正极固相电势虚部
        self.idxREφe_ = allocate(Ne)          # 索引：电解液电势实部
        self.idxIMφe_ = allocate(Ne)          # 索引：电解液电势虚部
        self.idxREjintneg_ = allocate(Nneg)   # 索引：负极主反应局部体积电流密度实部
        self.idxIMjintneg_ = allocate(Nneg)   # 索引：负极主反应局部体积电流密度虚部
        self.idxREjintpos_ = allocate(Npos)   # 索引：正极主反应局部体积电流密度实部
        self.idxIMjintpos_ = allocate(Npos)   # 索引：正极主反应局部体积电流密度虚部
        self.idxREjDLneg_ = allocate(Nneg)    # 索引：负极双电层局部体积电流密度实部
        self.idxIMjDLneg_ = allocate(Nneg)    # 索引：负极双电层局部体积电流密度虚部
        self.idxREjDLpos_ = allocate(Npos)    # 索引：正极双电层局部体积电流密度实部
        self.idxIMjDLpos_ = allocate(Npos)    # 索引：正极双电层局部体积电流密度虚部
        Ni0intneg = Nneg if self._i0intneg is None else 0
        Ni0intpos = Npos if self._i0intpos is None else 0
        self.idxREi0intneg_ = allocate(Ni0intneg)  # 索引：负极主反应交换电流密度实部
        self.idxIMi0intneg_ = allocate(Ni0intneg)  # 索引：负极主反应交换电流密度虚部
        self.idxREi0intpos_ = allocate(Ni0intpos)  # 索引：正极主反应交换电流密度实部
        self.idxIMi0intpos_ = allocate(Ni0intpos)  # 索引：正极主反应交换电流密度虚部
        self.idxREηintneg_ = allocate(Nneg)   # 索引：负极过电位实部
        self.idxIMηintneg_ = allocate(Nneg)   # 索引：负极过电位虚部
        self.idxREηintpos_ = allocate(Npos)   # 索引：正极过电位实部
        self.idxIMηintpos_ = allocate(Npos)   # 索引：正极过电位虚部
        NLP = Nneg if self.lithiumPlating else 0
        self.idxREjLP_ = allocate(NLP)  # 索引：析锂反应电流密度实部
        self.idxIMjLP_ = allocate(NLP)  # 索引：正极交换电流密度虚部
        self.idxREηLP_ = allocate(NLP)  # 索引：析锂反应过电位实部
        self.idxIMηLP_ = allocate(NLP)  # 索引：析锂反应过电位虚部
        return N  # 频域因变量总数

    def assign_Kf__with_constants(self):
        # 对Kf__矩阵赋恒定值
        Nneg, Npos = self.Nneg, self.Npos
        Kf__ = self.Kf__

        idx_ = concatenate([self.idxREcsnegsurf_, self.idxIMcsnegsurf_, self.idxREcspossurf_, self.idxIMcspossurf_])
        Kf__[idx_, idx_] = 1  # 负极、正极固相表面浓度实部REcssurf行REcssurf列、虚部IMcssurf行IMcssurf列

        # 负极固相电势实部REφsneg行、虚部IMφsneg行
        idxREφsneg_ = self.idxREφsneg_
        idxIMφsneg_ = self.idxIMφsneg_
        Kf__[idxREφsneg_[1:], idxREφsneg_[:-1]] = 1                     # REφsneg列下对角线
        Kf__[idxREφsneg_[:-1], idxREφsneg_[1:]] = 1                     # REφsneg列上对角线
        Kf__[idxREφsneg_, idxREφsneg_] = [-1] + [-2]*(Nneg - 2) + [-1]  # REφsneg列主对角线
        Kf__[idxIMφsneg_[1:], idxIMφsneg_[:-1]] = 1                     # IMφsneg列下对角线
        Kf__[idxIMφsneg_[:-1], idxIMφsneg_[1:]] = 1                     # IMφsneg列上对角线
        Kf__[idxIMφsneg_, idxIMφsneg_] = [-1] + [-2]*(Nneg - 2) + [-1]  # IMφsneg列主对角线
        # 正极固相电势实部REφspos行、虚部IMφspos行
        idxREφspos_ = self.idxREφspos_
        idxIMφspos_ = self.idxIMφspos_
        Kf__[idxREφspos_[1:], idxREφspos_[:-1]] = 1                     # REφspos列下对角线
        Kf__[idxREφspos_[:-1], idxREφspos_[1:]] = 1                     # REφspos列上对角线
        Kf__[idxREφspos_, idxREφspos_] = [-1] + [-2]*(Npos - 2) + [-1]  # REφspos列主对角线
        Kf__[idxIMφspos_[1:], idxIMφspos_[:-1]] = 1                     # IMφspos列下对角线
        Kf__[idxIMφspos_[:-1], idxIMφspos_[1:]] = 1                     # IMφspos列上对角线
        Kf__[idxIMφspos_, idxIMφspos_] = [-1] + [-2]*(Npos - 2) + [-1]  # IMφspos列主对角线

        # 电解液电势实部REφe行、虚部IMφe行
        idxREφe_, idxIMφe_ = self.idxREφe_, self.idxIMφe_
        idxREφeneg_ = idxREφe_[:Nneg]
        idxIMφeneg_ = idxIMφe_[:Nneg]
        idxREφepos_ = idxREφe_[-Npos:]
        idxIMφepos_ = idxIMφe_[-Npos:]
        Kf__[idxREφeneg_, idxREjintneg_ := self.idxREjintneg_] = \
        Kf__[idxREφeneg_, idxREjDLneg_  := self.idxREjDLneg_] = \
        Kf__[idxIMφeneg_, idxIMjintneg_ := self.idxIMjintneg_] = \
        Kf__[idxIMφeneg_, idxIMjDLneg_  := self.idxIMjDLneg_] = self.Δxneg   # REjneg、IMjneg列
        Kf__[idxREφepos_, idxREjintpos_ := self.idxREjintpos_] = \
        Kf__[idxREφepos_, idxREjDLpos_  := self.idxREjDLpos_] = \
        Kf__[idxIMφepos_, idxIMjintpos_ := self.idxIMjintpos_] = \
        Kf__[idxIMφepos_, idxIMjDLpos_  := self.idxIMjDLpos_] = self.Δxpos   # REjpos、IMjpos列
        if lithiumPlating := self.lithiumPlating:
            Kf__[idxREφeneg_, self.idxREjLP_] = \
            Kf__[idxIMφeneg_, self.idxIMjLP_] = self.Δxneg  # REφe行REjLP列、IMφe行IMjLP列

        # 负极、正极局部体积电流实部REj行REj列、虚部IMj行IMj列
        idx_ = concatenate([idxREjintneg_, idxIMjintneg_, idxREjintpos_, idxIMjintpos_,
                               idxREjDLneg_, idxIMjDLneg_, idxREjDLpos_, idxIMjDLpos_,
                               self.idxREjLP_, self.idxIMjLP_])
        Kf__[idx_, idx_] = 1
        # 负极、正极交换电流实部REi0int行REi0int列、虚部IMi0int行IMi0int列
        idx_ = concatenate([self.idxREi0intneg_, self.idxIMi0intneg_,
                               self.idxREi0intpos_, self.idxIMi0intpos_])
        Kf__[idx_, idx_] = 1

        # 负极、正极过电位实部REηint行REηint列、虚部IMηint行IMηint列
        idxREηintneg_ = self.idxREηintneg_
        idxIMηintneg_ = self.idxIMηintneg_
        idxREηintpos_ = self.idxREηintpos_
        idxIMηintpos_ = self.idxIMηintpos_
        idx_ = concatenate([idxREηintneg_, idxIMηintneg_, idxREηintpos_, idxIMηintpos_,
                               self.idxREηLP_, self.idxIMηLP_])
        Kf__[idx_, idx_] = 1
        # 负极、正极过电位实部REηint行、虚部IMηint行
        Kf__[idxREηintneg_, idxREφeneg_] = \
        Kf__[idxIMηintneg_, idxIMφeneg_] = \
        Kf__[idxREηintpos_, idxREφepos_] = \
        Kf__[idxIMηintpos_, idxIMφepos_] = 1   # REφe、IMφe列
        Kf__[idxREηintneg_, idxREφsneg_] = \
        Kf__[idxIMηintneg_, idxIMφsneg_] = \
        Kf__[idxREηintpos_, idxREφspos_] = \
        Kf__[idxIMηintpos_, idxIMφspos_] = -1  # REφs、IMφs列
        if lithiumPlating:
            Kf__[self.idxREηLP_, idxREφsneg_] = \
            Kf__[self.idxIMηLP_, idxIMφsneg_] = -1  # REηLP行REφsneg列、IMηLP行IMφsneg列
            Kf__[self.idxREηLP_, idxREφeneg_] = \
            Kf__[self.idxIMηLP_, idxIMφeneg_] = 1   # REηLP行REφeneg列、IMηLP行IMφeneg列

    def update_Kf__idxREce_idxREce_and_idxIMce_idxIMce_(self, DeeffWest_, DeeffEast_):
        # 更新Kf__矩阵REce行REce列、IMce行IMce列
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Nneg, Nsep = self.Nneg, self.Nsep
        Kf__ = self.Kf__
        idxREce_, idxIMce_ = self.idxREce_, self.idxIMce_
        a = DeeffEast_[0]/self.Δxneg
        Kf__[idxREce_[0], idxREce_[:2]] = [a, -a]    # 首行：电解液浓度实部首个控制体
        a = DeeffWest_[-1]/self.Δxpos
        Kf__[idxREce_[-1], idxREce_[-2:]] = [-a, a]  # 末行：电解液浓度实部末尾控制体
        Kf__[idxREce_[1:-1], idxREce_[:-2]] = a_ = -DeeffWest_[1:-1]/ΔxWest_[1:-1]  # 1~Ne-2行下对角线：电解液浓度实部内部控制体
        Kf__[idxREce_[1:-1], idxREce_[2:]]  = c_ = -DeeffEast_[1:-1]/ΔxEast_[1:-1]  # 1~Ne-2行上对角线：电解液浓度实部内部控制体
        Kf__[idxREce_[1:-1], idxREce_[1:-1]] = -(a_ + c_)  # 1~Ne-2行主对角线：电解液浓度实部内部控制体
        # 修正负极-隔膜界面、隔膜-正极界面
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            a, c = -DeeffWest_[nW]/ΔxWest_[nW], -2*DeeffEast_[nW]*DeeffWest_[nE]/(DeeffWest_[nE]*Δx_[nW] + DeeffEast_[nW]*Δx_[nE])
            Kf__[idxREce_[nW], idxREce_[nW-1:nW+2]] = [a, -(a + c), c]  # 界面左侧控制体
            a, c = c, -DeeffEast_[nE]/ΔxEast_[nE]
            Kf__[idxREce_[nE], idxREce_[nE-1:nE+2]] = [a, -(a + c), c]  # 界面右侧控制体

        Kf__[idxIMce_, idxIMce_] = Kf__[idxREce_, idxREce_]                    # IMce列主对角线
        Kf__[idxIMce_[1:], idxIMce_[:-1]] = Kf__[idxREce_[1:], idxREce_[:-1]]  # IMce列下对角线
        Kf__[idxIMce_[:-1], idxIMce_[1:]] = Kf__[idxREce_[:-1], idxREce_[1:]]  # IMce列上对角线

    def update_Kf__idxREφsneg_idxREjneg_and_idxIMφsneg_idxIMjneg_(self, σeffneg):
        # 更新Kf__矩阵REφsneg行REjneg列、IMφsneg行IMjneg列
        Kf__ = self.Kf__
        idxREφsneg_, idxIMφsneg_ = self.idxREφsneg_, self.idxIMφsneg_
        Kf__[idxREφsneg_, self.idxREjintneg_] = \
        Kf__[idxREφsneg_, self.idxREjDLneg_] = \
        Kf__[idxIMφsneg_, self.idxIMjintneg_] = \
        Kf__[idxIMφsneg_, self.idxIMjDLneg_] = a = -self.Δxneg**2/σeffneg
        if self.lithiumPlating:
            Kf__[idxREφsneg_, self.idxREjLP_] = \
            Kf__[idxIMφsneg_, self.idxIMjLP_] = a

    def update_Kf__idxREφspos_idxREjpos_and_idxIMφspos_idxIMjpos_(self, σeffpos):
        # 更新Kf__矩阵REφspos行REjpos列、IMφspos行IMjpos列
        Kf__ = self.Kf__
        idxREφspos_, idxIMφspos_ = self.idxREφspos_, self.idxIMφspos_
        Kf__[idxREφspos_, self.idxREjintpos_] = \
        Kf__[idxREφspos_, self.idxREjDLpos_] = \
        Kf__[idxIMφspos_, self.idxIMjintpos_] = \
        Kf__[idxIMφspos_, self.idxIMjDLpos_] = -self.Δxpos**2/σeffpos

    def update_Kf__idxREφe_idxREφe_and_idxIMφe_idxIMφe_(self, κeffWest_, κeffEast_):
        # 更新Kf__矩阵REφe行REφe列、IMφe行IMφe列
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Nneg, Nsep = self.Nneg, self.Nsep
        Kf__ = self.Kf__
        idxREφe_ = self.idxREφe_
        idxIMφe_ = self.idxIMφe_
        a = κeffEast_[0]/ΔxEast_[0]
        Kf__[idxREφe_[0], idxREφe_[:2]] = [-κeffWest_[0]/(0.5*Δx_[0]) - a, a]     # REφe列首行：电解液电势实部首个控制体
        a = κeffWest_[-1]/ΔxWest_[-1]
        Kf__[idxREφe_[-1], idxREφe_[-2:]] = [a, -a]                               # REφe列末行：电解液电势实部末尾控制体
        Kf__[idxREφe_[1:-1], idxREφe_[:-2]] = a_ = κeffWest_[1:-1]/ΔxWest_[1:-1]  # REφe列下对角线第1 ~ Ne-2行：电解液电势实部内部控制体
        Kf__[idxREφe_[1:-1], idxREφe_[2:]]  = c_ = κeffEast_[1:-1]/ΔxEast_[1:-1]  # REφe列上对角线第1 ~ Ne-2行：电解液电势实部内部控制体
        Kf__[idxREφe_[1:-1], idxREφe_[1:-1]] = -(a_ + c_)                         # REφe列主对角线第1 ~ Ne-2行：电解液电势实部内部控制体
        # 修正负极-隔膜界面、隔膜-正极界面
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            a, c = κeffWest_[nW]/ΔxWest_[nW], 2*κeffEast_[nW]*κeffWest_[nE]/(
                   κeffWest_[nE]*Δx_[nW] + κeffEast_[nW]*Δx_[nE])
            Kf__[idxREφe_[nW], idxREφe_[nW - 1:nW + 2]] = [a, -(a + c), c]  # 界面左侧控制体
            a, c = c, κeffEast_[nE]/ΔxEast_[nE]
            Kf__[idxREφe_[nE], idxREφe_[nE - 1:nE + 2]] = [a, -(a + c), c]  # 界面右侧控制体

        Kf__[idxIMφe_, idxIMφe_] = Kf__[idxREφe_, idxREφe_]                    # IMφe列主对角线
        Kf__[idxIMφe_[1:], idxIMφe_[:-1]] = Kf__[idxREφe_[1:], idxREφe_[:-1]]  # IMφe列下对角线
        Kf__[idxIMφe_[:-1], idxIMφe_[1:]] = Kf__[idxREφe_[:-1], idxREφe_[1:]]  # IMφe列上对角线

    def update_Kf__idxREηintneg_idxREjneg_and_idxIMηintneg_idxIMjneg_(self, RSEIneg, aeffneg):
        # 更新Kf__矩阵REηintneg行REjneg列、IMηintneg行IMjneg列
        Kf__ = self.Kf__
        idxREηintneg_, idxIMηintneg_ = self.idxREηintneg_, self.idxIMηintneg_
        Kf__[idxREηintneg_, self.idxREjintneg_] = \
        Kf__[idxREηintneg_, self.idxREjDLneg_] = \
        Kf__[idxIMηintneg_, self.idxIMjintneg_] = \
        Kf__[idxIMηintneg_, self.idxIMjDLneg_] = a = RSEIneg/aeffneg
        if self.lithiumPlating:
            Kf__[idxREηintneg_, self.idxREjLP_] = \
            Kf__[idxIMηintneg_, self.idxIMjLP_] = a

    def update_Kf__idxREηintpos_idxREjpos_and_idxIMηintpos_idxIMjpos_(self, RSEIpos, aeffpos):
        # 更新Kf__矩阵REηintpos行REjpos列、IMηintpos行IMjpos列
        Kf__ = self.Kf__
        idxREηintpos_, idxIMηintpos_ = self.idxREηintpos_, self.idxIMηintpos_
        Kf__[idxREηintpos_, self.idxREjintpos_] = \
        Kf__[idxREηintpos_, self.idxREjDLpos_] = \
        Kf__[idxIMηintpos_, self.idxIMjintpos_] = \
        Kf__[idxIMηintpos_, self.idxIMjDLpos_] = RSEIpos/aeffpos

    def update_Kf__idxREηLP_idxREjneg_and_idxIMηLP_idxIMjneg_(self, RSEIneg, aeffneg):
        # 更新Kf__矩阵REηLP行REJneg列、IMηLP行IMJneg列
        Kf__ = self.Kf__
        idxREηLP_, idxIMηLP_ = self.idxREηLP_, self.idxIMηLP_
        Kf__[idxREηLP_, self.idxREjintneg_] = \
        Kf__[idxREηLP_, self.idxREjDLneg_] = \
        Kf__[idxREηLP_, self.idxREjLP_] = \
        Kf__[idxIMηLP_, self.idxIMjintneg_] = \
        Kf__[idxIMηLP_, self.idxIMjDLneg_] = \
        Kf__[idxIMηLP_, self.idxIMjLP_] = RSEIneg/aeffneg

    def record_EISdata(self):
        """记录阻抗数据"""
        for dataname in self.EISdatanames_:
            value = getattr(self, dataname)
            if isscalar(value):
                pass
            else:
                value = value.copy()
            self.data[dataname].append(value)

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

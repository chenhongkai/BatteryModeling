from typing import Sequence
from decimal import Decimal
from math import cos, sin

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from numpy import pi, nan, ndarray,\
    array, arange, zeros, full, stack, hstack, concatenate, \
    empty, meshgrid, \
    sinh, cosh, outer, ptp, log10,\
    ix_, isscalar
from numpy.linalg import solve

from P2Dmodel import DFNP2D


class JTFP2D(DFNP2D):
    """锂离子电池时频联合准二维模型 Joint Time-Frequency Pseudo-two-Dimension model"""
    def __init__(self,
            f_: Sequence[float] = (100, 10, 1.,),   # 频率序列 [Hz]
            SOC0: float = 0.5,                      # 初始荷电状态 [–]
            T0: float = 298.15,                     # 初始温度 [K]
            fullyInitialize: bool = True,
            **kwargs):
        fullyInitialize = fullyInitialize or (type(self) is JTFP2D)
        if type(self) is JTFP2D:
            DFNP2D.__init__(self, **kwargs)
        self.f_ = f_ = array(f_); assert f_.ndim==1, f'频率序列f_应可转化为ndim==1的ndarray，当前{f_ = }'
        # 恒定量
        (self.ΔIAC,                      # 交流扰动电流振幅 [A]
        self.frequency_dependent_cache,  # 频率相关变量缓存
        self.EISdatanames_,              # 需记录的阻抗数据名称
        ) = (None,)*3
        # 状态量
        Nf, Nneg, Npos = len(f_), self.Nneg, self.Npos
        self.REφsneg__, self.IMφsneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极固相电势实部、虚部
        self.REφspos__, self.IMφspos__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极固相电势实部、虚部
        if self.complete:
            Ne = self.Ne
            self.REφe__, self.IMφe__ = empty((Nf, Ne)), empty((Nf, Ne))  # 电解液电势实部、虚部
            self.REηintneg__, self.IMηintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极主反应过电位实部、虚部
            self.REηintpos__, self.IMηintpos__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极主反应过电位实部、虚部
            if lithiumPlating:=self.lithiumPlating:
                self.REηLP__, self.IMηLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 析锂反应过电位实部、虚部
            if fullyInitialize:
                self.REcsnegsurf__, self.IMcsnegsurf__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极固相表面浓度实部、虚部
                self.REcspossurf__, self.IMcspossurf__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极固相表面浓度实部、虚部
                self.REce__, self.IMce__ = empty((Nf, Ne)), empty((Nf, Ne))                    # 电解液锂离子浓度实部、虚部
                self.REjintneg__, self.IMjintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))      # 负极主反应局部体积电流密度实部、虚部
                self.REjintpos__, self.IMjintpos__ = empty((Nf, Npos)), empty((Nf, Npos))      # 正极主反应局部体积电流密度实部、虚部
                self.REjDLneg__, self.IMjDLneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))        # 负极双电层局部体积电流密度实部、虚部
                self.REjDLpos__, self.IMjDLpos__ = empty((Nf, Npos)), empty((Nf, Npos))        # 正极双电层局部体积电流密度实部、虚部
                self.REi0intneg__, self.IMi0intneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))    # 负极交换电流密度实部、虚部
                self.REi0intpos__, self.IMi0intpos__ = empty((Nf, Npos)), empty((Nf, Npos))    # 正极交换电流密度实部、虚部
                if lithiumPlating:
                    self.REjLP__, self.IMjLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))   # 负极析锂反应局部体积电流密度实部、虚部
        (self.tEIS,          # 计算阻抗的时刻 [s]
        self.Kf__,           # 频域因变量线性矩阵
        self.bKf_,           # Kf__ @ X_ = bKf_
        self.bandwidthsKf_,  # Kf__矩阵上下带宽
        ) = (None,)*4
        # 索引频域因变量
        (self.idxREcsnegsurf_, self.idxIMcsnegsurf_,
        self.idxREcspossurf_, self.idxIMcspossurf_,
        self.idxREce_, self.idxIMce_,
        self.idxREφsneg_, self.idxIMφsneg_,
        self.idxREφspos_, self.idxIMφspos_,
        self.idxREφe_, self.idxIMφe_,
        self.idxREjintneg_, self.idxIMjintneg_,
        self.idxREjintpos_, self.idxIMjintpos_,
        self.idxREjDLneg_, self.idxIMjDLneg_,
        self.idxREjDLpos_, self.idxIMjDLpos_,
        self.idxREjLP_, self.idxIMjLP_,
        self.idxREi0intneg_, self.idxIMi0intneg_,
        self.idxREi0intpos_, self.idxIMi0intpos_,
        self.idxREηintneg_, self.idxIMηintneg_,
        self.idxREηintpos_, self.idxIMηintpos_,
        self.idxREηLP_, self.idxIMηLP_,
        self.idxKfReordered_, self.idxKfRecovered_,
        ) = (None,)*34
        # 初始化
        if type(self) is JTFP2D:
            self.initialize(
                SOC0=SOC0,  # 初始荷电状态[0]
                T0=T0,)     # 初始温度 [K]

    def initialize(self,
            SOC0: float = .5,     # 初始荷电状态 [–]
            T0: float = 298.15):  # 初始温度 [K]
        """初始化"""
        if self.verbose:
            print('时频联合P2D模型初始化...')
        DFNP2D.initialize(self, SOC0=SOC0, T0=T0)
        # 恒定量
        self.ΔIAC = self.Qcell*0.05   # 交流扰动电流振幅 [A]
        self.initialize_frequency_domain_linear_matrix()
        self.frequency_dependent_cache = self.solve_frequency_dependent_variables()
        # 需记录的阻抗数据名称
        self.EISdatanames_ = EISdatanames_ = ['tEIS', 'Z_']  # 频率 [Hz]、阻抗时刻 [s]、序列 、复阻抗 [Ω]
        if self.complete:
            self.EISdatanames_.extend([
                'Zneg_', 'Zsep_', 'Zpos_',         # 负极、隔膜、正极复阻抗 [Ω]
                'REcsnegsurf__', 'IMcsnegsurf__',  # 负极固相表面锂离子浓度实部、虚部 [mol/m^3]
                'REcspossurf__', 'IMcspossurf__',  # 正极固相表面锂离子浓度实部、虚部 [mol/m^3]
                'REce__', 'IMce__',            # 电解液锂离子浓度实部、虚部 [mol/m^3]
                'REφsneg__', 'IMφsneg__',      # 负极固相电势实部、虚部 [V]
                'REφspos__', 'IMφspos__',      # 正极固相电势实部、虚部 [V]
                'REφe__', 'IMφe__',            # 电解液电势实部、虚部 [V]
                'REjintneg__', 'IMjintneg__',  # 负极主反应局部体积电流密度实部、虚部 [A/m^3]
                'REjintpos__', 'IMjintpos__',  # 正极主反应局部体积电流密度实部、虚部 [A/m^3]
                'REjDLneg__', 'IMjDLneg__',    # 负极双电层效应局部体积电流密度实部、虚部 [A/m^3]
                'REjDLpos__', 'IMjDLpos__',    # 正极双电层效应局部体积电流密度实部、虚部 [A/m^3]
                'REi0intneg__', 'IMi0intneg__',  # 负极主反应交换电流密度实部、虚部 [A/m^2]
                'REi0intpos__', 'IMi0intpos__',  # 正极主反应交换电流密度实部、虚部 [A/m^2]
                'REηintneg__', 'IMηintneg__',    # 负极主反应过电位实部、虚部 [V]
                'REηintpos__', 'IMηintpos__',])  # 正极主反应过电位实部、虚部 [V]
        self.data.update({EISdataname: [] for EISdataname in EISdatanames_})  # 字典：存储呈时间序列的阻抗数据
        if self.verbose and type(self) is JTFP2D:
            print(self)
            print('时频联合P2D模型初始化完成!')
        return self

    def solve_frequency_dependent_variables(self) -> dict:
        """求解频率相关变量"""
        ω_ = self.ω_
        solve_Kcssurf__ = JTFP2D.solve_Kcssurf__
        Rsneg, Rspos, CDLneg, CDLpos, Dsneg, Dspos = (
            self.Rsneg, self.Rspos, self.CDLneg, self.CDLpos, self.Dsneg, self.Dspos)
        aneg, apos = self.aneg, self.apos
        frequency_dependent_variables = {
            'ωεeΔx__': outer(ω_, self.εe_*self.Δx_),  # (Nf, Ne) 各频率各控制体的ω*εe*Δx值
            'ωaCDLneg_': ω_ * (self.aeffneg*CDLneg),  # (Nf,)
            'ωaCDLpos_': ω_ * (self.aeffpos*CDLpos),  # (Nf,)
            'ωCDLRSEIneg_': ω_ * (CDLneg*self.RSEIneg),  # (Nf,)
            'ωCDLRSEIpos_': ω_ * (CDLpos*self.RSEIpos),  # (Nf,)
            'minusKcsnegsurf___': -array([solve_Kcssurf__(ω, Rsneg, Dsneg, aneg) for ω in ω_]),   # (Nf, 2, 2) 负极各频率Kcssurf__矩阵
            'minusKcspossurf___': -array([solve_Kcssurf__(ω, Rspos, Dspos, apos) for ω in ω_]),}  # (Nf, 2, 2) 正极各频率Kcssurf__矩阵
        return frequency_dependent_variables

    def generate_indices_of_frequency_domain_dependent_variables(self) -> int:
        """生成频域因变量索引"""
        Nneg, Nsep, Npos, Ne, Nr = self.Nneg, self.Nsep, self.Npos, self.Ne, self.Nr  # 读取：网格数
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
        self.idxREi0intneg_ = allocate(Nneg if self._i0intneg is None else 0)  # 索引：负极交换电流密度实部
        self.idxIMi0intneg_ = allocate(Nneg if self._i0intneg is None else 0)  # 索引：负极交换电流密度虚部
        self.idxREi0intpos_ = allocate(Npos if self._i0intpos is None else 0)  # 索引：正极交换电流密度实部
        self.idxIMi0intpos_ = allocate(Npos if self._i0intpos is None else 0)  # 索引：正极交换电流密度虚部
        self.idxREηintneg_ = allocate(Nneg)   # 索引：负极过电位实部
        self.idxIMηintneg_ = allocate(Nneg)   # 索引：负极过电位虚部
        self.idxREηintpos_ = allocate(Npos)   # 索引：正极过电位实部
        self.idxIMηintpos_ = allocate(Npos)   # 索引：正极过电位虚部
        lithiumPlating = self.lithiumPlating
        self.idxREjLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：析锂反应电流密度实部
        self.idxIMjLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：正极交换电流密度虚部
        self.idxREηLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：析锂反应过电位实部
        self.idxIMηLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：析锂反应过电位虚部
        return N  # 频域因变量总数

    def initialize_frequency_domain_linear_matrix(self) -> None:
        """初始化频域线性矩阵"""
        N = self.generate_indices_of_frequency_domain_dependent_variables()
        self.Kf__ = Kf__ = zeros([N, N])  # 频域因变量线性矩阵
        self.bKf_ = zeros(N)              # Kf__ @ X_ = bKf_
        if self.verbose:
            print(f'初始化频域因变量线性矩阵 Kf__.shape = {Kf__.shape}')

        ## 对频域因变量线性矩阵Kf__赋参数相关值 ##
        self.update_Kf__idxREce_idxREce_and_idxIMce_idxIMce_(Deeff_:=self.Deeff_, Deeff_)
        self.update_Kf__idxREce_idxREj_and_idxIMce_idxIMj_(self.tplus)
        self.update_Kf__idxREφsneg_idxREjneg_and_idxIMφsneg_idxIMjneg_(σeffneg := self.σeffneg)
        self.update_Kf__idxREφspos_idxREjpos_and_idxIMφspos_idxIMjpos_(σeffpos := self.σeffpos)
        self.update_bKf_idxREφsneg_0_and_idxREφspos_end(σeffneg, σeffpos)
        self.update_Kf__idxREφe_idxREφe_and_idxIMφe_idxIMφe_(κeff_:=self.κeff_, κeff_)
        self.update_Kf__idxREηintneg_idxREjneg_and_idxIMηintneg_idxIMjneg_(self.RSEIneg, self.aeffneg)
        self.update_Kf__idxREηintpos_idxREjpos_and_idxIMηintpos_idxIMjpos_(self.RSEIpos, self.aeffpos)
        if self.lithiumPlating:
            self.update_Kf__idxREηLP_idxREjneg_and_idxIMηLP_idxIMjneg_(self.RSEIneg, self.aeffneg)

        ## 对频域因变量线性矩阵Kf__赋恒定值 ##
        self.assign_Kf__with_constants()

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

    def update_Kf__idxREce_idxREj_and_idxIMce_idxIMj_(self, tplus):
        # 更新Kf__矩阵REce行REj列、IMce行IMj列
        a = (1 - tplus)/DFNP2D.F
        Nneg, Npos = self.Nneg, self.Npos
        Kf__ = self.Kf__
        idxREce_, idxIMce_ = self.idxREce_, self.idxIMce_
        idxREceneg_, idxREcepos_ = idxREce_[:Nneg], idxREce_[-Npos:]
        idxIMceneg_, idxIMcepos_ = idxIMce_[:Nneg], idxIMce_[-Npos:]
        idxREjintneg_, idxREjintpos_ = self.idxREjintneg_, self.idxREjintpos_
        idxREjDLneg_, idxREjDLpos_   = self.idxREjDLneg_, self.idxREjDLpos_
        idxIMjintneg_, idxIMjintpos_ = self.idxIMjintneg_, self.idxIMjintpos_
        idxIMjDLneg_, idxIMjDLpos_   = self.idxIMjDLneg_, self.idxIMjDLpos_
        Kf__[idxREceneg_, idxREjintneg_] = \
        Kf__[idxREceneg_, idxREjDLneg_]  = \
        Kf__[idxIMceneg_, idxIMjintneg_] = \
        Kf__[idxIMceneg_, idxIMjDLneg_] = n = -self.Δxneg*a
        Kf__[idxREcepos_, idxREjintpos_] = \
        Kf__[idxREcepos_, idxREjDLpos_]  = \
        Kf__[idxIMcepos_, idxIMjintpos_] = \
        Kf__[idxIMcepos_, idxIMjDLpos_] = -self.Δxpos*a
        if self.lithiumPlating:
            Kf__[idxREceneg_, self.idxREjLP_] = \
            Kf__[idxIMceneg_, self.idxIMjLP_] = n

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

    def update_bKf_idxREφsneg_0_and_idxREφspos_end(self, σeffneg, σeffpos):
        bKf_ = self.bKf_
        ΔiAC = self.ΔiAC
        # 更新bKf_向量REφsneg首元
        bKf_[self.idxREφsneg_[0]] = -self.Δxneg*ΔiAC/σeffneg
        # 更新bKf_向量REφspos末元
        bKf_[self.idxREφspos_[-1]] = self.Δxpos*ΔiAC/σeffpos

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

    def EIS(self):
        """计算电化学阻抗谱"""
        tEIS = self.t       # 读取：当前时刻 [s]
        data = self.data  # 读取：数据字典
        if data['tEIS'] and data['tEIS'][-1]==tEIS:
            if self.verbose:
                print(f'已计算时刻{tEIS = } s 电化学阻抗谱')
            return self
        # 频域因变量索引
        idxREcsnegsurf_, idxIMcsnegsurf_ = self.idxREcsnegsurf_, self.idxIMcsnegsurf_
        idxREcspossurf_, idxIMcspossurf_ = self.idxREcspossurf_, self.idxIMcspossurf_
        idxREce_, idxIMce_ = self.idxREce_, self.idxIMce_
        idxREφsneg_, idxIMφsneg_ = self.idxREφsneg_, self.idxIMφsneg_
        idxREφspos_, idxIMφspos_ = self.idxREφspos_, self.idxIMφspos_
        idxREφe_, idxIMφe_ = self.idxREφe_, self.idxIMφe_
        idxREjintneg_, idxIMjintneg_ = self.idxREjintneg_, self.idxIMjintneg_
        idxREjintpos_, idxIMjintpos_ = self.idxREjintpos_, self.idxIMjintpos_
        idxREjDLneg_, idxIMjDLneg_ = self.idxREjDLneg_, self.idxIMjDLneg_
        idxREjDLpos_, idxIMjDLpos_ = self.idxREjDLpos_, self.idxIMjDLpos_
        idxREi0intneg_, idxIMi0intneg_ = self.idxREi0intneg_, self.idxIMi0intneg_
        idxREi0intpos_, idxIMi0intpos_ = self.idxREi0intpos_, self.idxIMi0intpos_
        idxREηintneg_, idxIMηintneg_ = self.idxREηintneg_, self.idxIMηintneg_
        idxREηintpos_, idxIMηintpos_ = self.idxREηintpos_, self.idxIMηintpos_
        REIMi0intnegUnknown = idxREi0intneg_.size > 0
        REIMi0intposUnknown = idxREi0intpos_.size > 0
        lithiumPlating = self.lithiumPlating

        solve_banded_matrix = DFNP2D.solve_banded_matrix
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos            # 读取：网格数
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_  # 读取：网格距离 [m]
        κDeffWest_ = κDeffEast_ = self.κDeff_
        DeeffWest_ = DeeffEast_ = self.Deeff_

        if self.constants:
            cache = self.frequency_dependent_cache
        else:
            cache = self.solve_frequency_dependent_variables()
            κeffWest_ = κeffEast_ = self.κeff_
            self.update_Kf__idxREce_idxREce_and_idxIMce_idxIMce_(DeeffWest_, DeeffEast_)
            self.update_Kf__idxREce_idxREj_and_idxIMce_idxIMj_(self.tplus)
            self.update_Kf__idxREφsneg_idxREjneg_and_idxIMφsneg_idxIMjneg_(σeffneg := self.σeffneg)
            self.update_Kf__idxREφspos_idxREjpos_and_idxIMφspos_idxIMjpos_(σeffpos := self.σeffpos)
            self.update_bKf_idxREφsneg_0_and_idxREφspos_end(σeffneg, σeffpos)
            self.update_Kf__idxREφe_idxREφe_and_idxIMφe_idxIMφe_(κeffWest_, κeffEast_)
            self.update_Kf__idxREηintneg_idxREjneg_and_idxIMηintneg_idxIMjneg_(self.RSEIneg, self.aeffneg)
            self.update_Kf__idxREηintpos_idxREjpos_and_idxIMηintpos_idxIMjpos_(self.RSEIpos, self.aeffpos)
            if lithiumPlating:
                self.update_Kf__idxREηLP_idxREjneg_and_idxIMηLP_idxIMjneg_(self.RSEIneg, self.aeffneg)

        ce_ = self.ce_  # 读取：电解液锂离子浓度场 [mol/m^3]
        ceInterfaces_ = self.ceInterfaces_
        ceWest_ = ceInterfaces_[:-1]  # (Ne,) 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        ceEast_ = ceInterfaces_[1:]   # (Ne,) 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradceWest_ = hstack([0, (ce_[1:] - ce_[:-1])/ΔxWest_[1:]])   # (Ne,) 各控制体左界面的锂离子浓度梯度 [mol/m^4]
        gradceEast_ = hstack([(ce_[1:] - ce_[:-1])/ΔxEast_[:-1], 0])  # (Ne,) 各控制体右界面的锂离子浓度梯度 [mol/m^4]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradceEast_[nW] = (ceEast_[nW] - ce_[nW])/(0.5*Δx_[nW])
            gradceWest_[nE] = (ce_[nE] - ceWest_[nE])/(0.5*Δx_[nE])

        Kf__ = self.Kf__  # 频域因变量矩阵
        bKf_ = self.bKf_  # b向量

        ## 对Kf__矩阵赋时变值 ##

        # 电解液电势实部REφe行REce列
        a = κDeffEast_[0]/ceEast_[0]/ΔxEast_[0]
        aa = (κDeffEast_[0]*gradceEast_[0]/ceEast_[0]**2)/2
        Kf__[idxREφe_[0], idxREce_[:2]] = [a + aa, -a + aa]  # 首行
        a = κDeffWest_[-1]/ceWest_[-1]/ΔxWest_[-1]
        aa = (-κDeffWest_[-1]*gradceWest_[-1]/ceWest_[-1]**2)/2
        Kf__[idxREφe_[-1], idxREce_[-2:]] = [-a + aa, a + aa]  # 末行
        a_ = -κDeffWest_[1:-1]/ceWest_[1:-1]/ΔxWest_[1:-1]
        c_ = -κDeffEast_[1:-1]/ceEast_[1:-1]/ΔxEast_[1:-1]
        aa_ = (-κDeffWest_[1:-1]*gradceWest_[1:-1]/ceWest_[1:-1]**2)/2
        cc_ = ( κDeffEast_[1:-1]*gradceEast_[1:-1]/ceEast_[1:-1]**2)/2
        Kf__[idxREφe_[1:-1], idxREce_[:-2]] = a_ + aa_
        Kf__[idxREφe_[1:-1], idxREce_[2:]]  = c_ + cc_
        Kf__[idxREφe_[1:-1], idxREce_[1:-1]] = -(a_ + c_) + aa_ + cc_
        # 修正负极-隔膜界面、隔膜-正极界面
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            pDW = DeeffEast_[nW]*Δx_[nE] / (DeeffWest_[nE]*Δx_[nW] + DeeffEast_[nW]*Δx_[nE])
            pDE = 1 - pDW  # 即：DeeffWest_[nE]*Δx_[nW] / (DeeffWest_[nE]*Δx_[nW] + DeeffEast_[nW]*Δx_[nE])
            # 界面左侧控制体
            a = -κDeffWest_[nW]/ceWest_[nW]/ΔxWest_[nW]
            aa = (-κDeffWest_[nW]*gradceWest_[nW]/ceWest_[nW]**2)/2
            c = -2*κDeffEast_[nW]*DeeffWest_[nE] / ceEast_[nW] / (DeeffWest_[nE]*Δx_[nW] + DeeffEast_[nW]*Δx_[nE])
            cc = κDeffEast_[nW]*gradceEast_[nW]/ceEast_[nW]**2
            Kf__[idxREφe_[nW], idxREce_[nW-1:nW+2]] = [a + aa,
                                                       -(a + c) + aa + cc*pDW,
                                                       c + cc*pDE]
            # 界面右侧控制体
            a = -2*κDeffWest_[nE]*DeeffEast_[nW] / ceWest_[nE] / (DeeffWest_[nE]*Δx_[nW] + DeeffEast_[nW]*Δx_[nE])
            aa = -κDeffWest_[nE]*gradceWest_[nE]/ceWest_[nE]**2
            c = -κDeffEast_[nE]/ceEast_[nE]/ΔxEast_[nE]
            cc = (κDeffEast_[nE]*gradceEast_[nE]/ceEast_[nE]**2)/2
            Kf__[idxREφe_[nE], idxREce_[nE-1:nE+2]] = [a + aa*pDW ,
                                                       -(a + c) + aa*pDE + cc,
                                                       c + cc]
        # 电解液电势虚部IMφe行
        Kf__[idxIMφe_, idxIMce_] = Kf__[idxREφe_, idxREce_]                    # IMce列主对角线
        Kf__[idxIMφe_[1:], idxIMce_[:-1]] = Kf__[idxREφe_[1:], idxREce_[:-1]]  # IMce列下对角线
        Kf__[idxIMφe_[:-1], idxIMce_[1:]] = Kf__[idxREφe_[:-1], idxREce_[1:]]  # IMce列上对角线

        # 负极主反应局部体积电流密度实部REjintneg行、虚部IMjintneg行
        Kf__[idxREjintneg_, idxREηintneg_] = \
        Kf__[idxIMjintneg_, idxIMηintneg_] = -self.djintdηintneg_  # REIMηintneg列
        if REIMi0intnegUnknown:
            Kf__[idxREjintneg_, idxREi0intneg_] = \
            Kf__[idxIMjintneg_, idxIMi0intneg_] = -self.djintdi0intneg_  # REIMi0intneg列
        # 正极局部体积电流密度实部REjintpos行、虚部IMjintpos行
        Kf__[idxREjintpos_, idxREηintpos_] = \
        Kf__[idxIMjintpos_, idxIMηintpos_] = -self.djintdηintpos_  # IMηintpos列
        if REIMi0intposUnknown:
            Kf__[idxREjintpos_, idxREi0intpos_] = \
            Kf__[idxIMjintpos_, idxIMi0intpos_] = -self.djintdi0intpos_  # IMi0intpos列

        if REIMi0intnegUnknown:
            # 负极交换电流密度实部REi0intneg行、虚部IMi0intneg行
            Kf__[idxREi0intneg_, idxREce_[:Nneg]] = \
            Kf__[idxIMi0intneg_, idxIMce_[:Nneg]] = -self.di0intdceneg_      # REIMce列
            Kf__[idxREi0intneg_, idxREcsnegsurf_] = \
            Kf__[idxIMi0intneg_, idxIMcsnegsurf_] = -self.di0intdcsnegsurf_  # REIMcsnegsurf列

        if REIMi0intposUnknown:
            # 正极交换电流密度实部REi0intpos行、虚部IMi0intpos行
            Kf__[idxREi0intpos_, idxREce_[-Npos:]] = \
            Kf__[idxIMi0intpos_, idxIMce_[-Npos:]] = -self.di0intdcepos_     # REIMce列
            Kf__[idxREi0intpos_, idxREcspossurf_] = \
            Kf__[idxIMi0intpos_, idxIMcspossurf_] = -self.di0intdcspossurf_  # REIMcsnegsurf列

        # 负极过电位实部REηintneg行REcsnegsurf列、虚部IMηintneg行IMcsnegsurf列
        Kf__[idxREηintneg_, idxREcsnegsurf_] = \
        Kf__[idxIMηintneg_, idxIMcsnegsurf_] = self.dUOCPdcsnegsurf_
        # 正极过电位实部REηintpos行REcspossurf列、虚部IMηintpos行IMcsnegsurf列
        Kf__[idxREηintpos_, idxREcspossurf_] = \
        Kf__[idxIMηintpos_, idxIMcspossurf_] = self.dUOCPdcspossurf_

        if lithiumPlating:
            # 析锂补充
            idxREjLP_, idxIMjLP_ = self.idxREjLP_, self.idxIMjLP_
            idxREηLP_, idxIMηLP_ = self.idxREηLP_, self.idxIMηLP_
            # 析锂局部体积电流密度实部REjLP行REηLP列、虚部IMjLP行IMηLP列
            Kf__[idxREjLP_, idxREηLP_] = \
            Kf__[idxIMjLP_, idxIMηLP_] = -self.djLPdηLP_  # IMηLP列
            # 析锂局部体积电流实部REJLP行REθeneg列、虚部IMJLP行IMθeneg列
            Kf__[idxREjLP_, idxREce_[:Nneg]] = \
            Kf__[idxIMjLP_, idxIMce_[:Nneg]] = -self.djLPdce_

        Nf = self.f_.size
        X__ = empty((Nf, bKf_.shape[0]), dtype=bKf_.dtype)

        for nf, (ωεeΔx_,
             ωaCDLneg, ωaCDLpos,
             ωCDLRSEIneg, ωCDLRSEIpos,
             minusKcsnegsurf__, minusKcspossurf__) in enumerate(zip(
                cache['ωεeΔx__'],
                cache['ωaCDLneg_'], cache['ωaCDLpos_'],
                cache['ωCDLRSEIneg_'], cache['ωCDLRSEIpos_'],
                cache['minusKcsnegsurf___'], cache['minusKcspossurf___'])):
            ## 遍历所有频率f ##

            # 负极固相表面浓度实部REcsnegsurf行
            Kf__[idxREcsnegsurf_, idxREjintneg_] = minusKcsnegsurf__[0, 0]  # REjintneg列
            Kf__[idxREcsnegsurf_, idxIMjintneg_] = minusKcsnegsurf__[0, 1]  # IMjintneg列
            # 负极固相表面浓度虚部IMcsnegsurf行
            Kf__[idxIMcsnegsurf_, idxREjintneg_] = minusKcsnegsurf__[1, 0]  # REjintneg列
            Kf__[idxIMcsnegsurf_, idxIMjintneg_] = minusKcsnegsurf__[1, 1]  # IMjintneg列
            # 正极固相表面浓度实部REcspossurf行
            Kf__[idxREcspossurf_, idxREjintpos_] = minusKcspossurf__[0, 0]  # REjintpos列
            Kf__[idxREcspossurf_, idxIMjintpos_] = minusKcspossurf__[0, 1]  # IMjintpos列
            # 正极固相表面浓度虚部IMcspossurf行
            Kf__[idxIMcspossurf_, idxREjintpos_] = minusKcspossurf__[1, 0]  # REjintpos列
            Kf__[idxIMcspossurf_, idxIMjintpos_] = minusKcspossurf__[1, 1]  # IMjintpos列

            Kf__[idxREce_, idxIMce_] = -ωεeΔx_  # REce行IMce列
            Kf__[idxIMce_, idxREce_] = ωεeΔx_   # IMce行REce列

            # 负极双电层局部体积电流密度实部REjDLneg行
            Kf__[idxREjDLneg_, idxIMφe_[:Nneg]] = -ωaCDLneg   # IMφe负极列
            Kf__[idxREjDLneg_, idxIMφsneg_] = ωaCDLneg        # IMφsneg列
            Kf__[idxREjDLneg_, idxIMjintneg_] = \
            Kf__[idxREjDLneg_, idxIMjDLneg_] = -ωCDLRSEIneg   # IMjintneg列、IMjDLneg列
            # 负极双电层局部体积电流密度虚部IMjDLneg行
            Kf__[idxIMjDLneg_, idxREφe_[:Nneg]] = ωaCDLneg    # REφe负极列
            Kf__[idxIMjDLneg_, idxREφsneg_] = -ωaCDLneg       # REφsneg列
            Kf__[idxIMjDLneg_, idxREjintneg_] = \
            Kf__[idxIMjDLneg_, idxREjDLneg_]  = ωCDLRSEIneg   # REjintneg列、REjDLneg列
            # 正极双电层局部体积电流密度实部REjDLpos行
            Kf__[idxREjDLpos_, idxIMφe_[-Npos:]] = -ωaCDLpos  # IMφe正极列
            Kf__[idxREjDLpos_, idxIMφspos_] = ωaCDLpos        # IMφspos列
            Kf__[idxREjDLpos_, idxIMjintpos_] = \
            Kf__[idxREjDLpos_, idxIMjDLpos_] = -ωCDLRSEIpos   # IMjintpos列、IMjDLpos列
            # 正极双电层局部体积电流密度虚部IMjDLpos行
            Kf__[idxIMjDLpos_, idxREφe_[-Npos:]] = ωaCDLpos  # REφe正极列
            Kf__[idxIMjDLpos_, idxREφspos_] = -ωaCDLpos      # REφspos列
            Kf__[idxIMjDLpos_, idxREjintpos_] = \
            Kf__[idxIMjDLpos_, idxREjDLpos_] = ωCDLRSEIpos   # REjintpos列、REjDLpos列

            if lithiumPlating:
                # 补充
                Kf__[idxREjDLneg_, idxIMjLP_] = -ωCDLRSEIneg  # REjDLneg行IMjLP列
                Kf__[idxIMjDLneg_, idxREjLP_] =  ωCDLRSEIneg  # IMjDLneg行REjLP列

            if (self.bandwidthsKf_ is None) and any(self.data['I']):
                if verbose := self.verbose:
                    print('辨识重排频域因变量Kf__矩阵的带宽 -> ', end='')
                self.idxKfReordered_ = idxKfReordered_ = reverse_cuthill_mckee(csr_matrix(Kf__))
                self.idxKfRecovered_ = idxKfReordered_.argsort()
                self.bandwidthsKf_ = DFNP2D.identify_bandwidths(Kf__[ix_(idxKfReordered_, idxKfReordered_)])
                if verbose:
                    print(f'上带宽{self.bandwidthsKf_['upper']}，下带宽{self.bandwidthsKf_['lower']}')

            if bandwidthsKf_ := self.bandwidthsKf_:
                # 带状化求解
                X__[nf] = solve_banded_matrix(Kf__, bKf_,
                    self.idxKfReordered_, self.idxKfRecovered_, bandwidthsKf_)
            else:
                # 直接求解
                X__[nf] = solve(Kf__, bKf_)

        self.tEIS = tEIS
        self.REφsneg__[:] = X__[:, idxREφsneg_]  # 负极固相电势实部
        self.IMφsneg__[:] = X__[:, idxIMφsneg_]  # 负极固相电势虚部
        self.REφspos__[:] = X__[:, idxREφspos_]  # 正极固相电势实部
        self.IMφspos__[:] = X__[:, idxIMφspos_]  # 正极固相电势虚部
        if self.complete:
            self.REcsnegsurf__[:] = X__[:, idxREcsnegsurf_]  # 负极固相表面浓度实部
            self.IMcsnegsurf__[:] = X__[:, idxIMcsnegsurf_]  # 负极固相表面浓度虚部
            self.REcspossurf__[:] = X__[:, idxREcspossurf_]  # 正极固相表面浓度实部
            self.IMcspossurf__[:] = X__[:, idxIMcspossurf_]  # 正极固相表面浓度虚部
            self.REce__[:] = X__[:, idxREce_]        # 电解液锂离子浓度实部
            self.IMce__[:] = X__[:, idxIMce_]        # 电解液锂离子浓度虚部
            self.REφe__[:] = X__[:, idxREφe_]        # 电解液电势实部
            self.IMφe__[:] = X__[:, idxIMφe_]        # 电解液电势虚部
            self.REjintneg__[:] = X__[:, idxREjintneg_]  # 负极主反应局部体积电流密度实部
            self.IMjintneg__[:] = X__[:, idxIMjintneg_]  # 负极主反应局部体积电流密度虚部
            self.REjintpos__[:] = X__[:, idxREjintpos_]  # 正极主反应局部体积电流密度实部
            self.IMjintpos__[:] = X__[:, idxIMjintpos_]  # 正极主反应局部体积电流密度虚部
            self.REjDLneg__[:] = X__[:, idxREjDLneg_]  # 负极双电层局部体积电流密度实部
            self.IMjDLneg__[:] = X__[:, idxIMjDLneg_]  # 负极双电层局部体积电流密度虚部
            self.REjDLpos__[:] = X__[:, idxREjDLpos_]  # 正极双电层局部体积电流密度实部
            self.IMjDLpos__[:] = X__[:, idxIMjDLpos_]  # 正极双电层局部体积电流密度虚部
            self.REi0intneg__[:] = X__[:, idxREi0intneg_] if REIMi0intnegUnknown else zeros((Nf, Nneg)) # 负极交换电流密度实部
            self.IMi0intneg__[:] = X__[:, idxIMi0intneg_] if REIMi0intnegUnknown else zeros((Nf, Nneg)) # 负极交换电流密度虚部
            self.REi0intpos__[:] = X__[:, idxREi0intpos_] if REIMi0intposUnknown else zeros((Nf, Npos)) # 正极交换电流密度实部
            self.IMi0intpos__[:] = X__[:, idxIMi0intpos_] if REIMi0intposUnknown else zeros((Nf, Npos)) # 正极交换电流密度虚部
            self.REηintneg__[:] = X__[:, idxREηintneg_]  # 负极过电位实部
            self.IMηintneg__[:] = X__[:, idxIMηintneg_]  # 负极过电位虚部
            self.REηintpos__[:] = X__[:, idxREηintpos_]  # 正极过电位实部
            self.IMηintpos__[:] = X__[:, idxIMηintpos_]  # 正极过电位虚部
            if lithiumPlating:
                self.REjLP__[:] = X__[:, idxREjLP_]  # 负极析锂局部体积电流密度实部
                self.IMjLP__[:] = X__[:, idxIMjLP_]  # 负极析锂局部体积电流密度虚部
                self.REηLP__[:] = X__[:, idxREηLP_]  # 负极析锂过电位实部
                self.IMηLP__[:] = X__[:, idxIMηLP_]  # 负极析锂过电位虚部
        if self.verbose:
            print(f'计算时刻t = {tEIS:.1f}s 电化学阻抗谱')
        self.record_EISdata()  # 记录阻抗数据
        return self

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
    def Z_(self):
        """全电池复阻抗 [Ω]"""
        ΔiAC, ΔIAC = self.ΔiAC, self.ΔIAC
        REφsnegCollector_ = self.REφsneg__[:, 0] + 0.5*self.Δxneg*ΔiAC/self.σeffneg   # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = self.IMφsneg__[:, 0]                                      # (Nf,) 负极集流体电势虚部 [V]
        REφsposCollector_ = self.REφspos__[:, -1] - 0.5*self.Δxpos*ΔiAC/self.σeffpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = self.IMφspos__[:, -1]                                     # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗虚部 [Ω]
        return Zreal_ + 1j*Zimag_ + self.Zl_

    @property
    def Zl_(self):
        """全电池感抗 [Ω]"""
        return 1j*self.ωl_

    @property
    def Zneg_(self):
        """负极复阻抗 [Ω]"""
        Nneg = self.Nneg
        Δx_ = self.Δx_
        nW, nE = Nneg - 1, Nneg
        κeffWest_ = κeffEast_ = self.κeff_
        REφe__, IMφe__ = self.REφe__, self.IMφe__
        a, b = κeffWest_[nE]*Δx_[nW], κeffEast_[nW]*Δx_[nE]
        den = a + b
        REφenegsep_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
        IMφenegsep_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
        REφsnegCollector_ = self.REφsneg__[:, 0] + 0.5*self.Δxneg*self.ΔiAC/self.σeffneg  # 负极集流体电势实部 [V]
        IMφsnegCollector_ = self.IMφsneg__[:, 0]                                          # 负极集流体电势虚部 [V]
        ΔIAC = self.ΔIAC
        Zreal_ = (REφenegsep_ - REφsnegCollector_)/-ΔIAC  # 负极阻抗实部 [Ω]
        Zimag_ = (IMφenegsep_ - IMφsnegCollector_)/-ΔIAC  # 负极阻抗虚部 [Ω]
        return Zreal_ + 1j*Zimag_

    @property
    def Zpos_(self):
        """正极复阻抗 [Ω]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_ = self.Δx_
        nW, nE = Nneg + Nsep - 1, Nneg + Nsep
        κeffWest_ = κeffEast_ = self.κeff_
        REφe__, IMφe__ = self.REφe__, self.IMφe__
        a, b = κeffWest_[nE]*Δx_[nW], κeffEast_[nW]*Δx_[nE]
        den = a + b
        REφeseppos_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
        IMφeseppos_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
        REφsposCollector_ = self.REφspos__[:, -1] - 0.5*self.Δxpos*self.ΔiAC/self.σeffpos  # 正极集流体电势实部 [V]
        IMφsposCollector_ = self.IMφspos__[:, -1]                                          # 正极集流体电势虚部 [V]
        ΔIAC = self.ΔIAC
        Zreal_ = (REφsposCollector_ - REφeseppos_)/-ΔIAC  # 正极阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφeseppos_)/-ΔIAC  # 正极阻抗虚部 [Ω]
        return Zreal_ + 1j*Zimag_  # 正极复阻抗 [Ω]

    @property
    def Zsep_(self):
        """隔膜复阻抗 [Ω]"""
        return self.Z_ - self.Zneg_ - self.Zpos_ - self.Zl_   # 隔膜复阻抗 [Ω]

    @property
    def ΔiAC(self):
        """交流扰动电流密度振幅 [A/m^2]"""
        return self.ΔIAC/self.A

    @property
    def ω_(self):
        """角频率序列 [rad/s]"""
        return 2*pi*self.f_

    @property
    def ωl_(self):
        """感抗序列 [Ω]"""
        return self.ω_*self.l

    @staticmethod
    def solve_Kcssurf__(
            ω: float,   # 角频率 [rad/s]
            Rs: float,  # 颗粒半径 [m]
            Ds: float,  # 固相扩散系数 [m^2/s]
            a: float,   # 比表面积 [m^2/m^3]
            ):
        """求Kcssurf__矩阵，Kcssurf__ @ [REj_, IMj_] = [REcssurf_, IMcssurf_]"""
        W2 = ω*Rs**2/Ds
        W = W2**0.5
        root2 = 2**0.5
        γ = root2/2*W
        cosγ, sinγ = cos(γ), sin(γ)
        # 处理大数运算
        cosγ2, sinγ2 = Decimal(cosγ**2), Decimal(sinγ**2)
        cosγsinγ = Decimal(cosγ*sinγ)
        γ = Decimal(γ)
        W, W2 = Decimal(W), Decimal(W2)
        Rs = Decimal(Rs)
        aFDs = Decimal(a*DFNP2D.F*Ds)
        root2 = Decimal(root2)
        expγ, exp_γ = γ.exp(), (-γ).exp()
        coshγ = (expγ + exp_γ)/2
        sinhγ = (expγ - exp_γ)/2
        coshγ2 = coshγ**2
        coshγsinhγ = coshγ*sinhγ
        A = -root2*Rs*( W*coshγsinhγ
                      + W*cosγsinγ
                      + root2*cosγ2
                      - root2*coshγ2)
        B = -root2*W*Rs*(coshγsinhγ - cosγsinγ)
        den = 2*aFDs*((W2 + 1)*coshγ2
                      - root2*W*coshγsinhγ
                      - W2*sinγ2
                      - root2*W*cosγsinγ
                      - cosγ2)
        A /= den
        B /= den
        A, B = float(A), float(B)
        Kcssurf__ = array([[A,  B],
                              [-B, A]])
        return Kcssurf__

    @staticmethod
    def solve_REcs_IMcs(
            r: float,   # 径向坐标 [m]
            ω: float,   # 角频率 [rad/s]
            Rs: float,  # 颗粒半径 [m]
            Ds: float,  # 固相扩散系数 [m^2/s]
            a: float,   # 比表面积 [m^2/m^3]
            REjint_: ndarray,
            IMjint_: ndarray,
            ):
        """固相浓度实部、虚部在r处的解析解"""
        root2 = 2**0.5
        W2 = ω*Rs**2/Ds
        W = W2**0.5
        Y = r/Rs
        γ = root2/2*W
        γY = γ*Y
        sinγY, cosγY = Decimal(sin(γY)), Decimal(cos(γY))
        sinγ, cosγ   = Decimal(sin(γ)), Decimal(cos(γ))
        # 处理大数运算
        γY = Decimal(γY)
        γ = Decimal(γ)
        W, W2 =  Decimal(W), Decimal(W2)
        Rs = Decimal(Rs)
        aFDsr = Decimal(a*DFNP2D.F*Ds*r)
        root2 = Decimal(root2)

        expγY, exp_γY = γY.exp(), (-γY).exp()
        coshγY = (expγY + exp_γY)*0.5
        sinhγY = (expγY - exp_γY)*0.5
        expγ, exp_γ = γ.exp(), (-γ).exp()
        coshγ = (expγ + exp_γ)*0.5
        sinhγ = (expγ - exp_γ)*0.5
        A = -root2*Rs**2*(  W*coshγY*sinγY*coshγ*cosγ
                          - root2*coshγY*sinγY*coshγ*sinγ
                          + W*coshγY*sinγY*sinhγ*sinγ
                          + W*sinhγY*cosγY*coshγ*cosγ 
                          - W*sinhγY*cosγY*sinhγ*sinγ
                          - root2*sinhγY*cosγY*sinhγ*cosγ)
        B = -root2*Rs**2*(- W*coshγY*sinγY*coshγ*cosγ
                          + W*coshγY*sinγY*sinhγ*sinγ
                          + root2*coshγY*sinγY*sinhγ*cosγ
                          + W*sinhγY*cosγY*coshγ*cosγ
                          - root2*sinhγY*cosγY*coshγ*sinγ
                          + W*sinhγY*cosγY*sinhγ*sinγ)
        den = 2*aFDsr*((W2 + 1)*coshγ**2
                       - root2*W*coshγ*sinhγ
                       - W2*sinγ**2
                       - root2*W*cosγ*sinγ
                       - cosγ**2)
        A /= den
        B /= den
        A, B = float(A), float(B)
        Kcs__ = array([[A,  B],
                          [-B, A]])
        REcs_, IMcs_ = Kcs__ @ [REjint_, IMjint_]
        return REcs_, IMcs_

    @property
    def djintdi0intneg_(self):
        """负极主反应局部体积电流密度jintneg对交换电流密度i0intneg的偏导数 [A/m^3 / A/m^2]"""
        return DFNP2D.solve_djintdi0int_(self.T, self.aeffneg, self.ηintneg_)

    @property
    def djintdi0intpos_(self):
        """正极主反应局部体积电流密度jintpos对交换电流密度i0intpos的偏导数 [A/m^3 / A/m^2]"""
        return DFNP2D.solve_djintdi0int_(self.T, self.aeffpos, self.ηintpos_)

    @property
    def djintdηintneg_(self):
        """负极主反应局部体积电流密度jintneg对过电位ηintneg的偏导数 [A/m^3 / V]"""
        return DFNP2D.solve_djintdηint_(self.T, self.aeffneg, self.i0intneg_, self.ηintneg_)

    @property
    def djintdηintpos_(self):
        """正极主反应局部体积电流密度jintpos对过电位ηintpos的偏导数 [A/m^3 / V]"""
        return DFNP2D.solve_djintdηint_(self.T, self.aeffpos, self.i0intpos_, self.ηintpos_)

    @property
    def di0intdceneg_(self):
        """负极主反应交换电流密度i0int对电解液浓度ce的偏导数 [A/m2 / mol/m^3]"""
        return 0 if self._i0intneg \
            else self.solve_di0intdce_(self.ceneg_, self.i0intneg_)

    @property
    def di0intdcepos_(self):
        """正极主反应交换电流密度i0int对电解液浓度ce的偏导数 [A/m2 / mol/m^3]"""
        return 0 if self._i0intpos \
            else self.solve_di0intdce_(self.cepos_, self.i0intpos_)

    @property
    def di0intdcsnegsurf_(self):
        """负极主反应交换电流密度i0intneg对电极表面浓度csnegsurf的偏导数  [A/-]"""
        return 0 if self._i0intneg\
            else self.solve_di0intdcssurf_(self.kneg, self.csmaxneg,
                                           self.csnegsurf_, self.ceneg_, self.i0intneg_)

    @property
    def di0intdcspossurf_(self):
        """正极主反应交换电流密度i0intpos对电极表面浓度cspossurf的偏导数"""
        return 0 if self._i0intpos\
            else self.solve_di0intdcssurf_(self.kpos, self.csmaxpos,
                                           self.cspossurf_, self.cepos_, self.i0intpos_)

    @property
    def djLPdce_(self):
        """析锂反应局部体积电流密度jLP对电解液浓度ce的偏导数 [A/m^3 / mol/m^3]"""
        return 0 if self._i0LP \
            else DFNP2D.solve_djLPdce_(self.T, self.aeffneg,
                                       self.ceneg_, self.i0LP_, self.ηLPneg_)

    @property
    def djLPdηLP_(self):
        """析锂反应局部体积电流密度jLP对析锂过电位ηLP的偏导数 [A/m^3 / V]"""
        return DFNP2D.solve_djLPdηLP_(self.T, self.aeffneg, self.i0LP_, self.ηLPneg_)

    @property
    def dUOCPdcsnegsurf_(self):
        """负极电位UOCPnegsurf对负极表面锂离子浓度cssurf的导数 [V/mol/^3]"""
        csmaxneg = self.csmaxneg
        return self.solve_dUOCPdθsneg_(self.csnegsurf_/csmaxneg) / csmaxneg

    @property
    def dUOCPdcspossurf_(self):
        """正极电位UOCPpossurf对负极表面锂离子浓度cssurf的导数 [V/mol/^3]"""
        csmaxpos = self.csmaxpos
        return self.solve_dUOCPdθspos_(self.cspossurf_/csmaxpos) / csmaxpos

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
            return DFNP2D.interpolate(self, variableName, t_, x_, r_)

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
        Zsep_ = self.interpolate('Zsep_', t_=t_, f_=f)
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
        ax1.plot(t_, Zsep_.real*1000, 'g-d', label=r'${\it Z}_{sep}$')
        ax1.plot(t_, Zpos_.real*1000, 'r-s', label=r'${\it Z}_{neg}$')
        ax1.set_ylabel(r'${\it Z}′$ [mΩ]')
        ax1.set_xticks([])
        ax1.legend(facecolor='none', edgecolor='none', framealpha=0.8,
                   ncols=4, fontsize=16, loc=[0.2, 1.02], )

        ax2.plot(t_, -Z_.imag*1000, 'k-o', label='Full cell')
        ax2.plot(t_, -Zneg_.imag*1000, 'b-^')
        ax2.plot(t_, -Zsep_.imag*1000, 'g-d')
        ax2.plot(t_, -Zpos_.imag*1000, 'r-s')
        ax2.set_ylabel(r'$-{\it Z}″$ [mΩ]')
        ax2.set_xticks([])

        ax3.plot(t_, abs(Z_)*1000, 'k-o')
        ax3.plot(t_, abs(Zneg_)*1000, 'b-^')
        ax3.plot(t_, abs(Zsep_)*1000, 'g-d')
        ax3.plot(t_, abs(Zpos_)*1000, 'r-s')
        ax3.set_ylabel(r'$|{\it Z}|$ [mΩ]')
        ax3.set_xticks([])
        from numpy import angle

        ax4.plot(t_, -angle(Z_, deg=True), 'k-o')
        ax4.plot(t_, -angle(Zneg_, deg=True), 'b-^')
        ax4.plot(t_, -angle(Zsep_, deg=True), 'g-d')
        ax4.plot(t_, -angle(Zpos_, deg=True), 'r-s')
        ax4.set_ylabel(r'$-∠{\it Z}$ [°]')
        ax4.set_xlabel(r'Time $\it t$ [s]')

        duration = ptp(t_)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim([t_[0]-duration*0.02, t_[-1]+duration*0.02])
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
            ax.plot(Z_.real, -Z_.imag, 'o-', color=DFNP2D.get_color(t_, n),
                    label=rf'$\it t$ = {t:g} s')
        ax.set_ylabel(rf'Imaginary part of impedance $-{{\it Z″}}_{{{Z[1:-1]}}}\;\;{{[mΩ]}}$')
        ax.set_xlabel(rf'Real part of impedance ${{\it Z′}}_{{{Z[1:-1]}}}\;\;{{[mΩ]}}$')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(linestyle='--')

        i = ptp(abs(array(Z__)), axis=1).argmax()
        for x, y, f in zip(Z__[i].real, -Z__[i].imag, f_):
            ax.text(x, y,
                    f'  {f:g} Hz',
                    backgroundcolor='w',
                    va='center', ha='left',
                    fontsize=10,
                    bbox=dict(boxstyle='square,pad=0.4', fc='none', ec='none', lw=0.5, alpha=0.8)
                    )
        plt.show()

    def plot_REcssurfIMcssurf(self,
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
                     color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(f'{self.cSign}′$_{{s,AC}}$({self.xSign}, {self.rSign}; ${{\\it f}}$, ${{\\it t}}$)|$_{{ {self.rSign[1:-1]} = {"{\\it R}_{s,reg}" if self.rUnit else 1} }}$ [{self.cUnit or '–'}]')
        self.plot_interfaces(ax1)
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMcsnegsurf_, IMcspossurf_, label) in enumerate(zip(IMcsnegsurf__, IMcspossurf__, labels_)):
            ax2.plot(self.xPlot_, hstack([IMcsnegsurf_, full(self.Nsep, nan), IMcspossurf_]), 'o-',
                     color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        self.plot_interfaces(ax2)
        plt.show()

    def plot_REceIMce(self,
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
            ax1.plot(self.xPlot_, REce_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.cSign}′$_{{e,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.cUnit or '–'}]')
        self.plot_interfaces(ax1)
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMce_, label) in enumerate(zip(IMce__, labels_)):
            ax2.plot(self.xPlot_, IMce_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        self.plot_interfaces(ax2)
        plt.show()

    def plot_REφsIMφs(self,
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
            ax1.plot(self.xPlot_[:self.Nneg], REφsneg_*1e3, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'${{\it φ′}}_{{s,neg,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [mV]')
        ax1.set_xlim(0, self.xInterfacesPlot_[self.Nneg])

        for n, (REφspos_, label) in enumerate(zip(REφspos__, labels_)):
            ax2.plot(self.xPlot_[-self.Npos:], REφspos_*1e3, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('neg', 'pos'))
        ax2.set_xlim(self.xInterfacesPlot_[self.Nneg+self.Nsep], self.xInterfacesPlot_[-1])
        ax2.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMφsneg_, label) in enumerate(zip(IMφsneg__, labels_)):
            ax3.plot(self.xPlot_[:self.Nneg], IMφsneg_*1e3, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax3.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        ax3.set_xlim(0, self.xInterfacesPlot_[self.Nneg])

        for n, (IMφspos_, label) in enumerate(zip(IMφspos__, labels_)):
            ax4.plot(self.xPlot_[-self.Npos:], IMφspos_*1e3, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax4.set_ylabel(ax3.get_ylabel().replace('neg', 'pos'))
        ax4.set_xlim(self.xInterfacesPlot_[self.Nneg+self.Nsep], self.xInterfacesPlot_[-1])

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '—'}]')
            ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_REφeIMφe(self,
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
            ax1.plot(self.xPlot_, REφe_*1e3, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'${{\it φ′}}_{{e,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [mV]')
        self.plot_interfaces(ax1)
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMφe_, label) in enumerate(zip(IMφe__, labels_)):
            ax2.plot(self.xPlot_, IMφe_*1e3, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        self.plot_interfaces(ax2)
        plt.show()

    def plot_REjintIMjint(self,
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
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.jSign}′$_{{int,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.jUnit}]')
        self.plot_interfaces(ax1)
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMjintneg_, IMjintpos_, label) in enumerate(zip(IMjintneg__, IMjintpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMjintneg_, *([nan]*self.Nsep), *IMjintpos_
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        self.plot_interfaces(ax2)
        plt.show()

    def plot_REjDLIMjDL(self,
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
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.jSign}′$_{{DL,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.jUnit}]')
        self.plot_interfaces(ax1)
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMjDLneg_, IMjDLpos_, label) in enumerate(zip(IMjDLneg__, IMjDLpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMjDLneg_, *([nan]*self.Nsep), *IMjDLpos_
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        self.plot_interfaces(ax2)
        plt.show()

    def plot_REi0intIMi0int(self,
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
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'{self.i0Sign.replace('}', '′}').replace('0', '{0,AC}')}({self.xSign}; ${{\it f}}$, ${{\it t}}$) [{self.i0Unit}]')
        self.plot_interfaces(ax1)
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMi0intneg_, IMi0intpos_, label) in enumerate(zip(IMi0intneg__, IMi0intpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMi0intneg_, *([nan]*self.Nsep), *IMi0intpos_
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        self.plot_interfaces(ax2)
        plt.show()

    def plot_REηintIMηint(self,
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
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax1.set_ylabel(rf'${{\it η′}}_{{int,AC}}$({self.xSign}; ${{\it f}}$, ${{\it t}}$) [mV]')
        self.plot_interfaces(ax1)
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (IMηintneg_, IMηintpos_, t) in enumerate(zip(IMηintneg__, IMηintpos__, labels_)):
            x_ = self.xPlot_
            y_ = *IMηintneg_, *([nan]*self.Nsep), *IMηintpos_
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(labels_, n), label=label)
        ax2.set_ylabel(ax1.get_ylabel().replace('′', '″'))
        self.plot_interfaces(ax2)
        plt.show()

    def check_EIS(self):
        if not self.complete:
            print('complete==True的前提下才可检验频域控制方程')
            return
        if self.tEIS!=self.t:
            print('应在完成最新EIS计算后立刻检查结果')
            return
        print('='*100)
        print(f'检验频域控制方程：')
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos
        RSEI2aeffneg, RSEI2aeffpos = self.RSEIneg/self.aeffneg, self.RSEIpos/self.aeffpos
        REcsnegsurf__ = self.REcsnegsurf__  # 负极固相表面浓度实部
        IMcsnegsurf__ = self.IMcsnegsurf__  # 负极固相表面浓度虚部
        REcspossurf__ = self.REcspossurf__  # 正极固相表面浓度实部
        IMcspossurf__ = self.IMcspossurf__  # 正极固相表面浓度虚部
        REce__ = self.REce__            # 电解液锂离子浓度实部
        IMce__ = self.IMce__            # 电解液锂离子浓度虚部
        REφsneg__ = self.REφsneg__      # 负极固相电势实部
        IMφsneg__ = self.IMφsneg__       # 负极固相电势虚部
        REφspos__ = self.REφspos__      # 正极固相电势实部
        IMφspos__ = self.IMφspos__      # 正极固相电势虚部
        REφe__ = self.REφe__            # 电解液电势实部
        IMφe__ = self.IMφe__            # 电解液电势虚部
        REjintneg__ = self.REjintneg__  # 负极局部体积电流实部
        IMjintneg__ = self.IMjintneg__  # 负极局部体积电流虚部
        REjintpos__ = self.REjintpos__  # 正极局部体积电流实部
        IMjintpos__ = self.IMjintpos__  # 正极局部体积电流虚部
        REjDLneg__ = self.REjDLneg__    # 负极双电层局部体积电流实部
        IMjDLneg__ = self.IMjDLneg__    # 负极双电层局部体积电流虚部
        REjDLpos__ = self.REjDLpos__    # 正极双电层局部体积电流实部
        IMjDLpos__ = self.IMjDLpos__    # 正极双电层局部体积电流虚部
        REi0intneg__ = self.REi0intneg__  # 负极交换电流实部
        IMi0intneg__ = self.IMi0intneg__  # 负极交换电流虚部
        REi0intpos__ = self.REi0intpos__  # 正极交换电流实部
        IMi0intpos__ = self.IMi0intpos__  # 正极交换电流虚部
        REηintneg__ = self.REηintneg__  # 负极过电位实部
        IMηintneg__ = self.IMηintneg__  # 负极过电位虚部
        REηintpos__ = self.REηintpos__  # 正极过电位实部
        IMηintpos__ = self.IMηintpos__  # 正极过电位虚部
        if self.lithiumPlating:
            REjLP__ = self.REjLP__  # 负极析锂局部体积电流密度实部
            IMjLP__ = self.IMjLP__  # 负极析锂局部体积电流密度虚部
            REηLP__ = self.REηLP__  # 负极析锂过电位实部
            IMηLP__ = self.IMηLP__  # 负极析锂过电位虚部
        else:
            REjLP__ = 0
            IMjLP__ = 0
            REηLP__ = REφsneg__ - REφe__[:, :Nneg] - RSEI2aeffneg*(REjintneg__ + REjDLneg__)
            IMηLP__ = IMφsneg__ - IMφe__[:, :Nneg] - RSEI2aeffpos*(IMjintneg__ + IMjDLneg__)
        Nf = self.f_.size
        ω_ = self.ω_
        F = DFNP2D.F
        F2RT = F/2/DFNP2D.R/self.T
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        ΔIAC, ΔiAC = self.ΔIAC, self.ΔiAC
        σeffneg, σeffpos = self.σeffneg, self.σeffpos
        aeffneg, aeffpos = self.aeffneg, self.aeffpos
        εe_ = self.εe_
        DeeffWest_ = DeeffEast_ = self.Deeff_

        # 各控制体界面的电解液锂离子浓度实部 [mol/m^3]
        REceInterfaces__ = hstack([REce__[:, [0]], (REce__[:, :-1] + REce__[:, 1:])/2, REce__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 利用边界条件修正负极-隔膜界面、隔膜-正极界面锂离子浓度
            REceInterfaces__[:, nE] = (DeeffEast_[nW]*REce__[:, nW]*Δx_[nE] + DeeffWest_[nE]*REce__[:, nE]*Δx_[nW])/(
                                       DeeffEast_[nW]*Δx_[nE] + DeeffWest_[nE]*Δx_[nW])
        REceWest__ = REceInterfaces__[:, :-1]  # 各控制体左界面的电解液锂离子浓度 [–]
        REceEast__ = REceInterfaces__[:, 1:]   # 各控制体右界面的电解液锂离子浓度 [–]
        gradREceWest__ = hstack([zeros([Nf, 1]), (REce__[:, 1:] - REce__[:, :-1])/ΔxWest_[1:]])   # 各控制体左界面的锂离子浓度梯度实部 [mol/m^4]
        gradREceEast__ = hstack([(REce__[:, 1:] - REce__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # 各控制体右界面的锂离子浓度梯度实部 [mol/m^4]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradREceEast__[:, nW] = (REceEast__[:, nW] - REce__[:, nW])/(0.5*Δx_[nW])
            gradREceWest__[:, nE] = (REce__[:, nE] - REceWest__[:, nE])/(0.5*Δx_[nE])

        # 各控制体界面的电解液锂离子浓度虚部 [mol/m^3]
        IMceInterfaces__ = hstack([IMce__[:, [0]], (IMce__[:, :-1] + IMce__[:, 1:])/2, IMce__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 利用边界条件修正负极-隔膜界面、隔膜-正极界面锂离子浓度
            IMceInterfaces__[:, nE] = (DeeffEast_[nW]*IMce__[:, nW]*Δx_[nE] + DeeffWest_[nE]*IMce__[:, nE]*Δx_[nW])/(DeeffEast_[nW]*Δx_[nE] + DeeffWest_[nE]*Δx_[nW])
        IMceWest__ = IMceInterfaces__[:, :-1]  # 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        IMceEast__ = IMceInterfaces__[:, 1:]   # 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradIMceWest__ = hstack([zeros([Nf, 1]), (IMce__[:, 1:] - IMce__[:, :-1])/ΔxWest_[1:]])   # 各控制体左界面的锂离子浓度梯度虚部 [mol/m^4]
        gradIMceEast__ = hstack([(IMce__[:, 1:] - IMce__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # 各控制体右界面的锂离子浓度梯度虚部 [mol/m^4]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradIMceEast__[:, nW] = (IMceEast__[:, nW] - IMce__[:, nW])/(0.5*Δx_[nW])
            gradIMceWest__[:, nE] = (IMce__[:, nE] - IMceWest__[:, nE])/(0.5*Δx_[nE])

        c = self.solve_frequency_dependent_variables()
        maxError = max([
            abs(array([REcsnegsurf__, IMcsnegsurf__]).transpose(1, 0, 2) - -c['minusKcsnegsurf___'] @ array([REjintneg__, IMjintneg__]).transpose(1, 0, 2)).max(),
            abs(array([REcspossurf__, IMcspossurf__]).transpose(1, 0, 2) - -c['minusKcspossurf___'] @ array([REjintpos__, IMjintpos__]).transpose(1, 0, 2)).max(), ])
        print(f'固相表面浓度解析解方程最大误差{maxError} mol/m^3')

        LHS__ = -outer(ω_, εe_) * IMce__
        RHS__ = (DeeffEast_*gradREceEast__ - DeeffWest_*gradREceWest__)/Δx_ + (1 - self.tplus)/F*hstack([REjintneg__+REjDLneg__+REjLP__, zeros([Nf, Nsep]), REjintpos__+REjDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液实部浓度方程最大误差{maxError}')

        LHS__ = outer(ω_, εe_) * REce__
        RHS__ = (DeeffEast_*gradIMceEast__ - DeeffWest_*gradIMceWest__)/Δx_ + (1 - self.tplus)/F*hstack([IMjintneg__+IMjDLneg__+IMjLP__, zeros([Nf, Nsep]), IMjintpos__+IMjDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液虚部浓度方程最大误差{maxError}')

        gradREφsnegInterfaces__ = hstack([full([Nf, 1], -ΔiAC/σeffneg), (REφsneg__[:, 1:] - REφsneg__[:, :-1])/Δxneg, zeros([Nf, 1])])
        ΔREφsneg_ = (gradREφsnegInterfaces__[:, 1:] - gradREφsnegInterfaces__[:, :-1])/Δxneg
        gradIMφsnegInterfaces_ = hstack([zeros([Nf, 1]), (IMφsneg__[:, 1:] - IMφsneg__[:, :-1])/Δxneg, zeros([Nf, 1])])
        ΔIMφsneg_ = (gradIMφsnegInterfaces_[:, 1:] - gradIMφsnegInterfaces_[:, :-1])/Δxneg
        RE_LHS__ = σeffneg*ΔREφsneg_
        RE_RHS__ = REjintneg__ + REjDLneg__ + REjLP__
        IM_LHS__ = σeffneg*ΔIMφsneg_
        IM_RHS__ = IMjintneg__ + IMjDLneg__+ IMjLP__
        maxError = max([abs(RE_LHS__ - RE_RHS__).max(),
                        abs(IM_LHS__ - IM_RHS__).max(),])
        print(f'负极固相电势方程最大误差{maxError} A/m^3')

        gradREφsposInterfaces_ = hstack([zeros([Nf, 1]), (REφspos__[:, 1:] - REφspos__[:, :-1])/Δxpos, full([Nf, 1], -ΔiAC/σeffpos)])
        ΔREφspos__ = (gradREφsposInterfaces_[:, 1:] - gradREφsposInterfaces_[:, :-1])/Δxpos
        gradIMφsposInterfaces__ = hstack([zeros([Nf, 1]), (IMφspos__[:, 1:] - IMφspos__[:, :-1])/Δxpos, zeros([Nf, 1])])
        ΔIMφspos__ = (gradIMφsposInterfaces__[:, 1:] - gradIMφsposInterfaces__[:, :-1])/Δxpos
        RE_LHS__ = σeffpos*ΔREφspos__
        RE_RHS__ = REjintpos__ + REjDLpos__
        IM_LHS__ = σeffpos*ΔIMφspos__
        IM_RHS__ = IMjintpos__ + IMjDLpos__
        maxError = max([abs(RE_LHS__ - RE_RHS__).max(),
                        abs(IM_LHS__ - IM_RHS__).max(), ])
        print(f'正极固相电势方程最大误差{maxError} A/m^3')

        i0intneg_, i0intpos_ = self.i0intneg_, self.i0intpos_
        ηintneg_, ηintpos_ = self.ηintneg_, self.ηintpos_
        maxError = max([
            abs(REjintneg__ - 2*aeffneg*(REi0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMjintneg__ - 2*aeffneg*(IMi0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REjintpos__ - 2*aeffpos*(REi0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMjintpos__ - 2*aeffpos*(IMi0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),])
        print(f'主反应BV动力学方程最大误差{maxError} A/m^3')

        di0intdceneg_, di0intdcepos_ = self.di0intdceneg_, self.di0intdcepos_
        di0intdcsnegsurf_, di0intdcspossurf_ = self.di0intdcsnegsurf_, self.di0intdcspossurf_
        maxError = max([
            abs(REi0intneg__ - (di0intdceneg_*REce__[:, :Nneg] + di0intdcsnegsurf_*REcsnegsurf__)).max(),
            abs(IMi0intneg__ - (di0intdceneg_*IMce__[:, :Nneg] + di0intdcsnegsurf_*IMcsnegsurf__)).max(),
            abs(REi0intpos__ - (di0intdcepos_*REce__[:, -Npos:] + di0intdcspossurf_*REcspossurf__)).max(),
            abs(IMi0intpos__ - (di0intdcepos_*IMce__[:, -Npos:] + di0intdcspossurf_*IMcspossurf__)).max(), ])
        print(f'主反应交换电流密度方程最大误差{maxError} A/m^2')

        dUOCPdcsnegsurf_, dUOCPdcspossurf_ = self.dUOCPdcsnegsurf_, self.dUOCPdcspossurf_
        maxError = max([
            abs(REηintneg__ - (REφsneg__ - REφe__[:, :Nneg] - dUOCPdcsnegsurf_*REcsnegsurf__ - RSEI2aeffneg*(REjintneg__ + REjDLneg__ + REjLP__))).max(),
            abs(IMηintneg__ - (IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdcsnegsurf_*IMcsnegsurf__ - RSEI2aeffneg*(IMjintneg__ + IMjDLneg__ + IMjLP__))).max(),
            abs(REηintpos__ - (REφspos__ - REφe__[:, -Npos:] - dUOCPdcspossurf_*REcspossurf__ - RSEI2aeffpos*(REjintpos__ + REjDLpos__))).max(),
            abs(IMηintpos__ - (IMφspos__ - IMφe__[:, -Npos:] - dUOCPdcspossurf_*IMcspossurf__ - RSEI2aeffpos*(IMjintpos__ + IMjDLpos__))).max(),])
        print(f'主反应过电位方程最大误差{maxError} [V]')

        if self.lithiumPlating:
            maxError = max([
                abs(REjLP__ - (self.djLPdce_*REce__[:, :Nneg] + self.djLPdηLP_*REηLP__)).max(),
                abs(IMjLP__ - (self.djLPdce_*IMce__[:, :Nneg] + self.djLPdηLP_*IMηLP__)).max(),])
            print(f'析锂BV动力学方程最大误差{maxError} [A/m^3]')

            maxError = max([
                abs(REηLP__ - (REφsneg__ - REφe__[:, :Nneg] - RSEI2aeffneg*(REjintneg__ + REjDLneg__ + REjLP__))).max(),
                abs(IMηLP__ - (IMφsneg__ - IMφe__[:, :Nneg] - RSEI2aeffneg*(IMjintneg__ + IMjDLneg__ + IMjLP__))).max(),])
            print(f'析锂过电位方程最大误差{maxError} [V]')


if __name__=='__main__':
    import numpy as np
    cell = JTFP2D(
        Δt=10, SOC0=0.1,
        CDLpos=0.1, CDLneg=0.7,
        # i0intpos=3.67, i0intneg=3.30,
        Aeffneg=0.9, Aeffpos=0.8,
        Nneg=7, Nsep=8, Npos=7, Nr=10,
        f_=np.logspace(3, 0, 16),
        l=1e-7,
        lithiumPlating=True,
        # doubleLayerEffect=False,
        # constants=True,
        # complete=False
        )

    I = -10
    cell.count_lithium()
    thermalModel = True
    cell.EIS()
    cell.CC(I, 1500, thermalModel).EIS()
    cell.CC(-I, 2300, thermalModel).EIS()
    cell.CC(I, 2000, thermalModel).EIS()
    cell.CC(0, 3700, thermalModel).EIS()
    # cell.check_EIS()
    cell.count_lithium()

    '''
    cell.plot_UI()
    cell.plot_TQgen()
    cell.plot_SOC()
    cell.plot_c(arange(0, 2001, 200))
    cell.plot_φ(arange(0, 2001, 200))
    cell.plot_jint(arange(0, 2001, 200))
    cell.plot_jDL(arange(0, 2001, 200))
    cell.plot_i(arange(0, 2001, 200))
    cell.plot_csr(range(0, 2001, 200), 1)
    cell.plot_jLP(arange(4000, 4301, 100))
    cell.plot_ηLP()
    cell.plot_OCV()
    cell.plot_dUOCPdθs()
    
    cell.plot_Z(1)
    cell.plot_Nyquist()
    cell.plot_REcssurfIMcssurf()
    cell.plot_REceIMce()
    cell.plot_REφsIMφs()
    cell.plot_REφeIMφe()
    cell.plot_REjintIMjint()
    cell.plot_REjDLIMjDL()
    cell.plot_REi0intIMi0int()
    cell.plot_REηintIMηint()
    '''

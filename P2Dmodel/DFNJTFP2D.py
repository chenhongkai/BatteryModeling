#%%
from typing import Sequence
from decimal import Decimal
from math import cos, sin

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from numpy import ndarray,\
    array, zeros, full, hstack, \
    empty, logspace,\
    sinh, cosh, outer,\
    ix_
from numpy.linalg import solve

from P2Dmodel.JTFbase import JTFbase
from P2Dmodel.DFNP2D import DFNP2D
from P2Dmodel.P2Dbase import P2Dbase

class DFNJTFP2D(JTFbase, DFNP2D):
    """锂离子电池时频联合经典准二维模型 Doyle-Fuller-Newman Joint Time-Frequency Pseudo-two-Dimension model"""

    def __init__(self,
            f_: Sequence[float] = logspace(3, -1, 26),  # 频率序列 [Hz]
            **kwargs):
        DFNP2D.__init__(self, **kwargs)
        JTFbase.__init__(self, f_)
        lithiumPlating, complete, verbose = self.lithiumPlating, self.complete, self.verbose  # 读取：模式
        if complete:
            Nf, Nneg, Npos, Ne = len(f_), self.Nneg, self.Npos, self.Ne  # 读取：网格数
            # 状态量
            self.REcsnegsurf__, self.IMcsnegsurf__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极固相表面浓度实部、虚部
            self.REcspossurf__, self.IMcspossurf__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极固相表面浓度实部、虚部
            self.REce__, self.IMce__ = empty((Nf, Ne)), empty((Nf, Ne))                  # 电解液锂离子浓度实部、虚部
            self.REjintneg__, self.IMjintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))    # 负极主反应局部体积电流密度实部、虚部
            self.REjintpos__, self.IMjintpos__ = empty((Nf, Npos)), empty((Nf, Npos))    # 正极主反应局部体积电流密度实部、虚部
            self.REjDLneg__, self.IMjDLneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))      # 负极双电层局部体积电流密度实部、虚部
            self.REjDLpos__, self.IMjDLpos__ = empty((Nf, Npos)), empty((Nf, Npos))      # 正极双电层局部体积电流密度实部、虚部
            self.REi0intneg__, self.IMi0intneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极交换电流密度实部、虚部
            self.REi0intpos__, self.IMi0intpos__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极交换电流密度实部、虚部
            if lithiumPlating:
                self.REjLP__, self.IMjLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极析锂反应局部体积电流密度实部、虚部

            # 恒定量
            self.EISdatanames_.extend([            # 需记录的阻抗数据名称
                'Zneg_', 'Zpos_',                  # 负极、正极复阻抗 [Ω]
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
        self.data.update({EISdataname: [] for EISdataname in self.EISdatanames_})  # 字典：存储呈时间序列的阻抗数据
        N = self.generate_indices_of_frequency_domain_dependent_variables()  # 生成索引：频域因变量
        self.Kf__ = Kf__ = zeros((N, N))  # 频域因变量线性矩阵
        self.bKf_ = zeros(N)              # Kf__ @ X_ = bKf_
        # 对频域因变量线性矩阵Kf__赋参数相关值
        self.update_Kf__idxREce_idxREce_and_idxIMce_idxIMce_(Deeff_ := self.Deeff_, Deeff_)
        self.update_Kf__idxREce_idxREj_and_idxIMce_idxIMj_(self.tplus)
        self.update_Kf__idxREφsneg_idxREjneg_and_idxIMφsneg_idxIMjneg_(σeffneg := self.σeffneg)
        self.update_Kf__idxREφspos_idxREjpos_and_idxIMφspos_idxIMjpos_(σeffpos := self.σeffpos)
        self.update_bKf_idxREφsneg_0_and_idxREφspos_end(σeffneg, σeffpos)
        self.update_Kf__idxREφe_idxREφe_and_idxIMφe_idxIMφe_(κeff_ := self.κeff_, κeff_)
        self.update_Kf__idxREηintneg_idxREjneg_and_idxIMηintneg_idxIMjneg_(self.RSEIneg, self.aeffneg)
        self.update_Kf__idxREηintpos_idxREjpos_and_idxIMηintpos_idxIMjpos_(self.RSEIpos, self.aeffpos)
        if lithiumPlating:
            self.update_Kf__idxREηLP_idxREjneg_and_idxIMηLP_idxIMjneg_(self.RSEIneg, self.aeffneg)
        # 对频域因变量线性矩阵Kf__赋恒定值
        self.assign_Kf__with_constants()
        if verbose and type(self) is DFNJTFP2D:
            print(f'频域因变量线性矩阵 Kf__.shape = {Kf__.shape}')
            print(self)
            print('经典时频联合P2D模型(DFNJTFP2D)初始化完成!')

    def solve_frequency_dependent_variables(self) -> dict[str, ndarray]:
        """求解频率相关变量"""
        ω_ = self.ω_
        solve_Kcssurf__ = DFNJTFP2D.solve_Kcssurf__
        Rsneg, Rspos, CDLneg, CDLpos, Dsneg, Dspos = (
            self.Rsneg, self.Rspos, self.CDLneg, self.CDLpos, self.Dsneg, self.Dspos)
        aneg, apos = self.aneg, self.apos
        frequency_dependent_variables = {
            'ωεeΔx__': outer(ω_, self.εe_*self.Δx_),  # (Nf, Ne) 各频率各控制体的ω*εe*Δx值
            'ωaCDLneg_': ω_ * (self.aeffneg*CDLneg),     # (Nf,)
            'ωaCDLpos_': ω_ * (self.aeffpos*CDLpos),     # (Nf,)
            'ωCDLRSEIneg_': ω_ * (CDLneg*self.RSEIneg),  # (Nf,)
            'ωCDLRSEIpos_': ω_ * (CDLpos*self.RSEIpos),  # (Nf,)
            'minusKcsnegsurf___': -array([solve_Kcssurf__(ω, Rsneg, Dsneg, aneg) for ω in ω_]),   # (Nf, 2, 2) 负极各频率Kcssurf__矩阵
            'minusKcspossurf___': -array([solve_Kcssurf__(ω, Rspos, Dspos, apos) for ω in ω_]),}  # (Nf, 2, 2) 正极各频率Kcssurf__矩阵
        return frequency_dependent_variables

    def update_Kf__idxREce_idxREj_and_idxIMce_idxIMj_(self, tplus):
        # 更新Kf__矩阵REce行REj列、IMce行IMj列
        a = (1 - tplus)/P2Dbase.F
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

    def update_bKf_idxREφsneg_0_and_idxREφspos_end(self, σeffneg, σeffpos):
        bKf_ = self.bKf_
        ΔiAC = self.ΔiAC
        # 更新bKf_向量REφsneg首元
        bKf_[self.idxREφsneg_[0]] = -self.Δxneg*ΔiAC/σeffneg
        # 更新bKf_向量REφspos末元
        bKf_[self.idxREφspos_[-1]] = self.Δxpos*ΔiAC/σeffpos

    def EIS(self):
        """计算电化学阻抗谱"""
        tEIS = self.t     # 读取：当前时刻 [s]
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
        ΔiAC, ΔIAC = self.ΔiAC, self.ΔIAC
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        σeffneg, σeffpos = self.σeffneg, self.σeffpos
        κDeffWest_ = κDeffEast_ = self.κDeff_
        DeeffWest_ = DeeffEast_ = self.Deeff_

        if self.constants:
            cache = self.frequency_dependent_cache
        else:
            cache = self.solve_frequency_dependent_variables()
            κeffWest_ = κeffEast_ = self.κeff_
            self.update_Kf__idxREce_idxREce_and_idxIMce_idxIMce_(DeeffWest_, DeeffEast_)
            self.update_Kf__idxREce_idxREj_and_idxIMce_idxIMj_(self.tplus)
            self.update_Kf__idxREφsneg_idxREjneg_and_idxIMφsneg_idxIMjneg_(σeffneg)
            self.update_Kf__idxREφspos_idxREjpos_and_idxIMφspos_idxIMjpos_(σeffpos)
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

        REφsneg__ = X__[:, idxREφsneg_]  # 负极固相电势实部
        IMφsneg__ = X__[:, idxIMφsneg_]  # 负极固相电势虚部
        REφspos__ = X__[:, idxREφspos_]  # 正极固相电势实部
        IMφspos__ = X__[:, idxIMφspos_]  # 正极固相电势虚部

        self.tEIS = tEIS
        REφsnegCollector_ = REφsneg__[:, 0] + 0.5*Δxneg*ΔiAC/σeffneg   # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = IMφsneg__[:, 0]                            # (Nf,) 负极集流体电势虚部 [V]
        REφsposCollector_ = REφspos__[:, -1] - 0.5*Δxpos*ΔiAC/σeffpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = IMφspos__[:, -1]                           # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗虚部 [Ω]
        self.Z_[:] = Zreal_ + 1j*Zimag_ + self.Zl_

        if self.complete:
            self.REcsnegsurf__[:] = X__[:, idxREcsnegsurf_]  # 负极固相表面浓度实部
            self.IMcsnegsurf__[:] = X__[:, idxIMcsnegsurf_]  # 负极固相表面浓度虚部
            self.REcspossurf__[:] = X__[:, idxREcspossurf_]  # 正极固相表面浓度实部
            self.IMcspossurf__[:] = X__[:, idxIMcspossurf_]  # 正极固相表面浓度虚部
            self.REce__[:] = X__[:, idxREce_]  # 电解液锂离子浓度实部
            self.IMce__[:] = X__[:, idxIMce_]  # 电解液锂离子浓度虚部
            self.REφsneg__[:] = REφsneg__      # 负极固相电势实部
            self.IMφsneg__[:] = IMφsneg__      # 负极固相电势虚部
            self.REφspos__[:] = REφspos__      # 正极固相电势实部
            self.IMφspos__[:] = IMφspos__      # 正极固相电势虚部
            self.REφe__[:] = REφe__ = X__[:, idxREφe_]  # 电解液电势实部
            self.IMφe__[:] = IMφe__ = X__[:, idxIMφe_]  # 电解液电势虚部
            self.REjintneg__[:] = X__[:, idxREjintneg_]  # 负极主反应局部体积电流密度实部
            self.IMjintneg__[:] = X__[:, idxIMjintneg_]  # 负极主反应局部体积电流密度虚部
            self.REjintpos__[:] = X__[:, idxREjintpos_]  # 正极主反应局部体积电流密度实部
            self.IMjintpos__[:] = X__[:, idxIMjintpos_]  # 正极主反应局部体积电流密度虚部
            self.REjDLneg__[:] = X__[:, idxREjDLneg_]  # 负极双电层局部体积电流密度实部
            self.IMjDLneg__[:] = X__[:, idxIMjDLneg_]  # 负极双电层局部体积电流密度虚部
            self.REjDLpos__[:] = X__[:, idxREjDLpos_]  # 正极双电层局部体积电流密度实部
            self.IMjDLpos__[:] = X__[:, idxIMjDLpos_]  # 正极双电层局部体积电流密度虚部
            self.REi0intneg__[:] = X__[:, idxREi0intneg_] if REIMi0intnegUnknown else 0. # 负极交换电流密度实部
            self.IMi0intneg__[:] = X__[:, idxIMi0intneg_] if REIMi0intnegUnknown else 0. # 负极交换电流密度虚部
            self.REi0intpos__[:] = X__[:, idxREi0intpos_] if REIMi0intposUnknown else 0. # 正极交换电流密度实部
            self.IMi0intpos__[:] = X__[:, idxIMi0intpos_] if REIMi0intposUnknown else 0. # 正极交换电流密度虚部
            self.REηintneg__[:] = X__[:, idxREηintneg_]  # 负极过电位实部
            self.IMηintneg__[:] = X__[:, idxIMηintneg_]  # 负极过电位虚部
            self.REηintpos__[:] = X__[:, idxREηintpos_]  # 正极过电位实部
            self.IMηintpos__[:] = X__[:, idxIMηintpos_]  # 正极过电位虚部
            if lithiumPlating:
                self.REjLP__[:] = X__[:, idxREjLP_]  # 负极析锂局部体积电流密度实部
                self.IMjLP__[:] = X__[:, idxIMjLP_]  # 负极析锂局部体积电流密度虚部
                self.REηLP__[:] = X__[:, idxREηLP_]  # 负极析锂过电位实部
                self.IMηLP__[:] = X__[:, idxIMηLP_]  # 负极析锂过电位虚部

            nW, nE = Nneg - 1, Nneg
            κeffWest_ = κeffEast_ = self.κeff_
            a, b = κeffWest_[nE]*Δx_[nW], κeffEast_[nW]*Δx_[nE]
            den = a + b
            REφenegsep_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφenegsep_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφenegsep_ - REφsnegCollector_)/-ΔIAC  # 负极阻抗实部 [Ω]
            Zimag_ = (IMφenegsep_ - IMφsnegCollector_)/-ΔIAC  # 负极阻抗虚部 [Ω]
            self.Zneg_[:] = Zreal_ + 1j*Zimag_  # 负极复阻抗 [Ω]

            nW, nE = Nneg + Nsep - 1, Nneg + Nsep
            a, b = κeffWest_[nE]*Δx_[nW], κeffEast_[nW]*Δx_[nE]
            den = a + b
            REφeseppos_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφeseppos_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφsposCollector_ - REφeseppos_)/-ΔIAC  # 正极阻抗实部 [Ω]
            Zimag_ = (IMφsposCollector_ - IMφeseppos_)/-ΔIAC  # 正极阻抗虚部 [Ω]
            self.Zpos_[:] = Zreal_ + 1j*Zimag_  # 正极复阻抗 [Ω]

        if self.verbose:
            print(f'计算时刻t = {tEIS:.1f}s 电化学阻抗谱')
        self.record_EISdata()  # 记录阻抗数据
        return self

    @property
    def ΔiAC(self):
        """交流扰动电流密度振幅 [A/m^2]"""
        return self.ΔIAC/self.A

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
        aFDs = Decimal(a*P2Dbase.F*Ds)
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
        aFDsr = Decimal(a*P2Dbase.F*Ds*r)
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
        F2RT = P2Dbase.F/2/P2Dbase.R/self.T
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
        RHS__ = (DeeffEast_*gradREceEast__ - DeeffWest_*gradREceWest__)/Δx_ + (1 - self.tplus)/P2Dbase.F*hstack([REjintneg__ + REjDLneg__ + REjLP__, zeros([Nf, Nsep]), REjintpos__ + REjDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液实部浓度方程最大误差{maxError}')

        LHS__ = outer(ω_, εe_) * REce__
        RHS__ = (DeeffEast_*gradIMceEast__ - DeeffWest_*gradIMceWest__)/Δx_ + (1 - self.tplus)/P2Dbase.F*hstack([IMjintneg__ + IMjDLneg__ + IMjLP__, zeros([Nf, Nsep]), IMjintpos__ + IMjDLpos__])
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
    cell = DFNJTFP2D(
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
    cell.plot_c(np.arange(0, 2001, 200))
    cell.plot_φ(np.arange(0, 2001, 200))
    cell.plot_jint_i0int_ηint(np.arange(0, 2001, 200))
    cell.plot_jDL(np.arange(0, 2001, 200))
    cell.plot_csr(np.arange(0, 2001, 200), 1)
    cell.plot_jLP_ηLP(np.arange(4000, 4301, 100))
    cell.plot_LP()
    cell.plot_OCV_OCP()
    cell.plot_dUOCPdθs()
    cell.plot_i(np.arange(0, 2001, 200))
    
    cell.plot_Z(1)
    cell.plot_Nyquist()
    cell.plot_REcssurf_IMcssurf()
    cell.plot_REce_IMce()
    cell.plot_REφs_IMφs()
    cell.plot_REφe_IMφe()
    cell.plot_REjint_IMjint()
    cell.plot_REjDL_IMjDL()
    cell.plot_REi0int_IMi0int()
    cell.plot_REηint_IMηint()
    '''

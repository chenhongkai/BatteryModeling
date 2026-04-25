#%%
from math import cos, sin
from decimal import Decimal  # 处理大数

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from numpy import ndarray,\
    array, empty, zeros, full, hstack, \
    sinh, cosh, outer, ix_
from numpy.linalg import solve

from P2Dmodel import LPP2D, JTFP2D, DFNP2D


class LPJTFP2D(LPP2D, JTFP2D):
    """锂离子电池集总参数时频联合准二维模型（Lumped-Parameter Joint Time-Frequency Pseudo-two-Dimension model）"""
    def __init__(self,
            SOC0: float = 0.5,   # 初始荷电状态 [–]
            T0: float = 298.15,  # 初始温度 [K]
            **kwargs):
        f_ = kwargs.pop('f_', None)  # 从 kwargs 获取 f_
        init_kwargs = {}
        if f_ is not None:
            init_kwargs['f_'] = f_
        JTFP2D.__init__(self, fullyInitialize=False, **init_kwargs)
        LPP2D.__init__(self, **kwargs)
        # 状态量
        (self.REθsnegsurf__, self.IMθsnegsurf__,  # 负极固相表面浓度实部、虚部
         self.REθspossurf__, self.IMθspossurf__,  # 正极固相表面浓度实部、虚部
         self.REθe__, self.IMθe__,              # 电解液锂离子浓度实部、虚部
         self.REJintneg__, self.IMJintneg__,    # 负极主反应局部体积电流密度实部、虚部
         self.REJintpos__, self.IMJintpos__,    # 正极主反应局部体积电流密度实部、虚部
         self.REJDLneg__, self.IMJDLneg__,      # 负极双电层局部体积电流密度实部、虚部
         self.REJDLpos__, self.IMJDLpos__,      # 正极双电层局部体积电流密度实部、虚部
         self.REI0intneg__, self.IMI0intneg__,  # 负极交换电流密度实部、虚部
         self.REI0intpos__, self.IMI0intpos__,  # 正极交换电流密度实部、虚部
         self.REJLP__, self.IMJLP__,              # 析锂反应局部体积电流密度实部、虚部
         ) = (None,)*20
        # 索引频域因变量
        (self.idxREθsnegsurf_, self.idxIMθsnegsurf_,
         self.idxREθspossurf_, self.idxIMθspossurf_,
         self.idxREθe_, self.idxIMθe_,
         self.idxREJintneg_, self.idxIMJintneg_,
         self.idxREJintpos_, self.idxIMJintpos_,
         self.idxREJDLneg_, self.idxIMJDLneg_,
         self.idxREJDLpos_, self.idxIMJDLpos_,
         self.idxREI0intneg_, self.idxIMI0intneg_,
         self.idxREI0intpos_, self.idxIMI0intpos_,
         self.idxREJLP_, self.idxIMJLP_,
        ) = (None,)*20
        # 初始化
        self.initialize(
            SOC0=SOC0,  # 初始荷电状态 [–]
            T0=T0,)     # 初始温度 [K]

    def initialize(self,
            SOC0: int | float = 0.,    # 全电池荷电状态
            T0: int | float = 298.15,  # 温度 [K]
            ):
        """初始化"""
        if self.verbose and type(self) is LPJTFP2D:
            print(f'集总参数时频联合P2D模型初始化...')
        LPP2D.initialize(self, SOC0=SOC0, T0=T0)
        # 恒定量
        self.ΔIAC = self.Qcell*0.05  # 交流扰动电流振幅 [A]
        self.initialize_frequency_domain_linear_matrix()
        self.frequency_dependent_cache = self.solve_frequency_dependent_variables()
        # 需记录的阻抗数据名称
        self.EISdatanames_ = EISdatanames_ = ['tEIS', 'Z_']  # 频率 [Hz]、阻抗时刻 [s]、序列 、复阻抗 [Ω]
        if self.complete:
            self.EISdatanames_.extend([
                'Zneg_', 'Zsep_', 'Zpos_',         # 负极、隔膜、正极复阻抗 [Ω]
                'REθsnegsurf__', 'IMθsnegsurf__',  # 负极固相表面无量纲锂离子浓度实部、虚部 [–]
                'REθspossurf__', 'IMθspossurf__',  # 正极固相表面无量纲锂离子浓度实部、虚部 [–]
                'REθe__', 'IMθe__',                # 电解液无量纲锂离子浓度实部、虚部 [–]
                'REφsneg__', 'IMφsneg__',          # 负极固相电势实部、虚部 [V]
                'REφspos__', 'IMφspos__',          # 正极固相电势实部、虚部 [V]
                'REφe__', 'IMφe__',                # 电解液电势实部、虚部 [V]
                'REJintneg__', 'IMJintneg__',      # 负极主反应集总局部体积电流密度实部、虚部 [A]
                'REJintpos__', 'IMJintpos__',      # 正极主反应集总局部体积电流密度实部、虚部 [A]
                'REJDLneg__', 'IMJDLneg__',        # 负极双电层效应集总局部体积电流密度实部、虚部 [A]
                'REJDLpos__', 'IMJDLpos__',        # 正极双电层效应集总局部体积电流密度实部、虚部 [A]
                'REηintneg__', 'IMηintneg__',      # 负极过电位实部、虚部 [V]
                'REηintpos__', 'IMηintpos__',      # 正极过电位实部、虚部 [V]
                'REI0intneg__', 'IMI0intneg__',    # 负极集总交换电流密度实部、虚部 [A]
                'REI0intpos__', 'IMI0intpos__', ]) # 正极集总交换电流密度实部、虚部 [A]
        self.data.update({EISdataname: [] for EISdataname in EISdatanames_})  # 字典：存储呈时间序列的阻抗数据
        if self.verbose and type(self) is LPJTFP2D:
            print(self)
            print(f'集总参数时频联合P2D模型初始化完成!')
        return self

    def solve_frequency_dependent_variables(self):
        """求解频率相关变量"""
        Qneg, Qpos, Dsneg, Dspos = self.Qneg, self.Qpos, self.Dsneg, self.Dspos
        ω_ = self.ω_
        solve_Kθssurf__ = LPJTFP2D.solve_Kθssurf__
        frequency_dependent_variables = {
            'ωqeΔx__': outer(ω_, self.qe_*self.Δx_),  # (len(f_), Ne) 各频率各控制体的ω*qe*Δx值
            'ωCDLneg_': (ωCDLneg_ := ω_*self.CDLneg),
            'ωCDLpos_': (ωCDLpos_ := ω_*self.CDLpos),
            'ωCDLRSEIneg_': ωCDLneg_*self.RSEIneg,
            'ωCDLRSEIpos_': ωCDLpos_*self.RSEIpos,
            'minusKθsnegsurf___': -array([solve_Kθssurf__(ω, Qneg, Dsneg) for ω in ω_]),   # 负极各频率Kθssurf__矩阵
            'minusKθspossurf___': -array([solve_Kθssurf__(ω, Qpos, Dspos) for ω in ω_]),}  # 正极各频率Kθssurf__矩阵
        return frequency_dependent_variables

    def initialize_frequency_domain_linear_matrix(self):
        """初始化频域因变量矩阵"""
        N = JTFP2D.generate_indices_of_frequency_domain_dependent_variables(self)
        self.Kf__ = Kf__ = zeros([N, N])  # 频域因变量线性矩阵
        self.bKf_ = zeros(N)              # Kf__ @ X_ = bKf_
        if self.verbose:
            print(f'初始化频域因变量线性矩阵 Kf__.shape = {Kf__.shape}')
        Nneg, Npos = self.Nneg, self.Npos      # 读取：网格数
        Δxneg, Δxpos = self.Δxneg, self.Δxpos  # 读取：网格尺寸 [–]
        # 覆盖集总频域因变量索引
        self.idxREθsnegsurf_ = self.idxREcsnegsurf_  # 索引：负极固相表面浓度实部
        self.idxIMθsnegsurf_ = self.idxIMcsnegsurf_  # 索引：负极固相表面浓度虚部
        self.idxREθspossurf_ = self.idxREcspossurf_  # 索引：正极固相表面浓度实部
        self.idxIMθspossurf_ = self.idxIMcspossurf_  # 索引：正极固相表面浓度虚部
        idxREθe_ = self.idxREθe_ = self.idxREce_            # 索引：电解液锂离子浓度实部
        idxIMθe_ = self.idxIMθe_ = self.idxIMce_            # 索引：电解液锂离子浓度虚部
        idxREJintneg_ = self.idxREJintneg_ = self.idxREjintneg_  # 索引：负极局部体积电流实部
        idxIMJintneg_ = self.idxIMJintneg_ = self.idxIMjintneg_  # 索引：负极局部体积电流虚部
        idxREJintpos_ = self.idxREJintpos_ = self.idxREjintpos_  # 索引：正极局部体积电流实部
        idxIMJintpos_ = self.idxIMJintpos_ = self.idxIMjintpos_  # 索引：正极局部体积电流虚部
        idxREJDLneg_ = self.idxREJDLneg_ = self.idxREjDLneg_  # 索引：负极双电层局部体积电流实部
        idxIMJDLneg_ = self.idxIMJDLneg_ = self.idxIMjDLneg_  # 索引：负极双电层局部体积电流虚部
        idxREJDLpos_ = self.idxREJDLpos_ = self.idxREjDLpos_  # 索引：正极双电层局部体积电流实部
        idxIMJDLpos_ = self.idxIMJDLpos_ = self.idxIMjDLpos_  # 索引：正极双电层局部体积电流虚部
        self.idxREI0intneg_ = self.idxREi0intneg_  # 索引：负极交换电流实部
        self.idxIMI0intneg_ = self.idxIMi0intneg_  # 索引：负极交换电流虚部
        self.idxREI0intpos_ = self.idxREi0intpos_  # 索引：正极交换电流实部
        self.idxIMI0intpos_ = self.idxIMi0intpos_  # 索引：正极交换电流虚部
        self.idxREJLP_ = self.idxREjLP_  # 索引：析锂反应电流密度实部
        self.idxIMJLP_ = self.idxIMjLP_  # 索引：正极交换电流密度虚部

        ## 对频域因变量线性矩阵Kf__赋参数相关值 ##
        self.update_Kf__idxREθe_idxREθe_and_idxIMθe_idxIMθe_(self.Deκ_)
        self.update_Kf__idxREφsneg_idxREJneg_and_idxIMφsneg_idxIMJneg_(σneg := self.σneg)
        self.update_Kf__idxREφspos_idxREJpos_and_idxIMφspos_idxIMJpos_(σpos := self.σpos)
        self.update_bKf_idxREφsneg_0_and_idxREφspos_end(σneg, σpos)
        self.update_Kf__idxREφe_idxREφe_and_idxIMφe_idxIMφe_(κ_:=self.κ_, κ_)
        self.update_Kf__idxREηintneg_idxREJneg_and_idxIMηintneg_idxIMJneg_(self.RSEIneg)
        self.update_Kf__idxREηintpos_idxREJpos_and_idxIMηintpos_idxIMJpos_(self.RSEIpos)
        if lithiumPlating := self.lithiumPlating:
            self.update_Kf__idxREηLP_idxREJneg_and_idxIMηLP_idxIMJneg_(self.RSEIneg)

        ## 对频域因变量线性矩阵Kf__赋固定值 ##
        JTFP2D.assign_Kf__with_constants(self)
        # 集总参数模型需额外赋固定值（原始模型的此处为参数tplus相关的值）
        # 电解液浓度实部REθe行、虚部IMθe行
        idxREθeneg_ = idxREθe_[:Nneg]
        idxIMθeneg_ = idxIMθe_[:Nneg]
        idxREθepos_ = idxREθe_[-Npos:]
        idxIMθepos_ = idxIMθe_[-Npos:]
        Kf__[idxREθeneg_, idxREJintneg_] = \
        Kf__[idxREθeneg_, idxREJDLneg_] = \
        Kf__[idxIMθeneg_, idxIMJintneg_] = \
        Kf__[idxIMθeneg_, idxIMJDLneg_] = -Δxneg  # REJneg、IMJneg列
        Kf__[idxREθepos_, idxREJintpos_] = \
        Kf__[idxREθepos_, idxREJDLpos_] = \
        Kf__[idxIMθepos_, idxIMJintpos_] = \
        Kf__[idxIMθepos_, idxIMJDLpos_] = -Δxpos  # REJpos、IMJpos列
        if lithiumPlating:
            Kf__[idxREθeneg_, self.idxREJLP_] = \
            Kf__[idxIMθeneg_, self.idxIMJLP_] = -Δxneg  # REθe行REJLP列、IMθe行IMJLP列

    def update_Kf__idxREθe_idxREθe_and_idxIMθe_idxIMθe_(self, Deκ_):
        # 更新Kf__矩阵REθe行REθe列、IMθe行IMθe列
        JTFP2D.update_Kf__idxREce_idxREce_and_idxIMce_idxIMce_(self, Deκ_, Deκ_)

    def update_Kf__idxREφsneg_idxREJneg_and_idxIMφsneg_idxIMJneg_(self, σneg):
        # 更新Kf__矩阵REφsneg行REJneg列、IMφsneg行IMJneg列
        JTFP2D.update_Kf__idxREφsneg_idxREjneg_and_idxIMφsneg_idxIMjneg_(self, σneg)

    def update_Kf__idxREφspos_idxREJpos_and_idxIMφspos_idxIMJpos_(self, σpos):
        # 更新Kf__矩阵REφspos行REJpos列、IMφspos行IMJpos列
        JTFP2D.update_Kf__idxREφspos_idxREjpos_and_idxIMφspos_idxIMjpos_(self, σpos)

    def update_bKf_idxREφsneg_0_and_idxREφspos_end(self, σneg, σpos):
        bKf_ = self.bKf_
        ΔIAC = self.ΔIAC
        # 更新bKf_向量REφsneg首元
        bKf_[self.idxREφsneg_[0]] = -self.Δxneg*ΔIAC/σneg
        # 更新bKf_向量REφspos末元
        bKf_[self.idxREφspos_[-1]] = self.Δxpos*ΔIAC/σpos

    def update_Kf__idxREηintneg_idxREJneg_and_idxIMηintneg_idxIMJneg_(self, RSEIneg):
        # 更新Kf__矩阵REηintneg行REJneg列、IMηintneg行IMJneg列
        JTFP2D.update_Kf__idxREηintneg_idxREjneg_and_idxIMηintneg_idxIMjneg_(self, RSEIneg, 1)

    def update_Kf__idxREηintpos_idxREJpos_and_idxIMηintpos_idxIMJpos_(self, RSEIpos):
        # 更新Kf__矩阵REηintpos行REJpos列、IMηintpos行IMJpos列
        JTFP2D.update_Kf__idxREηintpos_idxREjpos_and_idxIMηintpos_idxIMjpos_(self, RSEIpos, 1)

    def update_Kf__idxREηLP_idxREJneg_and_idxIMηLP_idxIMJneg_(self, RSEIneg):
        # 更新Kf__矩阵REηLP行REJneg列、IMηLP行IMJneg列
        JTFP2D.update_Kf__idxREηLP_idxREjneg_and_idxIMηLP_idxIMjneg_(self, RSEIneg, 1)

    def EIS(self):
        """计算电化学阻抗谱"""
        tEIS = self.t     # 读取：当前时刻 [s]
        data = self.data  # 读取：数据字典
        if data['tEIS'] and data['tEIS'][-1]==tEIS:
            if self.verbose:
                print(f'已计算时刻{tEIS = } s 电化学阻抗谱')
            return self
        # 频域因变量索引
        idxREθsnegsurf_, idxIMθsnegsurf_ = self.idxREθsnegsurf_, self.idxIMθsnegsurf_
        idxREθspossurf_, idxIMθspossurf_ = self.idxREθspossurf_, self.idxIMθspossurf_
        idxREθe_, idxIMθe_ = self.idxREθe_, self.idxIMθe_
        idxREφsneg_, idxIMφsneg_ = self.idxREφsneg_, self.idxIMφsneg_
        idxREφspos_, idxIMφspos_ = self.idxREφspos_, self.idxIMφspos_
        idxREφe_, idxIMφe_ = self.idxREφe_, self.idxIMφe_
        idxREJintneg_, idxIMJintneg_ = self.idxREJintneg_, self.idxIMJintneg_
        idxREJintpos_, idxIMJintpos_ = self.idxREJintpos_, self.idxIMJintpos_
        idxREJDLneg_, idxIMJDLneg_ = self.idxREJDLneg_, self.idxIMJDLneg_
        idxREJDLpos_, idxIMJDLpos_ = self.idxREJDLpos_, self.idxIMJDLpos_
        idxREI0intneg_, idxIMI0intneg_ = self.idxREI0intneg_, self.idxIMI0intneg_
        idxREI0intpos_, idxIMI0intpos_ = self.idxREI0intpos_, self.idxIMI0intpos_
        idxREηintneg_, idxIMηintneg_ = self.idxREηintneg_, self.idxIMηintneg_
        idxREηintpos_, idxIMηintpos_ = self.idxREηintpos_, self.idxIMηintpos_
        REIMI0intnegUnknown = idxREI0intneg_.size > 0
        REIMI0intposUnknown = idxREI0intpos_.size > 0
        lithiumPlating = self.lithiumPlating

        solve_banded_matrix = DFNP2D.solve_banded_matrix
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos            # 读取：网格数
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_  # 读取：网格距离 [–]
        κ_ = self.κ_              # (Ne,) 各控制体电解液集总离子电导率 [S]
        Deκ_ = self.Deκ_          # (Ne,) 各控制体电解液集总扩散系数 [A]
        κDκT_ = (self.κD*self.T) * κ_  # (Ne,) 各控制体电解液集总扩散离子电导率 [A]

        if self.constants:
            cache = self.frequency_dependent_cache
        else:
            cache = self.solve_frequency_dependent_variables()
            # 更新Kf__矩阵的参数相关值
            σneg, σpos = self.σneg, self.σpos  # 读取：固相有效电导率 [S]
            self.update_Kf__idxREθe_idxREθe_and_idxIMθe_idxIMθe_(Deκ_)  # 更新：电解液浓度实部REθe行REθe列、虚部IMθe行IMθe列
            self.update_Kf__idxREφsneg_idxREJneg_and_idxIMφsneg_idxIMJneg_(σneg)
            self.update_Kf__idxREφspos_idxREJpos_and_idxIMφspos_idxIMJpos_(σpos)
            self.update_bKf_idxREφsneg_0_and_idxREφspos_end(σneg, σpos)
            self.update_Kf__idxREφe_idxREφe_and_idxIMφe_idxIMφe_(κ_, κ_)
            self.update_Kf__idxREηintneg_idxREJneg_and_idxIMηintneg_idxIMJneg_(self.RSEIneg)
            self.update_Kf__idxREηintpos_idxREJpos_and_idxIMηintpos_idxIMJpos_(self.RSEIpos)
            if lithiumPlating:
                self.update_Kf__idxREηLP_idxREJneg_and_idxIMηLP_idxIMJneg_(self.RSEIneg)

        θe_ = self.θe_                # 读取：(Ne,) 电解液锂离子浓度 [–]
        θeInterfaces_ = self.θeInterfaces_  # 读取：(Ne+1,) 各控制体界面的电解液锂离子浓度 [–]
        θeWest_ = θeInterfaces_[:-1]  # (Ne,) 各控制体左界面的电解液锂离子浓度 [–]
        θeEast_ = θeInterfaces_[1:]   # (Ne,) 各控制体右界面的电解液锂离子浓度 [–]
        gradθeWest_ = hstack([0, (θe_[1:] - θe_[:-1])/ΔxWest_[1:]])   # (Ne,) 各控制体左界面的锂离子浓度梯度 [–]
        gradθeEast_ = hstack([(θe_[1:] - θe_[:-1])/ΔxEast_[:-1], 0])  # (Ne,) 各控制体右界面的锂离子浓度梯度 [–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、修正隔膜-正极界面
            gradθeEast_[nW] = (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW])
            gradθeWest_[nE] = (θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE])

        Kf__ = self.Kf__  # 因变量矩阵
        bKf_ = self.bKf_  # b向量
        ## 对Kf__矩阵赋时变值 ##

        # 电解液电势实部REφe行REθe列
        a = κDκT_[0]/θeEast_[0]/ΔxEast_[0]
        aa = (κDκT_[0]*gradθeEast_[0]/θeEast_[0]**2)/2
        Kf__[idxREφe_[0], idxREθe_[:2]] = [a + aa, -a + aa]  # 首行
        a = κDκT_[-1]/θeWest_[-1]/ΔxWest_[-1]
        aa = (-κDκT_[-1]*gradθeWest_[-1]/θeWest_[-1]**2)/2
        Kf__[idxREφe_[-1], idxREθe_[-2:]] = [-a + aa, a + aa]  # 末行
        a_ = -κDκT_[1:-1]/θeWest_[1:-1]/ΔxWest_[1:-1]
        c_ = -κDκT_[1:-1]/θeEast_[1:-1]/ΔxEast_[1:-1]
        aa_ = (- κDκT_[1:-1]*gradθeWest_[1:-1]/θeWest_[1:-1]**2)/2
        cc_ = (+ κDκT_[1:-1]*gradθeEast_[1:-1]/θeEast_[1:-1]**2)/2
        Kf__[idxREφe_[1:-1], idxREθe_[:-2]] = a_ + aa_
        Kf__[idxREφe_[1:-1], idxREθe_[2:]]  = c_ + cc_
        Kf__[idxREφe_[1:-1], idxREθe_[1:-1]] = -(a_ + c_) + aa_ + cc_
        # 修正负极-隔膜界面、隔膜-正极界面
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            pDW = Deκ_[nW]*Δx_[nE]/(Deκ_[nE]*Δx_[nW] + Deκ_[nW]*Δx_[nE])
            pDE = 1 - pDW  # 即：Deκ_[nE]*Δx_[nW] / (Deκ_[nE]*Deκ_[nW] + Deκ_[nW]*Δx_[nE])
            # 界面左侧控制体
            a = -κDκT_[nW]/θeWest_[nW]/ΔxWest_[nW]
            aa = (-κDκT_[nW]*gradθeWest_[nW]/θeWest_[nW]**2)/2
            c = -2*κDκT_[nW]*Deκ_[nE]/θeEast_[nW]/(Deκ_[nE]*Δx_[nW] + Deκ_[nW]*Δx_[nE])
            cc = + κDκT_[nW]*gradθeEast_[nW]/θeEast_[nW]**2
            Kf__[idxREφe_[nW], idxREθe_[nW-1:nW+2]] = [a + aa,
                                                       -(a + c) + aa + cc*pDW,
                                                       c + cc*pDE]
            # 界面右侧控制体
            a = -2*κDκT_[nE]*Deκ_[nW]/θeWest_[nE]/(Deκ_[nE]*Δx_[nW] + Deκ_[nW]*Δx_[nE])
            aa = -κDκT_[nE]*gradθeWest_[nE]/θeWest_[nE]**2
            c = -κDκT_[nE]/θeEast_[nE]/ΔxEast_[nE]
            cc = (κDκT_[nE]*gradθeEast_[nE]/θeEast_[nE]**2)/2
            Kf__[idxREφe_[nE], idxREθe_[nE-1:nE+2]] = [a + aa*pDW,
                                                       -(a + c) + aa*pDE + cc,
                                                       c + cc]
        # 电解液电势虚部IMφe行
        Kf__[idxIMφe_, idxIMθe_] = Kf__[idxREφe_, idxREθe_]                    # IMθe列主对角线
        Kf__[idxIMφe_[1:], idxIMθe_[:-1]] = Kf__[idxREφe_[1:], idxREθe_[:-1]]  # IMθe列下对角线
        Kf__[idxIMφe_[:-1], idxIMθe_[1:]] = Kf__[idxREφe_[:-1], idxREθe_[1:]]  # IMθe列上对角线

        # 负极局部体积电流实部REJintneg行、虚部IMJintneg行
        if REIMI0intnegUnknown:
            Kf__[idxREJintneg_, idxREI0intneg_] = \
            Kf__[idxIMJintneg_, idxIMI0intneg_] = -self.dJintdI0intneg_ # REIMI0intneg列
        Kf__[idxREJintneg_, idxREηintneg_] = \
        Kf__[idxIMJintneg_, idxIMηintneg_] = -self.dJintdηintneg_  # REIMηintneg列
        # 正极局部体积电流实部REJintpos行、虚部IMJintpos行
        if REIMI0intposUnknown:
            Kf__[idxREJintpos_, idxREI0intpos_] = \
            Kf__[idxIMJintpos_, idxIMI0intpos_] = -self.dJintdI0intpos_  # REIMI0intpos列
        Kf__[idxREJintpos_, idxREηintpos_] = \
        Kf__[idxIMJintpos_, idxIMηintpos_] = -self.dJintdηintpos_        # REIMηintpos列
        
        if REIMI0intnegUnknown:
            # 负极交换电流实部REI0intneg行、虚部IMI0intneg行
            Kf__[idxREI0intneg_, idxREθsnegsurf_] = \
            Kf__[idxIMI0intneg_, idxIMθsnegsurf_] = -self.dI0intdθsnegsurf_  # REIMθsnegsurf列
            Kf__[idxREI0intneg_, idxREθe_[:Nneg]] = \
            Kf__[idxIMI0intneg_, idxIMθe_[:Nneg]] = -self.dI0intdθeneg_  # REIMθe列
        if REIMI0intposUnknown:
            # 正极交换电流实部REI0intpos行、虚部IMI0intpos行
            Kf__[idxREI0intpos_, idxREθspossurf_] = \
            Kf__[idxIMI0intpos_, idxIMθspossurf_] = -self.dI0intdθspossurf_  # REIMθsnegsurf列
            Kf__[idxREI0intpos_, idxREθe_[-Npos:]] = \
            Kf__[idxIMI0intpos_, idxIMθe_[-Npos:]] = -self.dI0intdθepos_     # REIMθe列

        # 负极过电位实部REηintneg行REθsnegsurf列、虚部IMηintneg行IMθsnegsurf列
        Kf__[idxREηintneg_, idxREθsnegsurf_] = \
        Kf__[idxIMηintneg_, idxIMθsnegsurf_] = self.dUOCPdθsnegsurf_
        # 正极过电位实部REηintpos行REθspossurf列、虚部IMηintpos行IMθsnegsurf列
        Kf__[idxREηintpos_, idxREθspossurf_] = \
        Kf__[idxIMηintpos_, idxIMθspossurf_] = self.dUOCPdθspossurf_

        if lithiumPlating:
            # 析锂补充
            idxREJLP_, idxIMJLP_ = self.idxREJLP_, self.idxIMJLP_
            idxREηLP_, idxIMηLP_ = self.idxREηLP_, self.idxIMηLP_
            # 析锂局部体积电流实部REJLP行REθe负极列、虚部IMJLP行IMθe负极列
            Kf__[idxREJLP_, idxREθe_[:Nneg]] = \
            Kf__[idxIMJLP_, idxIMθe_[:Nneg]] = -self.dJLPdθe_
            # 析锂局部体积电流实部REJLP行REηLP列、虚部IMJLP行IMηLP列
            Kf__[idxREJLP_, idxREηLP_] = \
            Kf__[idxIMJLP_, idxIMηLP_] = -self.dJLPdηLP_

        Nf = self.f_.size
        X__ = empty((Nf, bKf_.shape[0]), dtype=bKf_.dtype)

        for nf, (ωqeΔx_,
             ωCDLneg, ωCDLpos,
             ωCDLRSEIneg, ωCDLRSEIpos,
             minusKθsnegsurf__, minusKθspossurf__) in enumerate(zip(
                cache['ωqeΔx__'],
                cache['ωCDLneg_'], cache['ωCDLpos_'],
                cache['ωCDLRSEIneg_'], cache['ωCDLRSEIpos_'],
                cache['minusKθsnegsurf___'], cache['minusKθspossurf___'])):
            ## 遍历所有频率f ##

            # 负极固相表面浓度实部REθsnegsurf行
            Kf__[idxREθsnegsurf_, idxREJintneg_] = minusKθsnegsurf__[0, 0]  # REJintneg列
            Kf__[idxREθsnegsurf_, idxIMJintneg_] = minusKθsnegsurf__[0, 1]  # IMJintneg列
            # 负极固相表面浓度虚部IMθsnegsurf行
            Kf__[idxIMθsnegsurf_, idxREJintneg_] = minusKθsnegsurf__[1, 0]  # REJintneg列
            Kf__[idxIMθsnegsurf_, idxIMJintneg_] = minusKθsnegsurf__[1, 1]  # IMJintneg列
            # 正极固相表面浓度实部REθspossurf行
            Kf__[idxREθspossurf_, idxREJintpos_] = minusKθspossurf__[0, 0]  # REJintpos列
            Kf__[idxREθspossurf_, idxIMJintpos_] = minusKθspossurf__[0, 1]  # IMJintpos列
            # 正极固相表面浓度虚部IMθspossurf行
            Kf__[idxIMθspossurf_, idxREJintpos_] = minusKθspossurf__[1, 0]  # REJintpos列
            Kf__[idxIMθspossurf_, idxIMJintpos_] = minusKθspossurf__[1, 1]  # IMJintpos列

            Kf__[idxREθe_, idxIMθe_] = -ωqeΔx_  # REθe行IMθe列
            Kf__[idxIMθe_, idxREθe_] = ωqeΔx_   # IMθe行REθe列

            # 负极双电层局部体积电流实部REJDLneg行
            Kf__[idxREJDLneg_, idxIMφsneg_] = ωCDLneg         # IMφsneg列
            Kf__[idxREJDLneg_, idxIMφe_[:Nneg]] = -ωCDLneg    # IMφe负极列
            Kf__[idxREJDLneg_, idxIMJintneg_] = \
            Kf__[idxREJDLneg_, idxIMJDLneg_] = -ωCDLRSEIneg   # IMJintneg列、IMJDLneg列
            # 负极双电层局部体积电流虚部IMJDLneg行
            Kf__[idxIMJDLneg_, idxREφsneg_] = -ωCDLneg       # REφsneg列
            Kf__[idxIMJDLneg_, idxREφe_[:Nneg]] = ωCDLneg    # REφe负极列
            Kf__[idxIMJDLneg_, idxREJintneg_] = \
            Kf__[idxIMJDLneg_, idxREJDLneg_] = ωCDLRSEIneg   # REJintneg列、REJDLneg列
            # 正极双电层局部体积电流实部REJDLpos行
            Kf__[idxREJDLpos_, idxIMφspos_] = ωCDLpos        # IMφspos列
            Kf__[idxREJDLpos_, idxIMφe_[-Npos:]] = -ωCDLpos  # IMφe正极列
            Kf__[idxREJDLpos_, idxIMJintpos_] = \
            Kf__[idxREJDLpos_, idxIMJDLpos_] = -ωCDLRSEIpos  # IMJintpos列、IMJDLpos列
            # 正极双电层局部体积电流虚部IMJDLpos行
            Kf__[idxIMJDLpos_, idxREφspos_] = -ωCDLpos       # REφspos列
            Kf__[idxIMJDLpos_, idxREφe_[-Npos:]] = ωCDLpos   # REφe正极列
            Kf__[idxIMJDLpos_, idxREJintpos_] = \
            Kf__[idxIMJDLpos_, idxREJDLpos_] = ωCDLRSEIpos   # REJintpos列、REJDLpos列

            if lithiumPlating:
                # 补充
                Kf__[idxREJDLneg_, idxIMJLP_] = -ωCDLRSEIneg  # REJDLneg行IMJLP列
                Kf__[idxIMJDLneg_, idxREJLP_] = ωCDLRSEIneg   # IMJDLneg行REJLP列

            if (self.bandwidthsKf_ is None) and any(data['I']):
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
        self.REφsneg__ = X__[:, idxREφsneg_]  # 负极固相电势实部
        self.IMφsneg__ = X__[:, idxIMφsneg_]  # 负极固相电势虚部
        self.REφspos__ = X__[:, idxREφspos_]  # 正极固相电势实部
        self.IMφspos__ = X__[:, idxIMφspos_]  # 正极固相电势虚部
        if self.complete:
            self.REθsnegsurf__ = X__[:, idxREθsnegsurf_]  # 负极固相表面浓度实部
            self.IMθsnegsurf__ = X__[:, idxIMθsnegsurf_]  # 负极固相表面浓度虚部
            self.REθspossurf__ = X__[:, idxREθspossurf_]  # 正极固相表面浓度实部
            self.IMθspossurf__ = X__[:, idxIMθspossurf_]  # 正极固相表面浓度虚部
            self.REθe__ = X__[:, idxREθe_]            # 电解液锂离子浓度实部
            self.IMθe__ = X__[:, idxIMθe_]            # 电解液锂离子浓度虚部
            self.REφe__ = X__[:, idxREφe_]            # 电解液电势实部
            self.IMφe__ = X__[:, idxIMφe_]            # 电解液电势虚部
            self.REJintneg__ = X__[:, idxREJintneg_]  # 负极局部体积电流实部
            self.IMJintneg__ = X__[:, idxIMJintneg_]  # 负极局部体积电流虚部
            self.REJintpos__ = X__[:, idxREJintpos_]  # 正极局部体积电流实部
            self.IMJintpos__ = X__[:, idxIMJintpos_]  # 正极局部体积电流虚部
            self.REJDLneg__ = X__[:, idxREJDLneg_]    # 负极双电层局部体积电流实部
            self.IMJDLneg__ = X__[:, idxIMJDLneg_]    # 负极双电层局部体积电流虚部
            self.REJDLpos__ = X__[:, idxREJDLpos_]    # 正极双电层局部体积电流实部
            self.IMJDLpos__ = X__[:, idxIMJDLpos_]    # 正极双电层局部体积电流虚部
            self.REI0intneg__ = X__[:, idxREI0intneg_] if REIMI0intnegUnknown else zeros((Nf, Nneg))  # 负极交换电流实部
            self.IMI0intneg__ = X__[:, idxIMI0intneg_] if REIMI0intnegUnknown else zeros((Nf, Nneg))  # 负极交换电流虚部
            self.REI0intpos__ = X__[:, idxREI0intpos_] if REIMI0intposUnknown else zeros((Nf, Npos))  # 正极交换电流实部
            self.IMI0intpos__ = X__[:, idxIMI0intpos_] if REIMI0intposUnknown else zeros((Nf, Npos))  # 正极交换电流虚部
            self.REηintneg__ = X__[:, idxREηintneg_]  # 负极过电位实部
            self.IMηintneg__ = X__[:, idxIMηintneg_]  # 负极过电位虚部
            self.REηintpos__ = X__[:, idxREηintpos_]  # 正极过电位实部
            self.IMηintpos__ = X__[:, idxIMηintpos_]  # 正极过电位虚部
            if lithiumPlating:
                self.REJLP__ = X__[:, idxREJLP_]  # 负极析锂局部体积电流密度实部
                self.IMJLP__ = X__[:, idxIMJLP_]  # 负极析锂局部体积电流密度虚部
                self.REηLP__ = X__[:, idxREηLP_]  # 负极析锂过电位实部
                self.IMηLP__ = X__[:, idxIMηLP_]  # 负极析锂过电位虚部
        if self.verbose:
            print(f'计算时刻t = {tEIS:.1f} s 电化学阻抗谱')
        self.record_EISdata()  # 记录阻抗数据
        return self

    @property
    def Z_(self):
        """全电池复阻抗 [Ω]"""
        ΔIAC = self.ΔIAC
        σneg, σpos = self.σneg, self.σpos
        a  = 0.5*ΔIAC
        REφsnegCollector_ = self.REφsneg__[:, 0] + a*self.Δxneg/σneg   # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = self.IMφsneg__[:, 0]                       # (Nf,) 负极集流体电势虚部 [V]
        REφsposCollector_ = self.REφspos__[:, -1] - a*self.Δxpos/σpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = self.IMφspos__[:, -1]                      # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗虚部 [Ω]
        return Zreal_ + 1j*Zimag_ + self.Zl_                    # (Nf,) 全电池复阻抗 [Ω]

    @property
    def Zneg_(self):
        """负极复阻抗 [Ω]"""
        Nneg = self.Nneg
        Δx_ = self.Δx_
        nW, nE = Nneg - 1, Nneg
        κ_ = self.κ_
        ΔIAC = self.ΔIAC
        REφe__, IMφe__ = self.REφe__, self.IMφe__
        a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
        den = a + b
        REφenegsep_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
        IMφenegsep_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
        REφsnegCollector_ = self.REφsneg__[:, 0] + 0.5*self.Δxneg*ΔIAC/self.σneg  # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = self.IMφsneg__[:, 0]                                  # (Nf,) 负极集流体电势虚部 [V]
        Zreal_ = (REφenegsep_ - REφsnegCollector_)/-ΔIAC  # 负极阻抗实部 [Ω]
        Zimag_ = (IMφenegsep_ - IMφsnegCollector_)/-ΔIAC  # 负极阻抗虚部 [Ω]
        return Zreal_ + 1j*Zimag_  # 负极复阻抗 [Ω]

    @property
    def Zpos_(self):
        """正极复阻抗 [Ω]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_ = self.Δx_
        nW, nE = Nneg + Nsep - 1, Nneg + Nsep
        κ_ = self.κ_
        ΔIAC = self.ΔIAC
        REφe__, IMφe__ = self.REφe__, self.IMφe__
        a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
        den = a + b
        REφeseppos_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
        IMφeseppos_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
        REφsposCollector_ = self.REφspos__[:, -1] - 0.5*self.Δxpos*ΔIAC/self.σpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = self.IMφspos__[:, -1]                                  # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφeseppos_)/-ΔIAC  # 正极阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφeseppos_)/-ΔIAC  # 正极阻抗虚部 [Ω]
        return Zreal_ + 1j*Zimag_  # 正极复阻抗 [Ω]

    @staticmethod
    def solve_Kθssurf__(
            ω: float,  # 角频率 [rad/s]
            Q: float,  # 电极容量 [Ah]
            Ds: float,  # 集总固相锂离子扩散系数 [1/s]
            ):
        """求Kθssurf__矩阵，Kθssurf__ @ [REJ_, IMJ_] = [REθssurf_, IMθssurf_]"""
        W2 = ω/Ds
        W = W2**.5
        root2 = 1.4142135623730951
        γ = 0.7071067811865476*W
        cosγ, sinγ = cos(γ), sin(γ)
        # 处理大数运算
        cosγ2, sinγ2 = Decimal(cosγ*cosγ), Decimal(sinγ*sinγ)
        cosγsinγ = Decimal(cosγ*sinγ)
        γ = Decimal(γ)
        W, W2 = Decimal(W), Decimal(W2)
        root2 = Decimal(root2)
        Q3Ds = Decimal(10800*Q*Ds) # [A]
        expγ, exp_γ = γ.exp(), (-γ).exp()
        coshγ = (expγ + exp_γ)/2
        sinhγ = (expγ - exp_γ)/2
        coshγ2 = coshγ**2
        coshγsinhγ = coshγ*sinhγ

        a = -root2*(  W*coshγsinhγ
                    + W*cosγsinγ
                    + root2*cosγ2
                    - root2*coshγ2)
        b = -root2*W*(coshγsinhγ - cosγsinγ)
        d = 2*Q3Ds*((W2 + 1)*coshγ2
                   - root2*W*coshγsinhγ
                   - W2*sinγ2
                   - root2*W*cosγsinγ
                   - cosγ2)
        a /= d
        b /= d
        a, b = float(a), float(b)
        Kθssurf__ = array([[a,  b],
                              [-b, a]])
        return Kθssurf__

    @staticmethod
    def solve_REθs_IMθs(
            r: float,   # 径向坐标 [–]
            ω: float,   # 角频率 [rad/s]
            Q: float,   # 电极容量 [Ah]
            Ds: float,  # 集总固相锂离子扩散系数 [1/s]
            REJint_: ndarray,
            IMJint_: ndarray,
            ):
        """固相浓度实部、虚部在r处的解析解"""
        W2 = ω/Ds
        W = W2**.5
        root2 = 1.4142135623730951
        γ = .7071067811865476*W
        γr = γ*r
        # 处理大数运算
        sinγr, cosγr = Decimal(sin(γr)), Decimal(cos(γr))
        sinγ, cosγ = Decimal(sin(γ)),    Decimal(cos(γ))
        W, W2 = Decimal(W), Decimal(W2)
        root2 = Decimal(root2)
        Q3Dsr = Decimal(10800*Q*Ds*r)
        γr, γ = Decimal(γr), Decimal(γ)
        expγr, exp_γr = γr.exp(), (-γr).exp()
        coshγr = (expγr + exp_γr)/2
        sinhγr = (expγr - exp_γr)/2
        expγ, exp_γ = γ.exp(), (-γ).exp()
        coshγ = (expγ + exp_γ)/2
        sinhγ = (expγ - exp_γ)/2

        a = -root2*(  W*coshγr*sinγr*coshγ*cosγ
                    - root2*coshγr*sinγr*coshγ*sinγ
                    + W*coshγr*sinγr*sinhγ*sinγ
                    + W*sinhγr*cosγr*coshγ*cosγ
                    - W*sinhγr*cosγr*sinhγ*sinγ
                    - root2*sinhγr*cosγr*sinhγ*cosγ)

        b = -root2*(- W*coshγr*sinγr*coshγ*cosγ
                    + W*coshγr*sinγr*sinhγ*sinγ
                    + root2*coshγr*sinγr*sinhγ*cosγ
                    + W*sinhγr*cosγr*coshγ*cosγ
                    - root2*sinhγr*cosγr*coshγ*sinγ
                    + W*sinhγr*cosγr*sinhγ*sinγ)

        d = 2*Q3Dsr*((W2 + 1)*coshγ**2
                    - root2*W*coshγ*sinhγ
                    - W2*sinγ**2
                    - root2*W*cosγ*sinγ
                    - cosγ**2)

        a /= d
        b /= d
        a, b = float(a), float(b)
        Kθs__ = array([[a,  b],
                          [-b, a]])
        REθs_, IMθs_ = Kθs__ @ [REJint_, IMJint_]
        return REθs_, IMθs_

    @property
    def dJintdI0intneg_(self):
        """负极主反应局部体积电流密度Jintneg对交换电流密度I0intneg的偏导数 [A/A]"""
        return LPP2D.solve_dJintdI0int_(self.T, self.ηintneg_)

    @property
    def dJintdI0intpos_(self):
        """正极主反应局部体积电流密度Jintpos对交换电流密度I0pos的偏导数 [A/A]"""
        return LPP2D.solve_dJintdI0int_(self.T, self.ηintpos_)

    @property
    def dJintdηintneg_(self):
        """负极主反应局部体积电流密度Jintneg对过电位ηintneg的偏导数 [A/V]"""
        return LPP2D.solve_dJintdηint_(self.T, self.I0intneg_, self.ηintneg_)

    @property
    def dJintdηintpos_(self):
        """正极主反应局部体积电流密度Jintpos对过电位ηintpos的偏导数 [A/V]"""
        return LPP2D.solve_dJintdηint_(self.T, self.I0intpos_, self.ηintpos_)

    @property
    def dI0intdθsnegsurf_(self):
        """负极主反应交换电流密度I0intneg对电极表面浓度的偏导数 [A/-]"""
        return 0  if self._I0intneg\
            else LPP2D.solve_dI0intdθssurf_(self.kneg, self.θsnegsurf_, self.θeneg_, self.I0intneg_)

    @property
    def dI0intdθspossurf_(self):
        """正极主反应交换电流密度I0intpos对电极表面浓度的偏导数 [A/-]"""
        return 0 if self._I0intpos\
            else LPP2D.solve_dI0intdθssurf_(self.kpos, self.θspossurf_, self.θepos_, self.I0intpos_)

    @property
    def dI0intdθeneg_(self):
        """负极主反应交换电流密度I0int对电解液浓度θe的偏导数 [A/-]"""
        return 0 if self._I0intneg \
            else self.solve_dI0intdθe_(self.θeneg_, self.I0intneg_)

    @property
    def dI0intdθepos_(self):
        """正极主反应交换电流密度I0int对电解液浓度θe的偏导数 [A/-]"""
        return 0 if self._I0intpos \
            else self.solve_dI0intdθe_(self.θepos_, self.I0intpos_)

    @property
    def dJLPdθe_(self):
        """析锂反应局部体积电流密度JLP对电解液浓度θe的偏导数 [A/-]"""
        return 0 if self._I0LP \
            else LPP2D.solve_dJLPdθe_(self.T, self.θeneg_, self.I0LP_, self.ηLPneg_)

    @property
    def dJLPdηLP_(self):
        """析锂反应局部体积电流密度JLP对析锂过电位ηLP的偏导数 [A/V]"""
        return LPP2D.solve_dJLPdηLP_(self.T, self.I0LP_, self.ηLPneg_)

    @property
    def dUOCPdθsnegsurf_(self):
        """负极电位对负极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUOCPdθsneg_(self.θsnegsurf_)

    @property
    def dUOCPdθspossurf_(self):
        """正极电位对正极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUOCPdθspos_(self.θspossurf_)

    def plot_REθssurfIMEθssurf(self, *arg, **kwargs):
        JTFP2D.plot_REcssurfIMcssurf(self, *arg, **kwargs)

    def plot_REθeIMθe(self, *arg, **kwargs):
        JTFP2D.plot_REceIMce(self, *arg, **kwargs)

    def plot_REJintIMJint(self, *arg, **kwargs):
        JTFP2D.plot_REjintIMjint(self, *arg, **kwargs)

    def plot_REJDLIMJDL(self, *arg, **kwargs):
        JTFP2D.plot_REjDLIMjDL(self, *arg, **kwargs)

    def plot_REI0intIMI0int(self, *arg, **kwargs):
        JTFP2D.plot_REi0intIMi0int(self, *arg, **kwargs)

    def check_EIS(self):
        """检验频域控制方程"""
        if not self.complete:
            print('complete==True的前提下才可检验频域控制方程')
            return
        if self.tEIS!=self.t:
            print('应在完成最新EIS计算后立刻检查结果')
            return
        print('='*100)
        print(f'检验频域控制方程：')
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos
        REθsnegsurf__ = self.REθsnegsurf__  # 负极固相表面浓度实部
        IMθsnegsurf__ = self.IMθsnegsurf__  # 负极固相表面浓度虚部
        REθspossurf__ = self.REθspossurf__  # 正极固相表面浓度实部
        IMθspossurf__ = self.IMθspossurf__  # 正极固相表面浓度虚部
        REθe__ = self.REθe__            # 电解液锂离子浓度实部
        IMθe__ = self.IMθe__            # 电解液锂离子浓度虚部
        REφsneg__ = self.REφsneg__      # 负极固相电势实部
        IMφsneg__ = self.IMφsneg__       # 负极固相电势虚部
        REφspos__ = self.REφspos__      # 正极固相电势实部
        IMφspos__ = self.IMφspos__      # 正极固相电势虚部
        REφe__ = self.REφe__            # 电解液电势实部
        IMφe__ = self.IMφe__            # 电解液电势虚部
        REJintneg__ = self.REJintneg__  # 负极局部体积电流实部
        IMJintneg__ = self.IMJintneg__  # 负极局部体积电流虚部
        REJintpos__ = self.REJintpos__  # 正极局部体积电流实部
        IMJintpos__ = self.IMJintpos__  # 正极局部体积电流虚部
        REJDLneg__ = self.REJDLneg__    # 负极双电层局部体积电流实部
        IMJDLneg__ = self.IMJDLneg__    # 负极双电层局部体积电流虚部
        REJDLpos__ = self.REJDLpos__    # 正极双电层局部体积电流实部
        IMJDLpos__ = self.IMJDLpos__    # 正极双电层局部体积电流虚部
        REI0intneg__ = self.REI0intneg__  # 负极交换电流实部
        IMI0intneg__ = self.IMI0intneg__  # 负极交换电流虚部
        REI0intpos__ = self.REI0intpos__  # 正极交换电流实部
        IMI0intpos__ = self.IMI0intpos__  # 正极交换电流虚部
        REηintneg__ = self.REηintneg__  # 负极过电位实部
        IMηintneg__ = self.IMηintneg__  # 负极过电位虚部
        REηintpos__ = self.REηintpos__  # 正极过电位实部
        IMηintpos__ = self.IMηintpos__  # 正极过电位虚部
        if self.lithiumPlating:
            REJLP__ = self.REJLP__  # 负极析锂局部体积电流密度实部
            IMJLP__ = self.IMJLP__  # 负极析锂局部体积电流密度虚部
            REηLP__ = self.REηLP__  # 负极析锂过电位实部
            IMηLP__ = self.IMηLP__  # 负极析锂过电位虚部
        else:
            REJLP__ = 0
            IMJLP__ = 0
            REηLP__ = self.REφe__[:, :Nneg] - self.REφsneg__ - self.RSEIneg*(self.REJintneg__ + self.REJDLneg__)
            IMηLP__ = self.IMφe__[:, :Nneg] - self.IMφsneg__ - self.RSEIneg*(self.IMJintneg__ + self.IMJDLneg__)

        Nf = self.f_.size
        ω_ = self.ω_
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        x_ = self.x_
        ΔIAC = self.ΔIAC
        σneg, σpos = self.σneg, self.σpos
        RSEIneg, RSEIpos = self.RSEIneg, self.RSEIpos
        qe_ = self.qe_
        κ_ = self.κ_
        κDκT_ = (self.κD*self.T)*κ_
        Deκ_ = self.Deκ_
        F2RT = LPP2D.F/2/LPP2D.R/self.T
        I0intneg_, I0intpos_ = self.I0intneg_, self.I0intpos_
        ηintneg_, ηintpos_ = self.ηintneg_, self.ηintpos_
        θe_, θeInterfaces_ = self.θe_, self.θeInterfaces_
        θeWest_, θeEast_ = θeInterfaces_[:-1], θeInterfaces_[1:]
        gradθeWest_ = hstack([0, (θe_[1:] - θe_[:-1])/ΔxWest_[1:]])  # (Ne,) 各控制体左界面的锂离子浓度梯度 [–]
        gradθeEast_ = hstack([(θe_[1:] - θe_[:-1])/ΔxEast_[:-1], 0])  # (Ne,) 各控制体右界面的锂离子浓度梯度 [–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、修正隔膜-正极界面
            gradθeEast_[nW] = (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW])
            gradθeWest_[nE] = (θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE])

        # 各控制体界面的电解液浓度实部 [–]
        REθeInterfaces__ = hstack([REθe__[:, [0]], (REθe__[:, :-1] + REθe__[:, 1:])/2, REθe__[:, [-1]]])  # (Nf, Ne+1)
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 利用边界条件修正负极-隔膜界面、隔膜-正极界面锂离子浓度
            REθeInterfaces__[:, nE] = (Deκ_[nW]*Δx_[nE]*REθe__[:, nW] + Deκ_[nE]*Δx_[nW]*REθe__[:, nE])/(Deκ_[nW]*Δx_[nE] + Deκ_[nE]*Δx_[nW])
        REθeWest__ = REθeInterfaces__[:, :-1]  # 各控制体左界面的电解液锂离子浓度 [–]
        REθeEast__ = REθeInterfaces__[:, 1:]   # 各控制体右界面的电解液锂离子浓度 [–]
        gradREθeWest__ = hstack([zeros([Nf, 1]), (REθe__[:, 1:] - REθe__[:, :-1])/ΔxWest_[1:]])   # 各控制体左界面的锂离子浓度梯度实部 [–/–]
        gradREθeEast__ = hstack([(REθe__[:, 1:] - REθe__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # 各控制体右界面的锂离子浓度梯度实部 [–/–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradREθeEast__[:, nW] = (REθeEast__[:, nW] - REθe__[:, nW])/(0.5*Δx_[nW])
            gradREθeWest__[:, nE] = (REθe__[:, nE] - REθeWest__[:, nE])/(0.5*Δx_[nE])
        # 各控制体界面的电解液浓度虚部 [–]
        IMθeInterfaces__ = hstack([IMθe__[:, [0]], (IMθe__[:, :-1] + IMθe__[:, 1:])/2, IMθe__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 利用边界条件修正负极-隔膜界面、隔膜-正极界面锂离子浓度
            IMθeInterfaces__[:, nE] = (Deκ_[nW]*IMθe__[:, nW]*Δx_[nE] + Deκ_[nE]*IMθe__[:, nE]*Δx_[nW])/(Deκ_[nW]*Δx_[nE] + Deκ_[nE]*Δx_[nW])
        IMθeWest__ = IMθeInterfaces__[:, :-1]  # 各控制体左界面的电解液锂离子浓度虚部  [–]
        IMθeEast__ = IMθeInterfaces__[:, 1:]   # 各控制体右界面的电解液锂离子浓度虚部  [–]
        gradIMθeWest__ = hstack([zeros([Nf, 1]), (IMθe__[:, 1:] - IMθe__[:, :-1])/ΔxWest_[1:]])   # 各控制体左界面的锂离子浓度梯度虚部 [–/–]
        gradIMθeEast__ = hstack([(IMθe__[:, 1:] - IMθe__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # 各控制体右界面的锂离子浓度梯度虚部 [–/–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradIMθeEast__[:, nW] = (IMθeEast__[:, nW] - IMθe__[:, nW])/(0.5*Δx_[nW])
            gradIMθeWest__[:, nE] = (IMθe__[:, nE] - IMθeWest__[:, nE])/(0.5*Δx_[nE])

        # 各控制体界面的电解液电势实部 [V]
        REφeInterfaces__ = hstack([REφe__[:, [0]], (REφe__[:, :-1] + REφe__[:, 1:])/2, REφe__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            REφeInterfaces__[:, nE] = ( κ_[nW]*REφe__[:, nW]*Δx_[nE] + κ_[nE]*REφe__[:, nE]*Δx_[nW])/(κ_[nE]*Δx_[nW] + κ_[nW]*Δx_[nE])
        REφeWest__ = REφeInterfaces__[:, :-1]  # 各控制体左界面的电解液电势实部 [V]
        REφeEast__ = REφeInterfaces__[:, 1:]   # 各控制体右界面的电解液电势实部 [V]
        # 各控制体界面的电解液电势虚部 [V]
        IMφeInterfaces__ = hstack([IMφe__[:, [0]], (IMφe__[:, :-1] + IMφe__[:, 1:])/2, IMφe__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            IMφeInterfaces__[:, nE] = (κ_[nW]*IMφe__[:, nW]*Δx_[nE] + κ_[nE]*IMφe__[:, nE]*Δx_[nW])/(κ_[nE]*Δx_[nW] + κ_[nW]*Δx_[nE])
        IMφeWest__ = IMφeInterfaces__[:, :-1]  # 各控制体左界面的电解液电势虚部 [V]
        IMφeEast__ = IMφeInterfaces__[:, 1:]   # 各控制体右界面的电解液电势虚部 [V]
        # 各控制体界面的电解液电势实部梯度 [V/–]
        gradREφeInterfaces__ = hstack([zeros([Nf, 1]), (REφe__[:, 1:] - REφe__[:, :-1])/(x_[1:] - x_[:-1]), zeros([Nf, 1])])
        gradREφeWest__ = gradREφeInterfaces__[:, :-1].copy()
        gradREφeEast__ = gradREφeInterfaces__[:, 1:].copy()
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradREφeEast__[:, nW] = (REφeEast__[:, nW] - REφe__[:, nW])/(0.5*Δx_[nW])
            gradREφeWest__[:, nE] = (REφe__[:, nE] - REφeWest__[:, nE])/(0.5*Δx_[nE])
        # 各控制体界面的电解液电势虚部梯度 [V/–]
        gradIMφeInterfaces__ = hstack([zeros([Nf, 1]), (IMφe__[:, 1:] - IMφe__[:, :-1])/(x_[1:] - x_[:-1]), zeros([Nf, 1])])
        gradIMφeWest__ = gradIMφeInterfaces__[:, :-1].copy()
        gradIMφeEast__ = gradIMφeInterfaces__[:, 1:].copy()
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradIMφeEast__[:, nW] = (IMφeEast__[:, nW] - IMφe__[:, nW])/(0.5*Δx_[nW])
            gradIMφeWest__[:, nE] = (IMφe__[:, nE] - IMφeWest__[:, nE])/(0.5*Δx_[nE])

        c = self.solve_frequency_dependent_variables()
        maxError = max([
            abs(array([REθsnegsurf__, IMθsnegsurf__]).transpose(1, 0, 2) - -c['minusKθsnegsurf___'] @ array([REJintneg__, IMJintneg__]).transpose(1, 0, 2)).max(),
            abs(array([REθspossurf__, IMθspossurf__]).transpose(1, 0, 2) - -c['minusKθspossurf___'] @ array([REJintpos__, IMJintpos__]).transpose(1, 0, 2)).max(), ])
        print(f'固相表面浓度解析解方程最大误差{maxError: 8e} [–]')

        LHS__ = -outer(ω_, qe_) * IMθe__
        RHS__ = Deκ_*(gradREθeEast__ - gradREθeWest__)/Δx_ + hstack([REJintneg__ + REJDLneg__ + REJLP__, zeros([Nf, Nsep]), REJintpos__ + REJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液实部浓度方程最大误差{maxError: 8e} C')

        LHS__ = outer(ω_, qe_) * REθe__
        RHS__ = Deκ_*(gradIMθeEast__ - gradIMθeWest__)/Δx_ + hstack([IMJintneg__ + IMJDLneg__ + IMJLP__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液虚部浓度方程最大误差{maxError: 8e} C')

        gradREφsnegInterfaces_ = hstack([full([Nf, 1], -ΔIAC/σneg), (REφsneg__[:, 1:] - REφsneg__[:, :-1])/Δxneg, zeros([Nf, 1])])
        ΔREφsneg__ = (gradREφsnegInterfaces_[:, 1:] - gradREφsnegInterfaces_[:, :-1])/Δxneg
        gradIMφsnegInterfaces__ = hstack([zeros([Nf, 1]), (IMφsneg__[:, 1:] - IMφsneg__[:, :-1])/Δxneg, zeros([Nf, 1]),])
        ΔIMφsneg__ = (gradIMφsnegInterfaces__[:, 1:] - gradIMφsnegInterfaces__[:, :-1])/Δxneg
        RE_LHS__ = σneg*ΔREφsneg__
        RE_RHS__ = REJintneg__ + REJDLneg__ + REJLP__
        IM_LHS__ = σneg*ΔIMφsneg__
        IM_RHS__ = IMJintneg__ + IMJDLneg__ + IMJLP__
        maxError = max([abs(RE_LHS__ - RE_RHS__).max(),
                        abs(IM_LHS__ - IM_RHS__).max(),])
        print(f'负极固相电势方程最大误差{maxError: 8e} A')

        gradREφsposInterfaces__ = hstack([zeros([Nf, 1]), (REφspos__[:, 1:] - REφspos__[:, :-1])/Δxpos, full([Nf, 1], -ΔIAC/σpos)])
        ΔREφspos__ = (gradREφsposInterfaces__[:, 1:] - gradREφsposInterfaces__[:, :-1])/Δxpos
        gradIMφsposInterfaces__  = hstack([zeros([Nf, 1]), (IMφspos__[:, 1:] - IMφspos__[:, :-1])/Δxpos, zeros([Nf, 1])])
        ΔIMφspos__ = (gradIMφsposInterfaces__[:, 1:] - gradIMφsposInterfaces__[:, :-1])/Δxpos
        RE_LHS__ = σpos*ΔREφspos__
        RE_RHS__ = REJintpos__ + REJDLpos__
        IM_LHS__ = σpos*ΔIMφspos__
        IM_RHS__ = IMJintpos__ + IMJDLpos__
        maxError = max([abs(RE_LHS__ - RE_RHS__).max(),
                        abs(IM_LHS__ - IM_RHS__).max(), ])
        print(f'正极固相电势方程最大误差{maxError: 8e} A')

        term1__ = κ_*(gradREφeEast__ - gradREφeWest__)/Δx_
        term2__ = (κDκT_*(gradREθeEast__/θeEast_ - REθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradREθeWest__/θeWest_ - REθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([REJintneg__ + REJDLneg__ + REJLP__ , zeros([Nf, Nsep]), REJintpos__ + REJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势实部方程最大误差{maxError: 8e} A')

        term1__ = κ_*(gradIMφeEast__ - gradIMφeWest__)/Δx_
        term2__ = (κDκT_*(gradIMθeEast__/θeEast_ - IMθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradIMθeWest__/θeWest_ - IMθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([IMJintneg__ + IMJDLneg__ + IMJLP__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势虚部方程最大误差{maxError: 8e} A')

        maxError = max([
            abs(REJintneg__ - 2*(REI0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMJintneg__ - 2*(IMI0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REJintpos__ - 2*(REI0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMJintpos__ - 2*(IMI0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(), ])
        print(f'主反应BV动力学方程最大误差{maxError: 8e} A')

        dI0intdθeneg_, dI0intdθepos_ = self.dI0intdθeneg_, self.dI0intdθepos_
        dI0intdθsnegsurf_, dI0intdθspossurf_ = self.dI0intdθsnegsurf_, self.dI0intdθspossurf_
        maxError = max([
            abs(REI0intneg__ - (dI0intdθeneg_*REθe__[:, :Nneg] + dI0intdθsnegsurf_*REθsnegsurf__)).max(),
            abs(IMI0intneg__ - (dI0intdθeneg_*IMθe__[:, :Nneg] + dI0intdθsnegsurf_*IMθsnegsurf__)).max(),
            abs(REI0intpos__ - (dI0intdθepos_*REθe__[:, -Npos:] + dI0intdθspossurf_*REθspossurf__)).max(),
            abs(IMI0intpos__ - (dI0intdθepos_*IMθe__[:, -Npos:] + dI0intdθspossurf_*IMθspossurf__)).max(), ])
        print(f'交换电流方程最大误差{maxError: 8e} A')

        dUOCPdθsnegsurf_, dUOCPdθspossurf_ = self.dUOCPdθsnegsurf_, self.dUOCPdθspossurf_
        maxError = max([
            abs(REηintneg__ - (REφsneg__ - REφe__[:, :Nneg] - dUOCPdθsnegsurf_*REθsnegsurf__ - RSEIneg*(REJintneg__ + REJDLneg__ + REJLP__))).max(),
            abs(IMηintneg__ - (IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdθsnegsurf_*IMθsnegsurf__ - RSEIneg*(IMJintneg__ + IMJDLneg__ + IMJLP__))).max(),
            abs(REηintpos__ - (REφspos__ - REφe__[:, -Npos:] - dUOCPdθspossurf_*REθspossurf__ - RSEIpos*(REJintpos__ + REJDLpos__))).max(),
            abs(IMηintpos__ - (IMφspos__ - IMφe__[:, -Npos:] - dUOCPdθspossurf_*IMθspossurf__ - RSEIpos*(IMJintpos__ + IMJDLpos__))).max(), ])
        print(f'过电位方程最大误差{maxError: 8e} V')

        if self.lithiumPlating:
            dJLPdθe_, dJLPdηLP_ = self.dJLPdθe_, self.dJLPdηLP_
            maxError = max([
                abs(REJLP__ - (dJLPdθe_*REθe__[:, :Nneg] + dJLPdηLP_*REηLP__) ).max(),
                abs(IMJLP__ - (dJLPdθe_*IMθe__[:, :Nneg] + dJLPdηLP_*IMηLP__) ).max(),])
            print(f'析锂BV动力学方程最大误差{maxError: 8e} A')

            maxError = max([
                abs(REηLP__ - (REφsneg__ - REφe__[:, :Nneg] - RSEIneg*(REJintneg__ + REJDLneg__ + REJLP__))).max(),
                abs(IMηLP__ - (IMφsneg__ - IMφe__[:, :Nneg] - RSEIneg*(IMJintneg__ + IMJDLneg__ + IMJLP__))).max(), ])
            print(f'析锂过电位方程最大误差{maxError: 8e} V')


if __name__=='__main__':
    import numpy as np
    cell = LPJTFP2D(
        SOC0=0.1,
        # I0intneg=18, I0intpos=22,
        Nneg=8, Nsep=7, Npos=6, Nr=5,
        f_=np.logspace(3, -3, 61),
        lithiumPlating=True,
        # doubleLayerEffect=False,
        # complete=False,
        # constants=True,
        )

    cell.count_lithium()
    cell.EIS()
    thermalModel = 1
    cell.CC(-15, timeInterval=2000, thermalModel=thermalModel).EIS()
    cell.CC(-20, timeInterval=500, thermalModel=thermalModel).EIS()
    # cell.check_EIS()

    # cell.count_lithium()

    '''
    cell.plot_UI()
    cell.plot_TQgen()
    cell.plot_SOC()
    cell.plot_c(arange(0, 2001, 200))
    cell.plot_φ(arange(0, 2001, 200))
    cell.plot_jint(arange(0, 2001, 200))
    cell.plot_jDL(arange(0, 2001, 200))
    cell.plot_csr(range(0, 2001, 200), 1)
    cell.plot_jLP(arange(1000, 1601, 100))
    cell.plot_ηLP()
    cell.plot_OCV()

    cell.plot_Z()
    cell.plot_Nyquist()
    cell.plot_REθssurfIMEθssurf()
    cell.plot_REθeIMθe()
    cell.plot_REφsIMφs()
    cell.plot_REφeIMφe()
    cell.plot_REJintIMJint()
    cell.plot_REJDLIMJDL()
    cell.plot_REI0intIMI0int()
    cell.plot_REηintIMηint()
    '''
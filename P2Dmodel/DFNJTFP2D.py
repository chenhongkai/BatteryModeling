#%%
from typing import Sequence

from numpy import ndarray,\
    array, zeros, full, hstack, stack, \
    empty, logspace,\
    sinh, cosh, sqrt, exp, cos, sin, outer

from numpy.linalg import solve

from P2Dmodel.JTFbase import JTFbase, JTFbase__slots__
from P2Dmodel.DFNP2D import DFNP2D
from P2Dmodel.P2Dbase import P2Dbase


class DFNJTFP2D(JTFbase, DFNP2D):
    """锂离子电池时频联合经典准二维模型 Doyle-Fuller-Newman Joint Time-Frequency Pseudo-two-Dimension model"""

    __slots__ = (
        # DFNJTFP2D专有状态量
        'REcsnegsurf__', 'IMcsnegsurf__', 'REcspossurf__', 'IMcspossurf__', 'REce__', 'IMce__',
        'REjintneg__',   'IMjintneg__',   'REjintpos__',   'IMjintpos__',
        'REjDLneg__',    'IMjDLneg__',    'REjDLpos__',    'IMjDLpos__',
        'REi0intneg__',  'IMi0intneg__',  'REi0intpos__',  'IMi0intpos__',
        'REjLP__',       'IMjLP__',
        *JTFbase__slots__)

    def __init__(self,
            f_: Sequence[float] = logspace(3, -1, 26),  # 频率序列 [Hz]
            **kwargs):
        DFNP2D.__init__(self, **kwargs)
        JTFbase.__init__(self, f_)
        if self.complete:
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
            if self.lithiumPlating:
                self.REjLP__, self.IMjLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极析锂反应局部体积电流密度实部、虚部
            # 恒定量
            extra_EISdatanames_ = [                # 需记录的阻抗数据名称
                'REcsnegsurf__', 'IMcsnegsurf__',  # 负极固相表面锂离子浓度实部、虚部 [mol/m^3]
                'REcspossurf__', 'IMcspossurf__',  # 正极固相表面锂离子浓度实部、虚部 [mol/m^3]
                'REce__', 'IMce__',              # 电解液锂离子浓度实部、虚部 [mol/m^3]
                'REjintneg__', 'IMjintneg__',    # 负极主反应局部体积电流密度实部、虚部 [A/m^3]
                'REjintpos__', 'IMjintpos__',    # 正极主反应局部体积电流密度实部、虚部 [A/m^3]
                'REjDLneg__', 'IMjDLneg__',      # 负极双电层效应局部体积电流密度实部、虚部 [A/m^3]
                'REjDLpos__', 'IMjDLpos__',      # 正极双电层效应局部体积电流密度实部、虚部 [A/m^3]
                'REi0intneg__', 'IMi0intneg__',  # 负极主反应交换电流密度实部、虚部 [A/m^2]
                'REi0intpos__', 'IMi0intpos__',] # 正极主反应交换电流密度实部、虚部 [A/m^2]
            self.EISdatanames_.extend(extra_EISdatanames_)
            self.data.update({name: [] for name in extra_EISdatanames_})  # 字典：存储呈时间序列的阻抗数据

        if self.verbose and type(self) is DFNJTFP2D:
            print(self)
            print('经典时频联合P2D模型(DFNJTFP2D)初始化完成!')

    def solve_frequency_dependent_variables(self) -> dict[str, ndarray]:
        """求解频率相关变量"""
        ω_ = self.ω_
        solve_Kcssurf__ = DFNJTFP2D.solve_Kcssurf___
        CDLneg, CDLpos = self.CDLneg, self.CDLpos
        frequency_dependent_variables = {
            'ωεeΔx__': outer(ω_, self.εe_*self.Δx_),     # (Nf, Ne) 各频率各控制体的ω*εe*Δx值
            'ωaCDLneg_': ω_ * (self.aeffneg*CDLneg),     # (Nf,)
            'ωaCDLpos_': ω_ * (self.aeffpos*CDLpos),     # (Nf,)
            'ωCDLRSEIneg_': ω_ * (CDLneg*self.RSEIneg),  # (Nf,)
            'ωCDLRSEIpos_': ω_ * (CDLpos*self.RSEIpos),  # (Nf,)
            'minusKcsnegsurf___': -solve_Kcssurf__(ω_, self.Rsneg, self.Dsneg, self.aneg),   # (Nf, 2, 2) 负极各频率Kcssurf__矩阵
            'minusKcspossurf___': -solve_Kcssurf__(ω_, self.Rspos, self.Dspos, self.apos),}  # (Nf, 2, 2) 正极各频率Kcssurf__矩阵
        return frequency_dependent_variables

    def update_Kf__with_pure_electrochemical_parameters(self):
        # 对频域因变量线性矩阵Kf__赋纯电化学参数相关值
        self.update_Kf__REce_REce_and_IMce_IMce_(Deeff_ := self.Deeff_, Deeff_)
        self.update_Kf__REce_REj_and_IMce_IMj_(self.tplus)
        self.update_Kf__REφsneg_REjneg_and_IMφsneg_IMjneg_(σeffneg := self.σeffneg)
        self.update_Kf__REφspos_REjpos_and_IMφspos_IMjpos_(σeffpos := self.σeffpos)
        self.update_bKf_REφsneg0_and_REφsposEnd(σeffneg, σeffpos)
        self.update_Kf__REφe_REφe_and_IMφe_IMφe_(κeff_ := self.κeff_, κeff_)
        self.update_Kf__REηintneg_REjneg_and_IMηintneg_IMjneg_(self.RSEIneg, self.aeffneg)
        self.update_Kf__REηintpos_REjpos_and_IMηintpos_IMjpos_(self.RSEIpos, self.aeffpos)
        if self.lithiumPlating:
            self.update_Kf__REηLP_REjneg_and_IMηLP_IMjneg_(self.RSEIneg, self.aeffneg)

    def update_Kf__REce_REj_and_IMce_IMj_(self, tplus):
        # 更新Kf__矩阵REce行REj列、IMce行IMj列
        a = (1 - tplus)/P2Dbase.F
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        ravelKf_[sKf.sr_REceneg_REjintneg] = \
        ravelKf_[sKf.sr_REceneg_REjDLneg ]  = \
        ravelKf_[sKf.sr_IMceneg_IMjintneg] = \
        ravelKf_[sKf.sr_IMceneg_IMjDLneg ] = n = -self.Δxneg*a
        ravelKf_[sKf.sr_REcepos_REjintpos] = \
        ravelKf_[sKf.sr_REcepos_REjDLpos ]  = \
        ravelKf_[sKf.sr_IMcepos_IMjintpos] = \
        ravelKf_[sKf.sr_IMcepos_IMjDLpos ] = -self.Δxpos*a
        if self.lithiumPlating:
            ravelKf_[sKf.sr_REceneg_REjLP] = \
            ravelKf_[sKf.sr_IMceneg_IMjLP] = n

    def update_bKf_REφsneg0_and_REφsposEnd(self, σeffneg, σeffpos):
        bKf_ = self.bKf_
        ΔiAC = self.ΔiAC
        sKf = self.sKf
        # 更新bKf_向量REφsneg首元
        bKf_[sKf.s_REφsneg.start]    = -self.Δxneg*ΔiAC/σeffneg
        # 更新bKf_向量REφspos末元
        bKf_[sKf.s_REφspos.stop - 1] = self.Δxpos*ΔiAC/σeffpos

    def EIS(self):
        """计算电化学阻抗谱"""
        tEIS = self.t     # 读取：当前时刻 [s]
        if (tEIS_ := self.data['tEIS']) and tEIS_[-1:]==tEIS:
            if self.verbose:
                print(f'已计算时刻{tEIS = } s 电化学阻抗谱')
            return self
        REIMi0intnegUnknown = self._i0intneg is None
        REIMi0intposUnknown = self._i0intpos is None
        lithiumPlating = self.lithiumPlating

        # 读取参数
        solve_banded_matrix = P2Dbase.solve_banded_matrix
        Nneg, Nsep, Npos, Ne = self.Nneg, self.Nsep, self.Npos, self.Ne  # 读取：网格数
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_  # 读取：网格距离 [m]
        ΔiAC, ΔIAC = self.ΔiAC, self.ΔIAC
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        σeffneg, σeffpos = self.σeffneg, self.σeffpos
        κDeffWest_ = κDeffEast_ = self.κDeff_
        DeeffWest_ = DeeffEast_ = self.Deeff_

        if self.ravelKf_ is None:
            self._generate_Kf__bKf_and_slices()
            self.update_Kf__with_pure_electrochemical_parameters()

        ravelKf_ = self.ravelKf_  # 频域因变量矩阵Kf__展平视图
        bKf_ = self.bKf_          # 常数项向量
        Kf__ = ravelKf_.base

        if self.constants:
            cache = self.frequency_dependent_cache
        else:
            cache = self.solve_frequency_dependent_variables()
            self.update_Kf__with_pure_electrochemical_parameters()

        # 索引Kf__矩阵频率相关值
        sKf = self.sKf
        ravelKf_REcsnegsurf_REjintneg_ = ravelKf_[sKf.sr_REcsnegsurf_REjintneg]
        ravelKf_REcsnegsurf_IMjintneg_ = ravelKf_[sKf.sr_REcsnegsurf_IMjintneg]
        ravelKf_IMcsnegsurf_REjintneg_ = ravelKf_[sKf.sr_IMcsnegsurf_REjintneg]
        ravelKf_IMcsnegsurf_IMjintneg_ = ravelKf_[sKf.sr_IMcsnegsurf_IMjintneg]
        ravelKf_REcspossurf_REjintpos_ = ravelKf_[sKf.sr_REcspossurf_REjintpos]
        ravelKf_REcspossurf_IMjintpos_ = ravelKf_[sKf.sr_REcspossurf_IMjintpos]
        ravelKf_IMcspossurf_REjintpos_ = ravelKf_[sKf.sr_IMcspossurf_REjintpos]
        ravelKf_IMcspossurf_IMjintpos_ = ravelKf_[sKf.sr_IMcspossurf_IMjintpos]
        ravelKf_REce_IMce_ = ravelKf_[sKf.sr_REce_IMce]
        ravelKf_IMce_REce_ = ravelKf_[sKf.sr_IMce_REce]
        ravelKf_REjDLneg_IMφeneg_ = ravelKf_[sKf.sr_REjDLneg_IMφeneg]
        ravelKf_REjDLneg_IMφsneg_ = ravelKf_[sKf.sr_REjDLneg_IMφsneg]
        ravelKf_REjDLneg_IMjintneg_ = ravelKf_[sKf.sr_REjDLneg_IMjintneg]
        ravelKf_REjDLneg_IMjDLneg_ = ravelKf_[sKf.sr_REjDLneg_IMjDLneg]
        ravelKf_IMjDLneg_REφeneg_ = ravelKf_[sKf.sr_IMjDLneg_REφeneg]
        ravelKf_IMjDLneg_REφsneg_ = ravelKf_[sKf.sr_IMjDLneg_REφsneg]
        ravelKf_IMjDLneg_REjintneg_ = ravelKf_[sKf.sr_IMjDLneg_REjintneg]
        ravelKf_IMjDLneg_REjDLneg_ = ravelKf_[sKf.sr_IMjDLneg_REjDLneg]
        ravelKf_REjDLpos_IMφepos_ = ravelKf_[sKf.sr_REjDLpos_IMφepos]
        ravelKf_REjDLpos_IMφspos_ = ravelKf_[sKf.sr_REjDLpos_IMφspos]
        ravelKf_REjDLpos_IMjintpos_ = ravelKf_[sKf.sr_REjDLpos_IMjintpos]
        ravelKf_REjDLpos_IMjDLpos_ = ravelKf_[sKf.sr_REjDLpos_IMjDLpos]
        ravelKf_IMjDLpos_REφepos_ = ravelKf_[sKf.sr_IMjDLpos_REφepos]
        ravelKf_IMjDLpos_REφspos_ = ravelKf_[sKf.sr_IMjDLpos_REφspos]
        ravelKf_IMjDLpos_REjintpos_ = ravelKf_[sKf.sr_IMjDLpos_REjintpos]
        ravelKf_IMjDLpos_REjDLpos_ = ravelKf_[sKf.sr_IMjDLpos_REjDLpos]
        if lithiumPlating:
            ravelKf_REjDLneg_IMjLP_ = ravelKf_[sKf.sr_REjDLneg_IMjLP]
            ravelKf_IMjDLneg_REjLP_ = ravelKf_[sKf.sr_IMjDLneg_REjLP]

        ## 对Kf__矩阵赋时变值 ##

        self.update_Kf__REφe_REce_and_IMφe_IMce_(
            κDeffWest_, κDeffEast_,
            DeeffWest_, DeeffEast_,
            self.ce_, self.ceInterfaces_)

        # 负极主反应局部体积电流密度实部REjintneg行、虚部IMjintneg行
        if REIMi0intnegUnknown:
            ravelKf_[sKf.sr_REjintneg_REi0intneg] = \
            ravelKf_[sKf.sr_IMjintneg_IMi0intneg] = -self.djintdi0intneg_  # REIMi0intneg列
        ravelKf_[sKf.sr_REjintneg_REηintneg] = \
        ravelKf_[sKf.sr_IMjintneg_IMηintneg] = -self.djintdηintneg_        # REIMηintneg列
        if REIMi0intposUnknown:
            ravelKf_[sKf.sr_REjintpos_REi0intpos] = \
            ravelKf_[sKf.sr_IMjintpos_IMi0intpos] = -self.djintdi0intpos_  # IMi0intpos列
        # 正极局部体积电流密度实部REjintpos行、虚部IMjintpos行
        ravelKf_[sKf.sr_REjintpos_REηintpos] = \
        ravelKf_[sKf.sr_IMjintpos_IMηintpos] = -self.djintdηintpos_  # IMηintpos列

        if REIMi0intnegUnknown:
            # 负极交换电流密度实部REi0intneg行、虚部IMi0intneg行
            ravelKf_[sKf.sr_REi0intneg_REcsnegsurf] = \
            ravelKf_[sKf.sr_IMi0intneg_IMcsnegsurf] = -self.di0intdcsnegsurf_  # REIMcsnegsurf列
            ravelKf_[sKf.sr_REi0intneg_REceneg] = \
            ravelKf_[sKf.sr_IMi0intneg_IMceneg] = -self.di0intdceneg_          # REIMce列

        if REIMi0intposUnknown:
            # 正极交换电流密度实部REi0intpos行、虚部IMi0intpos行
            ravelKf_[sKf.sr_REi0intpos_REcspossurf] = \
            ravelKf_[sKf.sr_IMi0intpos_IMcspossurf] = -self.di0intdcspossurf_  # REIMcsnegsurf列
            ravelKf_[sKf.sr_REi0intpos_REcepos] = \
            ravelKf_[sKf.sr_IMi0intpos_IMcepos] = -self.di0intdcepos_          # REIMce列

        # 负极过电位实部REηintneg行REcsnegsurf列、虚部IMηintneg行IMcsnegsurf列
        ravelKf_[sKf.sr_REηintneg_REcsnegsurf] = \
        ravelKf_[sKf.sr_IMηintneg_IMcsnegsurf] = self.dUOCPdcsnegsurf_
        # 正极过电位实部REηintpos行REcspossurf列、虚部IMηintpos行IMcsnegsurf列
        ravelKf_[sKf.sr_REηintpos_REcspossurf] = \
        ravelKf_[sKf.sr_IMηintpos_IMcspossurf] = self.dUOCPdcspossurf_

        if lithiumPlating:
            # 析锂补充
            # 析锂局部体积电流实部REJLP行REθeneg列、虚部IMJLP行IMθeneg列
            ravelKf_[sKf.sr_REjLP_REceneg] = \
            ravelKf_[sKf.sr_IMjLP_IMceneg] = -self.djLPdce_
            # 析锂局部体积电流密度实部REjLP行REηLP列、虚部IMjLP行IMηLP列
            ravelKf_[sKf.sr_REjLP_REηLP] = \
            ravelKf_[sKf.sr_IMjLP_IMηLP] = -self.djLPdηLP_  # IMηLP列

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
            ravelKf_REcsnegsurf_REjintneg_[:] = minusKcsnegsurf__[0, 0]  # REjintneg列
            ravelKf_REcsnegsurf_IMjintneg_[:] = minusKcsnegsurf__[0, 1]  # IMjintneg列
            # 负极固相表面浓度虚部IMcsnegsurf行
            ravelKf_IMcsnegsurf_REjintneg_[:] = minusKcsnegsurf__[1, 0]  # REjintneg列
            ravelKf_IMcsnegsurf_IMjintneg_[:] = minusKcsnegsurf__[1, 1]  # IMjintneg列
            # 正极固相表面浓度实部REcspossurf行
            ravelKf_REcspossurf_REjintpos_[:] = minusKcspossurf__[0, 0]  # REjintpos列
            ravelKf_REcspossurf_IMjintpos_[:] = minusKcspossurf__[0, 1]  # IMjintpos列
            # 正极固相表面浓度虚部IMcspossurf行
            ravelKf_IMcspossurf_REjintpos_[:] = minusKcspossurf__[1, 0]  # REjintpos列
            ravelKf_IMcspossurf_IMjintpos_[:] = minusKcspossurf__[1, 1]  # IMjintpos列

            ravelKf_REce_IMce_[:] = -ωεeΔx_  # REce行IMce列
            ravelKf_IMce_REce_[:] =  ωεeΔx_   # IMce行REce列

            # 负极双电层局部体积电流密度实部REjDLneg行
            ravelKf_REjDLneg_IMφeneg_[:] = -ωaCDLneg   # IMφeneg列
            ravelKf_REjDLneg_IMφsneg_[:] = ωaCDLneg    # IMφsneg列
            ravelKf_REjDLneg_IMjintneg_[:] = \
            ravelKf_REjDLneg_IMjDLneg_[:] = -ωCDLRSEIneg  # IMjintneg列、IMjDLneg列
            # 负极双电层局部体积电流密度虚部IMjDLneg行
            ravelKf_IMjDLneg_REφeneg_[:] = ωaCDLneg       # REφeneg列
            ravelKf_IMjDLneg_REφsneg_[:] = -ωaCDLneg      # REφsneg列
            ravelKf_IMjDLneg_REjintneg_[:] = \
            ravelKf_IMjDLneg_REjDLneg_[:] = ωCDLRSEIneg  # REjintneg列、REjDLneg列
            # 正极双电层局部体积电流密度实部REjDLpos行
            ravelKf_REjDLpos_IMφepos_[:] = -ωaCDLpos       # IMφepos列
            ravelKf_REjDLpos_IMφspos_[:] = ωaCDLpos        # IMφspos列
            ravelKf_REjDLpos_IMjintpos_[:] = \
            ravelKf_REjDLpos_IMjDLpos_[:] = -ωCDLRSEIpos  # IMjintpos列、IMjDLpos列
            # 正极双电层局部体积电流密度虚部IMjDLpos行
            ravelKf_IMjDLpos_REφepos_[:] = ωaCDLpos       # REφepos列
            ravelKf_IMjDLpos_REφspos_[:] = -ωaCDLpos      # REφspos列
            ravelKf_IMjDLpos_REjintpos_[:] = \
            ravelKf_IMjDLpos_REjDLpos_[:] = ωCDLRSEIpos  # REjintpos列、REjDLpos列

            if lithiumPlating:
                # 补充
                ravelKf_REjDLneg_IMjLP_[:] = -ωCDLRSEIneg  # REjDLneg行IMjLP列
                ravelKf_IMjDLneg_REjLP_[:] =  ωCDLRSEIneg  # IMjDLneg行REjLP列

            if (self.banded_experience_of_Kf__ is None) and any(self.data['I']):
                self.banded_experience_of_Kf__ = expe = P2Dbase.banded_experience(Kf__)
                if self.verbose:
                    print(f'重排频域因变量矩阵Kf__的下带宽{expe['l']}，上带宽{expe['u']}')

            if expe := self.banded_experience_of_Kf__:
                # 带状化求解
                X__[nf] = solve_banded_matrix(Kf__, bKf_, **expe)
            else:
                # 直接求解
                X__[nf] = solve(Kf__, bKf_)

        REφsneg__ = X__[:, sKf.s_REφsneg]  # 负极固相电势实部
        IMφsneg__ = X__[:, sKf.s_IMφsneg]  # 负极固相电势虚部
        REφspos__ = X__[:, sKf.s_REφspos]  # 正极固相电势实部
        IMφspos__ = X__[:, sKf.s_IMφspos]  # 正极固相电势虚部

        self.tEIS = tEIS
        REφsnegCollector_ = REφsneg__[:, 0] + 0.5*Δxneg*ΔiAC/σeffneg   # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = IMφsneg__[:, 0]                            # (Nf,) 负极集流体电势虚部 [V]
        REφsposCollector_ = REφspos__[:, -1] - 0.5*Δxpos*ΔiAC/σeffpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = IMφspos__[:, -1]                           # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗虚部 [Ω]
        self.Z_[:] = Zreal_ + 1j*Zimag_ + self.Zl_

        if self.complete:
            self.REcsnegsurf__[:] = X__[:, sKf.s_REcsnegsurf]  # 负极固相表面浓度实部
            self.IMcsnegsurf__[:] = X__[:, sKf.s_IMcsnegsurf]  # 负极固相表面浓度虚部
            self.REcspossurf__[:] = X__[:, sKf.s_REcspossurf]  # 正极固相表面浓度实部
            self.IMcspossurf__[:] = X__[:, sKf.s_IMcspossurf]  # 正极固相表面浓度虚部
            self.REce__[:] = X__[:, sKf.s_REce]  # 电解液锂离子浓度实部
            self.IMce__[:] = X__[:, sKf.s_IMce]  # 电解液锂离子浓度虚部
            self.REφsneg__[:] = REφsneg__      # 负极固相电势实部
            self.IMφsneg__[:] = IMφsneg__      # 负极固相电势虚部
            self.REφspos__[:] = REφspos__      # 正极固相电势实部
            self.IMφspos__[:] = IMφspos__      # 正极固相电势虚部
            self.REφe__[:] = REφe__ = X__[:, sKf.s_REφe]  # 电解液电势实部
            self.IMφe__[:] = IMφe__ = X__[:, sKf.s_IMφe]  # 电解液电势虚部
            self.REjintneg__[:] = X__[:, sKf.s_REjintneg]  # 负极主反应局部体积电流密度实部
            self.IMjintneg__[:] = X__[:, sKf.s_IMjintneg]  # 负极主反应局部体积电流密度虚部
            self.REjintpos__[:] = X__[:, sKf.s_REjintpos]  # 正极主反应局部体积电流密度实部
            self.IMjintpos__[:] = X__[:, sKf.s_IMjintpos]  # 正极主反应局部体积电流密度虚部
            self.REjDLneg__[:] = X__[:, sKf.s_REjDLneg]  # 负极双电层局部体积电流密度实部
            self.IMjDLneg__[:] = X__[:, sKf.s_IMjDLneg]  # 负极双电层局部体积电流密度虚部
            self.REjDLpos__[:] = X__[:, sKf.s_REjDLpos]  # 正极双电层局部体积电流密度实部
            self.IMjDLpos__[:] = X__[:, sKf.s_IMjDLpos]  # 正极双电层局部体积电流密度虚部
            self.REi0intneg__[:] = X__[:, sKf.s_REi0intneg] if REIMi0intnegUnknown else 0. # 负极交换电流密度实部
            self.IMi0intneg__[:] = X__[:, sKf.s_IMi0intneg] if REIMi0intnegUnknown else 0. # 负极交换电流密度虚部
            self.REi0intpos__[:] = X__[:, sKf.s_REi0intpos] if REIMi0intposUnknown else 0. # 正极交换电流密度实部
            self.IMi0intpos__[:] = X__[:, sKf.s_IMi0intpos] if REIMi0intposUnknown else 0. # 正极交换电流密度虚部
            self.REηintneg__[:] = X__[:, sKf.s_REηintneg]  # 负极过电位实部
            self.IMηintneg__[:] = X__[:, sKf.s_IMηintneg]  # 负极过电位虚部
            self.REηintpos__[:] = X__[:, sKf.s_REηintpos]  # 正极过电位实部
            self.IMηintpos__[:] = X__[:, sKf.s_IMηintpos]  # 正极过电位虚部
            if lithiumPlating:
                self.REjLP__[:] = X__[:, sKf.s_REjLP]  # 负极析锂局部体积电流密度实部
                self.IMjLP__[:] = X__[:, sKf.s_IMjLP]  # 负极析锂局部体积电流密度虚部
                self.REηLP__[:] = X__[:, sKf.s_REηLP]  # 负极析锂过电位实部
                self.IMηLP__[:] = X__[:, sKf.s_IMηLP]  # 负极析锂过电位虚部

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
    def solve_Kcssurf___(
            ω_: ndarray,  # (Nf,) 角频率序列 [rad/s]
            Rs: float,    # 颗粒半径 [m]
            Ds: float,    # 固相扩散系数 [m^2/s]
            a: float,     # 比表面积 [m^2/m^3]
            ):
        """求Kcssurf___矩阵
        Kcssurf___ @ stack([REjint__, IMjint__], axis=1) = stack([REcssurf__, IMcssurf__], axis=1)"""
        W2_ = ω_*Rs**2/Ds
        W_ = sqrt(W2_)
        root2W_ = 1.4142135623730951*W_
        γ_ = 0.7071067811865476*W_
        cosγ_ = cos(γ_)
        sinγ_ = sin(γ_)
        cosγ2_ = cosγ_*cosγ_
        sinγ2_ = sinγ_*sinγ_
        cosγsinγ = cosγ_*sinγ_
        aFDs = a * P2Dbase.F * Ds

        # 指数缩放
        # cosh²γ 和 coshγ·sinhγ 是 ~exp(2γ) 级别的大数，容易溢出，不能直接算
        # 因此，把 cosh²γ、coshγ·sinhγ 全部乘 exp(-2γ)，转化成O(1)级别，防止溢出
        exp_2γ_ = exp(-2*γ_)
        m_ = 1 + exp_2γ_
        coshγ2_s_ = 0.25*(m_*m_)                    # cosh²γ * exp(-2γ)
        coshγsinhγ_s_ = 0.25*(1 - exp_2γ_*exp_2γ_)  # coshγ·sinhγ * exp(-2γ)

        a_ = -Rs*root2W_ * (coshγsinhγ_s_ + cosγsinγ * exp_2γ_) + Rs*2 * (coshγ2_s_ - cosγ2_ * exp_2γ_)
        b_ = -Rs*root2W_ * (coshγsinhγ_s_ - cosγsinγ * exp_2γ_)
        d_ = 2*aFDs*((W2_ + 1) * coshγ2_s_
                     - root2W_ * coshγsinhγ_s_
                     - W2_     * sinγ2_   * exp_2γ_
                     - root2W_ * cosγsinγ * exp_2γ_
                     -           cosγ2_   * exp_2γ_)
        a_ /= d_
        b_ /= d_
        Kcssurf___ = empty((ω_.size, 2, 2))
        Kcssurf___[:, 0, 0] = Kcssurf___[:, 1, 1] = a_
        Kcssurf___[:, 0, 1] = b_
        Kcssurf___[:, 1, 0] = -b_
        return Kcssurf___  # (Nf, 2, 2)

    @staticmethod
    def solve_REcs_IMcs(
            r: float,      # 径向坐标 [m]
            ω_: ndarray,   # (Nf,) 角频率序列 [rad/s]
            Rs: float,  # 颗粒半径 [m]
            Ds: float,  # 固相扩散系数 [m^2/s]
            a: float,   # 比表面积 [m^2/m^3]
            REjint__: ndarray,  # (Nf, Nreg) 主反应局部体积电流密度实部 [A/m^3]
            IMjint__: ndarray,  # (Nf, Nreg) 主反应局部体积电流密度虚部 [A/m^3]
            ):
        """固相浓度实部、虚部在r处的解析解"""
        Rs2 = Rs*Rs
        W2_ = Rs2/Ds*ω_   # [–]
        W_ = sqrt(W2_)    # [–]
        root2W_ = 1.4142135623730951*W_
        γ_ = 0.7071067811865476*W_ # [–]
        γr_ = γ_*r/Rs  # [–]
        sinγ_ = sin(γ_)
        cosγ_ = cos(γ_)
        sinγr_ = sin(γr_)
        cosγr_ = cos(γr_)
        aFDsr = a * P2Dbase.F * Ds * r
        # 指数缩放
        # coshγ coshγr sinhγ sinhγr 是 ~exp(γ) 级别的大数，容易溢出，不能直接算
        # 应缩放：统一乘exp(-γ)
        exp_γ_ = exp(-γ_)
        exp_2γ_ = exp_γ_*exp_γ_
        half_exp_2γ_ = 0.5*exp_2γ_
        coshγ_s_ = 0.5 + half_exp_2γ_  # coshγ*exp(-γ_)
        sinhγ_s_ = 0.5 - half_exp_2γ_  # sinhγ*exp(-γ_)

        exp_γr_ = exp(-γr_)
        q_ = exp(γ_*(r/Rs - 1))
        p_ = exp_γ_*exp_γr_
        coshγr_s_ = 0.5*(q_ + p_)  # coshγr*exp(-γ_)
        sinhγr_s_ = 0.5*(q_ - p_)  # sinhγr*exp(-γ_)

        a_ = Rs2*(- root2W_  * coshγr_s_ * sinγr_ * coshγ_s_ * cosγ_
                  + 2        * coshγr_s_ * sinγr_ * coshγ_s_ * sinγ_
                  - root2W_  * coshγr_s_ * sinγr_ * sinhγ_s_ * sinγ_
                  - root2W_  * sinhγr_s_ * cosγr_ * coshγ_s_ * cosγ_
                  + root2W_  * sinhγr_s_ * cosγr_ * sinhγ_s_ * sinγ_
                  + 2        * sinhγr_s_ * cosγr_ * sinhγ_s_ * cosγ_)
        b_ = Rs2*(  root2W_ * coshγ_s_ * sinγr_ * coshγ_s_ * cosγ_
                  - root2W_ * coshγ_s_ * sinγr_ * sinhγ_s_ * sinγ_
                  - 2       * coshγ_s_ * sinγr_ * sinhγ_s_ * cosγ_
                  - root2W_ * sinhγ_s_ * cosγr_ * coshγ_s_ * cosγ_
                  + 2       * sinhγ_s_ * cosγr_ * coshγ_s_ * sinγ_
                  - root2W_ * sinhγ_s_ * cosγr_ * sinhγ_s_ * sinγ_)
        d_ = 2*aFDsr*( (W2_ + 1) * coshγ_s_ * coshγ_s_
                       - root2W_ * coshγ_s_ * sinhγ_s_
                       - W2_     * sinγ_    * sinγ_ * exp_2γ_
                       - root2W_ * cosγ_    * sinγ_ * exp_2γ_
                       -           cosγ_    * cosγ_ * exp_2γ_)
        a_ /= d_
        b_ /= d_
        Kcs___ = empty((ω_.size, 2, 2))
        Kcs___[:, 0, 0] = Kcs___[:, 1, 1] = a_
        Kcs___[:, 0, 1] = b_
        Kcs___[:, 1, 0] = -b_
        results___ = Kcs___ @ stack([REjint__, IMjint__], axis=1)  # (Nf, 2, Nreg)
        REcs__, IMcs__ = results___[:, 0, :], results___[:, 1, :]
        return REcs__, IMcs__  # (Nf, Nreg)

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
        F2RT = 0.5 * P2Dbase.F/P2Dbase.R/self.T
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        ΔIAC, ΔiAC = self.ΔIAC, self.ΔiAC
        σeffneg, σeffpos = self.σeffneg, self.σeffpos
        aeffneg, aeffpos = self.aeffneg, self.aeffpos
        εe_ = self.εe_
        DeeffWest_ = DeeffEast_ = self.Deeff_

        # 各控制体界面的电解液锂离子浓度实部 [mol/m^3]
        REceInterfaces__ = hstack([REce__[:, [0]], (REce__[:, :-1] + REce__[:, 1:]) * 0.5, REce__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面锂离子浓度
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
        IMceInterfaces__ = hstack([IMce__[:, [0]], (IMce__[:, :-1] + IMce__[:, 1:]) * 0.5, IMce__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面锂离子浓度
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
        print(f'固相表面浓度解析解方程 cssurf 最大误差{maxError} mol/m^3')

        LHS__ = -outer(ω_, εe_) * IMce__
        RHS__ = (DeeffEast_*gradREceEast__ - DeeffWest_*gradREceWest__)/Δx_ + (1 - self.tplus)/P2Dbase.F*hstack([REjintneg__ + REjDLneg__ + REjLP__, zeros([Nf, Nsep]), REjintpos__ + REjDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液实部浓度方程 REce 最大误差{maxError} [mol/m^3/s]')

        LHS__ = outer(ω_, εe_) * REce__
        RHS__ = (DeeffEast_*gradIMceEast__ - DeeffWest_*gradIMceWest__)/Δx_ + (1 - self.tplus)/P2Dbase.F*hstack([IMjintneg__ + IMjDLneg__ + IMjLP__, zeros([Nf, Nsep]), IMjintpos__ + IMjDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液虚部浓度方程 IMce 最大误差{maxError} [mol/m^3/s]')

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
        print(f'负极固相电势方程 REφsneg IMEφsneg 最大误差{maxError} A/m^3')

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
        print(f'正极固相电势方程 REφspos IMEφspos 最大误差{maxError} A/m^3')

        i0intneg_, i0intpos_ = self.i0intneg_, self.i0intpos_
        ηintneg_, ηintpos_ = self.ηintneg_, self.ηintpos_
        maxError = max([
            abs(REjintneg__ - 2*aeffneg*(REi0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMjintneg__ - 2*aeffneg*(IMi0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REjintpos__ - 2*aeffpos*(REi0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMjintpos__ - 2*aeffpos*(IMi0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),])
        print(f'主反应BV动力学方程 REjing IMjing 最大误差{maxError} A/m^3')

        di0intdceneg_, di0intdcepos_ = self.di0intdceneg_, self.di0intdcepos_
        di0intdcsnegsurf_, di0intdcspossurf_ = self.di0intdcsnegsurf_, self.di0intdcspossurf_
        maxError = max([
            abs(REi0intneg__ - (di0intdceneg_*REce__[:, :Nneg] + di0intdcsnegsurf_*REcsnegsurf__)).max(),
            abs(IMi0intneg__ - (di0intdceneg_*IMce__[:, :Nneg] + di0intdcsnegsurf_*IMcsnegsurf__)).max(),
            abs(REi0intpos__ - (di0intdcepos_*REce__[:, -Npos:] + di0intdcspossurf_*REcspossurf__)).max(),
            abs(IMi0intpos__ - (di0intdcepos_*IMce__[:, -Npos:] + di0intdcspossurf_*IMcspossurf__)).max(), ])
        print(f'主反应交换电流密度方程 REi0int IMi0int 最大误差{maxError} A/m^2')

        dUOCPdcsnegsurf_, dUOCPdcspossurf_ = self.dUOCPdcsnegsurf_, self.dUOCPdcspossurf_
        maxError = max([
            abs(REηintneg__ - (REφsneg__ - REφe__[:, :Nneg] - dUOCPdcsnegsurf_*REcsnegsurf__ - RSEI2aeffneg*(REjintneg__ + REjDLneg__ + REjLP__))).max(),
            abs(IMηintneg__ - (IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdcsnegsurf_*IMcsnegsurf__ - RSEI2aeffneg*(IMjintneg__ + IMjDLneg__ + IMjLP__))).max(),
            abs(REηintpos__ - (REφspos__ - REφe__[:, -Npos:] - dUOCPdcspossurf_*REcspossurf__ - RSEI2aeffpos*(REjintpos__ + REjDLpos__))).max(),
            abs(IMηintpos__ - (IMφspos__ - IMφe__[:, -Npos:] - dUOCPdcspossurf_*IMcspossurf__ - RSEI2aeffpos*(IMjintpos__ + IMjDLpos__))).max(),])
        print(f'主反应过电位方程 REηint IMηint 最大误差{maxError} [V]')

        if self.lithiumPlating:
            djLPdce_, djLPdηLP_ = self.djLPdce_, self.djLPdηLP_
            maxError = max([
                abs(REjLP__ - (djLPdce_*REce__[:, :Nneg] + djLPdηLP_*REηLP__)).max(),
                abs(IMjLP__ - (djLPdce_*IMce__[:, :Nneg] + djLPdηLP_*IMηLP__)).max(),])
            print(f'析锂BV动力学方程 REjLP IMjLP 最大误差{maxError} [A/m^3]')

            maxError = max([
                abs(REηLP__ - (REφsneg__ - REφe__[:, :Nneg] - RSEI2aeffneg*(REjintneg__ + REjDLneg__ + REjLP__))).max(),
                abs(IMηLP__ - (IMφsneg__ - IMφe__[:, :Nneg] - RSEI2aeffneg*(IMjintneg__ + IMjDLneg__ + IMjLP__))).max(),])
            print(f'析锂过电位方程 REηLP IMηLP 最大误差{maxError} [V]')


if __name__=='__main__':
    import numpy as np
    cell = DFNJTFP2D(
        Δt=10, SOC0=0.1,
        CDLpos=0.1, CDLneg=0.7,
        # i0intpos=3.67, i0intneg=3.30,
        Aeffneg=0.9, Aeffpos=0.8,
        Nneg=8, Nsep=6, Npos=7, Nr=9,
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
    # cell.checkEIS()
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

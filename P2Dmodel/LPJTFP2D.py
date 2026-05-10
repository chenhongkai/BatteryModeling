#%%
from typing import Sequence

from numpy import ndarray,\
    array, empty, zeros, full, logspace, hstack, stack, \
    sinh, cosh, cos, sin, exp, sqrt, outer
from numpy.linalg import solve

from P2Dmodel.LPP2D import LPP2D
from P2Dmodel.JTFbase import JTFbase, JTFbase__slots__
from P2Dmodel.P2Dbase import P2Dbase

class LPJTFP2D(JTFbase, LPP2D):
    """锂离子电池集总参数时频联合准二维模型（Lumped-Parameter Joint Time-Frequency Pseudo-two-Dimension model）"""

    __slots__ = (
        # LPJTFP2D专有状态量
        'REθsnegsurf__', 'IMθsnegsurf__', 'REθspossurf__', 'IMθspossurf__', 'REθe__', 'IMθe__',
        'REJintneg__',   'IMJintneg__',   'REJintpos__',   'IMJintpos__',
        'REJDLneg__',    'IMJDLneg__',    'REJDLpos__',    'IMJDLpos__',
        'REI0intneg__',  'IMI0intneg__',  'REI0intpos__',  'IMI0intpos__',
        'REJLP__', 'IMJLP__',
        *JTFbase__slots__)

    def __init__(self,
            f_: Sequence[float] = logspace(3, -1, 26),  # 频率序列 [Hz]
            **kwargs):
        LPP2D.__init__(self, **kwargs)
        JTFbase.__init__(self, f_)
        if self.complete:
            Nf = len(f_)
            Nneg, Npos, Ne = self.Nneg, self.Npos, self.Ne
            # 状态量
            self.REθsnegsurf__, self.IMθsnegsurf__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极固相表面浓度实部、虚部
            self.REθspossurf__, self.IMθspossurf__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极固相表面浓度实部、虚部
            self.REθe__, self.IMθe__ = empty((Nf, Ne)), empty((Nf, Ne))                    # 电解液锂离子浓度实部、虚部
            self.REJintneg__, self.IMJintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))      # 负极主反应局部体积电流密度实部、虚部
            self.REJintpos__, self.IMJintpos__ = empty((Nf, Npos)), empty((Nf, Npos))      # 正极主反应局部体积电流密度实部、虚部
            self.REJDLneg__, self.IMJDLneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))        # 负极双电层局部体积电流密度实部、虚部
            self.REJDLpos__, self.IMJDLpos__ = empty((Nf, Npos)), empty((Nf, Npos))        # 正极双电层局部体积电流密度实部、虚部
            self.REI0intneg__, self.IMI0intneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))    # 负极交换电流密度实部、虚部
            self.REI0intpos__, self.IMI0intpos__ = empty((Nf, Npos)), empty((Nf, Npos))    # 正极交换电流密度实部、虚部
            if self.lithiumPlating:
                self.REJLP__, self.IMJLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 析锂反应局部体积电流密度实部、虚部
            # 恒定量
            extra_EISdatanames_ = [   # 需记录的阻抗数据名称
                'REθsnegsurf__', 'IMθsnegsurf__',  # 负极固相表面无量纲锂离子浓度实部、虚部 [–]
                'REθspossurf__', 'IMθspossurf__',  # 正极固相表面无量纲锂离子浓度实部、虚部 [–]
                'REθe__', 'IMθe__',                # 电解液无量纲锂离子浓度实部、虚部 [–]
                'REJintneg__', 'IMJintneg__',      # 负极主反应集总局部体积电流密度实部、虚部 [A]
                'REJintpos__', 'IMJintpos__',      # 正极主反应集总局部体积电流密度实部、虚部 [A]
                'REJDLneg__', 'IMJDLneg__',        # 负极双电层效应集总局部体积电流密度实部、虚部 [A]
                'REJDLpos__', 'IMJDLpos__',        # 正极双电层效应集总局部体积电流密度实部、虚部 [A]
                'REI0intneg__', 'IMI0intneg__',    # 负极主反应集总交换电流密度实部、虚部 [A]
                'REI0intpos__', 'IMI0intpos__',]   # 正极主反应集总交换电流密度实部、虚部 [A]
            self.EISdatanames_.extend(extra_EISdatanames_)
            self.data.update({name: [] for name in extra_EISdatanames_})  # 字典：存储呈时间序列的阻抗数据

        if self.verbose and type(self) is LPJTFP2D:
            print(self)
            print('集总参数时频联合P2D模型(LPJTFP2D)初始化完成!')

    def solve_frequency_dependent_variables(self):
        """求解频率相关变量"""
        ω_ = self.ω_
        solve_Kθssurf___ = LPJTFP2D.solve_Kθssurf___
        frequency_dependent_variables = {
            'ωqeΔx__': outer(ω_, self.qe_*self.Δx_),   # (len(f_), Ne) 各频率各控制体的ω*qe*Δx值
            'ωCDLneg_': (ωCDLneg_ := ω_*self.CDLneg),  # (Ne,)
            'ωCDLpos_': (ωCDLpos_ := ω_*self.CDLpos),  # (Ne,)
            'ωCDLRSEIneg_': ωCDLneg_*self.RSEIneg,     # (Ne,)
            'ωCDLRSEIpos_': ωCDLpos_*self.RSEIpos,     # (Ne,)
            'minusKθsnegsurf___': -solve_Kθssurf___(ω_, self.Qneg, self.Dsneg),   # (Nf, 2, 2) 负极各频率Kθssurf__矩阵
            'minusKθspossurf___': -solve_Kθssurf___(ω_, self.Qpos, self.Dspos),}  # (Nf, 2, 2) 正极各频率Kθssurf__矩阵
        return frequency_dependent_variables

    def _generate_Kf__bKf_and_slices(self):
        # 生成频域因变量矩阵Kf__、常数项向量bKf_及切片索引，并对Kf__赋常系数、几何网格相关参数
        JTFbase._generate_Kf__bKf_and_slices(self)
        # 集总参数模型需额外赋固定值（经典DFN-P2D模型此处为参数tplus相关的值）
        # 电解液浓度实部REθe行、虚部IMθe行
        ravelKf_ = self.ravelKf_
        sKf = self.sKf
        ravelKf_[sKf.sr_REceneg_REjintneg] = \
        ravelKf_[sKf.sr_REceneg_REjDLneg ] = \
        ravelKf_[sKf.sr_IMceneg_IMjintneg] = \
        ravelKf_[sKf.sr_IMceneg_IMjDLneg ] = -self.Δxneg  # REJneg、IMJneg列
        ravelKf_[sKf.sr_REcepos_REjintpos] = \
        ravelKf_[sKf.sr_REcepos_REjDLpos ] = \
        ravelKf_[sKf.sr_IMcepos_IMjintpos] = \
        ravelKf_[sKf.sr_IMcepos_IMjDLpos ] = -self.Δxpos  # REJpos、IMJpos列
        if self.lithiumPlating:
            ravelKf_[sKf.sr_REceneg_REjLP] = \
            ravelKf_[sKf.sr_IMceneg_IMjLP] = -self.Δxneg  # REθe行REJLP列、IMθe行IMJLP列

    def update_Kf__with_pure_electrochemical_parameters(self):
        # 对频域因变量线性矩阵Kf__赋纯电化学参数相关值
        self.update_Kf__REθe_REθe_and_IMθe_IMθe_(self.Deκ_)
        self.update_Kf__REφsneg_REJneg_and_IMφsneg_IMJneg_(σneg := self.σneg)
        self.update_Kf__REφspos_REJpos_and_IMφspos_IMJpos_(σpos := self.σpos)
        self.update_bKf_REφsneg0_and_REφsposEnd(σneg, σpos)
        self.update_Kf__REφe_REφe_and_IMφe_IMφe_(κ_ := self.κ_, κ_)
        self.update_Kf__REηintneg_REJneg_and_IMηintneg_IMJneg_(self.RSEIneg)
        self.update_Kf__REηintpos_REJpos_and_IMηintpos_IMJpos_(self.RSEIpos)
        if self.lithiumPlating:
            self.update_Kf__REηLP_REJneg_and_IMηLP_IMJneg_(self.RSEIneg)

    def update_Kf__REθe_REθe_and_IMθe_IMθe_(self, Deκ_):
        # 更新Kf__矩阵REθe行REθe列、IMθe行IMθe列
        JTFbase.update_Kf__REce_REce_and_IMce_IMce_(self, Deκ_, Deκ_)

    def update_Kf__REφsneg_REJneg_and_IMφsneg_IMJneg_(self, σneg):
        # 更新Kf__矩阵REφsneg行REJneg列、IMφsneg行IMJneg列
        JTFbase.update_Kf__REφsneg_REjneg_and_IMφsneg_IMjneg_(self, σneg)

    def update_Kf__REφspos_REJpos_and_IMφspos_IMJpos_(self, σpos):
        # 更新Kf__矩阵REφspos行REJpos列、IMφspos行IMJpos列
        JTFbase.update_Kf__REφspos_REjpos_and_IMφspos_IMjpos_(self, σpos)

    def update_bKf_REφsneg0_and_REφsposEnd(self, σneg, σpos):
        bKf_ = self.bKf_
        ΔIAC = self.ΔIAC
        sKf = self.sKf
        # 更新bKf_向量REφsneg首元
        bKf_[sKf.s_REφsneg.start]   = -self.Δxneg*ΔIAC/σneg
        # 更新bKf_向量REφspos末元
        bKf_[sKf.s_REφspos.stop - 1] = self.Δxpos*ΔIAC/σpos

    def update_Kf__REηintneg_REJneg_and_IMηintneg_IMJneg_(self, RSEIneg):
        # 更新Kf__矩阵REηintneg行REJneg列、IMηintneg行IMJneg列
        JTFbase.update_Kf__REηintneg_REjneg_and_IMηintneg_IMjneg_(self, RSEIneg, 1)

    def update_Kf__REηintpos_REJpos_and_IMηintpos_IMJpos_(self, RSEIpos):
        # 更新Kf__矩阵REηintpos行REJpos列、IMηintpos行IMJpos列
        JTFbase.update_Kf__REηintpos_REjpos_and_IMηintpos_IMjpos_(self, RSEIpos, 1)

    def update_Kf__REηLP_REJneg_and_IMηLP_IMJneg_(self, RSEIneg):
        # 更新Kf__矩阵REηLP行REJneg列、IMηLP行IMJneg列
        JTFbase.update_Kf__REηLP_REjneg_and_IMηLP_IMjneg_(self, RSEIneg, 1)

    def EIS(self):
        """计算电化学阻抗谱"""
        tEIS = self.t     # 读取：当前时刻 [s]
        if (tEIS_ := self.data['tEIS']) and tEIS_[-1:]==tEIS:
            if self.verbose:
                print(f'已计算时刻{tEIS = } s 电化学阻抗谱')
            return self

        # 读取参数
        solve_banded_matrix = P2Dbase.solve_banded_matrix
        ΔIAC = self.ΔIAC
        κ_ = self.κ_
        κDκT_ = self.κD * self.T * κ_

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
        ravelKf_REθsnegsurf_REJintneg_ = ravelKf_[sKf.sr_REcsnegsurf_REjintneg]
        ravelKf_REθsnegsurf_IMJintneg_ = ravelKf_[sKf.sr_REcsnegsurf_IMjintneg]
        ravelKf_IMθsnegsurf_REJintneg_ = ravelKf_[sKf.sr_IMcsnegsurf_REjintneg]
        ravelKf_IMθsnegsurf_IMJintneg_ = ravelKf_[sKf.sr_IMcsnegsurf_IMjintneg]
        ravelKf_REθspossurf_REJintpos_ = ravelKf_[sKf.sr_REcspossurf_REjintpos]
        ravelKf_REθspossurf_IMJintpos_ = ravelKf_[sKf.sr_REcspossurf_IMjintpos]
        ravelKf_IMθspossurf_REJintpos_ = ravelKf_[sKf.sr_IMcspossurf_REjintpos]
        ravelKf_IMθspossurf_IMJintpos_ = ravelKf_[sKf.sr_IMcspossurf_IMjintpos]
        ravelKf_REθe_IMθe_ = ravelKf_[sKf.sr_REce_IMce]
        ravelKf_IMθe_REθe_ = ravelKf_[sKf.sr_IMce_REce]
        ravelKf_REJDLneg_IMφeneg_ = ravelKf_[sKf.sr_REjDLneg_IMφeneg]
        ravelKf_REJDLneg_IMφsneg_ = ravelKf_[sKf.sr_REjDLneg_IMφsneg]
        ravelKf_REJDLneg_IMJintneg_ = ravelKf_[sKf.sr_REjDLneg_IMjintneg]
        ravelKf_REJDLneg_IMJDLneg_ = ravelKf_[sKf.sr_REjDLneg_IMjDLneg]
        ravelKf_IMJDLneg_REφeneg_ = ravelKf_[sKf.sr_IMjDLneg_REφeneg]
        ravelKf_IMJDLneg_REφsneg_ = ravelKf_[sKf.sr_IMjDLneg_REφsneg]
        ravelKf_IMJDLneg_REJintneg_ = ravelKf_[sKf.sr_IMjDLneg_REjintneg]
        ravelKf_IMJDLneg_REJDLneg_ = ravelKf_[sKf.sr_IMjDLneg_REjDLneg]
        ravelKf_REJDLpos_IMφepos_ = ravelKf_[sKf.sr_REjDLpos_IMφepos]
        ravelKf_REJDLpos_IMφspos_ = ravelKf_[sKf.sr_REjDLpos_IMφspos]
        ravelKf_REJDLpos_IMJintpos_ = ravelKf_[sKf.sr_REjDLpos_IMjintpos]
        ravelKf_REJDLpos_IMJDLpos_ = ravelKf_[sKf.sr_REjDLpos_IMjDLpos]
        ravelKf_IMJDLpos_REφepos_ = ravelKf_[sKf.sr_IMjDLpos_REφepos]
        ravelKf_IMJDLpos_REφspos_ = ravelKf_[sKf.sr_IMjDLpos_REφspos]
        ravelKf_IMJDLpos_REJintpos_ = ravelKf_[sKf.sr_IMjDLpos_REjintpos]
        ravelKf_IMJDLpos_REJDLpos_ = ravelKf_[sKf.sr_IMjDLpos_REjDLpos]
        if lithiumPlating := self.lithiumPlating:
            ravelKf_REJDLneg_IMJLP_ = ravelKf_[sKf.sr_REjDLneg_IMjLP]
            ravelKf_IMJDLneg_REJLP_ = ravelKf_[sKf.sr_IMjDLneg_REjLP]

        ## 对Kf__矩阵赋时变值 ##

        self.update_Kf__REφe_REce_and_IMφe_IMce_(
            κDκT_, κDκT_, Deκ_:=self.Deκ_, Deκ_,
            self.θe_, self.θeInterfaces_)

        # 负极局部体积电流实部REJintneg行、虚部IMJintneg行
        if REIMI0intnegUnknown := (self._I0intneg is None):
            ravelKf_[sKf.sr_REjintneg_REi0intneg] = \
            ravelKf_[sKf.sr_IMjintneg_IMi0intneg] = -self.dJintdI0intneg_ # REIMI0intneg列
        ravelKf_[sKf.sr_REjintneg_REηintneg] = \
        ravelKf_[sKf.sr_IMjintneg_IMηintneg] = -self.dJintdηintneg_       # REIMηintneg列
        # 正极局部体积电流实部REJintpos行、虚部IMJintpos行
        if REIMI0intposUnknown := (self._I0intpos is None):
            ravelKf_[sKf.sr_REjintpos_REi0intpos] = \
            ravelKf_[sKf.sr_IMjintpos_IMi0intpos] = -self.dJintdI0intpos_  # REIMI0intpos列
        ravelKf_[sKf.sr_REjintpos_REηintpos] = \
        ravelKf_[sKf.sr_IMjintpos_IMηintpos] = -self.dJintdηintpos_        # REIMηintpos列
        
        if REIMI0intnegUnknown:
            # 负极交换电流实部REI0intneg行、虚部IMI0intneg行
            ravelKf_[sKf.sr_REi0intneg_REcsnegsurf] = \
            ravelKf_[sKf.sr_IMi0intneg_IMcsnegsurf] = -self.dI0intdθsnegsurf_  # REIMθsnegsurf列
            ravelKf_[sKf.sr_REi0intneg_REceneg] = \
            ravelKf_[sKf.sr_IMi0intneg_IMceneg] = -self.dI0intdθeneg_  # REIMθe列
        if REIMI0intposUnknown:
            # 正极交换电流实部REI0intpos行、虚部IMI0intpos行
            ravelKf_[sKf.sr_REi0intpos_REcspossurf] = \
            ravelKf_[sKf.sr_IMi0intpos_IMcspossurf] = -self.dI0intdθspossurf_  # REIMθsnegsurf列
            ravelKf_[sKf.sr_REi0intpos_REcepos] = \
            ravelKf_[sKf.sr_IMi0intpos_IMcepos] = -self.dI0intdθepos_  # REIMθe列

        # 负极过电位实部REηintneg行REθsnegsurf列、虚部IMηintneg行IMθsnegsurf列
        ravelKf_[sKf.sr_REηintneg_REcsnegsurf] = \
        ravelKf_[sKf.sr_IMηintneg_IMcsnegsurf] = self.dUOCPdθsnegsurf_
        # 正极过电位实部REηintpos行REθspossurf列、虚部IMηintpos行IMθsnegsurf列
        ravelKf_[sKf.sr_REηintpos_REcspossurf] = \
        ravelKf_[sKf.sr_IMηintpos_IMcspossurf] = self.dUOCPdθspossurf_

        if lithiumPlating:
            # 析锂局部体积电流实部REJLP行REθe负极列、虚部IMJLP行IMθe负极列
            ravelKf_[sKf.sr_REjLP_REceneg] = \
            ravelKf_[sKf.sr_IMjLP_IMceneg] = -self.dJLPdθe_
            # 析锂局部体积电流实部REJLP行REηLP列、虚部IMJLP行IMηLP列
            ravelKf_[sKf.sr_REjLP_REηLP] = \
            ravelKf_[sKf.sr_IMjLP_IMηLP] = -self.dJLPdηLP_

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
            ravelKf_REθsnegsurf_REJintneg_[:] = minusKθsnegsurf__[0, 0]  # REJintneg列
            ravelKf_REθsnegsurf_IMJintneg_[:] = minusKθsnegsurf__[0, 1]  # IMJintneg列
            # 负极固相表面浓度虚部IMθsnegsurf行
            ravelKf_IMθsnegsurf_REJintneg_[:] = minusKθsnegsurf__[1, 0]  # REJintneg列
            ravelKf_IMθsnegsurf_IMJintneg_[:] = minusKθsnegsurf__[1, 1]  # IMJintneg列
            # 正极固相表面浓度实部REθspossurf行
            ravelKf_REθspossurf_REJintpos_[:] = minusKθspossurf__[0, 0]  # REJintpos列
            ravelKf_REθspossurf_IMJintpos_[:] = minusKθspossurf__[0, 1]  # IMJintpos列
            # 正极固相表面浓度虚部IMθspossurf行
            ravelKf_IMθspossurf_REJintpos_[:] = minusKθspossurf__[1, 0]  # REJintpos列
            ravelKf_IMθspossurf_IMJintpos_[:] = minusKθspossurf__[1, 1]  # IMJintpos列

            ravelKf_REθe_IMθe_[:] = -ωqeΔx_  # REθe行IMθe列
            ravelKf_IMθe_REθe_[:] = ωqeΔx_   # IMθe行REθe列

            # 负极双电层局部体积电流实部REJDLneg行
            ravelKf_REJDLneg_IMφsneg_[:] = ωCDLneg   # IMφsneg列
            ravelKf_REJDLneg_IMφeneg_[:] = -ωCDLneg  # IMφe负极列
            ravelKf_REJDLneg_IMJintneg_[:] = \
            ravelKf_REJDLneg_IMJDLneg_[:] = -ωCDLRSEIneg  # IMJintneg列、IMJDLneg列
            # 负极双电层局部体积电流虚部IMJDLneg行
            ravelKf_IMJDLneg_REφsneg_[:] = -ωCDLneg  # REφsneg列
            ravelKf_IMJDLneg_REφeneg_[:] = ωCDLneg   # REφe负极列
            ravelKf_IMJDLneg_REJintneg_[:] = \
            ravelKf_IMJDLneg_REJDLneg_[:]  = ωCDLRSEIneg  # REJintneg列、REJDLneg列
            # 正极双电层局部体积电流实部REJDLpos行
            ravelKf_REJDLpos_IMφspos_[:] = ωCDLpos   # IMφspos列
            ravelKf_REJDLpos_IMφepos_[:] = -ωCDLpos  # IMφe正极列
            ravelKf_REJDLpos_IMJintpos_[:] = \
            ravelKf_REJDLpos_IMJDLpos_[:] = -ωCDLRSEIpos  # IMJintpos列、IMJDLpos列
            # 正极双电层局部体积电流虚部IMJDLpos行
            ravelKf_IMJDLpos_REφspos_[:] = -ωCDLpos  # REφspos列
            ravelKf_IMJDLpos_REφepos_[:] = ωCDLpos   # REφe正极列
            ravelKf_IMJDLpos_REJintpos_[:] = \
            ravelKf_IMJDLpos_REJDLpos_[:] = ωCDLRSEIpos  # REJintpos列、REJDLpos列

            if lithiumPlating:
                ravelKf_REJDLneg_IMJLP_[:] = -ωCDLRSEIneg  # REJDLneg行IMJLP列
                ravelKf_IMJDLneg_REJLP_[:] =  ωCDLRSEIneg  # IMJDLneg行REJLP列

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

        self.tEIS = tEIS
        REφsneg__ = X__[:, sKf.s_REφsneg]  # 负极固相电势实部
        IMφsneg__ = X__[:, sKf.s_IMφsneg]  # 负极固相电势虚部
        REφspos__ = X__[:, sKf.s_REφspos]  # 正极固相电势实部
        IMφspos__ = X__[:, sKf.s_IMφspos]  # 正极固相电势虚部

        hΔIAC  = 0.5*ΔIAC
        REφsnegCollector_ = REφsneg__[:, 0]  + hΔIAC*self.Δxneg/self.σneg  # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = IMφsneg__[:, 0]                  # (Nf,) 负极集流体电势虚部 [V]
        REφsposCollector_ = REφspos__[:, -1] - hΔIAC*self.Δxpos/self.σpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = IMφspos__[:, -1]                 # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗虚部 [Ω]
        self.Z_[:] = Zreal_ + 1j*Zimag_ + self.Zl_                    # (Nf,) 全电池复阻抗 [Ω]

        if self.complete:
            self.REθsnegsurf__[:] = X__[:, sKf.s_REcsnegsurf]  # 负极固相表面浓度实部
            self.IMθsnegsurf__[:] = X__[:, sKf.s_IMcsnegsurf]  # 负极固相表面浓度虚部
            self.REθspossurf__[:] = X__[:, sKf.s_REcspossurf]  # 正极固相表面浓度实部
            self.IMθspossurf__[:] = X__[:, sKf.s_IMcspossurf]  # 正极固相表面浓度虚部
            self.REθe__[:] = X__[:, sKf.s_REce]  # 电解液锂离子浓度实部
            self.IMθe__[:] = X__[:, sKf.s_IMce]  # 电解液锂离子浓度虚部
            self.REφsneg__[:] = REφsneg__  # 负极固相电势实部
            self.IMφsneg__[:] = IMφsneg__  # 负极固相电势虚部
            self.REφspos__[:] = REφspos__  # 正极固相电势实部
            self.IMφspos__[:] = IMφspos__  # 正极固相电势虚部
            self.REφe__[:] = REφe__ = X__[:, sKf.s_REφe]   # 电解液电势实部
            self.IMφe__[:] = IMφe__ = X__[:, sKf.s_IMφe]   # 电解液电势虚部
            self.REJintneg__[:] = X__[:, sKf.s_REjintneg]  # 负极局部体积电流实部
            self.IMJintneg__[:] = X__[:, sKf.s_IMjintneg]  # 负极局部体积电流虚部
            self.REJintpos__[:] = X__[:, sKf.s_REjintpos]  # 正极局部体积电流实部
            self.IMJintpos__[:] = X__[:, sKf.s_IMjintpos]  # 正极局部体积电流虚部
            self.REJDLneg__[:] = X__[:, sKf.s_REjDLneg]    # 负极双电层局部体积电流实部
            self.IMJDLneg__[:] = X__[:, sKf.s_IMjDLneg]    # 负极双电层局部体积电流虚部
            self.REJDLpos__[:] = X__[:, sKf.s_REjDLpos]    # 正极双电层局部体积电流实部
            self.IMJDLpos__[:] = X__[:, sKf.s_IMjDLpos]    # 正极双电层局部体积电流虚部
            self.REI0intneg__[:] = X__[:, sKf.s_REi0intneg] if REIMI0intnegUnknown else 0  # 负极交换电流实部
            self.IMI0intneg__[:] = X__[:, sKf.s_IMi0intneg] if REIMI0intnegUnknown else 0  # 负极交换电流虚部
            self.REI0intpos__[:] = X__[:, sKf.s_REi0intpos] if REIMI0intposUnknown else 0  # 正极交换电流实部
            self.IMI0intpos__[:] = X__[:, sKf.s_IMi0intpos] if REIMI0intposUnknown else 0  # 正极交换电流虚部
            self.REηintneg__[:] = X__[:, sKf.s_REηintneg]  # 负极过电位实部
            self.IMηintneg__[:] = X__[:, sKf.s_IMηintneg]  # 负极过电位虚部
            self.REηintpos__[:] = X__[:, sKf.s_REηintpos]  # 正极过电位实部
            self.IMηintpos__[:] = X__[:, sKf.s_IMηintpos]  # 正极过电位虚部
            if lithiumPlating:
                self.REJLP__[:] = X__[:, sKf.s_REjLP]  # 负极析锂局部体积电流密度实部
                self.IMJLP__[:] = X__[:, sKf.s_IMjLP]  # 负极析锂局部体积电流密度虚部
                self.REηLP__[:] = X__[:, sKf.s_REηLP]  # 负极析锂过电位实部
                self.IMηLP__[:] = X__[:, sKf.s_IMηLP]  # 负极析锂过电位虚部

            Nneg, Nsep = self.Nneg, self.Nsep  # 读取：网格数
            Δx_ = self.Δx_
            nW, nE = Nneg - 1, Nneg
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            den = a + b
            REφenegsep_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφenegsep_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφenegsep_ - REφsnegCollector_)/-ΔIAC  # 负极阻抗实部 [Ω]
            Zimag_ = (IMφenegsep_ - IMφsnegCollector_)/-ΔIAC  # 负极阻抗虚部 [Ω]
            self.Zneg_[:] = Zreal_ + 1j*Zimag_  # 负极复阻抗 [Ω]

            nW, nE = Nneg + Nsep - 1, Nneg + Nsep
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            den = a + b
            REφeseppos_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφeseppos_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφsposCollector_ - REφeseppos_)/-ΔIAC  # 正极阻抗实部 [Ω]
            Zimag_ = (IMφsposCollector_ - IMφeseppos_)/-ΔIAC  # 正极阻抗虚部 [Ω]
            self.Zpos_[:] = Zreal_ + 1j*Zimag_  # 正极复阻抗 [Ω]

        if self.verbose:
            print(f'计算时刻t = {tEIS:.1f} s 电化学阻抗谱')
        self.record_EISdata()  # 记录阻抗数据
        return self


    @staticmethod
    def solve_Kθssurf___(
            ω_: ndarray,  # (Nf,) 角频率序列 [rad/s]
            Q: float,     # 电极容量 [Ah]
            Ds: float,    # 集总固相锂离子扩散系数 [1/s]
            ):
        """求Kθssurf___矩阵
        Kθssurf___ @ stack([REJint__, IMJint__], axis=1) = stack([REθssurf__, IMθssurf__], axis=1)"""
        W2_ = ω_/Ds
        W_ = sqrt(W2_)
        root2W_ = 1.4142135623730951*W_
        γ_ = 0.7071067811865476*W_
        cosγ_ = cos(γ_)
        sinγ_ = sin(γ_)
        cosγ2_ = cosγ_*cosγ_
        sinγ2_ = sinγ_*sinγ_
        cosγsinγ_ = cosγ_*sinγ_
        Q6Ds = 21600 * Q * Ds  # [A]

        # 指数缩放
        # cosh²γ 和 coshγ·sinhγ 是 ~exp(2γ) 级别的大数，容易溢出，不能直接算
        # 因此，把 cosh²γ、coshγ·sinhγ 全部乘 exp(-2γ)，转化成O(1)级别，防止溢出
        exp_2γ_ = exp(-2*γ_)
        m_ = 1 + exp_2γ_
        coshγ2_s_ = 0.25*(m_*m_)                    # cosh²γ * exp(-2γ)
        coshγsinhγ_s_ = 0.25*(1 - exp_2γ_*exp_2γ_)  # coshγ·sinhγ * exp(-2γ)

        a_ = -root2W_*(coshγsinhγ_s_ + cosγsinγ_ * exp_2γ_) + 2*(coshγ2_s_ - cosγ2_ * exp_2γ_)
        b_ = -root2W_*(coshγsinhγ_s_ - cosγsinγ_ * exp_2γ_)
        d_ = Q6Ds*((W2_ + 1) * coshγ2_s_
                   - root2W_ * coshγsinhγ_s_
                   - W2_     * sinγ2_    * exp_2γ_
                   - root2W_ * cosγsinγ_ * exp_2γ_
                   -           cosγ2_    * exp_2γ_)
        a_ /= d_
        b_ /= d_
        Kθssurf___ = empty((ω_.size, 2, 2))
        Kθssurf___[:, 0, 0] = Kθssurf___[:, 1, 1] = a_
        Kθssurf___[:, 0, 1] = b_
        Kθssurf___[:, 1, 0] = -b_
        return Kθssurf___  # (Nf, 2, 2)

    @staticmethod
    def solve_REθs_IMθs(
            r: float,     # 径向坐标 [–]
            ω_: ndarray,  # (Nf,) 角频率序列 [rad/s]
            Q: float,     # 电极容量 [Ah]
            Ds: float,    # 集总固相锂离子扩散系数 [1/s]
            REJint__: ndarray,  # (Nf, Nreg) 主反应局部体积电流密度实部 [A]
            IMJint__: ndarray,  # (Nf, Nreg) 主反应局部体积电流密度虚部 [A]
            ):
        """固相浓度实部、虚部在r处的解析解"""
        W2_ = ω_/Ds
        W_ = sqrt(W2_)
        root2 = 1.4142135623730951
        root2W_ = root2*W_
        γ_ = .7071067811865476*W_
        γr_ = γ_*r
        sinγ_ = sin(γ_)
        cosγ_ = cos(γ_)
        sinγr_ = sin(γr_)
        cosγr_ = cos(γr_)
        Q6Dsr = 21600 * Q * Ds * r
        # 指数缩放
        # coshγ coshγr sinhγ sinhγr 是 ~exp(γ) 级别的大数，容易溢出，不能直接算
        # 应缩放：统一乘exp(-γ)
        exp_γ_ = exp(-γ_)
        exp_2γ_ = exp_γ_*exp_γ_
        half_exp_2γ_ = 0.5*exp_2γ_
        coshγ_s_ = 0.5 + half_exp_2γ_  # coshγ*exp(-γ_)
        sinhγ_s_ = 0.5 - half_exp_2γ_  # sinhγ*exp(-γ_)

        exp_γr_ = exp(-γr_)
        q_ = exp(γ_*(r - 1))
        p_ = exp_γ_ * exp_γr_
        coshγr_s_ = 0.5 * (q_ + p_)  # coshγr*exp(-γ_)
        sinhγr_s_ = 0.5 * (q_ - p_)  # sinhγr*exp(-γ_)

        a_ = ( - root2W_ * coshγr_s_ * sinγr_ * coshγ_s_ * cosγ_
               + 2       * coshγr_s_ * sinγr_ * coshγ_s_ * sinγ_
               - root2W_ * coshγr_s_ * sinγr_ * sinhγ_s_ * sinγ_
               - root2W_ * sinhγr_s_ * cosγr_ * coshγ_s_ * cosγ_
               + root2W_ * sinhγr_s_ * cosγr_ * sinhγ_s_ * sinγ_
               + 2       * sinhγr_s_ * cosγr_ * sinhγ_s_ * cosγ_ )

        b_ = (   root2W_ * coshγr_s_ * sinγr_ * coshγ_s_ * cosγ_
               - root2W_ * coshγr_s_ * sinγr_ * sinhγ_s_ * sinγ_
               - 2       * coshγr_s_ * sinγr_ * sinhγ_s_ * cosγ_
               - root2W_ * sinhγr_s_ * cosγr_ * coshγ_s_ * cosγ_
               + 2       * sinhγr_s_ * cosγr_ * coshγ_s_ * sinγ_
               - root2W_ * sinhγr_s_ * cosγr_ * sinhγ_s_ * sinγ_ )

        d_ = Q6Dsr*( (W2_ + 1) * coshγ_s_ * coshγ_s_
                    - root2W_  * coshγ_s_ * sinhγ_s_
                    - W2_      * sinγ_    * sinγ_ * exp_2γ_
                    - root2W_  * cosγ_    * sinγ_ * exp_2γ_
                    -            cosγ_    * cosγ_ * exp_2γ_ )
        a_ /= d_
        b_ /= d_
        Kθs___ = empty((ω_.size, 2, 2))
        Kθs___[:, 0, 0] = Kθs___[:, 1, 1] = a_
        Kθs___[:, 0, 1] = b_
        Kθs___[:, 1, 0] = -b_
        results___ = Kθs___ @ stack([REJint__, IMJint__], axis=1)  # (Nf, 2, Nreg)
        REθs__, IMθs__ = results___[:, 0, :], results___[:, 1, :]
        return REθs__, IMθs__  # (Nf, Nreg)

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
        """正极主反应交换电流密度I0intpos对电极表面嵌锂状态的偏导数 [A/-]"""
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

    plot_REθssurf_IMEθssurf = JTFbase.plot_REcssurf_IMcssurf
    plot_REθe_IMθe = JTFbase.plot_REce_IMce
    plot_REJint_IMJint = JTFbase.plot_REjint_IMjint
    plot_REJDL_IMJDL = JTFbase.plot_REjDL_IMjDL
    plot_REI0int_IMI0int = JTFbase.plot_REi0int_IMi0int

    def checkEIS(self):
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
        IMφsneg__ = self.IMφsneg__      # 负极固相电势虚部
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
            REJLP__ = 0.
            IMJLP__ = 0.
            REηLP__ = REφsneg__ - REφe__[:, :Nneg] - self.RSEIneg*(REJintneg__ + REJDLneg__)
            IMηLP__ = IMφsneg__ - IMφe__[:, :Nneg] - self.RSEIneg*(IMJintneg__ + IMJDLneg__)

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
        F2RT = 0.5*LPP2D.F/LPP2D.R/self.T
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
        REθeInterfaces__ = hstack([REθe__[:, [0]], (REθe__[:, :-1] + REθe__[:, 1:])*0.5, REθe__[:, [-1]]])  # (Nf, Ne+1)
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面锂离子浓度
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
        IMθeInterfaces__ = hstack([IMθe__[:, [0]], (IMθe__[:, :-1] + IMθe__[:, 1:])*0.5, IMθe__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面锂离子浓度
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
        REφeInterfaces__ = hstack([REφe__[:, [0]], (REφe__[:, :-1] + REφe__[:, 1:])*0.5, REφe__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            REφeInterfaces__[:, nE] = ( κ_[nW]*REφe__[:, nW]*Δx_[nE] + κ_[nE]*REφe__[:, nE]*Δx_[nW])/(κ_[nE]*Δx_[nW] + κ_[nW]*Δx_[nE])
        REφeWest__ = REφeInterfaces__[:, :-1]  # 各控制体左界面的电解液电势实部 [V]
        REφeEast__ = REφeInterfaces__[:, 1:]   # 各控制体右界面的电解液电势实部 [V]
        # 各控制体界面的电解液电势虚部 [V]
        IMφeInterfaces__ = hstack([IMφe__[:, [0]], (IMφe__[:, :-1] + IMφe__[:, 1:])*0.5, IMφe__[:, [-1]]])
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
        print(f'固相表面浓度解析解方程 REθssurf IMθssurf 最大误差{maxError: 8e} [–]')

        LHS__ = -outer(ω_, qe_) * IMθe__
        RHS__ = Deκ_*(gradREθeEast__ - gradREθeWest__)/Δx_ + hstack([REJintneg__ + REJDLneg__ + REJLP__, zeros([Nf, Nsep]), REJintpos__ + REJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液实部浓度方程 REθe 最大误差{maxError: 8e} C')

        LHS__ = outer(ω_, qe_) * REθe__
        RHS__ = Deκ_*(gradIMθeEast__ - gradIMθeWest__)/Δx_ + hstack([IMJintneg__ + IMJDLneg__ + IMJLP__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液虚部浓度方程 IMθe 最大误差{maxError: 8e} C')

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
        print(f'负极固相电势方程 REφsneg IMφsneg 最大误差{maxError: 8e} A')

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
        print(f'正极固相电势方程 REφspos IMφspos 最大误差{maxError: 8e} A')

        term1__ = κ_*(gradREφeEast__ - gradREφeWest__)/Δx_
        term2__ = (κDκT_*(gradREθeEast__/θeEast_ - REθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradREθeWest__/θeWest_ - REθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([REJintneg__ + REJDLneg__ + REJLP__ , zeros([Nf, Nsep]), REJintpos__ + REJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势实部方程 REφe 最大误差{maxError: 8e} A')

        term1__ = κ_*(gradIMφeEast__ - gradIMφeWest__)/Δx_
        term2__ = (κDκT_*(gradIMθeEast__/θeEast_ - IMθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradIMθeWest__/θeWest_ - IMθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([IMJintneg__ + IMJDLneg__ + IMJLP__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势虚部方程 IMφe 最大误差{maxError: 8e} A')

        maxError = max([
            abs(REJintneg__ - 2*(REI0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMJintneg__ - 2*(IMI0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REJintpos__ - 2*(REI0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMJintpos__ - 2*(IMI0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(), ])
        print(f'主反应BV动力学方程 REJint IMJint 最大误差{maxError: 8e} A')

        dI0intdθeneg_, dI0intdθepos_ = self.dI0intdθeneg_, self.dI0intdθepos_
        dI0intdθsnegsurf_, dI0intdθspossurf_ = self.dI0intdθsnegsurf_, self.dI0intdθspossurf_
        maxError = max([
            abs(REI0intneg__ - (dI0intdθeneg_*REθe__[:, :Nneg] + dI0intdθsnegsurf_*REθsnegsurf__)).max(),
            abs(IMI0intneg__ - (dI0intdθeneg_*IMθe__[:, :Nneg] + dI0intdθsnegsurf_*IMθsnegsurf__)).max(),
            abs(REI0intpos__ - (dI0intdθepos_*REθe__[:, -Npos:] + dI0intdθspossurf_*REθspossurf__)).max(),
            abs(IMI0intpos__ - (dI0intdθepos_*IMθe__[:, -Npos:] + dI0intdθspossurf_*IMθspossurf__)).max(), ])
        print(f'交换电流密度方程 REI0int IMI0int 最大误差{maxError: 8e} A')

        dUOCPdθsnegsurf_, dUOCPdθspossurf_ = self.dUOCPdθsnegsurf_, self.dUOCPdθspossurf_
        maxError = max([
            abs(REηintneg__ - (REφsneg__ - REφe__[:, :Nneg] - dUOCPdθsnegsurf_*REθsnegsurf__ - RSEIneg*(REJintneg__ + REJDLneg__ + REJLP__))).max(),
            abs(IMηintneg__ - (IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdθsnegsurf_*IMθsnegsurf__ - RSEIneg*(IMJintneg__ + IMJDLneg__ + IMJLP__))).max(),
            abs(REηintpos__ - (REφspos__ - REφe__[:, -Npos:] - dUOCPdθspossurf_*REθspossurf__ - RSEIpos*(REJintpos__ + REJDLpos__))).max(),
            abs(IMηintpos__ - (IMφspos__ - IMφe__[:, -Npos:] - dUOCPdθspossurf_*IMθspossurf__ - RSEIpos*(IMJintpos__ + IMJDLpos__))).max(), ])
        print(f'主反应过电位方程 REηint IMηint 最大误差{maxError: 8e} V')

        if self.lithiumPlating:
            dJLPdθe_, dJLPdηLP_ = self.dJLPdθe_, self.dJLPdηLP_
            maxError = max([
                abs(REJLP__ - (dJLPdθe_*REθe__[:, :Nneg] + dJLPdηLP_*REηLP__) ).max(),
                abs(IMJLP__ - (dJLPdθe_*IMθe__[:, :Nneg] + dJLPdηLP_*IMηLP__) ).max(),])
            print(f'析锂BV动力学方程 REJLP IMJLP 最大误差{maxError: 8e} A')

            maxError = max([
                abs(REηLP__ - (REφsneg__ - REφe__[:, :Nneg] - RSEIneg*(REJintneg__ + REJDLneg__ + REJLP__))).max(),
                abs(IMηLP__ - (IMφsneg__ - IMφe__[:, :Nneg] - RSEIneg*(IMJintneg__ + IMJDLneg__ + IMJLP__))).max(), ])
            print(f'析锂过电位方程 REηLP IMηLP 最大误差{maxError: 8e} V')


if __name__=='__main__':
    import numpy as np
    cell = LPJTFP2D(
        SOC0=0.1,
        # I0intneg=18, I0intpos=22,
        Nneg=8, Nsep=7, Npos=6, Nr=5,
        f_=np.logspace(3, -1, 16),
        lithiumPlating=True,
        # doubleLayerEffect=False,
        # complete=False,
        # constants=True,
        )

    cell.count_lithium()
    cell.EIS()
    thermalModel = 1
    cell.CC(-15, 2000, thermalModel=thermalModel).EIS()
    cell.CC(20, 1000, thermalModel=thermalModel).EIS()
    cell.CC(0, 300, thermalModel=thermalModel).EIS()
    # cell.checkEIS()

    # cell.count_lithium()

    '''
    cell.plot_UI()
    cell.plot_TQgen()
    cell.plot_SOC()
    cell.plot_c(np.arange(0, 2001, 200))
    cell.plot_φ(np.arange(0, 2001, 200))
    cell.plot_Jint_I0int_ηint(np.arange(0, 2001, 200))
    cell.plot_JDL(np.arange(0, 2001, 200))
    cell.plot_csr(np.arange(0, 2001, 200), 1)
    cell.plot_JLP_ηLP(np.arange(1000, 1601, 100))
    cell.plot_LP()
    cell.plot_OCV_OCP()

    cell.plot_Z()
    cell.plot_Nyquist()
    cell.plot_REθssurf_IMEθssurf()
    cell.plot_REθe_IMθe()
    cell.plot_REφs_IMφs()
    cell.plot_REφe_IMφe()
    cell.plot_REJint_IMJint()
    cell.plot_REJDL_IMJDL()
    cell.plot_REI0int_IMI0int()
    cell.plot_REηint_IMηint()
    '''
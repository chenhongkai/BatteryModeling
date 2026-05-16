#%%
import warnings
from typing import Sequence
from functools import partial

import matplotlib.pyplot as plt
from numpy import ndarray, array, zeros, zeros_like, full, empty, hstack, stack, concatenate, \
    exp, sqrt, cos, sin, cosh, sinh, arcsinh, outer, \
    ix_, isnan
from numpy.linalg import solve

from P2Dmodel.P2Dbase import P2Dbase
from P2Dmodel.tools import diagonalSliceRavel, get_color


class DFNJTFP2D(P2Dbase):
    """锂离子电池经典时频联合准二维模型 Doyle-Fuller-Newman Joint-Time Frequency Pseudo-two-Dimensional model"""

    __slots__ = (
        # 专有参数名
        'A', 'Lneg', 'Lsep', 'Lpos', 'εsneg', 'εspos', 'εeneg', 'εesep', 'εepos', 'Rsneg', 'Rspos',
        'bneg', 'bsep', 'bpos',
        '_De', '_κ', 'tplus', 'TDF',
        'csmaxneg', 'csmaxpos', 'ce0',
        'Aeffneg', 'Aeffpos',
        # 专有状态量
        'csneg__', 'cspos__', 'csnegsurf_', 'cspossurf_', 'ce_',
        'jintneg_', 'jintpos_', 'jDLneg_', 'jDLpos_',
        'i0intneg_', 'i0intpos_',
        'jneg_', 'jpos_',
        'jLP_',
        # 专有状态量
        'REcsnegsurf__', 'IMcsnegsurf__', 'REcspossurf__', 'IMcspossurf__', 'REce__', 'IMce__',
        'REjintneg__', 'IMjintneg__', 'REjintpos__', 'IMjintpos__',
        'REjDLneg__', 'IMjDLneg__', 'REjDLpos__', 'IMjDLpos__',
        'REi0intneg__', 'IMi0intneg__', 'REi0intpos__', 'IMi0intpos__',
        'REjLP__', 'IMjLP__',
        )

    def __init__(self,
            A: float = 1.,           # 电极面积 [m^2]
            Lneg: float = 100e-6,    # 负极厚度 [m]
            Lsep: float = 52e-6,     # 隔膜厚度 [m]
            Lpos: float = 174e-6,    # 正极厚度 [m]
            εsneg: float = .471,     # 负极区域固相体积分数 [–]
            εspos: float = .297,     # 正极区域固相体积分数 [–]
            εeneg: float = .503,     # 负极区域电解液体积分数 [–]
            εesep: float = .7,       # 隔膜区域电解液体积分数 [–]
            εepos: float = .630,     # 正极区域电解液体积分数 [–]
            Rsneg: float = 12.5e-6,  # 负极球形固相颗粒半径 [m]
            Rspos: float = 8.0e-6,   # 正极球形固相颗粒半径 [m]
            bneg: float = 3.3,       # 负极Bruggman指数 [–]
            bsep: float = 3.3,       # 隔膜Bruggman指数 [–]
            bpos: float = 3.3,       # 正极Bruggman指数 [–]
            Dsneg: float = 3.9e-14,  # 负极固相的锂离子扩散系数 [m^2/s]
            Dspos: float = 2.5e-14,  # 正极固相的锂离子扩散系数 [m^2/s]
            De: float = 7.5e-11,     # 电解液扩散系数 [m^2/s]
            σneg: float = 100.,      # 负极固相电导率 [S/m]
            σpos: float = 3.8,       # 正极固相电导率 [S/m]
            κ: float = .2,           # 电解液离子电导率 [S/m]
            tplus: float = 0.363,    # 电解液阳离子迁移数 [–]
            TDF: float = 1.,         # 电解液活度系数项1+∂lnf/∂lnce
            kneg: float = 1.948e-11,    # 负极主反应速率常数 [m^2.5/(mol^0.5·s)]
            kpos: float = 2.156e-11,    # 正极主反应速率常数 [m^2.5/(mol^0.5·s)]
            kLP: float = 2.23e-7,       # 负极析锂反应速率常数 2017GeHao 2.23e-7; 2019Christian2.5e-7; 2020Simon1e-10~2e-8
            RSEIneg: float = 5e-3,      # 负极SEI膜的面积电阻 [Ω·m^2]
            RSEIpos: float = 2e-3,      # 正极SEI膜的面积电阻 [Ω·m^2]
            csmaxneg: float = 26390.,   # 负极固相最大锂离子浓度 [mol/m^3]
            csmaxpos: float = 22860.,   # 正极固相最大锂离子浓度 [mol/m^3]
            ce0: float = 2000.,         # 电解液的初始浓度 [mol/m^3]
            CDLneg: float = .8,         # 负极颗粒表面的双电层面积电容 [F/m^2]
            CDLpos: float = .8,         # 正极颗粒表面的双电层面积电容 [F/m^2]
            l: float = 0.,              # 等效电感 [H]
            i0intneg: float | None = None,  # 负极主反应交换电流密度 [A/m^2]
            i0intpos: float | None = None,  # 正极主反应交换电流密度 [A/m^2]
            i0LP: float | None = None,  # 负极析锂反应交换电流密度 [A/m^2]
            Aeffneg: float = 1.,   # 负极活性材料与电解质的有效接触面积比 [–]
            Aeffpos: float = 1.,   # 正极活性材料与电解质的有效接触面积比 [–]
            θminneg: float = .0370744,  # SOC=0%的负极嵌锂状态 [–]
            θmaxneg: float = .8775600,  # SOC=100%的负极嵌锂状态 [–]
            θminpos: float = .0746557,  # SOC=100%的正极嵌锂状态 [–]
            θmaxpos: float = .9589741,  # SOC=0%的正极嵌锂状态 [–]
            SOC0: float = 0.2,          # 初始荷电状态 [–]
            **kwargs,
            ):
        # 11几何参数
        self.A = A; assert A>0, f'电极面积{A = }，应大于0 [m^2]'
        self.Lneg = Lneg; assert Lneg>0, f'负极厚度{Lneg = }，应大于0 [m]'
        self.Lsep = Lsep; assert Lsep>0, f'隔膜厚度{Lsep = }，应大于0 [m]'
        self.Lpos = Lpos; assert Lpos>0, f'正极厚度{Lpos = }，应大于0 [m]'
        self.εsneg = εsneg; assert 0<εsneg<1, f'负极固相体积分数{εsneg = }，取值范围应为(0, 1)'
        self.εspos = εspos; assert 0<εspos<1, f'正极固相体积分数{εspos = }，取值范围应为(0, 1)'
        self.εeneg = εeneg; assert 0<εeneg<1,  f'负极电解液体积分数{εeneg = }，取值范围应为(0, 1)'
        self.εesep = εesep; assert 0<εesep<=1, f'隔膜电解液体积分数{εesep = }，取值范围应为(0, 1]'
        self.εepos = εepos; assert 0<εepos<1,  f'正极电解液体积分数{εepos = }，取值范围应为(0, 1)'
        self.Rsneg = Rsneg; assert Rsneg>0, f'负极球形固相颗粒半径{Rsneg = }，应大于0 [m]'
        self.Rspos = Rspos; assert Rspos>0, f'正极球形固相颗粒半径{Rspos = }，应大于0 [m]'
        # 13输运参数
        self.bneg = bneg; assert bneg>=1, f'负极Bruggman指数{bneg = }，应大于或等于1'
        self.bsep = bsep; assert bsep>=1, f'隔膜Bruggman指数{bsep = }，应大于或等于1'
        self.bpos = bpos; assert bpos>=1, f'正极Bruggman指数{bpos = }，应大于或等于1'
        self.Dsneg = Dsneg; assert Dsneg>0, f'负极固相的锂离子扩散系数{Dsneg = }，应大于0 [m^2/s]'
        self.Dspos = Dspos; assert Dspos>0, f'正极固相的锂离子扩散系数{Dspos = }，应大于0 [m^2/s]'
        self.De = De;       assert De>0, f'电解液扩散系数{De = }，应大于0 [m^2/s]'
        self.σneg = σneg;   assert σneg>0, f'负极固相电导率{σneg = }，应大于0 [S/m]'
        self.σpos = σpos;   assert σpos>0, f'正极固相电导率{σpos = }，应大于0 [S/m]'
        self.κ = κ;         assert κ>0, f'电解液离子电导率{κ = }，应大于0 [S/m]'
        self.tplus = tplus; assert tplus>0, f'电解液迁移数{tplus = }，应大于0'
        self.TDF = TDF;     assert TDF>0, f'热力学因子1 + ∂lnf/∂lnce = {TDF = }，应大于0'
        self.RSEIneg = RSEIneg; assert RSEIneg>=0, f'负极SEI膜的面积电阻{RSEIneg = }，应大于或等于0 [Ω·m^2]'
        self.RSEIpos = RSEIpos; assert RSEIpos>=0, f'正极SEI膜的面积电阻{RSEIpos = }，应大于或等于0 [Ω·m^2]'
        # 5动力学参数
        self.kneg = kneg; assert kneg>0, f'负极主反应速率常数{kneg = }，应大于0 m^2.5/(mol^0.5·s)'
        self.kpos = kpos; assert kpos>0, f'正极主反应速率常数{kpos = }，应大于0 m^2.5/(mol^0.5·s)'
        self.kLP = kLP; assert kLP>0,  f'负极析锂反应速率常数{kLP = }，应大于0'
        # 3浓度参数
        self.csmaxneg = csmaxneg; assert csmaxneg>0, f'负极固相最大锂离子浓度{csmaxneg = }，应大于0 [mol/m^3]'
        self.csmaxpos = csmaxpos; assert csmaxpos>0, f'正极固相最大锂离子浓度{csmaxpos = }，应大于0 [mol/m^3]'
        self.ce0 = ce0; assert ce0>0, f'电解液的初始浓度{ce0 = }，应大于0 [mol/m^3]'
        # 3电抗参数
        self.CDLneg = CDLneg; assert CDLneg>=0, f'负极表面双电层面积电容{CDLneg = }，应大于或等于0 [F/m^2]'
        self.CDLpos = CDLpos; assert CDLpos>=0, f'正极表面双电层面积电容{CDLpos = }，应大于或等于0 [F/m^2]'
        self.l = l; assert l>=0, f'等效电感{l = }，应大于或等于0 [H]'
        # 3交换电流密度
        self.i0intneg = i0intneg; assert (i0intneg is None) or (i0intneg>0), f'负极主反应交换电流密度{i0intneg = }，应大于0 [A/m^2]'
        self.i0intpos = i0intpos; assert (i0intpos is None) or (i0intpos>0), f'正极主反应交换电流密度{i0intpos = }，应大于0 [A/m^2]'
        self.i0LP = i0LP;         assert (i0LP is None) or (i0LP>0), f'负极析锂反应交换电流密度{i0LP = }，应大于0 [A/m^2]'
        # 2全固态电池参数
        self.Aeffneg = Aeffneg; assert 0<Aeffneg<=1, f'负极固相与电解质的有效接触面积比{Aeffneg = }，取值范围应为(0, 1]'
        self.Aeffpos = Aeffpos; assert 0<Aeffpos<=1, f'正极固相与电解质的有效接触面积比{Aeffpos = }，取值范围应为(0, 1]'
        # P2D通用参数
        P2Dbase.__init__(self,
                         Lneg=Lneg, Lsep=Lsep, Lpos=Lpos,
                         Rsneg=Rsneg, Rspos=Rspos,
                         SOC0=SOC0,
                         θminneg=θminneg, θmaxneg=θmaxneg,
                         θminpos=θminpos, θmaxpos=θmaxpos, **kwargs)
        if (verbose := self.verbose) and (εsneg + εeneg)>1:
            warnings.warn(f'负极固相体积分数εsneg与负极电解液体积分数εeneg之和大于1，{εsneg + εeneg = } > 1')
        if verbose and (εspos + εepos)>1:
            warnings.warn(f'正极固相体积分数εspos与正极电解液体积分数εepos之和大于1，{εspos + εepos = } > 1')
        Nneg, Npos, Ne, Nr = self.Nneg, self.Npos, self.Ne, self.Nr  # 读取：网格数
        # DFNP2D专有状态量
        csneg = csmaxneg*(θminneg + SOC0*(θmaxneg - θminneg))  # 初始负极固相锂离子浓度 [mol/m^3]
        cspos = csmaxpos*(θmaxpos + SOC0*(θminpos - θmaxpos))  # 初始正极固相锂离子浓度 [mol/m^3]
        self.csneg__ = full((Nr, Nneg), csneg)  # 初始化：负极固相颗粒锂离子浓度场 [mol/m^3]
        self.cspos__ = full((Nr, Npos), cspos)  # 初始化：正极固相颗粒锂离子浓度场 [mol/m^3]
        self.csnegsurf_ = full(Nneg, csneg)     # 初始化：负极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.cspossurf_ = full(Npos, cspos)     # 初始化：正极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.ce_ = full(Ne, ce0)           # 初始化：电解液锂离子浓度场 [mol/m^3]
        self.jintneg_, self.jintpos_ = zeros(Nneg), zeros(Npos)  # 初始化：负极、正极主反应局部体积电流密度场 [A/m^3]
        self.jDLneg_, self.jDLpos_   = zeros(Nneg), zeros(Npos)  # 初始化：负极、正极双电层效应局部体积电流密度场 [A/m^3]
        i0intneg = self.i0intneg if self._i0intneg else DFNJTFP2D.solve_i0int_(self.kneg, csmaxneg, csneg, ce0)
        i0intpos = self.i0intpos if self._i0intpos else DFNJTFP2D.solve_i0int_(self.kpos, csmaxpos, cspos, ce0)
        self.i0intneg_ = full(Nneg, i0intneg)
        self.i0intpos_ = full(Npos, i0intpos)  # 初始化：负极、正极主反应交换电流密度场 [A/m^2]
        self.jneg_ = zeros(Nneg)
        self.jpos_ = zeros(Npos)  # 初始化：负极、正极总局部体积电流密度 [A/m^3]
        if lithiumPlating := self.lithiumPlating:
            self.jLP_  = zeros(Nneg)  # 初始化：负极析锂局部体积电流密度场 [A/m^3]
        if self.complete:
            # 状态量
            Nf = self.f_.size
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

            extra_datanames_ = [             # 需记录的数据名称
                'csneg__', 'cspos__',        # 负极、正极固相锂离子浓度场 [mol/m^3]
                'csnegsurf_', 'cspossurf_',  # 负极、正极表面锂离子浓度场 [mol/m^3]
                'ce_',                       # 电解液锂离子浓度场 [mol/m^3]
                'jintneg_', 'jintpos_',      # 负极、正极主反应局部体积电流密度场 [A/m^3]
                'jDLpos_', 'jDLneg_',        # 负极、正极双电层效应局部体积电流密度场 [A/m^3]
                'i0intneg_', 'i0intpos_',    # 负极、正极主反应交换电流密度场 [A/m^2]
                'isneg_', 'ispos_', 'ie_',]  # 负极、正极固相电流密度场、电解液电流密度场 [A/m^2]
            if lithiumPlating:
                extra_datanames_.append('jLP_')  # 负极析锂局部体积电流密度场 [A/m^3]
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
            self.datanames_.extend(extra_datanames_)
            self.EISdatanames_.extend(extra_EISdatanames_)
            self.data.update({name: [] for name in (extra_datanames_ + extra_EISdatanames_)})

        if verbose and type(self) is DFNJTFP2D:
            print(self)
            print('经典时频联合P2D模型(DFNJTFP2D)初始化完成!')

    def update_K__with_pure_electrochemical_parameters(self):
        # 对K__矩阵赋纯电化学参数相关值
        if self.decouple:
            pass
        else:
            self._update_K__csnegsurf_jintneg_when_coupling(self.aneg, self.Dsneg)
            self._update_K__cspossurf_jintpos_when_coupling(self.apos, self.Dspos)
        self._update_K__φsneg_jneg(self.σeffneg)
        self._update_K__φspos_jpos(self.σeffpos)
        self._update_K__φe_φe(κeff_ := self.κeff_, κeff_)
        self._update_K__ηintneg_jneg(self.RSEIneg, self.aeffneg)
        self._update_K__ηintpos_jpos(self.RSEIpos, self.aeffpos)
        if self.lithiumPlating:
            self._update_K__ηLP_jneg(self.RSEIneg, self.aeffneg)

    def _update_K__bK_csnegsurf_jintneg_when_decoupling(self, Dsneg, aneg, Δt, old_csneg__, old_jintneg_):
        # 更新K__矩阵csnegsurf行jintneg列
        # 更新bK_向量csnegsurf行
        Kcsjintneg = DFNJTFP2D._solve_Kcsjint(aneg, Δt, self.Rsneg, self.Δrneg_)
        csnegI__, γneg_ = P2Dbase._update_K__bK_csnegsurf_jintneg_when_decoupling(
            self, Dsneg, Kcsjintneg, Δt, old_csneg__, old_jintneg_)
        return csnegI__, γneg_

    def _update_K__bK_cspossurf_jintpos_when_decoupling(self, Dspos, apos, Δt, old_cspos__, old_jintpos_):
        # 更新K__矩阵cspossurf行jintpos列
        # 更新bK_向量cspossurf行
        Kcsjintpos = DFNJTFP2D._solve_Kcsjint(apos, Δt, self.Rspos, self.Δrpos_)
        csposI__, γpos_ = P2Dbase._update_K__bK_cspossurf_jintpos_when_decoupling(
            self, Dspos, Kcsjintpos, Δt, old_cspos__, old_jintpos_)
        return csposI__, γpos_

    def _update_K__bK_csneg_csneg_jintneg_when_coupling(self, Dsneg, aneg, Δt, old_csneg__, old_jintneg_):
        # 更新K__矩阵csneg行csneg列
        # 更新K__矩阵csneg末尾球壳控制体行jintneg列
        # 更新bK_向量csneg行
        Kcsjintneg = DFNJTFP2D._solve_Kcsjint(aneg, Δt, self.Rsneg, self.Δrneg_)
        P2Dbase._update_K__bK_csneg_csneg_jintneg_when_coupling(
            self, Dsneg, Δt, Kcsjintneg, old_csneg__, old_jintneg_)

    def _update_K__bK_cspos_cspos_jintpos_when_coupling(self, Dspos, apos, Δt, old_cspos__, old_jintpos_):
        # 更新K__矩阵cspos行cspos列
        # 更新K__矩阵cspos末尾球壳控制体行jintpos列
        # 更新bK_向量cspos行
        Kcsjintpos = DFNJTFP2D._solve_Kcsjint(apos, Δt, self.Rspos, self.Δrpos_)
        P2Dbase._update_K__bK_cspos_cspos_jintpos_when_coupling(
            self, Dspos, Δt, Kcsjintpos, old_cspos__, old_jintpos_)

    @staticmethod
    def _solve_Kcsjint(areg, Δt, Rsreg, Δr_):
        Rsreg2 = Rsreg*Rsreg
        Δ = Rsreg - Δr_[-1]
        return Δt*Rsreg2/areg/P2Dbase.F/((Rsreg2*Rsreg - Δ*Δ*Δ)/3)

    def _update_K__csnegsurf_jintneg_when_coupling(self, aneg, Dsneg):
        # 更新K__矩阵csnegsurf行jintneg列
        self.ravelK_[self.sK.sr_csnegsurf_jintneg] = 1/(aneg*P2Dbase.F*Dsneg)

    def _update_K__cspossurf_jintpos_when_coupling(self, apos, Dspos):
        # 更新K__矩阵cspossurf行jintpos列
        self.ravelK_[self.sK.sr_cspossurf_jintpos] = 1/(apos*P2Dbase.F*Dspos)

    def _update_K__bK_ce_ce_j(self,
            DeeffWest_: ndarray, DeeffEast_: ndarray,
            εe_: ndarray, tplus: float | None,
            Δt: float,
            old_ce_, old_jneg_, old_jpos_,
            ):
        # 更新K__矩阵ce行ce列
        # 更新K__矩阵ce行j列
        # 更新bK_向量ce行
        Kcej = -Δt*(1 - tplus) / P2Dbase.F
        P2Dbase._update_K__bK_ce_ce_j(self,
            DeeffWest_, DeeffEast_, εe_, Kcej, Δt, old_ce_, old_jneg_, old_jpos_)

    def _update_bK_φsneg0_φsposEnd(self, σeffneg, σeffpos, I):
        # 更新bK_向量φsneg行首元、φspos行末元
        i = I/self.A
        bK_ = self.bK_
        sK = self.sK
        bK_[sK.s_φsneg.start]    = -self.Δxneg*i/σeffneg
        bK_[sK.s_φspos.stop - 1] =  self.Δxpos*i/σeffpos

    def _stepping(self, Δt):
        """时间步进：Newton法迭代因变量"""
        # 读取模式
        lithiumPlating = self.lithiumPlating
        timeDiscretization = self.timeDiscretization
        decouple = self.decouple
        doubleLayerEffect = self.doubleLayerEffect
        verbose = self.verbose

        # 读取索引
        sK = self.sK
        s_φe = sK.s_φe
        s_jintneg = sK.s_jintneg
        s_jintpos = sK.s_jintpos
        s_i0intneg = sK.s_i0intneg
        s_i0intpos = sK.s_i0intpos
        s_ηintneg = sK.s_ηintneg
        s_ηintpos = sK.s_ηintpos
        s_c = sK.s_c
        s_φ = sK.s_φ
        s_j = sK.s_j
        if lithiumPlating:
            s_jLP = sK.s_jLP

        # 读取方法
        solve_banded_matrix = DFNJTFP2D.solve_banded_matrix
        solve_jint_ = DFNJTFP2D.solve_jint_
        solve_djintdηint_ = DFNJTFP2D.solve_djintdηint_
        solve_djintdi0int_ = DFNJTFP2D.solve_djintdi0int_
        solve_i0int_ = DFNJTFP2D.solve_i0int_
        solve_di0intdcssurf_ = DFNJTFP2D.solve_di0intdcssurf_
        solve_di0intdce_ = DFNJTFP2D.solve_di0intdce_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_
        solve_dUOCPdθsneg_ = self.solve_dUOCPdθsneg_
        solve_dUOCPdθspos_ = self.solve_dUOCPdθspos_
        if lithiumPlating:
            solve_jLP_      = DFNJTFP2D.solve_jLP_
            solve_djLPdce_  = DFNJTFP2D.solve_djLPdce_
            solve_djLPdηLP_ = DFNJTFP2D.solve_djLPdηLP_
            solve_i0LP_     = DFNJTFP2D.solve_i0LP_

        # 读取参数
        Nneg, Nsep, Npos, Nr = self.Nneg, self.Nsep, self.Npos, self.Nr  # 读取：网格数
        Δx_ = self.Δx_                    # 读取：网格尺寸 [m]
        ΔxWest_, ΔxEast_ = self.ΔxWest_, self.ΔxEast_  # 读取：网格距离 [m]
        aneg, apos = self.aneg, self.apos              # 读取：固相颗粒的比表面积 [m^2/m^3]
        aeffneg, aeffpos = self.aeffneg, self.aeffpos  # 读取：负极、正极材料与电解质的有效比表面积
        csmaxneg, csmaxpos = self.csmaxneg, self.csmaxpos  # 读取：固相最大锂离子浓度 [mol/m^3]
        RSEIneg, RSEIpos = self.RSEIneg, self.RSEIpos  # 读取：负极、正极SEI膜面积电阻 [Ω·m^2]
        Dsneg, Dspos = self.Dsneg, self.Dspos          # 读取：负极、正极固相扩散系数 [m^2/s]
        DeeffWest_ = DeeffEast_ = self.Deeff_
        κeffWest_ = κeffEast_ = self.κeff_
        κDeffWest_ = κDeffEast_ = self.κDeff_
        RSEI2aeffneg = RSEIneg/aeffneg
        RSEI2aeffpos = RSEIpos/aeffpos
        if i0intnegUnknown := (self._i0intneg is None):
            kneg = self.kneg          # 读取：负极主反应速率常数
        else:
            i0intneg = self.i0intneg  # 读取：负极主反应交换电流密度 [A/m^2]
        if i0intposUnknown := (self._i0intpos is None):
            kpos = self.kpos          # 读取：正极主反应速率常数
        else:
            i0intpos = self.i0intpos  # 读取：正极主反应交换电流密度 [A/m^2]
        if lithiumPlating:
            if i0LPUnknown := (self._i0LP is None):
                kLP = self.kLP    # 读取：负极析锂反应速率常数
            else:
                i0LP = self.i0LP  # 读取：负极析锂反应交换电流密度 [A/m^2]

        # 读取状态
        i = (I := self.I)/self.A  # 电流密度 [A/m^2]
        T = self.T                # 温度 [K]
        F2RT = 0.5*P2Dbase.F/(P2Dbase.R*T)  # 常数 [1/V]
        data = self.data  # 运行数据字典

        ravelK_ = self.ravelK_  # 因变量线性矩阵K__展平视图
        bK_ = self.bK_          # 常数项向量，F_ = K__ @ X_ - bK_
        K__ = ravelK_.base

        if decouple:
            # 固相表面浓度cssurf行jint列
            # 历史固相浓度影响分量csI__、系数向量γ_
            csnegI__, γneg_ = self._update_K__bK_csnegsurf_jintneg_when_decoupling(Dsneg, aneg, Δt, self.csneg__, self.jintneg_)
            csposI__, γpos_ = self._update_K__bK_cspossurf_jintpos_when_decoupling(Dspos, apos, Δt, self.cspos__, self.jintpos_)
        else:
            # 固相内部浓度cs行cs行、cs末尾球壳控制体行jint列
            self._update_K__bK_csneg_csneg_jintneg_when_coupling(Dsneg, aneg, Δt, self.csneg__, self.jintneg_)
            self._update_K__bK_cspos_cspos_jintpos_when_coupling(Dspos, apos, Δt, self.cspos__, self.jintpos_)

        # 电解液浓度ce行ce列、ce行j列
        self._update_K__bK_ce_ce_j(
            DeeffWest_, DeeffEast_, self.εe_,
            self.tplus, Δt, self.ce_, self.jneg_, self.jpos_)

        # 固相电流边界条件
        self._update_bK_φsneg0_φsposEnd(self.σeffneg, self.σeffpos, I)

        # 双电层局部体积电流密度jDLneg、jDLpos行
        if doubleLayerEffect:
            self._update_K__bK_jDL_φs_φe_j(
                aeffneg, aeffpos, RSEIneg, RSEIpos,
                self.CDLneg,  self.CDLpos, Δt)

        # 索引解
        X_ = zeros_like(bK_)
        if decouple:
            pass
        else:
            csneg_ = X_[sK.s_csneg]
            cspos_ = X_[sK.s_cspos]
        csnegsurf_ = X_[sK.s_csnegsurf]
        cspossurf_ = X_[sK.s_cspossurf]
        ce_    = X_[sK.s_ce]
        ceneg_ = X_[sK.s_ceneg]
        cepos_ = X_[sK.s_cepos]
        φsneg_ = X_[sK.s_φsneg]
        φspos_ = X_[sK.s_φspos]
        φe_    = X_[s_φe]
        jintneg_ = X_[s_jintneg]
        jintpos_ = X_[s_jintpos]
        if doubleLayerEffect:
            jDLneg_ = X_[sK.s_jDLneg]
            jDLpos_ = X_[sK.s_jDLpos]
        i0intneg_ = X_[s_i0intneg] if i0intnegUnknown else i0intneg
        i0intpos_ = X_[s_i0intpos] if i0intposUnknown else i0intpos
        ηintneg_ = X_[s_ηintneg]
        ηintpos_ = X_[s_ηintpos]
        if lithiumPlating:
            φeneg_ = X_[sK.s_φeneg]
            jLP_ = X_[s_jLP]
            ηLP_ = X_[sK.s_ηLP]

        # 解向量赋初值
        if decouple:
            pass
        else:
            csneg_[:] = self.csneg__.ravel('F')
            cspos_[:] = self.cspos__.ravel('F')
        csnegsurf_[:] = self.csnegsurf_
        cspossurf_[:] = self.cspossurf_
        ce_[:] = self.ce_
        if i0intnegUnknown:
            i0intneg_[:] = self.i0intneg_
        if i0intposUnknown:
            i0intpos_[:] = self.i0intpos_
        if I==data['I'][-1]:
            # 恒电流
            φsneg_[:] = self.φsneg_
            φspos_[:] = self.φspos_
            φe_[:] = self.φe_
            jintneg_[:] = self.jintneg_
            jintpos_[:] = self.jintpos_
            ηintneg_[:] = self.ηintneg_
            ηintpos_[:] = self.ηintpos_
        else:
            # 变电流瞬间
            jintneg_[:] = jintneg =  i/self.Lneg
            jintpos_[:] = jintpos = -i/self.Lpos
            ηintneg_[:] = arcsinh(jintneg/(2 * aeffneg * i0intneg_))/F2RT
            ηintpos_[:] = arcsinh(jintpos/(2 * aeffpos * i0intpos_))/F2RT
            φsneg_[:] = ηintneg_ + RSEI2aeffneg*jintneg + solve_UOCPneg_(csnegsurf_/csmaxneg)
            φspos_[:] = ηintpos_ + RSEI2aeffpos*jintpos + solve_UOCPpos_(cspossurf_/csmaxpos)
        if lithiumPlating:
            ηLP_[:] = φsneg_ - φeneg_ - RSEI2aeffneg*jintneg_

        # 初始化Jacobi矩阵
        J__ = K__.copy()
        ravelJ_ = J__.ravel()    # (NK*NK,) Jacobi矩阵展平视图
        ravelJ_φe_ce_l_ = ravelJ_[sK.sr_φe_ce_l]
        ravelJ_φe_ce_u_ = ravelJ_[sK.sr_φe_ce_u]
        ravelJ_φe_ce_   = ravelJ_[sr_φe_ce := sK.sr_φe_ce]
        start0φece = sr_φe_ce.start
        NJ1 = bK_.size + 1
        ravelJ_jintneg_ηintneg_  = ravelJ_[sK.sr_jintneg_ηintneg]
        ravelJ_jintpos_ηintpos_ = ravelJ_[sK.sr_jintpos_ηintpos]
        ravelJ_ηintneg_csnegsurf_ = ravelJ_[sK.sr_ηintneg_csnegsurf]
        ravelJ_ηintpos_cspossurf_ = ravelJ_[sK.sr_ηintpos_cspossurf]
        if i0intnegUnknown:
            ravelJ_jintneg_i0intneg_ = ravelJ_[sK.sr_jintneg_i0intneg]
            ravelJ_i0intneg_csnegsurf_ = ravelJ_[sK.sr_i0intneg_csnegsurf]
            ravelJ_i0intneg_ceneg_ = ravelJ_[sK.sr_i0intneg_ceneg]
        if i0intposUnknown:
            ravelJ_jintpos_i0intpos_ = ravelJ_[sK.sr_jintpos_i0intpos]
            ravelJ_i0intpos_cspossurf_ = ravelJ_[sK.sr_i0intpos_cspossurf]
            ravelJ_i0intpos_cepos_     = ravelJ_[sK.sr_i0intpos_cepos]
        if lithiumPlating:
            ravelJ_jLP_ceneg_ = ravelJ_[sK.sr_jLP_ceneg]
            ravelJ_jLP_ηLP_ = ravelJ_[sK.sr_jLP_ηLP]

        # 预计算
        κDeff2ΔxWest_ = κDeffWest_[1:]  / ΔxWest_[1:]   # (Ne-1,)
        κDeff2ΔxEast_ = κDeffEast_[:-1] / ΔxEast_[:-1]  # (Ne-1,)

        for nNewton in range(1, 201):
            ## Newton迭代
            F_ = K__.dot(X_) - bK_  # F残差向量的线性部分

            # F向量非线性部分
            ceM_ = 0.5*(ce_[1:] + ce_[:-1])   # (Ne-1,) 相邻浓度均值
            q_ = (ce_[1:] - ce_[:-1]) / ceM_  # (Ne-1,)
            a_ = κDeff2ΔxWest_ * q_   # (Ne-1,)
            c_ = κDeff2ΔxEast_ * q_   # (Ne-1,)
            ΔFφe_ = hstack([0, a_]) - hstack([c_, 0]) # (Ne,)
            for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
                # 修正负极-隔膜界面、隔膜-正极界面
                a, b = DeeffWest_[nE]*Δx_[nW], DeeffEast_[nW]*Δx_[nE]
                cInterface = (a*ce_[nE] + b*ce_[nW])/(a + b)
                ΔFφe_[nW] = (κDeffWest_[nW]  * (ce_[nW] - ce_[nW-1])  / ΔxWest_[nW]   / (0.5*(ce_[nW] + ce_[nW-1]))
                            -κDeffEast_[nW]  * (cInterface - ce_[nW]) / (0.5*Δx_[nW]) / cInterface)
                ΔFφe_[nE] = ( κDeffWest_[nE] * (ce_[nE] - cInterface) / (0.5*Δx_[nE]) / cInterface
                             -κDeffEast_[nE] * (ce_[nE+1] - ce_[nE] ) / ΔxEast_[nE]   / (0.5*(ce_[nE+1] + ce_[nE])) )
            F_[s_φe] += ΔFφe_
            F_[s_jintneg] -= solve_jint_(T, aeffneg, i0intneg_, ηintneg_) # F向量jintneg部分
            F_[s_jintpos] -= solve_jint_(T, aeffpos, i0intpos_, ηintpos_)  # F向量jintpos部分
            if i0intnegUnknown:
                F_[s_i0intneg] -= solve_i0int_(kneg, csmaxneg, csnegsurf_, ceneg_) # F向量i0intneg部分
            if i0intposUnknown:
                F_[s_i0intpos] -= solve_i0int_(kpos, csmaxpos, cspossurf_, cepos_) # F向量i0intpos部分
            F_[s_ηintneg] += solve_UOCPneg_(csnegsurf_/csmaxneg)  # F向量ηintneg非线性部分
            F_[s_ηintpos] += solve_UOCPpos_(cspossurf_/csmaxpos)  # F向量ηintpos非线性部分
            if lithiumPlating:
                i0LP_ = solve_i0LP_(kLP, ceneg_) if i0LPUnknown else i0LP  # 负极析锂反应的交换电流密度场 [A/m^2]
                F_[s_jLP] -= solve_jLP_(T, aeffneg, i0LP_, ηLP_)  # F向量jLP部分

            # 更新Jacobi矩阵非线性部分
            # φe行ce列
            q_ *= 0.5
            a_ = κDeff2ΔxWest_ / ceM_
            aa_ = a_ * q_
            c_ = κDeff2ΔxEast_ / ceM_
            cc_ = c_ * q_
            ravelJ_φe_ce_l_[:] = -aa_ - a_  # 下对角线
            ravelJ_φe_ce_u_[:] = cc_ - c_   # 上对角线
            ravelJ_φe_ce_[:]   = hstack([0, a_ - aa_]) + hstack([cc_ + c_, 0])  # 主对角线
            for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
                # 修正负极-隔膜界面、隔膜-正极界面
                nrW = start0φece + nW*NJ1
                nrE = start0φece + nE*NJ1
                κDeffEast_nW_DeeffWest_nE = κDeffEast_[nW]*DeeffWest_[nE]
                κDeffWest_nE_DeeffEast_nW = κDeffWest_[nE]*DeeffEast_[nW]
                num  = κDeffEast_nW_DeeffWest_nE - κDeffWest_nE_DeeffEast_nW
                κeffWest_nE_Δx_nW = κeffWest_[nE] * Δx_[nW]
                κeffEast_nW_Δx_nE = κeffEast_[nW] * Δx_[nE]
                DeeffWest_nE_Δx_nW = DeeffWest_[nE]*Δx_[nW]
                DeeffEast_nW_Δx_nE = DeeffEast_[nW]*Δx_[nE]
                den1 = κeffWest_nE_Δx_nW + κeffEast_nW_Δx_nE
                den2 = DeeffWest_nE_Δx_nW * ce_[nE] + DeeffEast_nW_Δx_nE * ce_[nW]
                quotient = num / (den1*den2)
                ΔceEW = ce_[nE] - ce_[nW]
                SceW  = ce_[nW] + ce_[nW-1]
                SceE  = ce_[nE] + ce_[nE+1]

                coeff = ΔceEW * DeeffWest_nE_Δx_nW / den2
                a  = 2 * κDeffWest_[nW] / (SceW * Δx_[nW])
                aa = a * (ce_[nW] - ce_[nW-1]) / SceW
                c  = 2 * κeffEast_nW_Δx_nE * quotient
                cc = c * coeff
                d  = 2 * κDeffEast_nW_DeeffWest_nE  / den2
                dd = d * coeff
                p  = DeeffEast_nW_Δx_nE / DeeffWest_nE_Δx_nW
                ravelJ_[nrW-1:nrW+2] = -a - aa, -c - cc*p + d + dd*p + a - aa, c - cc - d + dd  # 界面左侧控制体

                coeff = ΔceEW * DeeffEast_nW_Δx_nE / den2
                a  = 2 * κeffWest_nE_Δx_nW * quotient
                aa = a * coeff
                c  = 2 * κDeffWest_nE_DeeffEast_nW / den2
                cc = c * coeff
                d  = 2 * κDeffEast_[nE] / (SceE * Δx_[nE])
                dd = d * (ce_[nE] - ce_[nE+1]) / SceE
                p = DeeffWest_nE_Δx_nW / DeeffEast_nW_Δx_nE
                ravelJ_[nrE-1:nrE+2] = -a - aa - c - cc, a - aa*p + c - cc*p + d - dd, -d - dd  # 界面右侧控制体

            ravelJ_jintneg_ηintneg_[:] = -solve_djintdηint_(T, aeffneg, i0intneg_, ηintneg_) # ∂Fjintneg/∂ηintneg
            ravelJ_jintpos_ηintpos_[:] = -solve_djintdηint_(T, aeffpos, i0intpos_, ηintpos_) # ∂Fjintpos/∂ηintpos
            if i0intnegUnknown:
                ravelJ_jintneg_i0intneg_[:]   = -solve_djintdi0int_(T, aeffneg, ηintneg_)  # ∂Fjintneg/∂i0intneg
                ravelJ_i0intneg_csnegsurf_[:] = -solve_di0intdcssurf_(kneg, csmaxneg, csnegsurf_, ceneg_, i0intneg_)  # ∂Fi0intneg/∂csnegsurf
                ravelJ_i0intneg_ceneg_[:]     = -solve_di0intdce_(ceneg_, i0intneg_)       # ∂Fi0intneg/∂ce
            if i0intposUnknown:
                ravelJ_jintpos_i0intpos_[:]   = -solve_djintdi0int_(T, aeffpos, ηintpos_)  # ∂Fjintpos/∂i0intpos
                ravelJ_i0intpos_cspossurf_[:] = -solve_di0intdcssurf_(kpos, csmaxpos, cspossurf_, cepos_, i0intpos_)  # ∂Fi0intpos/∂cspossurf
                ravelJ_i0intpos_cepos_[:]     = -solve_di0intdce_(cepos_, i0intpos_)       # ∂Fi0intpos/∂ce
            ravelJ_ηintneg_csnegsurf_[:] = solve_dUOCPdθsneg_(csnegsurf_/csmaxneg) / csmaxneg  # ∂Fηintneg/∂csnegsurf
            ravelJ_ηintpos_cspossurf_[:] = solve_dUOCPdθspos_(cspossurf_/csmaxpos) / csmaxpos  # ∂Fηintpos/∂cspossurf
            if lithiumPlating:
                ravelJ_jLP_ceneg_[:] = -solve_djLPdce_(T, aeffneg, ceneg_, i0LP_, ηLP_)  # ∂FjLP/∂ce
                ravelJ_jLP_ηLP_[:]   = -solve_djLPdηLP_(T, aeffneg, i0LP_, ηLP_)         # ∂FjLP/∂ηLP

            if (self.banded_experience_of_J__ is None) and any(data['I']):
                self.banded_experience_of_J__ = expe = P2Dbase.banded_experience(J__)
                if verbose:
                    print(f'重排因变量Jacobi矩阵J__的下带宽{expe['l']}，上带宽{expe['u']}')

            # Newton迭代新解向量
            if expe := self.banded_experience_of_J__:
                # 带状化求解
                ΔX_ = solve_banded_matrix(J__, F_, **expe)
            else:
                # 直接求解
                ΔX_ = solve(J__, F_)

            X_ -= ΔX_

            if isnan(X_).any():
                return nNewton, False, 'nan'
            if (ce_<=0).any():
                return nNewton, False, 'ce<=0'
            if (csnegsurf_<=0).any():
                return nNewton, False, 'csnegsurf<=0'
            if (csnegsurf_>=csmaxneg).any():
                return nNewton, False, 'csnegsurf>=csmaxneg'
            if (cspossurf_<=0).any():
                return nNewton, False, 'cspossurf<=0'
            if (cspossurf_>=csmaxpos).any():
                return nNewton, False, 'cspossurf>=csmaxpos'

            ΔX_ = abs(ΔX_)
            maxΔφ = ΔX_[s_φ].max()  # 新旧电势场最大绝对误差
            maxΔc = ΔX_[s_c].max()  # 新旧浓度场最大绝对误差
            maxΔj = ΔX_[s_j].max()  # 新旧局部体积电流密度场最大绝对误差
            if maxΔc<0.2 and maxΔφ<1e-3 and maxΔj<1:
                break
        else:
            if verbose:
                print(f'时刻t = {data['t'][-1]}s，步进{Δt = }s，Newton迭代达到最大次数{nNewton}，'
                      f'{maxΔc = :.4f} mol/m^3，{maxΔφ = :.6f} V，{maxΔj = :.3f} A/m^3')

        # Newton迭代收敛，更新状态量
        if decouple:
            match timeDiscretization:
                case 'CN':
                    self.csneg__[:] = csnegI__ + outer(γneg_, jintneg_ + self.jintneg_)
                    self.cspos__[:] = csposI__ + outer(γpos_, jintpos_ + self.jintpos_)
                case 'backward':
                    self.csneg__[:] = csnegI__ + outer(γneg_, jintneg_)
                    self.cspos__[:] = csposI__ + outer(γpos_, jintpos_)
        else:
            self.csneg__[:] = csneg_.reshape(Nr, Nneg, order='F')
            self.cspos__[:] = cspos_.reshape(Nr, Npos, order='F')
        self.csnegsurf_[:] = self.solve_csnegsurf_(aneg, Dsneg, self.csneg__, jintneg_)
        self.cspossurf_[:] = self.solve_cspossurf_(apos, Dspos, self.cspos__, jintpos_)
        # self.csnegsurf_[:] = csnegsurf_
        # self.cspossurf_[:] = cspossurf_
        self.ce_[:] = ce_
        self.φe_[:] = φe_
        self.φsneg_[:] = φsneg_
        self.φspos_[:] = φspos_
        self.jintneg_[:] = jintneg_
        self.jintpos_[:] = jintpos_
        self.jneg_[:] = jintneg_
        self.jpos_[:] = jintpos_
        if doubleLayerEffect:
            self.jDLneg_[:] = jDLneg_
            self.jDLpos_[:] = jDLpos_
            self.jneg_ += jDLneg_
            self.jpos_ += jDLpos_
        self.i0intneg_[:] = i0intneg_
        self.i0intpos_[:] = i0intpos_
        self.ηintneg_[:] = ηintneg_
        self.ηintpos_[:] = ηintpos_
        if lithiumPlating:
            self.jLP_[:] = jLP_
            self.jneg_ += jLP_

        return nNewton, True, None  # 返回Newton迭代次数

    def count_lithium(self):
        """统计锂电荷量"""
        qsneg = self.θsneg*self.Qneg
        qspos = self.θspos*self.Qpos
        qe = (self.ce_*self.Δx_*self.A*self.εe_).sum()*P2Dbase.F/3600
        print(f'合计锂电荷总量{qsneg + qspos + qe + self.QLP:.8g} Ah = '
              f'负极嵌锂{qsneg:.8g} Ah + 正极嵌锂{qspos:.8g} Ah + '
              f'电解液锂{qe:.8g} Ah + 负极析锂{self.QLP:.8g} Ah')

    @property
    def i0intneg(self):
        """负极主反应交换电流密度 [A/m^2]"""
        return self.Arrhenius(self._i0intneg, self.Ekneg)
    @i0intneg.setter
    def i0intneg(self, i0intneg):
        self._i0intneg = i0intneg

    @property
    def i0intpos(self):
        """正极主反应交换电流密度 [A/m^2]"""
        return self.Arrhenius(self._i0intpos, self.Ekpos)
    @i0intpos.setter
    def i0intpos(self, i0intpos):
        self._i0intpos = i0intpos

    @property
    def i0LP(self):
        """负极析锂反应交换电流密度 [A/m^2]"""
        return self.Arrhenius(self._i0LP, self.EkLP)
    @i0LP.setter
    def i0LP(self, i0LP):
        self._i0LP = i0LP

    @property
    def De(self):
        """电解液锂离子扩散系数 [m^2/s]"""
        return self.Arrhenius(self._De, self.EDe)
    @De.setter
    def De(self, De):
        self._De = De

    @property
    def Deeff_(self):
        """(Ne,) 全区域各控制体电解液扩散系数 [m2/s]"""
        return self.De*self.εeb_

    @property
    def κ(self):
        """电解液离子电导率 [S/m]"""
        return self.Arrhenius(self._κ, self.Eκ)
    @κ.setter
    def κ(self, κ):
        self._κ = κ

    @property
    def κeff_(self):
        """(Ne,) 全区域各控制体电解液有效离子电导率 [S/m]"""
        return self.κ*self.εeb_

    @property
    def κDeff_(self):
        """(Ne,) 全区域各控制体电解液有效扩散离子电导率 [S/m]"""
        return (2*P2Dbase.R*self.T*(1 - self.tplus)/P2Dbase.F*self.TDF) * self.κeff_

    @property
    def Qneg(self):
        """负极容量 [Ah]"""
        return P2Dbase.F*self.A*self.Lneg*self.εsneg*self.csmaxneg/3600

    @property
    def Qpos(self):
        """正极容量 [Ah]"""
        return P2Dbase.F*self.A*self.Lpos*self.εspos*self.csmaxpos/3600

    @property
    def Qcell(self):
        """全电池理论可用容量 [Ah]"""
        return min(self.Qneg*(self.θmaxneg - self.θminneg),
                   self.Qpos*(self.θmaxpos - self.θminpos),)

    @property
    def σeffneg(self):
        """负极固相有效电导率 [S/m]"""
        return self.σneg*self.εsneg**self.bneg  # 负极固相有效电导率 [S/m]

    @property
    def σeffpos(self):
        """正极固相有效电导率 [S/m]"""
        return self.σpos*self.εspos**self.bpos

    @property
    def aneg(self):
        """负极固相单位体积的表面积 [m^2/m^3]"""
        return 3/self.Rsneg*self.εsneg

    @property
    def apos(self):
        """正极固相单位体积的表面积 [m^2/m^3]"""
        return 3/self.Rspos*self.εspos

    @property
    def aeffneg(self):
        """负极固相单位体积的有效接触表面积 [m^2/m^3]"""
        return self.aneg*self.Aeffneg

    @property
    def aeffpos(self):
        """正极固相单位体积的有效接触表面积 [m^2/m^3]"""
        return self.apos*self.Aeffpos

    @property
    def εe_(self):
        """(Ne,) 全区域各控制体电解液体积分数 [C]"""
        return concatenate([
            full(self.Nneg, self.εeneg),
            full(self.Nsep, self.εesep),
            full(self.Npos, self.εepos),])

    @property
    def εeb_(self):
        """(Ne,) 全区域各控制体电解液体积分数的Bruggman指数次幂"""
        return concatenate([
            full(self.Nneg, self.εeneg**self.bneg),
            full(self.Nsep, self.εesep**self.bsep),
            full(self.Npos, self.εepos**self.bpos), ])

    @property
    def U(self):
        """正负极端电压 [V]"""
        a = 0.5*self.I/self.A
        φsposCollector = self.φspos_[-1] - a*self.Δxpos/self.σeffpos
        φsnegCollector = self.φsneg_[0]  + a*self.Δxneg/self.σeffneg
        return φsposCollector - φsnegCollector + self.Ul

    @property
    def ceneg_(self):
        """(Nneg,) 负极区域电解液锂离子浓度 [mol/m^3]"""
        return self.ce_[:self.Nneg]

    def solve_csnegsurf_(self, aneg, Dsneg, csneg__, jintneg_):
        """求解负极固相表面锂离子浓度场 [mol/m^3]"""
        if self.decouple:
            return self.coeffsExpl_.dot(csneg__[-3:])
        else:
            Nr = self.Nr
            csneg_ = csneg__.ravel('F')
            c_ = self.coeffs_csneg_
            return -(  c_[-3] * csneg_[Nr-3::Nr]
                     + c_[-2] * csneg_[Nr-2::Nr]
                     + c_[-1] * csneg_[Nr-1::Nr]
                     + 1/(aneg * P2Dbase.F * Dsneg) * jintneg_
                     )/self.coeff_csnegsurf_csnegsurf

    def solve_cspossurf_(self, apos, Dspos, cspos__, jintpos_):
        """求解正极固相表面锂离子浓度场 [mol/m^3]"""
        if self.decouple:
            return self.coeffsExpl_.dot(cspos__[-3:])
        else:
            Nr = self.Nr
            cspos_ = cspos__.ravel('F')
            c_ = self.coeffs_cspos_
            return -(  c_[-3] * cspos_[Nr-3::Nr]
                     + c_[-2] * cspos_[Nr-2::Nr]
                     + c_[-1] * cspos_[Nr-1::Nr]
                     + 1/(apos * P2Dbase.F * Dspos) * jintpos_
                     )/self.coeff_cspossurf_cspossurf

    @property
    def cepos_(self):
        """(Npos,) 正极区域电解液锂离子浓度 [mol/m^3]"""
        return self.ce_[-self.Npos:]

    @property
    def ceInterfaces_(self):
        """(Ne+1,)各控制体界面的锂离子浓度 [mol/m^3]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_ = self.Δx_
        ce_ = self.ce_
        DeeffWest_ = DeeffEast_ = self.Deeff_
        ceInterfaces_ = hstack([ce_[0], (ce_[:-1] + ce_[1:])*0.5, ce_[-1]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = DeeffWest_[nE]*Δx_[nW], DeeffEast_[nW]*Δx_[nE]
            ceInterfaces_[nW+1] = (a*ce_[nE] + b*ce_[nW])/(a + b)
        return ceInterfaces_

    @property
    def φeInterfaces_(self):
        """(Ne+1,) 各控制体界面的电解液电势 [V]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_, ΔxWest_, ΔxEast_ = self.Δx_, self.ΔxWest_, self.ΔxEast_
        φe_, ce_ = self.φe_, self.ce_
        ceInterfaces_ = self.ceInterfaces_
        ceWest_ = ceInterfaces_[:-1]  # (Ne,) 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        ceEast_ = ceInterfaces_[1:]   # (Ne,) 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradceWest_ = hstack([0, (ce_[1:] - ce_[:-1])/ΔxWest_[1:]])   # (Ne,) 各控制体左界面的锂离子浓度梯度 [mol/m^4]
        gradceEast_ = hstack([(ce_[1:] - ce_[:-1])/ΔxEast_[:-1], 0])  # (Ne,) 各控制体右界面的锂离子浓度梯度 [mol/m^4]
        for nW, nE in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradceEast_[nW] = (ceEast_[nW] - ce_[nW])/(0.5*Δx_[nW])
            gradceWest_[nE] = (ce_[nE] - ceWest_[nE])/(0.5*Δx_[nE])
        gradlnceWest_ = gradceWest_/ceWest_  # (Ne,) 各控制体左界面的对数锂离子浓度梯度 [ln mol/m^4]
        gradlnceEast_ = gradceEast_/ceEast_  # (Ne,) 各控制体右界面的对数锂离子浓度梯度 [ln mol/m^4]

        φeInterfaces_ = hstack([φe_[0], (φe_[:-1] + φe_[1:])*0.5, φe_[-1]])
        κeffWest_ = κeffEast_ = self.κeff_
        κDeffWest_ = κDeffEast_ = self.κDeff_
        for nW, nE in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = κeffEast_[nW]*Δx_[nE], κeffWest_[nE]*Δx_[nW]
            c = 0.5*Δx_[nW]*Δx_[nE]
            φeInterfaces_[nE] = (  a*φe_[nW] + b*φe_[nE]
                                 + c*κDeffEast_[nW]*gradlnceEast_[nW]
                                 - c*κDeffWest_[nE]*gradlnceWest_[nE]
                                 )/(a + b)
        return φeInterfaces_

    @staticmethod
    def solve_jint_(T, aeff, i0int_, ηint_) -> ndarray:
        """求解主反应局部体积电流密度jint [A/m^3]"""
        return 2*aeff*i0int_*sinh(P2Dbase.F/(2*P2Dbase.R*T)*ηint_)

    @property
    def djintdi0intneg_(self):
        """负极主反应局部体积电流密度jintneg对交换电流密度i0intneg的偏导数 [A/m^3 / A/m^2]"""
        return DFNJTFP2D.solve_djintdi0int_(self.T, self.aeffneg, self.ηintneg_)

    @property
    def djintdi0intpos_(self):
        """正极主反应局部体积电流密度jintpos对交换电流密度i0intpos的偏导数 [A/m^3 / A/m^2]"""
        return DFNJTFP2D.solve_djintdi0int_(self.T, self.aeffpos, self.ηintpos_)

    @staticmethod
    def solve_djintdi0int_(T, aeff, ηint_) -> ndarray:
        """求解主反应局部体积电流密度jint对交换电流密度i0int的偏导数 [A/m^3 / A/m^2]"""
        return 2*aeff*sinh(P2Dbase.F/(2*P2Dbase.R*T)*ηint_)

    @property
    def djintdηintneg_(self):
        """负极主反应局部体积电流密度jintneg对过电位ηintneg的偏导数 [A/m^3 / V]"""
        return DFNJTFP2D.solve_djintdηint_(self.T, self.aeffneg, self.i0intneg_, self.ηintneg_)

    @property
    def djintdηintpos_(self):
        """正极主反应局部体积电流密度jintpos对过电位ηintpos的偏导数 [A/m^3 / V]"""
        return DFNJTFP2D.solve_djintdηint_(self.T, self.aeffpos, self.i0intpos_, self.ηintpos_)

    @staticmethod
    def solve_djintdηint_(T, aeff, i0int_, ηint_) -> ndarray:
        """求解主反应局部体积电流密度jint对过电位ηint的偏导数 [A/m^3 / V]"""
        FRT = P2Dbase.F/(P2Dbase.R*T)
        return FRT*aeff*i0int_*cosh(FRT*0.5*ηint_)

    @staticmethod
    def solve_i0int_(k, csmax, cssurf_, ce_) -> ndarray:
        """求主反应交换电流密度 [A/m^2]"""
        return P2Dbase.F * k * sqrt(ce_*(csmax - cssurf_)*cssurf_)

    @property
    def di0intdceneg_(self):
        """负极主反应交换电流密度i0int对电解液浓度ce的偏导数 [A/m2 / mol/m^3]"""
        return 0 if self._i0intneg \
            else DFNJTFP2D.solve_di0intdce_(self.ceneg_, self.i0intneg_)

    @property
    def di0intdcepos_(self):
        """正极主反应交换电流密度i0int对电解液浓度ce的偏导数 [A/m2 / mol/m^3]"""
        return 0 if self._i0intpos \
            else DFNJTFP2D.solve_di0intdce_(self.cepos_, self.i0intpos_)

    @staticmethod
    def solve_di0intdce_(ce_, i0int_):
        """求解主反应交换电流密度i0int对电解液锂离子浓度ce的偏导数 [A/m2 / mol/m^3]"""
        return 0.5*i0int_/ce_

    @property
    def di0intdcsnegsurf_(self):
        """负极主反应交换电流密度i0intneg对电极表面浓度csnegsurf的偏导数  [A/-]"""
        return 0 if self._i0intneg\
            else DFNJTFP2D.solve_di0intdcssurf_(
            self.kneg, self.csmaxneg, self.csnegsurf_, self.ceneg_, self.i0intneg_)

    @property
    def di0intdcspossurf_(self):
        """正极主反应交换电流密度i0intpos对电极表面浓度cspossurf的偏导数"""
        return 0 if self._i0intpos\
            else DFNJTFP2D.solve_di0intdcssurf_(
            self.kpos, self.csmaxpos, self.cspossurf_, self.cepos_, self.i0intpos_)

    @staticmethod
    def solve_di0intdcssurf_(k, csmax, cssurf_, ce_, i0int_):
        """求解主反应交换电流密度i0int对固相颗粒表面锂离子浓度cssurf的偏导数"""
        Fk = P2Dbase.F*k
        return 0.5*Fk*Fk * ce_*(csmax - 2*cssurf_)/i0int_

    @staticmethod
    def solve_jLP_(T, aeffneg, i0LP_, ηLP_) -> ndarray:
        """求解析锂反应局部体积电流密度jLP [A]"""
        FRT = P2Dbase.F/P2Dbase.R/T
        a, b = 0.3*FRT, -0.7*FRT
        jLP_ = aeffneg * i0LP_ * (exp(a*ηLP_) - exp(b*ηLP_))
        jLP_[ηLP_>=0] = 0
        return jLP_

    @property
    def djLPdce_(self):
        """析锂反应局部体积电流密度jLP对电解液浓度ce的偏导数 [A/m^3 / mol/m^3]"""
        return 0 if self._i0LP \
            else DFNJTFP2D.solve_djLPdce_(self.T, self.aeffneg,
                                       self.ceneg_, self.i0LP_, self.ηLPneg_)

    @staticmethod
    def solve_djLPdce_(T, aeffneg, ceneg_, i0LP_, ηLP_):
        """析锂反应局部体积电流密度jLP对电解液锂离子浓度ce的偏导数 [A/m^3 / mol/m^3]"""
        FRT = P2Dbase.F/P2Dbase.R/T
        a, b = 0.3*FRT, -0.7*FRT
        djLPdi0LP_ = aeffneg*(exp(a*ηLP_) - exp(b*ηLP_))
        di0LPdce_ = 0.3*i0LP_/ceneg_
        djLPdce_ = djLPdi0LP_ * di0LPdce_
        djLPdce_[ηLP_>=0] = 0
        return djLPdce_

    @property
    def djLPdηLP_(self):
        """析锂反应局部体积电流密度jLP对析锂过电位ηLP的偏导数 [A/m^3 / V]"""
        return DFNJTFP2D.solve_djLPdηLP_(self.T, self.aeffneg, self.i0LP_, self.ηLPneg_)

    @staticmethod
    def solve_djLPdηLP_(T, aeffneg, i0LP_, ηLP_):
        """求解析锂反应局部体积电流密度jLP对析锂过电位ηLP的偏导数 [A/m^3 / V]"""
        FRT = P2Dbase.F/(P2Dbase.R*T)
        a, b = 0.3*FRT, -0.7*FRT
        djLPdηLP_ = aeffneg * i0LP_ * (a*exp(a*ηLP_) - b*exp(b*ηLP_))
        djLPdηLP_[ηLP_>=0] = 0
        return djLPdηLP_

    @property
    def i0LP_(self):
        """(Nneg,) 析锂反应交换电流密度场 [A/m^2]"""
        return full(self.Nneg, self.i0LP) if self._i0LP else \
            DFNJTFP2D.solve_i0LP_(self.kLP, self.ceneg_)

    @staticmethod
    def solve_i0LP_(kLP, ceneg_) -> ndarray:
        """由液相浓度场求析锂反应交换电流密度 [A/m^2]"""
        return P2Dbase.F * kLP * ceneg_**0.3

    @property
    def ηLPneg_(self):
        """负极析锂反应过电位场 [V]"""
        return self.φsneg_ - self.φeneg_ - self.RSEIneg/self.aeffneg * self.jneg_

    @property
    def ηLPpos_(self):
        """正极析锂反应过电位场 [V]"""
        return self.φspos_ - self.φepos_ - self.RSEIpos/self.aeffpos * self.jpos_

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

    @property
    def gradlnce_(self):
        """对数电解液锂离子浓度场的梯度 [ln(mol/m^3)/m]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_, ΔxWest_, ΔxEast_ = self.Δx_, self.ΔxWest_, self.ΔxEast_
        x_ = self.x_
        ce_ = self.ce_
        ceInterfaces_ = self.ceInterfaces_
        ceWest_ = ceInterfaces_[:-1]  # 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        ceEast_ = ceInterfaces_[1:]  # 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradce_ = hstack([
            (ce_[1] - ce_[0])/(x_[1] - x_[0])*0.5,       # 负极首个控制体
            (ce_[2:] - ce_[:-2])/(x_[2:] - x_[:-2]),         # 内部控制体
            (ce_[-1] - ce_[-2])/(x_[-1] - x_[-2])*0.5])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradce_[nW] = ((ce_[nW] - ce_[nW - 1])/ΔxWest_[nW] + (ceEast_[nW] - ce_[nW])/(0.5*Δx_[nW])) * 0.5  # 界面左侧控制体
            gradce_[nE] = ((ce_[nE] - ceWest_[nE])/(0.5*Δx_[nE]) + (ce_[nE + 1] - ce_[nE])/ΔxEast_[nE]) * 0.5  # 界面右侧控制体
        return gradce_/ce_

    @property
    def gradφsneg_(self):
        """负极固相电势场的梯度 [V/m]"""
        φsneg_ = self.φsneg_
        Δxneg = self.Δxneg
        return hstack([
            (-self.I/self.A/self.σeffneg + (φsneg_[1] - φsneg_[0])/Δxneg)*0.5, # 负极首个控制体
            (φsneg_[2:] - φsneg_[:-2])/(2*Δxneg),    # 负极内部控制体
            (φsneg_[-1] - φsneg_[-2])/Δxneg * 0.5])  # 负极末尾控制体

    @property
    def gradφspos_(self):
        """正极固相电势场的梯度 [V/m]"""
        φspos_ = self.φspos_
        Δxpos = self.Δxpos
        return hstack([
            (φspos_[1] - φspos_[0])/Δxpos * 0.5,  # 正极首个控制体
            (φspos_[2:] - φspos_[:-2])/(2*Δxpos),   # 正极内部控制体
            ((φspos_[-1] - φspos_[-2])/Δxpos + -self.I/self.A/self.σeffpos) * 0.5])  # 正极末尾控制体

    @property
    def gradφe_(self):
        """电解液电势场的梯度∂φe/∂x [V/m]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        x_, Δx_, ΔxWest_, ΔxEast_ = self.x_, self.Δx_, self.ΔxWest_, self.ΔxEast_
        φe_ = self.φe_
        φeInterfaces_ = self.φeInterfaces_
        φeWest_ = φeInterfaces_[:-1]  # 各控制体左界面的电解液电势 [V]
        φeEast_ = φeInterfaces_[1:]   # 各控制体右界面的电解液电势 [V]
        gradφe_ = hstack([
            (φe_[1] - φe_[0])/(x_[1] - x_[0]) * 0.5,       # 负极首个控制体
            (φe_[2:] - φe_[:-2])/(x_[2:] - x_[:-2]),       # 内部控制体
            (φe_[-1] - φe_[-2])/(x_[-1] - x_[-2]) * 0.5])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradφe_[nW] = ((φe_[nW] - φe_[nW - 1])/ΔxWest_[nW] + (φeEast_[nW] - φe_[nW])/(0.5*Δx_[nW])) * 0.5
            gradφe_[nE] = ((φe_[nE] - φeWest_[nE])/(0.5*Δx_[nE]) + (φe_[nE + 1] - φe_[nE])/ΔxEast_[nE]) * 0.5
        return gradφe_

    @property
    def IDLneg(self):
        """负极双电层电流 [A]"""
        return self.jDLneg_.sum()*self.Δxneg*self.A

    @property
    def IDLpos(self):
        """正极双电层电流 [A]"""
        return self.jDLpos_.sum()*self.Δxpos*self.A

    @property
    def ILP(self):
        """析锂反应电流 [A]"""
        return self.jLP_.sum()*self.Δxneg*self.A

    @property
    def Qohme(self):
        """电解液欧姆热 [W]"""
        return self.A*((self.κeff_*self.gradφe_ + self.κDeff_*self.gradlnce_)*self.gradφe_*self.Δx_).sum()

    @property
    def Qohmneg(self):
        """负极固相欧姆热 [W]"""
        grad_ = self.gradφsneg_
        return self.A*self.σeffneg*(grad_*grad_).sum()*self.Δxneg

    @property
    def Qohmpos(self):
        """正极固相欧姆热 [W]"""
        grad_ = self.gradφspos_
        return self.A*self.σeffpos*(grad_*grad_).sum()*self.Δxpos

    @property
    def Qrxnneg(self):
        """负极反应热 [W]"""
        return self.A*(self.jintneg_*self.ηintneg_).sum()*self.Δxneg

    @property
    def Qrxnpos(self):
        """正极反应热 [W]"""
        return self.A*(self.jintpos_*self.ηintpos_).sum()*self.Δxpos

    @property
    def Qrevneg(self):
        """负极可逆热 [W]"""
        dUOCPdTnegsurf_ = self.dUOCPdTneg(self.csnegsurf_/self.csmaxneg) if callable(self.dUOCPdTneg) else self.dUOCPdTneg
        return self.A*(self.T*dUOCPdTnegsurf_*self.jintneg_).sum()*self.Δxneg

    @property
    def Qrevpos(self):
        """正极可逆热 [W]"""
        dUOCPdTpossurf_ = self.dUOCPdTpos(self.cspossurf_/self.csmaxpos) if callable(self.dUOCPdTpos) else self.dUOCPdTpos
        return self.A*(self.T*dUOCPdTpossurf_*self.jintpos_).sum()*self.Δxpos

    @property
    def θsneg(self):
        """负极嵌锂状态 [–]"""
        return self.Vr_.dot(self.csneg__).mean()/self.csmaxneg

    @property
    def θspos(self):
        """正极嵌锂状态 [–]"""
        return self.Vr_.dot(self.cspos__).mean()/self.csmaxpos

    @property
    def isneg_(self):
        """负极固相电流密度场 [A/m^2]"""
        return -self.σeffneg*self.gradφsneg_

    @property
    def ispos_(self):
        """正极固相电流密度场 [A/m^2]"""
        return -self.σeffpos*self.gradφspos_

    @property
    def ie_(self):
        return -self.κeff_*self.gradφe_ + self.κDeff_*self.gradlnce_

    @property
    def xPlot_(self):
        """全区域控制体中心的坐标（用于作图） [μm]"""
        return self.x_*1e6

    @property
    def xInterfacesPlot_(self):
        """各控制体交界面的坐标（用于作图） [μm]"""
        return self.xInterfaces_*1e6

    def plot_i(self,
               t_: Sequence | None = None,  # 时刻序列
               ):
        """固液相电流密度-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        isneg__ = self('isneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极固相电流密度场 [V]
        ispos__ = self('ispos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极固相电流密度场 [V]
        ie__ = self('ie_', t_=t_, x_=self.x_)  # 呈时间序列的电解液电流密度场 [V]
        i_   = self('I', t_=t_)/self.A         # 呈时间序列的总电流密度 [A/m^2]
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (isneg_, ispos_, i, t) in enumerate(zip(isneg__, ispos__, i_, t_)):
            x_ = (0, *self.xPlot_[:Nneg], self.xInterfacesPlot_[Nneg],
                  *self.xPlot_[Nneg:Nneg+Nsep], self.xInterfacesPlot_[Nneg+Nsep],
                  *self.xPlot_[-Npos:], self.xInterfacesPlot_[-1])
            y_ = i, *isneg_, 0, *[0]*Nsep, 0, *ispos_, i
            ax1.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax1.set_ylabel(rf'Solid-phase current density ${{\it i}}_{{s}}$({self.xSign}, {self.tSign}) [A/m$^2$]')
        ax1.legend(bbox_to_anchor=[1, 1])

        for n, (ie_, i, t) in enumerate(zip(ie__, i_, t_)):
            x_ = 0, *self.xPlot_, self.xInterfacesPlot_[-1]
            y_ = 0, *ie_, 0
            ax2.plot(x_, y_, 'o-', color=get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax2.set_ylabel(rf'Liquid-phase current density ${{\it i}}_{{e}}$({self.xSign}, {self.tSign}) [A/m$^2$]')

        self.plot_interfaces(ax1, ax2)
        plt.show()

    def initialize_consistent(self,
            csneg__: ndarray,
            cspos__: ndarray,
            ce_: ndarray,
            I: float | int = 0):
        """一致性初始化
        已知：csneg__、cspos__、ce_、I
        求解：csnegsurf_、cspossurf__、φsneg_、φspos_、φe_、jintneg_、jintpos、i0intneg_、i0intpos_、ηintneg_、ηintpos_
        令：jDLneg_ = jDLpos_ = jLP_ = 0
        """
        Nr, Nneg, Nsep, Npos, Ne = self.Nr, self.Nneg, self.Nsep, self.Npos, self.Ne  # 读取网格数
        assert csneg__.shape==(Nr, Nneg), f'负极固相颗粒内部锂离子浓度csneg__.shape应为({Nr}, {Nneg})'
        assert csneg__.shape==(Nr, Nneg), f'正极固相颗粒内部锂离子浓度cspos__.shape应为({Nr}, {Npos})'
        assert ce_.shape==(self.Ne,), f'电解液锂离子浓度ce_.shape应为({self.Ne},)'
        assert ((0<=csneg__) & (csneg__<=self.csmaxneg)).all(), 'csneg__取值范围应为(0, csmaxneg) [mol/m^3]'
        assert ((0<=cspos__) & (cspos__<=self.csmaxpos)).all(), 'cspos__取值范围应为(0, csmaxpos) [mol/m^3]'
        assert (0<ce_).all(), 'ce_取值应大于0 [mol/m^3]'
        ceneg_, cepos_ = ce_[:Nneg], ce_[-Npos:]

        if self.ravelK_ is None:
            self._generate_K__bK_and_slices()

        # 更新K__矩阵纯电化学参数相关值
        self.update_K__with_pure_electrochemical_parameters()

        # 生成索引
        sK = self.sK  # 原索引
        s2idx = P2Dbase.s2idx
        idx_ = concatenate([
            s2idx(sK.s_csnegsurf), s2idx(sK.s_cspossurf),
            s2idx(sK.s_φsneg),     s2idx(sK.s_φspos),     s2idx(sK.s_φe),
            s2idx(sK.s_jintneg),   s2idx(sK.s_jintpos),
            s2idx(sK.s_i0intneg),  s2idx(sK.s_i0intpos),
            s2idx(sK.s_ηintneg),   s2idx(sK.s_ηintpos),])
        Kinit__ = self.ravelK_.base[ix_(idx_, idx_)]  # 提取K__矩阵
        Ninit = Kinit__.shape[0]             # Kinit__矩阵的行列数
        start = 0
        def reassign(s_old: slice) -> slice:
            """对矩阵Kinit__重新安排切片索引"""
            nonlocal start
            length = s_old.stop - s_old.start
            s_new = slice(start, start + length)
            start += length
            return s_new
        # 重新安排切片索引
        s_csnegsurf = reassign(sK.s_csnegsurf)
        s_cspossurf = reassign(sK.s_cspossurf)
        s_φsneg = reassign(sK.s_φsneg)
        s_φspos = reassign(sK.s_φspos)
        s_φe = reassign(sK.s_φe)
        s_jintneg = reassign(sK.s_jintneg)
        s_jintpos = reassign(sK.s_jintpos)
        s_i0intneg = reassign(sK.s_i0intneg)
        s_i0intpos = reassign(sK.s_i0intpos)
        s_ηintneg = reassign(sK.s_ηintneg)
        s_ηintpos = reassign(sK.s_ηintpos)

        # 读取方法
        solve_jint_ = DFNJTFP2D.solve_jint_
        solve_djintdηint_ = DFNJTFP2D.solve_djintdηint_
        solve_djintdi0int_ = DFNJTFP2D.solve_djintdi0int_
        solve_i0int_ = DFNJTFP2D.solve_i0int_
        solve_di0intdcssurf_ = DFNJTFP2D.solve_di0intdcssurf_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_  # 读取：负极、正极开路电位函数 [V]
        solve_dUOCPdθsneg_ = self.solve_dUOCPdθsneg_
        solve_dUOCPdθspos_ = self.solve_dUOCPdθspos_  # 读取：负极、正极开路电位对嵌锂状态的偏导数函数 [V/–]

        # 读取参数
        aeffneg = self.aeffneg
        aeffpos = self.aeffpos
        RSEI2aeffneg = self.RSEIneg/aeffneg
        RSEI2aeffpos = self.RSEIpos/aeffpos
        csmaxneg, csmaxpos = self.csmaxneg, self.csmaxpos
        κDeffWest_ = κDeffEast_ = self.κDeff_
        DeeffWest_ = DeeffEast_ = self.Deeff_
        T = self.T
        F2RT = P2Dbase.F/(2*P2Dbase.R*T)
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        if i0intnegUnknown := (self._i0intneg is None):
            kneg = self.kneg          # 读取：负极主反应速率常数
        else:
            i0intneg = self.i0intneg  # 读取：负极主反应交换电流密度 [A/m^2]
        if i0intposUnknown := (self._i0intpos is None):
            kpos = self.kpos          # 读取：正极主反应速率常数
        else:
            i0intpos = self.i0intpos  # 读取：正极主反应交换电流密度 [A/m^2]

        # 外推表面浓度
        c_ = self.coeffsExpl_
        csnegsurfExpl_ = c_.dot(csneg__[-3:])
        cspossurfExpl_ = c_.dot(cspos__[-3:])

        ## 对Kinit__的右端项bKinit_赋值 ##
        bKinit_ = zeros(Ninit)  # 右端项
        if self.decouple:
            # 强制表面浓度约束：认为 csnegsurf_、cspossurf_ 是外推得到的已知值
            bKinit_[s_csnegsurf] = csnegsurfExpl_
            bKinit_[s_cspossurf] = cspossurfExpl_
        else:
            # 用颗粒扩散边界条件：方程关联jint、cssurf及靠近颗粒表面的3个内部节点浓度
            csneg_ = csneg__.ravel('F')
            cspos_ = cspos__.ravel('F')
            c_ = self.coeffs_csneg_
            bKinit_[s_csnegsurf] = -(
                    c_[-3] * csneg_[Nr-3::Nr]
                  + c_[-2] * csneg_[Nr-2::Nr]
                  + c_[-1] * csneg_[Nr-1::Nr])
            c_ = self.coeffs_cspos_
            bKinit_[s_cspossurf] = -(
                   c_[-3] * cspos_[Nr-3::Nr]
                 + c_[-2] * cspos_[Nr-2::Nr]
                 + c_[-1] * cspos_[Nr-1::Nr])
        # 固相电流边界条件
        i = I/self.A  # 电极电流密度 [A/m^2]
        bKinit_[s_φsneg.start]    = -self.Δxneg*i/self.σeffneg
        bKinit_[s_φspos.stop - 1] =  self.Δxpos*i/self.σeffpos
        # 电解液电势方程的电解液锂离子浓度项
        q_ = 2*(ce_[1:]  - ce_[:-1])/(ce_[1:] + ce_[:-1])  # (Ne-1,)
        a_ = κDeffWest_[1:]  * q_ / ΔxWest_[1:]   # (Ne-1,)
        c_ = κDeffEast_[:-1] * q_ / ΔxEast_[:-1]  # (Ne-1,)
        bKinit_φe_ = bKinit_[s_φe]
        bKinit_φe_[:] = hstack([c_, 0]) - hstack([0, a_])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = DeeffWest_[nE]*Δx_[nW], DeeffEast_[nW]*Δx_[nE]
            cInterface = (a*ce_[nE] + b*ce_[nW])/(a + b)
            bKinit_φe_[nW] = (κDeffEast_[nW] * (cInterface - ce_[nW]) / (0.5*Δx_[nW]) / cInterface
                            - κDeffWest_[nW] * (ce_[nW] - ce_[nW-1])  / ΔxWest_[nW]   / (0.5*(ce_[nW] + ce_[nW-1])))
            bKinit_φe_[nE] = (κDeffEast_[nE] * (ce_[nE+1] - ce_[nE])  / ΔxEast_[nE]   / (0.5*(ce_[nE+1] + ce_[nE]))
                            - κDeffWest_[nE] * (ce_[nE] - cInterface) / (0.5*Δx_[nE]) / cInterface)

        # 索引解
        X_ = zeros(Ninit)
        csnegsurf_ = X_[s_csnegsurf]
        cspossurf_ = X_[s_cspossurf]
        φsneg_ = X_[s_φsneg]
        φspos_ = X_[s_φspos]
        φe_ = X_[s_φe]
        jintneg_ = X_[s_jintneg]
        jintpos_ = X_[s_jintpos]
        i0intneg_ = X_[s_i0intneg] if i0intnegUnknown else i0intneg
        i0intpos_ = X_[s_i0intpos] if i0intposUnknown else i0intpos
        ηintneg_ = X_[s_ηintneg]
        ηintpos_ = X_[s_ηintpos]

        # 迭代初值
        csnegsurf_[:] = csnegsurfExpl_
        cspossurf_[:] = cspossurfExpl_
        jintneg_[:] = jintneg =  i/self.Lneg  # 初值：主反应平均局部体积电流密度 [A/m^3]
        jintpos_[:] = jintpos = -i/self.Lpos
        if i0intnegUnknown:
            i0intneg_[:] = solve_i0int_(kneg, csmaxneg, csnegsurfExpl_, ceneg_)
        if i0intposUnknown:
            i0intpos_[:] = solve_i0int_(kpos, csmaxpos, cspossurfExpl_, cepos_)
        ηintneg_[:] = arcsinh(jintneg/(2*aeffneg*i0intneg_))/F2RT
        ηintpos_[:] = arcsinh(jintpos/(2*aeffpos*i0intpos_))/F2RT
        φsneg_[:] = ηintneg_ + RSEI2aeffneg*jintneg + solve_UOCPneg_(csnegsurfExpl_/csmaxneg)
        φspos_[:] = ηintpos_ + RSEI2aeffpos*jintpos + solve_UOCPpos_(cspossurfExpl_/csmaxpos)

        # Newton迭代
        J__ = Kinit__.copy()
        ravelJ_ = J__.ravel()
        dsr = partial(diagonalSliceRavel, Ninit)
        ravelJ_jintneg_ηintneg_ = ravelJ_[dsr(s_jintneg, s_ηintneg)]
        ravelJ_jintpos_ηintpos_ = ravelJ_[dsr(s_jintpos, s_ηintpos)]
        ravelJ_ηintneg_csnegsurf_ = ravelJ_[dsr(s_ηintneg, s_csnegsurf)]
        ravelJ_ηintpos_cspossurf_ = ravelJ_[dsr(s_ηintpos, s_cspossurf)]
        if i0intnegUnknown:
            ravelJ_jintneg_i0intneg_ = ravelJ_[dsr(s_jintneg, s_i0intneg)]
            ravelJ_i0intneg_csnegsurf_ = ravelJ_[dsr(s_i0intneg, s_csnegsurf)]
        if i0intposUnknown:
            ravelJ_jintpos_i0intpos_ = ravelJ_[dsr(s_jintpos, s_i0intpos)]
            ravelJ_i0intpos_cspossurf_ = ravelJ_[dsr(s_i0intpos, s_cspossurf)]

        for nNewton in range(1, 101):
            F_ = Kinit__.dot(X_) - bKinit_  # (Ninit,) F残差向量

            # F向量非线性部分
            F_[s_jintneg] -= solve_jint_(T, aeffneg, i0intneg_, ηintneg_)  # F向量jintneg部分
            F_[s_jintpos] -= solve_jint_(T, aeffpos, i0intpos_, ηintpos_)  # F向量jintpos部分
            if i0intnegUnknown:
                F_[s_i0intneg] -= solve_i0int_(kneg, csmaxneg, csnegsurf_, ceneg_)  # F向量i0intneg部分
            if i0intposUnknown:
                F_[s_i0intpos] -= solve_i0int_(kpos, csmaxpos, cspossurf_, cepos_)  # F向量i0intpos部分
            F_[s_ηintneg] += solve_UOCPneg_(csnegsurf_/csmaxneg)  # F向量ηintneg非线性部分
            F_[s_ηintpos] += solve_UOCPpos_(cspossurf_/csmaxpos)  # F向量ηintpos非线性部分

            # 更新Jacobi矩阵
            ravelJ_jintneg_ηintneg_[:] = -solve_djintdηint_(T, aeffneg, i0intneg_, ηintneg_)  # ∂Fjintneg/∂ηintneg
            ravelJ_jintpos_ηintpos_[:] = -solve_djintdηint_(T, aeffpos, i0intpos_, ηintpos_)  # ∂Fjintpos/∂ηintpos
            if i0intnegUnknown:
                ravelJ_jintneg_i0intneg_[:]   = -solve_djintdi0int_(T, aeffneg, ηintneg_)  # ∂Fjintneg/∂i0intneg
                ravelJ_i0intneg_csnegsurf_[:] = -solve_di0intdcssurf_(kneg, csmaxneg, csnegsurf_, ceneg_, i0intneg_)  # ∂Fi0intneg/∂csnegsurf
            if i0intposUnknown:
                ravelJ_jintpos_i0intpos_[:]   = -solve_djintdi0int_(T, aeffpos, ηintpos_)  # ∂Fjintpos/∂i0intpos
                ravelJ_i0intpos_cspossurf_[:] = -solve_di0intdcssurf_(kpos, csmaxpos, cspossurf_, cepos_, i0intpos_)  # ∂Fi0intpos/∂cspossurf
            ravelJ_ηintneg_csnegsurf_[:] = solve_dUOCPdθsneg_(csnegsurf_/csmaxneg) / csmaxneg  # ∂Fηintneg/∂csnegsurf
            ravelJ_ηintpos_cspossurf_[:] = solve_dUOCPdθspos_(cspossurf_/csmaxpos) / csmaxpos  # ∂Fηintpos/∂cspossurf

            ΔX_ = solve(J__, F_)
            X_ -= ΔX_

            if abs(ΔX_).max()<1e-6:
                break
        else:
            raise DFNJTFP2D.Error(f'一致性初始化失败，Newton迭代{nNewton = }次，不收敛，{abs(ΔX_).max() = }')

        # 初始化状态
        self.I = I
        self.csneg__[:] = csneg__
        self.cspos__[:] = cspos__
        self.csnegsurf_[:] = csnegsurf_
        self.cspossurf_[:] = cspossurf_
        self.ce_[:] = ce_
        self.φsneg_[:] = φsneg_
        self.φspos_[:] = φspos_
        self.φe_[:] = φe_
        self.jintneg_[:] = jintneg_
        self.jintpos_[:] = jintpos_
        self.jDLneg_[:] = 0
        self.jDLpos_[:] = 0
        self.i0intneg_[:] = i0intneg_
        self.i0intpos_[:] = i0intpos_
        self.ηintneg_[:] = ηintneg_
        self.ηintpos_[:] = ηintpos_
        if self.lithiumPlating:
            self.jLP_[:] = 0

        self.jneg_[:] = jintneg_
        self.jpos_[:] = jintpos_

        if self.verbose:
            print(f'一致性初始化完成。Newton迭代{nNewton = }。Consistent initial conditions are solved! ')

    def update_Kf__with_pure_electrochemical_parameters(self):
        # 对频域因变量线性矩阵Kf__赋纯电化学参数相关值
        self._update_Kf__REce_REce_and_IMce_IMce(Deeff_ := self.Deeff_, Deeff_)
        self._update_Kf__REce_REj_and_IMce_IMj(self.tplus)
        self._update_Kf__REφsneg_REjneg_and_IMφsneg_IMjneg(σeffneg := self.σeffneg)
        self._update_Kf__REφspos_REjpos_and_IMφspos_IMjpos(σeffpos := self.σeffpos)
        self._update_bKf_REφsneg0_and_REφsposEnd(σeffneg, σeffpos)
        self._update_Kf__REφe_REφe_and_IMφe_IMφe(κeff_ := self.κeff_, κeff_)
        self._update_Kf__REηintneg_REjneg_and_IMηintneg_IMjneg(self.RSEIneg, self.aeffneg)
        self._update_Kf__REηintpos_REjpos_and_IMηintpos_IMjpos(self.RSEIpos, self.aeffpos)
        if self.lithiumPlating:
            self._update_Kf__REηLP_REjneg_and_IMηLP_IMjneg(self.RSEIneg, self.aeffneg)

    def _update_Kf__REce_REj_and_IMce_IMj(self, tplus):
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

    def _update_bKf_REφsneg0_and_REφsposEnd(self, σeffneg, σeffpos):
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
        Rs2 = Rs*Rs
        W2_ = ω_*Rs2/Ds
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
    def solve_REcs__IMcs__(
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

    def solve_frequency_dependent_variables(self) -> dict[str, ndarray]:
        """求解频率相关变量"""
        ω_ = self.ω_
        solve_Kcssurf__ = DFNJTFP2D.solve_Kcssurf___
        CDLneg, CDLpos = self.CDLneg, self.CDLpos
        frequency_dependent_variables = {
            'ωεeΔx__' : outer(ω_, self.εe_*self.Δx_),     # (Nf, Ne) 各频率各控制体的ω*εe*Δx值
            'ωaCDLneg_' : ω_ * (self.aeffneg*CDLneg),     # (Nf,)
            'ωaCDLpos_' : ω_ * (self.aeffpos*CDLpos),     # (Nf,)
            'ωCDLRSEIneg_' : ω_ * (CDLneg*self.RSEIneg),  # (Nf,)
            'ωCDLRSEIpos_' : ω_ * (CDLpos*self.RSEIpos),  # (Nf,)
            'minusKcsnegsurf___' : -solve_Kcssurf__(ω_, self.Rsneg, self.Dsneg, self.aneg),   # (Nf, 2, 2) 负极各频率Kcssurf__矩阵
            'minusKcspossurf___' : -solve_Kcssurf__(ω_, self.Rspos, self.Dspos, self.apos),}  # (Nf, 2, 2) 正极各频率Kcssurf__矩阵
        return frequency_dependent_variables

    def checkEIS(self):
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
        print(f'固相表面浓度解析解方程 cssurf 最大误差{maxError} [mol/m^3]')

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
        print(f'负极固相电势方程 REφsneg IMEφsneg 最大误差{maxError} [A/m^3]')

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
        print(f'正极固相电势方程 REφspos IMEφspos 最大误差{maxError} [A/m^3]')

        i0intneg_, i0intpos_ = self.i0intneg_, self.i0intpos_
        ηintneg_, ηintpos_ = self.ηintneg_, self.ηintpos_
        maxError = max([
            abs(REjintneg__ - 2*aeffneg*(REi0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMjintneg__ - 2*aeffneg*(IMi0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REjintpos__ - 2*aeffpos*(REi0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMjintpos__ - 2*aeffpos*(IMi0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),])
        print(f'主反应BV动力学方程 REjing IMjing 最大误差{maxError} [A/m^3]')

        di0intdceneg_, di0intdcepos_ = self.di0intdceneg_, self.di0intdcepos_
        di0intdcsnegsurf_, di0intdcspossurf_ = self.di0intdcsnegsurf_, self.di0intdcspossurf_
        maxError = max([
            abs(REi0intneg__ - (di0intdceneg_*REce__[:, :Nneg] + di0intdcsnegsurf_*REcsnegsurf__)).max(),
            abs(IMi0intneg__ - (di0intdceneg_*IMce__[:, :Nneg] + di0intdcsnegsurf_*IMcsnegsurf__)).max(),
            abs(REi0intpos__ - (di0intdcepos_*REce__[:, -Npos:] + di0intdcspossurf_*REcspossurf__)).max(),
            abs(IMi0intpos__ - (di0intdcepos_*IMce__[:, -Npos:] + di0intdcspossurf_*IMcspossurf__)).max(), ])
        print(f'主反应交换电流密度方程 REi0int IMi0int 最大误差{maxError} [A/m^2]')

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
        Nneg=9, Nsep=6, Npos=7, Nr=8,
        i0LP=1e-4,
        CDLneg=8, CDLpos=9,
        # Aeffneg=0.5, # Aeffpos=0.4,
        # i0intpos=0.1, i0intneg=0.75,
        lithiumPlating=True,
        # doubleLayerEffect=False,
        # timeDiscretization='backward',
        # radialDiscretization='EI',
        # complete=False,
        # constants=True,
        # verbose=False,
        # decouple=False,
        )

    I = cell.Qcell
    cell.count_lithium()

    thermalModel = True
    cell.EIS()
    # cell.CC(-I, 2300, thermalModel).EIS()
    # cell.CC(I, 2000, thermalModel).EIS()
    # cell.CC(0, 500, thermalModel).EIS()

    # cell.count_lithium()

    # cell.checkEIS()

    plt.close('all')

    '''
    cell.plot_UI()
    cell.plot_TQgen()
    cell.plot_SOC()
    cell.plot_c(np.arange(0, 2001, 200))
    cell.plot_φ(np.arange(0, 2001, 200))
    cell.plot_jint_i0int_ηint(np.arange(0, 2001, 200))
    cell.plot_jDL(np.arange(0, 2001, 200))
    cell.plot_csr(np.arange(0, 2001, 200), 1)
    cell.plot_jLP_ηLP(np.arange(1700, 2301, 100))
    cell.plot_LP()
    cell.plot_OCV_OCP()
    cell.plot_dUOCPdθs()
    cell.plot_nNewton()
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



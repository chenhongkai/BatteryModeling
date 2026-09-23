#%%
import warnings, os
os.environ['NUMBA_CACHE_DIR'] = r'C:\numba_cache'
from typing import Sequence
from functools import partial


import matplotlib.pyplot as plt
from numpy import ndarray, array, zeros, zeros_like, ones, eye, full, empty, empty_like, hstack, stack, concatenate, \
    exp, sqrt, cos, sin, cosh, sinh, arcsinh, outer, \
    isfinite, minimum, maximum
from numpy.linalg import solve
from scipy.linalg.lapack import dgtsv
from numba import njit

from P2Dmodel.P2Dbase import P2Dbase
from P2Dmodel.tools import (diagonalSliceRavel, tridiagonal_matmul,
                            stepping_aware_cached_property,
                            get_color, batch_inv_2x2,
                            F, R)

εi0 = 0.1  # i0计算中的固液相浓度保护阈值 [mol/m^3]

@njit(cache=True, fastmath=True)
def solve_Kcssurf___(
        ω_: ndarray,  # (Nf,) 角频率序列 [rad/s]
        Rs: float,    # 颗粒半径 [m]
        Ds: float,    # 固相扩散系数 [m^2/s]
        a: float,     # 比表面积 [m^2/m^3]
        ) -> ndarray:
    # 求Kcssurf___矩阵
    # Kcssurf___ @ stack([REjint__, IMjint__], axis=1) = stack([REcssurf__, IMcssurf__], axis=1)
    Rs2 = Rs*Rs
    W2_ = ω_*Rs2/Ds
    W_ = sqrt(W2_)
    root2W_ = 1.4142135623730951*W_
    ψ_ = 0.7071067811865476*W_
    cosψ_ = cos(ψ_)
    sinψ_ = sin(ψ_)
    cosψ2_ = cosψ_*cosψ_
    sinψ2_ = sinψ_*sinψ_
    cosψsinψ = cosψ_*sinψ_
    aFDs = a * 96485.33289 * Ds

    # 指数缩放
    # cosh²ψ 和 coshψ·sinhψ 是 ~exp(2ψ) 级别的大数，容易溢出，不能直接算
    # 因此，把 cosh²ψ、coshψ·sinhψ 全部乘 exp(-2ψ)，转化成O(1)级别，防止溢出
    exp_2ψ_ = exp(-2*ψ_)
    m_ = 1 + exp_2ψ_
    coshψ2_s_ = 0.25*(m_*m_)                    # cosh²ψ * exp(-2ψ)
    coshψsinhψ_s_ = 0.25*(1 - exp_2ψ_*exp_2ψ_)  # coshψ·sinhψ * exp(-2ψ)

    a_ = -Rs*root2W_ * (coshψsinhψ_s_ + cosψsinψ * exp_2ψ_) + Rs*2 * (coshψ2_s_ - cosψ2_ * exp_2ψ_)
    b_ = -Rs*root2W_ * (coshψsinhψ_s_ - cosψsinψ * exp_2ψ_)
    d_ = 2*aFDs*((W2_ + 1) * coshψ2_s_
                 - root2W_ * coshψsinhψ_s_
                 - W2_     * sinψ2_   * exp_2ψ_
                 - root2W_ * cosψsinψ * exp_2ψ_
                 -           cosψ2_   * exp_2ψ_)
    a_ /= d_
    b_ /= d_
    Kcssurf___ = empty((ω_.size, 2, 2))
    Kcssurf___[:, 0, 0] = Kcssurf___[:, 1, 1] = a_
    Kcssurf___[:, 0, 1] = b_
    Kcssurf___[:, 1, 0] = -b_
    return Kcssurf___  # (Nf, 2, 2)


@njit(cache=True, fastmath=True)
def solve_frequency_dependent_variables(
        ω_: ndarray, εe_: ndarray, Δx_: ndarray,
        Rsneg: float, Rspos: float,
        Dsneg: float, Dspos: float,
        aneg: float, apos: float) -> tuple[ndarray, ndarray, ndarray]:
    # 求解频率相关变量
    ωεeΔx__ =  outer(ω_, εe_*Δx_)                             # (Nf, Ne) 各频率各控制体的ω*εe*Δx值 [m/s]
    Kcsnegsurf___ = solve_Kcssurf___(ω_, Rsneg, Dsneg, aneg)  # (Nf, 2, 2) 负极各频率Kcssurf__矩阵
    Kcspossurf___ = solve_Kcssurf___(ω_, Rspos, Dspos, apos)  # (Nf, 2, 2) 正极各频率Kcssurf__矩阵
    return Kcsnegsurf___, Kcspossurf___, ωεeΔx__


@njit(cache=True, fastmath=True)
def solve_jint_(T: float, aeff: float, i0int_: ndarray, ηint_: ndarray) -> ndarray:
    # 求解主反应局部体积电流密度jint [A/m^3]
    return 2*aeff*i0int_*sinh(F/(2*R*T)*ηint_)


@njit(cache=True, fastmath=True)
def solve_djintdi0int_(T: float, aeff: float, ηint_: ndarray) -> ndarray:
    # 求解主反应局部体积电流密度jint对交换电流密度i0int的偏导数 [A/m^3 / A/m^2]
    return 2*aeff*sinh(F/(2*R*T)*ηint_)


@njit(cache=True, fastmath=True)
def solve_djintdηint_(T: float, aeff: float, i0int_: ndarray, ηint_: ndarray) -> ndarray:
    # 求解主反应局部体积电流密度jint对过电位ηint的偏导数 [A/m^3 / V]
    FRT = F/(R*T)
    return FRT*aeff*i0int_*cosh(FRT*0.5*ηint_)


@njit(cache=True, fastmath=True)
def solve_i0int_(k: float, csmax: float, cssurf_: ndarray, ce_: ndarray) -> ndarray:
    # 求主反应交换电流密度 [A/m^2]
    cssurfEval_ = maximum(minimum(cssurf_, csmax - εi0), εi0)
    ceEval_ = maximum(ce_, εi0)
    return F * k * sqrt(ceEval_*(csmax - cssurfEval_)*cssurfEval_)


@njit(cache=True, fastmath=True)
def solve_di0intdcssurf_(k: float, csmax: float,
                         cssurf_: ndarray, ce_: ndarray, i0int_: ndarray) -> ndarray:
    # 求解主反应交换电流密度i0int对固相颗粒表面锂离子浓度cssurf的偏导数 [A/m^2 / (mol/m^3)]
    Fk = F*k
    cssurfEval_ = maximum(minimum(cssurf_, csmax - εi0), εi0)
    ceEval_ = maximum(ce_, εi0)
    di0intdcssurf_ = 0.5*Fk*Fk * ceEval_*(csmax - 2*cssurfEval_)/i0int_
    return di0intdcssurf_ * ((cssurf_ > εi0) & (cssurf_ < csmax - εi0))


@njit(cache=True, fastmath=True)
def solve_di0intdce_(ce_: ndarray, i0int_: ndarray):
    # 求解主反应交换电流密度i0int对电解液锂离子浓度ce的偏导数 [A/m^2 / (mol/m^3)]
    ceEval_ = maximum(ce_, εi0)
    return 0.5*i0int_/ceEval_ * (ce_ > εi0)


class DFNJTFP2D(P2Dbase):
    """锂离子电池经典时频联合准二维模型 Doyle-Fuller-Newman Joint Time-Frequency Pseudo-two-Dimensional model"""

    __slots__ = (
        # 专有参数名
        'A', 'Lneg', 'Lsep', 'Lpos', 'εsneg', 'εspos', 'εeneg', 'εesep', 'εepos', 'Rsneg', 'Rspos',
        'bneg', 'bsep', 'bpos',
        '_De', '_κ', 'tplus', 'TDF',
        'csmaxneg', 'csmaxpos', 'ce0',
        'Aeffneg', 'Aeffpos',
        # 专有时域状态量
        'csneg__', 'cspos__', 'csnegsurf_', 'cspossurf_', 'ce_',
        'jintneg_', 'jintpos_', 'jDLneg_', 'jDLpos_',
        'i0intneg_', 'i0intpos_',
        # 专有频域状态量
        'REcsnegsurf__', 'IMcsnegsurf__', 'REcspossurf__', 'IMcspossurf__', 'REce__', 'IMce__',
        'REjintneg__', 'IMjintneg__', 'REjintpos__', 'IMjintpos__',
        'REjDLneg__', 'IMjDLneg__', 'REjDLpos__', 'IMjDLpos__',
        'REi0intneg__', 'IMi0intneg__', 'REi0intpos__', 'IMi0intpos__',
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
            bneg: float = 3.3,       # 负极Bruggeman指数 [–]
            bsep: float = 3.3,       # 隔膜Bruggeman指数 [–]
            bpos: float = 3.3,       # 正极Bruggeman指数 [–]
            Dsneg: float = 3.9e-14,  # 负极固相锂离子扩散系数 [m^2/s]
            Dspos: float = 2.5e-14,  # 正极固相锂离子扩散系数 [m^2/s]
            De: float = 7.5e-11,     # 电解液扩散系数 [m^2/s]
            σneg: float = 100.,      # 负极固相电导率 [S/m]
            σpos: float = 3.8,       # 正极固相电导率 [S/m]
            κ: float = .2,           # 电解液离子电导率 [S/m]
            tplus: float = 0.363,    # 电解液阳离子迁移数 [–]
            TDF: float = 1.,         # 电解液热力学因子1+∂lnf/∂lnce [–]
            kneg: float = 1.948e-11,    # 负极主反应速率常数 [m^2.5/(mol^0.5·s)]
            kpos: float = 2.156e-11,    # 正极主反应速率常数 [m^2.5/(mol^0.5·s)]
            Rfneg: float = 5e-3,      # 负极SEI膜的面积电阻 [Ω·m^2]
            Rfpos: float = 2e-3,      # 正极SEI膜的面积电阻 [Ω·m^2]
            csmaxneg: float = 26390.,   # 负极固相最大锂离子浓度 [mol/m^3]
            csmaxpos: float = 22860.,   # 正极固相最大锂离子浓度 [mol/m^3]
            ce0: float = 2000.,         # 电解液的初始浓度 [mol/m^3]
            CDLneg: float = .8,         # 负极颗粒表面的双电层面积电容 [F/m^2]
            CDLpos: float = .8,         # 正极颗粒表面的双电层面积电容 [F/m^2]
            l: float = 0.,              # 等效电感 [H]
            i0intneg: float | None = None,  # 负极主反应交换电流密度 [A/m^2]
            i0intpos: float | None = None,  # 正极主反应交换电流密度 [A/m^2]
            Aeffneg: float = 1.,        # 负极固相颗粒与电解质的有效接触面积比 [–]
            Aeffpos: float = 1.,        # 正极固相颗粒与电解质的有效接触面积比 [–]
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
        self.bneg = bneg; assert bneg>=1, f'负极Bruggeman指数{bneg = }，应大于或等于1'
        self.bsep = bsep; assert bsep>=1, f'隔膜Bruggeman指数{bsep = }，应大于或等于1'
        self.bpos = bpos; assert bpos>=1, f'正极Bruggeman指数{bpos = }，应大于或等于1'
        self.Dsneg = Dsneg; assert Dsneg>0, f'负极固相的锂离子扩散系数{Dsneg = }，应大于0 [m^2/s]'
        self.Dspos = Dspos; assert Dspos>0, f'正极固相的锂离子扩散系数{Dspos = }，应大于0 [m^2/s]'
        self.De = De;       assert De>0, f'电解液扩散系数{De = }，应大于0 [m^2/s]'
        self.σneg = σneg;   assert σneg>0, f'负极固相电导率{σneg = }，应大于0 [S/m]'
        self.σpos = σpos;   assert σpos>0, f'正极固相电导率{σpos = }，应大于0 [S/m]'
        self.κ = κ;         assert κ>0, f'电解液离子电导率{κ = }，应大于0 [S/m]'
        self.tplus = tplus; assert 0<tplus<1, f'电解液迁移数{tplus = }，取值范围应为(0, 1)'
        self.TDF = TDF;     assert TDF>0, f'热力学因子1 + ∂lnf/∂lnce = {TDF = }，应大于0'
        self.Rfneg = Rfneg; assert Rfneg>=0, f'负极SEI膜的面积电阻{Rfneg = }，应大于或等于0 [Ω·m^2]'
        self.Rfpos = Rfpos; assert Rfpos>=0, f'正极SEI膜的面积电阻{Rfpos = }，应大于或等于0 [Ω·m^2]'
        # 2动力学参数
        self.kneg = kneg; assert kneg>0, f'负极主反应速率常数{kneg = }，应大于0 m^2.5/(mol^0.5·s)'
        self.kpos = kpos; assert kpos>0, f'正极主反应速率常数{kpos = }，应大于0 m^2.5/(mol^0.5·s)'
        # 3浓度参数
        self.csmaxneg = csmaxneg; assert csmaxneg>0, f'负极固相最大锂离子浓度{csmaxneg = }，应大于0 [mol/m^3]'
        self.csmaxpos = csmaxpos; assert csmaxpos>0, f'正极固相最大锂离子浓度{csmaxpos = }，应大于0 [mol/m^3]'
        self.ce0 = ce0; assert ce0>0, f'电解液的初始浓度{ce0 = }，应大于0 [mol/m^3]'
        # 3电抗参数
        self.CDLneg = CDLneg; assert CDLneg>=0, f'负极表面双电层面积电容{CDLneg = }，应大于或等于0 [F/m^2]'
        self.CDLpos = CDLpos; assert CDLpos>=0, f'正极表面双电层面积电容{CDLpos = }，应大于或等于0 [F/m^2]'
        self.l = l; assert l>=0, f'等效电感{l = }，应大于或等于0 [H]'
        # 2交换电流密度
        self.i0intneg = i0intneg; assert (i0intneg is None) or (i0intneg>0), f'负极主反应交换电流密度{i0intneg = }，应大于0 [A/m^2]'
        self.i0intpos = i0intpos; assert (i0intpos is None) or (i0intpos>0), f'正极主反应交换电流密度{i0intpos = }，应大于0 [A/m^2]'
        # 2有效接触面积参数
        self.Aeffneg = Aeffneg; assert 0<Aeffneg<=1, f'负极固相颗粒与电解质的有效接触面积比{Aeffneg = }，取值范围应为(0, 1]'
        self.Aeffpos = Aeffpos; assert 0<Aeffpos<=1, f'正极固相颗粒与电解质的有效接触面积比{Aeffpos = }，取值范围应为(0, 1]'
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
        # DFNJTFP2D专有状态量
        csneg = csmaxneg*(θminneg + SOC0*(θmaxneg - θminneg))  # 初始负极固相锂离子浓度 [mol/m^3]
        cspos = csmaxpos*(θmaxpos + SOC0*(θminpos - θmaxpos))  # 初始正极固相锂离子浓度 [mol/m^3]
        self.csneg__ = full((Nr, Nneg), csneg, float)  # 初始化：负极固相颗粒内部锂离子浓度场 [mol/m^3]
        self.cspos__ = full((Nr, Npos), cspos, float)  # 初始化：正极固相颗粒内部锂离子浓度场 [mol/m^3]
        self.csnegsurf_ = full(Nneg, csneg, float)     # 初始化：负极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.cspossurf_ = full(Npos, cspos, float)     # 初始化：正极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.ce_ = full(Ne, ce0, float)                # 初始化：电解液锂离子浓度场 [mol/m^3]
        self.jintneg_, self.jintpos_ = zeros(Nneg), zeros(Npos)  # 初始化：负极、正极主反应局部体积电流密度场 [A/m^3]
        self.jDLneg_, self.jDLpos_   = zeros(Nneg), zeros(Npos)  # 初始化：负极、正极双电层效应局部体积电流密度场 [A/m^3]
        i0intneg = self.i0intneg if self._i0intneg else DFNJTFP2D.solve_i0int_(self.kneg, csmaxneg, csneg, ce0)
        i0intpos = self.i0intpos if self._i0intpos else DFNJTFP2D.solve_i0int_(self.kpos, csmaxpos, cspos, ce0)
        self.i0intneg_ = full(Nneg, i0intneg, float)  # 初始化：负极主反应交换电流密度场 [A/m^2]
        self.i0intpos_ = full(Npos, i0intpos, float)  # 初始化：正极主反应交换电流密度场 [A/m^2]
        if self.complete:
            # 状态量
            Nf = self.f_.size
            self.REcsnegsurf__, self.IMcsnegsurf__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极固相表面浓度实部、虚部 [mol/m^3]
            self.REcspossurf__, self.IMcspossurf__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极固相表面浓度实部、虚部 [mol/m^3]
            self.REce__, self.IMce__ = empty((Nf, Ne)), empty((Nf, Ne))                  # 电解液锂离子浓度实部、虚部 [mol/m^3]
            self.REjintneg__, self.IMjintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))    # 负极主反应局部体积电流密度实部、虚部 [A/m^3]
            self.REjintpos__, self.IMjintpos__ = empty((Nf, Npos)), empty((Nf, Npos))    # 正极主反应局部体积电流密度实部、虚部 [A/m^3]
            self.REjDLneg__, self.IMjDLneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))      # 负极双电层局部体积电流密度实部、虚部 [A/m^3]
            self.REjDLpos__, self.IMjDLpos__ = empty((Nf, Npos)), empty((Nf, Npos))      # 正极双电层局部体积电流密度实部、虚部 [A/m^3]
            self.REi0intneg__, self.IMi0intneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极交换电流密度实部、虚部 [A/m^2]
            self.REi0intpos__, self.IMi0intpos__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极交换电流密度实部、虚部 [A/m^2]
            extra_datanames_ = [             # 需记录的数据名称
                'csneg__', 'cspos__',        # 负极、正极固相锂离子浓度场 [mol/m^3]
                'csnegsurf_', 'cspossurf_',  # 负极、正极表面锂离子浓度场 [mol/m^3]
                'ce_',                       # 电解液锂离子浓度场 [mol/m^3]
                'jintneg_', 'jintpos_',      # 负极、正极主反应局部体积电流密度场 [A/m^3]
                'jDLneg_', 'jDLpos_',        # 负极、正极双电层效应局部体积电流密度场 [A/m^3]
                'i0intneg_', 'i0intpos_',    # 负极、正极主反应交换电流密度场 [A/m^2]
                'isneg_', 'ispos_', 'ie_',]  # 负极、正极固相电流密度场、电解液电流密度场 [A/m^2]
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
            print(f'{self.__class__.__name__}初始化完成!')

    def _update_K__bK_(self, Δt):
        # 更新K__矩阵和bK_向量

        # 读取模式
        is_CN = self.timeDiscretization == 'CN'

        ## 计算α_, β, γ_
        # cssurf = α + β*jint  -->  jint = cssurf/β - α/β
        Rsneg = self.Rsneg
        Rspos = self.Rspos
        Rsneg2, Rspos2 = Rsneg*Rsneg, Rspos*Rspos
        Δneg = Rsneg - self.Δrneg_[-1]
        Δpos = Rspos - self.Δrpos_[-1]
        Kcsjintneg = Δt*Rsneg2/self.aneg/P2Dbase.F/((Rsneg2*Rsneg - Δneg*Δneg*Δneg)/3)
        Kcsjintpos = Δt*Rspos2/self.apos/P2Dbase.F/((Rspos2*Rspos - Δpos*Δpos*Δpos)/3)
        bandKcsneg__ = (Δt*self.Dsneg)*self.bandKcsneg__
        bandKcspos__ = (Δt*self.Dspos)*self.bandKcspos__
        if is_CN:
            bandKcsneg__ *= .5
            bandKcspos__ *= .5
            Kcsjintneg *= .5
            Kcsjintpos *= .5
            bandBcsneg__ = -bandKcsneg__
            bandBcspos__ = -bandKcspos__
            bandBcsneg__[1] += 1
            bandBcspos__[1] += 1
            RHScsneg__ = tridiagonal_matmul(bandBcsneg__, self.csneg__)
            RHScspos__ = tridiagonal_matmul(bandBcspos__, self.cspos__)
        else:
            RHScsneg__ = self.csneg__
            RHScspos__ = self.cspos__
        bandKcsneg__[1] += 1
        bandKcspos__[1] += 1
        e__ = self.e__
        RHScsneg__ = concatenate((RHScsneg__, e__), axis=1)
        RHScspos__ = concatenate((RHScspos__, e__), axis=1)
        Scsneg__ = dgtsv(bandKcsneg__[2, :-1], bandKcsneg__[1], bandKcsneg__[0, 1:], RHScsneg__, True, True, True, True)[3]
        Scspos__ = dgtsv(bandKcspos__[2, :-1], bandKcspos__[1], bandKcspos__[0, 1:], RHScspos__, True, True, True, True)[3]
        csnegI__ = Scsneg__[:, :-1]  # 负极内部锂离子浓度的历史影响分量
        csposI__ = Scspos__[:, :-1]  # 正极内部锂离子浓度的历史影响分量
        γneg_ = Scsneg__[:, -1] * -Kcsjintneg
        γpos_ = Scspos__[:, -1] * -Kcsjintpos
        coeffsExpl_ = self.coeffsExpl_
        αneg_ = coeffsExpl_.dot(csnegI__[-3:])
        αpos_ = coeffsExpl_.dot(csposI__[-3:])
        βneg = coeffsExpl_.dot(γneg_[-3:])
        βpos = coeffsExpl_.dot(γpos_[-3:])
        if is_CN:
            αneg_ += βneg*self.jintneg_
            αpos_ += βpos*self.jintpos_

        ## 双电层局部体积电流密度jDL显式表达式系数及偏导数
        # jDL = (bjDL + C*(φs - φe - Rf/aeff*jint))/D，其中 D = 1 + C*Rf/aeff
        if self.timeDL:
            data_t_ = self.data['t']
            Δφseneg__ = self.ΔφsenegHistory__
            Δφsepos__ = self.ΔφseposHistory__
            NΔφse = len(Δφseneg__)
            t_1 = data_t_[-1]  # 步初时刻 [s]
            t_2 = t_1
            t_3 = t_1
            Δφseneg_1_ = Δφseneg__[-1]
            Δφsepos_1_ = Δφsepos__[-1]
            Δφseneg_2_ = Δφseneg_1_
            Δφsepos_2_ = Δφsepos_1_
            Δφseneg_3_ = Δφseneg_1_
            Δφsepos_3_ = Δφsepos_1_
            if NΔφse > 1:
                t_2 = data_t_[-2]
                Δφseneg_2_ = Δφseneg__[-2]
                Δφsepos_2_ = Δφsepos__[-2]
            if NΔφse > 2:
                t_3 = data_t_[-3]
                Δφseneg_3_ = Δφseneg__[-3]
                Δφsepos_3_ = Δφsepos__[-3]
        else:
            # 时域关闭双电层：不读取积分历史，以下数组仅用于保持JIT参数形状
            NΔφse = 0
            t_1 = t_2 = t_3 = self.t
            Δφseneg_1_ = Δφseneg_2_ = Δφseneg_3_ = self.φsneg_
            Δφsepos_1_ = Δφsepos_2_ = Δφsepos_3_ = self.φspos_

        (bjDLneg_, bjDLpos_, Cneg, Cpos, Dneg, Dpos,
         djdcsnegsurf, djdcspossurf,
         djDLdφsneg, djDLdφeneg,
         djDLdφspos, djDLdφepos,) = DFNJTFP2D._update_K__bK_JIT(
            # 矩阵
            self.ravelK_, self.bK_, self.sK,
            # 模式
            is_CN,
            # 网格参数
            self.Nneg, self.Nsep, self.Npos, Δt, self.Δx_, self.ΔxWest_, self.ΔxEast_,
            # 电化学参数
            self.σeffneg, self.σeffpos, self.Rfneg, self.Rfpos,
            self.aeffneg, self.aeffpos, self.CDLneg, self.CDLpos,
            self.Deeff_, self.κeff_, self.εe_, self.tplus,
            # 状态
            self.i, self.ce_, self.jneg_, self.jpos_,
            # 双电层历史
            NΔφse, t_1, t_2, t_3,
            Δφseneg_1_, Δφseneg_2_, Δφseneg_3_,
            Δφsepos_1_, Δφsepos_2_, Δφsepos_3_,
            # 固相扩散消元结果
            αneg_, βneg, αpos_, βpos,)

        return (
            csnegI__, αneg_, βneg, γneg_,
            csposI__, αpos_, βpos, γpos_,
            bjDLneg_, bjDLpos_, Cneg, Cpos, Dneg, Dpos,
            djdcsnegsurf, djdcspossurf,
            djDLdφsneg, djDLdφeneg,
            djDLdφspos, djDLdφepos,)

    @staticmethod
    @njit(cache=True)
    def _update_K__bK_JIT(
            # 矩阵
            ravelK_, bK_, sK,
            # 模式
            is_CN,
            # 网格参数
            Nneg, Nsep, Npos, Δt, Δx_, ΔxWest_, ΔxEast_,
            # 电化学参数
            σeffneg, σeffpos, Rfneg, Rfpos,
            aeffneg, aeffpos, CDLneg, CDLpos,
            Deeff_, κeff_, εe_, tplus,
            # 状态
            i, ce_, jneg_, jpos_,
            # 双电层历史
            NΔφse, t_1, t_2, t_3,
            Δφseneg_1_, Δφseneg_2_, Δφseneg_3_,
            Δφsepos_1_, Δφsepos_2_, Δφsepos_3_,
            # 固相扩散消元结果
            αneg_, βneg, αpos_, βpos,
            ):
        # 更新K__矩阵和bK_向量

        # cssurf = α + β*jint  -->  jint = 1/β*cssurf - α/β
        djintdcsnegsurf = 1/βneg
        djintdcspossurf = 1/βpos
        jintneg_const_ = αneg_/-βneg
        jintpos_const_ = αpos_/-βpos

        ## 双电层局部体积电流密度jDL显式表达式系数及偏导数
        # jDL = (bjDL + C*(φs - φe - Rf/aeff*jint))/D，其中 D = 1 + C*Rf/aeff
        Rf2aeffneg = Rfneg/aeffneg
        Rf2aeffpos = Rfpos/aeffpos
        if NΔφse == 0:  # timeDL=False，双电层电流及其偏导数为零
            Cneg = Cpos = 0.
            Dneg = Dpos = 1.
            bjDLneg_ = zeros(Nneg)
            bjDLpos_ = zeros(Npos)
        else:
            t = t_1 + Δt  # 步后时刻 [s]
            c = 1/Δt
            if NΔφse > 1:
                c += 1/(t - t_2)
            if NΔφse > 2:
                c += 1/(t - t_3)
            Cneg = aeffneg*CDLneg*c
            Cpos = aeffpos*CDLpos*c
            Dneg = 1 + Cneg*Rf2aeffneg
            Dpos = 1 + Cpos*Rf2aeffpos
            if NΔφse > 2:
                A = (t - t_2)*(t - t_3)/-Δt/(t_1 - t_2)/(t_1 - t_3)
                B = Δt*(t - t_3)/(t_2 - t)/(t_2 - t_1)/(t_2 - t_3)
                C = Δt*(t - t_2)/(t_3 - t)/(t_3 - t_1)/(t_3 - t_2)
                bjDLneg_ = aeffneg*CDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_ + C*Δφseneg_3_)
                bjDLpos_ = aeffpos*CDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_ + C*Δφsepos_3_)
            elif NΔφse == 2:
                A = (t - t_2)/(-Δt*(t_1 - t_2))
                B = Δt/((t_2 - t)*(t_2 - t_1))
                bjDLneg_ = aeffneg*CDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_)
                bjDLpos_ = aeffpos*CDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_)
            else:
                bjDLneg_ = -Cneg*Δφseneg_1_
                bjDLpos_ = -Cpos*Δφsepos_1_
        djDLdφsneg = Cneg/Dneg
        djDLdφeneg = -djDLdφsneg
        djDLdjintneg = djDLdφeneg*Rf2aeffneg
        djDLdφspos = Cpos/Dpos
        djDLdφepos = -djDLdφspos
        djDLdjintpos = djDLdφepos*Rf2aeffpos
        djDLdcsnegsurf = djDLdjintneg*djintdcsnegsurf
        djDLdcspossurf = djDLdjintpos*djintdcspossurf
        jDLneg_const_ = bjDLneg_/Dneg + djDLdjintneg*jintneg_const_
        jDLpos_const_ = bjDLpos_/Dpos + djDLdjintpos*jintpos_const_

        # 被消去的反应源项在各方程中的线性系数
        Kcej = -Δt*(1 - tplus)/F
        if is_CN:
            Kcej *= .5

        # 读取网格
        Δxneg, Δxpos = Δx_[0], Δx_[-1]
        Kφsnegj = -Δxneg*Δxneg/σeffneg
        Kφsposj = -Δxpos*Δxpos/σeffpos
        Kφenegj = Δxneg
        Kφeposj = Δxpos

        djdcsnegsurf = djintdcsnegsurf + djDLdcsnegsurf
        djdcspossurf = djintdcspossurf + djDLdcspossurf
        jneg_const_ = jintneg_const_ + jDLneg_const_
        jpos_const_ = jintpos_const_ + jDLpos_const_

        ## 赋值K__矩阵
        # ce行cssurf列
        ravelK_[sK.sr_ceneg_csnegsurf] = Kcej*djdcsnegsurf
        ravelK_[sK.sr_cepos_cspossurf] = Kcej*djdcspossurf
        # ce行ce列
        s_ce = sK.s_ce
        sr_ce_ce = sK.sr_ce_ce
        dl_ = ravelK_[sK.sr_ce_ce_l]  # 下对角线
        du_ = ravelK_[sK.sr_ce_ce_u]  # 上对角线
        d_ = ravelK_[sr_ce_ce]        # 主对角线
        dl_[:] = -Deeff_[1:]  / ΔxWest_[1:]
        du_[:] = -Deeff_[:-1] / ΔxEast_[:-1]
        d_[0] = -du_[0]
        d_[1:-1] = -(dl_[:-1] + du_[1:])
        d_[-1] = -dl_[-1]
        NK1 = bK_.size + 1
        start0 = sr_ce_ce.start
        for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NK1
            nrE = start0 + nE*NK1
            a, c = -Deeff_[nW]/ΔxWest_[nW], -2*Deeff_[nW]*Deeff_[nE]/(Deeff_[nW]*Δx_[nE] + Deeff_[nE]*Δx_[nW])
            ravelK_[nrW-1:nrW+2] = a, -(a + c), c
            a, c = c, -Deeff_[nE]/ΔxEast_[nE]
            ravelK_[nrE-1:nrE+2] = a, -(a + c), c
        Δt2Δx_ = Δt/Δx_
        dl_ *= Δt2Δx_[1:]
        du_ *= Δt2Δx_[:-1]
        d_  *= Δt2Δx_
        if is_CN:
            dl_ *= .5
            du_ *= .5
            d_ *= .5
            # ce行CN格式历史项，等价于Bce__ @ ce_
            bKce_ = bK_[s_ce]
            bKce_[:] = εe_*ce_
            bKce_[0] -= d_[0]*ce_[0] + du_[0]*ce_[1]
            bKce_[1:-1] -= dl_[:-1]*ce_[:-2] + d_[1:-1]*ce_[1:-1] + du_[1:]*ce_[2:]
            bKce_[-1] -= dl_[-1]*ce_[-2] + d_[-1]*ce_[-1]
        d_ += εe_
        # ce行φs列
        ravelK_[sK.sr_ceneg_φsneg] = Kcej*djDLdφsneg
        ravelK_[sK.sr_cepos_φspos] = Kcej*djDLdφspos
        # ce行φe列
        ravelK_[sK.sr_ceneg_φeneg] = Kcej*djDLdφeneg
        ravelK_[sK.sr_cepos_φepos] = Kcej*djDLdφepos

        # φs行cssurf列
        ravelK_[sK.sr_φsneg_csnegsurf] = Kφsnegj*djdcsnegsurf
        ravelK_[sK.sr_φspos_cspossurf] = Kφsposj*djdcspossurf
        # φs行φs列
        dφsneg_ = ravelK_[sK.sr_φsneg_φsneg]
        dφspos_ = ravelK_[sK.sr_φspos_φspos]
        dφsneg_[0] = -1 + Kφsnegj*djDLdφsneg
        dφsneg_[1:-1] = -2 + Kφsnegj*djDLdφsneg
        dφsneg_[-1] = -1 + Kφsnegj*djDLdφsneg
        dφspos_[0] = -1 + Kφsposj*djDLdφspos
        dφspos_[1:-1] = -2 + Kφsposj*djDLdφspos
        dφspos_[-1] = -1 + Kφsposj*djDLdφspos
        # φs行φe列
        ravelK_[sK.sr_φsneg_φeneg] = Kφsnegj*djDLdφeneg
        ravelK_[sK.sr_φspos_φepos] = Kφsposj*djDLdφepos

        # φe行cssurf列
        ravelK_[sK.sr_φeneg_csnegsurf] = Kφenegj*djdcsnegsurf
        ravelK_[sK.sr_φepos_cspossurf] = Kφeposj*djdcspossurf
        # φe行φs列
        ravelK_[sK.sr_φeneg_φsneg] = Kφenegj*djDLdφsneg
        ravelK_[sK.sr_φepos_φspos] = Kφeposj*djDLdφspos
        # φe行φe列
        dl_ = ravelK_[sK.sr_φe_φe_l]
        du_ = ravelK_[sK.sr_φe_φe_u]
        sr_φe_φe = sK.sr_φe_φe
        d_ = ravelK_[sr_φe_φe]  # 主对角线
        dl_[:] = κeff_[1:] / ΔxWest_[1:]
        du_[:] = κeff_[:-1] / ΔxEast_[:-1]
        d_[0] = -du_[0]
        d_[1:-1] = -(dl_[:-1] + du_[1:])
        d_[-1] = -dl_[-1]
        d_[0] -= κeff_[0]/(0.5*Δx_[0])
        start0 = sr_φe_φe.start
        for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NK1
            nrE = start0 + nE*NK1
            a, c = κeff_[nW]/ΔxWest_[nW], 2*κeff_[nW]*κeff_[nE]/(κeff_[nW]*Δx_[nE] + κeff_[nE]*Δx_[nW])
            ravelK_[nrW-1:nrW+2] = a, -(a + c), c
            a, c = c, κeff_[nE]/ΔxEast_[nE]
            ravelK_[nrE-1:nrE+2] = a, -(a + c), c
        d_[:Nneg]  += Kφenegj*djDLdφeneg
        d_[-Npos:] += Kφeposj*djDLdφepos

        ## 更新bK_
        # cssurf行
        bK_[sK.s_csnegsurf] = αneg_
        bK_[sK.s_cspossurf] = αpos_
        # ce行
        bKce_ = bK_[s_ce]
        if is_CN:
            bKce_[:Nneg]  -= Kcej*jneg_
            bKce_[-Npos:] -= Kcej*jpos_
        else:
            bKce_[:] = εe_*ce_
        bKce_[:Nneg]  -= Kcej*jneg_const_
        bKce_[-Npos:] -= Kcej*jpos_const_
        # φs行
        bKφsneg_ = bK_[sK.s_φsneg]
        bKφspos_ = bK_[sK.s_φspos]
        bKφsneg_[:] = -Kφsnegj*jneg_const_
        bKφspos_[:] = -Kφsposj*jpos_const_
        bKφsneg_[0]  += -Δxneg*i/σeffneg
        bKφspos_[-1] +=  Δxpos*i/σeffpos  # 固相电流边界条件
        # φe行
        bKφe_ = bK_[sK.s_φe]
        bKφe_[:Nneg]  = -Kφenegj*jneg_const_
        bKφe_[-Npos:] = -Kφeposj*jpos_const_

        return (
            bjDLneg_, bjDLpos_,
            Cneg, Cpos, Dneg, Dpos,
            djdcsnegsurf,
            djdcspossurf,
            djDLdφsneg, djDLdφeneg,
            djDLdφspos, djDLdφepos,)

    def _stepping(self, Δt):
        # 时间步进：Newton迭代

        # 读取矩阵
        K__ = self.ravelK_.base  # 读取：因变量线性矩阵K__
        bK_ = self.bK_           # 读取：常数项向量，F_ = K__ @ X_ - bK_

        # 读取索引
        sK = self.sK
        s_c = sK.s_c
        s_φ = sK.s_φ

        # 读取方法
        solve_banded_matrix = DFNJTFP2D.solve_banded_matrix
        solve_i0int_ = DFNJTFP2D.solve_i0int_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_
        solve_dUOCPdθsneg_ = self.solve_dUOCPdθsneg_
        solve_dUOCPdθspos_ = self.solve_dUOCPdθspos_

        # 读取参数
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_ = self.Δx_
        ΔxWest_, ΔxEast_ = self.ΔxWest_, self.ΔxEast_
        aeffneg, aeffpos = self.aeffneg, self.aeffpos
        csmaxneg, csmaxpos = self.csmaxneg, self.csmaxpos
        Rf2aeffneg = self.Rfneg/aeffneg
        Rf2aeffpos = self.Rfpos/aeffpos
        DeeffWest_ = DeeffEast_ = self.Deeff_
        κeffWest_ = κeffEast_ = self.κeff_
        κDeffWest_ = κDeffEast_ = self.κDeff_
        if i0intnegUnknown := (self._i0intneg is None):
            kneg = self.kneg
        else:
            kneg = 0.
            i0intneg = self.i0intneg
        if i0intposUnknown := (self._i0intpos is None):
            kpos = self.kpos
        else:
            kpos = 0.
            i0intpos = self.i0intpos

        # 读取状态
        I = self.I    # 电流 [A]
        i = I/self.A  # 电流密度 [A/m^2]
        T = self.T    # 温度 [K]
        Ihistory_ = self.Ihistory_  # 历史4电流序列

        # 更新K__矩阵、bK_向量
        (csnegI__, αneg_, βneg, γneg_,
        csposI__, αpos_, βpos, γpos_,
        bjDLneg_, bjDLpos_, Cneg, Cpos, Dneg, Dpos,
        djdcsnegsurf, djdcspossurf,
        djDLdφsneg, djDLdφeneg,
        djDLdφspos, djDLdφepos,
        ) = self._update_K__bK_(Δt)
        dηintdφsneg = 1 - Rf2aeffneg*djDLdφsneg
        dηintdφspos = 1 - Rf2aeffpos*djDLdφspos
        dηintdφeneg = -1 - Rf2aeffneg*djDLdφeneg
        dηintdφepos = -1 - Rf2aeffpos*djDLdφepos

        # 初始化解X_，用切片索引绑定因变量
        X_ = zeros_like(bK_)
        csnegsurf_ = X_[sK.s_csnegsurf]
        cspossurf_ = X_[sK.s_cspossurf]
        ce_ = X_[sK.s_ce]
        ceneg_ = X_[sK.s_ceneg]
        cepos_ = X_[sK.s_cepos]
        φsneg_ = X_[sK.s_φsneg]
        φspos_ = X_[sK.s_φspos]
        φe_ = X_[sK.s_φe]
        φeneg_ = X_[sK.s_φeneg]
        φepos_ = X_[sK.s_φepos]

        # 对X_赋初值
        jintneg0_ = self.jintneg_ if I==Ihistory_[-1] else (i/self.Lneg)
        jintpos0_ = self.jintpos_ if I==Ihistory_[-1] else (-i/self.Lpos)
        csnegsurf_[:] = αneg_ + βneg*jintneg0_
        cspossurf_[:] = αpos_ + βpos*jintpos0_
        ce_[:] = self.ce_
        i0intneg_ = self.i0intneg_ if i0intnegUnknown else i0intneg
        i0intpos_ = self.i0intpos_ if i0intposUnknown else i0intpos
        if I==Ihistory_[-1]:
            # 恒电流
            φsneg_[:] = self.φsneg_
            φspos_[:] = self.φspos_
            φe_[:] = self.φe_
        else:
            # 变电流瞬间
            F2RT = 0.5*P2Dbase.F/(P2Dbase.R*T)
            ηintneg0_ = arcsinh(jintneg0_/(2*aeffneg*i0intneg_))/F2RT
            ηintpos0_ = arcsinh(jintpos0_/(2*aeffpos*i0intpos_))/F2RT
            φsneg_[:] = ηintneg0_ + Rf2aeffneg*jintneg0_ + solve_UOCPneg_(csnegsurf_/csmaxneg)
            φspos_[:] = ηintpos0_ + Rf2aeffpos*jintpos0_ + solve_UOCPpos_(cspossurf_/csmaxpos)

        ## Newton迭代变量预备
        J__ = K__.copy()       # (NK, NK) 初始化Jacobi矩阵
        ravelJ_ = J__.ravel()  # (NK*NK,) Jacobi矩阵展平视图
        F_ = empty_like(bK_)   # (NK,) F残差向量
        κDeff2ΔxWest_ = κDeffWest_[1:] / ΔxWest_[1:]
        κDeff2ΔxEast_ = κDeffEast_[:-1] / ΔxEast_[:-1]
        _update_F_J__nonlinear_terms_JIT = DFNJTFP2D._update_F_J__nonlinear_terms_JIT

        for nNewton in range(1, 201):
            ## Newton迭代

            # 更新4派生因变量
            jintneg_ = (csnegsurf_ - αneg_)/βneg  # 由cssurf反算jint
            jintpos_ = (cspossurf_ - αpos_)/βpos
            jDLneg_ = (bjDLneg_ + Cneg*(φsneg_ - φeneg_) - Cneg*Rf2aeffneg*jintneg_)/Dneg
            jDLpos_ = (bjDLpos_ + Cpos*(φspos_ - φepos_) - Cpos*Rf2aeffpos*jintpos_)/Dpos
            if i0intnegUnknown:
                i0intneg_ = solve_i0int_(kneg, csmaxneg, csnegsurf_, ceneg_)
            if i0intposUnknown:
                i0intpos_ = solve_i0int_(kpos, csmaxpos, cspossurf_, cepos_)
            ηintneg_ = φsneg_ - φeneg_ - solve_UOCPneg_(csnegsurf_/csmaxneg) - Rf2aeffneg*(jintneg_ + jDLneg_)
            ηintpos_ = φspos_ - φepos_ - solve_UOCPpos_(cspossurf_/csmaxpos) - Rf2aeffpos*(jintpos_ + jDLpos_)

            # 更新F向量线性部分
            F_[:] = K__.dot(X_) - bK_

            # 更新F_向量非线性部分、J__矩阵非线性部分
            _update_F_J__nonlinear_terms_JIT(
                # 矩阵及其索引
                F_, ravelJ_, sK,
                # 网格参数
                Nneg, Nsep, Δx_, ΔxWest_, ΔxEast_,
                # 模式
                i0intnegUnknown, i0intposUnknown,
                # 电化学参数
                T,
                aeffneg, aeffpos,
                kneg, kpos,
                csmaxneg, csmaxpos,
                Rf2aeffneg, Rf2aeffpos,
                κDeffWest_, κDeffEast_, κeffWest_, κeffEast_, DeeffWest_, DeeffEast_,
                κDeff2ΔxWest_, κDeff2ΔxEast_,
                # 固相扩散消元结果
                βneg, βpos,
                # 因变量及其偏导数
                csnegsurf_, cspossurf_, ce_, ceneg_, cepos_,
                i0intneg_, i0intpos_, ηintneg_, ηintpos_,
                djdcsnegsurf, djdcspossurf,
                dηintdφsneg, dηintdφeneg, dηintdφspos, dηintdφepos,
                solve_dUOCPdθsneg_(csnegsurf_/csmaxneg)/csmaxneg,
                solve_dUOCPdθspos_(cspossurf_/csmaxpos)/csmaxpos)

            if (self.banded_experience_of_J__ is None) and any(Ihistory_):
                self.banded_experience_of_J__ = self.banded_J__(J__)

            # Newton迭代新解向量
            if expe := self.banded_experience_of_J__:
                # 带状化求解
                ΔX_ = solve_banded_matrix(J__, F_, **expe)
            else:
                # 直接求解
                ΔX_ = solve(J__, F_)

            X_ -= ΔX_

            if not isfinite(X_).all():
                return nNewton, False, 'X_中出现nan或inf'
            if (ce_ <= 0).any():
                return nNewton, False, 'ce<=0'
            if (csnegsurf_ <= 0).any():
                return nNewton, False, 'csnegsurf<=0'
            if (csnegsurf_ >= csmaxneg).any():
                return nNewton, False, 'csnegsurf>=csmaxneg'
            if (cspossurf_ <= 0).any():
                return nNewton, False, 'cspossurf<=0'
            if (cspossurf_ >= csmaxpos).any():
                return nNewton, False, 'cspossurf>=csmaxpos'

            ΔX_ = abs(ΔX_)
            maxΔc = ΔX_[s_c].max()
            maxΔφ = ΔX_[s_φ].max()
            if maxΔc < 2e-1 and maxΔφ < 2e-4:
                break
        else:
            t = self.t
            return nNewton, False, f't = {t} -> {t + Δt} s，Newton迭代达最大次数{nNewton}，{maxΔc = :.4f} mol/m^3，{maxΔφ = :.6f} V'

        # Newton迭代收敛，更新状态量
        jintneg_ = (csnegsurf_ - αneg_)/βneg
        jintpos_ = (cspossurf_ - αpos_)/βpos
        match self.timeDiscretization:
            case 'CN':
                self.csneg__[:] = csnegI__ + outer(γneg_, jintneg_ + self.jintneg_)
                self.cspos__[:] = csposI__ + outer(γpos_, jintpos_ + self.jintpos_)
            case 'backward':
                self.csneg__[:] = csnegI__ + outer(γneg_, jintneg_)
                self.cspos__[:] = csposI__ + outer(γpos_, jintpos_)
        self.csnegsurf_[:] = csnegsurf_
        self.cspossurf_[:] = cspossurf_
        self.ce_[:] = ce_
        self.φe_[:] = φe_
        self.φsneg_[:] = φsneg_
        self.φspos_[:] = φspos_
        self.jintneg_[:] = jintneg_
        self.jintpos_[:] = jintpos_
        self.jDLneg_[:] = jDLneg_ = (bjDLneg_ + Cneg*(φsneg_ - φeneg_) - Cneg*Rf2aeffneg*jintneg_)/Dneg
        self.jDLpos_[:] = jDLpos_ = (bjDLpos_ + Cpos*(φspos_ - φepos_) - Cpos*Rf2aeffpos*jintpos_)/Dpos
        self.i0intneg_[:] = solve_i0int_(kneg, csmaxneg, csnegsurf_, ceneg_) if i0intnegUnknown else i0intneg_
        self.i0intpos_[:] = solve_i0int_(kpos, csmaxpos, cspossurf_, cepos_) if i0intposUnknown else i0intpos_
        self.ηintneg_[:] = φsneg_ - φeneg_ - solve_UOCPneg_(csnegsurf_/csmaxneg) - Rf2aeffneg*(jintneg_ + jDLneg_)
        self.ηintpos_[:] = φspos_ - φepos_ - solve_UOCPpos_(cspossurf_/csmaxpos) - Rf2aeffpos*(jintpos_ + jDLpos_)
        return nNewton, True, None

    @staticmethod
    @njit(cache=True)
    def _update_F_J__nonlinear_terms_JIT(
            # 矩阵及其索引
            F_, ravelJ_, sK,
            # 网格参数
            Nneg, Nsep, Δx_, ΔxWest_, ΔxEast_,
            # 模式
            i0intnegUnknown, i0intposUnknown,
            # 电化学参数
            T, aeffneg, aeffpos, kneg, kpos, csmaxneg, csmaxpos,
            Rf2aeffneg, Rf2aeffpos,
            κDeffWest_, κDeffEast_, κeffWest_, κeffEast_, DeeffWest_, DeeffEast_,
            κDeff2ΔxWest_, κDeff2ΔxEast_,
            # 固相扩散消元结果
            βneg, βpos,
            # 因变量及其导数
            csnegsurf_, cspossurf_, ce_, ceneg_, cepos_,
            i0intneg_, i0intpos_, ηintneg_, ηintpos_,
            djdcsnegsurf, djdcspossurf,
            dηintdφsneg, dηintdφeneg, dηintdφspos, dηintdφepos,
            dUOCPdcsnegsurf_, dUOCPdcspossurf_):
        # 更新F_残差向量非线性部分、Jacobi矩阵J__非线性部分

        ## 更新F_向量非线性部分
        # cssurf行
        F_[sK.s_csnegsurf] -= βneg*solve_jint_(T, aeffneg, i0intneg_, ηintneg_)
        F_[sK.s_cspossurf] -= βpos*solve_jint_(T, aeffpos, i0intpos_, ηintpos_)

        # φe行
        ΔFφe_ = zeros(ce_.size)
        ceW_ = ce_[:-1]
        ceE_ = ce_[1:]
        ceM_ = 0.5*(ceE_ + ceW_)  # (Ne-1,) 相邻浓度均值
        q_ = (ceE_ - ceW_)/ceM_   # (Ne-1,)
        a_ = κDeff2ΔxWest_*q_
        c_ = κDeff2ΔxEast_*q_
        ΔFφe_[:-1] -= c_
        ΔFφe_[1:]  += a_
        for nW, nE in zip((Nneg - 1, Nneg + Nsep - 1), (Nneg, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            a = DeeffWest_[nE]*Δx_[nW]
            b = DeeffEast_[nW]*Δx_[nE]
            cInterface = (a*ce_[nE] + b*ce_[nW])/(a + b)
            ΔFφe_[nW] = ( κDeffWest_[nW] * (ce_[nW] - ce_[nW-1])/ΔxWest_[nW]/(0.5*(ce_[nW] + ce_[nW-1]))
                         -κDeffEast_[nW] * (cInterface - ce_[nW])/(0.5*Δx_[nW])/cInterface)
            ΔFφe_[nE] = ( κDeffWest_[nE] * (ce_[nE] - cInterface)/(0.5*Δx_[nE])/cInterface
                         -κDeffEast_[nE] * (ce_[nE+1] - ce_[nE])/ΔxEast_[nE]/(0.5*(ce_[nE+1] + ce_[nE])))

        F_[sK.s_φe] += ΔFφe_

        ## 更新J__矩阵非线性部分
        # cssurf行
        djintdi0intneg_ = solve_djintdi0int_(T, aeffneg, ηintneg_)
        djintdi0intpos_ = solve_djintdi0int_(T, aeffpos, ηintpos_)
        djintdηintneg_ = solve_djintdηint_(T, aeffneg, i0intneg_, ηintneg_)
        djintdηintpos_ = solve_djintdηint_(T, aeffpos, i0intpos_, ηintpos_)
        dηintdcsnegsurf_ = -dUOCPdcsnegsurf_ - Rf2aeffneg*djdcsnegsurf
        dηintdcspossurf_ = -dUOCPdcspossurf_ - Rf2aeffpos*djdcspossurf
        if i0intnegUnknown:
            di0intdcsnegsurf_ = solve_di0intdcssurf_(kneg, csmaxneg, csnegsurf_, ceneg_, i0intneg_)
            di0intdceneg_     = solve_di0intdce_(ceneg_, i0intneg_)
        else:
            di0intdcsnegsurf_ = di0intdceneg_ = zeros(Nneg)
        ravelJ_[sK.sr_csnegsurf_csnegsurf] = 1 - βneg*(djintdi0intneg_*di0intdcsnegsurf_ + djintdηintneg_*dηintdcsnegsurf_)
        ravelJ_[sK.sr_csnegsurf_ceneg] = -βneg*djintdi0intneg_*di0intdceneg_
        ravelJ_[sK.sr_csnegsurf_φsneg] = -βneg*dηintdφsneg*djintdηintneg_
        ravelJ_[sK.sr_csnegsurf_φeneg] = -βneg*dηintdφeneg*djintdηintneg_
        if i0intposUnknown:
            di0intdcspossurf_ = solve_di0intdcssurf_(kpos, csmaxpos, cspossurf_, cepos_, i0intpos_)
            di0intdcepos_     = solve_di0intdce_(cepos_, i0intpos_)
        else:
            di0intdcspossurf_ = di0intdcepos_ = zeros(cspossurf_.size)
        ravelJ_[sK.sr_cspossurf_cspossurf] = 1 - βpos*(djintdi0intpos_*di0intdcspossurf_ + djintdηintpos_*dηintdcspossurf_)
        ravelJ_[sK.sr_cspossurf_cepos] = -βpos*djintdi0intpos_*di0intdcepos_
        ravelJ_[sK.sr_cspossurf_φspos] = -βpos*dηintdφspos*djintdηintpos_
        ravelJ_[sK.sr_cspossurf_φepos] = -βpos*dηintdφepos*djintdηintpos_

        # φe行ce列
        q_ *= 0.5
        a_ = κDeff2ΔxWest_/ceM_
        aa_ = a_*q_
        c_ = κDeff2ΔxEast_/ceM_
        cc_ = c_*q_
        ravelJ_[sK.sr_φe_ce_l] = -aa_ - a_  # 下对角线
        ravelJ_[sK.sr_φe_ce_u] = cc_ - c_   # 上对角线
        sr_φe_ce = sK.sr_φe_ce
        ravelJ_φe_ce_ = ravelJ_[sr_φe_ce]  # 主对角线
        ravelJ_φe_ce_[:] = 0.
        ravelJ_φe_ce_[:-1] += cc_ + c_
        ravelJ_φe_ce_[1:]  += a_ - aa_
        start0φece = sr_φe_ce.start
        NJ1 = F_.size + 1
        for nW, nE in zip((Nneg - 1, Nneg + Nsep - 1), (Nneg, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0φece + nW*NJ1
            nrE = start0φece + nE*NJ1
            κDeffEast_nW_DeeffWest_nE = κDeffEast_[nW]*DeeffWest_[nE]
            κDeffWest_nE_DeeffEast_nW = κDeffWest_[nE]*DeeffEast_[nW]
            num = κDeffEast_nW_DeeffWest_nE - κDeffWest_nE_DeeffEast_nW
            κeffWest_nE_Δx_nW = κeffWest_[nE]*Δx_[nW]
            κeffEast_nW_Δx_nE = κeffEast_[nW]*Δx_[nE]
            DeeffWest_nE_Δx_nW = DeeffWest_[nE]*Δx_[nW]
            DeeffEast_nW_Δx_nE = DeeffEast_[nW]*Δx_[nE]
            den1 = κeffWest_nE_Δx_nW + κeffEast_nW_Δx_nE
            den2 = DeeffWest_nE_Δx_nW*ce_[nE] + DeeffEast_nW_Δx_nE*ce_[nW]
            quotient = num/(den1*den2)
            ΔceEW = ce_[nE] - ce_[nW]
            SceW = ce_[nW] + ce_[nW-1]
            SceE = ce_[nE] + ce_[nE+1]

            coeff = ΔceEW*DeeffWest_nE_Δx_nW/den2
            a = 2*κDeffWest_[nW]/(SceW*Δx_[nW])
            aa = a*(ce_[nW] - ce_[nW-1])/SceW
            c = 2*κeffEast_nW_Δx_nE*quotient
            cc = c*coeff
            d = 2*κDeffEast_nW_DeeffWest_nE/den2
            dd = d*coeff
            p = DeeffEast_nW_Δx_nE/DeeffWest_nE_Δx_nW
            ravelJ_[nrW-1:nrW+2] = -a - aa, -c - cc*p + d + dd*p + a - aa, c - cc - d + dd

            coeff = ΔceEW*DeeffEast_nW_Δx_nE/den2
            a = 2*κeffWest_nE_Δx_nW*quotient
            aa = a*coeff
            c = 2*κDeffWest_nE_DeeffEast_nW/den2
            cc = c*coeff
            d = 2*κDeffEast_[nE]/(SceE*Δx_[nE])
            dd = d*(ce_[nE] - ce_[nE+1])/SceE
            p = DeeffWest_nE_Δx_nW/DeeffEast_nW_Δx_nE
            ravelJ_[nrE-1:nrE+2] = -a - aa - c - cc, a - aa*p + c - cc*p + d - dd, -d - dd

    def count_lithium(self):
        """统计锂电荷量"""
        qsneg = self.θsneg*self.Qneg
        qspos = self.θspos*self.Qpos
        qe = (self.ce_*self.Δx_*self.A*self.εe_).sum()*P2Dbase.F/3600
        print(f'合计锂电荷总量{qsneg + qspos + qe:.8g} Ah = '
              f'负极嵌锂{qsneg:.8g} Ah + 正极嵌锂{qspos:.8g} Ah + '
              f'电解液锂{qe:.8g} Ah')

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
    def De(self):
        """电解液锂离子扩散系数 [m^2/s]"""
        return self.Arrhenius(self._De, self.EDe)
    @De.setter
    def De(self, De):
        self._De = De

    @property
    def Deeff_(self):
        """(Ne,) 全区域各控制体电解液有效扩散系数 [m^2/s]"""
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
        """(Ne,) 全区域各控制体电解液有效扩散离子电导率 [A/m]"""
        return 2*P2Dbase.R*self.T*(1 - self.tplus)/P2Dbase.F*self.TDF * self.κeff_

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
        """单位负极体积的固相颗粒比表面积 [m^2/m^3]"""
        return 3/self.Rsneg*self.εsneg

    @property
    def apos(self):
        """单位正极体积的固相颗粒比表面积 [m^2/m^3]"""
        return 3/self.Rspos*self.εspos

    @property
    def aeffneg(self):
        """单位负极体积的固相颗粒有效接触比表面积 [m^2/m^3]"""
        return self.aneg*self.Aeffneg

    @property
    def aeffpos(self):
        """单位正极体积的固相颗粒有效接触比表面积 [m^2/m^3]"""
        return self.apos*self.Aeffpos

    @property
    def εe_(self):
        """(Ne,) 全区域各控制体电解液体积分数 [–]"""
        return concatenate([
            full(self.Nneg, self.εeneg),
            full(self.Nsep, self.εesep),
            full(self.Npos, self.εepos),])

    @property
    def εeb_(self):
        """(Ne,) 全区域各控制体电解液体积分数的Bruggeman指数次幂 [–]"""
        return concatenate([
            full(self.Nneg, self.εeneg**self.bneg),
            full(self.Nsep, self.εesep**self.bsep),
            full(self.Npos, self.εepos**self.bpos), ])

    @property
    def U(self):
        """全电池端电压 [V]"""
        a = 0.5*self.I/self.A
        φsposCollector = self.φspos_[-1] - a*self.Δxpos/self.σeffpos
        φsnegCollector = self.φsneg_[0]  + a*self.Δxneg/self.σeffneg
        return φsposCollector - φsnegCollector + self.Ul

    @property
    def i(self):
        """电流密度 [A/m^2]"""
        return self.I/self.A

    @property
    def ceneg_(self):
        """(Nneg,) 负极区域电解液锂离子浓度 [mol/m^3]"""
        return self.ce_[:self.Nneg]

    def solve_csnegsurf_(self, csneg__):
        """求解负极固相表面锂离子浓度场 [mol/m^3]"""
        return self.coeffsExpl_.dot(csneg__[-3:])

    def solve_cspossurf_(self, cspos__):
        """求解正极固相表面锂离子浓度场 [mol/m^3]"""
        return self.coeffsExpl_.dot(cspos__[-3:])

    @property
    def cepos_(self):
        """(Npos,) 正极区域电解液锂离子浓度 [mol/m^3]"""
        return self.ce_[-self.Npos:]

    @stepping_aware_cached_property
    def ceInterfaces_(self):
        """(Ne+1,)各控制体界面的锂离子浓度 [mol/m^3]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_ = self.Δx_
        ce_ = self.ce_
        DeeffWest_ = DeeffEast_ = self.Deeff_
        ceInterfaces_ = hstack([ce_[0], (ce_[:-1] + ce_[1:])*0.5, ce_[-1]])
        for (nW, nE) in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            a, b = DeeffWest_[nE]*Δx_[nW], DeeffEast_[nW]*Δx_[nE]
            ceInterfaces_[nW+1] = (a*ce_[nE] + b*ce_[nW])/(a + b)
        return ceInterfaces_

    @stepping_aware_cached_property
    def φeInterfaces_(self):
        """(Ne+1,) 各控制体界面的电解液电势 [V]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_, ΔxWest_, ΔxEast_ = self.Δx_, self.ΔxWest_, self.ΔxEast_
        φe_, ce_ = self.φe_, self.ce_
        ceInterfaces_ = self.ceInterfaces_
        ceWest_ = ceInterfaces_[:-1]    # (Ne,) 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        ceEast_ = ceInterfaces_[1:]     # (Ne,) 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradceWest_ = self.gradceWest_  # (Ne,) 各控制体左界面的锂离子浓度梯度 [mol/m^4]
        gradceEast_ = self.gradceEast_  # (Ne,) 各控制体右界面的锂离子浓度梯度 [mol/m^4]
        gradlnceWest_ = gradceWest_/ceWest_  # (Ne,) 各控制体左界面的对数锂离子浓度梯度 [1/m]
        gradlnceEast_ = gradceEast_/ceEast_  # (Ne,) 各控制体右界面的对数锂离子浓度梯度 [1/m]
        φeInterfaces_ = hstack([φe_[0], (φe_[:-1] + φe_[1:])*0.5, φe_[-1]])
        κeffWest_  = κeffEast_ = self.κeff_
        κDeffWest_ = κDeffEast_ = self.κDeff_
        for nW, nE in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            a, b = κeffEast_[nW]*Δx_[nE], κeffWest_[nE]*Δx_[nW]
            c = 0.5*Δx_[nW]*Δx_[nE]
            φeInterfaces_[nE] = (  a*φe_[nW] + b*φe_[nE]
                                 + c*κDeffEast_[nW]*gradlnceEast_[nW]
                                 - c*κDeffWest_[nE]*gradlnceWest_[nE]
                                 )/(a + b)
        return φeInterfaces_

    # 保留类级调用接口，实际公式使用模块级njit函数，便于JIT kernel直接调用
    solve_jint_ = staticmethod(solve_jint_)

    @stepping_aware_cached_property
    def djintdi0intneg_(self):
        """负极主反应局部体积电流密度jintneg对交换电流密度i0intneg的偏导数 [A/m^3 / A/m^2]"""
        return DFNJTFP2D.solve_djintdi0int_(self.T, self.aeffneg, self.ηintneg_)

    @stepping_aware_cached_property
    def djintdi0intpos_(self):
        """正极主反应局部体积电流密度jintpos对交换电流密度i0intpos的偏导数 [A/m^3 / A/m^2]"""
        return DFNJTFP2D.solve_djintdi0int_(self.T, self.aeffpos, self.ηintpos_)

    solve_djintdi0int_ = staticmethod(solve_djintdi0int_)

    @stepping_aware_cached_property
    def djintdηintneg_(self):
        """负极主反应局部体积电流密度jintneg对过电位ηintneg的偏导数 [A/m^3 / V]"""
        return DFNJTFP2D.solve_djintdηint_(self.T, self.aeffneg, self.i0intneg_, self.ηintneg_)

    @stepping_aware_cached_property
    def djintdηintpos_(self):
        """正极主反应局部体积电流密度jintpos对过电位ηintpos的偏导数 [A/m^3 / V]"""
        return DFNJTFP2D.solve_djintdηint_(self.T, self.aeffpos, self.i0intpos_, self.ηintpos_)

    solve_djintdηint_ = staticmethod(solve_djintdηint_)

    solve_i0int_ = staticmethod(solve_i0int_)

    @stepping_aware_cached_property
    def di0intdcsnegsurf_(self):
        """负极主反应交换电流密度i0intneg对电极表面浓度csnegsurf的偏导数 [A/m^2 / (mol/m^3)]"""
        return 0 if self._i0intneg\
            else DFNJTFP2D.solve_di0intdcssurf_(
            self.kneg, self.csmaxneg, self.csnegsurf_, self.ceneg_, self.i0intneg_)

    @stepping_aware_cached_property
    def di0intdcspossurf_(self):
        """正极主反应交换电流密度i0intpos对电极表面浓度cspossurf的偏导数 [A/m^2 / (mol/m^3)]"""
        return 0 if self._i0intpos\
            else DFNJTFP2D.solve_di0intdcssurf_(
            self.kpos, self.csmaxpos, self.cspossurf_, self.cepos_, self.i0intpos_)

    solve_di0intdcssurf_ = staticmethod(solve_di0intdcssurf_)

    @stepping_aware_cached_property
    def di0intdceneg_(self):
        """负极主反应交换电流密度i0int对电解液浓度ce的偏导数 [A/m^2 / (mol/m^3)]"""
        return 0 if self._i0intneg \
            else DFNJTFP2D.solve_di0intdce_(self.ceneg_, self.i0intneg_)

    @stepping_aware_cached_property
    def di0intdcepos_(self):
        """正极主反应交换电流密度i0int对电解液浓度ce的偏导数 [A/m^2 / (mol/m^3)]"""
        return 0 if self._i0intpos \
            else DFNJTFP2D.solve_di0intdce_(self.cepos_, self.i0intpos_)

    solve_di0intdce_ = staticmethod(solve_di0intdce_)

    @property
    def ηLPneg_(self):
        """负极析锂反应过电位场 [V]"""
        return self.φsneg_ - self.φeneg_ - self.Rfneg/self.aeffneg * self.jneg_

    @property
    def ηLPpos_(self):
        """正极析锂反应过电位场 [V]"""
        return self.φspos_ - self.φepos_ - self.Rfpos/self.aeffpos * self.jpos_

    @stepping_aware_cached_property
    def jneg_(self):
        """负极总局部体积电流密度场 [A/m^3]"""
        return self.jintneg_ + self.jDLneg_

    @stepping_aware_cached_property
    def jpos_(self):
        """正极总局部体积电流密度场 [A/m^3]"""
        return self.jintpos_ + self.jDLpos_

    @stepping_aware_cached_property
    def dUOCPdcsnegsurf_(self):
        """负极开路电位UOCPnegsurf对负极表面锂离子浓度cssurf的导数 [V/(mol/m^3)]"""
        csmaxneg = self.csmaxneg
        return self.solve_dUOCPdθsneg_(self.csnegsurf_/csmaxneg) / csmaxneg

    @stepping_aware_cached_property
    def dUOCPdcspossurf_(self):
        """正极开路电位UOCPpossurf对正极表面锂离子浓度cssurf的导数 [V/(mol/m^3)]"""
        csmaxpos = self.csmaxpos
        return self.solve_dUOCPdθspos_(self.cspossurf_/csmaxpos) / csmaxpos

    @stepping_aware_cached_property
    def gradlnce_(self):
        """对数电解液锂离子浓度场的梯度 [1/m]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_, ΔxWest_, ΔxEast_ = self.Δx_, self.ΔxWest_, self.ΔxEast_
        x_ = self.x_
        ce_ = self.ce_
        ceInterfaces_ = self.ceInterfaces_
        ceWest_ = ceInterfaces_[:-1]  # 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        ceEast_ = ceInterfaces_[1:]   # 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradce_ = hstack([
            (ce_[1] - ce_[0])/(x_[1] - x_[0])*0.5,       # 负极首个控制体
            (ce_[2:] - ce_[:-2])/(x_[2:] - x_[:-2]),         # 内部控制体
            (ce_[-1] - ce_[-2])/(x_[-1] - x_[-2])*0.5])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            gradce_[nW] = ((ce_[nW] - ce_[nW - 1])/ΔxWest_[nW] + (ceEast_[nW] - ce_[nW])/(0.5*Δx_[nW])) * 0.5  # 界面左侧控制体
            gradce_[nE] = ((ce_[nE] - ceWest_[nE])/(0.5*Δx_[nE]) + (ce_[nE + 1] - ce_[nE])/ΔxEast_[nE]) * 0.5  # 界面右侧控制体
        return gradce_/ce_

    @stepping_aware_cached_property
    def gradceWest_(self):
        """(Ne,) 各控制体左侧界面电解液锂离子浓度梯度 [mol/m^4]"""
        ce_ = self.ce_
        ceWest_ = self.ceInterfaces_[:-1]
        Nneg = self.Nneg
        gradceWest_ = hstack([0, (ce_[1:] - ce_[:-1])/self.ΔxWest_[1:]])
        # 修正负极-隔膜、隔膜-正极界面
        idx_ = [Nneg, Nneg + self.Nsep]
        gradceWest_[idx_] = (ce_[idx_] - ceWest_[idx_])/(0.5*self.Δx_[idx_])
        return gradceWest_

    @stepping_aware_cached_property
    def gradceEast_(self):
        """(Ne,) 各控制体右侧界面电解液锂离子浓度梯度 [mol/m^4]"""
        ce_ = self.ce_
        ceEast_ = self.ceInterfaces_[1:]
        Nneg = self.Nneg
        gradceEast_ = hstack([(ce_[1:] - ce_[:-1])/self.ΔxEast_[:-1], 0])
        # 修正负极-隔膜、隔膜-正极界面
        idx_ = [Nneg - 1, Nneg + self.Nsep - 1]
        gradceEast_[idx_] = (ceEast_[idx_] - ce_[idx_])/(0.5*self.Δx_[idx_])
        return gradceEast_

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
            # 修正负极-隔膜、隔膜-正极界面
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
    def Qe(self):
        """电解液欧姆热功率 [W]"""
        gradφe_ = self.gradφe_
        return self.A*((self.κeff_*gradφe_ - self.κDeff_*self.gradlnce_)*gradφe_*self.Δx_).sum()

    @property
    def Qohmneg(self):
        """负极固相欧姆热功率 [W]"""
        grad_ = self.gradφsneg_
        return self.A*self.σeffneg*(grad_*grad_).sum()*self.Δxneg

    @property
    def Qohmpos(self):
        """正极固相欧姆热功率 [W]"""
        grad_ = self.gradφspos_
        return self.A*self.σeffpos*(grad_*grad_).sum()*self.Δxpos

    @property
    def Qrxnneg(self):
        """负极反应热功率 [W]"""
        return self.A*(self.jintneg_*self.ηintneg_).sum()*self.Δxneg

    @property
    def Qrxnpos(self):
        """正极反应热功率 [W]"""
        return self.A*(self.jintpos_*self.ηintpos_).sum()*self.Δxpos

    @property
    def QSEIneg(self):
        """负极SEI膜电阻热功率 [W]"""
        jneg_ = self.jneg_
        return self.A*self.Rfneg/self.aeffneg*(jneg_*jneg_).sum()*self.Δxneg

    @property
    def QSEIpos(self):
        """正极SEI膜电阻热功率 [W]"""
        jpos_ = self.jpos_
        return self.A*self.Rfpos/self.aeffpos*(jpos_*jpos_).sum()*self.Δxpos

    @property
    def Qrevneg(self):
        """负极可逆热功率 [W]"""
        dUOCPdTnegsurf_ = self.dUOCPdTneg(self.csnegsurf_/self.csmaxneg) if callable(self.dUOCPdTneg) else self.dUOCPdTneg
        return self.A*(self.T*dUOCPdTnegsurf_*self.jintneg_).sum()*self.Δxneg

    @property
    def Qrevpos(self):
        """正极可逆热功率 [W]"""
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
        """电解液电流密度场 [A/m^2]"""
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
        isneg__ = self('isneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极固相电流密度场 [A/m^2]
        ispos__ = self('ispos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极固相电流密度场 [A/m^2]
        ie__ = self('ie_', t_=t_, x_=self.x_)  # 呈时间序列的电解液电流密度场 [A/m^2]
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
            I: float | int = 0,
            T: float | int = 298.15):
        # 一致性初始化
        # 已知：csneg__、cspos__、ce_、I、T
        # 基本因变量：φsneg_、φspos_、φe_
        # 派生量：csnegsurf_、cspossurf_、jintneg_、jintpos_、i0intneg_、i0intpos_、ηintneg_、ηintpos_
        # 令：jDLneg_ = jDLpos_ = 0，jint_由固相电势方程显式反算
        self.T = T; assert T>0, f'温度{T = }，应大于0 [K]'
        Nr, Nneg, Nsep, Npos, Ne = self.Nr, self.Nneg, self.Nsep, self.Npos, self.Ne  # 读取：网格数
        assert csneg__.shape==(Nr, Nneg), f'负极固相颗粒内部锂离子浓度csneg__.shape应为({Nr}, {Nneg})'
        assert cspos__.shape==(Nr, Npos), f'正极固相颗粒内部锂离子浓度cspos__.shape应为({Nr}, {Npos})'
        assert ce_.shape==(Ne,), f'电解液锂离子浓度ce_.shape应为({Ne},)'
        assert ((0<=csneg__) & (csneg__<=self.csmaxneg)).all(), 'csneg__取值范围应为[0, csmaxneg] [mol/m^3]'
        assert ((0<=cspos__) & (cspos__<=self.csmaxpos)).all(), 'cspos__取值范围应为[0, csmaxpos] [mol/m^3]'
        assert (0<ce_).all(), 'ce_取值应大于0 [mol/m^3]'

        # 外推表面浓度
        c_ = self.coeffsExpl_
        csnegsurf_ = c_.dot(csneg__[-3:])
        cspossurf_ = c_.dot(cspos__[-3:])
        csmaxneg, csmaxpos = self.csmaxneg, self.csmaxpos
        if (csnegsurf_<=0).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到csnegsurf<=0')
        if (csnegsurf_>=csmaxneg).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到csnegsurf>=csmaxneg')
        if (cspossurf_<=0).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到cspossurf<=0')
        if (cspossurf_>=csmaxpos).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到cspossurf>=csmaxpos')

        # 负极、正极电解液锂离子浓度
        ceneg_, cepos_ = ce_[:Nneg], ce_[-Npos:]

        # 生成局部一致性初始化矩阵切片，只分配3个基本因变量
        s_φsneg = slice(0, Nneg)
        s_φspos = slice(s_φsneg.stop, s_φsneg.stop + Npos)
        s_φe    = slice(s_φspos.stop, s_φspos.stop + Ne)
        s_φeneg = slice(s_φe.start, s_φe.start + Nneg)
        s_φepos = slice(s_φe.stop - Npos, s_φe.stop)
        NKinit = s_φe.stop
        dsr = partial(diagonalSliceRavel, NKinit)
        sr_φsneg_φsneg = dsr(s_φsneg, s_φsneg)
        sr_φspos_φspos = dsr(s_φspos, s_φspos)
        sr_φsneg_φeneg = dsr(s_φsneg, s_φeneg)
        sr_φspos_φepos = dsr(s_φspos, s_φepos)
        sr_φe_φe = dsr(s_φe, s_φe)

        # 读取方法
        solve_jint_ = DFNJTFP2D.solve_jint_
        solve_djintdηint_ = DFNJTFP2D.solve_djintdηint_
        solve_i0int_ = DFNJTFP2D.solve_i0int_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_

        # 读取参数
        aeffneg, aeffpos = self.aeffneg, self.aeffpos
        Rf2aeffneg = self.Rfneg/aeffneg
        Rf2aeffpos = self.Rfpos/aeffpos
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        T = self.T
        F2RT = P2Dbase.F/(2*P2Dbase.R*T)
        σeffneg, σeffpos = self.σeffneg, self.σeffpos
        κeff_ = self.κeff_
        κDeff_ = self.κDeff_
        Deeff_ = self.Deeff_
        if i0intnegUnknown := (self._i0intneg is None):
            kneg = self.kneg          # 读取：负极主反应速率常数 [m^2.5/(mol^0.5·s)]
        else:
            i0intneg = self.i0intneg  # 读取：负极主反应交换电流密度 [A/m^2]
        if i0intposUnknown := (self._i0intpos is None):
            kpos = self.kpos          # 读取：正极主反应速率常数 [m^2.5/(mol^0.5·s)]
        else:
            i0intpos = self.i0intpos  # 读取：正极主反应交换电流密度 [A/m^2]

        # 显式计算主反应交换电流密度
        i0intneg_ = solve_i0int_(kneg, csmaxneg, csnegsurf_, ceneg_) if i0intnegUnknown else i0intneg
        i0intpos_ = solve_i0int_(kpos, csmaxpos, cspossurf_, cepos_) if i0intposUnknown else i0intpos

        # 初始化Kinit__矩阵、bKinit_向量
        Kinit__ = zeros((NKinit, NKinit))
        bKinit_ = zeros(NKinit)
        ravelKinit_ = Kinit__.ravel()

        ## 赋值Kinit__矩阵
        # φs行φs列：常数
        ravelKinit_[sr_φsneg_φsneg] = [-1] + [-2]*(Nneg - 2) + [-1]
        ravelKinit_[sr_φspos_φspos] = [-1] + [-2]*(Npos - 2) + [-1]
        ravelKinit_[dsr(s_φsneg, s_φsneg, -1)] = ravelKinit_[dsr(s_φsneg, s_φsneg, 1)] = 1
        ravelKinit_[dsr(s_φspos, s_φspos, -1)] = ravelKinit_[dsr(s_φspos, s_φspos, 1)] = 1
        Kφsnegj = -Δxneg*Δxneg/σeffneg
        Kφsposj = -Δxpos*Δxpos/σeffpos
        # φe行φe列
        dl_ = ravelKinit_[dsr(s_φe, s_φe, -1)]
        du_ = ravelKinit_[dsr(s_φe, s_φe, 1)]
        d_ = ravelKinit_[sr_φe_φe]
        dl_[:] = κeff_[1:] / ΔxWest_[1:]
        du_[:] = κeff_[:-1] / ΔxEast_[:-1]
        d_[:] = -(hstack([0, dl_]) + hstack([du_, 0]))
        d_[0] -= κeff_[0]/(0.5*Δx_[0])  # 首元占优，固定电解液电势参考
        start0 = sr_φe_φe.start
        NKinit1 = NKinit + 1
        for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NKinit1
            nrE = start0 + nE*NKinit1
            a, c = κeff_[nW]/ΔxWest_[nW], 2*κeff_[nW]*κeff_[nE]/(κeff_[nW]*Δx_[nE] + κeff_[nE]*Δx_[nW])
            ravelKinit_[nrW-1:nrW+2] = a, -(a + c), c
            a, c = c, κeff_[nE]/ΔxEast_[nE]
            ravelKinit_[nrE-1:nrE+2] = a, -(a + c), c
        # φe行φs列
        Kinit__[s_φeneg, s_φsneg] = -Δxneg/Kφsnegj*Kinit__[s_φsneg, s_φsneg]
        Kinit__[s_φepos, s_φspos] = -Δxpos/Kφsposj*Kinit__[s_φspos, s_φspos]

        # 赋值bKinit_向量
        # φs行
        i = I/self.A  # 电极电流密度 [A/m^2]
        bKinit_[s_φsneg.start]    = -Δxneg*i/σeffneg  # 固相电流边界条件
        bKinit_[s_φspos.stop - 1] =  Δxpos*i/σeffpos
        # φe行
        q_ = 2*(ce_[1:] - ce_[:-1])/(ce_[1:] + ce_[:-1])  # (Ne-1,)
        c_ = κDeff_[:-1]*q_ / ΔxEast_[:-1]  # (Ne-1,)
        a_ = κDeff_[1:] *q_ / ΔxWest_[1:]   # (Ne-1,)
        bKinit_[s_φe] = hstack([c_, 0]) - hstack([0, a_])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            a, b = Deeff_[nE]*Δx_[nW], Deeff_[nW]*Δx_[nE]
            cInterface = (a*ce_[nE] + b*ce_[nW])/(a + b)
            bKinit_[s_φe.start + nW] = (κDeff_[nW] * (cInterface - ce_[nW]) / (0.5*Δx_[nW]) / cInterface
                                      - κDeff_[nW] * (ce_[nW] - ce_[nW-1])  / ΔxWest_[nW]   / (0.5*(ce_[nW] + ce_[nW-1])))
            bKinit_[s_φe.start + nE] = (κDeff_[nE] * (ce_[nE+1] - ce_[nE])  / ΔxEast_[nE]   / (0.5*(ce_[nE+1] + ce_[nE]))
                                      - κDeff_[nE] * (ce_[nE] - cInterface) / (0.5*Δx_[nE]) / cInterface)
        bKinit_[s_φeneg.start]    += -Δxneg/Kφsnegj*bKinit_[s_φsneg.start]
        bKinit_[s_φepos.stop - 1] += -Δxpos/Kφsposj*bKinit_[s_φspos.stop - 1]

        ## Newton迭代初值 ##
        X_ = zeros(NKinit)
        # 索引解向量
        φsneg_ = X_[s_φsneg]
        φspos_ = X_[s_φspos]
        φe_ = X_[s_φe]
        φeneg_ = X_[s_φeneg]
        φepos_ = X_[s_φepos]

        # 初始化解向量
        jintneg0 = i/self.Lneg
        jintpos0 = -i/self.Lpos
        ηintneg0_ = arcsinh(jintneg0/(2*aeffneg*i0intneg_))/F2RT
        ηintpos0_ = arcsinh(jintpos0/(2*aeffpos*i0intpos_))/F2RT
        φsneg_[:] = ηintneg0_ + Rf2aeffneg*jintneg0 + solve_UOCPneg_(csnegsurf_/csmaxneg)
        φspos_[:] = ηintpos0_ + Rf2aeffpos*jintpos0 + solve_UOCPpos_(cspossurf_/csmaxpos)

        # Newton迭代
        Kinit_φsneg_φsneg__ = Kinit__[s_φsneg, s_φsneg]
        Kinit_φspos_φspos__ = Kinit__[s_φspos, s_φspos]
        F_ = zeros(NKinit)    # F残差向量
        J__ = Kinit__.copy()  # Jacobi矩阵
        ravelJ_ = J__.ravel()
        J_φsneg_φsneg__ = J__[s_φsneg, s_φsneg]
        J_φspos_φspos__ = J__[s_φspos, s_φspos]

        for nNewton in range(1, 201):
            ## Newton迭代
            # F向量线性部分
            F_[:] = Kinit__.dot(X_) - bKinit_

            # 更新派生因变量
            jintneg_ = F_[s_φsneg]/-Kφsnegj
            jintpos_ = F_[s_φspos]/-Kφsposj
            ηintneg_ = φsneg_ - φeneg_ - solve_UOCPneg_(csnegsurf_/csmaxneg) - Rf2aeffneg*jintneg_
            ηintpos_ = φspos_ - φepos_ - solve_UOCPpos_(cspossurf_/csmaxpos) - Rf2aeffpos*jintpos_
            djintdηintneg_ = solve_djintdηint_(T, aeffneg, i0intneg_, ηintneg_)
            djintdηintpos_ = solve_djintdηint_(T, aeffpos, i0intpos_, ηintpos_)

            # F向量非线性部分
            # φs行：由固相电势方程反算的jint应满足BV方程
            F_[s_φsneg] += Kφsnegj*solve_jint_(T, aeffneg, i0intneg_, ηintneg_)
            F_[s_φspos] += Kφsposj*solve_jint_(T, aeffpos, i0intpos_, ηintpos_)

            # 更新J__矩阵非线性部分
            J_φsneg_φsneg__[:] = Kinit_φsneg_φsneg__ + Rf2aeffneg*djintdηintneg_[:, None]*Kinit_φsneg_φsneg__
            J_φspos_φspos__[:] = Kinit_φspos_φspos__ + Rf2aeffpos*djintdηintpos_[:, None]*Kinit_φspos_φspos__
            ravelJ_[sr_φsneg_φsneg] += Kφsnegj*djintdηintneg_
            ravelJ_[sr_φspos_φspos] += Kφsposj*djintdηintpos_
            ravelJ_[sr_φsneg_φeneg] = -Kφsnegj*djintdηintneg_
            ravelJ_[sr_φspos_φepos] = -Kφsposj*djintdηintpos_

            ΔX_ = solve(J__, F_)
            X_ -= ΔX_

            if (maxΔX := abs(ΔX_).max())<1e-5:
                break
        else:
            raise P2Dbase.Error(f'一致性初始化失败，Newton迭代{nNewton = }次，不收敛，{maxΔX = }')

        # Newton迭代收敛，更新派生因变量
        F_[:] = Kinit__.dot(X_) - bKinit_

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
        self.jintneg_[:] = jintneg_ = F_[s_φsneg]/-Kφsnegj
        self.jintpos_[:] = jintpos_ = F_[s_φspos]/-Kφsposj
        self.jDLneg_[:] = 0
        self.jDLpos_[:] = 0
        self.i0intneg_[:] = i0intneg_
        self.i0intpos_[:] = i0intpos_
        self.ηintneg_[:] = φsneg_ - φeneg_ - solve_UOCPneg_(csnegsurf_/csmaxneg) - Rf2aeffneg*jintneg_
        self.ηintpos_[:] = φspos_ - φepos_ - solve_UOCPpos_(cspossurf_/csmaxpos) - Rf2aeffpos*jintpos_

        if self.verbose:
            print(f'一致性初始化完成。Newton迭代{nNewton = }。Consistent initial conditions are solved! ')

    def _update_Kf__bKf_with_pure_parameters(self):
        # 更新Kf__矩阵的纯电化学参数相关项
        NKf1 = self.bKf_.size + 1
        DFNJTFP2D._update_Kf__bKf_with_pure_parameters_JIT(
            # 矩阵
            self.ravelKf_, self.bKf_, self.sKf, NKf1,
            # 网格参数
            self.Nneg, self.Nsep, self.ΔxWest_, self.ΔxEast_, self.Δx_,
            # 电化学参数
            self.ΔiAC, self.σeffneg, self.σeffpos, self.Deeff_,)

    @staticmethod
    @njit(cache=True)
    def _update_Kf__bKf_with_pure_parameters_JIT(
            # 矩阵
            ravelKf_, bKf_, sKf, NKf1,
            # 网格参数
            Nneg, Nsep, ΔxWest_, ΔxEast_, Δx_,
            # 电化学参数
            ΔiAC, σeffneg, σeffpos, Deeff_,
            ):
        # 更新Kf__矩阵的纯电化学参数相关项

        # REce行REce列
        dl_ = ravelKf_[sKf.sr_REce_REce_l]
        du_ = ravelKf_[sKf.sr_REce_REce_u]
        sr_REce_REce = sKf.sr_REce_REce
        d_  = ravelKf_[sr_REce_REce]
        dl_[:] = -Deeff_[1:] /ΔxWest_[1:]
        du_[:] = -Deeff_[:-1]/ΔxEast_[:-1]
        d_[0] = -du_[0]
        d_[1:-1] = -(dl_[:-1] + du_[1:])
        d_[-1] = -dl_[-1]
        start0 = sr_REce_REce.start
        for (nW, nE) in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NKf1
            nrE = start0 + nE*NKf1
            a, c = -Deeff_[nW]/ΔxWest_[nW], -2*Deeff_[nW]*Deeff_[nE]/(Deeff_[nE]*Δx_[nW] + Deeff_[nW]*Δx_[nE])
            ravelKf_[nrW-1:nrW+2] = a, -(a + c), c
            a, c = c, -Deeff_[nE]/ΔxEast_[nE]
            ravelKf_[nrE-1:nrE+2] = a, -(a + c), c
        # IMce行IMce列
        ravelKf_[sKf.sr_IMce_IMce] = d_
        ravelKf_[sKf.sr_IMce_IMce_l] = dl_
        ravelKf_[sKf.sr_IMce_IMce_u] = du_

        # 更新bKf_向量
        Δxneg, Δxpos = Δx_[0], Δx_[-1]
        bKf_[sKf.s_REφsneg.start]    = -Δxneg*ΔiAC/σeffneg
        bKf_[sKf.s_REφspos.stop - 1] =  Δxpos*ΔiAC/σeffpos

    def _update_Kf__with_states(self):
        # 更新Kf__矩阵的状态（含“参数+状态”）相关项
        NKf1 = self.bKf_.size + 1
        DFNJTFP2D._update_Kf__with_states_JIT(
            # 矩阵
            self.ravelKf_, self.sKf, NKf1,
            # 网格参数
            self.Nneg, self.Nsep, self.ΔxWest_, self.ΔxEast_, self.Δx_,
            # 电化学参数
            self.Deeff_, self.κDeff_,
            # 状态及其偏导数
            self.ceInterfaces_, self.gradceWest_, self.gradceEast_,
            self.djintdi0intneg_, self.di0intdceneg_,
            self.djintdi0intpos_, self.di0intdcepos_,)

    @staticmethod
    @njit(cache=True)
    def _update_Kf__with_states_JIT(
            # 矩阵
            ravelKf_, sKf, NKf1,
            # 网格参数
            Nneg, Nsep, ΔxWest_, ΔxEast_, Δx_,
            # 电化学参数
            Deeff_, κDeff_,
            # 状态及其偏导数
            ceInterfaces_, gradceWest_, gradceEast_,
            djintdi0intneg_, di0intdceneg_,
            djintdi0intpos_, di0intdcepos_,):
        # 更新Kf__矩阵的状态（含“参数+状态”）相关项

        # 读取状态及其偏导数
        ceWest_ = ceInterfaces_[:-1]
        ceEast_ = ceInterfaces_[1:]
        dREjintBVdREceneg_ = djintdi0intneg_*di0intdceneg_
        dIMjintBVdIMceneg_ = dREjintBVdREceneg_
        dREjintBVdREcepos_ = djintdi0intpos_*di0intdcepos_
        dIMjintBVdIMcepos_ = dREjintBVdREcepos_

        # REcssurf行REce列
        ravelKf_[sKf.sr_REcsnegsurf_REceneg] = -dREjintBVdREceneg_
        ravelKf_[sKf.sr_REcspossurf_REcepos] = -dREjintBVdREcepos_
        # IMcssurf行IMce列
        ravelKf_[sKf.sr_IMcsnegsurf_IMceneg] = -dIMjintBVdIMceneg_
        ravelKf_[sKf.sr_IMcspossurf_IMcepos] = -dIMjintBVdIMcepos_
        # REφe行REce列
        sr_REφe_REce = sKf.sr_REφe_REce
        d_REφe_REce_  = ravelKf_[sr_REφe_REce]
        dl_REφe_REce_ = ravelKf_[sKf.sr_REφe_REce_l]
        du_REφe_REce_ = ravelKf_[sKf.sr_REφe_REce_u]

        # 读取参数
        κDeff2ceWest_ = κDeff_[1:]/ceWest_[1:]
        κDeff2ceEast_ = κDeff_[:-1]/ceEast_[:-1]
        a_ = κDeff2ceWest_/ΔxWest_[1:]
        c_ = κDeff2ceEast_/ΔxEast_[:-1]
        aa_ = κDeff2ceWest_*gradceWest_[1:]/ceWest_[1:]*0.5
        cc_ = κDeff2ceEast_*gradceEast_[:-1]/ceEast_[:-1]*0.5
        d_REφe_REce_[0]    = c_[0] + cc_[0]
        d_REφe_REce_[1:-1] = a_[:-1] - aa_[:-1] + c_[1:] + cc_[1:]
        d_REφe_REce_[-1]   = a_[-1] - aa_[-1]
        dl_REφe_REce_[:] = -aa_ - a_
        du_REφe_REce_[:] = cc_ - c_
        start0 = sr_REφe_REce.start
        for (nW, nE) in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NKf1
            nrE = start0 + nE*NKf1
            Deeff_nE_Δx_nW = Deeff_[nE]*Δx_[nW]
            Deeff_nW_Δx_nE = Deeff_[nW]*Δx_[nE]
            den = Deeff_nE_Δx_nW + Deeff_nW_Δx_nE
            pDW = Deeff_nW_Δx_nE/den
            pDE = 1 - pDW
            κDeff2ceWest = κDeff_[nW]/ceWest_[nW]
            κDeff2ceEast = κDeff_[nW]/ceEast_[nW]
            a = κDeff2ceWest/ΔxWest_[nW]
            aa = κDeff2ceWest*gradceWest_[nW]/ceWest_[nW]*0.5
            c = κDeff2ceEast*Deeff_[nE]/den*2
            cc = κDeff2ceEast*gradceEast_[nW]/ceEast_[nW]
            ravelKf_[nrW-1:nrW+2] = -aa - a, a - aa + c + cc*pDW, cc*pDE - c
            κDeff2ceWest = κDeff_[nE]/ceWest_[nE]
            κDeff2ceEast = κDeff_[nE]/ceEast_[nE]
            a = κDeff2ceWest*Deeff_[nW]/den*2
            aa = κDeff2ceWest*gradceWest_[nE]/ceWest_[nE]
            c = κDeff2ceEast/ΔxEast_[nE]
            cc = κDeff2ceEast*gradceEast_[nE]/ceEast_[nE]*0.5
            ravelKf_[nrE-1:nrE+2] = -a - aa*pDW, a - aa*pDE + c + cc, cc - c
        # IMφe行IMce列
        ravelKf_[sKf.sr_IMφe_IMce] = d_REφe_REce_
        ravelKf_[sKf.sr_IMφe_IMce_l] = dl_REφe_REce_
        ravelKf_[sKf.sr_IMφe_IMce_u] = du_REφe_REce_

    def _update_Kf__with_frequencies(self, ravelKf__, base_ravelKf_):
        # 更新所有频率Kf__矩阵频率相关项
        DFNJTFP2D._update_Kf__with_frequencies_JIT(
            # 矩阵
            ravelKf__, base_ravelKf_, self.sKf, self.bKf_.size + 1,
            # 网格参数
            self.Nneg, self.Nsep, self.Npos,
            self.ΔxWest_, self.ΔxEast_, self.Δx_,
            # 电化学参数
            self.Rfneg, self.Rfpos,
            self.εe_, self.Rsneg, self.Rspos,
            self.Dsneg, self.Dspos,
            self.aneg, self.apos,
            self.aeffneg, self.aeffpos,
            self.σeffneg, self.σeffpos, self.κeff_,
            self.CDLneg, self.CDLpos, self.tplus,
            # 状态偏导数
            self.djintdi0intneg_, self.djintdi0intpos_,
            self.djintdηintneg_, self.djintdηintpos_,
            self.di0intdcsnegsurf_, self.di0intdcspossurf_,
            self.dUOCPdcsnegsurf_, self.dUOCPdcspossurf_,
            # 频率
            self.ω_,)

    @staticmethod
    @njit(cache=True)
    def _update_Kf__with_frequencies_JIT(
            # 矩阵
            ravelKf__, base_ravelKf_, sKf, NKf1,
            # 网格参数
            Nneg, Nsep, Npos, ΔxWest_, ΔxEast_, Δx_,
            # 电化学参数
            Rfneg, Rfpos,
            εe_, Rsneg, Rspos,
            Dsneg, Dspos,
            aneg, apos,
            aeffneg, aeffpos,
            σeffneg, σeffpos, κeff_,
            CDLneg, CDLpos, tplus,
            # 状态偏导数
            djintdi0intneg_, djintdi0intpos_,
            djintdηintneg_, djintdηintpos_,
            di0intdcsnegsurf_, di0intdcspossurf_,
            dUOCPdcsnegsurf_, dUOCPdcspossurf_,
            # 频率
            ω_,
            ):
        # 更新所有频率Kf__矩阵频率相关项
        Kcsnegsurf___, Kcspossurf___, ωεeΔx__ = solve_frequency_dependent_variables(
            ω_, εe_, Δx_, Rsneg, Rspos, Dsneg, Dspos, aneg, apos)
        for nf in range(ω_.size):
            ravelKf_ = ravelKf__[nf]
            ravelKf_[:] = base_ravelKf_
            ω = ω_[nf]
            Kcsnegsurf__ = Kcsnegsurf___[nf]
            Kcspossurf__ = Kcspossurf___[nf]
            ωεeΔx_ = ωεeΔx__[nf]

            # 读取参数
            Δxneg, Δxpos = Δx_[0], Δx_[-1]
            Rf2aeffneg = Rfneg/aeffneg
            Rf2aeffpos = Rfpos/aeffpos

            det = Kcsnegsurf__[0, 0]*Kcsnegsurf__[1, 1] - Kcsnegsurf__[0, 1]*Kcsnegsurf__[1, 0]
            dREjintdREcsnegsurf =  Kcsnegsurf__[1, 1]/det
            dREjintdIMcsnegsurf = -Kcsnegsurf__[0, 1]/det
            dIMjintdREcsnegsurf = -Kcsnegsurf__[1, 0]/det
            dIMjintdIMcsnegsurf =  Kcsnegsurf__[0, 0]/det
            det = Kcspossurf__[0, 0]*Kcspossurf__[1, 1] - Kcspossurf__[0, 1]*Kcspossurf__[1, 0]
            dREjintdREcspossurf =  Kcspossurf__[1, 1]/det
            dREjintdIMcspossurf = -Kcspossurf__[0, 1]/det
            dIMjintdREcspossurf = -Kcspossurf__[1, 0]/det
            dIMjintdIMcspossurf =  Kcspossurf__[0, 0]/det

            ωaeffCDLneg = ω*aeffneg*CDLneg
            ωaeffCDLRfneg = ωaeffCDLneg*Rf2aeffneg
            den = 1 + ωaeffCDLRfneg*ωaeffCDLRfneg
            HDLneg2RRE = ωaeffCDLneg*ωaeffCDLRfneg/den
            HDLneg2RIM = ωaeffCDLneg/den
            a = -Rf2aeffneg*dREjintdREcsnegsurf
            b = -Rf2aeffneg*dIMjintdREcsnegsurf
            dREjDLdREcsnegsurf = HDLneg2RRE*a - HDLneg2RIM*b
            dIMjDLdREcsnegsurf = HDLneg2RRE*b + HDLneg2RIM*a
            a = -Rf2aeffneg*dREjintdIMcsnegsurf
            b = -Rf2aeffneg*dIMjintdIMcsnegsurf
            dREjDLdIMcsnegsurf = HDLneg2RRE*a - HDLneg2RIM*b
            dIMjDLdIMcsnegsurf = HDLneg2RRE*b + HDLneg2RIM*a
            dREjDLdREφsneg = HDLneg2RRE
            dREjDLdIMφsneg = -HDLneg2RIM
            dIMjDLdREφsneg = HDLneg2RIM
            dIMjDLdIMφsneg = HDLneg2RRE
            dREjDLdREφeneg = -HDLneg2RRE
            dREjDLdIMφeneg = HDLneg2RIM
            dIMjDLdREφeneg = -HDLneg2RIM
            dIMjDLdIMφeneg = -HDLneg2RRE
            dREjdREcsnegsurf = dREjintdREcsnegsurf + dREjDLdREcsnegsurf
            dREjdIMcsnegsurf = dREjintdIMcsnegsurf + dREjDLdIMcsnegsurf
            dIMjdREcsnegsurf = dIMjintdREcsnegsurf + dIMjDLdREcsnegsurf
            dIMjdIMcsnegsurf = dIMjintdIMcsnegsurf + dIMjDLdIMcsnegsurf
            dREjdREφsneg = dREjDLdREφsneg
            dREjdIMφsneg = dREjDLdIMφsneg
            dIMjdREφsneg = dIMjDLdREφsneg
            dIMjdIMφsneg = dIMjDLdIMφsneg
            dREjdREφeneg = dREjDLdREφeneg
            dREjdIMφeneg = dREjDLdIMφeneg
            dIMjdREφeneg = dIMjDLdREφeneg
            dIMjdIMφeneg = dIMjDLdIMφeneg
            dREηintdREcsnegsurf_ = -dUOCPdcsnegsurf_ - Rf2aeffneg*dREjdREcsnegsurf
            dREηintdIMcsnegsurf = -Rf2aeffneg*dREjdIMcsnegsurf
            dIMηintdREcsnegsurf = -Rf2aeffneg*dIMjdREcsnegsurf
            dIMηintdIMcsnegsurf_ = -dUOCPdcsnegsurf_ - Rf2aeffneg*dIMjdIMcsnegsurf
            dREηintdREφsneg = 1 - Rf2aeffneg*dREjdREφsneg
            dREηintdIMφsneg = -Rf2aeffneg*dREjdIMφsneg
            dIMηintdREφsneg = -Rf2aeffneg*dIMjdREφsneg
            dIMηintdIMφsneg = 1 - Rf2aeffneg*dIMjdIMφsneg
            dREηintdREφeneg = -1 - Rf2aeffneg*dREjdREφeneg
            dREηintdIMφeneg = -Rf2aeffneg*dREjdIMφeneg
            dIMηintdREφeneg = -Rf2aeffneg*dIMjdREφeneg
            dIMηintdIMφeneg = -1 - Rf2aeffneg*dIMjdIMφeneg

            dREjintBVdREcsnegsurf_ = djintdi0intneg_*di0intdcsnegsurf_ + djintdηintneg_*dREηintdREcsnegsurf_
            dREjintBVdIMcsnegsurf_ = djintdηintneg_*dREηintdIMcsnegsurf
            dIMjintBVdREcsnegsurf_ = djintdηintneg_*dIMηintdREcsnegsurf
            dIMjintBVdIMcsnegsurf_ = djintdi0intneg_*di0intdcsnegsurf_ + djintdηintneg_*dIMηintdIMcsnegsurf_
            dREjintBVdREφsneg_ = djintdηintneg_*dREηintdREφsneg
            dREjintBVdIMφsneg_ = djintdηintneg_*dREηintdIMφsneg
            dIMjintBVdREφsneg_ = djintdηintneg_*dIMηintdREφsneg
            dIMjintBVdIMφsneg_ = djintdηintneg_*dIMηintdIMφsneg
            dREjintBVdREφeneg_ = djintdηintneg_*dREηintdREφeneg
            dREjintBVdIMφeneg_ = djintdηintneg_*dREηintdIMφeneg
            dIMjintBVdREφeneg_ = djintdηintneg_*dIMηintdREφeneg
            dIMjintBVdIMφeneg_ = djintdηintneg_*dIMηintdIMφeneg

            ωaeffCDLpos = ω*aeffpos*CDLpos
            ωaeffCDLRfpos = ωaeffCDLpos*Rf2aeffpos
            den = 1 + ωaeffCDLRfpos*ωaeffCDLRfpos
            HDLpos2RRE = ωaeffCDLpos*ωaeffCDLRfpos/den
            HDLpos2RIM = ωaeffCDLpos/den
            a = -Rf2aeffpos*dREjintdREcspossurf
            b = -Rf2aeffpos*dIMjintdREcspossurf
            dREjDLdREcspossurf = HDLpos2RRE*a - HDLpos2RIM*b
            dIMjDLdREcspossurf = HDLpos2RRE*b + HDLpos2RIM*a
            a = -Rf2aeffpos*dREjintdIMcspossurf
            b = -Rf2aeffpos*dIMjintdIMcspossurf
            dREjDLdIMcspossurf = HDLpos2RRE*a - HDLpos2RIM*b
            dIMjDLdIMcspossurf = HDLpos2RRE*b + HDLpos2RIM*a
            dREjDLdREφspos = HDLpos2RRE
            dREjDLdIMφspos = -HDLpos2RIM
            dIMjDLdREφspos = HDLpos2RIM
            dIMjDLdIMφspos = HDLpos2RRE
            dREjDLdREφepos = -HDLpos2RRE
            dREjDLdIMφepos = HDLpos2RIM
            dIMjDLdREφepos = -HDLpos2RIM
            dIMjDLdIMφepos = -HDLpos2RRE
            dREjdREcspossurf = dREjintdREcspossurf + dREjDLdREcspossurf
            dREjdIMcspossurf = dREjintdIMcspossurf + dREjDLdIMcspossurf
            dIMjdREcspossurf = dIMjintdREcspossurf + dIMjDLdREcspossurf
            dIMjdIMcspossurf = dIMjintdIMcspossurf + dIMjDLdIMcspossurf
            dREjdREφspos = dREjDLdREφspos
            dREjdIMφspos = dREjDLdIMφspos
            dIMjdREφspos = dIMjDLdREφspos
            dIMjdIMφspos = dIMjDLdIMφspos
            dREjdREφepos = dREjDLdREφepos
            dREjdIMφepos = dREjDLdIMφepos
            dIMjdREφepos = dIMjDLdREφepos
            dIMjdIMφepos = dIMjDLdIMφepos
            dREηintdREcspossurf = -dUOCPdcspossurf_ - Rf2aeffpos*dREjdREcspossurf
            dREηintdIMcspossurf = -Rf2aeffpos*dREjdIMcspossurf
            dIMηintdREcspossurf = -Rf2aeffpos*dIMjdREcspossurf
            dIMηintdIMcspossurf = -dUOCPdcspossurf_ - Rf2aeffpos*dIMjdIMcspossurf
            dREηintdREφspos = 1 - Rf2aeffpos*dREjdREφspos
            dREηintdIMφspos = -Rf2aeffpos*dREjdIMφspos
            dIMηintdREφspos = -Rf2aeffpos*dIMjdREφspos
            dIMηintdIMφspos = 1 - Rf2aeffpos*dIMjdIMφspos
            dREηintdREφepos = -1 - Rf2aeffpos*dREjdREφepos
            dREηintdIMφepos = -Rf2aeffpos*dREjdIMφepos
            dIMηintdREφepos = -Rf2aeffpos*dIMjdREφepos
            dIMηintdIMφepos = -1 - Rf2aeffpos*dIMjdIMφepos
            dREjintBVdREcspossurf_ = djintdi0intpos_*di0intdcspossurf_ + djintdηintpos_*dREηintdREcspossurf
            dREjintBVdIMcspossurf_ = djintdηintpos_*dREηintdIMcspossurf
            dIMjintBVdREcspossurf_ = djintdηintpos_*dIMηintdREcspossurf
            dIMjintBVdIMcspossurf_ = djintdi0intpos_*di0intdcspossurf_ + djintdηintpos_*dIMηintdIMcspossurf
            dREjintBVdREφspos_ = djintdηintpos_*dREηintdREφspos
            dREjintBVdIMφspos_ = djintdηintpos_*dREηintdIMφspos
            dIMjintBVdREφspos_ = djintdηintpos_*dIMηintdREφspos
            dIMjintBVdIMφspos_ = djintdηintpos_*dIMηintdIMφspos
            dREjintBVdREφepos_ = djintdηintpos_*dREηintdREφepos
            dREjintBVdIMφepos_ = djintdηintpos_*dREηintdIMφepos
            dIMjintBVdREφepos_ = djintdηintpos_*dIMηintdREφepos
            dIMjintBVdIMφepos_ = djintdηintpos_*dIMηintdIMφepos

            ## 更新Kf__矩阵

            # REcsnegsurf行
            ravelKf_[sKf.sr_REcsnegsurf_REcsnegsurf] = dREjintdREcsnegsurf - dREjintBVdREcsnegsurf_
            ravelKf_[sKf.sr_REcsnegsurf_IMcsnegsurf] = dREjintdIMcsnegsurf - dREjintBVdIMcsnegsurf_
            ravelKf_[sKf.sr_REcsnegsurf_REφsneg] = -dREjintBVdREφsneg_
            ravelKf_[sKf.sr_REcsnegsurf_IMφsneg] = -dREjintBVdIMφsneg_
            ravelKf_[sKf.sr_REcsnegsurf_REφeneg] = -dREjintBVdREφeneg_
            ravelKf_[sKf.sr_REcsnegsurf_IMφeneg] = -dREjintBVdIMφeneg_
            # IMcsnegsurf行
            ravelKf_[sKf.sr_IMcsnegsurf_REcsnegsurf] = dIMjintdREcsnegsurf - dIMjintBVdREcsnegsurf_
            ravelKf_[sKf.sr_IMcsnegsurf_IMcsnegsurf] = dIMjintdIMcsnegsurf - dIMjintBVdIMcsnegsurf_
            ravelKf_[sKf.sr_IMcsnegsurf_REφsneg] = -dIMjintBVdREφsneg_
            ravelKf_[sKf.sr_IMcsnegsurf_IMφsneg] = -dIMjintBVdIMφsneg_
            ravelKf_[sKf.sr_IMcsnegsurf_REφeneg] = -dIMjintBVdREφeneg_
            ravelKf_[sKf.sr_IMcsnegsurf_IMφeneg] = -dIMjintBVdIMφeneg_
            # REcspossurf行
            ravelKf_[sKf.sr_REcspossurf_REcspossurf] = dREjintdREcspossurf - dREjintBVdREcspossurf_
            ravelKf_[sKf.sr_REcspossurf_IMcspossurf] = dREjintdIMcspossurf - dREjintBVdIMcspossurf_
            ravelKf_[sKf.sr_REcspossurf_REφspos] = -dREjintBVdREφspos_
            ravelKf_[sKf.sr_REcspossurf_IMφspos] = -dREjintBVdIMφspos_
            ravelKf_[sKf.sr_REcspossurf_REφepos] = -dREjintBVdREφepos_
            ravelKf_[sKf.sr_REcspossurf_IMφepos] = -dREjintBVdIMφepos_
            # IMcspossurf行
            ravelKf_[sKf.sr_IMcspossurf_REcspossurf] = dIMjintdREcspossurf - dIMjintBVdREcspossurf_
            ravelKf_[sKf.sr_IMcspossurf_IMcspossurf] = dIMjintdIMcspossurf - dIMjintBVdIMcspossurf_
            ravelKf_[sKf.sr_IMcspossurf_REφspos] = -dIMjintBVdREφspos_
            ravelKf_[sKf.sr_IMcspossurf_IMφspos] = -dIMjintBVdIMφspos_
            ravelKf_[sKf.sr_IMcspossurf_REφepos] = -dIMjintBVdREφepos_
            ravelKf_[sKf.sr_IMcspossurf_IMφepos] = -dIMjintBVdIMφepos_

            a = (1 - tplus)/F
            Kcejneg = -Δxneg*a
            Kcejpos = -Δxpos*a
            # REce行
            ravelKf_[sKf.sr_REceneg_REcsnegsurf] = Kcejneg*dREjdREcsnegsurf
            ravelKf_[sKf.sr_REceneg_IMcsnegsurf] = Kcejneg*dREjdIMcsnegsurf
            ravelKf_[sKf.sr_REcepos_REcspossurf] = Kcejpos*dREjdREcspossurf
            ravelKf_[sKf.sr_REcepos_IMcspossurf] = Kcejpos*dREjdIMcspossurf
            ravelKf_[sKf.sr_REce_IMce] = -ωεeΔx_
            ravelKf_[sKf.sr_REceneg_REφsneg] = Kcejneg*dREjdREφsneg
            ravelKf_[sKf.sr_REceneg_IMφsneg] = Kcejneg*dREjdIMφsneg
            ravelKf_[sKf.sr_REcepos_REφspos] = Kcejpos*dREjdREφspos
            ravelKf_[sKf.sr_REcepos_IMφspos] = Kcejpos*dREjdIMφspos
            ravelKf_[sKf.sr_REceneg_REφeneg] = Kcejneg*dREjdREφeneg
            ravelKf_[sKf.sr_REceneg_IMφeneg] = Kcejneg*dREjdIMφeneg
            ravelKf_[sKf.sr_REcepos_REφepos] = Kcejpos*dREjdREφepos
            ravelKf_[sKf.sr_REcepos_IMφepos] = Kcejpos*dREjdIMφepos
            # IMce行
            ravelKf_[sKf.sr_IMceneg_REcsnegsurf] = Kcejneg*dIMjdREcsnegsurf
            ravelKf_[sKf.sr_IMceneg_IMcsnegsurf] = Kcejneg*dIMjdIMcsnegsurf
            ravelKf_[sKf.sr_IMcepos_REcspossurf] = Kcejpos*dIMjdREcspossurf
            ravelKf_[sKf.sr_IMcepos_IMcspossurf] = Kcejpos*dIMjdIMcspossurf
            ravelKf_[sKf.sr_IMce_REce] = ωεeΔx_
            ravelKf_[sKf.sr_IMceneg_REφsneg] = Kcejneg*dIMjdREφsneg
            ravelKf_[sKf.sr_IMceneg_IMφsneg] = Kcejneg*dIMjdIMφsneg
            ravelKf_[sKf.sr_IMcepos_REφspos] = Kcejpos*dIMjdREφspos
            ravelKf_[sKf.sr_IMcepos_IMφspos] = Kcejpos*dIMjdIMφspos
            ravelKf_[sKf.sr_IMceneg_REφeneg] = Kcejneg*dIMjdREφeneg
            ravelKf_[sKf.sr_IMceneg_IMφeneg] = Kcejneg*dIMjdIMφeneg
            ravelKf_[sKf.sr_IMcepos_REφepos] = Kcejpos*dIMjdREφepos
            ravelKf_[sKf.sr_IMcepos_IMφepos] = Kcejpos*dIMjdIMφepos

            Kφsnegj = -Δxneg*Δxneg/σeffneg
            Kφsposj = -Δxpos*Δxpos/σeffpos
            # REφsneg行
            ravelKf_[sKf.sr_REφsneg_REcsnegsurf] = Kφsnegj*dREjdREcsnegsurf
            ravelKf_[sKf.sr_REφsneg_IMcsnegsurf] = Kφsnegj*dREjdIMcsnegsurf
            d_REφsneg_REφsneg_ = ravelKf_[sKf.sr_REφsneg_REφsneg]
            d_REφsneg_REφsneg_[:] = -2. + Kφsnegj*dREjdREφsneg
            d_REφsneg_REφsneg_[0] += 1.
            d_REφsneg_REφsneg_[-1] += 1.
            ravelKf_[sKf.sr_REφsneg_IMφsneg] = Kφsnegj*dREjdIMφsneg
            ravelKf_[sKf.sr_REφsneg_REφeneg] = Kφsnegj*dREjdREφeneg
            ravelKf_[sKf.sr_REφsneg_IMφeneg] = Kφsnegj*dREjdIMφeneg
            # REφspos行
            ravelKf_[sKf.sr_REφspos_REcspossurf] = Kφsposj*dREjdREcspossurf
            ravelKf_[sKf.sr_REφspos_IMcspossurf] = Kφsposj*dREjdIMcspossurf
            d_REφspos_REφspos_ = ravelKf_[sKf.sr_REφspos_REφspos]
            d_REφspos_REφspos_[:] = -2. + Kφsposj*dREjdREφspos
            d_REφspos_REφspos_[0] += 1.
            d_REφspos_REφspos_[-1] += 1.
            ravelKf_[sKf.sr_REφspos_IMφspos] = Kφsposj*dREjdIMφspos
            ravelKf_[sKf.sr_REφspos_REφepos] = Kφsposj*dREjdREφepos
            ravelKf_[sKf.sr_REφspos_IMφepos] = Kφsposj*dREjdIMφepos
            # IMφsneg行
            ravelKf_[sKf.sr_IMφsneg_REcsnegsurf] = Kφsnegj*dIMjdREcsnegsurf
            ravelKf_[sKf.sr_IMφsneg_IMcsnegsurf] = Kφsnegj*dIMjdIMcsnegsurf
            ravelKf_[sKf.sr_IMφsneg_REφsneg] = Kφsnegj*dIMjdREφsneg
            d_IMφsneg_IMφsneg_ = ravelKf_[sKf.sr_IMφsneg_IMφsneg]
            d_IMφsneg_IMφsneg_[:] = -2. + Kφsnegj*dIMjdIMφsneg
            d_IMφsneg_IMφsneg_[0] += 1.
            d_IMφsneg_IMφsneg_[-1] += 1.
            ravelKf_[sKf.sr_IMφsneg_REφeneg] = Kφsnegj*dIMjdREφeneg
            ravelKf_[sKf.sr_IMφsneg_IMφeneg] = Kφsnegj*dIMjdIMφeneg
            # IMφspos行
            ravelKf_[sKf.sr_IMφspos_REcspossurf] = Kφsposj*dIMjdREcspossurf
            ravelKf_[sKf.sr_IMφspos_IMcspossurf] = Kφsposj*dIMjdIMcspossurf
            ravelKf_[sKf.sr_IMφspos_REφspos] = Kφsposj*dIMjdREφspos
            d_IMφspos_IMφspos_ = ravelKf_[sKf.sr_IMφspos_IMφspos]
            d_IMφspos_IMφspos_[:] = -2. + Kφsposj*dIMjdIMφspos
            d_IMφspos_IMφspos_[0] += 1.
            d_IMφspos_IMφspos_[-1] += 1.
            ravelKf_[sKf.sr_IMφspos_REφepos] = Kφsposj*dIMjdREφepos
            ravelKf_[sKf.sr_IMφspos_IMφepos] = Kφsposj*dIMjdIMφepos

            # REφe行
            ravelKf_[sKf.sr_REφeneg_REcsnegsurf] = Δxneg*dREjdREcsnegsurf
            ravelKf_[sKf.sr_REφeneg_IMcsnegsurf] = Δxneg*dREjdIMcsnegsurf
            ravelKf_[sKf.sr_REφepos_REcspossurf] = Δxpos*dREjdREcspossurf
            ravelKf_[sKf.sr_REφepos_IMcspossurf] = Δxpos*dREjdIMcspossurf
            ravelKf_[sKf.sr_REφeneg_REφsneg] = Δxneg*dREjdREφsneg
            ravelKf_[sKf.sr_REφeneg_IMφsneg] = Δxneg*dREjdIMφsneg
            ravelKf_[sKf.sr_REφepos_REφspos] = Δxpos*dREjdREφspos
            ravelKf_[sKf.sr_REφepos_IMφspos] = Δxpos*dREjdIMφspos
            sr_REφe_REφe = sKf.sr_REφe_REφe
            d_REφe_REφe_  = ravelKf_[sr_REφe_REφe]
            dl_REφe_REφe_ = ravelKf_[sKf.sr_REφe_REφe_l]
            du_REφe_REφe_ = ravelKf_[sKf.sr_REφe_REφe_u]
            dl_REφe_REφe_[:] = κeff_[1:]/ΔxWest_[1:]
            du_REφe_REφe_[:] = κeff_[:-1]/ΔxEast_[:-1]
            d_REφe_REφe_[0] = -du_REφe_REφe_[0]
            d_REφe_REφe_[1:-1] = -(dl_REφe_REφe_[:-1] + du_REφe_REφe_[1:])
            d_REφe_REφe_[-1] = -dl_REφe_REφe_[-1]
            d_REφe_REφe_[0] -= κeff_[0]/(0.5*Δx_[0])
            start0 = sr_REφe_REφe.start
            for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
                # 修正负极-隔膜、隔膜-正极界面
                nrW = start0 + nW*NKf1
                nrE = start0 + nE*NKf1
                a, c = κeff_[nW]/ΔxWest_[nW], 2*κeff_[nW]*κeff_[nE]/(κeff_[nE]*Δx_[nW] + κeff_[nW]*Δx_[nE])
                ravelKf_[nrW-1:nrW+2] = a, -(a + c), c
                a, c = c, κeff_[nE]/ΔxEast_[nE]
                ravelKf_[nrE-1:nrE+2] = a, -(a + c), c
            base_d_REφe_REφe_  = d_REφe_REφe_.copy()
            base_dl_REφe_REφe_ = dl_REφe_REφe_.copy()
            base_du_REφe_REφe_ = du_REφe_REφe_.copy()
            ravelKf_[sKf.sr_REφe_REφe][:Nneg]  += Δxneg*dREjdREφeneg
            ravelKf_[sKf.sr_REφe_REφe][-Npos:] += Δxpos*dREjdREφepos
            ravelKf_[sKf.sr_REφe_IMφe][:Nneg]  = Δxneg*dREjdIMφeneg
            ravelKf_[sKf.sr_REφe_IMφe][-Npos:] = Δxpos*dREjdIMφepos

            # IMφe行
            ravelKf_[sKf.sr_IMφeneg_REcsnegsurf] = Δxneg*dIMjdREcsnegsurf
            ravelKf_[sKf.sr_IMφeneg_IMcsnegsurf] = Δxneg*dIMjdIMcsnegsurf
            ravelKf_[sKf.sr_IMφepos_REcspossurf] = Δxpos*dIMjdREcspossurf
            ravelKf_[sKf.sr_IMφepos_IMcspossurf] = Δxpos*dIMjdIMcspossurf
            ravelKf_[sKf.sr_IMφeneg_REφsneg] = Δxneg*dIMjdREφsneg
            ravelKf_[sKf.sr_IMφeneg_IMφsneg] = Δxneg*dIMjdIMφsneg
            ravelKf_[sKf.sr_IMφepos_REφspos] = Δxpos*dIMjdREφspos
            ravelKf_[sKf.sr_IMφepos_IMφspos] = Δxpos*dIMjdIMφspos
            ravelKf_[sKf.sr_IMφe_REφe][:Nneg] = Δxneg*dIMjdREφeneg
            ravelKf_[sKf.sr_IMφe_REφe][-Npos:] = Δxpos*dIMjdREφepos
            ravelKf_[sKf.sr_IMφe_IMφe] = base_d_REφe_REφe_
            ravelKf_[sKf.sr_IMφe_IMφe_l] = base_dl_REφe_REφe_
            ravelKf_[sKf.sr_IMφe_IMφe_u] = base_du_REφe_REφe_
            ravelKf_[sKf.sr_IMφe_IMφe][:Nneg]  += Δxneg*dIMjdIMφeneg
            ravelKf_[sKf.sr_IMφe_IMφe][-Npos:] += Δxpos*dIMjdIMφepos

    def EIS(self):
        # 计算电化学阻抗谱
        tEIS = self.t
        if (tEIS_ := self.data['tEIS']) and tEIS_[-1]==tEIS:
            # 若计算过时刻tEIS的阻抗，直接返回
            if self.verbose:
                print(f'已计算时刻{tEIS = } s 电化学阻抗谱')
            return self

        if self.ravelKf_ is None:
            # 生成Kf__、bKf_及其索引
            self._generate_Kf__bKf_and_slices()

        # 读取矩阵
        bKf_ = self.bKf_
        sKf = self.sKf
        NKf = bKf_.size

        # 读取方法
        solve_banded_matrix = P2Dbase.solve_banded_matrix

        # 读取参数
        ω_ = self.ω_
        Nf = ω_.size

        self._update_Kf__bKf_with_pure_parameters()  # 更新Kf__、bKf_纯电化学参数相关项
        self._update_Kf__with_states()               # 更新Kf__状态相关且频率无关项

        X__ = empty((Nf, NKf))
        ravelKf__ = empty((Nf, NKf*NKf))
        self._update_Kf__with_frequencies(ravelKf__, self.ravelKf_.copy())
        for nf in range(Nf):
            # 遍历频率序列
            Kf__ = ravelKf__[nf].reshape(NKf, NKf)

            if self.banded_experience_of_Kf__ is None:
                self.banded_experience_of_Kf__ = expe = self.banded_Kf__by_space(Kf__)
                if self.verbose:
                    print(f"重排频域因变量矩阵Kf__的下带宽 l = {expe['l']}，上带宽 u = {expe['u']}")
            if expe := self.banded_experience_of_Kf__:
                # 带状化求解
                X__[nf] = solve_banded_matrix(Kf__, bKf_, **expe)
            else:
                # 直接求解
                X__[nf] = solve(Kf__, bKf_)
        self.ravelKf_[:] = ravelKf__[-1]

        self.tEIS = tEIS
        REφsneg__ = X__[:, sKf.s_REφsneg]  # 负极固相电势实部
        IMφsneg__ = X__[:, sKf.s_IMφsneg]  # 负极固相电势虚部
        REφspos__ = X__[:, sKf.s_REφspos]  # 正极固相电势实部
        IMφspos__ = X__[:, sKf.s_IMφspos]  # 正极固相电势虚部
        ΔIAC = self.ΔIAC
        hΔiAC = 0.5*self.ΔiAC
        REφsnegCollector_ = REφsneg__[:, 0]  + hΔiAC*self.Δxneg/self.σeffneg  # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = IMφsneg__[:, 0]                                   # (Nf,) 负极集流体电势虚部 [V]
        REφsposCollector_ = REφspos__[:, -1] - hΔiAC*self.Δxpos/self.σeffpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = IMφspos__[:, -1]                        # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφsnegCollector_)/-ΔIAC      # (Nf,) 全电池阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφsnegCollector_)/-ΔIAC      # (Nf,) 全电池阻抗虚部 [Ω]
        self.Z_[:] = Zreal_ + 1j*Zimag_ + self.Zl_                  # (Nf,) 全电池复阻抗 [Ω]

        if self.complete:
            # 读取参数
            Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos
            CDLneg = self.CDLneg
            CDLpos = self.CDLpos
            aeffneg = self.aeffneg
            aeffpos = self.aeffpos
            Rf2aeffneg = self.Rfneg/aeffneg
            Rf2aeffpos = self.Rfpos/aeffpos
            κeff_ = self.κeff_
            Δx_ = self.Δx_
            # 读取状态偏导数
            di0intdcsnegsurf_ = self.di0intdcsnegsurf_
            di0intdcspossurf_ = self.di0intdcspossurf_
            di0intdceneg_ = self.di0intdceneg_
            di0intdcepos_ = self.di0intdcepos_
            dUOCPdcsnegsurf_ = self.dUOCPdcsnegsurf_
            dUOCPdcspossurf_ = self.dUOCPdcspossurf_
            Kcsnegsurf___, Kcspossurf___, _ = solve_frequency_dependent_variables(
                ω_, self.εe_, self.Δx_,
                self.Rsneg, self.Rspos,
                self.Dsneg, self.Dspos,
                self.aneg, self.apos)

            self.REcsnegsurf__[:] = REcsnegsurf__ = X__[:, sKf.s_REcsnegsurf]  # 负极固相表面浓度实部
            self.IMcsnegsurf__[:] = IMcsnegsurf__ = X__[:, sKf.s_IMcsnegsurf]  # 负极固相表面浓度虚部
            self.REcspossurf__[:] = REcspossurf__ = X__[:, sKf.s_REcspossurf]  # 正极固相表面浓度实部
            self.IMcspossurf__[:] = IMcspossurf__ = X__[:, sKf.s_IMcspossurf]  # 正极固相表面浓度虚部
            self.REce__[:] = REce__ = X__[:, sKf.s_REce]  # 电解液锂离子浓度实部
            self.IMce__[:] = IMce__ = X__[:, sKf.s_IMce]  # 电解液锂离子浓度虚部
            self.REφsneg__[:] = REφsneg__  # 负极固相电势实部
            self.IMφsneg__[:] = IMφsneg__  # 负极固相电势虚部
            self.REφspos__[:] = REφspos__  # 正极固相电势实部
            self.IMφspos__[:] = IMφspos__  # 正极固相电势虚部
            self.REφe__[:] = REφe__ = X__[:, sKf.s_REφe]  # 电解液电势实部
            self.IMφe__[:] = IMφe__ = X__[:, sKf.s_IMφe]  # 电解液电势虚部

            inv__ = empty((Nf, 2, 2))

            inv__[:] = batch_inv_2x2(Kcsnegsurf___)
            results__ = inv__ @ stack([REcsnegsurf__, IMcsnegsurf__], axis=1)
            self.REjintneg__[:] = REjintneg__ = results__[:, 0, :]
            self.IMjintneg__[:] = IMjintneg__ = results__[:, 1, :]
            inv__[:] = batch_inv_2x2(Kcspossurf___)
            results__ = inv__ @ stack([REcspossurf__, IMcspossurf__], axis=1)
            self.REjintpos__[:] = REjintpos__ = results__[:, 0, :]
            self.IMjintpos__[:] = IMjintpos__ = results__[:, 1, :]

            A__ = ones((Nf, 2, 2))

            ωaCDLneg_ = ω_*aeffneg*CDLneg
            A__[:, 0, 1] = -ωaCDLneg_*Rf2aeffneg
            A__[:, 1, 0] = -A__[:, 0, 1]
            inv__[:] = batch_inv_2x2(A__)
            results__ = inv__ @ stack([
                -ωaCDLneg_[:, None]*(IMφsneg__ - IMφe__[:, :Nneg] - Rf2aeffneg*IMjintneg__),
                 ωaCDLneg_[:, None]*(REφsneg__ - REφe__[:, :Nneg] - Rf2aeffneg*REjintneg__)], axis=1)
            self.REjDLneg__[:] = REjDLneg__ = results__[:, 0, :]
            self.IMjDLneg__[:] = IMjDLneg__ = results__[:, 1, :]

            ωaCDLpos_ = ω_*aeffpos*CDLpos
            A__[:, 0, 1] = -ωaCDLpos_*Rf2aeffpos
            A__[:, 1, 0] = -A__[:, 0, 1]
            inv__[:] = batch_inv_2x2(A__)
            results__ = inv__ @ stack([
                -ωaCDLpos_[:, None]*(IMφspos__ - IMφe__[:, -Npos:] - Rf2aeffpos*IMjintpos__),
                 ωaCDLpos_[:, None]*(REφspos__ - REφe__[:, -Npos:] - Rf2aeffpos*REjintpos__)], axis=1)
            self.REjDLpos__[:] = REjDLpos__ = results__[:, 0, :]
            self.IMjDLpos__[:] = IMjDLpos__ = results__[:, 1, :]

            self.REi0intneg__[:] = di0intdcsnegsurf_*REcsnegsurf__ + di0intdceneg_*REce__[:, :Nneg]
            self.IMi0intneg__[:] = di0intdcsnegsurf_*IMcsnegsurf__ + di0intdceneg_*IMce__[:, :Nneg]
            self.REi0intpos__[:] = di0intdcspossurf_*REcspossurf__ + di0intdcepos_*REce__[:, -Npos:]
            self.IMi0intpos__[:] = di0intdcspossurf_*IMcspossurf__ + di0intdcepos_*IMce__[:, -Npos:]

            self.REηintneg__[:] = REφsneg__ - REφe__[:, :Nneg] - dUOCPdcsnegsurf_*REcsnegsurf__ - Rf2aeffneg*(REjintneg__ + REjDLneg__)
            self.IMηintneg__[:] = IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdcsnegsurf_*IMcsnegsurf__ - Rf2aeffneg*(IMjintneg__ + IMjDLneg__)
            self.REηintpos__[:] = REφspos__ - REφe__[:, -Npos:] - dUOCPdcspossurf_*REcspossurf__ - Rf2aeffpos*(REjintpos__ + REjDLpos__)
            self.IMηintpos__[:] = IMφspos__ - IMφe__[:, -Npos:] - dUOCPdcspossurf_*IMcspossurf__ - Rf2aeffpos*(IMjintpos__ + IMjDLpos__)

            nW, nE = Nneg - 1, Nneg
            a, b = κeff_[nE]*Δx_[nW], κeff_[nW]*Δx_[nE]
            den = a + b
            REφenegsep_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφenegsep_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφenegsep_ - REφsnegCollector_)/-ΔIAC
            Zimag_ = (IMφenegsep_ - IMφsnegCollector_)/-ΔIAC
            self.Zneg_[:] = Zreal_ + 1j*Zimag_  # 负极阻抗

            nW, nE = Nneg + Nsep - 1, Nneg + Nsep
            a, b = κeff_[nE]*Δx_[nW], κeff_[nW]*Δx_[nE]
            den = a + b
            REφeseppos_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφeseppos_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφsposCollector_ - REφeseppos_)/-ΔIAC
            Zimag_ = (IMφsposCollector_ - IMφeseppos_)/-ΔIAC
            self.Zpos_[:] = Zreal_ + 1j*Zimag_  # 正极阻抗

        if self.verbose:
            print(f'计算时刻{tEIS = :.1f} s 电化学阻抗谱')
        self.record_EISdata()
        return self

    @property
    def ΔiAC(self):
        """交流扰动电流密度振幅 [A/m^2]"""
        return self.ΔIAC/self.A

    @staticmethod
    def solve_REcs__IMcs__(
            r: float,      # 径向坐标 [m]
            ω_: ndarray,   # (Nf,) 角频率序列 [rad/s]
            Rs: float,  # 颗粒半径 [m]
            Ds: float,  # 固相扩散系数 [m^2/s]
            a: float,   # 比表面积 [m^2/m^3]
            REjint__: ndarray,  # (Nf, Nreg) 主反应局部体积电流密度实部 [A/m^3]
            IMjint__: ndarray,  # (Nf, Nreg) 主反应局部体积电流密度虚部 [A/m^3]
            ) -> tuple[ndarray, ndarray]:
        """固相浓度实部、虚部在r处的解析解"""
        Rs2 = Rs*Rs
        W2_ = Rs2/Ds*ω_   # [–]
        W_ = sqrt(W2_)    # [–]
        root2W_ = 1.4142135623730951*W_
        ψ_ = 0.7071067811865476*W_ # [–]
        ψr_ = ψ_*r/Rs  # [–]
        sinψ_ = sin(ψ_)
        cosψ_ = cos(ψ_)
        sinψr_ = sin(ψr_)
        cosψr_ = cos(ψr_)
        aFDsr = a * P2Dbase.F * Ds * r
        # 指数缩放
        # coshψ coshψr sinhψ sinhψr 是 ~exp(ψ) 级别的大数，容易溢出，不能直接算
        # 应缩放：统一乘exp(-ψ)
        exp_ψ_ = exp(-ψ_)
        exp_2ψ_ = exp_ψ_*exp_ψ_
        half_exp_2ψ_ = 0.5*exp_2ψ_
        coshψ_s_ = 0.5 + half_exp_2ψ_  # coshψ*exp(-ψ_)
        sinhψ_s_ = 0.5 - half_exp_2ψ_  # sinhψ*exp(-ψ_)

        exp_ψr_ = exp(-ψr_)
        q_ = exp(ψ_*(r/Rs - 1))
        p_ = exp_ψ_*exp_ψr_
        coshψr_s_ = 0.5*(q_ + p_)  # coshψr*exp(-ψ_)
        sinhψr_s_ = 0.5*(q_ - p_)  # sinhψr*exp(-ψ_)

        a_ = Rs2*(- root2W_  * coshψr_s_ * sinψr_ * coshψ_s_ * cosψ_
                  + 2        * coshψr_s_ * sinψr_ * coshψ_s_ * sinψ_
                  - root2W_  * coshψr_s_ * sinψr_ * sinhψ_s_ * sinψ_
                  - root2W_  * sinhψr_s_ * cosψr_ * coshψ_s_ * cosψ_
                  + root2W_  * sinhψr_s_ * cosψr_ * sinhψ_s_ * sinψ_
                  + 2        * sinhψr_s_ * cosψr_ * sinhψ_s_ * cosψ_)
        b_ = Rs2*(  root2W_ * coshψ_s_ * sinψr_ * coshψ_s_ * cosψ_
                  - root2W_ * coshψ_s_ * sinψr_ * sinhψ_s_ * sinψ_
                  - 2       * coshψ_s_ * sinψr_ * sinhψ_s_ * cosψ_
                  - root2W_ * sinhψ_s_ * cosψr_ * coshψ_s_ * cosψ_
                  + 2       * sinhψ_s_ * cosψr_ * coshψ_s_ * sinψ_
                  - root2W_ * sinhψ_s_ * cosψr_ * sinhψ_s_ * sinψ_)
        d_ = 2*aFDsr*( (W2_ + 1) * coshψ_s_ * coshψ_s_
                       - root2W_ * coshψ_s_ * sinhψ_s_
                       - W2_     * sinψ_    * sinψ_ * exp_2ψ_
                       - root2W_ * cosψ_    * sinψ_ * exp_2ψ_
                       -           cosψ_    * cosψ_ * exp_2ψ_)
        a_ /= d_
        b_ /= d_
        Kcs___ = empty((ω_.size, 2, 2))
        Kcs___[:, 0, 0] = Kcs___[:, 1, 1] = a_
        Kcs___[:, 0, 1] = b_
        Kcs___[:, 1, 0] = -b_
        results___ = Kcs___ @ stack([REjint__, IMjint__], axis=1)  # (Nf, 2, Nreg)
        REcs__, IMcs__ = results___[:, 0, :], results___[:, 1, :]
        return REcs__, IMcs__  # (Nf, Nreg)

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
        Rf2aeffneg, Rf2aeffpos = self.Rfneg/self.aeffneg, self.Rfpos/self.aeffpos
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
        REjintneg__ = self.REjintneg__  # 负极局部体积电流密度实部
        IMjintneg__ = self.IMjintneg__  # 负极局部体积电流密度虚部
        REjintpos__ = self.REjintpos__  # 正极局部体积电流密度实部
        IMjintpos__ = self.IMjintpos__  # 正极局部体积电流密度虚部
        REjDLneg__ = self.REjDLneg__    # 负极双电层局部体积电流密度实部
        IMjDLneg__ = self.IMjDLneg__    # 负极双电层局部体积电流密度虚部
        REjDLpos__ = self.REjDLpos__    # 正极双电层局部体积电流密度实部
        IMjDLpos__ = self.IMjDLpos__    # 正极双电层局部体积电流密度虚部
        REi0intneg__ = self.REi0intneg__  # 负极交换电流密度实部
        IMi0intneg__ = self.IMi0intneg__  # 负极交换电流密度虚部
        REi0intpos__ = self.REi0intpos__  # 正极交换电流密度实部
        IMi0intpos__ = self.IMi0intpos__  # 正极交换电流密度虚部
        REηintneg__ = self.REηintneg__  # 负极过电位实部
        IMηintneg__ = self.IMηintneg__  # 负极过电位虚部
        REηintpos__ = self.REηintpos__  # 正极过电位实部
        IMηintpos__ = self.IMηintpos__  # 正极过电位虚部
        Nf = self.f_.size
        ω_ = self.ω_
        F2RT = 0.5 * P2Dbase.F/(P2Dbase.R*self.T)
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
            # 修正负极-隔膜、隔膜-正极界面锂离子浓度
            REceInterfaces__[:, nE] = (DeeffEast_[nW]*REce__[:, nW]*Δx_[nE] + DeeffWest_[nE]*REce__[:, nE]*Δx_[nW])/(
                                       DeeffEast_[nW]*Δx_[nE] + DeeffWest_[nE]*Δx_[nW])
        REceWest__ = REceInterfaces__[:, :-1]  # 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        REceEast__ = REceInterfaces__[:, 1:]   # 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradREceWest__ = hstack([zeros([Nf, 1]), (REce__[:, 1:] - REce__[:, :-1])/ΔxWest_[1:]])   # 各控制体左界面的锂离子浓度梯度实部 [mol/m^4]
        gradREceEast__ = hstack([(REce__[:, 1:] - REce__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # 各控制体右界面的锂离子浓度梯度实部 [mol/m^4]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            gradREceEast__[:, nW] = (REceEast__[:, nW] - REce__[:, nW])/(0.5*Δx_[nW])
            gradREceWest__[:, nE] = (REce__[:, nE] - REceWest__[:, nE])/(0.5*Δx_[nE])

        # 各控制体界面的电解液锂离子浓度虚部 [mol/m^3]
        IMceInterfaces__ = hstack([IMce__[:, [0]], (IMce__[:, :-1] + IMce__[:, 1:]) * 0.5, IMce__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面锂离子浓度
            IMceInterfaces__[:, nE] = (DeeffEast_[nW]*IMce__[:, nW]*Δx_[nE] + DeeffWest_[nE]*IMce__[:, nE]*Δx_[nW])/(DeeffEast_[nW]*Δx_[nE] + DeeffWest_[nE]*Δx_[nW])
        IMceWest__ = IMceInterfaces__[:, :-1]  # 各控制体左界面的电解液锂离子浓度 [mol/m^3]
        IMceEast__ = IMceInterfaces__[:, 1:]   # 各控制体右界面的电解液锂离子浓度 [mol/m^3]
        gradIMceWest__ = hstack([zeros([Nf, 1]), (IMce__[:, 1:] - IMce__[:, :-1])/ΔxWest_[1:]])   # 各控制体左界面的锂离子浓度梯度虚部 [mol/m^4]
        gradIMceEast__ = hstack([(IMce__[:, 1:] - IMce__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # 各控制体右界面的锂离子浓度梯度虚部 [mol/m^4]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            gradIMceEast__[:, nW] = (IMceEast__[:, nW] - IMce__[:, nW])/(0.5*Δx_[nW])
            gradIMceWest__[:, nE] = (IMce__[:, nE] - IMceWest__[:, nE])/(0.5*Δx_[nE])

        Kcsnegsurf___, Kcspossurf___, ωεeΔx__ = solve_frequency_dependent_variables(
            ω_, εe_, Δx_,
            self.Rsneg, self.Rspos,
            self.Dsneg, self.Dspos,
            self.aneg, self.apos)
        maxError = max([
            abs(array([REcsnegsurf__, IMcsnegsurf__]).transpose(1, 0, 2) - Kcsnegsurf___ @ array([REjintneg__, IMjintneg__]).transpose(1, 0, 2)).max(),
            abs(array([REcspossurf__, IMcspossurf__]).transpose(1, 0, 2) - Kcspossurf___ @ array([REjintpos__, IMjintpos__]).transpose(1, 0, 2)).max(), ])
        print(f'固相表面浓度解析解方程 cssurf 最大误差{maxError} [mol/m^3]')

        LHS__ = -outer(ω_, εe_) * IMce__
        RHS__ = (DeeffEast_*gradREceEast__ - DeeffWest_*gradREceWest__)/Δx_ + (1 - self.tplus)/P2Dbase.F*hstack([REjintneg__ + REjDLneg__, zeros([Nf, Nsep]), REjintpos__ + REjDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液实部浓度方程 REce 最大误差{maxError} [mol/m^3/s]')

        LHS__ = outer(ω_, εe_) * REce__
        RHS__ = (DeeffEast_*gradIMceEast__ - DeeffWest_*gradIMceWest__)/Δx_ + (1 - self.tplus)/P2Dbase.F*hstack([IMjintneg__ + IMjDLneg__, zeros([Nf, Nsep]), IMjintpos__ + IMjDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液虚部浓度方程 IMce 最大误差{maxError} [mol/m^3/s]')

        gradREφsnegInterfaces__ = hstack([full([Nf, 1], -ΔiAC/σeffneg), (REφsneg__[:, 1:] - REφsneg__[:, :-1])/Δxneg, zeros([Nf, 1])])
        ΔREφsneg_ = (gradREφsnegInterfaces__[:, 1:] - gradREφsnegInterfaces__[:, :-1])/Δxneg
        gradIMφsnegInterfaces_ = hstack([zeros([Nf, 1]), (IMφsneg__[:, 1:] - IMφsneg__[:, :-1])/Δxneg, zeros([Nf, 1])])
        ΔIMφsneg_ = (gradIMφsnegInterfaces_[:, 1:] - gradIMφsnegInterfaces_[:, :-1])/Δxneg
        RE_LHS__ = σeffneg*ΔREφsneg_
        RE_RHS__ = REjintneg__ + REjDLneg__
        IM_LHS__ = σeffneg*ΔIMφsneg_
        IM_RHS__ = IMjintneg__ + IMjDLneg__
        maxError = max([abs(RE_LHS__ - RE_RHS__).max(),
                        abs(IM_LHS__ - IM_RHS__).max(),])
        print(f'负极固相电势方程 REφsneg IMφsneg 最大误差{maxError} [A/m^3]')

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
        print(f'正极固相电势方程 REφspos IMφspos 最大误差{maxError} [A/m^3]')

        i0intneg_, i0intpos_ = self.i0intneg_, self.i0intpos_
        ηintneg_, ηintpos_ = self.ηintneg_, self.ηintpos_
        maxError = max([
            abs(REjintneg__ - 2*aeffneg*(REi0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMjintneg__ - 2*aeffneg*(IMi0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*i0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REjintpos__ - 2*aeffpos*(REi0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMjintpos__ - 2*aeffpos*(IMi0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*i0intpos_*cosh(F2RT*ηintpos_))).max(),])
        print(f'主反应BV动力学方程 REjint IMjint 最大误差{maxError} [A/m^3]')

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
            abs(REηintneg__ - (REφsneg__ - REφe__[:, :Nneg] - dUOCPdcsnegsurf_*REcsnegsurf__ - Rf2aeffneg*(REjintneg__ + REjDLneg__))).max(),
            abs(IMηintneg__ - (IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdcsnegsurf_*IMcsnegsurf__ - Rf2aeffneg*(IMjintneg__ + IMjDLneg__))).max(),
            abs(REηintpos__ - (REφspos__ - REφe__[:, -Npos:] - dUOCPdcspossurf_*REcspossurf__ - Rf2aeffpos*(REjintpos__ + REjDLpos__))).max(),
            abs(IMηintpos__ - (IMφspos__ - IMφe__[:, -Npos:] - dUOCPdcspossurf_*IMcspossurf__ - Rf2aeffpos*(IMjintpos__ + IMjDLpos__))).max(),])
        print(f'主反应过电位方程 REηint IMηint 最大误差{maxError} [V]')


if __name__=='__main__':
    import numpy as np
    cell = DFNJTFP2D(
        Δt=10, SOC0=0.1,
        Nneg=9, Nsep=6, Npos=7, Nr=8,
        CDLneg=8, CDLpos=9,
        # Aeffneg=0.5, # Aeffpos=0.4,
        # i0intpos=0.1, i0intneg=0.75,
        # timeDiscretization='backward',
        # radialDiscretization='EI',
        # complete=False,
        # verbose=False,
        )

    I = cell.Qcell
    cell.count_lithium()

    thermalModel = True
    cell.EIS()
    cell.CC(-I, 2300, thermalModel).EIS()
    cell.CC(I, 2000, thermalModel).EIS()
    cell.CC(0, 500, thermalModel).EIS()

    cell.count_lithium()

    cell.checkEIS()

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



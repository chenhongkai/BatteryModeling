#%%
from functools import partial
from typing import Callable

from numpy import ndarray, array, zeros, zeros_like, ones, empty, empty_like,\
    full, hstack, concatenate, stack, \
    exp, sqrt, cos, sin, sinh, cosh, arcsinh, outer, \
    isfinite, minimum, maximum
from numpy.linalg import solve
from scipy.linalg.lapack import dgtsv
from numba import njit

from P2Dmodel.P2Dbase import P2Dbase
from P2Dmodel.OCP import NMC111, Graphite
from P2Dmodel.tools import (diagonalSliceRavel, tridiagonal_matmul,
                            stepping_aware_cached_property, batch_inv_2x2,
                            F, R)

εI0 = 1e-4  # I0计算中的无量纲浓度保护阈值 [–]


@njit(cache=True, fastmath=True)
def solve_Kθssurf___(
        ω_: ndarray,  # (Nf,) 角频率序列 [rad/s]
        Q: float,     # 电极容量 [Ah]
        Ds: float,    # 集总固相锂离子扩散系数 [1/s]
        ) -> ndarray:
    # 求Kθssurf___矩阵
    # Kθssurf___ @ stack([REJint__, IMJint__], axis=1) = stack([REθssurf__, IMθssurf__], axis=1)
    W2_ = ω_/Ds
    W_ = sqrt(W2_)
    root2W_ = 1.4142135623730951*W_
    ψ_ = 0.7071067811865476*W_
    cosψ_ = cos(ψ_)
    sinψ_ = sin(ψ_)
    cosψ2_ = cosψ_*cosψ_
    sinψ2_ = sinψ_*sinψ_
    cosψsinψ_ = cosψ_*sinψ_
    Q6Ds = 21600 * Q * Ds  # [A]

    # 指数缩放
    # cosh²ψ 和 coshψ·sinhψ 是 ~exp(2ψ) 级别的大数，容易溢出，不能直接算
    # 因此，把 cosh²ψ、coshψ·sinhψ 全部乘 exp(-2ψ)，转化成O(1)级别，防止溢出
    exp_2ψ_ = exp(-2*ψ_)
    m_ = 1 + exp_2ψ_
    coshψ2_s_ = 0.25*(m_*m_)                    # cosh²ψ * exp(-2ψ)
    coshψsinhψ_s_ = 0.25*(1 - exp_2ψ_*exp_2ψ_)  # coshψ·sinhψ * exp(-2ψ)

    a_ = -root2W_*(coshψsinhψ_s_ + cosψsinψ_ * exp_2ψ_) + 2*(coshψ2_s_ - cosψ2_ * exp_2ψ_)
    b_ = -root2W_*(coshψsinhψ_s_ - cosψsinψ_ * exp_2ψ_)
    d_ = Q6Ds*((W2_ + 1) * coshψ2_s_
               - root2W_ * coshψsinhψ_s_
               - W2_     * sinψ2_    * exp_2ψ_
               - root2W_ * cosψsinψ_ * exp_2ψ_
               -           cosψ2_    * exp_2ψ_)
    a_ /= d_
    b_ /= d_
    Kθssurf___ = empty((ω_.size, 2, 2))
    Kθssurf___[:, 0, 0] = Kθssurf___[:, 1, 1] = a_
    Kθssurf___[:, 0, 1] = b_
    Kθssurf___[:, 1, 0] = -b_
    return Kθssurf___  # (Nf, 2, 2)


@njit(cache=True, fastmath=True)
def solve_frequency_dependent_variables(
        ω_: ndarray, qe_: ndarray, Δx_: ndarray,
        Qneg: float, Qpos: float,
        Dsneg: float, Dspos: float,
        ) -> tuple[ndarray, ndarray, ndarray]:
    # 求解频率相关变量
    ωqeΔx__ = outer(ω_, qe_*Δx_)                       # (Nf, Ne) 各频率各控制体的ω*qe*Δx值 [A]
    Kθsnegsurf___ = solve_Kθssurf___(ω_, Qneg, Dsneg)  # (Nf, 2, 2) 负极各频率Kθssurf__矩阵
    Kθspossurf___ = solve_Kθssurf___(ω_, Qpos, Dspos)  # (Nf, 2, 2) 正极各频率Kθssurf__矩阵
    return Kθsnegsurf___, Kθspossurf___, ωqeΔx__


@njit(cache=True, fastmath=True)
def solve_Jint_(T: float, I0int_: ndarray, ηint_: ndarray) -> ndarray:
    # 求解主反应集总局部体积电流密度Jint [A]
    return 2*I0int_*sinh(F/(2*R*T)*ηint_)


@njit(cache=True, fastmath=True)
def solve_dJintdI0int_(T: float, ηint_: ndarray) -> ndarray:
    # 求解主反应集总局部体积电流密度Jint对集总交换电流密度I0int的偏导数 [A/A]
    return 2*sinh(F/(2*R*T)*ηint_)


@njit(cache=True, fastmath=True)
def solve_dJintdηint_(T: float, I0int_: ndarray, ηint_: ndarray) -> ndarray:
    # 求解主反应集总局部体积电流密度Jint对过电位ηint的偏导数 [A/V]
    FRT = F/(R*T)
    return FRT*I0int_*cosh(FRT*0.5*ηint_)


@njit(cache=True, fastmath=True)
def solve_I0int_(k: float, θssurf_: ndarray, θe_: ndarray) -> ndarray:
    # 由固液相浓度场求主反应集总交换电流密度I0int [A]
    θssurf_ = maximum(minimum(θssurf_, 1 - εI0), εI0)
    θe_ = maximum(θe_, εI0)
    return k * sqrt(θe_*(1 - θssurf_)*θssurf_)


@njit(cache=True, fastmath=True)
def solve_dI0intdθssurf_(k: float, θssurf_: ndarray, θe_: ndarray, I0int_: ndarray):
    # 求解主反应集总交换电流密度I0int对固相颗粒表面无量纲锂离子浓度θssurf的偏导数 [A/–]
    θssurfEval_ = maximum(minimum(θssurf_, 1 - εI0), εI0)
    θeEval_ = maximum(θe_, εI0)
    dI0intdθssurf_ = k*k * θeEval_*(0.5 - θssurfEval_)/I0int_
    return dI0intdθssurf_ * ((θssurf_ > εI0) & (θssurf_ < 1 - εI0))


@njit(cache=True, fastmath=True)
def solve_dI0intdθe_(θe_: ndarray, I0int_: ndarray) -> ndarray:
    # 求解主反应集总交换电流密度I0int对电解液无量纲锂离子浓度θe的偏导数  [A/–]
    θeEval_ = maximum(θe_, εI0)
    return 0.5*I0int_/θeEval_ * (θe_ > εI0)


class LPJTFP2D(P2Dbase):
    """锂离子电池集总参数时频联合准二维模型 Lumped-Parameter Joint Time-Frequency Pseudo-two-Dimensional model"""

    __slots__ = (
        # 专有参数名
        'Qcell', 'ξQneg', 'ξQpos', 'qeneg', 'qesep', 'qepos',
        '_κneg', '_κsep', '_κpos', 'De', 'κD',
        '_I0intneg', '_I0intpos',
        # 专有时域状态量
        'θsneg__', 'θspos__', 'θsnegsurf_', 'θspossurf_', 'θe_',
        'Jintneg_', 'Jintpos_', 'JDLneg_', 'JDLpos_',
        'I0intneg_', 'I0intpos_',
        # 专有频域状态量
        'REθsnegsurf__', 'IMθsnegsurf__', 'REθspossurf__', 'IMθspossurf__', 'REθe__', 'IMθe__',
        'REJintneg__',   'IMJintneg__',   'REJintpos__',   'IMJintpos__',
        'REJDLneg__',    'IMJDLneg__',    'REJDLpos__',    'IMJDLpos__',
        'REI0intneg__',  'IMI0intneg__',  'REI0intpos__',  'IMI0intpos__',
        # 专有恒定量
        'r_', 'Δr_',
        )

    def __init__(self,
            Qcell: float = 20.,     # 电池理论可用容量 [Ah]
            ξQneg: float = 1.2,     # 负极容量与全电池可用容量之比 [–]
            ξQpos: float = 1.65,    # 正极容量与全电池可用容量之比 [–]
            qeneg: float = 13507.,  # 负极电解液锂离子电荷量 [C]
            qesep: float = 16402.,  # 隔膜电解液锂离子电荷量 [C]
            qepos: float = 25086.,  # 正极电解液锂离子电荷量 [C]
            σneg: float = 3.68e5,   # 负极集总固相电导率 [S]
            σpos: float = 5.94e3,   # 正极集总固相电导率 [S]
            κneg: float = 175.62,   # 负极电解液集总离子电导率 [S]
            κsep: float = 854.24,   # 隔膜电解液集总离子电导率 [S]
            κpos: float = 164.10,   # 正极电解液集总离子电导率 [S]
            Dsneg: float = 2.5e-4,  # 负极集总固相锂离子扩散系数 [1/s]
            Dspos: float = 1.4e-3,  # 正极集总固相锂离子扩散系数 [1/s]
            De: float = 0.2,        # 集总离子扩散率/电导率之比 [V]
            κD: float = 4.39e-4,    # 电解液集总扩散离子电导率系数 [V/K]
            Rfneg: float = 6.91e-5,  # 负极集总SEI膜内阻 [Ω]
            Rfpos: float = 2e-5,     # 正极集总SEI膜内阻 [Ω]
            kneg: float = 32.,      # 负极集总反应速率常数 [A]
            kpos: float = 42.,      # 正极集总反应速率常数 [A]
            CDLneg: float = 144.691,   # 负极集总双电层电容 [F]
            CDLpos: float = 19.971,    # 正极集总双电层电容 [F]
            l: float = 1e-13,          # 等效电感 [H]
            I0intneg: float | None = None,  # 负极主反应集总交换电流密度 [A]
            I0intpos: float | None = None,  # 正极主反应集总交换电流密度 [A]
            Umin: float = 2.8,      # SOC=0%开路电压 [V]
            Umax: float  = 4.2,     # SOC=100%开路电压 [V]
            θminneg: float = None,  # SOC=0%的负极嵌锂状态 [–]，默认需要由Qcell、ξQneg、ξQpos计算4个边界嵌锂状态
            θmaxneg: float = None,  # SOC=100%的负极嵌锂状态 [–]
            θminpos: float = None,  # SOC=100%的正极嵌锂状态 [–]
            θmaxpos: float = None,  # SOC=0%的正极嵌锂状态 [–]
            SOC0: float = 0.5,      # 初始荷电状态 [–]
            UOCPneg: Callable = Graphite().Graphite_COMSOL,  # 负极开路电位函数
            UOCPpos: Callable = NMC111().NMC111_COMSOL,      # 正极开路电位函数
            **kwargs,):
        self.Qcell = Qcell; assert Qcell>0, f'电池理论可用容量{Qcell = }，应大于0 [Ah]'
        # 4边界嵌锂状态参数；负极、正极容量比ξQneg、ξQpos
        if all(v is not None for v in (θminneg, θmaxneg, θminpos, θmaxpos)):
            # 4θ均非None，使用给定4θ，忽略4等式关系，并重新计算ξQneg、ξQpos
            assert 0<θminneg<θmaxneg<1, f'负极最小、最大嵌锂状态{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1'
            assert 0<θminpos<θmaxpos<1, f'正极最小、最大嵌锂状态{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1'
            self.ξQneg = 1/(θmaxneg - θminneg)
            self.ξQpos = 1/(θmaxpos - θminpos)
        else:
            # 使用4等式由Qcell、ξQneg、ξQpos计算4θ
            self.ξQneg = ξQneg; assert ξQneg>1, f'负极容量与全电池可用容量之比{ξQneg = }，应大于1'
            self.ξQpos = ξQpos; assert ξQpos>1, f'正极容量与全电池可用容量之比{ξQpos = }，应大于1'
            assert Umax>Umin>0, f'运行电压{Umax = }，{Umin = }，应满足Umax > Umin > 0 [V]'
            θminneg, θmaxneg, θminpos, θmaxpos = P2Dbase.solve_4θ(
                UOCPneg, UOCPpos, Qcell, self.Qneg, self.Qpos,
                Umin, Umax)
        # 3电解液锂电荷量
        self.qeneg = qeneg; assert qeneg>0, f'负极电解液锂离子电荷量{qeneg = }，应大于0 [C]'
        self.qesep = qesep; assert qesep>0, f'隔膜电解液锂离子电荷量{qesep = }，应大于0 [C]'
        self.qepos = qepos; assert qepos>0, f'正极电解液锂离子电荷量{qepos = }，应大于0 [C]'
        # 11输运参数
        self.σneg = σneg; assert σneg>0, f'负极集总固相电导率{σneg = }，应大于0 [S]'
        self.σpos = σpos; assert σpos>0, f'正极集总固相电导率{σpos = }，应大于0 [S]'
        self.κneg = κneg; assert κneg>0, f'负极电解液集总离子电导率{κneg = }，应大于0 [S]'
        self.κsep = κsep; assert κsep>0, f'隔膜电解液集总离子电导率{κsep = }，应大于0 [S]'
        self.κpos = κpos; assert κpos>0, f'正极电解液集总离子电导率{κpos = }，应大于0 [S]'
        self.Dsneg = Dsneg; assert Dsneg>0, f'负极集总固相锂离子扩散系数{Dsneg = }，应大于0 [1/s]'
        self.Dspos = Dspos; assert Dspos>0, f'正极集总固相锂离子扩散系数{Dspos = }，应大于0 [1/s]'
        self.De = De; assert De>0, f'集总离子扩散率/电导率之比{De = }，应大于0 [V]'
        self.κD = κD; assert κD>0, f'集总扩散电解液离子电导率系数{κD = }，应大于0 [V/K]'
        self.Rfneg = Rfneg; assert Rfneg>=0, f'负极集总SEI膜电阻{Rfneg = }，应大于或等于0 [Ω]'
        self.Rfpos = Rfpos; assert Rfpos>=0, f'正极集总SEI膜电阻{Rfpos = }，应大于或等于0 [Ω]'
        # 2动力学参数
        self.kneg = kneg; assert kneg>0, f'负极主反应集总速率常数{kneg = }，应大于0 [A]'
        self.kpos = kpos; assert kpos>0, f'正极主反应集总速率常数{kpos = }，应大于0 [A]'
        # 3电抗参数
        self.CDLneg = CDLneg; assert CDLneg>=0, f'负极集总双电层电容{CDLneg = }，应大于或等于0 [F]'
        self.CDLpos = CDLpos; assert CDLpos>=0, f'正极集总双电层电容{CDLpos = }，应大于或等于0 [F]'
        self.l = l;           assert l>=0, f'等效电感{l = }，应大于或等于0 [H]'
        # 2集总交换电流密度
        self._I0intneg = self._i0intneg = I0intneg; assert (I0intneg is None) or (I0intneg>0), f'负极主反应集总交换电流密度{I0intneg = }，应大于0 [A]'
        self._I0intpos = self._i0intpos = I0intpos; assert (I0intpos is None) or (I0intpos>0), f'正极主反应集总交换电流密度{I0intpos = }，应大于0 [A]'
        # P2D通用参数
        P2Dbase.__init__(self,
                         Lneg=1, Lsep=1, Lpos=1,
                         Rsneg=1, Rspos=1,
                         SOC0=SOC0,
                         UOCPneg=UOCPneg, UOCPpos=UOCPpos,
                         θminneg=θminneg, θmaxneg=θmaxneg,
                         θminpos=θminpos, θmaxpos=θmaxpos, **kwargs)
        # LPJTFP2D专有状态量
        Nneg, Npos, Ne, Nr = self.Nneg, self.Npos, self.Ne, self.Nr  # 读取：网格数
        θsneg = θminneg + SOC0*(θmaxneg - θminneg)  # 初始负极嵌锂状态 [–]
        θspos = θmaxpos - SOC0*(θmaxpos - θminpos)  # 初始正极嵌锂状态 [–]
        self.θsneg__ = full((Nr, Nneg), θsneg, float)  # 初始化：负极固相内部无量纲锂离子浓度场 [–]
        self.θspos__ = full((Nr, Npos), θspos, float)  # 初始化：正极固相内部无量纲锂离子浓度场 [–]
        self.θsnegsurf_ = full(Nneg, θsneg, float)  # 初始化：负极固相表面无量纲锂离子浓度场 [–]
        self.θspossurf_ = full(Npos, θspos, float)  # 初始化：正极固相表面无量纲锂离子浓度场 [–]
        self.θe_ = ones(Ne)          # 初始化：电解液无量纲锂离子浓度场 [–]
        self.Jintneg_ = zeros(Nneg)  # 初始化：负极主反应集总局部体积电流密度场 [A]
        self.Jintpos_ = zeros(Npos)  # 初始化：正极主反应集总局部体积电流密度场 [A]
        self.JDLneg_ = zeros(Nneg)   # 初始化：负极双电层效应集总局部体积电流密度场 [A]
        self.JDLpos_ = zeros(Npos)   # 初始化：正极双电层效应集总局部体积电流密度场 [A]
        I0intneg = self.I0intneg if self._I0intneg else LPJTFP2D.solve_I0int_(self.kneg, θsneg, 1)
        I0intpos = self.I0intpos if self._I0intpos else LPJTFP2D.solve_I0int_(self.kpos, θspos, 1)
        self.I0intneg_ = full(Nneg, I0intneg, float)  # 初始化：负极主反应集总交换电流密度场 [A]
        self.I0intpos_ = full(Npos, I0intpos, float)  # 初始化：正极主反应集总交换电流密度场 [A]
        # 恒定量
        self.Δr_ = self.Δrneg_
        self.r_  = self.rneg_
        if self.complete:
            # 状态量
            Nf = self.f_.size
            self.REθsnegsurf__, self.IMθsnegsurf__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极固相表面无量纲浓度实部、虚部 [–]
            self.REθspossurf__, self.IMθspossurf__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极固相表面无量纲浓度实部、虚部 [–]
            self.REθe__, self.IMθe__ = empty((Nf, Ne)), empty((Nf, Ne))                    # 电解液无量纲锂离子浓度实部、虚部 [–]
            self.REJintneg__, self.IMJintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))      # 负极主反应集总局部体积电流密度实部、虚部 [A]
            self.REJintpos__, self.IMJintpos__ = empty((Nf, Npos)), empty((Nf, Npos))      # 正极主反应集总局部体积电流密度实部、虚部 [A]
            self.REJDLneg__, self.IMJDLneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))        # 负极双电层集总局部体积电流密度实部、虚部 [A]
            self.REJDLpos__, self.IMJDLpos__ = empty((Nf, Npos)), empty((Nf, Npos))        # 正极双电层集总局部体积电流密度实部、虚部 [A]
            self.REI0intneg__, self.IMI0intneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))    # 负极主反应集总交换电流密度实部、虚部 [A]
            self.REI0intpos__, self.IMI0intpos__ = empty((Nf, Npos)), empty((Nf, Npos))    # 正极主反应集总交换电流密度实部、虚部 [A]
            extra_datanames_ = [             # 需记录的数据名称
                'θsneg__', 'θspos__',        # 负极、正极固相无量纲锂离子浓度场 [–]
                'θsnegsurf_', 'θspossurf_',  # 负极、正极表面无量纲锂离子浓度场 [–]
                'θe_',                       # 电解液无量纲锂离子浓度场 [–]
                'Jintneg_', 'Jintpos_',      # 负极、正极主反应集总局部体积电流密度场 [A]
                'JDLneg_', 'JDLpos_',        # 负极、正极双电层效应集总局部体积电流密度场 [A]
                'I0intneg_', 'I0intpos_',]   # 负极、正极主反应集总交换电流密度场 [A]
            extra_EISdatanames_ = [                # 额外需记录的阻抗数据名称
                'REθsnegsurf__', 'IMθsnegsurf__',  # 负极固相表面无量纲锂离子浓度实部、虚部 [–]
                'REθspossurf__', 'IMθspossurf__',  # 正极固相表面无量纲锂离子浓度实部、虚部 [–]
                'REθe__', 'IMθe__',                # 电解液无量纲锂离子浓度实部、虚部 [–]
                'REJintneg__', 'IMJintneg__',      # 负极主反应集总局部体积电流密度实部、虚部 [A]
                'REJintpos__', 'IMJintpos__',      # 正极主反应集总局部体积电流密度实部、虚部 [A]
                'REJDLneg__', 'IMJDLneg__',        # 负极双电层效应集总局部体积电流密度实部、虚部 [A]
                'REJDLpos__', 'IMJDLpos__',        # 正极双电层效应集总局部体积电流密度实部、虚部 [A]
                'REI0intneg__', 'IMI0intneg__',    # 负极主反应集总交换电流密度实部、虚部 [A]
                'REI0intpos__', 'IMI0intpos__',]   # 正极主反应集总交换电流密度实部、虚部 [A]
            self.datanames_.extend(extra_datanames_)
            self.EISdatanames_.extend(extra_EISdatanames_)
            self.data.update({name: [] for name in (extra_datanames_ + extra_EISdatanames_)})

            # 作图变量单位
            self.xSign, self.xUnit = r'$\overline{\it x}$', ''  # 电极厚度方向坐标x符号、单位
            self.rSign, self.rUnit = r'$\overline{\it r}$', ''  # 颗粒径向坐标r符号、单位
            self.cSign, self.cUnit = r'${\it θ}$', ''   # 锂离子浓度θ符号、单位
            self.jSign, self.jUnit = r'${\it J}$', 'A'  # 集总局部体积电流密度J符号、单位
            self.i0Sign, self.i0Unit = r'${\it I}_{0}$', 'A'  # 集总交换电流密度I0符号、单位

        if self.verbose:
            print(self)
            print(f'{self.__class__.__name__}初始化完成!')

    def _update_K__bK_(self, Δt):
        # 更新K__矩阵、bK_向量

        # 读取模式
        is_CN = self.timeDiscretization == 'CN'

        ## 计算α_, β, γ_
        # θssurf = α + β*Jint  -->  Jint = θssurf/β - α/β
        Δ = 1 - self.Δr_[-1]
        KθsJintneg = Δt/(10800*self.Qneg)/((1 - Δ*Δ*Δ)/3)
        KθsJintpos = Δt/(10800*self.Qpos)/((1 - Δ*Δ*Δ)/3)
        bandKθsneg__ = (Δt*self.Dsneg)*self.bandKcsneg__
        bandKθspos__ = (Δt*self.Dspos)*self.bandKcspos__
        if is_CN:
            bandKθsneg__ *= .5
            bandKθspos__ *= .5
            KθsJintneg *= .5
            KθsJintpos *= .5
            bandBθsneg__ = -bandKθsneg__
            bandBθspos__ = -bandKθspos__
            bandBθsneg__[1] += 1
            bandBθspos__[1] += 1
            RHSθsneg__ = tridiagonal_matmul(bandBθsneg__, self.θsneg__)
            RHSθspos__ = tridiagonal_matmul(bandBθspos__, self.θspos__)
        else:
            RHSθsneg__ = self.θsneg__
            RHSθspos__ = self.θspos__
        bandKθsneg__[1] += 1
        bandKθspos__[1] += 1
        e__ = self.e__
        RHSθsneg__ = concatenate((RHSθsneg__, e__), axis=1)
        RHSθspos__ = concatenate((RHSθspos__, e__), axis=1)
        Sθsneg__ = dgtsv(bandKθsneg__[2, :-1], bandKθsneg__[1], bandKθsneg__[0, 1:], RHSθsneg__, True, True, True, True)[3]
        Sθspos__ = dgtsv(bandKθspos__[2, :-1], bandKθspos__[1], bandKθspos__[0, 1:], RHSθspos__, True, True, True, True)[3]
        θsnegI__ = Sθsneg__[:, :-1]  # 负极内部无量纲锂离子浓度的历史影响分量
        θsposI__ = Sθspos__[:, :-1]  # 正极内部无量纲锂离子浓度的历史影响分量
        γneg_ = Sθsneg__[:, -1] * -KθsJintneg
        γpos_ = Sθspos__[:, -1] * -KθsJintpos
        coeffsExpl_ = self.coeffsExpl_
        αneg_ = coeffsExpl_.dot(θsnegI__[-3:])
        αpos_ = coeffsExpl_.dot(θsposI__[-3:])
        βneg = coeffsExpl_.dot(γneg_[-3:])
        βpos = coeffsExpl_.dot(γpos_[-3:])
        if is_CN:
            αneg_ += βneg*self.Jintneg_
            αpos_ += βpos*self.Jintpos_

        ## 双电层效应集总局部体积电流密度JDL显式表达式系数及偏导数
        # JDL = (bJDL + C*(φs - φe - Rf*Jint))/D，其中 D = 1 + C*Rf
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

        (bJDLneg_, bJDLpos_, Cneg, Cpos, Dneg, Dpos,
         dJdθsnegsurf, dJdθspossurf,
         dJDLdφsneg, dJDLdφeneg,
         dJDLdφspos, dJDLdφepos,) = LPJTFP2D._update_K__bK_JIT(
            # 矩阵
            self.ravelK_, self.bK_, self.sK,
            # 模式
            is_CN,
            # 网格参数
            self.Nneg, self.Nsep, self.Npos, Δt, self.Δx_, self.ΔxWest_, self.ΔxEast_,
            # 电化学参数
            self.σneg, self.σpos, self.Rfneg, self.Rfpos, self.CDLneg, self.CDLpos,
            self.Deκ_, self.κ_, self.qe_,
            # 状态
            self.I, self.θe_, self.Jneg_, self.Jpos_,
            # 双电层历史
            NΔφse, t_1, t_2, t_3,
            Δφseneg_1_, Δφseneg_2_, Δφseneg_3_,
            Δφsepos_1_, Δφsepos_2_, Δφsepos_3_,
            # 固相扩散消元结果
            αneg_, βneg, αpos_, βpos,)

        return (
            θsnegI__, αneg_, βneg, γneg_,
            θsposI__, αpos_, βpos, γpos_,
            bJDLneg_, bJDLpos_, Cneg, Cpos, Dneg, Dpos,
            dJdθsnegsurf, dJdθspossurf,
            dJDLdφsneg, dJDLdφeneg,
            dJDLdφspos, dJDLdφepos,)

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
            σneg, σpos, Rfneg, Rfpos, CDLneg, CDLpos, Deκ_, κ_, qe_,
            # 状态
            I, θe_, Jneg_, Jpos_,
            # 双电层历史
            NΔφse, t_1, t_2, t_3,
            Δφseneg_1_, Δφseneg_2_, Δφseneg_3_,
            Δφsepos_1_, Δφsepos_2_, Δφsepos_3_,
            # 固相扩散消元结果
            αneg_, βneg, αpos_, βpos,
            ):
        # 更新K__矩阵、bK_向量

        # 读取网格
        Δxneg, Δxpos = Δx_[0], Δx_[-1]

        # θssurf = α + β*Jint  -->  Jint = θssurf/β - α/β
        dJintdθsnegsurf = 1/βneg
        dJintdθspossurf = 1/βpos
        Jintneg_const_ = αneg_/-βneg
        Jintpos_const_ = αpos_/-βpos

        ## 双电层效应集总局部体积电流密度JDL显式表达式系数及偏导数
        # JDL = (bJDL + C*(φs - φe - Rf*Jint))/D，其中 D = 1 + C*Rf
        if NΔφse == 0:  # timeDL=False，双电层电流及其偏导数为零
            Cneg = Cpos = 0
            Dneg = Dpos = 1
            bJDLneg_ = zeros(Nneg)
            bJDLpos_ = zeros(Npos)
        else:
            t = t_1 + Δt  # 步后时刻 [s]
            c = 1/Δt
            if NΔφse > 1:
                c += 1/(t - t_2)
            if NΔφse > 2:
                c += 1/(t - t_3)
            Cneg = CDLneg*c
            Cpos = CDLpos*c
            Dneg = 1 + Cneg*Rfneg
            Dpos = 1 + Cpos*Rfpos
            if NΔφse > 2:
                A = (t - t_2)*(t - t_3)/-Δt/(t_1 - t_2)/(t_1 - t_3)
                B = Δt*(t - t_3)/(t_2 - t)/(t_2 - t_1)/(t_2 - t_3)
                C = Δt*(t - t_2)/(t_3 - t)/(t_3 - t_1)/(t_3 - t_2)
                bJDLneg_ = CDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_ + C*Δφseneg_3_)
                bJDLpos_ = CDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_ + C*Δφsepos_3_)
            elif NΔφse==2:
                A = (t - t_2)/(-Δt*(t_1 - t_2))
                B = Δt/((t_2 - t)*(t_2 - t_1))
                bJDLneg_ = CDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_)
                bJDLpos_ = CDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_)
            else:
                bJDLneg_ = -Cneg*Δφseneg_1_
                bJDLpos_ = -Cpos*Δφsepos_1_
        dJDLdφsneg = Cneg/Dneg
        dJDLdφeneg = -dJDLdφsneg
        dJDLdJintneg = dJDLdφeneg*Rfneg
        dJDLdφspos = Cpos/Dpos
        dJDLdφepos = -dJDLdφspos
        dJDLdJintpos = dJDLdφepos*Rfpos
        dJDLdθsnegsurf = dJDLdJintneg*dJintdθsnegsurf
        dJDLdθspossurf = dJDLdJintpos*dJintdθspossurf
        JDLneg_const_ = bJDLneg_/Dneg + dJDLdJintneg*Jintneg_const_
        JDLpos_const_ = bJDLpos_/Dpos + dJDLdJintpos*Jintpos_const_

        dJdθsnegsurf = dJintdθsnegsurf + dJDLdθsnegsurf
        dJdθspossurf = dJintdθspossurf + dJDLdθspossurf
        Jneg_const_ = Jintneg_const_ + JDLneg_const_
        Jpos_const_ = Jintpos_const_ + JDLpos_const_

        # 被消去的反应源项在各方程中的线性系数
        KθeJ = -Δt
        if is_CN:
            KθeJ *= .5
        KφsnegJ = -Δxneg*Δxneg/σneg
        KφsposJ = -Δxpos*Δxpos/σpos
        KφenegJ = Δxneg
        KφeposJ = Δxpos

        ## 赋值K__矩阵

        # θe行θssurf列
        ravelK_[sK.sr_ceneg_csnegsurf] = KθeJ*dJdθsnegsurf
        ravelK_[sK.sr_cepos_cspossurf] = KθeJ*dJdθspossurf
        # θe行θe列
        s_ce = sK.s_ce
        sr_ce_ce = sK.sr_ce_ce
        dl_ = ravelK_[sK.sr_ce_ce_l]  # 下对角线
        du_ = ravelK_[sK.sr_ce_ce_u]  # 上对角线
        d_ = ravelK_[sr_ce_ce]        # 主对角线
        dl_[:] = -Deκ_[1:] / ΔxWest_[1:]
        du_[:] = -Deκ_[:-1] / ΔxEast_[:-1]
        d_[0] = -du_[0]
        d_[1:-1] = -(dl_[:-1] + du_[1:])
        d_[-1] = -dl_[-1]
        NK1 = bK_.size + 1
        start0 = sr_ce_ce.start
        for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NK1
            nrE = start0 + nE*NK1
            a, c = -Deκ_[nW]/ΔxWest_[nW], -2*Deκ_[nW]*Deκ_[nE]/(Deκ_[nW]*Δx_[nE] + Deκ_[nE]*Δx_[nW])
            ravelK_[nrW-1:nrW+2] = a, -(a + c), c
            a, c = c, -Deκ_[nE]/ΔxEast_[nE]
            ravelK_[nrE-1:nrE+2] = a, -(a + c), c
        Δt2Δx_ = Δt/Δx_
        dl_ *= Δt2Δx_[1:]
        du_ *= Δt2Δx_[:-1]
        d_  *= Δt2Δx_
        if is_CN:
            dl_ *= .5
            du_ *= .5
            d_ *= .5
            # θe行CN格式历史项，等价于Bθe__ @ θe_
            bKθe_ = bK_[s_ce]
            bKθe_[:] = qe_*θe_
            bKθe_[0] -= d_[0]*θe_[0] + du_[0]*θe_[1]
            bKθe_[1:-1] -= dl_[:-1]*θe_[:-2] + d_[1:-1]*θe_[1:-1] + du_[1:]*θe_[2:]
            bKθe_[-1] -= dl_[-1]*θe_[-2] + d_[-1]*θe_[-1]
        d_ += qe_
        # θe行φs列
        ravelK_[sK.sr_ceneg_φsneg] = KθeJ*dJDLdφsneg
        ravelK_[sK.sr_cepos_φspos] = KθeJ*dJDLdφspos
        # θe行φe列
        ravelK_[sK.sr_ceneg_φeneg] = KθeJ*dJDLdφeneg
        ravelK_[sK.sr_cepos_φepos] = KθeJ*dJDLdφepos

        # φs行θssurf列
        ravelK_[sK.sr_φsneg_csnegsurf] = KφsnegJ*dJdθsnegsurf
        ravelK_[sK.sr_φspos_cspossurf] = KφsposJ*dJdθspossurf
        # φs行φs列
        dφsneg_ = ravelK_[sK.sr_φsneg_φsneg]
        dφspos_ = ravelK_[sK.sr_φspos_φspos]
        dφsneg_[0] = -1 + KφsnegJ*dJDLdφsneg
        dφsneg_[1:-1] = -2 + KφsnegJ*dJDLdφsneg
        dφsneg_[-1] = -1 + KφsnegJ*dJDLdφsneg
        dφspos_[0] = -1 + KφsposJ*dJDLdφspos
        dφspos_[1:-1] = -2 + KφsposJ*dJDLdφspos
        dφspos_[-1] = -1 + KφsposJ*dJDLdφspos
        # φs行φe列
        ravelK_[sK.sr_φsneg_φeneg] = KφsnegJ*dJDLdφeneg
        ravelK_[sK.sr_φspos_φepos] = KφsposJ*dJDLdφepos

        # φe行θssurf列
        ravelK_[sK.sr_φeneg_csnegsurf] = KφenegJ*dJdθsnegsurf
        ravelK_[sK.sr_φepos_cspossurf] = KφeposJ*dJdθspossurf
        # φe行φs列
        ravelK_[sK.sr_φeneg_φsneg] = KφenegJ*dJDLdφsneg
        ravelK_[sK.sr_φepos_φspos] = KφeposJ*dJDLdφspos
        # φe行φe列
        dl_ = ravelK_[sK.sr_φe_φe_l]
        du_ = ravelK_[sK.sr_φe_φe_u]
        sr_φe_φe = sK.sr_φe_φe
        d_ = ravelK_[sr_φe_φe]  # 主对角线
        dl_[:] = κ_[1:] / ΔxWest_[1:]
        du_[:] = κ_[:-1] / ΔxEast_[:-1]
        d_[0] = -du_[0]
        d_[1:-1] = -(dl_[:-1] + du_[1:])
        d_[-1] = -dl_[-1]
        d_[0] -= κ_[0]/(0.5*Δx_[0])  # 首元占优，固定电解液电势参考
        start0 = sr_φe_φe.start
        for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NK1
            nrE = start0 + nE*NK1
            a, c = κ_[nW]/ΔxWest_[nW], 2*κ_[nW]*κ_[nE]/(κ_[nW]*Δx_[nE] + κ_[nE]*Δx_[nW])
            ravelK_[nrW-1:nrW+2] = a, -(a + c), c
            a, c = c, κ_[nE]/ΔxEast_[nE]
            ravelK_[nrE-1:nrE+2] = a, -(a + c), c
        d_[:Nneg]  += KφenegJ*dJDLdφeneg
        d_[-Npos:] += KφeposJ*dJDLdφepos

        ## 更新bK_

        # θssurf行
        bK_[sK.s_csnegsurf] = αneg_
        bK_[sK.s_cspossurf] = αpos_
        # θe行
        bKθe_ = bK_[s_ce]
        if is_CN:
            bKθe_[:Nneg]  -= KθeJ*Jneg_
            bKθe_[-Npos:] -= KθeJ*Jpos_
        else:
            bKθe_[:] = qe_*θe_
        bKθe_[:Nneg]  -= KθeJ*Jneg_const_
        bKθe_[-Npos:] -= KθeJ*Jpos_const_
        # φs行
        bKφsneg_ = bK_[sK.s_φsneg]
        bKφspos_ = bK_[sK.s_φspos]
        bKφsneg_[:] = -KφsnegJ*Jneg_const_
        bKφspos_[:] = -KφsposJ*Jpos_const_
        bKφsneg_[0]  += -Δxneg*I/σneg
        bKφspos_[-1] +=  Δxpos*I/σpos  # 固相电流边界条件
        # φe行
        bKφe_ = bK_[sK.s_φe]
        bKφe_[:Nneg]  = -KφenegJ*Jneg_const_
        bKφe_[-Npos:] = -KφeposJ*Jpos_const_

        return (
            bJDLneg_, bJDLpos_,
            Cneg, Cpos, Dneg, Dpos,
            dJdθsnegsurf,
            dJdθspossurf,
            dJDLdφsneg, dJDLdφeneg,
            dJDLdφspos, dJDLdφepos,)

    def _stepping(self, Δt):
        # 时间步进：Newton迭代

        # 读取矩阵
        K__ = self.ravelK_.base  # 读取：因变量线性矩阵K__
        bK_ = self.bK_           # 读取：常数项向量，F_ = K__ @ X_ - bK_

        # 读取索引
        sK = self.sK
        s_θ = sK.s_c
        s_φ = sK.s_φ

        # 读取方法
        solve_banded_matrix = P2Dbase.solve_banded_matrix
        solve_I0int_ = LPJTFP2D.solve_I0int_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_
        solve_dUOCPdθsneg_ = self.solve_dUOCPdθsneg_
        solve_dUOCPdθspos_ = self.solve_dUOCPdθspos_

        # 读取参数
        Nneg, Nsep = self.Nneg, self.Nsep  # 读取：网格数
        Δx_ = self.Δx_                                 # 网格尺寸 [–]
        ΔxWest_, ΔxEast_ = self.ΔxWest_, self.ΔxEast_  # 网格距离 [–]
        Rfneg, Rfpos = self.Rfneg, self.Rfpos  # 负极、正极集总SEI膜内阻 [Ω]
        κ_ = self.κ_
        if I0intnegUnknown := (self._I0intneg is None):
            kneg = self.kneg          # 读取：负极主反应集总速率常数 [A]
        else:
            kneg = 0.
            I0intneg = self.I0intneg  # 读取：负极主反应集总交换电流密度 [A]
        if I0intposUnknown := (self._I0intpos is None):
            kpos = self.kpos          # 读取：正极主反应集总速率常数 [A]
        else:
            kpos = 0.
            I0intpos = self.I0intpos  # 读取：正极主反应集总交换电流密度 [A]

        # 读取状态
        I = self.I  # 电流 [A]
        T = self.T  # 温度 [K]
        κDκT_ = (self.κD * T) * κ_
        Ihistory_ = self.Ihistory_  # 历史电流序列

        # 更新K__矩阵、bK_向量
        (θsnegI__, αneg_, βneg, γneg_,
         θsposI__, αpos_, βpos, γpos_,
         bJDLneg_, bJDLpos_,
         Cneg, Cpos, Dneg, Dpos,
         dJdθsnegsurf_, dJdθspossurf_,
         dJDLdφsneg, dJDLdφeneg,
         dJDLdφspos, dJDLdφepos,
        ) = self._update_K__bK_(Δt)
        dηintdφsneg = 1 - Rfneg*dJDLdφsneg
        dηintdφspos = 1 - Rfpos*dJDLdφspos
        dηintdφeneg = -1 - Rfneg*dJDLdφeneg
        dηintdφepos = -1 - Rfpos*dJDLdφepos

        # 初始化解X_，用切片索引绑定因变量
        X_ = zeros_like(bK_)
        θsnegsurf_ = X_[sK.s_csnegsurf]
        θspossurf_ = X_[sK.s_cspossurf]
        θe_ = X_[sK.s_ce]
        θeneg_ = X_[sK.s_ceneg]
        θepos_ = X_[sK.s_cepos]
        φsneg_ = X_[sK.s_φsneg]
        φspos_ = X_[sK.s_φspos]
        φe_ = X_[sK.s_φe]
        φeneg_ = X_[sK.s_φeneg]
        φepos_ = X_[sK.s_φepos]

        # 对X_赋初值
        Jintneg0_ = self.Jintneg_ if I==Ihistory_[-1] else  I
        Jintpos0_ = self.Jintpos_ if I==Ihistory_[-1] else -I
        θsnegsurf_[:] = αneg_ + βneg*Jintneg0_
        θspossurf_[:] = αpos_ + βpos*Jintpos0_
        θe_[:] = self.θe_
        I0intneg_ = self.I0intneg_ if I0intnegUnknown else I0intneg
        I0intpos_ = self.I0intpos_ if I0intposUnknown else I0intpos
        if I == Ihistory_[-1]:
            # 恒电流
            φsneg_[:] = self.φsneg_
            φspos_[:] = self.φspos_
            φe_[:] = self.φe_
        else:
            # 变电流瞬间
            F2RT = 0.5*P2Dbase.F/(P2Dbase.R*T)  # 常数 [1/V]
            ηintneg0_ = arcsinh( I/(2*I0intneg_))/F2RT
            ηintpos0_ = arcsinh(-I/(2*I0intpos_))/F2RT
            φsneg_[:] = ηintneg0_ + Rfneg*Jintneg0_ + solve_UOCPneg_(θsnegsurf_)
            φspos_[:] = ηintpos0_ + Rfpos*Jintpos0_ + solve_UOCPpos_(θspossurf_)

        ## Newton迭代预备
        J__ = K__.copy()       # (NK, NK) 初始化Jacobi矩阵
        ravelJ_ = J__.ravel()  # (NK*NK,) Jacobi矩阵展平视图
        F_ = empty_like(bK_)   # (NK,) F残差向量
        κDκT2ΔxWest_ = κDκT_[1:]  / ΔxWest_[1:]
        κDκT2ΔxEast_ = κDκT_[:-1] / ΔxEast_[:-1]
        _update_F_J__nonlinear_terms_JIT = LPJTFP2D._update_F_J__nonlinear_terms_JIT

        for nNewton in range(1, 201):
            ## Newton迭代

            # 更新4派生因变量
            Jintneg_ = (θsnegsurf_ - αneg_)/βneg  # 由θssurf反算Jint
            Jintpos_ = (θspossurf_ - αpos_)/βpos
            JDLneg_ = (bJDLneg_ + Cneg*(φsneg_ - φeneg_ - Rfneg*Jintneg_))/Dneg
            JDLpos_ = (bJDLpos_ + Cpos*(φspos_ - φepos_ - Rfpos*Jintpos_))/Dpos
            if I0intnegUnknown:
                I0intneg_ = solve_I0int_(kneg, θsnegsurf_, θeneg_)
            if I0intposUnknown:
                I0intpos_ = solve_I0int_(kpos, θspossurf_, θepos_)
            ηintneg_ = φsneg_ - φeneg_ - solve_UOCPneg_(θsnegsurf_) - Rfneg*(Jintneg_ + JDLneg_)
            ηintpos_ = φspos_ - φepos_ - solve_UOCPpos_(θspossurf_) - Rfpos*(Jintpos_ + JDLpos_)

            # 更新F向量线性部分
            F_[:] = K__.dot(X_) - bK_

            # 更新F向量非线性部分、J__矩阵非线性部分
            _update_F_J__nonlinear_terms_JIT(
                # 矩阵及其索引
                F_, ravelJ_, sK,
                # 网格参数
                Nneg, Nsep, Δx_, ΔxWest_, ΔxEast_,
                # 模式
                I0intnegUnknown, I0intposUnknown,
                # 电化学参数
                T,
                kneg, kpos,
                Rfneg, Rfpos,
                κDκT_, κ_, κDκT2ΔxWest_, κDκT2ΔxEast_,
                # 固相扩散消元结果
                βneg, βpos,
                # 因变量及其导数
                θsnegsurf_, θspossurf_, θe_, θeneg_, θepos_,
                I0intneg_, I0intpos_, ηintneg_, ηintpos_,
                dJdθsnegsurf_, dJdθspossurf_,
                dηintdφsneg, dηintdφeneg, dηintdφspos, dηintdφepos,
                solve_dUOCPdθsneg_(θsnegsurf_), solve_dUOCPdθspos_(θspossurf_))

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
            if (θsnegsurf_ <= 0).any():
                return nNewton, False, 'θsnegsurf<=0'
            if (θsnegsurf_ >= 1).any():
                return nNewton, False, 'θsnegsurf>=1'
            if (θspossurf_ <= 0).any():
                return nNewton, False, 'θspossurf<=0'
            if (θspossurf_ >= 1).any():
                return nNewton, False, 'θspossurf>=1'
            if θe_.min() < εI0:
                return nNewton, False, f'θe < {εI0}'

            ΔX_ = abs(ΔX_)
            maxΔθ = ΔX_[s_θ].max()  # 新旧无量纲浓度场最大绝对误差
            maxΔφ = ΔX_[s_φ].max()  # 新旧电势场最大绝对误差
            if maxΔθ < 2e-4 and maxΔφ < 2e-4:
                break
        else:
            t = self.t
            return nNewton, False, f't = {t} -> {t + Δt} s，Newton迭代达最大次数{nNewton}，{maxΔθ = :.4f}，{maxΔφ = :.6f} V'

        # Newton迭代收敛，更新状态量
        Jintneg_ = (θsnegsurf_ - αneg_)/βneg
        Jintpos_ = (θspossurf_ - αpos_)/βpos
        match self.timeDiscretization:
            case 'CN':
                self.θsneg__[:] = θsnegI__ + outer(γneg_, Jintneg_ + self.Jintneg_)
                self.θspos__[:] = θsposI__ + outer(γpos_, Jintpos_ + self.Jintpos_)
            case 'backward':
                self.θsneg__[:] = θsnegI__ + outer(γneg_, Jintneg_)
                self.θspos__[:] = θsposI__ + outer(γpos_, Jintpos_)
        self.θsnegsurf_[:] = θsnegsurf_
        self.θspossurf_[:] = θspossurf_
        self.θe_[:] = θe_
        self.φsneg_[:] = φsneg_
        self.φspos_[:] = φspos_
        self.φe_[:] = φe_
        self.Jintneg_[:] = Jintneg_
        self.Jintpos_[:] = Jintpos_
        self.JDLneg_[:] = JDLneg_ = (bJDLneg_ + Cneg*(φsneg_ - φeneg_) - Cneg*Rfneg*Jintneg_)/Dneg
        self.JDLpos_[:] = JDLpos_ = (bJDLpos_ + Cpos*(φspos_ - φepos_) - Cpos*Rfpos*Jintpos_)/Dpos
        self.I0intneg_[:] = solve_I0int_(kneg, θsnegsurf_, θeneg_) if I0intnegUnknown else I0intneg_
        self.I0intpos_[:] = solve_I0int_(kpos, θspossurf_, θepos_) if I0intposUnknown else I0intpos_
        self.ηintneg_[:] = φsneg_ - φeneg_ - solve_UOCPneg_(θsnegsurf_) - Rfneg*(Jintneg_ + JDLneg_)
        self.ηintpos_[:] = φspos_ - φepos_ - solve_UOCPpos_(θspossurf_) - Rfpos*(Jintpos_ + JDLpos_)
        return nNewton, True, None

    @staticmethod
    @njit(cache=True)
    def _update_F_J__nonlinear_terms_JIT(
            # 矩阵及其索引
            F_, ravelJ_, sK,
            # 网格参数
            Nneg, Nsep, Δx_, ΔxWest_, ΔxEast_,
            # 模式
            I0intnegUnknown, I0intposUnknown,
            # 电化学参数
            T, kneg, kpos, Rfneg, Rfpos,
            κDκT_, κ_, κDκT2ΔxWest_, κDκT2ΔxEast_,
            # 固相扩散消元结果
            βneg, βpos,
            # 因变量及其导数
            θsnegsurf_, θspossurf_, θe_, θeneg_, θepos_,
            I0intneg_, I0intpos_, ηintneg_, ηintpos_,
            dJdθsnegsurf, dJdθspossurf,
            dηintdφsneg, dηintdφeneg, dηintdφspos, dηintdφepos,
            dUOCPdθsnegsurf_, dUOCPdθspossurf_):
        # 更新残差向量F_非线性部分、Jacobi矩阵J__非线性部分

        ## 更新F_非线性部分
        # θssurf行
        F_[sK.s_csnegsurf] -= βneg*solve_Jint_(T, I0intneg_, ηintneg_)
        F_[sK.s_cspossurf] -= βpos*solve_Jint_(T, I0intpos_, ηintpos_)

        # φe行
        ΔFφe_ = zeros(θe_.size)
        θeW_ = θe_[:-1]
        θeE_ = θe_[1:]
        θeM_ = 0.5*(θeE_ + θeW_)  # (Ne-1,) 相邻无量纲浓度均值
        q_ = (θeE_ - θeW_)/θeM_   # (Ne-1,)
        ΔFφe_[1:]  += κDκT2ΔxWest_*q_
        ΔFφe_[:-1] -= κDκT2ΔxEast_*q_
        for nW, nE in zip((Nneg - 1, Nneg + Nsep - 1), (Nneg, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            a = κ_[nE]*Δx_[nW]
            b = κ_[nW]*Δx_[nE]
            θinterface = (a*θe_[nE] + b*θe_[nW])/(a + b)
            ΔFφe_[nW] = ( κDκT_[nW] * (θe_[nW] - θe_[nW-1])/ΔxWest_[nW]/(0.5*(θe_[nW] + θe_[nW-1]))
                         -κDκT_[nW] * (θinterface - θe_[nW])/(0.5*Δx_[nW])/θinterface)
            ΔFφe_[nE] = ( κDκT_[nE] * (θe_[nE] - θinterface)/(0.5*Δx_[nE])/θinterface
                         -κDκT_[nE] * (θe_[nE+1] - θe_[nE])/ΔxEast_[nE]/(0.5*(θe_[nE+1] + θe_[nE])))
        F_[sK.s_φe] += ΔFφe_

        ## 更新J__非线性部分
        # θssurf行
        dJintdI0intneg_ = solve_dJintdI0int_(T, ηintneg_)
        dJintdI0intpos_ = solve_dJintdI0int_(T, ηintpos_)
        dJintdηintneg_ = solve_dJintdηint_(T, I0intneg_, ηintneg_)
        dJintdηintpos_ = solve_dJintdηint_(T, I0intpos_, ηintpos_)
        dηintdθssurfneg_ = -dUOCPdθsnegsurf_ - Rfneg*dJdθsnegsurf
        dηintdθssurfpos_ = -dUOCPdθspossurf_ - Rfpos*dJdθspossurf
        if I0intnegUnknown:
            dI0intdθsnegsurf_ = solve_dI0intdθssurf_(kneg, θsnegsurf_, θeneg_, I0intneg_)
            dI0intdθeneg_     = solve_dI0intdθe_(θeneg_, I0intneg_)
        else:
            dI0intdθsnegsurf_ = dI0intdθeneg_ = zeros(Nneg)
        ravelJ_[sK.sr_csnegsurf_csnegsurf] = 1 - βneg*(dJintdI0intneg_*dI0intdθsnegsurf_ + dJintdηintneg_*dηintdθssurfneg_)
        ravelJ_[sK.sr_csnegsurf_ceneg] = -βneg*dJintdI0intneg_*dI0intdθeneg_
        ravelJ_[sK.sr_csnegsurf_φsneg] = -βneg*dηintdφsneg*dJintdηintneg_
        ravelJ_[sK.sr_csnegsurf_φeneg] = -βneg*dηintdφeneg*dJintdηintneg_
        if I0intposUnknown:
            dI0intdθspossurf_ = solve_dI0intdθssurf_(kpos, θspossurf_, θepos_, I0intpos_)
            dI0intdθepos_     = solve_dI0intdθe_(θepos_, I0intpos_)
        else:
            dI0intdθspossurf_ = dI0intdθepos_ = zeros(θspossurf_.size)
        ravelJ_[sK.sr_cspossurf_cspossurf] = 1 - βpos*(dJintdI0intpos_*dI0intdθspossurf_ + dJintdηintpos_*dηintdθssurfpos_)
        ravelJ_[sK.sr_cspossurf_cepos] = -βpos*dJintdI0intpos_*dI0intdθepos_
        ravelJ_[sK.sr_cspossurf_φspos] = -βpos*dηintdφspos*dJintdηintpos_
        ravelJ_[sK.sr_cspossurf_φepos] = -βpos*dηintdφepos*dJintdηintpos_

        # φe行θe列
        q_ *= 0.5
        a_ = κDκT2ΔxWest_/θeM_
        aa_ = a_*q_
        c_ = κDκT2ΔxEast_/θeM_
        cc_ = c_*q_
        ravelJ_[sK.sr_φe_ce_l] = -aa_ - a_  # 下对角线
        ravelJ_[sK.sr_φe_ce_u] = cc_ - c_   # 上对角线
        sr_φe_ce = sK.sr_φe_ce
        ravelJ_φe_θe_ = ravelJ_[sr_φe_ce]  # 主对角线
        ravelJ_φe_θe_[:] = 0.
        ravelJ_φe_θe_[:-1] += cc_ + c_
        ravelJ_φe_θe_[1:]  += a_ - aa_
        start0φece = sr_φe_ce.start
        NJ1 = F_.size + 1
        for nW, nE in zip((Nneg - 1, Nneg + Nsep - 1), (Nneg, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0φece + nW*NJ1
            nrE = start0φece + nE*NJ1
            κDκTWκE = κDκT_[nW]*κ_[nE]
            κDκTEκW = κDκT_[nE]*κ_[nW]
            num = κDκTWκE - κDκTEκW
            κEΔxW = κ_[nE]*Δx_[nW]
            κWΔxE = κ_[nW]*Δx_[nE]
            den1 = κEΔxW + κWΔxE
            den2 = κEΔxW*θe_[nE] + κWΔxE*θe_[nW]
            quotient = num/(den1*den2)
            ΔθeEW = θe_[nE] - θe_[nW]
            SθeW = θe_[nW] + θe_[nW-1]
            SθeE = θe_[nE] + θe_[nE+1]

            a = 2*κDκT_[nW]/(SθeW*Δx_[nW])
            aa = a*(θe_[nW] - θe_[nW-1])/SθeW
            c = 2*κWΔxE*quotient
            coeff = ΔθeEW*κEΔxW/den2
            cc = c*coeff
            d = 2*κDκTWκE/den2
            dd = d*coeff
            p = κWΔxE/κEΔxW
            ravelJ_[nrW-1:nrW+2] = -a - aa, -c - cc*p + d + dd*p + a - aa, c - cc - d + dd

            a = 2*κEΔxW*quotient
            coeff = ΔθeEW*κWΔxE/den2
            aa = a*coeff
            c = 2*κDκTEκW/den2
            cc = c*coeff
            d = 2*κDκT_[nE]/(SθeE*Δx_[nE])
            dd = d*(θe_[nE] - θe_[nE+1])/SθeE
            p = κEΔxW/κWΔxE
            ravelJ_[nrE-1:nrE+2] = -a - aa - c - cc, a - aa*p + c - cc*p + d - dd, -d - dd

    def count_lithium(self):
        """统计锂电荷量"""
        qsneg = self.θsneg*self.Qneg  # 负极固相锂电荷量 [Ah]
        qspos = self.θspos*self.Qpos  # 正极固相锂电荷量 [Ah]
        qe = (self.θe_*self.Δx_*self.qe_).sum()/3600  # 电解液锂电荷量 [Ah]
        qtot = qsneg + qspos + qe
        print(f'合计锂电荷总量 {qtot:.6f} Ah = '
              f'负极嵌锂{qsneg:.6f} Ah + 正极嵌锂{qspos:.6f} Ah'
              f' + 电解液锂{qe:.6f} Ah')

    @property
    def Qneg(self):
        """负极容量 [Ah]"""
        return self.Qcell*self.ξQneg

    @property
    def Qpos(self):
        """正极容量 [Ah]"""
        return self.Qcell*self.ξQpos

    @property
    def I0intneg(self):
        """负极主反应集总交换电流密度 [A]"""
        return self.Arrhenius(self._I0intneg, self.Ekneg)
    @I0intneg.setter
    def I0intneg(self, I0intneg):
        self._I0intneg = I0intneg

    @property
    def I0intpos(self):
        """正极主反应集总交换电流密度 [A]"""
        return self.Arrhenius(self._I0intpos, self.Ekpos)
    @I0intpos.setter
    def I0intpos(self, I0intpos):
        self._I0intpos = I0intpos

    @property
    def κneg(self):
        """负极电解液集总离子电导率 [S]"""
        return self.Arrhenius(self._κneg, self.Eκ)
    @κneg.setter
    def κneg(self, κneg):
        self._κneg = κneg

    @property
    def κsep(self):
        """隔膜电解液集总离子电导率 [S]"""
        return self.Arrhenius(self._κsep, self.Eκ)
    @κsep.setter
    def κsep(self, κsep):
        self._κsep = κsep

    @property
    def κpos(self):
        """正极电解液集总离子电导率 [S]"""
        return self.Arrhenius(self._κpos, self.Eκ)
    @κpos.setter
    def κpos(self, κpos):
        self._κpos = κpos

    @property
    def κ_(self):
        """(Ne,) 各控制体集总电解液离子电导率 [S]"""
        return concatenate([
            full(self.Nneg, self.κneg),
            full(self.Nsep, self.κsep),
            full(self.Npos, self.κpos)])

    @property
    def Deκ_(self):
        """(Ne,) 各控制体集总电解液锂离子扩散项系数 [A]"""
        De3κ_ = self.De * array([self._κneg, self._κsep, self._κpos])
        De3κ_ = self.Arrhenius(De3κ_, self.EDe)
        return concatenate([
            full(self.Nneg, De3κ_[0]),
            full(self.Nsep, De3κ_[1]),
            full(self.Npos, De3κ_[2])])

    @property
    def qe_(self):
        """(Ne,) 各控制体所属区域的电解液锂电荷量系数 [C]"""
        return concatenate([
            full(self.Nneg, self.qeneg),
            full(self.Nsep, self.qesep),
            full(self.Npos, self.qepos),])

    @property
    def U(self):
        """全电池端电压 [V]"""
        a = 0.5*self.I
        φsposCollector = self.φspos_[-1] - a*self.Δxpos/self.σpos
        φsnegCollector = self.φsneg_[0]  + a*self.Δxneg/self.σneg
        return φsposCollector - φsnegCollector + self.Ul

    @property
    def θeneg_(self):
        """(Nneg,) 负极区域电解液无量纲锂离子浓度 [–]"""
        return self.θe_[:self.Nneg]

    @property
    def θepos_(self):
        """(Npos,) 正极区域电解液无量纲锂离子浓度 [–]"""
        return self.θe_[-self.Npos:]

    @stepping_aware_cached_property
    def θeInterfaces_(self) -> ndarray:
        """(Ne+1,) 各控制体界面的无量纲锂离子浓度 [–]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_ = self.Δx_
        θe_ = self.θe_
        κ_ = self.κ_
        θeInterfaces_ = hstack([θe_[0], (θe_[:-1] + θe_[1:])/2, θe_[-1]])  # 各控制体界面的无量纲锂离子浓度
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            θeInterfaces_[nW + 1] = (a*θe_[nE] + b*θe_[nW])/(a + b)
        return θeInterfaces_

    @stepping_aware_cached_property
    def φeInterfaces_(self) -> ndarray:
        """(Ne+1,) 各控制体界面的电解液电势 [V]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_, ΔxWest_, ΔxEast_ = self.Δx_, self.ΔxWest_, self.ΔxEast_
        φe_, θe_ = self.φe_, self.θe_
        θeInterfaces_ = self.θeInterfaces_
        θeWest_ = θeInterfaces_[:-1]    # 各控制体左界面的电解液无量纲锂离子浓度
        θeEast_ = θeInterfaces_[1:]     # 各控制体右界面的电解液无量纲锂离子浓度
        gradθeWest_ = self.gradθeWest_  # 各控制体左界面的无量纲锂离子浓度梯度
        gradθeEast_ = self.gradθeEast_  # 各控制体右界面的无量纲锂离子浓度梯度
        gradlnθeWest_ = gradθeWest_/θeWest_  # 各控制体左界面的对数无量纲锂离子浓度梯度 [–/–]
        gradlnθeEast_ = gradθeEast_/θeEast_  # 各控制体右界面的对数无量纲锂离子浓度梯度 [–/–]
        φeInterfaces_ = hstack([φe_[0], (φe_[:-1] + φe_[1:])/2, φe_[-1]])
        κ_ = self.κ_
        κDκT_ = (self.κD*self.T) * κ_
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            a, b = κ_[nW]*Δx_[nE], κ_[nE]*Δx_[nW]
            c = 0.5*Δx_[nW]*Δx_[nE]
            φeInterfaces_[nE] = (  a*φe_[nW] + b*φe_[nE]
                                 + c*κDκT_[nW]*gradlnθeEast_[nW]
                                 - c*κDκT_[nE]*gradlnθeWest_[nE]
                                 )/(a + b)
        return φeInterfaces_

    # 保留类级调用接口，实际公式使用模块级njit函数，便于JIT kernel直接调用
    solve_Jint_ = staticmethod(solve_Jint_)

    @stepping_aware_cached_property
    def dJintdI0intneg_(self):
        """负极主反应集总局部体积电流密度 Jintneg 对集总交换电流密度 I0intneg 的偏导数 [A/A]"""
        return LPJTFP2D.solve_dJintdI0int_(self.T, self.ηintneg_)

    @stepping_aware_cached_property
    def dJintdI0intpos_(self):
        """正极主反应集总局部体积电流密度 Jintpos 对集总交换电流密度 I0intpos 的偏导数 [A/A]"""
        return LPJTFP2D.solve_dJintdI0int_(self.T, self.ηintpos_)

    solve_dJintdI0int_ = staticmethod(solve_dJintdI0int_)

    @stepping_aware_cached_property
    def dJintdηintneg_(self):
        """负极主反应集总局部体积电流密度 Jintneg 对过电位ηintneg的偏导数 [A/V]"""
        return LPJTFP2D.solve_dJintdηint_(self.T, self.I0intneg_, self.ηintneg_)

    @stepping_aware_cached_property
    def dJintdηintpos_(self):
        """正极主反应集总局部体积电流密度Jintpos对过电位ηintpos的偏导数 [A/V]"""
        return LPJTFP2D.solve_dJintdηint_(self.T, self.I0intpos_, self.ηintpos_)

    solve_dJintdηint_ = staticmethod(solve_dJintdηint_)

    solve_I0int_ = staticmethod(solve_I0int_)

    @stepping_aware_cached_property
    def dI0intdθsnegsurf_(self):
        """负极主反应集总交换电流密度 I0intneg 对负极表面嵌锂状态的偏导数 [A/–]"""
        return 0  if self._I0intneg\
            else LPJTFP2D.solve_dI0intdθssurf_(self.kneg, self.θsnegsurf_, self.θeneg_, self.I0intneg_)

    @stepping_aware_cached_property
    def dI0intdθspossurf_(self):
        """正极主反应集总交换电流密度 I0intpos 对正极表面嵌锂状态的偏导数 [A/–]"""
        return 0 if self._I0intpos\
            else LPJTFP2D.solve_dI0intdθssurf_(self.kpos, self.θspossurf_, self.θepos_, self.I0intpos_)

    solve_dI0intdθssurf_ = staticmethod(solve_dI0intdθssurf_)

    @stepping_aware_cached_property
    def dI0intdθeneg_(self):
        """负极主反应集总交换电流密度 I0int 对电解液无量纲浓度θe的偏导数 [A/–]"""
        return 0 if self._I0intneg \
            else LPJTFP2D.solve_dI0intdθe_(self.θeneg_, self.I0intneg_)

    @stepping_aware_cached_property
    def dI0intdθepos_(self):
        """正极主反应集总交换电流密度 I0int 对电解液无量纲浓度θe的偏导数 [A/–]"""
        return 0 if self._I0intpos \
            else LPJTFP2D.solve_dI0intdθe_(self.θepos_, self.I0intpos_)

    solve_dI0intdθe_ = staticmethod(solve_dI0intdθe_)

    @property
    def ηLPneg_(self):
        """负极析锂反应过电位场 [V]"""
        return self.φsneg_ - self.φeneg_ - self.Rfneg*self.Jneg_

    @property
    def ηLPpos_(self):
        """正极析锂反应过电位场 [V]"""
        return self.φspos_ - self.φepos_ - self.Rfpos*self.Jpos_

    @stepping_aware_cached_property
    def Jneg_(self):
        """负极总的集总局部体积电流密度场 [A]"""
        return self.Jintneg_ + self.JDLneg_

    @stepping_aware_cached_property
    def Jpos_(self):
        """正极总的集总局部体积电流密度场 [A]"""
        return self.Jintpos_ + self.JDLpos_

    @stepping_aware_cached_property
    def dUOCPdθsnegsurf_(self):
        """负极开路电位对负极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUOCPdθsneg_(self.θsnegsurf_)

    @stepping_aware_cached_property
    def dUOCPdθspossurf_(self):
        """正极开路电位对正极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUOCPdθspos_(self.θspossurf_)

    @stepping_aware_cached_property
    def gradlnθe_(self):
        """对数电解液无量纲锂离子浓度场的梯度 [–/–]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        x_, Δx_, ΔxWest_, ΔxEast_ = self.x_, self.Δx_, self.ΔxWest_, self.ΔxEast_
        θe_ = self.θe_
        θeInterfaces_ = self.θeInterfaces_
        θeWest_ = θeInterfaces_[:-1]  # 各控制体左界面的电解液无量纲锂离子浓度
        θeEast_ = θeInterfaces_[1:]   # 各控制体右界面的电解液无量纲锂离子浓度
        gradθe_ = hstack([
            (θe_[1] - θe_[0])/(x_[1] - x_[0]) * 0.5,       # 负极首个控制体 [–/–]
            (θe_[2:] - θe_[:-2])/(x_[2:] - x_[:-2]),       # 内部控制体 [–/–]
            (θe_[-1] - θe_[-2])/(x_[-1] - x_[-2]) * 0.5])  # 正极末尾控制体 [–/–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            gradθe_[nW] = ((θe_[nW] - θe_[nW-1])  /ΔxWest_[nW]  + (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW])) * 0.5  # 界面左侧控制体 [–/–]
            gradθe_[nE] = ((θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE]) + (θe_[nE+1] - θe_[nE]) / ΔxEast_[nE]) * 0.5   # 界面右侧控制体 [–/–]
        return gradθe_/θe_

    @stepping_aware_cached_property
    def gradθeWest_(self) -> ndarray:
        """(Ne,) 各控制体左侧界面电解液无量纲锂离子浓度梯度 [–/–]"""
        θe_ = self.θe_
        θeWest_ = self.θeInterfaces_[:-1]
        Nneg = self.Nneg
        gradθeWest_ = hstack([0, (θe_[1:] - θe_[:-1])/self.ΔxWest_[1:]])
        # 修正负极-隔膜、隔膜-正极界面
        idx_ = [Nneg, Nneg + self.Nsep]
        gradθeWest_[idx_] = (θe_[idx_] - θeWest_[idx_])/(0.5*self.Δx_[idx_])
        return gradθeWest_

    @stepping_aware_cached_property
    def gradθeEast_(self) -> ndarray:
        """(Ne,) 各控制体右侧界面电解液无量纲锂离子浓度梯度 [–/–]"""
        θe_ = self.θe_
        θeEast_ = self.θeInterfaces_[1:]
        Nneg = self.Nneg
        gradθeEast_ = hstack([(θe_[1:] - θe_[:-1])/self.ΔxEast_[:-1], 0])
        # 修正负极-隔膜、隔膜-正极界面
        idx_ = [Nneg - 1, Nneg + self.Nsep - 1]
        gradθeEast_[idx_] = (θeEast_[idx_] - θe_[idx_])/(0.5*self.Δx_[idx_])
        return gradθeEast_

    @property
    def gradφsneg_(self):
        """负极固相电势场的梯度 [V/–]"""
        φsneg_ = self.φsneg_
        Δxneg = self.Δxneg
        gradφsneg_ = hstack([
            (-self.I/self.σneg + (φsneg_[1] - φsneg_[0])/Δxneg) * 0.5, # 负极首个控制体
            (φsneg_[2:] - φsneg_[:-2])/(2*Δxneg),      # 负极内部控制体
            (φsneg_[-1] - φsneg_[-2])/Δxneg * 0.5])  # 负极末尾控制体
        return gradφsneg_

    @property
    def gradφspos_(self):
        """正极固相电势场的梯度 [V/–]"""
        φspos_ = self.φspos_
        Δxpos = self.Δxpos
        gradφspos_ = hstack([
            (0 + (φspos_[1] - φspos_[0])/Δxpos)/2,              # 正极首个控制体
            (φspos_[2:] - φspos_[:-2])/(2*Δxpos),               # 正极内部控制体
            ((φspos_[-1] - φspos_[-2])/Δxpos + -self.I/self.σpos)/2])  # 正极末尾控制体
        return gradφspos_

    @property
    def gradφe_(self):
        """电解液电势场的梯度∂φe/∂x [V/–]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        x_, Δx_, ΔxWest_, ΔxEast_ = self.x_, self.Δx_, self.ΔxWest_, self.ΔxEast_
        φe_ = self.φe_
        φeInterfaces_ = self.φeInterfaces_
        φeWest_ = φeInterfaces_[:-1]  # 各控制体左界面的电解液电势 [V]
        φeEast_ = φeInterfaces_[1:]   # 各控制体右界面的电解液电势 [V]
        gradφe_ = hstack([
            (0 + (φe_[1] - φe_[0])/(x_[1] - x_[0]))/2,       # 负极首个控制体
            (φe_[2:] - φe_[:-2])/(x_[2:] - x_[:-2]),         # 内部控制体
            ((φe_[-1] - φe_[-2])/(x_[-1] - x_[-2]) + 0)/2])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            gradφe_[nW] = ((φe_[nW] - φe_[nW - 1])/ΔxWest_[nW] + (φeEast_[nW] - φe_[nW])/(0.5*Δx_[nW]))/2
            gradφe_[nE] = ((φe_[nE] - φeWest_[nE])/(0.5*Δx_[nE]) + (φe_[nE + 1] - φe_[nE])/ΔxEast_[nE])/2
        return gradφe_

    @property
    def IDLneg(self):
        """负极双电层电流 [A]"""
        return self.JDLneg_.mean()

    @property
    def IDLpos(self):
        """正极双电层电流 [A]"""
        return self.JDLpos_.mean()

    @property
    def Qe(self):
        """电解液化学势产热功率 [W]"""
        gradφe_ = self.gradφe_
        return (self.κ_*(gradφe_ - self.κD*self.T*self.gradlnθe_)*gradφe_*self.Δx_).sum()

    @property
    def Qohmneg(self):
        """负极固相欧姆热功率 [W]"""
        gradφsneg_ = self.gradφsneg_
        return self.σneg*(gradφsneg_*gradφsneg_).mean()

    @property
    def Qohmpos(self):
        """正极固相欧姆热功率 [W]"""
        gradφspos_ = self.gradφspos_
        return self.σpos*(gradφspos_*gradφspos_).mean()

    @property
    def Qrxnneg(self):
        """负极反应热功率 [W]"""
        return (self.Jintneg_*self.ηintneg_).mean()

    @property
    def Qrxnpos(self):
        """正极反应热功率 [W]"""
        return (self.Jintpos_*self.ηintpos_).mean()

    @property
    def QSEIneg(self):
        """负极SEI膜电阻热功率 [W]"""
        Jneg_ = self.Jneg_
        return self.Rfneg*(Jneg_*Jneg_).mean()

    @property
    def QSEIpos(self):
        """正极SEI膜电阻热功率 [W]"""
        Jpos_ = self.Jpos_
        return self.Rfpos*(Jpos_*Jpos_).mean()

    @property
    def Qrevneg(self):
        """负极可逆热功率 [W]"""
        dUOCPdTnegsurf_ = self.dUOCPdTneg(self.θsnegsurf_) if callable(self.dUOCPdTneg) else self.dUOCPdTneg
        return (self.T*dUOCPdTnegsurf_*self.Jintneg_).mean()

    @property
    def Qrevpos(self):
        """正极可逆热功率 [W]"""
        dUOCPdTpossurf_ = self.dUOCPdTpos(self.θspossurf_) if callable(self.dUOCPdTpos) else self.dUOCPdTpos
        return (self.T*dUOCPdTpossurf_*self.Jintpos_).mean()

    @property
    def θsneg(self):
        """负极嵌锂状态 [–]"""
        return self.Vr_.dot(self.θsneg__).mean()

    @property
    def θspos(self):
        """正极嵌锂状态 [–]"""
        return self.Vr_.dot(self.θspos__).mean()

    def initialize_consistent(self,
            θsneg__: ndarray,
            θspos__: ndarray,
            θe_: ndarray,
            I: float | int = 0,
            T: float | int = 298.15):
        # 一致性初始化
        # 已知：θsneg__、θspos__、θe_、I、T
        # 基本因变量：φsneg_、φspos_、φe_
        # 派生量：θsnegsurf_、θspossurf_、Jintneg_、Jintpos_、I0intneg_、I0intpos_、ηintneg_、ηintpos_
        # 令：JDLneg_ = JDLpos_ = 0，Jint_由固相电势方程显式反算
        self.T = T; assert T>0, f'温度{T = }，应大于0 [K]'
        Nr, Nneg, Nsep, Npos, Ne = self.Nr, self.Nneg, self.Nsep, self.Npos, self.Ne  # 读取：网格数
        assert θsneg__.shape==(Nr, Nneg), f'负极固相颗粒内部无量纲锂离子浓度θsneg__.shape应为({Nr}, {Nneg})'
        assert θspos__.shape==(Nr, Npos), f'正极固相颗粒内部无量纲锂离子浓度θspos__.shape应为({Nr}, {Npos})'
        assert θe_.shape==(self.Ne,), f'电解液无量纲锂离子浓度θe_.shape应为({Ne},)'
        assert ((0<=θsneg__) & (θsneg__<=1)).all(), 'θsneg__取值范围应为[0, 1]'
        assert ((0<=θspos__) & (θspos__<=1)).all(), 'θspos__取值范围应为[0, 1]'
        assert (0<θe_).all(), 'θe_取值应大于0'

        # 外推表面浓度
        c_ = self.coeffsExpl_
        θsnegsurf_ = c_.dot(θsneg__[-3:])
        θspossurf_ = c_.dot(θspos__[-3:])
        if (θsnegsurf_<=0).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到θsnegsurf<=0')
        if (θsnegsurf_>=1).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到θsnegsurf>=1')
        if (θspossurf_<=0).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到θspossurf<=0')
        if (θspossurf_>=1).any():
            raise P2Dbase.Error('一致性初始化失败，外推得到θspossurf>=1')

        # 负极、正极电解液无量纲锂离子浓度
        θeneg_, θepos_ = θe_[:Nneg], θe_[-Npos:]

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
        solve_Jint_ = LPJTFP2D.solve_Jint_
        solve_dJintdηint_  = LPJTFP2D.solve_dJintdηint_
        solve_I0int_ = LPJTFP2D.solve_I0int_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_

        # 读取参数
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        T = self.T   # 温度 [K]
        κ_ = self.κ_
        κDκT_ = (self.κD * T) * κ_
        σneg, σpos = self.σneg, self.σpos
        Rfneg = self.Rfneg
        Rfpos = self.Rfpos
        if I0intnegUnknown := (self._I0intneg is None):
            kneg = self.kneg          # 读取：负极主反应集总速率常数 [A]
        else:
            I0intneg = self.I0intneg  # 读取：负极主反应集总交换电流密度 [A]
        if I0intposUnknown := (self._I0intpos is None):
            kpos = self.kpos          # 读取：正极主反应集总速率常数 [A]
        else:
            I0intpos = self.I0intpos  # 读取：正极主反应集总交换电流密度 [A]

        # 显式计算主反应集总交换电流密度
        I0intneg_ = solve_I0int_(kneg, θsnegsurf_, θeneg_) if I0intnegUnknown else I0intneg
        I0intpos_ = solve_I0int_(kpos, θspossurf_, θepos_) if I0intposUnknown else I0intpos

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
        KφsnegJ = -Δxneg*Δxneg/σneg
        KφsposJ = -Δxpos*Δxpos/σpos
        # φe行φe列
        dl_ = ravelKinit_[dsr(s_φe, s_φe, -1)]
        du_ = ravelKinit_[dsr(s_φe, s_φe, 1)]
        d_  = ravelKinit_[sr_φe_φe]
        dl_[:] = κ_[1:] / ΔxWest_[1:]
        du_[:] = κ_[:-1] / ΔxEast_[:-1]
        d_[:] = -(hstack([0, dl_]) + hstack([du_, 0]))
        d_[0] -= κ_[0]/(0.5*Δx_[0])  # 首元占优，固定电解液电势参考
        start0 = sr_φe_φe.start
        NKinit1 = NKinit + 1
        for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NKinit1
            nrE = start0 + nE*NKinit1
            a, c = κ_[nW]/ΔxWest_[nW], 2*κ_[nW]*κ_[nE]/(κ_[nW]*Δx_[nE] + κ_[nE]*Δx_[nW])
            ravelKinit_[nrW-1:nrW+2] = a, -(a + c), c
            a, c = c, κ_[nE]/ΔxEast_[nE]
            ravelKinit_[nrE-1:nrE+2] = a, -(a + c), c
        # φe行φs列
        Kinit__[s_φeneg, s_φsneg] = -Δxneg/KφsnegJ*Kinit__[s_φsneg, s_φsneg]
        Kinit__[s_φepos, s_φspos] = -Δxpos/KφsposJ*Kinit__[s_φspos, s_φspos]

        # 赋值bKinit_向量
        # φs行
        bKinit_[s_φsneg.start]    = -Δxneg*I/σneg  # 固相电流边界条件
        bKinit_[s_φspos.stop - 1] =  Δxpos*I/σpos
        # φe行
        q_ = 2*(θe_[1:] - θe_[:-1])/(θe_[1:] + θe_[:-1])  # (Ne-1,)
        c_ = κDκT_[:-1]*q_ / ΔxEast_[:-1]  # (Ne-1,)
        a_ = κDκT_[1:] *q_ / ΔxWest_[1:]   # (Ne-1,)
        bKinit_[s_φe] = hstack([c_, 0]) - hstack([0, a_])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            θinterface = (a*θe_[nE] + b*θe_[nW])/(a + b)
            bKinit_[s_φe.start + nW] = (κDκT_[nW] * (θinterface - θe_[nW])  / (0.5*Δx_[nW]) / θinterface
                                      - κDκT_[nW] * (θe_[nW] - θe_[nW-1])   / ΔxWest_[nW]   / (0.5*(θe_[nW] + θe_[nW-1])))
            bKinit_[s_φe.start + nE] = (κDκT_[nE] * (θe_[nE+1] - θe_[nE])   / ΔxEast_[nE]   / (0.5*(θe_[nE+1] + θe_[nE]))
                                      - κDκT_[nE] * (θe_[nE] - θinterface)  / (0.5*Δx_[nE]) / θinterface)
        bKinit_[s_φeneg.start]    += -Δxneg/KφsnegJ*bKinit_[s_φsneg.start]
        bKinit_[s_φepos.stop - 1] += -Δxpos/KφsposJ*bKinit_[s_φspos.stop - 1]

        ## Newton迭代初值 ##
        X_ = zeros(NKinit)
        # 索引解向量
        φsneg_ = X_[s_φsneg]
        φspos_ = X_[s_φspos]
        φe_ = X_[s_φe]
        φeneg_ = X_[s_φeneg]
        φepos_ = X_[s_φepos]

        # 初始化解向量
        Jintneg0 = I
        Jintpos0 = -I
        F2RT = P2Dbase.F/(2*P2Dbase.R*T)
        ηintneg0_ = arcsinh(Jintneg0/(2*I0intneg_))/F2RT
        ηintpos0_ = arcsinh(Jintpos0/(2*I0intpos_))/F2RT
        φsneg_[:] = ηintneg0_ + Rfneg*Jintneg0 + solve_UOCPneg_(θsnegsurf_)
        φspos_[:] = ηintpos0_ + Rfpos*Jintpos0 + solve_UOCPpos_(θspossurf_)

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
            Jintneg_ = F_[s_φsneg]/-KφsnegJ
            Jintpos_ = F_[s_φspos]/-KφsposJ
            ηintneg_ = φsneg_ - φeneg_ - solve_UOCPneg_(θsnegsurf_) - Rfneg*Jintneg_
            ηintpos_ = φspos_ - φepos_ - solve_UOCPpos_(θspossurf_) - Rfpos*Jintpos_
            dJintdηintneg_ = solve_dJintdηint_(T, I0intneg_, ηintneg_)
            dJintdηintpos_ = solve_dJintdηint_(T, I0intpos_, ηintpos_)

            # F向量非线性部分
            # φs行：由固相电势方程反算的Jint应满足BV方程
            F_[s_φsneg] += KφsnegJ*solve_Jint_(T, I0intneg_, ηintneg_)
            F_[s_φspos] += KφsposJ*solve_Jint_(T, I0intpos_, ηintpos_)

            # 更新J__矩阵非线性部分
            J_φsneg_φsneg__[:] = Kinit_φsneg_φsneg__ + Rfneg*dJintdηintneg_[:, None]*Kinit_φsneg_φsneg__
            J_φspos_φspos__[:] = Kinit_φspos_φspos__ + Rfpos*dJintdηintpos_[:, None]*Kinit_φspos_φspos__
            ravelJ_[sr_φsneg_φsneg] += KφsnegJ*dJintdηintneg_
            ravelJ_[sr_φspos_φspos] += KφsposJ*dJintdηintpos_
            ravelJ_[sr_φsneg_φeneg] = -KφsnegJ*dJintdηintneg_
            ravelJ_[sr_φspos_φepos] = -KφsposJ*dJintdηintpos_

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
        self.θsneg__[:] = θsneg__
        self.θspos__[:] = θspos__
        self.θsnegsurf_[:] = θsnegsurf_
        self.θspossurf_[:] = θspossurf_
        self.θe_[:] = θe_
        self.φsneg_[:] = φsneg_
        self.φspos_[:] = φspos_
        self.φe_[:] = φe_
        self.Jintneg_[:] = Jintneg_ = F_[s_φsneg]/-KφsnegJ
        self.Jintpos_[:] = Jintpos_ = F_[s_φspos]/-KφsposJ
        self.JDLneg_[:] = 0
        self.JDLpos_[:] = 0
        self.I0intneg_[:] = I0intneg_
        self.I0intpos_[:] = I0intpos_
        self.ηintneg_[:] = φsneg_ - φeneg_ - solve_UOCPneg_(θsnegsurf_) - Rfneg*Jintneg_
        self.ηintpos_[:] = φspos_ - φepos_ - solve_UOCPpos_(θspossurf_) - Rfpos*Jintpos_

        if self.verbose:
            print(f'一致性初始化完成。Newton迭代{nNewton = }。Consistent initial conditions are solved! ')
        return self

    def _update_Kf__bKf_with_pure_parameters(self):
        # 更新Kf__矩阵的纯电化学参数相关项
        LPJTFP2D._update_Kf__bKf_with_pure_parameters_JIT(
            # 矩阵
            self.ravelKf_, self.bKf_, self.sKf,
            # 网格参数
            self.Nneg, self.Nsep, self.ΔxWest_, self.ΔxEast_, self.Δx_,
            # 电化学参数
            self.ΔIAC, self.σneg, self.σpos, self.Deκ_,)

    @staticmethod
    @njit(cache=True)
    def _update_Kf__bKf_with_pure_parameters_JIT(
            # 矩阵
            ravelKf_, bKf_, sKf,
            # 网格参数
            Nneg, Nsep, ΔxWest_, ΔxEast_, Δx_,
            # 电化学参数
            ΔIAC, σneg, σpos, Deκ_,
            ):
        # 更新Kf__矩阵的纯电化学参数相关项
        # REθe行REθe列
        dl_ = ravelKf_[sKf.sr_REce_REce_l]
        du_ = ravelKf_[sKf.sr_REce_REce_u]
        sr_REce_REce = sKf.sr_REce_REce
        d_ = ravelKf_[sr_REce_REce]       # 主对角线
        dl_[:] = -Deκ_[1:]/ΔxWest_[1:]    # 下对角线
        du_[:] = -Deκ_[:-1]/ΔxEast_[:-1]  # 上对角线
        d_[0] = -du_[0]
        d_[1:-1] = -(dl_[:-1] + du_[1:])
        d_[-1] = -dl_[-1]
        start0 = sr_REce_REce.start
        NKf1 = bKf_.size + 1
        for (nW, nE) in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜、隔膜-正极界面
            nrW = start0 + nW*NKf1
            nrE = start0 + nE*NKf1
            a, c = -Deκ_[nW]/ΔxWest_[nW], -2*Deκ_[nW]*Deκ_[nE]/(Deκ_[nE]*Δx_[nW] + Deκ_[nW]*Δx_[nE])
            ravelKf_[nrW - 1:nrW + 2] = a, -(a + c), c
            a, c = c, -Deκ_[nE]/ΔxEast_[nE]
            ravelKf_[nrE - 1:nrE + 2] = a, -(a + c), c
        # IMθe行IMθe列
        ravelKf_[sKf.sr_IMce_IMce] = d_
        ravelKf_[sKf.sr_IMce_IMce_l] = dl_
        ravelKf_[sKf.sr_IMce_IMce_u] = du_

        ## 更新bKf_向量
        Δxneg, Δxpos = Δx_[0], Δx_[-1]
        bKf_[sKf.s_REφsneg.start]    = -Δxneg*ΔIAC/σneg
        bKf_[sKf.s_REφspos.stop - 1] =  Δxpos*ΔIAC/σpos

    def _update_Kf__with_states(self):
        # 更新Kf__矩阵的状态（含“参数+状态”）相关项
        NKf1 = self.bKf_.size + 1
        LPJTFP2D._update_Kf__with_states_JIT(
            # 矩阵
            self.ravelKf_, self.sKf, NKf1,
            # 网格参数
            self.Nneg, self.Nsep, self.ΔxWest_, self.ΔxEast_, self.Δx_,
            # 电化学参数
            self.κD, self.T, self.κ_, self.Deκ_,
            # 状态及其偏导数
            self.θeInterfaces_, self.gradθeWest_, self.gradθeEast_,
            self.dJintdI0intneg_, self.dI0intdθeneg_,
            self.dJintdI0intpos_, self.dI0intdθepos_,)

    @staticmethod
    @njit(cache=True)
    def _update_Kf__with_states_JIT(
            # 矩阵
            ravelKf_, sKf, NKf1,
            # 网格参数
            Nneg, Nsep, ΔxWest_, ΔxEast_, Δx_,
            # 电化学参数
            κD, T, κ_, Deκ_,
            # 状态及其偏导数
            θeInterfaces_, gradθeWest_, gradθeEast_,
            dJintdI0intneg_, dI0intdθeneg_,
            dJintdI0intpos_, dI0intdθepos_,):
        # 更新Kf__矩阵的状态（含“参数+状态”）相关项

        # 读取状态及其偏导数
        θeWest_ = θeInterfaces_[:-1]
        θeEast_ = θeInterfaces_[1:]
        dREJintBVdREθeneg_ = dJintdI0intneg_*dI0intdθeneg_
        dIMJintBVdIMθeneg_ = dREJintBVdREθeneg_
        dREJintBVdREθepos_ = dJintdI0intpos_*dI0intdθepos_
        dIMJintBVdIMθepos_ = dREJintBVdREθepos_

        # 读取参数
        κDκT_ = (κD*T)*κ_

        # REθssurf行REθe列
        ravelKf_[sKf.sr_REcsnegsurf_REceneg] = -dREJintBVdREθeneg_
        ravelKf_[sKf.sr_REcspossurf_REcepos] = -dREJintBVdREθepos_
        # IMθssurf行IMθe列
        ravelKf_[sKf.sr_IMcsnegsurf_IMceneg] = -dIMJintBVdIMθeneg_
        ravelKf_[sKf.sr_IMcspossurf_IMcepos] = -dIMJintBVdIMθepos_
        # REφe行REθe列
        sr_REφe_REce = sKf.sr_REφe_REce
        d_REφe_REce_  = ravelKf_[sr_REφe_REce]
        dl_REφe_REce_ = ravelKf_[sKf.sr_REφe_REce_l]
        du_REφe_REce_ = ravelKf_[sKf.sr_REφe_REce_u]
        κDκT2θeWest_ = κDκT_[1:]/θeWest_[1:]
        κDκT2θeEast_ = κDκT_[:-1]/θeEast_[:-1]
        a_ = κDκT2θeWest_/ΔxWest_[1:]
        c_ = κDκT2θeEast_/ΔxEast_[:-1]
        aa_ = κDκT2θeWest_*gradθeWest_[1:]/θeWest_[1:]*0.5
        cc_ = κDκT2θeEast_*gradθeEast_[:-1]/θeEast_[:-1]*0.5
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
            Deκ_nE_Δx_nW = Deκ_[nE]*Δx_[nW]
            Deκ_nW_Δx_nE = Deκ_[nW]*Δx_[nE]
            den = Deκ_nE_Δx_nW + Deκ_nW_Δx_nE
            pDW = Deκ_nW_Δx_nE/den
            pDE = 1 - pDW
            κDκT2θeWest = κDκT_[nW]/θeWest_[nW]
            κDκT2θeEast = κDκT_[nW]/θeEast_[nW]
            a = κDκT2θeWest/ΔxWest_[nW]
            aa = κDκT2θeWest*gradθeWest_[nW]/θeWest_[nW]*0.5
            c = κDκT2θeEast*Deκ_[nE]/den*2
            cc = κDκT2θeEast*gradθeEast_[nW]/θeEast_[nW]
            ravelKf_[nrW-1:nrW+2] = -aa - a, a - aa + c + cc*pDW, cc*pDE - c
            κDκT2θeWest = κDκT_[nE]/θeWest_[nE]
            κDκT2θeEast = κDκT_[nE]/θeEast_[nE]
            a = κDκT2θeWest*Deκ_[nW]/den*2
            aa = κDκT2θeWest*gradθeWest_[nE]/θeWest_[nE]
            c = κDκT2θeEast/ΔxEast_[nE]
            cc = κDκT2θeEast*gradθeEast_[nE]/θeEast_[nE]*0.5
            ravelKf_[nrE-1:nrE+2] = -a - aa*pDW, a - aa*pDE + c + cc, cc - c
        # IMφe行IMθe列
        ravelKf_[sKf.sr_IMφe_IMce]   = d_REφe_REce_
        ravelKf_[sKf.sr_IMφe_IMce_l] = dl_REφe_REce_
        ravelKf_[sKf.sr_IMφe_IMce_u] = du_REφe_REce_

    def _update_Kf__with_frequencies(self, ravelKf__, base_ravelKf_):
        # 更新所有频率Kf__矩阵频率相关项
        LPJTFP2D._update_Kf__with_frequencies_JIT(
            # 矩阵
            ravelKf__, base_ravelKf_, self.sKf, self.bKf_.size + 1,
            # 网格参数
            self.Nneg, self.Nsep, self.Npos, self.ΔxWest_, self.ΔxEast_, self.Δx_,
            # 电化学参数
            self.Rfneg, self.Rfpos, self.σneg, self.σpos, self.κ_,
            self.qe_, self.Qneg, self.Qpos, self.Dsneg, self.Dspos,
            self.CDLneg, self.CDLpos,
            # 状态偏导数
            self.dJintdI0intneg_, self.dJintdI0intpos_,
            self.dJintdηintneg_, self.dJintdηintpos_,
            self.dI0intdθsnegsurf_, self.dI0intdθspossurf_,
            self.dUOCPdθsnegsurf_, self.dUOCPdθspossurf_,
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
            Rfneg, Rfpos, σneg, σpos, κ_,
            qe_, Qneg, Qpos, Dsneg, Dspos,
            CDLneg, CDLpos,
            # 状态偏导数
            dJintdI0intneg_, dJintdI0intpos_,
            dJintdηintneg_, dJintdηintpos_,
            dI0intdθsnegsurf_, dI0intdθspossurf_,
            dUOCPdθsnegsurf_, dUOCPdθspossurf_,
            # 频率
            ω_,
            ):
        # 更新所有频率Kf__矩阵频率相关项
        Kθsnegsurf___, Kθspossurf___, ωqeΔx__ = solve_frequency_dependent_variables(
            ω_, qe_, Δx_, Qneg, Qpos, Dsneg, Dspos)
        for nf in range(ω_.size):
            ravelKf_ = ravelKf__[nf]
            ravelKf_[:] = base_ravelKf_
            ω = ω_[nf]
            Kθsnegsurf__ = Kθsnegsurf___[nf]
            Kθspossurf__ = Kθspossurf___[nf]
            ωqeΔx_ = ωqeΔx__[nf]

            det = Kθsnegsurf__[0, 0]*Kθsnegsurf__[1, 1] - Kθsnegsurf__[0, 1]*Kθsnegsurf__[1, 0]
            dREJintdREθsnegsurf =  Kθsnegsurf__[1, 1]/det
            dREJintdIMθsnegsurf = -Kθsnegsurf__[0, 1]/det
            dIMJintdREθsnegsurf = -Kθsnegsurf__[1, 0]/det
            dIMJintdIMθsnegsurf =  Kθsnegsurf__[0, 0]/det
            det = Kθspossurf__[0, 0]*Kθspossurf__[1, 1] - Kθspossurf__[0, 1]*Kθspossurf__[1, 0]
            dREJintdREθspossurf =  Kθspossurf__[1, 1]/det
            dREJintdIMθspossurf = -Kθspossurf__[0, 1]/det
            dIMJintdREθspossurf = -Kθspossurf__[1, 0]/det
            dIMJintdIMθspossurf =  Kθspossurf__[0, 0]/det

            ωCDLneg = ω*CDLneg
            ωCDLRfneg = ωCDLneg*Rfneg
            den = 1 + ωCDLRfneg*ωCDLRfneg
            HDLneg2RRE = ωCDLneg*ωCDLRfneg/den
            HDLneg2RIM = ωCDLneg/den
            a = -Rfneg*dREJintdREθsnegsurf
            b = -Rfneg*dIMJintdREθsnegsurf
            dREJDLdREθsnegsurf = HDLneg2RRE*a - HDLneg2RIM*b
            dIMJDLdREθsnegsurf = HDLneg2RRE*b + HDLneg2RIM*a
            a = -Rfneg*dREJintdIMθsnegsurf
            b = -Rfneg*dIMJintdIMθsnegsurf
            dREJDLdIMθsnegsurf = HDLneg2RRE*a - HDLneg2RIM*b
            dIMJDLdIMθsnegsurf = HDLneg2RRE*b + HDLneg2RIM*a
            dREJDLdREφsneg = HDLneg2RRE
            dREJDLdIMφsneg = -HDLneg2RIM
            dIMJDLdREφsneg = HDLneg2RIM
            dIMJDLdIMφsneg = HDLneg2RRE
            dREJDLdREφeneg = -HDLneg2RRE
            dREJDLdIMφeneg = HDLneg2RIM
            dIMJDLdREφeneg = -HDLneg2RIM
            dIMJDLdIMφeneg = -HDLneg2RRE
            dREJdREθsnegsurf = dREJintdREθsnegsurf + dREJDLdREθsnegsurf
            dREJdIMθsnegsurf = dREJintdIMθsnegsurf + dREJDLdIMθsnegsurf
            dIMJdREθsnegsurf = dIMJintdREθsnegsurf + dIMJDLdREθsnegsurf
            dIMJdIMθsnegsurf = dIMJintdIMθsnegsurf + dIMJDLdIMθsnegsurf
            dREJdREφsneg = dREJDLdREφsneg
            dREJdIMφsneg = dREJDLdIMφsneg
            dIMJdREφsneg = dIMJDLdREφsneg
            dIMJdIMφsneg = dIMJDLdIMφsneg
            dREJdREφeneg = dREJDLdREφeneg
            dREJdIMφeneg = dREJDLdIMφeneg
            dIMJdREφeneg = dIMJDLdREφeneg
            dIMJdIMφeneg = dIMJDLdIMφeneg
            dREηintdREθsnegsurf_ = -dUOCPdθsnegsurf_ - Rfneg*dREJdREθsnegsurf
            dREηintdIMθsnegsurf = -Rfneg*dREJdIMθsnegsurf
            dIMηintdREθsnegsurf = -Rfneg*dIMJdREθsnegsurf
            dIMηintdIMθsnegsurf_ = -dUOCPdθsnegsurf_ - Rfneg*dIMJdIMθsnegsurf
            dREηintdREφsneg = 1 - Rfneg*dREJdREφsneg
            dREηintdIMφsneg = -Rfneg*dREJdIMφsneg
            dIMηintdREφsneg = -Rfneg*dIMJdREφsneg
            dIMηintdIMφsneg = 1 - Rfneg*dIMJdIMφsneg
            dREηintdREφeneg = -1 - Rfneg*dREJdREφeneg
            dREηintdIMφeneg = -Rfneg*dREJdIMφeneg
            dIMηintdREφeneg = -Rfneg*dIMJdREφeneg
            dIMηintdIMφeneg = -1 - Rfneg*dIMJdIMφeneg

            dREJintBVdREθsnegsurf_ = dJintdI0intneg_*dI0intdθsnegsurf_ + dJintdηintneg_*dREηintdREθsnegsurf_
            dREJintBVdIMθsnegsurf_ = dJintdηintneg_*dREηintdIMθsnegsurf
            dIMJintBVdREθsnegsurf_ = dJintdηintneg_*dIMηintdREθsnegsurf
            dIMJintBVdIMθsnegsurf_ = dJintdI0intneg_*dI0intdθsnegsurf_ + dJintdηintneg_*dIMηintdIMθsnegsurf_
            dREJintBVdREφsneg_ = dJintdηintneg_*dREηintdREφsneg
            dREJintBVdIMφsneg_ = dJintdηintneg_*dREηintdIMφsneg
            dIMJintBVdREφsneg_ = dJintdηintneg_*dIMηintdREφsneg
            dIMJintBVdIMφsneg_ = dJintdηintneg_*dIMηintdIMφsneg
            dREJintBVdREφeneg_ = dJintdηintneg_*dREηintdREφeneg
            dREJintBVdIMφeneg_ = dJintdηintneg_*dREηintdIMφeneg
            dIMJintBVdREφeneg_ = dJintdηintneg_*dIMηintdREφeneg
            dIMJintBVdIMφeneg_ = dJintdηintneg_*dIMηintdIMφeneg

            ωCDLpos = ω*CDLpos
            ωCDLRfpos = ωCDLpos*Rfpos
            den = 1 + ωCDLRfpos*ωCDLRfpos
            HDLpos2RRE = ωCDLpos*ωCDLRfpos/den
            HDLpos2RIM = ωCDLpos/den
            a = -Rfpos*dREJintdREθspossurf
            b = -Rfpos*dIMJintdREθspossurf
            dREJDLdREθspossurf = HDLpos2RRE*a - HDLpos2RIM*b
            dIMJDLdREθspossurf = HDLpos2RRE*b + HDLpos2RIM*a
            a = -Rfpos*dREJintdIMθspossurf
            b = -Rfpos*dIMJintdIMθspossurf
            dREJDLdIMθspossurf = HDLpos2RRE*a - HDLpos2RIM*b
            dIMJDLdIMθspossurf = HDLpos2RRE*b + HDLpos2RIM*a
            dREJDLdREφspos = HDLpos2RRE
            dREJDLdIMφspos = -HDLpos2RIM
            dIMJDLdREφspos = HDLpos2RIM
            dIMJDLdIMφspos = HDLpos2RRE
            dREJDLdREφepos = -HDLpos2RRE
            dREJDLdIMφepos = HDLpos2RIM
            dIMJDLdREφepos = -HDLpos2RIM
            dIMJDLdIMφepos = -HDLpos2RRE
            dREJdREθspossurf = dREJintdREθspossurf + dREJDLdREθspossurf
            dREJdIMθspossurf = dREJintdIMθspossurf + dREJDLdIMθspossurf
            dIMJdREθspossurf = dIMJintdREθspossurf + dIMJDLdREθspossurf
            dIMJdIMθspossurf = dIMJintdIMθspossurf + dIMJDLdIMθspossurf
            dREJdREφspos = dREJDLdREφspos
            dREJdIMφspos = dREJDLdIMφspos
            dIMJdREφspos = dIMJDLdREφspos
            dIMJdIMφspos = dIMJDLdIMφspos
            dREJdREφepos = dREJDLdREφepos
            dREJdIMφepos = dREJDLdIMφepos
            dIMJdREφepos = dIMJDLdREφepos
            dIMJdIMφepos = dIMJDLdIMφepos
            dREηintdREθspossurf = -dUOCPdθspossurf_ - Rfpos*dREJdREθspossurf
            dREηintdIMθspossurf = -Rfpos*dREJdIMθspossurf
            dIMηintdREθspossurf = -Rfpos*dIMJdREθspossurf
            dIMηintdIMθspossurf = -dUOCPdθspossurf_ - Rfpos*dIMJdIMθspossurf
            dREηintdREφspos = 1 - Rfpos*dREJdREφspos
            dREηintdIMφspos = -Rfpos*dREJdIMφspos
            dIMηintdREφspos = -Rfpos*dIMJdREφspos
            dIMηintdIMφspos = 1 - Rfpos*dIMJdIMφspos
            dREηintdREφepos = -1 - Rfpos*dREJdREφepos
            dREηintdIMφepos = -Rfpos*dREJdIMφepos
            dIMηintdREφepos = -Rfpos*dIMJdREφepos
            dIMηintdIMφepos = -1 - Rfpos*dIMJdIMφepos
            dREJintBVdREθspossurf_ = dJintdI0intpos_*dI0intdθspossurf_ + dJintdηintpos_*dREηintdREθspossurf
            dREJintBVdIMθspossurf_ = dJintdηintpos_*dREηintdIMθspossurf
            dIMJintBVdREθspossurf_ = dJintdηintpos_*dIMηintdREθspossurf
            dIMJintBVdIMθspossurf_ = dJintdI0intpos_*dI0intdθspossurf_ + dJintdηintpos_*dIMηintdIMθspossurf
            dREJintBVdREφspos_ = dJintdηintpos_*dREηintdREφspos
            dREJintBVdIMφspos_ = dJintdηintpos_*dREηintdIMφspos
            dIMJintBVdREφspos_ = dJintdηintpos_*dIMηintdREφspos
            dIMJintBVdIMφspos_ = dJintdηintpos_*dIMηintdIMφspos
            dREJintBVdREφepos_ = dJintdηintpos_*dREηintdREφepos
            dREJintBVdIMφepos_ = dJintdηintpos_*dREηintdIMφepos
            dIMJintBVdREφepos_ = dJintdηintpos_*dIMηintdREφepos
            dIMJintBVdIMφepos_ = dJintdηintpos_*dIMηintdIMφepos

            ## 更新Kf__矩阵

            # REθsnegsurf行
            ravelKf_[sKf.sr_REcsnegsurf_REcsnegsurf] = dREJintdREθsnegsurf - dREJintBVdREθsnegsurf_
            ravelKf_[sKf.sr_REcsnegsurf_IMcsnegsurf] = dREJintdIMθsnegsurf - dREJintBVdIMθsnegsurf_
            ravelKf_[sKf.sr_REcsnegsurf_REφsneg] = -dREJintBVdREφsneg_
            ravelKf_[sKf.sr_REcsnegsurf_IMφsneg] = -dREJintBVdIMφsneg_
            ravelKf_[sKf.sr_REcsnegsurf_REφeneg] = -dREJintBVdREφeneg_
            ravelKf_[sKf.sr_REcsnegsurf_IMφeneg] = -dREJintBVdIMφeneg_
            # IMθsnegsurf行
            ravelKf_[sKf.sr_IMcsnegsurf_REcsnegsurf] = dIMJintdREθsnegsurf - dIMJintBVdREθsnegsurf_
            ravelKf_[sKf.sr_IMcsnegsurf_IMcsnegsurf] = dIMJintdIMθsnegsurf - dIMJintBVdIMθsnegsurf_
            ravelKf_[sKf.sr_IMcsnegsurf_REφsneg] = -dIMJintBVdREφsneg_
            ravelKf_[sKf.sr_IMcsnegsurf_IMφsneg] = -dIMJintBVdIMφsneg_
            ravelKf_[sKf.sr_IMcsnegsurf_REφeneg] = -dIMJintBVdREφeneg_
            ravelKf_[sKf.sr_IMcsnegsurf_IMφeneg] = -dIMJintBVdIMφeneg_
            # REθspossurf行
            ravelKf_[sKf.sr_REcspossurf_REcspossurf] = dREJintdREθspossurf - dREJintBVdREθspossurf_
            ravelKf_[sKf.sr_REcspossurf_IMcspossurf] = dREJintdIMθspossurf - dREJintBVdIMθspossurf_
            ravelKf_[sKf.sr_REcspossurf_REφspos] = -dREJintBVdREφspos_
            ravelKf_[sKf.sr_REcspossurf_IMφspos] = -dREJintBVdIMφspos_
            ravelKf_[sKf.sr_REcspossurf_REφepos] = -dREJintBVdREφepos_
            ravelKf_[sKf.sr_REcspossurf_IMφepos] = -dREJintBVdIMφepos_
            # IMθspossurf行
            ravelKf_[sKf.sr_IMcspossurf_REcspossurf] = dIMJintdREθspossurf - dIMJintBVdREθspossurf_
            ravelKf_[sKf.sr_IMcspossurf_IMcspossurf] = dIMJintdIMθspossurf - dIMJintBVdIMθspossurf_
            ravelKf_[sKf.sr_IMcspossurf_REφspos] = -dIMJintBVdREφspos_
            ravelKf_[sKf.sr_IMcspossurf_IMφspos] = -dIMJintBVdIMφspos_
            ravelKf_[sKf.sr_IMcspossurf_REφepos] = -dIMJintBVdREφepos_
            ravelKf_[sKf.sr_IMcspossurf_IMφepos] = -dIMJintBVdIMφepos_

            # 读取参数
            Δxneg, Δxpos = Δx_[0], Δx_[-1]
            # REθe行
            ravelKf_[sKf.sr_REceneg_REcsnegsurf] = -Δxneg*dREJdREθsnegsurf
            ravelKf_[sKf.sr_REceneg_IMcsnegsurf] = -Δxneg*dREJdIMθsnegsurf
            ravelKf_[sKf.sr_REcepos_REcspossurf] = -Δxpos*dREJdREθspossurf
            ravelKf_[sKf.sr_REcepos_IMcspossurf] = -Δxpos*dREJdIMθspossurf
            ravelKf_[sKf.sr_REce_IMce] = -ωqeΔx_
            ravelKf_[sKf.sr_REceneg_REφsneg] = -Δxneg*dREJdREφsneg
            ravelKf_[sKf.sr_REceneg_IMφsneg] = -Δxneg*dREJdIMφsneg
            ravelKf_[sKf.sr_REcepos_REφspos] = -Δxpos*dREJdREφspos
            ravelKf_[sKf.sr_REcepos_IMφspos] = -Δxpos*dREJdIMφspos
            ravelKf_[sKf.sr_REceneg_REφeneg] = -Δxneg*dREJdREφeneg
            ravelKf_[sKf.sr_REceneg_IMφeneg] = -Δxneg*dREJdIMφeneg
            ravelKf_[sKf.sr_REcepos_REφepos] = -Δxpos*dREJdREφepos
            ravelKf_[sKf.sr_REcepos_IMφepos] = -Δxpos*dREJdIMφepos
            # IMθe行
            ravelKf_[sKf.sr_IMceneg_REcsnegsurf] = -Δxneg*dIMJdREθsnegsurf
            ravelKf_[sKf.sr_IMceneg_IMcsnegsurf] = -Δxneg*dIMJdIMθsnegsurf
            ravelKf_[sKf.sr_IMcepos_REcspossurf] = -Δxpos*dIMJdREθspossurf
            ravelKf_[sKf.sr_IMcepos_IMcspossurf] = -Δxpos*dIMJdIMθspossurf
            ravelKf_[sKf.sr_IMce_REce] = ωqeΔx_
            ravelKf_[sKf.sr_IMceneg_REφsneg] = -Δxneg*dIMJdREφsneg
            ravelKf_[sKf.sr_IMceneg_IMφsneg] = -Δxneg*dIMJdIMφsneg
            ravelKf_[sKf.sr_IMcepos_REφspos] = -Δxpos*dIMJdREφspos
            ravelKf_[sKf.sr_IMcepos_IMφspos] = -Δxpos*dIMJdIMφspos
            ravelKf_[sKf.sr_IMceneg_REφeneg] = -Δxneg*dIMJdREφeneg
            ravelKf_[sKf.sr_IMceneg_IMφeneg] = -Δxneg*dIMJdIMφeneg
            ravelKf_[sKf.sr_IMcepos_REφepos] = -Δxpos*dIMJdREφepos
            ravelKf_[sKf.sr_IMcepos_IMφepos] = -Δxpos*dIMJdIMφepos

            KφsnegJ = -Δxneg*Δxneg/σneg
            KφsposJ = -Δxpos*Δxpos/σpos
            # REφsneg行
            ravelKf_[sKf.sr_REφsneg_REcsnegsurf] = KφsnegJ*dREJdREθsnegsurf
            ravelKf_[sKf.sr_REφsneg_IMcsnegsurf] = KφsnegJ*dREJdIMθsnegsurf
            d_REφsneg_REφsneg_ = ravelKf_[sKf.sr_REφsneg_REφsneg]
            d_REφsneg_REφsneg_[:] = -2. + KφsnegJ*dREJdREφsneg
            d_REφsneg_REφsneg_[0] += 1.
            d_REφsneg_REφsneg_[-1] += 1.
            ravelKf_[sKf.sr_REφsneg_IMφsneg] = KφsnegJ*dREJdIMφsneg
            ravelKf_[sKf.sr_REφsneg_REφeneg] = KφsnegJ*dREJdREφeneg
            ravelKf_[sKf.sr_REφsneg_IMφeneg] = KφsnegJ*dREJdIMφeneg
            # REφspos行
            ravelKf_[sKf.sr_REφspos_REcspossurf] = KφsposJ*dREJdREθspossurf
            ravelKf_[sKf.sr_REφspos_IMcspossurf] = KφsposJ*dREJdIMθspossurf
            d_REφspos_REφspos_ = ravelKf_[sKf.sr_REφspos_REφspos]
            d_REφspos_REφspos_[:] = -2. + KφsposJ*dREJdREφspos
            d_REφspos_REφspos_[0] += 1.
            d_REφspos_REφspos_[-1] += 1.
            ravelKf_[sKf.sr_REφspos_IMφspos] = KφsposJ*dREJdIMφspos
            ravelKf_[sKf.sr_REφspos_REφepos] = KφsposJ*dREJdREφepos
            ravelKf_[sKf.sr_REφspos_IMφepos] = KφsposJ*dREJdIMφepos
            # IMφsneg行
            ravelKf_[sKf.sr_IMφsneg_REcsnegsurf] = KφsnegJ*dIMJdREθsnegsurf
            ravelKf_[sKf.sr_IMφsneg_IMcsnegsurf] = KφsnegJ*dIMJdIMθsnegsurf
            ravelKf_[sKf.sr_IMφsneg_REφsneg] = KφsnegJ*dIMJdREφsneg
            d_IMφsneg_IMφsneg_ = ravelKf_[sKf.sr_IMφsneg_IMφsneg]
            d_IMφsneg_IMφsneg_[:] = -2. + KφsnegJ*dIMJdIMφsneg
            d_IMφsneg_IMφsneg_[0] += 1.
            d_IMφsneg_IMφsneg_[-1] += 1.
            ravelKf_[sKf.sr_IMφsneg_REφeneg] = KφsnegJ*dIMJdREφeneg
            ravelKf_[sKf.sr_IMφsneg_IMφeneg] = KφsnegJ*dIMJdIMφeneg
            # IMφspos行
            ravelKf_[sKf.sr_IMφspos_REcspossurf] = KφsposJ*dIMJdREθspossurf
            ravelKf_[sKf.sr_IMφspos_IMcspossurf] = KφsposJ*dIMJdIMθspossurf
            ravelKf_[sKf.sr_IMφspos_REφspos] = KφsposJ*dIMJdREφspos
            d_IMφspos_IMφspos_ = ravelKf_[sKf.sr_IMφspos_IMφspos]
            d_IMφspos_IMφspos_[:] = -2. + KφsposJ*dIMJdIMφspos
            d_IMφspos_IMφspos_[0] += 1.
            d_IMφspos_IMφspos_[-1] += 1.
            ravelKf_[sKf.sr_IMφspos_REφepos] = KφsposJ*dIMJdREφepos
            ravelKf_[sKf.sr_IMφspos_IMφepos] = KφsposJ*dIMJdIMφepos

            # REφe行
            ravelKf_[sKf.sr_REφeneg_REcsnegsurf] = Δxneg*dREJdREθsnegsurf
            ravelKf_[sKf.sr_REφeneg_IMcsnegsurf] = Δxneg*dREJdIMθsnegsurf
            ravelKf_[sKf.sr_REφepos_REcspossurf] = Δxpos*dREJdREθspossurf
            ravelKf_[sKf.sr_REφepos_IMcspossurf] = Δxpos*dREJdIMθspossurf
            ravelKf_[sKf.sr_REφeneg_REφsneg] = Δxneg*dREJdREφsneg
            ravelKf_[sKf.sr_REφeneg_IMφsneg] = Δxneg*dREJdIMφsneg
            ravelKf_[sKf.sr_REφepos_REφspos] = Δxpos*dREJdREφspos
            ravelKf_[sKf.sr_REφepos_IMφspos] = Δxpos*dREJdIMφspos
            d_REφe_REφe_  = ravelKf_[sKf.sr_REφe_REφe]
            dl_REφe_REφe_ = ravelKf_[sKf.sr_REφe_REφe_l]
            du_REφe_REφe_ = ravelKf_[sKf.sr_REφe_REφe_u]
            dl_REφe_REφe_[:] = κ_[1:]/ΔxWest_[1:]
            du_REφe_REφe_[:] = κ_[:-1]/ΔxEast_[:-1]
            d_REφe_REφe_[0] = -du_REφe_REφe_[0]
            d_REφe_REφe_[1:-1] = -(dl_REφe_REφe_[:-1] + du_REφe_REφe_[1:])
            d_REφe_REφe_[-1] = -dl_REφe_REφe_[-1]
            d_REφe_REφe_[0] -= κ_[0]/(0.5*Δx_[0])
            start0 = sKf.sr_REφe_REφe.start
            for (nW, nE) in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
                # 修正负极-隔膜、隔膜-正极界面
                nrW = start0 + nW*NKf1
                nrE = start0 + nE*NKf1
                a = κ_[nW]/ΔxWest_[nW]
                c = 2*κ_[nW]*κ_[nE]/(κ_[nE]*Δx_[nW] + κ_[nW]*Δx_[nE])
                ravelKf_[nrW-1:nrW+2] = a, -(a + c), c
                a, c = c, κ_[nE]/ΔxEast_[nE]
                ravelKf_[nrE-1:nrE+2] = a, -(a + c), c
            base_d_REφe_REφe_  = d_REφe_REφe_.copy()
            base_dl_REφe_REφe_ = dl_REφe_REφe_.copy()
            base_du_REφe_REφe_ = du_REφe_REφe_.copy()
            ravelKf_[sKf.sr_REφe_REφe][:Nneg]  += Δxneg*dREJdREφeneg
            ravelKf_[sKf.sr_REφe_REφe][-Npos:] += Δxpos*dREJdREφepos
            ravelKf_[sKf.sr_REφe_IMφe][:Nneg]  = Δxneg*dREJdIMφeneg
            ravelKf_[sKf.sr_REφe_IMφe][-Npos:] = Δxpos*dREJdIMφepos

            # IMφe行
            ravelKf_[sKf.sr_IMφeneg_REcsnegsurf] = Δxneg*dIMJdREθsnegsurf
            ravelKf_[sKf.sr_IMφeneg_IMcsnegsurf] = Δxneg*dIMJdIMθsnegsurf
            ravelKf_[sKf.sr_IMφepos_REcspossurf] = Δxpos*dIMJdREθspossurf
            ravelKf_[sKf.sr_IMφepos_IMcspossurf] = Δxpos*dIMJdIMθspossurf
            ravelKf_[sKf.sr_IMφeneg_REφsneg] = Δxneg*dIMJdREφsneg
            ravelKf_[sKf.sr_IMφeneg_IMφsneg] = Δxneg*dIMJdIMφsneg
            ravelKf_[sKf.sr_IMφepos_REφspos] = Δxpos*dIMJdREφspos
            ravelKf_[sKf.sr_IMφepos_IMφspos] = Δxpos*dIMJdIMφspos
            ravelKf_[sKf.sr_IMφe_REφe][:Nneg]  = Δxneg*dIMJdREφeneg
            ravelKf_[sKf.sr_IMφe_REφe][-Npos:] = Δxpos*dIMJdREφepos
            ravelKf_[sKf.sr_IMφe_IMφe]   = base_d_REφe_REφe_
            ravelKf_[sKf.sr_IMφe_IMφe_l] = base_dl_REφe_REφe_
            ravelKf_[sKf.sr_IMφe_IMφe_u] = base_du_REφe_REφe_
            ravelKf_[sKf.sr_IMφe_IMφe][:Nneg]  += Δxneg*dIMJdIMφeneg
            ravelKf_[sKf.sr_IMφe_IMφe][-Npos:] += Δxpos*dIMJdIMφepos

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
        hΔIAC  = 0.5*ΔIAC
        REφsnegCollector_ = REφsneg__[:, 0]  + hΔIAC*self.Δxneg/self.σneg  # (Nf,) 负极集流体电势实部 [V]
        IMφsnegCollector_ = IMφsneg__[:, 0]                                # (Nf,) 负极集流体电势虚部 [V]
        REφsposCollector_ = REφspos__[:, -1] - hΔIAC*self.Δxpos/self.σpos  # (Nf,) 正极集流体电势实部 [V]
        IMφsposCollector_ = IMφspos__[:, -1]                    # (Nf,) 正极集流体电势虚部 [V]
        Zreal_ = (REφsposCollector_ - REφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗实部 [Ω]
        Zimag_ = (IMφsposCollector_ - IMφsnegCollector_)/-ΔIAC  # (Nf,) 全电池阻抗虚部 [Ω]
        self.Z_[:] = Zreal_ + 1j*Zimag_ + self.Zl_              # (Nf,) 全电池复阻抗 [Ω]

        if self.complete:
            # 读取参数
            Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos
            CDLneg = self.CDLneg
            CDLpos = self.CDLpos
            Rfneg = self.Rfneg
            Rfpos = self.Rfpos
            κ_ = self.κ_
            Δx_ = self.Δx_
            # 读取状态偏导数
            dI0intdθsnegsurf_ = self.dI0intdθsnegsurf_
            dI0intdθspossurf_ = self.dI0intdθspossurf_
            dI0intdθeneg_ = self.dI0intdθeneg_
            dI0intdθepos_ = self.dI0intdθepos_
            dUOCPdθsnegsurf_ = self.dUOCPdθsnegsurf_
            dUOCPdθspossurf_ = self.dUOCPdθspossurf_
            Kθsnegsurf___, Kθspossurf___, _ = solve_frequency_dependent_variables(
                ω_, self.qe_, self.Δx_,
                self.Qneg, self.Qpos,
                self.Dsneg, self.Dspos)

            self.REθsnegsurf__[:] = REθsnegsurf__ = X__[:, sKf.s_REcsnegsurf]  # 负极固相表面无量纲浓度实部
            self.IMθsnegsurf__[:] = IMθsnegsurf__ = X__[:, sKf.s_IMcsnegsurf]  # 负极固相表面无量纲浓度虚部
            self.REθspossurf__[:] = REθspossurf__ = X__[:, sKf.s_REcspossurf]  # 正极固相表面无量纲浓度实部
            self.IMθspossurf__[:] = IMθspossurf__ = X__[:, sKf.s_IMcspossurf]  # 正极固相表面无量纲浓度虚部
            self.REθe__[:] = REθe__ = X__[:, sKf.s_REce]  # 电解液无量纲锂离子浓度实部
            self.IMθe__[:] = IMθe__ = X__[:, sKf.s_IMce]  # 电解液无量纲锂离子浓度虚部
            self.REφsneg__[:] = REφsneg__  # 负极固相电势实部
            self.IMφsneg__[:] = IMφsneg__  # 负极固相电势虚部
            self.REφspos__[:] = REφspos__  # 正极固相电势实部
            self.IMφspos__[:] = IMφspos__  # 正极固相电势虚部
            self.REφe__[:] = REφe__ = X__[:, sKf.s_REφe]   # 电解液电势实部
            self.IMφe__[:] = IMφe__ = X__[:, sKf.s_IMφe]   # 电解液电势虚部

            inv__ = empty((Nf, 2, 2))

            inv__[:] = batch_inv_2x2(Kθsnegsurf___)
            results__ = inv__ @ stack([REθsnegsurf__, IMθsnegsurf__], axis=1)
            self.REJintneg__[:] = REJintneg__ = results__[:, 0, :]
            self.IMJintneg__[:] = IMJintneg__ = results__[:, 1, :]
            inv__[:] = batch_inv_2x2(Kθspossurf___)
            results__ = inv__ @ stack([REθspossurf__, IMθspossurf__], axis=1)
            self.REJintpos__[:] = REJintpos__ = results__[:, 0, :]
            self.IMJintpos__[:] = IMJintpos__ = results__[:, 1, :]

            A__ = ones((Nf, 2, 2))

            A__[:, 0, 1] = -CDLneg*Rfneg*ω_
            A__[:, 1, 0] = -A__[:, 0, 1]
            inv__[:] = batch_inv_2x2(A__)
            results__ = inv__ @ stack([
                -ω_[:, None]*CDLneg*(IMφsneg__ - IMφe__[:, :Nneg] - Rfneg*IMJintneg__),
                 ω_[:, None]*CDLneg*(REφsneg__ - REφe__[:, :Nneg] - Rfneg*REJintneg__)], axis=1)
            self.REJDLneg__[:] = REJDLneg__ = results__[:, 0, :]
            self.IMJDLneg__[:] = IMJDLneg__ = results__[:, 1, :]

            A__[:, 0, 1] = -CDLpos*Rfpos*ω_
            A__[:, 1, 0] = -A__[:, 0, 1]
            inv__[:] = batch_inv_2x2(A__)
            results__ = inv__ @ stack([
                -ω_[:, None]*CDLpos*(IMφspos__ - IMφe__[:, -Npos:] - Rfpos*IMJintpos__),
                 ω_[:, None]*CDLpos*(REφspos__ - REφe__[:, -Npos:] - Rfpos*REJintpos__)], axis=1)
            self.REJDLpos__[:] = REJDLpos__ = results__[:, 0, :]
            self.IMJDLpos__[:] = IMJDLpos__ = results__[:, 1, :]

            self.REI0intneg__[:] = dI0intdθsnegsurf_*REθsnegsurf__ + dI0intdθeneg_*REθe__[:, :Nneg]
            self.IMI0intneg__[:] = dI0intdθsnegsurf_*IMθsnegsurf__ + dI0intdθeneg_*IMθe__[:, :Nneg]
            self.REI0intpos__[:] = dI0intdθspossurf_*REθspossurf__ + dI0intdθepos_*REθe__[:, -Npos:]
            self.IMI0intpos__[:] = dI0intdθspossurf_*IMθspossurf__ + dI0intdθepos_*IMθe__[:, -Npos:]

            self.REηintneg__[:] = REφsneg__ - REφe__[:, :Nneg] - dUOCPdθsnegsurf_*REθsnegsurf__  - Rfneg*(REJintneg__ + REJDLneg__)
            self.IMηintneg__[:] = IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdθsnegsurf_*IMθsnegsurf__  - Rfneg*(IMJintneg__ + IMJDLneg__)
            self.REηintpos__[:] = REφspos__ - REφe__[:, -Npos:] - dUOCPdθspossurf_*REθspossurf__ - Rfpos*(REJintpos__ + REJDLpos__)
            self.IMηintpos__[:] = IMφspos__ - IMφe__[:, -Npos:] - dUOCPdθspossurf_*IMθspossurf__ - Rfpos*(IMJintpos__ + IMJDLpos__)

            nW, nE = Nneg - 1, Nneg
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            den = a + b
            REφenegsep_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφenegsep_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφenegsep_ - REφsnegCollector_)/-ΔIAC
            Zimag_ = (IMφenegsep_ - IMφsnegCollector_)/-ΔIAC
            self.Zneg_[:] = Zreal_ + 1j*Zimag_  # 负极阻抗

            nW, nE = Nneg + Nsep - 1, Nneg + Nsep
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            den = a + b
            REφeseppos_ = (b*REφe__[:, nW] + a*REφe__[:, nE])/den
            IMφeseppos_ = (b*IMφe__[:, nW] + a*IMφe__[:, nE])/den
            Zreal_ = (REφsposCollector_ - REφeseppos_)/-ΔIAC
            Zimag_ = (IMφsposCollector_ - IMφeseppos_)/-ΔIAC
            self.Zpos_[:] = Zreal_ + 1j*Zimag_ # 正极阻抗

        if self.verbose:
            print(f'计算时刻{tEIS = :.1f} s 电化学阻抗谱')
        self.record_EISdata()
        return self

    @staticmethod
    def solve_REθs__IMθs__(
            r: float,     # 径向坐标 [–]
            ω_: ndarray,  # (Nf,) 角频率序列 [rad/s]
            Q: float,     # 电极容量 [Ah]
            Ds: float,    # 集总固相锂离子扩散系数 [1/s]
            REJint__: ndarray,  # (Nf, Nreg) 集总局部体积电流密度实部 [A]
            IMJint__: ndarray,  # (Nf, Nreg) 集总局部体积电流密度虚部 [A]
            ) -> tuple[ndarray, ndarray]:
        """固相无量纲锂离子浓度实部、虚部在r处的解析解"""
        W2_ = ω_/Ds
        W_ = sqrt(W2_)
        root2 = 1.4142135623730951
        root2W_ = root2*W_
        ψ_ = .7071067811865476*W_
        ψr_ = ψ_*r
        sinψ_ = sin(ψ_)
        cosψ_ = cos(ψ_)
        sinψr_ = sin(ψr_)
        cosψr_ = cos(ψr_)
        Q6Dsr = 21600 * Q * Ds * r
        # 指数缩放
        # coshψ coshψr sinhψ sinhψr 是 ~exp(ψ) 级别的大数，容易溢出，不能直接算
        # 应缩放：统一乘exp(-ψ)
        exp_ψ_ = exp(-ψ_)
        exp_2ψ_ = exp_ψ_*exp_ψ_
        half_exp_2ψ_ = 0.5*exp_2ψ_
        coshψ_s_ = 0.5 + half_exp_2ψ_  # coshψ*exp(-ψ_)
        sinhψ_s_ = 0.5 - half_exp_2ψ_  # sinhψ*exp(-ψ_)

        exp_ψr_ = exp(-ψr_)
        q_ = exp(ψ_*(r - 1))
        p_ = exp_ψ_ * exp_ψr_
        coshψr_s_ = 0.5 * (q_ + p_)  # coshψr*exp(-ψ_)
        sinhψr_s_ = 0.5 * (q_ - p_)  # sinhψr*exp(-ψ_)

        a_ = ( - root2W_ * coshψr_s_ * sinψr_ * coshψ_s_ * cosψ_
               + 2       * coshψr_s_ * sinψr_ * coshψ_s_ * sinψ_
               - root2W_ * coshψr_s_ * sinψr_ * sinhψ_s_ * sinψ_
               - root2W_ * sinhψr_s_ * cosψr_ * coshψ_s_ * cosψ_
               + root2W_ * sinhψr_s_ * cosψr_ * sinhψ_s_ * sinψ_
               + 2       * sinhψr_s_ * cosψr_ * sinhψ_s_ * cosψ_ )

        b_ = (   root2W_ * coshψr_s_ * sinψr_ * coshψ_s_ * cosψ_
               - root2W_ * coshψr_s_ * sinψr_ * sinhψ_s_ * sinψ_
               - 2       * coshψr_s_ * sinψr_ * sinhψ_s_ * cosψ_
               - root2W_ * sinhψr_s_ * cosψr_ * coshψ_s_ * cosψ_
               + 2       * sinhψr_s_ * cosψr_ * coshψ_s_ * sinψ_
               - root2W_ * sinhψr_s_ * cosψr_ * sinhψ_s_ * sinψ_ )

        d_ = Q6Dsr*( (W2_ + 1) * coshψ_s_ * coshψ_s_
                    - root2W_  * coshψ_s_ * sinhψ_s_
                    - W2_      * sinψ_    * sinψ_ * exp_2ψ_
                    - root2W_  * cosψ_    * sinψ_ * exp_2ψ_
                    -            cosψ_    * cosψ_ * exp_2ψ_ )
        a_ /= d_
        b_ /= d_
        Kθs___ = empty((ω_.size, 2, 2))
        Kθs___[:, 0, 0] = Kθs___[:, 1, 1] = a_
        Kθs___[:, 0, 1] = b_
        Kθs___[:, 1, 0] = -b_
        results___ = Kθs___ @ stack([REJint__, IMJint__], axis=1)  # (Nf, 2, Nreg)
        REθs__, IMθs__ = results___[:, 0, :], results___[:, 1, :]
        return REθs__, IMθs__  # (Nf, Nreg)

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
        REθsnegsurf__ = self.REθsnegsurf__  # 负极固相表面无量纲浓度实部
        IMθsnegsurf__ = self.IMθsnegsurf__  # 负极固相表面无量纲浓度虚部
        REθspossurf__ = self.REθspossurf__  # 正极固相表面无量纲浓度实部
        IMθspossurf__ = self.IMθspossurf__  # 正极固相表面无量纲浓度虚部
        REθe__ = self.REθe__            # 电解液无量纲锂离子浓度实部
        IMθe__ = self.IMθe__            # 电解液无量纲锂离子浓度虚部
        REφsneg__ = self.REφsneg__      # 负极固相电势实部
        IMφsneg__ = self.IMφsneg__      # 负极固相电势虚部
        REφspos__ = self.REφspos__      # 正极固相电势实部
        IMφspos__ = self.IMφspos__      # 正极固相电势虚部
        REφe__ = self.REφe__            # 电解液电势实部
        IMφe__ = self.IMφe__            # 电解液电势虚部
        REJintneg__ = self.REJintneg__  # 负极主反应集总局部体积电流密度实部
        IMJintneg__ = self.IMJintneg__  # 负极主反应集总局部体积电流密度虚部
        REJintpos__ = self.REJintpos__  # 正极主反应集总局部体积电流密度实部
        IMJintpos__ = self.IMJintpos__  # 正极主反应集总局部体积电流密度虚部
        REJDLneg__ = self.REJDLneg__    # 负极双电层集总局部体积电流密度实部
        IMJDLneg__ = self.IMJDLneg__    # 负极双电层集总局部体积电流密度虚部
        REJDLpos__ = self.REJDLpos__    # 正极双电层集总局部体积电流密度实部
        IMJDLpos__ = self.IMJDLpos__    # 正极双电层集总局部体积电流密度虚部
        REI0intneg__ = self.REI0intneg__  # 负极主反应集总交换电流密度实部
        IMI0intneg__ = self.IMI0intneg__  # 负极主反应集总交换电流密度虚部
        REI0intpos__ = self.REI0intpos__  # 正极主反应集总交换电流密度实部
        IMI0intpos__ = self.IMI0intpos__  # 正极主反应集总交换电流密度虚部
        REηintneg__ = self.REηintneg__  # 负极过电位实部
        IMηintneg__ = self.IMηintneg__  # 负极过电位虚部
        REηintpos__ = self.REηintpos__  # 正极过电位实部
        IMηintpos__ = self.IMηintpos__  # 正极过电位虚部
        Nf = self.f_.size
        ω_ = self.ω_
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        Δxneg, Δxpos = self.Δxneg, self.Δxpos
        x_ = self.x_
        ΔIAC = self.ΔIAC
        σneg, σpos = self.σneg, self.σpos
        Rfneg, Rfpos = self.Rfneg, self.Rfpos
        CDLneg, CDLpos = self.CDLneg, self.CDLpos
        κ_ = self.κ_
        qe_ = self.qe_
        κDκT_ = (self.κD*self.T)*κ_
        Deκ_ = self.Deκ_
        F2RT = 0.5*P2Dbase.F/(P2Dbase.R*self.T)
        I0intneg_, I0intpos_ = self.I0intneg_, self.I0intpos_
        ηintneg_, ηintpos_ = self.ηintneg_, self.ηintpos_
        θe_, θeInterfaces_ = self.θe_, self.θeInterfaces_
        θeWest_, θeEast_ = θeInterfaces_[:-1], θeInterfaces_[1:]
        gradθeWest_ = hstack([0, (θe_[1:] - θe_[:-1])/ΔxWest_[1:]])   # (Ne,) 各控制体左界面的无量纲锂离子浓度梯度 [–/–]
        gradθeEast_ = hstack([(θe_[1:] - θe_[:-1])/ΔxEast_[:-1], 0])  # (Ne,) 各控制体右界面的无量纲锂离子浓度梯度 [–/–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、修正隔膜-正极界面
            gradθeEast_[nW] = (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW])
            gradθeWest_[nE] = (θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE])

        # 各控制体界面的电解液无量纲锂离子浓度实部 [–]
        REθeInterfaces__ = hstack([REθe__[:, [0]], (REθe__[:, :-1] + REθe__[:, 1:])*0.5, REθe__[:, [-1]]])  # (Nf, Ne+1)
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面锂离子浓度
            REθeInterfaces__[:, nE] = (Deκ_[nW]*Δx_[nE]*REθe__[:, nW] + Deκ_[nE]*Δx_[nW]*REθe__[:, nE])/(Deκ_[nW]*Δx_[nE] + Deκ_[nE]*Δx_[nW])
        REθeWest__ = REθeInterfaces__[:, :-1]  # 各控制体左界面的电解液无量纲锂离子浓度实部 [–]
        REθeEast__ = REθeInterfaces__[:, 1:]   # 各控制体右界面的电解液无量纲锂离子浓度实部 [–]
        gradREθeWest__ = hstack([zeros([Nf, 1]), (REθe__[:, 1:] - REθe__[:, :-1])/ΔxWest_[1:]])   # (Nf, Ne) 各控制体左界面的无量纲锂离子浓度梯度实部 [–/–]
        gradREθeEast__ = hstack([(REθe__[:, 1:] - REθe__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # (Nf, Ne) 各控制体右界面的无量纲锂离子浓度梯度实部 [–/–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            gradREθeEast__[:, nW] = (REθeEast__[:, nW] - REθe__[:, nW])/(0.5*Δx_[nW])
            gradREθeWest__[:, nE] = (REθe__[:, nE] - REθeWest__[:, nE])/(0.5*Δx_[nE])
        # 各控制体界面的电解液无量纲锂离子浓度虚部 [–]
        IMθeInterfaces__ = hstack([IMθe__[:, [0]], (IMθe__[:, :-1] + IMθe__[:, 1:])*0.5, IMθe__[:, [-1]]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面锂离子浓度
            IMθeInterfaces__[:, nE] = (Deκ_[nW]*IMθe__[:, nW]*Δx_[nE] + Deκ_[nE]*IMθe__[:, nE]*Δx_[nW])/(Deκ_[nW]*Δx_[nE] + Deκ_[nE]*Δx_[nW])
        IMθeWest__ = IMθeInterfaces__[:, :-1]  # 各控制体左界面的电解液无量纲锂离子浓度虚部 [–]
        IMθeEast__ = IMθeInterfaces__[:, 1:]   # 各控制体右界面的电解液无量纲锂离子浓度虚部 [–]
        gradIMθeWest__ = hstack([zeros([Nf, 1]), (IMθe__[:, 1:] - IMθe__[:, :-1])/ΔxWest_[1:]])   # (Nf, Ne) 各控制体左界面的无量纲锂离子浓度梯度虚部 [–/–]
        gradIMθeEast__ = hstack([(IMθe__[:, 1:] - IMθe__[:, :-1])/ΔxEast_[:-1], zeros([Nf, 1])])  # (Nf, Ne) 各控制体右界面的无量纲锂离子浓度梯度虚部 [–/–]
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
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
            # 修正负极-隔膜、隔膜-正极界面
            gradREφeEast__[:, nW] = (REφeEast__[:, nW] - REφe__[:, nW])/(0.5*Δx_[nW])
            gradREφeWest__[:, nE] = (REφe__[:, nE] - REφeWest__[:, nE])/(0.5*Δx_[nE])
        # 各控制体界面的电解液电势虚部梯度 [V/–]
        gradIMφeInterfaces__ = hstack([zeros([Nf, 1]), (IMφe__[:, 1:] - IMφe__[:, :-1])/(x_[1:] - x_[:-1]), zeros([Nf, 1])])
        gradIMφeWest__ = gradIMφeInterfaces__[:, :-1].copy()
        gradIMφeEast__ = gradIMφeInterfaces__[:, 1:].copy()
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜、隔膜-正极界面
            gradIMφeEast__[:, nW] = (IMφeEast__[:, nW] - IMφe__[:, nW])/(0.5*Δx_[nW])
            gradIMφeWest__[:, nE] = (IMφe__[:, nE] - IMφeWest__[:, nE])/(0.5*Δx_[nE])

        Kθsnegsurf___, Kθspossurf___, ωqeΔx__ = solve_frequency_dependent_variables(
            self.ω_, self.qe_, self.Δx_,
            self.Qneg, self.Qpos,
            self.Dsneg, self.Dspos)
        maxError = max([
            abs(array([REθsnegsurf__, IMθsnegsurf__]).transpose(1, 0, 2) - Kθsnegsurf___ @ array([REJintneg__, IMJintneg__]).transpose(1, 0, 2)).max(),
            abs(array([REθspossurf__, IMθspossurf__]).transpose(1, 0, 2) - Kθspossurf___ @ array([REJintpos__, IMJintpos__]).transpose(1, 0, 2)).max(), ])
        print(f'固相表面无量纲浓度 REθssurf IMθssurf 解析解方程最大误差{maxError: 3e} [–]')

        LHS__ = -outer(ω_, qe_) * IMθe__
        RHS__ = Deκ_*(gradREθeEast__ - gradREθeWest__)/Δx_ + hstack([REJintneg__ + REJDLneg__, zeros([Nf, Nsep]), REJintpos__ + REJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液锂电荷守恒方程实部 REθe 电流残差最大误差{maxError: 3e} [A]')

        LHS__ = outer(ω_, qe_) * REθe__
        RHS__ = Deκ_*(gradIMθeEast__ - gradIMθeWest__)/Δx_ + hstack([IMJintneg__ + IMJDLneg__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液锂电荷守恒方程虚部 IMθe 电流残差最大误差{maxError: 3e} [A]')

        gradREφsnegInterfaces_ = hstack([full([Nf, 1], -ΔIAC/σneg), (REφsneg__[:, 1:] - REφsneg__[:, :-1])/Δxneg, zeros([Nf, 1])])
        ΔREφsneg__ = (gradREφsnegInterfaces_[:, 1:] - gradREφsnegInterfaces_[:, :-1])/Δxneg
        gradIMφsnegInterfaces__ = hstack([zeros([Nf, 1]), (IMφsneg__[:, 1:] - IMφsneg__[:, :-1])/Δxneg, zeros([Nf, 1]),])
        ΔIMφsneg__ = (gradIMφsnegInterfaces__[:, 1:] - gradIMφsnegInterfaces__[:, :-1])/Δxneg
        RE_LHS__ = σneg*ΔREφsneg__
        RE_RHS__ = REJintneg__ + REJDLneg__
        IM_LHS__ = σneg*ΔIMφsneg__
        IM_RHS__ = IMJintneg__ + IMJDLneg__
        maxError = max([abs(RE_LHS__ - RE_RHS__).max(),
                        abs(IM_LHS__ - IM_RHS__).max(),])
        print(f'负极固相电势 REφsneg IMφsneg 方程电流残差最大误差{maxError: 3e} [A]')

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
        print(f'正极固相电势 REφspos IMφspos 方程电流残差最大误差{maxError: 3e} [A]')

        term1__ = κ_*(gradREφeEast__ - gradREφeWest__)/Δx_
        term2__ = (κDκT_*(gradREθeEast__/θeEast_ - REθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradREθeWest__/θeWest_ - REθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([REJintneg__ + REJDLneg__ , zeros([Nf, Nsep]), REJintpos__ + REJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势实部 REφe 方程电流残差最大误差{maxError: 3e} [A]')

        term1__ = κ_*(gradIMφeEast__ - gradIMφeWest__)/Δx_
        term2__ = (κDκT_*(gradIMθeEast__/θeEast_ - IMθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradIMθeWest__/θeWest_ - IMθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([IMJintneg__ + IMJDLneg__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势虚部 IMφe 方程电流残差最大误差{maxError: 3e} [A]')

        maxError = max([
            abs(REJintneg__ - 2*(REI0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMJintneg__ - 2*(IMI0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REJintpos__ - 2*(REI0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMJintpos__ - 2*(IMI0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(), ])
        print(f'主反应BV动力学 REJint IMJint 方程最大误差{maxError: 3e} [A]')

        maxError = max([
            abs(REJDLneg__ - -ω_[:, None]*CDLneg*(IMφsneg__ - IMφe__[:, :Nneg] - Rfneg*(IMJintneg__ + IMJDLneg__))).max(),
            abs(IMJDLneg__ -  ω_[:, None]*CDLneg*(REφsneg__ - REφe__[:, :Nneg] - Rfneg*(REJintneg__ + REJDLneg__))).max(),
            abs(REJDLpos__ - -ω_[:, None]*CDLpos*(IMφspos__ - IMφe__[:, -Npos:] - Rfpos*(IMJintpos__ + IMJDLpos__))).max(),
            abs(IMJDLpos__ -  ω_[:, None]*CDLpos*(REφspos__ - REφe__[:, -Npos:] - Rfpos*(REJintpos__ + REJDLpos__))).max(), ])
        print(f'双电层电流 REJDL IMJDL 方程最大误差{maxError: 3e} [A]')

        dI0intdθeneg_, dI0intdθepos_ = self.dI0intdθeneg_, self.dI0intdθepos_
        dI0intdθsnegsurf_, dI0intdθspossurf_ = self.dI0intdθsnegsurf_, self.dI0intdθspossurf_
        maxError = max([
            abs(REI0intneg__ - (dI0intdθeneg_*REθe__[:, :Nneg] + dI0intdθsnegsurf_*REθsnegsurf__)).max(),
            abs(IMI0intneg__ - (dI0intdθeneg_*IMθe__[:, :Nneg] + dI0intdθsnegsurf_*IMθsnegsurf__)).max(),
            abs(REI0intpos__ - (dI0intdθepos_*REθe__[:, -Npos:] + dI0intdθspossurf_*REθspossurf__)).max(),
            abs(IMI0intpos__ - (dI0intdθepos_*IMθe__[:, -Npos:] + dI0intdθspossurf_*IMθspossurf__)).max(), ])
        print(f'集总交换电流密度 REI0int IMI0int 方程最大误差{maxError: 3e} [A]')

        dUOCPdθsnegsurf_, dUOCPdθspossurf_ = self.dUOCPdθsnegsurf_, self.dUOCPdθspossurf_
        maxError = max([
            abs(REηintneg__ - (REφsneg__ - REφe__[:, :Nneg] - dUOCPdθsnegsurf_*REθsnegsurf__ - Rfneg*(REJintneg__ + REJDLneg__))).max(),
            abs(IMηintneg__ - (IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdθsnegsurf_*IMθsnegsurf__ - Rfneg*(IMJintneg__ + IMJDLneg__))).max(),
            abs(REηintpos__ - (REφspos__ - REφe__[:, -Npos:] - dUOCPdθspossurf_*REθspossurf__ - Rfpos*(REJintpos__ + REJDLpos__))).max(),
            abs(IMηintpos__ - (IMφspos__ - IMφe__[:, -Npos:] - dUOCPdθspossurf_*IMθspossurf__ - Rfpos*(IMJintpos__ + IMJDLpos__))).max(), ])
        print(f'主反应过电位 REηint IMηint 方程最大误差{maxError: 3e} [V]')

    plot_θ = P2Dbase.plot_c      # 参数辨识作图：无量纲浓度场-空间、时间
    plot_Jint_I0int_ηint = P2Dbase.plot_jint_i0int_ηint  # 参数辨识作图：主反应集总局部体积电流密度、集总交换电流密度、过电位-空间、时间
    plot_JDL = P2Dbase.plot_jDL  # 参数辨识作图：双电层效应集总局部体积电流密度、电流
    plot_θsr = P2Dbase.plot_csr  # 参数辨识作图：固相颗粒径向无量纲锂离子浓度场-空间、时间

    plot_REθssurf_IMθssurf = P2Dbase.plot_REcssurf_IMcssurf
    plot_REθe_IMθe = P2Dbase.plot_REce_IMce
    plot_REJint_IMJint = P2Dbase.plot_REjint_IMjint
    plot_REJDL_IMJDL = P2Dbase.plot_REjDL_IMjDL
    plot_REI0int_IMI0int = P2Dbase.plot_REi0int_IMi0int

if __name__=='__main__':
    import numpy as np
    cell = LPJTFP2D(
        SOC0=0.1,

        Nneg=10, Nsep=9, Npos=11, Nr=8,
        f_=np.logspace(3, -1, 21),
        # I0intneg=18, I0intpos=22,
        # CDLneg=0, CDLpos=0,
        # complete=False,
        )

    cell.count_lithium()
    thermalModel = True  # False
    cell.CC(-20, 2000, thermalModel=thermalModel, tEIS_=range(0, 2001, 200),)
    cell.CC(20, 1000, thermalModel=thermalModel).EIS()
    cell.CC(0, 300, thermalModel=thermalModel).EIS()
    cell.checkEIS()

    cell.count_lithium()
    '''
    cell.plot_UI()
    cell.plot_TQgen()
    cell.plot_SOC()
    cell.plot_θ(np.arange(0, 2001, 200))
    cell.plot_φ(np.arange(0, 2001, 200))
    cell.plot_Jint_I0int_ηint(np.arange(0, 2001, 200))
    cell.plot_JDL(np.arange(0, 2001, 200))
    cell.plot_csr(np.arange(0, 2001, 200), 1)
    cell.plot_OCV_OCP()
    cell.plot_dUOCPdθs()
    cell.plot_nNewton()

    cell.plot_Z()
    cell.plot_Nyquist()
    cell.plot_REθssurf_IMθssurf()
    cell.plot_REθe_IMθe()
    cell.plot_REφs_IMφs()
    cell.plot_REφe_IMφe()
    cell.plot_REJint_IMJint()
    cell.plot_REJDL_IMJDL()
    cell.plot_REI0int_IMI0int()
    cell.plot_REηint_IMηint()
    '''

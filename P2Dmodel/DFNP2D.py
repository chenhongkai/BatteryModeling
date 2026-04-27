#%%
import time, warnings, pathlib
from math import exp
from typing import Sequence, Callable
from collections.abc import Iterable

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.linalg.lapack import dgbsv, dgtsv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.optimize import root
from numpy import pi, nan, ndarray,\
    array, arange, zeros, empty, full, linspace, stack, hstack, concatenate, tile, \
    zeros_like, meshgrid,\
    exp, sqrt, sinh, cosh, arcsinh, outer, \
    ix_, asfortranarray, \
    isnan, isscalar, \
    savez
from numpy.linalg import solve


from P2Dmodel import set_matplotlib, Interpolate1D, triband_to_dense
from P2Dmodel.OCP import NMC111, Graphite


class DFNP2D:
    """锂离子电池经典准二维模型 Doyle-Fuller-Newman Pseudo-two-Dimensional model"""
    F = 96485.33289  # 法拉第常数 [C/mol]
    R = 8.314472     # 摩尔气体常数 [J/(mol·K)]
    M = 6.941e-3     # 锂的摩尔质量 [kg/mol]
    ρ = 534.         # 锂金属密度 [kg/m^3]

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
            Dspos: float = 10e-14,   # 正极固相的锂离子扩散系数 [m^2/s]
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
            θminneg: float = .0370744,  # SOC=0%的负极嵌锂状态 [–]
            θmaxneg: float = .8775600,  # SOC=100%的负极嵌锂状态 [–]
            θminpos: float = .0746557,  # SOC=100%的正极嵌锂状态 [–]
            θmaxpos: float = .9589741,  # SOC=0%的正极嵌锂状态 [–]
            UOCPneg: Callable = Graphite().Graphite_COMSOL,  # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位UOCPneg_ [V]
            UOCPpos: Callable = NMC111().NMC111_COMSOL,      # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位UOCPpos_ [V]
            dUOCPdTneg: Callable | float = 0.,   # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位的熵热系数 [V/K]
            dUOCPdTpos: Callable | float = 0.,   # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位的熵热系数 [V/K]
            dUOCPdθsneg: Callable | None = None,  # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位对嵌锂状态的导数 [V/–]
            dUOCPdθspos: Callable | None = None,  # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位对嵌锂状态的导数 [V/–]
            i0intneg: float | None = None,    # 负极主反应交换电流密度 [A/m^2]
            i0intpos: float | None = None,    # 正极主反应交换电流密度 [A/m^2]
            i0LP: float | None = None,        # 负极析锂反应交换电流密度 [A/m^2]
            EDsneg: float = 30e3,  # 负极锂离子扩散系数的比活化能 [J/mol]
            EDspos: float = 30e3,  # 正极锂离子扩散系数的比活化能 [J/mol]
            Ekneg: float = 68e3,   # 负极主反应速率常数的比活化能 [J/mol]
            Ekpos: float = 50e3,   # 正极主反应速率常数的比活化能 [J/mol]
            Eκ: float = 4e3,       # 电解液锂离子电导率的比活化能 [J/mol]
            EDe: float = 16.5e3,   # 电解液锂离子扩散系数的比活化能 [J/mol]
            EkLP: float = 3.53e4,  # 负极析锂反应速率常数的比活化能 [J/mol]
            Tref: float = 298.15,  # 参考温度 [K]
            Aeffneg: float = 1.,   # 负极活性材料与电解质的有效接触面积比 [–]
            Aeffpos: float = 1.,   # 正极活性材料与电解质的有效接触面积比 [–]
            Δt: float | int = 10., # 时间步长 [s]
            Nneg: int = 10,        # 负极区域网格数
            Nsep: int = 10,        # 隔膜区域网格数
            Npos: int = 10,        # 正极区域网格数
            Nr: int = 10,          # 球形固相颗粒半径方向网格数
            hA: float = 1.5,       # 表面传热系数与传热面积之积 [W/K]
            Cth: float = 600.,     # 热容 [J/K]
            Tamb: float = 298.15,  # 环境温度 [K]
            SOC0: float = 0.5,     # 初始荷电状态 [–]
            T0: float = 298.15,    # 初始温度 [K]
            lithiumPlating: bool = False,     # 是否考虑析锂反应
            doubleLayerEffect: bool = True,   # 是否考虑电极颗粒双电层效应
            timeDiscretization: str = 'CN',   # 时间离散格式 'CN'/'backward'
            radialDiscretization: str = 'EV', # 球形颗粒径向离散方法 等体积'EV'/等间隔'EI'
            decouple_cs: bool = True,      # 是否解耦固相锂离子浓度的求解，decouple_cs==True可加速，几乎无代价
            constants: bool = False,       # 是否使用常量参数缓存，设置constants==True可加速，但应在考虑参数为恒定值的情况下
            complete: bool = True,         # 是否确保功能完备，设置complete==False可加速，但省略不必要的计算
            verbose: bool = True,          # 是否显示初始化、运行进度
            fullyInitialize: bool = True,  # 是否完全初始化
            ):
        fullyInitialize = fullyInitialize or (type(self) is DFNP2D)  # 类DFNP2D的初始化必须完整
        if fullyInitialize:
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
            if verbose and (εsneg + εeneg)>1:
                warnings.warn(f'负极固相体积分数εsneg与负极电解液体积分数εeneg之和大于1，{εsneg + εeneg = } > 1')
            if verbose and (εspos + εepos)>1:
                warnings.warn(f'正极固相体积分数εspos与正极电解液体积分数εepos之和大于1，{εspos + εepos = } > 1')
            self.Rsneg = Rsneg; assert Rsneg>0, f'负极球形固相颗粒半径{Rsneg = }，应大于0 [m]'
            self.Rspos = Rspos; assert Rspos>0, f'正极球形固相颗粒半径{Rspos = }，应大于0 [m]'
            # 11输运参数
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
            # 5动力学参数
            self.kneg = kneg; assert kneg>0, f'负极主反应速率常数{kneg = }，应大于0 m^2.5/(mol^0.5·s)'
            self.kpos = kpos; assert kpos>0, f'正极主反应速率常数{kpos = }，应大于0 m^2.5/(mol^0.5·s)'
            self.kLP = kLP; assert kLP>0,  f'负极析锂反应速率常数{kLP = }，应大于0'
            self.RSEIneg = RSEIneg; assert RSEIneg>=0, f'负极SEI膜的面积电阻{RSEIneg = }，应大于或等于0 [Ω·m^2]'
            self.RSEIpos = RSEIpos; assert RSEIpos>=0, f'正极SEI膜的面积电阻{RSEIpos = }，应大于或等于0 [Ω·m^2]'
            # 3浓度参数
            self.csmaxneg = csmaxneg; assert csmaxneg>0, f'负极固相最大锂离子浓度{csmaxneg = }，应大于0 [mol/m^3]'
            self.csmaxpos = csmaxpos; assert csmaxpos>0, f'正极固相最大锂离子浓度{csmaxpos = }，应大于0 [mol/m^3]'
            self.ce0 = ce0; assert ce0>0, f'电解液的初始浓度{ce0 = }，应大于0 [mol/m^3]'
            # 3电抗参数
            self.CDLneg = CDLneg; assert CDLneg>=0, f'负极表面双电层面积电容{CDLneg = }，应大于或等于0 [F/m^2]'
            self.CDLpos = CDLpos; assert CDLpos>=0, f'正极表面双电层面积电容{CDLpos = }，应大于或等于0 [F/m^2]'
            self.l = l; assert l>=0, f'等效电感{l = }，应大于或等于0 [H]'
            # 4边界嵌锂状态参数
            assert 0<θminneg<θmaxneg<1, f'负极最小、最大嵌锂状态{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1'
            assert 0<θminpos<θmaxpos<1, f'正极最小、最大嵌锂状态{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1'
            self.θminneg = θminneg  # SOC=0%的负极嵌锂状态
            self.θmaxneg = θmaxneg  # SOC=100%的负极嵌锂状态
            self.θmaxpos = θmaxpos  # SOC=0%的正极嵌锂状态
            self.θminpos = θminpos  # SOC=100%的正极嵌锂状态
            # 交换电流密度
            self.i0intneg = i0intneg; assert (i0intneg is None) or (i0intneg>0), f'负极主反应交换电流密度{i0intneg = }，应大于0 [A/m^2]'
            self.i0intpos = i0intpos; assert (i0intpos is None) or (i0intpos>0), f'正极主反应交换电流密度{i0intpos = }，应大于0 [A/m^2]'
            self.i0LP = i0LP;         assert (i0LP is None) or (i0LP>0), f'负极析锂反应交换电流密度{i0LP = }，应大于0 [A/m^2]'
            # 全固态电池参数
            self.Aeffneg = Aeffneg; assert 0<Aeffneg<=1, f'负极固相与电解质的有效接触面积比{Aeffneg = }，取值范围应为(0, 1]'
            self.Aeffpos = Aeffpos; assert 0<Aeffpos<=1, f'正极固相与电解质的有效接触面积比{Aeffpos = }，取值范围应为(0, 1]'
        # 函数
        self.UOCPneg = UOCPneg; assert callable(UOCPneg), '函数UOCPneg，输入负极嵌锂状态θsneg_ [–]，输出正极开路电位UOCPneg_ [V]'
        self.UOCPpos = UOCPpos; assert callable(UOCPpos), '函数UOCPpos，输入正极嵌锂状态θspos_ [–]，输出负极开路电位UOCPpos_ [V]'
        self.dUOCPdTneg = dUOCPdTneg; assert callable(dUOCPdTneg) or isinstance(dUOCPdTneg, (int, float)), '负极开路电位的熵热系数dUOCPdTneg [V/K]，标量或函数（输入负极嵌锂状态θsneg_ [–]）'
        self.dUOCPdTpos = dUOCPdTpos; assert callable(dUOCPdTpos) or isinstance(dUOCPdTpos, (int, float)), '正极开路电位的熵热系数dUOCPdTpos [V/K]，标量或函数（输入正极嵌锂状态θspos_ [–]）'
        assert callable(dUOCPdθsneg) or (dUOCPdθsneg is None), '负极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入负极嵌锂状态θsneg_ [–]）'
        self.solve_dUOCPdθsneg_ = DFNP2D.generate_solve_dUOCPdθs_(UOCPneg) if (dUOCPdθsneg is None) else dUOCPdθsneg
        assert callable(dUOCPdθspos) or (dUOCPdθspos is None), '正极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入正极嵌锂状态θspos_ [–]）'
        self.solve_dUOCPdθspos_ = DFNP2D.generate_solve_dUOCPdθs_(UOCPpos) if (dUOCPdθspos is None) else dUOCPdθspos
        # 比活化能
        self.EDsneg = EDsneg; assert EDsneg>=0, f'负极锂离子扩散系数的比活化能{EDsneg = }，应大于或等于0 [J/mol]'
        self.EDspos = EDspos; assert EDspos>=0, f'正极锂离子扩散系数的比活化能{EDspos = }，应大于或等于0 [J/mol]'
        self.Ekneg = Ekneg; assert Ekneg>=0,    f'负极主反应速率常数的比活化能{Ekneg = }，应大于或等于0 [J/mol]'
        self.Ekpos = Ekpos; assert Ekpos>=0,    f'正极主反应速率常数的比活化能{Ekpos = }，应大于或等于0 [J/mol]'
        self.Eκ = Eκ;     assert Eκ>=0,   f'电解液锂离子电导率的比活化能{Eκ = }，应大于或等于0 [J/mol]'
        self.EDe = EDe;   assert EDe>=0,  f'电解液锂离子扩散系数的比活化能{EDe = }，应大于或等于0 [J/mol]'
        self.EkLP = EkLP; assert EkLP>=0, f'负极析锂反应速率常数的比活化能{EkLP = }，应大于或等于0 [J/mol]'
        self.Tref = Tref; assert Tref>0, f'参考温度{Tref = }，应大于0 [K]'
        # 网格参数
        self.Δt = Δt; assert Δt>0, f'时间步长{Δt = }，应大于0 [s]'
        self.Nneg = Nneg; assert isinstance(Nneg, int) and Nneg>=3, f'负极区域网格数{Nneg = }，应为不小于3的正整数'
        self.Nsep = Nsep; assert isinstance(Nsep, int) and Nsep>=3, f'隔膜区域网格数{Nsep = }，应为不小于3的正整数'
        self.Npos = Npos; assert isinstance(Npos, int) and Npos>=3, f'正极区域网格数{Npos = }，应为不小于3的正整数'
        self.Nr = Nr; assert isinstance(Nr, int) and Nr>=3, f'球形固相颗粒半径方向网格数{Nr = }，应为不小于3的正整数'
        self.Ne = Ne = Nneg + Nsep + Npos   # 电解液网格总数
        # 热参数
        self.hA = hA; assert hA>0, f'表面传热系数与传热面积之积{hA = }，应大于0 [W/K]'
        self.Cth = Cth; assert Cth>0, f'电池热容{Cth = }，应大于0 [J/K]'
        self.Tamb = Tamb; assert Tamb>0, f'环境温度{Tamb = }，应大于0 [K]'
        # 作图变量单位
        self.tSign = r'${\it t}$'  # 时间t符号
        self.tUnit = 's'           # 时间t单位
        if fullyInitialize:
            self.xUnit = 'μm'  # 横坐标x单位
            self.rUnit = 'μm'  # 径向坐标r单位
            self.cUnit = 'mol/m$^3$'   # 浓度单位
            self.jUnit = 'A/m$^3$'     # 局部体积电流密度单位
            self.i0Unit = 'A/m$^2$'    # 交换电流密度单位
            self.xSign = r'${\it x}$'  # 横坐标x符号
            self.rSign = r'${\it r}$'  # 径向坐标r符号
            self.cSign = r'${\it c}$'  # 浓度符号
            self.jSign = r'${\it j}$'  # 局部体积电流密度符号
            self.i0Sign = r'${\it i}_0$'  # 交换电流密度符号
        # 模式
        self.lithiumPlating = lithiumPlating        # 是否考虑析锂反应
        self.doubleLayerEffect = doubleLayerEffect  # 是否考虑双电层效应
        self.timeDiscretization = timeDiscretization;     assert timeDiscretization in {'backward', 'CN'}, f'时间离散格式{timeDiscretization = }，应为 "CN"（Crank-Nicolson） 或 "backward"（后向差分格式）'
        self.radialDiscretization = radialDiscretization; assert radialDiscretization in {'EV', 'EI'}, f'球形颗粒径向离散方法{radialDiscretization = }，应为 "EV"（等体积）/"EI"（等间隔）'
        self.decouple_cs = decouple_cs  # 是否解耦固相锂离子浓度的求解
        self.constants = constants      # 是否使用常量参数缓存
        self.complete = complete        # 是否确保功能完备
        self.verbose = verbose          # 是否显示初始化、运行进度
        # 恒定量
        (self.Δxneg, self.Δxsep, self.Δxpos,  # 负极、隔膜、正极网格厚度 [m]
        self.x_,                              # (Ne,) 全区域控制体中心的坐标 [m]
        self.Δx_,                             # (Ne,) 全区域控制体厚度 [m]
        self.xneg_, self.xsep_, self.xpos_,   # (Nneg,), (Nsep,), (Npos,) 负极、隔膜、正极区控制体中心的坐标 [m]
        self.ΔxWest_, self.ΔxEast_,           # (Ne,) 当前控制体中心到左侧、右侧控制体中心的距离 [m]
        self.xInterfaces_,                    # (Ne+1,) 各控制体交界面的坐标（包括负极-集流体界面、正极集流体界面） [m]
        self.Δrneg_, self.Δrpos_,             # (Nr,) 负极、正极颗粒球壳网格厚度 [m]
        self.rneg_, self.rpos_,               # (Nr,) 负极、正极球形固相颗粒控制体中心径向坐标 [m]
        self.Vr_,                             # (Nr,) 从中心到表面负极固相颗粒球壳体积分数序列 [–]
        self.coeffs_,                         # (3,) 用于外推颗粒表面锂离子浓度的系数
        self.bandKcsneg__, self.bandKcspos__, # (3, Nr) 负极、正极固相浓度矩阵的带
        self.datanames_,                      # 列表：需记录的数据名称
        ) = (None,)*20
        self.e__ = zeros((Nr, 1))  # (Nr, 1) 非零末元为1的向量
        self.e__[-1] = 1.
        # 状态量
        if fullyInitialize:
            self.csneg__, self.cspos__ = empty((Nr, Nneg)), empty((Nr, Npos))  # 负极、正极固相颗粒内部锂离子浓度场 [mol/m^3]
            self.csnegsurf_, self.cspossurf_ = empty(Nneg), empty(Npos)  # 负极、正极固相颗粒表面锂离子浓度场 [mol/m^3]
            self.ce_ = empty(Ne)                                         # 电解液锂离子浓度场 [mol/m^3]
            self.jintneg_, self.jintpos_ = empty(Nneg), empty(Npos)      # 负极、正极主反应局部体积电流密度场 [A/m^3]
            self.jDLneg_, self.jDLpos_ = empty(Nneg), empty(Npos)        # 负极、正极双电层效应局部体积电流密度场 [A/m^3]
            self.i0intneg_, self.i0intpos_ = empty(Nneg), empty(Npos)    # 负极、正极主反应交换电流密度场 [A/m^2]
            self.jneg_, self.jpos_ = empty(Nneg), empty(Npos)            # 负极、正极总局部体积电流密度场 [A/m^3]
            if lithiumPlating:
                self.jLP_ = empty(Nneg)   # 负极析锂局部体积电流密度场 [A/m^3]
                self.i0LP_ = empty(Nneg)  # 负极析锂反应交换电流密度场 [A/m^2]
        self.φsneg_, self.φspos_ = empty(Nneg), empty(Npos)     # 负极、正极固相电势场 [V]
        self.φe_ = empty(Ne)                                    # 电解液电势场 [V]
        self.ηintneg_, self.ηintpos_ =empty(Nneg), empty(Npos)  # 负极、正极主反应过电位场 [V]
        self.ηLPneg_, self.ηLPpos_ = empty(Nneg), empty(Npos)   # 负极、正极析锂反应过电位场 [V]
        (self.I, self.t, self.T,    # 电流 [A]、时刻 [s]、电池温度 [K]
        self.QLP,                   # 累计析锂量 [Ah]
        self.data,                  # 字典：存储呈时间序列的运行数据
        self.K__,                   # 因变量线性矩阵、常数项向量向量 K__ @ X_ = bK_
        self.bandwidthsJ_,          # Jacobi矩阵带宽
         ) = (None,)*7
        # 因变量索引
        (self.idxcsneg_, self.idxcspos_, self.idxcsnegsurf_, self.idxcspossurf_, self.idxce_,
        self.idxφsneg_, self.idxφspos_, self.idxφe_,
        self.idxjintneg_, self.idxjintpos_, self.idxjDLneg_, self.idxjDLpos_, self.idxjLP_,
        self.idxi0intneg_, self.idxi0intpos_,
        self.idxηintneg_, self.idxηintpos_, self.idxηLP_,
        self.idxc_, self.idxφ_, self.idxj_,
        self.idxJreordered_, self.idxJrecovered_,
        ) = (None,)*23
        if complete:
            set_matplotlib()  # matplotlib作图设置
        if type(self) is DFNP2D:
            self.initialize(
                SOC0=SOC0,  # 初始荷电状态 [–]
                T0=T0,)     # 初始温度 [K]

    def initialize(self,
            SOC0: int | float = 0.5,     # 全电池荷电状态
            T0: int | float = 298.15,):  # 温度 [K]
        """初始化"""
        if self.verbose:
            print(f'DFN-P2D模型初始化...')
        assert 0<=SOC0<=1, f'初始荷电状态{SOC0 = }，取值范围应为[0, 1]'
        self.T = T0; assert T0>0, f'初始温度{T0 = }，应大于0 [K]'
        self.I = 0.  # 初始化：电流 [A]
        self.t = 0.  # 初始化：时刻 [s]
        Nneg, Nsep, Npos, Ne, Nr = self.Nneg, self.Nsep, self.Npos, self.Ne, self.Nr  # 读取：网格数
        Lneg, Lsep, Lpos = self.Lneg, self.Lsep, self.Lpos  # 读取：负极、隔膜、正极厚度 [m]
        # 恒定量
        self.Δxneg = Δxneg = Lneg/Nneg  # 负极网格厚度 [m]
        self.Δxsep = Δxsep = Lsep/Nsep  # 隔膜网格厚度 [m]
        self.Δxpos = Δxpos = Lpos/Npos  # 正极网格厚度 [m]
        self.x_ = hstack([
            linspace(0, Lneg, Nneg + 1)[:-1] + Δxneg/2,
            linspace(Lneg, Lneg + Lsep, Nsep + 1)[:-1] + Δxsep/2,
            linspace(Lneg + Lsep, Lneg + Lsep + Lpos, Npos + 1)[:-1] + Δxpos/2,
            ])  # (Ne,) 各控制体中心坐标 [m]
        self.generate_x_related_coordinates()
        if not self.doubleLayerEffect:
            self.CDLneg = self.CDLpos = 0  # 若不考虑双电层效应，正负极双电层电容赋0
        match self.radialDiscretization:
            case 'EI':  # 等间隔划分球壳网格
                Δrneg = self.Rsneg/Nr  # 负极颗粒球壳网格厚度 [m]
                Δrpos = self.Rspos/Nr  # 正极颗粒球壳网格厚度 [m]
                self.rneg_ = linspace(0, self.Rsneg, Nr + 1)[:-1] + Δrneg/2  # 负极球形颗粒径向控制体中心的坐标 [m]
                self.rpos_ = linspace(0, self.Rspos, Nr + 1)[:-1] + Δrpos/2  # 正极球形颗粒径向控制体中心的坐标 [m]
                self.Δrneg_ = full(Nr, Δrneg)  # 负极颗粒球壳网格厚度序列 [m]
                self.Δrpos_ = full(Nr, Δrpos)  # 正极颗粒球壳网格厚度序列 [m]
            case 'EV':  # 等体积划分球壳网格
                Vneg, Vpos = 4/3*pi*self.Rsneg**3, 4/3*pi*self.Rspos**3  # 负极、正极颗粒体积 [m^3]
                ΔVneg, ΔVpos = Vneg/Nr, Vpos/Nr                          # 负极、正极球壳控制体体积 [m^3]
                rW_ = (ΔVneg*arange(0, Nr)/(4/3*pi))**(1/3)              # Nr维向量：负极球壳内界面坐标 [m^3]
                rE_ = (ΔVneg*arange(1, Nr + 1)/(4/3*pi))**(1/3)          # Nr维向量：负极球壳外界面坐标 [m^3]
                self.rneg_ = (rW_ + rE_)/2
                self.Δrneg_ = rE_ - rW_
                rW_ = (ΔVpos*arange(0, Nr)/(4/3*pi))**(1/3)      # Nr维向量：正极球壳内界面坐标 [m^3]
                rE_ = (ΔVpos*arange(1, Nr + 1)/(4/3*pi))**(1/3)  # Nr维向量：正极球壳外界面坐标 [m^3]
                self.rpos_ = (rW_ + rE_)/2
                self.Δrpos_ = rE_ - rW_
        self.Vr_ = ((self.rneg_ + self.Δrneg_/2)**3
                  - (self.rneg_ - self.Δrneg_/2)**3)/self.Rsneg**3  # 从中心到表面负极固相颗粒球壳体积分数序列 [–]
        self.initialize_linear_matrix()
        # 状态量
        θsneg = self.θminneg + SOC0*(self.θmaxneg - self.θminneg)  # 初始化：负极嵌锂状态
        θspos = self.θmaxpos + SOC0*(self.θminpos - self.θmaxpos)  # 初始化：正极嵌锂状态
        csneg = θsneg*(csmaxneg := self.csmaxneg)
        cspos = θspos*(csmaxpos := self.csmaxpos)
        self.csneg__[:] = csneg  # 初始化：负极固相颗粒锂离子浓度场 [mol/m^3]
        self.cspos__[:] = cspos  # 初始化：正极固相颗粒锂离子浓度场 [mol/m^3]
        self.csnegsurf_[:] = csneg    # 初始化：负极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.cspossurf_[:] = cspos    # 初始化：正极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.ce_[:] = ce0 = self.ce0  # 初始化：电解液锂离子浓度场 [mol/m^3]
        self.φsneg_[:] = UOCPneg = self.solve_UOCPneg_(θsneg)  # 初始化：负极固相电势场 [V]
        self.φspos_[:] = UOCPpos = self.solve_UOCPpos_(θspos)  # 初始化：正极固相电势场 [V]
        self.φe_[:] = 0.                                       # 初始化：电解液电势场 [V]
        self.jintneg_[:], self.jintpos_[:] = 0., 0.  # 初始化：负极、正极主反应局部体积电流密度场 [A/m^3]
        self.jDLneg_[:], self.jDLpos_[:] = 0., 0.    # 初始化：负极、正极双电层效应局部体积电流密度场 [A/m^3]
        self.i0intneg_[:] = self.i0intneg if self._i0intneg else DFNP2D.solve_i0int_(self.kneg, csmaxneg, csneg, ce0)
        self.i0intpos_[:] = self.i0intpos if self._i0intpos else DFNP2D.solve_i0int_(self.kpos, csmaxpos, cspos, ce0)
        self.ηintneg_[:], self.ηintpos_[:] = 0., 0.           # 初始化：负极、正极主反应过电位场 [V]
        self.jneg_[:], self.jpos_[:] = 0., 0.         # 初始化：负极、正极总局部体积电流密度 [A/m^3]
        self.ηLPneg_[:], self.ηLPpos_[:]  = UOCPneg, UOCPpos  # 初始化：负极、正极析锂反应过电位场 [V]
        if lithiumPlating:=self.lithiumPlating:
            self.jLP_[:]  = 0.     # 初始化：负极析锂局部体积电流密度场 [A/m^3]
            self.i0LP_[:] = self.i0LP if self._i0LP else DFNP2D.solve_i0LP_(self.kLP, ce0)  # 初始化：负极析锂反应交换电流密度 [A/^2]
        self.QLP = 0.  # 初始化：累计析锂量 [Ah]
        # 需记录的数据名称
        self.datanames_ = ['U', 'I', 't',          # 端电压 [V]、电流 [A]、时刻 [s]
                           'ηLPneg_', 'ηLPpos_',]  # 负极、正极表面析锂反应过电位场 [V]
        if self.complete:
            self.datanames_.extend([
                'csneg__', 'cspos__',        # 负极、正极固相锂离子浓度场 [mol/m^3]
                'csnegsurf_', 'cspossurf_',  # 负极、正极表面锂离子浓度场 [mol/m^3]
                'ce_',                       # 电解液锂离子浓度场 [mol/m^3]
                'φsneg_', 'φspos_',          # 负极、正极固相电势场 [V]
                'φe_',                       # 电解液电势场 [V]
                'jintneg_', 'jintpos_',      # 负极、正极主反应局部体积电流密度场 [A/m^3]
                'jDLpos_', 'jDLneg_',        # 负极、正极双电层效应局部体积电流密度场 [A/m^3]
                'i0intneg_', 'i0intpos_',    # 负极、正极主反应交换电流密度场 [A/m^2]
                'ηintneg_', 'ηintpos_',      # 负极、正极主反应过电位场 [V]
                'isneg_', 'ispos_', 'ie_',   # 负极、正极固相电流密度场、电解液电流密度场 [A/m^2]
                'θsneg', 'θspos', 'SOC',     # 负极、正极嵌锂状态、全电池荷电状态 [–]
                'T', 'Qgen',])               # 温度 [K]、产热量 [W]
            if lithiumPlating:
                self.datanames_.extend(['jLP_', 'i0LP_'])  # 负极析锂局部体积电流密度场 [A/m^3]、交换电流密度场 [A/m^2]
        self.data = {dataname: [] for dataname in self.datanames_}  # 字典：存储呈时间序列的运行数据
        if self.verbose and type(self) is DFNP2D:
            print(self)
            print('DFN-P2D模型初始化完成!')
        return self

    def generate_x_related_coordinates(self):
        """生成厚度方向x相关坐标"""
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos        # 读取：网格数
        Δxneg, Δxsep, Δxpos = self.Δxneg, self.Δxsep, self.Δxpos  # 读取：网格厚度
        self.xneg_ = self.x_[:Nneg]               # (Nneg,) 负极区控制体中心坐标
        self.xsep_ = self.x_[Nneg:(Nneg + Nsep)]  # (Nsep,) 隔膜区控制体中心坐标
        self.xpos_ = self.x_[-Npos:]              # (Npos,) 正极区控制体中心坐标
        self.Δx_ = concatenate([full(Nneg, Δxneg),
                                full(Nsep, Δxsep),
                                full(Npos, Δxpos)])  # (Ne,) 全区域控制体厚度
        self.ΔxWest_ = hstack([
            full(Nneg, Δxneg),
            (Δxneg + Δxsep)/2, full(Nsep - 1, Δxsep),
            (Δxsep + Δxpos)/2, full(Npos - 1, Δxpos)])  # (Ne,) 当前控制体中心到左侧控制体中心的距离
        self.ΔxEast_ = hstack([
            full(Nneg - 1, Δxneg), (Δxneg + Δxsep)/2,
            full(Nsep - 1, Δxsep), (Δxsep + Δxpos)/2,
            full(Npos, Δxpos)])                                # (Ne,) 当前控制体中心到右侧控制体中心的距离
        self.xInterfaces_ = hstack([0, self.Δx_.cumsum()])  # (Ne+1,) 各控制体界面的坐标（包括负极-集流体界面、正极集流体界面）

    def generate_indices_of_dependent_variables(self):
        """生成因变量索引"""
        Nneg, Npos, Ne, Nr = self.Nneg, self.Npos, self.Ne, self.Nr  # 读取：网格数
        decouple_cs, complete, lithiumPlating = self.decouple_cs, self.complete, self.lithiumPlating
        N = 0  # 全局索引游标
        def allocate(n):
            # 分配索引
            nonlocal N
            idx_ = arange(N, N + n)
            N += n
            return idx_
        self.idxcsneg_ = allocate(0 if decouple_cs else Nr*Nneg)  # 索引：负极固相内部浓度 先排颗粒径向r，再排x方向
        self.idxcspos_ = allocate(0 if decouple_cs else Nr*Npos)  # 索引：正极固相内部浓度
        self.idxcsnegsurf_ = allocate(Nneg)  # 索引：正极固相表面浓度
        self.idxcspossurf_ = allocate(Npos)  # 索引：正极固相表面浓度
        self.idxce_ = allocate(Ne)         # 索引：电解液浓度
        self.idxφsneg_ = allocate(Nneg)    # 索引：负极固相电势
        self.idxφspos_ = allocate(Npos)    # 索引：正极固相电势
        self.idxφe_ = allocate(Ne)         # 索引：电解液电势
        self.idxjintneg_ = allocate(Nneg)  # 索引：负极主反应局部体积电流密度
        self.idxjintpos_ = allocate(Npos)  # 索引：正极主反应局部体积电流密度
        self.idxjDLneg_ = allocate(Nneg if self.CDLneg else 0)  # 索引：负极双电层局部体积电流密度
        self.idxjDLpos_ = allocate(Npos if self.CDLpos else 0)  # 索引：正极双电层局部体积电流密度
        self.idxi0intneg_ = allocate(Nneg if self._i0intneg is None else 0)  # 索引：负极主反应交换电流密度
        self.idxi0intpos_ = allocate(Npos if self._i0intpos is None else 0)  # 索引：正极主反应交换电流密度
        self.idxηintneg_ = allocate(Nneg)   # 索引：负极过电位
        self.idxηintpos_ = allocate(Npos)   # 索引：正极过电位
        self.idxjLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：负极析锂反应局部体积电流密度
        self.idxηLP_ = allocate(Nneg if lithiumPlating else 0)  # 索引：负极析锂反应过电位
        self.idxc_ = concatenate([
            self.idxce_, self.idxcsneg_, self.idxcspos_,
            self.idxcsnegsurf_, self.idxcspossurf_])            # 索引：所有浓度量
        self.idxφ_ = concatenate([self.idxφe_, self.idxφsneg_, self.idxφspos_,
            self.idxηintneg_, self.idxηintpos_, self.idxηLP_])  # 索引：所有电势量
        self.idxj_ = concatenate([self.idxjintneg_, self.idxjDLneg_, self.idxjLP_,
                                     self.idxjintpos_, self.idxjDLpos_])  # 索引：电流量
        return N  # 因变量总数

    def initialize_linear_matrix(self):
        """初始化因变量线性矩阵"""
        N = self.generate_indices_of_dependent_variables()  # 生成因变量索引
        self.K__ = K__ = zeros([N, N])  # 因变量线性矩阵
        if self.verbose:
            print(f'初始化因变量线性矩阵 K__.shape = {K__.shape}')

        ## 对K__矩阵赋参数相关值 ##
        if decouple_cs := self.decouple_cs:
            pass
        else:
            self.update_K__idxcsnegsurf_idxjintneg_(self.aneg, self.Dsneg)
            self.update_K__idxcspossurf_idxjintpos_(self.apos, self.Dspos)
        self.update_K__idxφsneg_idxjneg_(self.σeffneg)
        self.update_K__idxφspos_idxjpos_(self.σeffpos)
        self.update_K__idxφe_idxφe_(κeff_ := self.κeff_, κeff_)
        self.update_K__idxηintneg_idxjneg_(self.RSEIneg, self.aeffneg)
        self.update_K__idxηintpos_idxjpos_(self.RSEIpos, self.aeffpos)
        if self.lithiumPlating:
            self.update_K__idxηLP_idxjneg_(self.RSEIneg, self.aeffneg)

        ## 对K__矩阵赋固定值（x方向网格相关值，常数值） ##
        self.assign_K__with_constants()
        # 对K__额外赋固定值（此处为参数Rsneg、Rspos相关，据其物理意义，这两参数在电池运行过程中不应变化）
        Nr = self.Nr  # 读取：网格数
        # 负极、正极固相表面浓度cssurf行
        for i, (idxcs_, idxcssurf_, r_, Rs, Nreg) in enumerate(zip(
                [self.idxcsneg_, self.idxcspos_],
                [self.idxcsnegsurf_, self.idxcspossurf_],
                [self.rneg_, self.rpos_],
                [self.Rsneg, self.Rspos],
                [self.Nneg, self.Npos],)):
            r_3, r_2, r_1 = r_[-3:]
            a3, a2, a1  = Rs - r_[-3:]
            if i==0:
                self.coeffs_ = array([
                    a1*a2/((r_3 - r_1)*(r_3 - r_2)),
                    a1*a3/((r_2 - r_1)*(r_2 - r_3)),
                    a2*a3/((r_1 - r_2)*(r_1 - r_3))])  # 用于由3个颗粒内部节点浓度外推表面浓度的系数
            if decouple_cs:
                K__[idxcssurf_, idxcssurf_] = 1
            else:
                K__[idxcssurf_, idxcs_[Nr-3:Nreg*Nr:Nr]] = a1*a2/(-a3*(r_3 - r_1)*(r_3 - r_2))
                K__[idxcssurf_, idxcs_[Nr-2:Nreg*Nr:Nr]] = a1*a3/(-a2*(r_2 - r_1)*(r_2 - r_3))
                K__[idxcssurf_, idxcs_[Nr-1:Nreg*Nr:Nr]] = a2*a3/(-a1*(r_1 - r_2)*(r_1 - r_3))
                K__[idxcssurf_, idxcssurf_] = 1/a1 + 1/a2 + 1/a3

        self.bandKcsneg__ = zeros((3, Nr))  # 负极固相浓度矩阵的带 [m^-2]，仅与Rsneg、Nr相关
        self.bandKcspos__ = zeros((3, Nr))  # 正极固相浓度矩阵的带 [m^-2]，仅与Rspos、Nr相关
        idx_ = arange(Nr)
        idxm_ = idx_[1:-1]
        for band__, r_, Δr_ in zip([self.bandKcsneg__, self.bandKcspos__], [self.rneg_, self.rpos_], [self.Δrneg_, self.Δrpos_]):
            Kcs__ = zeros((Nr, Nr))
            a = (r_[0] + Δr_[0]/2)**2/(r_[1] - r_[0])
            Kcs__[0, :2] = a, -a    # 首行
            a = (r_[-1] - Δr_[-1]/2)**2/(r_[-1] - r_[-2])
            Kcs__[-1, -2:] = -a, a  # 末行
            Kcs__[idxm_, idx_[:-2]]  = a_ = -(r_[1:-1] - Δr_[1:-1]/2)**2/(r_[1:-1] - r_[:-2])  # 下对角线
            Kcs__[idxm_, idx_[2:]]   = c_ = -(r_[1:-1] + Δr_[1:-1]/2)**2/(r_[2:] - r_[1:-1])   # 上对角线
            Kcs__[idxm_, idx_[1:-1]] = -(a_ + c_)                                              # 主对角线
            Kcs__ /= (((r_ + Δr_/2)**3 - (r_ - Δr_/2)**3)/3).reshape(-1, 1)
            diag = Kcs__.diagonal
            band__[0, 1:]  = diag(1)   # 上对角线
            band__[1, :]   = diag(0)   # 主对角线
            band__[2, :-1] = diag(-1)  # 下对角线

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
        idxjintneg_, idxjintpos_ = self.idxjintneg_, self.idxjintpos_
        K__[idxφeneg_, idxjintneg_] = Δxneg  # jintneg列
        K__[idxφepos_, idxjintpos_] = Δxpos  # jintpos列
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
        self.K__[self.idxcsnegsurf_, self.idxjintneg_] = 1/(aneg*DFNP2D.F*Dsneg)

    def update_K__idxcspossurf_idxjintpos_(self, apos, Dspos):
        # 更新K__矩阵cspossurf行jintpos列
        self.K__[self.idxcspossurf_, self.idxjintpos_] = 1/(apos*DFNP2D.F*Dspos)

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

    def __str__(self):
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

    @staticmethod
    def solve_banded_matrix(
            A__: ndarray,          # 矩阵
            b_:  ndarray,          # A__ @ X_ = b_
            idxReordered_: Sequence[int],  # 索引：重排，使矩阵A__带状化
            idxRecovered_: Sequence[int],  # 索引：恢复排序
            bandwidths_: dict):
        """解带状化矩阵"""
        N = A__.shape[0]  # 因变量总数
        u, l = bandwidths_['upper'], bandwidths_['lower']  # 上下带宽
        Areordered__ = A__[ix_(idxReordered_, idxReordered_)]       # 重新排列矩阵A__，使之带状化
        ab__ = zeros((2*l + u + 1, N), dtype=A__.dtype, order='F')  # 适合dgbsv
        band__ = ab__[l:, :]   # (u + l + 1, N) 矩阵A__的带
        diag = Areordered__.diagonal
        for row, offset in enumerate(range(u, -l-1, -1)):
            d_ = diag(offset)  # 提取矩阵Areordered__的带
            start = max(0, offset)
            end   = min(N, N + offset)
            band__[row, start:end] = d_
        X_ = dgbsv(l, u, ab__, asfortranarray(b_[idxReordered_]), True, True)[2]
        # X_ = solve_banded((l, u), band__, asfortranarray(b_[idxReordered_]), True, True, False)
        return X_[idxRecovered_]

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

    def count_lithium(self):
        """统计锂电荷量"""
        qsneg = self.θsneg*self.Qneg
        qspos = self.θspos*self.Qpos
        qe = (self.ce_*self.Δx_*self.A*self.εe_).sum()*DFNP2D.F/3600
        print(f'合计锂电荷总量{qsneg + qspos + qe + self.QLP:.8g} Ah = '
              f'负极嵌锂{qsneg:.8g} Ah + 正极嵌锂{qspos:.8g} Ah + '
              f'电解液锂{qe:.8g} Ah + 负极析锂{self.QLP:.8g} Ah')

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
        assert self.Δt>=minΔt, f'时间步长Δt = {self.Δt} s，应大于{minΔt = } s'
        verbose = self.verbose
        if verbose:
            startTime = time.time()  # 开始时间戳 [s]
        lithiumPlating = self.lithiumPlating
        if self.t==0:
            self.record_data()        # 记录初始时刻数据
        data = self.data              # 读取：运行数据字典
        tStart = data['t'][-1]        # 开始时刻 [s]
        tEnd = tStart + timeInterval  # 终止时刻 [s]
        self.I = I  # 电流 [A]
        C = abs(I)/self.Qcell  # 电流倍率 [C]
        运行状态 = f'{C:.2f}C放电' if I>0 else (f'{C:.2f}C充电' if I<0 else '静置')

        while True:
            ### 持续时间步进...
            # 检查终止条件
            if self.t>=tEnd:
                if verbose:
                    print(f'\n达到运行时长{timeInterval} s，停止{运行状态}')
                break
            if I<0 and (Umax is not None) and self.U>=Umax:
                if verbose:
                    print(f'\n充电达到最高电压{Umax} V，停止充电')
                break
            if I<0 and (SOCmax is not None) and self.SOC>=SOCmax:
                if verbose:
                    print(f'\n充电达到最大SOC {SOCmax}，停止充电')
                break
            if I>0 and (Umin is not None) and self.U<=Umin:
                if verbose:
                    print(f'\n放电达到最低电压{Umin} V，停止放电')
                break
            if I>0 and (SOCmin is not None) and self.SOC<=SOCmin:
                if verbose:
                    print(f'\n放电达到最小SOC {SOCmin}，停止放电')
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
                except DFNP2D.Error as message:
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
                print(f'|{finishedBar}{unfinishedBar}|已完成{percentage:.0f}%，耗时{timeStamp:.1f}s，'
                      f't={self.t:g}s-->{tEnd:g}s，{运行状态}，'
                      f'电压{self.U:.3f}V，SOC{self.SOC*100:.1f}%，温度{self.T - 273.15:.1f}°C，'
                      f'析锂过电位{self.ηLPneg_[-1]*1000:.0f}mV，'
                      f'{nNewton}次Newton迭代'
                      f'\r', end='')
        return self

    def step(self, Δt):
        """时间步进：Newton法迭代因变量"""
        idxcsneg_ = self.idxcsneg_
        idxcspos_ = self.idxcspos_
        idxcsnegsurf_ = self.idxcsnegsurf_
        idxcspossurf_ = self.idxcspossurf_
        idxce_ = self.idxce_
        idxφsneg_ = self.idxφsneg_
        idxφspos_ = self.idxφspos_
        idxφe_ = self.idxφe_
        idxjintneg_ = self.idxjintneg_
        idxjintpos_ = self.idxjintpos_
        idxjDLneg_ = self.idxjDLneg_
        idxjDLpos_ = self.idxjDLpos_
        idxi0intneg_ = self.idxi0intneg_
        idxi0intpos_ = self.idxi0intpos_
        idxηintneg_ = self.idxηintneg_
        idxηintpos_ = self.idxηintpos_
        idxηLP_ = self.idxηLP_
        idxjLP_ = self.idxjLP_
        idxc_ = self.idxc_
        idxφ_ = self.idxφ_
        idxj_ = self.idxj_

        # 读取方法
        solve_banded_matrix = DFNP2D.solve_banded_matrix
        solve_jint_ = DFNP2D.solve_jint_
        solve_djintdηint_ = DFNP2D.solve_djintdηint_
        solve_djintdi0int_ = DFNP2D.solve_djintdi0int_
        solve_i0int_ = DFNP2D.solve_i0int_
        solve_di0intdcssurf_ = DFNP2D.solve_di0intdcssurf_
        solve_di0intdce_ = DFNP2D.solve_di0intdce_
        solve_UOCPneg_, solve_UOCPpos_ = self.solve_UOCPneg_, self.solve_UOCPpos_                  # 读取：负极、正极开路电位函数 [V]
        solve_dUOCPdθsneg_, solve_dUOCPdθspos_ = self.solve_dUOCPdθsneg_, self.solve_dUOCPdθspos_  # 读取：负极、正极开路电位对嵌锂状态的偏导数函数 [V/–]

        data = self.data  # 读取：运行数据字典
        Nneg, Nsep, Npos, Ne, Nr = self.Nneg, self.Nsep, self.Npos, self.Ne, self.Nr  # 读取：网格数
        Δxneg, Δxpos, Δx_, x_ = self.Δxneg, self.Δxpos, self.Δx_, self.x_             # 读取：网格尺寸 [m]
        ΔxWest_, ΔxEast_ = self.ΔxWest_, self.ΔxEast_  # 读取：网格距离 [m]
        lithiumPlating = self.lithiumPlating          # 是否考虑析锂反应
        timeDiscretization = self.timeDiscretization  # 时间离散格式
        decouple_cs = self.decouple_cs   # 是否解耦固相锂离子浓度的求解
        verbose = self.verbose

        Rsneg, Rspos = self.Rsneg, self.Rspos              # 读取：颗粒半径 [m]
        aneg, apos = self.aneg, self.apos                  # 读取：固相颗粒的比表面积 [m^2/m^3]
        aeffneg, aeffpos = self.aeffneg, self.aeffpos      # 读取：负极、正极材料与电解质的有效比表面积
        σeffneg, σeffpos = self.σeffneg, self.σeffpos      # 读取：固相有效电导率 [S/m]
        csmaxneg, csmaxpos = self.csmaxneg, self.csmaxpos  # 读取：固相最大锂离子浓度 [mol/m^3]
        RSEIneg, RSEIpos = self.RSEIneg, self.RSEIpos      # 读取：负极、正极SEI膜面积电阻 [Ω·m^2]
        Dsneg, Dspos = self.Dsneg, self.Dspos              # 读取：负极、正极固相扩散系数 [m^2/s]
        DeeffWest_ = DeeffEast_ = self.Deeff_
        κeffWest_ = κeffEast_ = self.κeff_
        κDeffWest_ = κDeffEast_ = self.κDeff_
        εe_ = self.εe_
        T, F, R= self.T, DFNP2D.F, DFNP2D.R  # 读取：温度 [K]、法拉第常数 [C/mol]、摩尔气体常数 [J/(mol·K)]
        F2RT = F/(2*R*T)          # 一个常数 [1/V]
        i = (I := self.I)/self.A  # 电流密度 [A/m^2]
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
            solve_jLP_      = DFNP2D.solve_jLP_
            solve_djLPdce_  = DFNP2D.solve_djLPdce_
            solve_djLPdηLP_ = DFNP2D.solve_djLPdηLP_
            solve_i0LP_     = DFNP2D.solve_i0LP_
            if i0LPUnknown := (self._i0LP is None):
                kLP = self.kLP    # 读取：负极析锂反应速率常数
            else:
                i0LP = self.i0LP  # 读取：负极析锂反应交换电流密度 [A/m^2]

        if self.constants:
            pass
        else:
            # 更新K__矩阵的参数相关值
            if decouple_cs:
                pass
            else:
                self.update_K__idxcsnegsurf_idxjintneg_(aneg, Dsneg)
                self.update_K__idxcspossurf_idxjintpos_(apos, Dspos)
            self.update_K__idxφsneg_idxjneg_(σeffneg)
            self.update_K__idxφspos_idxjpos_(σeffpos)
            self.update_K__idxφe_idxφe_(κeffWest_, κeffEast_)
            self.update_K__idxηintneg_idxjneg_(RSEIneg, aeffneg)
            self.update_K__idxηintpos_idxjpos_(RSEIpos, aeffpos)
            if lithiumPlating:
                self.update_K__idxηLP_idxjneg_(RSEIneg, aeffneg)

        K__ = self.K__                # 读取：因变量线性矩阵
        bK_ = zeros(K__.shape[0])  # 读取：常数项向量，F_ = K__ @ X_ - bK_

        bandKcsneg__ = (Δt*Dsneg) * self.bandKcsneg__  # (3, Nr)
        bandKcspos__ = (Δt*Dspos) * self.bandKcspos__  # (3, Nr)
        Kcs_jintneg =  Δt * Rsneg**2 / aneg / F / ((Rsneg**3 - (Rsneg - self.Δrneg_[-1])**3)/3)
        Kcs_jintpos =  Δt * Rspos**2 / apos / F / ((Rspos**3 - (Rspos - self.Δrpos_[-1])**3)/3)
        if timeDiscretization=='CN':
            bandKcsneg__ *= .5
            bandKcspos__ *= .5
            Kcs_jintneg *= .5
            Kcs_jintpos *= .5
            bandBcsneg__ = -bandKcsneg__  # (3, Nr)
            bandBcspos__ = -bandKcspos__  # (3, Nr)
            bandBcsneg__[1] += 1  # 对角元+1
            bandBcspos__[1] += 1  # 对角元+1
        bandKcsneg__[1] += 1  # 对角元+1
        bandKcspos__[1] += 1  # 对角元+1

        ## 对K__矩阵赋值 ##
        if decouple_cs:
            # 历史浓度影响的浓度分量
            match timeDiscretization:
                case 'backward':
                    RHSneg__ = self.csneg__  # (Nr, Nneg)
                    RHSpos__ = self.cspos__  # (Nr, Npos)
                case 'CN':
                    RHSneg__ = triband_to_dense(bandBcsneg__) @ self.csneg__  # (Nr, Nneg)
                    RHSpos__ = triband_to_dense(bandBcspos__) @ self.cspos__  # (Nr, Npos)
            e__ = self.e__
            RHSneg__ = concatenate([RHSneg__, e__], axis=1)  # (Nr, Nneg+1)
            RHSpos__ = concatenate([RHSpos__, e__], axis=1)  # (Nr, Npos+1)
            Sneg__ = dgtsv(bandKcsneg__[2, :-1], bandKcsneg__[1], bandKcsneg__[0, 1:], RHSneg__, True, True, True, True)[3]  # (Nr, Nneg+1)
            Spos__ = dgtsv(bandKcspos__[2, :-1], bandKcspos__[1], bandKcspos__[0, 1:], RHSpos__, True, True, True, True)[3]  # (Nr, Npos+1)
            csnegI__ = Sneg__[:, :-1]  # (Nr, Nneg) 内部锂离子浓度的历史影响分量
            csposI__ = Spos__[:, :-1]  # (Nr, Npos)
            γneg_ = Sneg__[:, -1] * -Kcs_jintneg  # (Nr,)
            γpos_ = Spos__[:, -1] * -Kcs_jintpos  # (Nr,)
            # 3点2次多项式外推颗粒表面锂离子浓度的历史影响分量
            # backward: cssurf_ = α_ + jint_*β
            # CN:       cssurf_ = α_ + (jint_ + jintold)*β
            coeffs_ = self.coeffs_
            αneg_ = coeffs_.dot(csnegI__[-3:])  # (Nneg,)
            αpos_ = coeffs_.dot(csposI__[-3:])  # (Npos,)
            βneg = coeffs_.dot(γneg_[-3:])
            βpos = coeffs_.dot(γpos_[-3:])
            # 负极、正极固相表面浓度cssurf行
            K__[idxcsnegsurf_, idxjintneg_] = -βneg
            K__[idxcspossurf_, idxjintpos_] = -βpos
        else:
            # 负极、正极固相内部浓度csneg行、cspos行
            for band__, idxcs_, Nreg in zip(
                    [bandKcsneg__, bandKcspos__], [idxcsneg_, idxcspos_], [Nneg, Npos]):
                idx__ = idxcs_.reshape(Nreg, Nr)
                K__[idx__[:, :-1].ravel(), idx__[:, 1:].ravel()] = tile(band__[0, 1:], Nreg)   # 上对角线
                K__[idxcs_, idxcs_]                              = tile(band__[1], Nreg)       # 主对角线
                K__[idx__[:, 1:].ravel(), idx__[:, :-1].ravel()] = tile(band__[2, :-1], Nreg)  # 下对角线
            K__[idxcsneg_[Nr-1::Nr], idxjintneg_] = Kcs_jintneg  # jintneg列
            K__[idxcspos_[Nr-1::Nr], idxjintpos_] = Kcs_jintpos  # jintpos列
        # 电解液浓度ce行ce列
        a = DeeffEast_[0]/ΔxEast_[0]
        K__[idxce_[0], idxce_[:2]] = [a, -a]    # ce列首行
        a = DeeffWest_[-1]/ΔxWest_[-1]
        K__[idxce_[-1], idxce_[-2:]] = [-a, a]  # ce列末行
        K__[idxce_[1:-1], idxce_[:-2]]  = a_ = -DeeffWest_[1:-1]/ΔxWest_[1:-1]  # ce列下对角线
        K__[idxce_[1:-1], idxce_[2:]]   = c_ = -DeeffEast_[1:-1]/ΔxEast_[1:-1]  # ce列上对角线
        K__[idxce_[1:-1], idxce_[1:-1]] = -(a_ + c_)                            # ce列主对角线
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, c = -DeeffWest_[nW]/ΔxWest_[nW], -2*DeeffEast_[nW]*DeeffWest_[nE]/(DeeffEast_[nW]*Δx_[nE] + DeeffWest_[nE]*Δx_[nW])
            K__[idxce_[nW], idxce_[nW-1:nW+2]] = [a, -(a + c), c]  # 界面左侧控制体
            a, c = c, -DeeffEast_[nE]/ΔxEast_[nE]
            K__[idxce_[nE], idxce_[nE-1:nE+2]] = [a, -(a + c), c]  # 界面右侧控制体
        Δt2Δx_ = Δt/Δx_
        K__[idxce_[1:], idxce_[:-1]] *= Δt2Δx_[1:]   # ce列下对角线
        K__[idxce_[:-1], idxce_[1:]] *= Δt2Δx_[:-1]  # ce列上对角线
        K__[idxce_, idxce_] *= Δt2Δx_                # ce列主对角线
        if timeDiscretization=='CN':
            K__[idxce_[1:], idxce_[:-1]] *= .5
            K__[idxce_[:-1], idxce_[1:]] *= .5
            K__[idxce_, idxce_] *= .5
            start = idxce_[0]
            end = idxce_[-1] + 1
            Kce__ = -K__[start:end, start:end]
            Kce__.ravel()[::Ne+1] += εe_  # 对角元+εe_
        K__[idxce_, idxce_] += εe_

        # 电解液浓度ce行j列
        Kce_j = -Δt*(1 - self.tplus)/F
        if timeDiscretization=='CN':
            Kce_j *= .5
        idxceneg_, idxcepos_ = idxce_[:Nneg], idxce_[-Npos:]
        K__[idxceneg_, idxjintneg_] = Kce_j  # jintneg列
        K__[idxcepos_, idxjintpos_] = Kce_j  # jintpos列
        if jDLnegUnknown := (idxjDLneg_.size>0):
            K__[idxceneg_, idxjDLneg_] = Kce_j  # jDLneg列
        if jDLposUnknown := (idxjDLpos_.size>0):
            K__[idxcepos_, idxjDLpos_] = Kce_j  # jDLpos列
        if lithiumPlating:
            K__[idxceneg_, idxjLP_] = Kce_j  # jLP列

        # 双电层局部体积电流密度jDLneg、jDLpos行
        if jDLnegUnknown or jDLposUnknown:
            if jDLnegUnknown:
                aCDLneg = aeffneg*self.CDLneg  # 负极双电层体积电容 [F/m^3]
            if jDLposUnknown:
                aCDLpos = aeffpos*self.CDLpos  # 正极双电层体积电容 [F/m^3]
            Nt = len(data['t'])  # 存储数据时刻数
            t_1 = data['t'][-1]  # 上一时刻 [s]
            t_2 = data['t'][-2] if Nt>1 else None  # 上上一时刻
            t_3 = data['t'][-3] if Nt>2 else None  # 上上上一时刻
            t = t_1 + Δt  # 当前时刻 [s]
            c = 1/Δt
            if Nt>1:
                c += 1/(t - t_2)
            if Nt>2:
                c += 1/(t - t_3)
            if jDLnegUnknown:
                aC2Δtneg = aCDLneg*c
                # 负极双电层局部体积电流密度jDLneg行
                K__[idxjDLneg_, idxφe_[:Nneg]] = aC2Δtneg  # φe负极列
                K__[idxjDLneg_, idxφsneg_] = -aC2Δtneg     # φsneg列
                K__[idxjDLneg_, idxjintneg_] = a = aC2Δtneg*RSEI2aeffneg  # jintneg列
                K__[idxjDLneg_, idxjDLneg_]  = 1 + a       # jDLneg列
                if lithiumPlating:
                    K__[idxjDLneg_, idxjLP_] = a  # jLP列
            if jDLposUnknown:
                aC2Δtpos = aCDLpos*c
                # 正极双电层局部体积电流密度jDLpos行
                K__[idxjDLpos_, idxφe_[-Npos:]] = aC2Δtpos  # φe正极列
                K__[idxjDLpos_, idxφspos_] = -aC2Δtpos      # φspos列
                K__[idxjDLpos_, idxjintpos_] = a = aC2Δtpos*RSEI2aeffpos      # jintpos列
                K__[idxjDLpos_, idxjDLpos_]  = 1 + a  # jDLpos列

        ## b向量（常数值、固液相浓度场旧值）##
        bK_[idxφsneg_[0]]  = -Δxneg*i/σeffneg
        bK_[idxφspos_[-1]] =  Δxpos*i/σeffpos
        match timeDiscretization:
            case 'backward':
                if decouple_cs:
                    bK_[idxcsnegsurf_] = αneg_
                    bK_[idxcspossurf_] = αpos_
                else:
                    bK_[idxcsneg_] = self.csneg__.ravel('F')
                    bK_[idxcspos_] = self.cspos__.ravel('F')
                bK_[idxce_] = εe_*self.ce_
            case 'CN':
                if decouple_cs:
                    bK_[idxcsnegsurf_] = αneg_ + βneg*self.jintneg_
                    bK_[idxcspossurf_] = αpos_ + βpos*self.jintpos_
                else:
                    bK_[idxcsneg_] = (triband_to_dense(bandBcsneg__) @ self.csneg__).ravel('F')
                    bK_[idxcspos_] = (triband_to_dense(bandBcspos__) @ self.cspos__).ravel('F')
                    bK_[idxcsneg_[Nr-1::Nr]] -= Kcs_jintneg * self.jintneg_
                    bK_[idxcspos_[Nr-1::Nr]] -= Kcs_jintpos * self.jintpos_
                bK_[idxce_] = Kce__.dot(self.ce_)
                bK_[idxceneg_] -= Kce_j * self.jneg_
                bK_[idxcepos_] -= Kce_j * self.jpos_

        if jDLnegUnknown or jDLposUnknown:
            # 上一时刻负极、正极固液相电势场之差
            Δφseneg_1_ = data['ηLPneg_'][-1]
            Δφsepos_1_ = data['ηLPpos_'][-1]
            # 上上时刻
            Δφseneg_2_ = data['ηLPneg_'][-2] if (jDLnegUnknown and Nt>1) else None
            Δφsepos_2_ = data['ηLPpos_'][-2] if (jDLposUnknown and Nt>1) else None
            # 上上上时刻
            Δφseneg_3_ = data['ηLPneg_'][-3] if (jDLnegUnknown and Nt>2) else None
            Δφsepos_3_ = data['ηLPpos_'][-3] if (jDLposUnknown and Nt>2) else None
            if Nt>2:
                A = (t - t_2)*(t - t_3)/-Δt/(t_1 - t_2)/(t_1 - t_3)
                B = Δt*(t - t_3)/(t_2 - t)/(t_2 - t_1)/(t_2 - t_3)
                C = Δt*(t - t_2)/(t_3 - t)/(t_3 - t_1)/(t_3 - t_2)
                if jDLnegUnknown:
                    bK_[idxjDLneg_] = aCDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_ + C*Δφseneg_3_)
                if jDLposUnknown:
                    bK_[idxjDLpos_] = aCDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_ + C*Δφsepos_3_)
            elif Nt==2:
                A = (t - t_2)/(-Δt*(t_1 - t_2))
                B = Δt/((t_2 - t)*(t_2 - t_1))
                if jDLnegUnknown:
                    bK_[idxjDLneg_] = aCDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_)
                if jDLposUnknown:
                    bK_[idxjDLpos_] = aCDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_)
            else:
                if jDLnegUnknown:
                    bK_[idxjDLneg_] = -aC2Δtneg*Δφseneg_1_
                if jDLposUnknown:
                    bK_[idxjDLpos_] = -aC2Δtpos*Δφsepos_1_

        # 初始化解向量
        X_ = zeros_like(bK_)
        if decouple_cs:
            pass
        else:
            X_[idxcsneg_] = self.csneg__.ravel('F')
            X_[idxcspos_] = self.cspos__.ravel('F')
        X_[idxcsnegsurf_] = self.csnegsurf_
        X_[idxcspossurf_] = self.cspossurf_
        X_[idxce_] = self.ce_
        if i0intnegUnknown:
            X_[idxi0intneg_] = self.i0intneg_
        if i0intposUnknown:
            X_[idxi0intpos_] = self.i0intpos_
        if I==data['I'][-1]:
            # 恒电流
            X_[idxφsneg_] = self.φsneg_
            X_[idxφspos_] = self.φspos_
            X_[idxφe_] = self.φe_
            X_[idxjintneg_] = self.jintneg_
            X_[idxjintpos_] = self.jintpos_
            X_[idxηintneg_] = self.ηintneg_
            X_[idxηintpos_] = self.ηintpos_
        else:
            X_[idxφe_] = 0
            jintneg =  i/self.Lneg
            jintpos = -i/self.Lpos
            X_[idxjintneg_] = jintneg
            X_[idxjintpos_] = jintpos
            i0intneg_ = X_[idxi0intneg_] if i0intnegUnknown else i0intneg
            i0intpos_ = X_[idxi0intpos_] if i0intposUnknown else i0intpos
            X_[idxηintneg_] = arcsinh(jintneg/(2*aeffneg*i0intneg_))/F2RT
            X_[idxηintpos_] = arcsinh(jintpos/(2*aeffpos*i0intpos_))/F2RT
            X_[idxφsneg_] = X_[idxηintneg_] + RSEI2aeffneg*jintneg + solve_UOCPneg_(X_[idxcsnegsurf_]/csmaxneg)
            X_[idxφspos_] = X_[idxηintpos_] + RSEI2aeffpos*jintpos + solve_UOCPpos_(X_[idxcspossurf_]/csmaxpos)

        if lithiumPlating:
            X_[idxηLP_] = ηLP_ = X_[idxφsneg_] - X_[idxφe_[:Nneg]] - RSEI2aeffneg*X_[idxjintneg_]  # 初始化ηLPneg_

        J__ = K__.copy()  # 初始化Jacobi矩阵
        ΔFφe_ = zeros(Ne)
        for nNewton in range(1, 201):
            ## Newton迭代
            F_ = K__.dot(X_) - bK_  # F残差向量的线性部分

            # 提取解
            csnegsurf_, cspossurf_ = X_[idxcsnegsurf_], X_[idxcspossurf_]
            ce_ = X_[idxce_]
            ceneg_, cepos_ = ce_[:Nneg], ce_[-Npos:]
            i0intneg_ = X_[idxi0intneg_] if i0intnegUnknown else i0intneg
            i0intpos_ = X_[idxi0intpos_] if i0intposUnknown else i0intpos
            ηintneg_, ηintpos_ = X_[idxηintneg_], X_[idxηintpos_]

            # F向量非线性部分
            ΔFφe_[0]  = -κDeffEast_[0]  * (ce_[1] - ce_[0]  )/ΔxEast_[0] / (0.5*(ce_[1] + ce_[0]))
            ΔFφe_[-1] =  κDeffWest_[-1] * (ce_[-1] - ce_[-2])/ΔxWest_[-1] / (0.5*(ce_[-1] + ce_[-2]))
            ΔFφe_[1:-1] = -( κDeffEast_[1:-1] * (ce_[2:] - ce_[1:-1] )/ΔxEast_[1:-1] / (0.5*(ce_[2:] + ce_[1:-1]))
                            -κDeffWest_[1:-1] * (ce_[1:-1] - ce_[:-2])/ΔxWest_[1:-1] / (0.5*(ce_[1:-1] + ce_[:-2])) )
            for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
                # 修正负极-隔膜界面、隔膜-正极界面
                a, b = DeeffWest_[nE]*Δx_[nW], DeeffEast_[nW]*Δx_[nE]
                cInterface = (a*ce_[nE] + b*ce_[nW])/(a + b)
                ΔFφe_[nW] = -( κDeffEast_[nW] * (cInterface - ce_[nW])/(0.5*Δx_[nW]) / cInterface
                              -κDeffWest_[nW] * (ce_[nW] - ce_[nW-1])/ΔxWest_[nW]  / (0.5*(ce_[nW] + ce_[nW-1])) )
                ΔFφe_[nE] = -( κDeffEast_[nE] * (ce_[nE+1] - ce_[nE] )/ΔxEast_[nE] / (0.5*(ce_[nE+1] + ce_[nE]))
                              -κDeffWest_[nE] * (ce_[nE] - cInterface)/(0.5*Δx_[nE]) / cInterface )
            F_[idxφe_] += ΔFφe_
            F_[idxjintneg_] -= solve_jint_(T, aeffneg, i0intneg_, ηintneg_) # F向量jintneg部分
            F_[idxjintpos_] -= solve_jint_(T, aeffpos, i0intpos_, ηintpos_)  # F向量jintpos部分
            if i0intnegUnknown:
                F_[idxi0intneg_] -= solve_i0int_(kneg, csmaxneg, csnegsurf_, ceneg_) # F向量i0intneg部分
            if i0intposUnknown:
                F_[idxi0intpos_] -= solve_i0int_(kpos, csmaxpos, cspossurf_, cepos_) # F向量i0intpos部分
            F_[idxηintneg_] += solve_UOCPneg_(csnegsurf_/csmaxneg)  # F向量ηintneg非线性部分
            F_[idxηintpos_] += solve_UOCPpos_(cspossurf_/csmaxpos)  # F向量ηintpos非线性部分
            if lithiumPlating:
                ηLP_ = X_[idxηLP_]
                i0LP_ = solve_i0LP_(kLP, ceneg_) if i0LPUnknown else i0LP  # 负极析锂反应的交换电流密度场 [A/m^2]
                F_[idxjLP_] -= solve_jLP_(T, aeffneg, i0LP_, ηLP_)  # F向量jLP部分

            # 更新Jacobi矩阵非线性部分
            # φe行ce列
            a = κDeffEast_[0] /  (0.5*(ce_[1] + ce_[0]) * ΔxEast_[0])
            aa = a * (ce_[1] - ce_[0]) / (ce_[1] + ce_[0])
            J__[idxφe_[0], idxce_[:2]] = [aa + a, aa - a]      # ce首行起始2列

            a = κDeffWest_[-1] / (0.5*(ce_[-1] + ce_[-2]) * ΔxWest_[-1])
            aa = a * (ce_[-1] - ce_[-2]) / (ce_[-1] + ce_[-2])
            J__[idxφe_[-1], idxce_[-2:]] = [-aa - a, -aa + a]  # ce末行末尾2列

            a_ = κDeffWest_[1:-1] / (0.5*(ce_[1:-1] + ce_[:-2]) * ΔxWest_[1:-1])
            aa_ = a_ * (ce_[1:-1] - ce_[:-2]) / (ce_[1:-1] + ce_[:-2])
            c_ = κDeffEast_[1:-1] / (0.5*(ce_[1:-1] + ce_[2:]) * ΔxEast_[1:-1])
            cc_ = c_ * (ce_[2:] - ce_[1:-1]) / (ce_[2:] + ce_[1:-1])
            J__[idxφe_[1:-1], idxce_[:-2]] = - aa_ - a_            # 下对角线
            J__[idxφe_[1:-1], idxce_[2:]] = cc_ - c_               # 上对角线
            J__[idxφe_[1:-1], idxce_[1:-1]] = cc_ + c_ - aa_ + a_  # 主对角线
            for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
                # 修正负极-隔膜界面、隔膜-正极界面
                num = κDeffEast_[nW]*DeeffWest_[nE] - κDeffWest_[nE]*DeeffEast_[nW]
                den1 = κeffWest_[nE]*Δx_[nW] + κeffEast_[nW]*Δx_[nE]
                den2 = DeeffWest_[nE]*ce_[nE]*Δx_[nW] + DeeffEast_[nW]*ce_[nW]*Δx_[nE]
                product = den1*den2

                a = 2*κDeffWest_[nW] / ((ce_[nW] + ce_[nW-1]) * Δx_[nW])
                aa = a * (ce_[nW] - ce_[nW-1]) / (ce_[nW] + ce_[nW-1])
                c = 2*Δx_[nE]*κeffEast_[nW]*num / product
                cc = c * (ce_[nE] - ce_[nW])*DeeffWest_[nE]*Δx_[nW] / den2
                d = 2*κDeffEast_[nW]*DeeffWest_[nE] / den2
                dd = d * (ce_[nE] - ce_[nW])*DeeffWest_[nE]*Δx_[nW] / den2
                J__[idxφe_[nW], idxce_[nW-1:nW+2]] = [
                    -a - aa,
                    -c - cc/Δx_[nW]*Δx_[nE]/DeeffWest_[nE]*DeeffEast_[nW] + d + dd/Δx_[nW]*Δx_[nE]/DeeffWest_[nE]*DeeffEast_[nW] + a - aa,
                    c - cc - d + dd]  # 界面左侧控制体

                a = 2*κeffWest_[nE]*Δx_[nW]*num / product
                aa = a * (ce_[nE] - ce_[nW])*DeeffEast_[nW]*Δx_[nE] / den2
                c = 2*κDeffWest_[nE]*DeeffEast_[nW] / den2
                cc = c * (ce_[nE] - ce_[nW])*DeeffEast_[nW]*Δx_[nE] / den2
                d = 2*κDeffEast_[nE] / ((ce_[nE] + ce_[nE+1]) * Δx_[nE])
                dd = d * (ce_[nE] - ce_[nE+1]) / (ce_[nE] + ce_[nE+1])
                J__[idxφe_[nE], idxce_[nE-1:nE+2]] = [
                    -a - aa - c - cc,
                    a - aa/Δx_[nE]*Δx_[nW]/DeeffEast_[nW]*DeeffWest_[nE] + c - cc/Δx_[nE]*Δx_[nW]/DeeffEast_[nW]*DeeffWest_[nE] + d - dd,
                    -d - dd]  # 界面右侧

            J__[idxjintneg_, idxηintneg_] = -solve_djintdηint_(T, aeffneg, i0intneg_, ηintneg_) # ∂Fjintneg/∂ηintneg
            J__[idxjintpos_, idxηintpos_] = -solve_djintdηint_(T, aeffpos, i0intpos_, ηintpos_) # ∂Fjintpos/∂ηintpos
            if i0intnegUnknown:
                J__[idxjintneg_, idxi0intneg_] = -solve_djintdi0int_(T, aeffneg, ηintneg_)  # ∂Fjintneg/∂i0intneg
                J__[idxi0intneg_, idxce_[:Nneg]] = -solve_di0intdce_(ceneg_, i0intneg_)     # ∂Fi0intneg/∂ce
                J__[idxi0intneg_, idxcsnegsurf_] = -solve_di0intdcssurf_(kneg, csmaxneg, csnegsurf_, ceneg_, i0intneg_)  # ∂Fi0intneg/∂csnegsurf
            if i0intposUnknown:
                J__[idxjintpos_, idxi0intpos_] = -solve_djintdi0int_(T, aeffpos, ηintpos_)  # ∂Fjintpos/∂i0intpos
                J__[idxi0intpos_, idxce_[-Npos:]] = -solve_di0intdce_(cepos_, i0intpos_)    # ∂Fi0intpos/∂ce
                J__[idxi0intpos_, idxcspossurf_] = -solve_di0intdcssurf_(kpos, csmaxpos, cspossurf_, cepos_, i0intpos_)  # ∂Fi0intpos/∂cspossurf
            J__[idxηintneg_, idxcsnegsurf_] = solve_dUOCPdθsneg_(csnegsurf_/csmaxneg) / csmaxneg  # ∂Fηintneg/∂csnegsurf
            J__[idxηintpos_, idxcspossurf_] = solve_dUOCPdθspos_(cspossurf_/csmaxpos) / csmaxpos  # ∂Fηintpos/∂cspossurf
            if lithiumPlating:
                J__[idxjLP_, idxce_[:Nneg]] = -solve_djLPdce_(T, aeffneg, ceneg_, i0LP_, ηLP_)  # ∂FjLP/∂ce
                J__[idxjLP_, idxηLP_] = -solve_djLPdηLP_(T, aeffneg, i0LP_, ηLP_)               # ∂FjLP/∂ηLP

            if self.bandwidthsJ_ is None and any(self.data['I']):
                if verbose:
                    print('辨识重排因变量Jacobi矩阵的带宽 -> ', end='')
                self.idxJreordered_ = idxJreordered_ = reverse_cuthill_mckee(csr_matrix(J__))
                self.idxJrecovered_ = idxJreordered_.argsort()
                self.bandwidthsJ_ = DFNP2D.identify_bandwidths(J__[ix_(idxJreordered_, idxJreordered_)])
                if verbose:
                    print(f'上带宽{self.bandwidthsJ_['upper']}，下带宽{self.bandwidthsJ_['lower']}')

            # Newton迭代新解向量
            if bandwidthsJ_ := self.bandwidthsJ_:
                # 带状化求解
                ΔX_ = solve_banded_matrix(J__, F_,
                    self.idxJreordered_, self.idxJrecovered_, bandwidthsJ_)
            else:
                # 直接求解
                ΔX_ = solve(J__, F_)

            X_ -= ΔX_

            if isnan(X_).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = } s，Newton迭代出现nan')
            if (X_[idxce_]<=0).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = } s，Newton迭代出现ce<=0')
            csnegsurf_ = X_[idxcsnegsurf_]
            if (csnegsurf_<=0).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = } s，Newton迭代出现csnegsurf<=0')
            if (csnegsurf_>=csmaxneg).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = } s，Newton迭代出现csnegsurf>=csmaxneg')
            cspossurf_ = X_[idxcspossurf_]
            if (cspossurf_<=0).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = } s，Newton迭代出现cspossurf<=0')
            if (cspossurf_>=csmaxpos).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = } s，Newton迭代出现cspossurf>=csmaxpos')

            ΔX_ = abs(ΔX_)
            maxΔφ = ΔX_[idxφ_].max()  # 新旧电势场最大绝对误差
            maxΔc = ΔX_[idxc_].max()  # 新旧浓度场最大绝对误差
            maxΔj = ΔX_[idxj_].max()  # 新旧局部体积电流密度场最大绝对误差
            if maxΔj<0.1 and maxΔc<0.1 and maxΔφ<1e-3:
                break
        else:
            if verbose:
                t = self.t
                print(f'时刻{t = } s，Newton迭代达到最大次数{nNewton}，'
                      f'{maxΔφ = :.6f} V，'
                      f'{maxΔc = :.4f} mol/m^3，'
                      f'{maxΔj = :.3f} A/m^3')

        jintneg_ = X_[idxjintneg_]
        jintpos_ = X_[idxjintpos_]
        if decouple_cs:
            match timeDiscretization:
                case 'CN':
                    self.csneg__[:] = csnegI__ + outer(γneg_, jintneg_ + self.jintneg_)
                    self.cspos__[:] = csposI__ + outer(γpos_, jintpos_ + self.jintpos_)
                case 'backward':
                    self.csneg__[:] = csnegI__ + outer(γneg_, jintneg_)
                    self.cspos__[:] = csposI__ + outer(γpos_, jintpos_)
        else:
            self.csneg__[:] = X_[idxcsneg_].reshape(Nr, Nneg, order='F')
            self.cspos__[:] = X_[idxcspos_].reshape(Nr, Npos, order='F')
        self.csnegsurf_[:] = self.solve_csnegsurf_(self.csneg__, jintneg_)
        self.cspossurf_[:] = self.solve_cspossurf_(self.cspos__, jintpos_)
        # self.csnegsurf_[:] = csnegsurf_
        # self.cspossurf_[:] = cspossurf_
        self.ce_[:] = X_[idxce_]
        self.φe_[:] = X_[idxφe_]
        self.φsneg_[:] = φsneg_ = X_[idxφsneg_]
        self.φspos_[:] = φspos_ = X_[idxφspos_]
        self.jintneg_[:] = jintneg_
        self.jintpos_[:] = jintpos_
        self.jneg_[:] = jintneg_
        self.jpos_[:] = jintpos_
        if jDLnegUnknown:
            self.jDLneg_[:] = jDLneg_ = X_[idxjDLneg_]
            self.jneg_ += jDLneg_
        if jDLposUnknown:
            self.jDLpos_[:] = jDLpos_ = X_[idxjDLpos_]
            self.jpos_ += jDLpos_
        self.i0intneg_[:] = X_[idxi0intneg_] if i0intnegUnknown else full(Nneg, i0intneg)
        self.i0intpos_[:] = X_[idxi0intpos_] if i0intposUnknown else full(Npos, i0intpos)
        self.ηintneg_[:] = X_[idxηintneg_]
        self.ηintpos_[:] = X_[idxηintpos_]

        if lithiumPlating:
            self.jLP_[:] = jLP_ = X_[idxjLP_]
            self.i0LP_[:] = solve_i0LP_(kLP, self.ceneg_) if i0LPUnknown else full(Nneg, i0LP)
            self.jneg_ += jLP_

        self.ηLPneg_[:] = φsneg_ - self.φeneg_ - RSEI2aeffneg*self.jneg_
        self.ηLPpos_[:] = φspos_ - self.φepos_ - RSEI2aeffpos*self.jpos_

        return nNewton  # 返回Newton迭代次数

    @staticmethod
    def Arrhenius(X,  # 参考温度下的参数值
                  E,                      # 活化能 [J/mol]
                  T,                      # 温度 [K]
                  Tref#: float = 278.15,  # 参考温度 [K]
                  ):
        """Arrhenius温度修正"""
        if T==Tref or X is None:
            return X
        else:
            return X * exp(E/DFNP2D.R*(1/Tref - 1/T))

    @property
    def kneg(self):
        """负极主反应速率常数"""
        if self.constants:
            return self._kneg
        else:
            return DFNP2D.Arrhenius(self._kneg, self.Ekneg, self.T, self.Tref)
    @kneg.setter
    def kneg(self, kneg):
        self._kneg = kneg

    @property
    def kpos(self):
        """正极主反应速率常数"""
        if self.constants:
            return self._kpos
        else:
            return DFNP2D.Arrhenius(self._kpos, self.Ekpos, self.T, self.Tref)
    @kpos.setter
    def kpos(self, kpos):
        self._kpos = kpos

    @property
    def kLP(self):
        """负极析锂反应速率常数"""
        if self.constants:
            return self._kLP
        else:
            return DFNP2D.Arrhenius(self._kLP, self.EkLP, self.T, self.Tref)
    @kLP.setter
    def kLP(self, kLP):
        self._kLP = kLP

    @property
    def i0intneg(self):
        """负极主反应交换电流密度 [A/m^2]"""
        if self.constants:
            return self._i0intneg
        else:
            return DFNP2D.Arrhenius(self._i0intneg, self.Ekneg, self.T, self.Tref)
    @i0intneg.setter
    def i0intneg(self, i0intneg):
        self._i0intneg = i0intneg

    @property
    def i0intpos(self):
        """正极主反应交换电流密度 [A/m^2]"""
        if self.constants:
            return self._i0intpos
        else:
            return DFNP2D.Arrhenius(self._i0intpos, self.Ekpos, self.T, self.Tref)
    @i0intpos.setter
    def i0intpos(self, i0intpos):
        self._i0intpos = i0intpos

    @property
    def i0LP(self):
        """负极析锂反应交换电流密度 [A/m^2]"""
        if self.constants:
            return self._i0LP
        else:
            return DFNP2D.Arrhenius(self._i0LP, self.EkLP, self.T, self.Tref)
    @i0LP.setter
    def i0LP(self, i0LP):
        self._i0LP = i0LP

    @property
    def Dsneg(self):
        """负极固相扩散系数"""
        if self.constants:
            return self._Dsneg
        else:
            return DFNP2D.Arrhenius(self._Dsneg, self.EDsneg, self.T, self.Tref)
    @Dsneg.setter
    def Dsneg(self, Dsneg):
        self._Dsneg = Dsneg

    @property
    def Dspos(self):
        """正极固相扩散系数"""
        if self.constants:
            return self._Dspos
        else:
            return DFNP2D.Arrhenius(self._Dspos, self.EDspos, self.T, self.Tref)
    @Dspos.setter
    def Dspos(self, Dspos):
        self._Dspos = Dspos

    @property
    def De(self):
        """电解液锂离子扩散系数 [m^2/s]"""
        if self.constants:
            return self._De
        else:
            return DFNP2D.Arrhenius(self._De, self.EDe, self.T, self.Tref)
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
        if self.constants:
            return self._κ
        else:
            return DFNP2D.Arrhenius(self._κ, self.Eκ, self.T, self.Tref)
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
        return (2*DFNP2D.R*self.T*(1 - self.tplus)/DFNP2D.F*self.TDF) * self.κeff_

    @property
    def Qneg(self):
        """负极容量 [Ah]"""
        return DFNP2D.F*self.A*self.Lneg*self.εsneg*self.csmaxneg/3600

    @property
    def Qpos(self):
        """正极容量 [Ah]"""
        return DFNP2D.F*self.A*self.Lpos*self.εspos*self.csmaxpos/3600

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
        return φsposCollector - φsnegCollector

    @property
    def OCV(self):
        """开路电压 [V]"""
        OCPpos = self.solve_UOCPpos_(self.θspos)
        OCPneg = self.solve_UOCPneg_(self.θsneg)
        return OCPpos - OCPneg

    @property
    def SOC(self):
        return (self.θsneg - self.θminneg)/(self.θmaxneg - self.θminneg)

    # @property
    def solve_csnegsurf_(self, csneg__, jintneg_):
        """负极固相表面锂离子浓度场 [mol/m^3]"""
        if self.decouple_cs:
            return self.coeffs_.dot(csneg__[-3:])
        else:
            Nr = self.Nr
            idxcsneg_, idxcsnegsurf_ = self.idxcsneg_,  self.idxcsnegsurf_
            csneg_ = csneg__.ravel('F')
            K__ = self.K__
            return -( K__[idxcsnegsurf_, idxcsneg_[Nr-3::Nr]] * csneg_[Nr-3::Nr]
                    + K__[idxcsnegsurf_, idxcsneg_[Nr-2::Nr]] * csneg_[Nr-2::Nr]
                    + K__[idxcsnegsurf_, idxcsneg_[Nr-1::Nr]] * csneg_[Nr-1::Nr]
                    + K__[idxcsnegsurf_, self.idxjintneg_] * jintneg_)/K__[idxcsnegsurf_, idxcsnegsurf_]

    # @property
    def solve_cspossurf_(self, cspos__, jintpos_):
        """正极固相表面锂离子浓度场 [mol/m^3]"""
        if self.decouple_cs:
            return self.coeffs_.dot(cspos__[-3:])
        else:
            Nr = self.Nr
            idxcspos_, idxcspossurf_ = self.idxcspos_, self.idxcspossurf_
            cspos_ = cspos__.ravel('F')
            K__ = self.K__
            return -(K__[idxcspossurf_, idxcspos_[Nr-3::Nr]] * cspos_[Nr-3::Nr]
                   + K__[idxcspossurf_, idxcspos_[Nr-2::Nr]] * cspos_[Nr-2::Nr]
                   + K__[idxcspossurf_, idxcspos_[Nr-1::Nr]] * cspos_[Nr-1::Nr]
                   + K__[idxcspossurf_, self.idxjintpos_] * jintpos_)/K__[idxcspossurf_, idxcspossurf_]

    @property
    def ceneg_(self):
        """(Nneg,) 负极区域电解液锂离子浓度 [mol/m^3]"""
        return self.ce_[:self.Nneg]

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
        ceInterfaces_ = hstack([ce_[0], (ce_[:-1] + ce_[1:])/2, ce_[-1]])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = DeeffWest_[nE]*Δx_[nW], DeeffEast_[nW]*Δx_[nE]
            ceInterfaces_[nW+1] = (a*ce_[nE] + b*ce_[nW])/(a + b)
        return ceInterfaces_

    @property
    def φeneg_(self):
        """(Nneg,) 负极区域电解液电势 [V]"""
        return self.φe_[:self.Nneg]

    @property
    def φepos_(self):
        """(Npos,) 正极区域电解液电势 [V]"""
        return self.φe_[-self.Npos:]

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
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradceEast_[nW] = (ceEast_[nW] - ce_[nW])/(0.5*Δx_[nW])
            gradceWest_[nE] = (ce_[nE] - ceWest_[nE])/(0.5*Δx_[nE])
        gradlnceWest_ = gradceWest_/ceWest_  # (Ne,) 各控制体左界面的对数锂离子浓度梯度 [ln mol/m^4]
        gradlnceEast_ = gradceEast_/ceEast_  # (Ne,) 各控制体右界面的对数锂离子浓度梯度 [ln mol/m^4]

        φeInterfaces_ = hstack([φe_[0], (φe_[:-1] + φe_[1:])/2, φe_[-1]])
        κeffWest_ = κeffEast_ = self.κeff_
        κDeffWest_ = κDeffEast_ = self.κDeff_
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
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
        return 2*aeff*i0int_*sinh(DFNP2D.F/(2*DFNP2D.R*T) * ηint_)

    @staticmethod
    def solve_djintdi0int_(T, aeff, ηint_) -> ndarray:
        """求解主反应局部体积电流密度jint对交换电流密度i0int的偏导数 [A/m^3 / A/m^2]"""
        return 2*aeff*sinh(DFNP2D.F/(2*DFNP2D.R*T) * ηint_)

    @staticmethod
    def solve_djintdηint_(T, aeff, i0int_, ηint_) -> ndarray:
        """求解主反应局部体积电流密度jint对过电位ηint的偏导数 [A/m^3 / V]"""
        FRT = DFNP2D.F / (DFNP2D.R*T)
        return FRT*aeff*i0int_*cosh(FRT*0.5*ηint_)

    @staticmethod
    def solve_i0int_(k, csmax, cssurf_, ce_) -> ndarray:
        """求主反应交换电流密度 [A/m^2]"""
        return DFNP2D.F * k * sqrt(ce_*(csmax - cssurf_)*cssurf_)

    @staticmethod
    def solve_di0intdce_(ce_, i0int_):
        """求解主反应交换电流密度i0int对电解液锂离子浓度ce的偏导数 [A/m2 / mol/m^3]"""
        return 0.5*i0int_/ce_

    @staticmethod
    def solve_di0intdcssurf_(k, csmax, cssurf_, ce_, i0int_):
        """求解主反应交换电流密度i0int对固相颗粒表面锂离子浓度cssurf的偏导数"""
        Fk = DFNP2D.F*k
        return 0.5*Fk*Fk * ce_*(csmax - 2*cssurf_)/i0int_

    @staticmethod
    def solve_jLP_(T, aeffneg, i0LP_, ηLP_) -> ndarray:
        """求解析锂反应局部体积电流密度jLP [A]"""
        FRT = DFNP2D.F/DFNP2D.R/T
        a, b = 0.3*FRT, -0.7*FRT
        jLP_ = aeffneg * i0LP_ * (exp(a*ηLP_) - exp(b*ηLP_))
        jLP_[ηLP_>=0] = 0
        return jLP_

    @staticmethod
    def solve_djLPdce_(T, aeffneg, ceneg_, i0LP_, ηLP_):
        """析锂反应局部体积电流密度jLP对电解液锂离子浓度ce的偏导数 [A/m^3 / mol/m^3]"""
        FRT = DFNP2D.F/DFNP2D.R/T
        a, b = 0.3*FRT, -0.7*FRT
        djLPdi0LP_ = aeffneg*(exp(a*ηLP_) - exp(b*ηLP_))
        di0LPdce_ = 0.3*i0LP_/ceneg_
        djLPdce_ = djLPdi0LP_ * di0LPdce_
        djLPdce_[ηLP_>=0] = 0
        return djLPdce_

    @staticmethod
    def solve_djLPdηLP_(T, aeffneg, i0LP_, ηLP_):
        """求解析锂反应局部体积电流密度jLP对析锂过电位ηLP的偏导数 [A/m^3 / V]"""
        FRT = DFNP2D.F / (DFNP2D.R*T)
        a, b = 0.3*FRT, -0.7*FRT
        djLPdηLP_ = aeffneg * i0LP_ * (a*exp(a*ηLP_) - b*exp(b*ηLP_))
        djLPdηLP_[ηLP_>=0] = 0
        return djLPdηLP_

    @staticmethod
    def solve_i0LP_(kLP, ceneg_) -> ndarray:
        """由液相浓度场求析锂反应交换电流密度 [A/m^2]"""
        return DFNP2D.F * kLP * ceneg_**0.3

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
            (0 + (ce_[1] - ce_[0])/(x_[1] - x_[0]))/2,       # 负极首个控制体
            (ce_[2:] - ce_[:-2])/(x_[2:] - x_[:-2]),         # 内部控制体
            ((ce_[-1] - ce_[-2])/(x_[-1] - x_[-2]) + 0)/2])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradce_[nW] = ((ce_[nW] - ce_[nW - 1])/ΔxWest_[nW] + (ceEast_[nW] - ce_[nW])/(0.5*Δx_[nW]))/2  # 界面左侧控制体
            gradce_[nE] = ((ce_[nE] - ceWest_[nE])/(0.5*Δx_[nE]) + (ce_[nE + 1] - ce_[nE])/ΔxEast_[nE])/2  # 界面右侧控制体
        return gradce_/ce_

    @property
    def gradφsneg_(self):
        """负极固相电势场的梯度 [V/m]"""
        φsneg_ = self.φsneg_
        Δxneg = self.Δxneg
        return hstack([
            (-self.I/self.A/self.σeffneg + (φsneg_[1] - φsneg_[0])/Δxneg)/2, # 负极首个控制体
            (φsneg_[2:] - φsneg_[:-2])/(2*Δxneg),      # 负极内部控制体
            ((φsneg_[-1] - φsneg_[-2])/Δxneg + 0)/2])  # 负极末尾控制体

    @property
    def gradφspos_(self):
        """正极固相电势场的梯度 [V/m]"""
        φspos_ = self.φspos_
        Δxpos = self.Δxpos
        return hstack([
            (0 + (φspos_[1] - φspos_[0])/Δxpos)/2,  # 正极首个控制体
            (φspos_[2:] - φspos_[:-2])/(2*Δxpos),   # 正极内部控制体
            ((φspos_[-1] - φspos_[-2])/Δxpos + -self.I/self.A/self.σeffpos)/2])  # 正极末尾控制体

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
            (0 + (φe_[1] - φe_[0])/(x_[1] - x_[0]))/2,       # 负极首个控制体
            (φe_[2:] - φe_[:-2])/(x_[2:] - x_[:-2]),         # 内部控制体
            ((φe_[-1] - φe_[-2])/(x_[-1] - x_[-2]) + 0)/2])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradφe_[nW] = ((φe_[nW] - φe_[nW - 1])/ΔxWest_[nW] + (φeEast_[nW] - φe_[nW])/(0.5*Δx_[nW]))/2
            gradφe_[nE] = ((φe_[nE] - φeWest_[nE])/(0.5*Δx_[nE]) + (φe_[nE + 1] - φe_[nE])/ΔxEast_[nE])/2
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
    def Qgen(self):
        """总产热量 [W]"""
        Qohm = self.Qohme + self.Qohmneg + self.Qohmpos  # 总欧姆热 [W]
        Qrxn = self.Qrxnneg + self.Qrxnpos  # 总反应热 [W]
        Qrev = self.Qrevneg + self.Qrevpos  # 总可逆热 [W]
        return Qohm + Qrxn + Qrev

    @property
    def Qohme(self):
        """电解液欧姆热 [W]"""
        return self.A*((self.κeff_*self.gradφe_ + self.κDeff_*self.gradlnce_)*self.gradφe_*self.Δx_).sum()

    @property
    def Qohmneg(self):
        """负极固相欧姆热 [W]"""
        return self.A*(self.σeffneg*self.gradφsneg_**2).sum()*self.Δxneg

    @property
    def Qohmpos(self):
        """正极固相欧姆热 [W]"""
        return self.A*(self.σeffpos*self.gradφspos_**2).sum()*self.Δxpos

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
    def UOCPnegsurf_(self):
        """负极表面电位场 [V]"""
        return self.solve_UOCPneg_(self.csnegsurf_/self.csmaxneg)

    @property
    def UOCPpossurf_(self):
        """正极表面电位场 [V]"""
        return self.solve_UOCPpos_(self.cspossurf_/self.csmaxpos)

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

    @property
    def θsneg(self):
        """负极嵌锂状态"""
        return self.Vr_.dot(self.csneg__).mean()/self.csmaxneg

    @property
    def θspos(self):
        """正极嵌锂状态"""
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
        dUOCPdθs_ = ( UOCP_[:-2]  * (θs_[1:-1] - θs_[2:]) / (θs_[:-2] - θs_[1:-1]) / (θs_[:-2] - θs_[2:])
                 + UOCP_[1:-1] * (1/(θs_[1:-1] - θs_[:-2]) + 1/(θs_[1:-1] - θs_[2:]))
                 + UOCP_[2:]   * (θs_[1:-1] - θs_[:-2]) / (θs_[2:] - θs_[:-2]) / (θs_[2:] - θs_[1:-1]) )  # 内部点开路电位导数
        dUOCPdθs0 = ( UOCP_[0] * (1/(θs_[0] - θs_[1]) + 1/(θs_[0] - θs_[2]))
                 + UOCP_[1] * (θs_[0] - θs_[2]) / (θs_[1] - θs_[0]) / (θs_[1] - θs_[2])
                 + UOCP_[2] * (θs_[0] - θs_[1]) / (θs_[2] - θs_[0]) / (θs_[2] - θs_[1]) )  # 左界点开路电位导数
        dUOCPdθsEnd = (UOCP_[-3] * (θs_[-1] - θs_[-2]) / (θs_[-3] - θs_[-2]) / (θs_[-3] - θs_[-1])
                   + UOCP_[-2] * (θs_[-1] - θs_[-3]) / (θs_[-2] - θs_[-3]) / (θs_[-2] - θs_[-1])
                   + UOCP_[-1] * (1/(θs_[-1] - θs_[-3]) + 1/(θs_[-1] - θs_[-2]))  )  # 右界点开路电位导数
        return Interpolate1D(θs_, hstack([dUOCPdθs0, dUOCPdθs_, dUOCPdθsEnd]))    # 插值函数
        # return interp1d(θ_, hstack([dUOCPdθs0, dUOCPdθs_, dUOCPdθsEnd]), bounds_error=False, fill_value='extrapolate')

    def record_data(self,):
        """记录数据"""
        for dataname in self.datanames_:
            value = getattr(self, dataname)
            if isscalar(value):
                pass
            else:
                value = value.copy()
            self.data[dataname].append(value)

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

    class Error(Exception):
        """P2D模型专属异常类"""
        def __init__(self, information: str, *args):
            super().__init__(information, *args)

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
        ax1.set_xlabel(r'Time $\it t$ [s]')
        ax1.set_ylim([2.7, 4.5])  # 电压范围
        ax1.set_yticks(arange(2.7, 4.5 + 1e-6, 0.3))  # 电压范围
        duration = t_[-1] - t_[0]
        ax1.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax1.grid(axis='y', linestyle='--', color=[.5,.5,.5])
        ax1.minorticks_on()

        ax2.plot(t_, I_/self.Qcell, 'r-')
        ax2.set_ylabel('Current C-rate [C]')
        ax2.set_xlabel(r'Time $\it t$ [s]')
        ax2.set_ylim([-6, 6])  # 电流范围
        ax2.set_yticks(range(-6, 6 + 1, 1))  # 电流范围
        ax2.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax2.grid(axis='y', linestyle='--', color=[.5, .5, .5])
        ax2.minorticks_on()
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
        ax1.set_xlabel(r'Time $\it t$ [s]')
        duration = t_[-1] - t_[0]
        ax1.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax1.grid(axis='y', linestyle='--', color=[.5, .5, .5])
        ax1.minorticks_on()

        ax2.plot(t_, Qgen_, 'k-')
        ax2.set_ylabel(r'Heat generation rate ${\it Q}_{gen}({\it t})$ [W]')
        ax2.set_xlabel(r'Time $\it t$ [s]')
        ax2.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax2.grid(axis='y', linestyle='--', color=[.5, .5, .5])
        ax2.minorticks_on()
        plt.show()

    def plot_SOC(self,
                 t_: Sequence | None = None,  # 时刻序列
                 ):
        """SOC-时间"""
        if t_ is None:
            t_ = self.data['t']
        SOC_ = self.interpolate('SOC', t_)    # 呈时间序列的全电池荷电状态
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
        csnegcent___ = self.interpolate(f'{cθ}sneg__', t_=t_, x_=self.xneg_, r_=self.rneg_[[0]])  # 呈时间序列的负极固相颗粒中心锂离子浓度
        csposcent___ = self.interpolate(f'{cθ}spos__', t_=t_, x_=self.xpos_, r_=self.rpos_[[0]])  # 呈时间序列的正极固相颗粒中心锂离子浓度
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
            y_ = *csnegcent__.ravel(), *([nan]*self.Nsep), *csposcent__.ravel()
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{{self.rSign[1:-1]}=0}}$ [{self.cUnit or '–'}]')
        self.plot_interfaces(ax1)
        ax1.legend(bbox_to_anchor=(1, 1))

        ax2.set_title('Lithium concentration at electrode particle surface', fontsize=12)
        for n, (csnegsurf_, cspossurf_, t) in enumerate(zip(csnegsurf__, cspossurf__, t_)):
            x_ = self.xPlot_
            y_ = *csnegsurf_, *([nan]*self.Nsep), *cspossurf_
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.rSign[1:-1]} = {"{\\it R}_{s,reg}" if self.rUnit else 1} }}$ [{self.cUnit or '–'}]')
        self.plot_interfaces(ax2)
        plt.show()

        ax3.set_title('Lithium-ion concentration in electrolyte', fontsize=12)
        for n, (ce_, t) in enumerate(zip(ce__, t_)):
            x_ = [0, *self.xPlot_, self.xInterfacesPlot_[-1]]
            y_ = [ce_[0], *ce_, ce_[-1]]
            ax3.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax3.set_ylabel(rf'{self.cSign}$_e$({self.xSign}, {self.tSign}) [{self.cUnit or '–'}]')
        self.plot_interfaces(ax3)
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
                y_ = 1e3*hstack([φsneg_[0] + I/self.A/self.σeffneg*0.5*self.Δxneg, φsneg_, φsneg_[-1]])
            else:
                y_ = 1e3*hstack([φsneg_[0] + I/self.σneg*0.5*self.Δxneg, φsneg_, φsneg_[-1]])
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(r'Electrode potential ${\it φ}_{s,neg}$' + f'({self.xSign}, {self.tSign}) [mV]')
        ax1.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
        ax1.set_xlim(x_[0], x_[-1])
        ax1.grid(axis='y', linestyle='--')

        x_ = [self.xInterfacesPlot_[Nneg+Nsep], *self.xPlot_[-Npos:], self.xInterfacesPlot_[-1]]
        for n, (φspos_, I, t) in enumerate(zip(φspos__, I_, t_)):
            if self.xUnit:
                y_ = 1e3*hstack([φspos_[0], φspos_, φspos_[-1] - I/self.A/self.σeffpos*0.5*self.Δxpos])
            else:
                y_ = 1e3*hstack([φspos_[0], φspos_, φspos_[-1] - I/self.σpos*0.5*self.Δxpos])
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(ax1.get_ylabel().replace('neg', 'pos'))
        ax2.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
        ax2.set_xlim(x_[0], x_[-1])
        ax2.grid(axis='y', linestyle='--')
        ax2.legend(bbox_to_anchor=[1, 1])

        for n, (φe_, t) in enumerate(zip(φe__, t_)):
            ax3.plot([0, *self.xPlot_, self.xInterfacesPlot_[-1]],
                     hstack([φe_[0], φe_, φe_[-1]]), 'o-', color=DFNP2D.get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(r'Electrolyte potential ${\it φ}_e$' + f'({self.xSign}, {self.tSign}) [mV]')

        self.plot_interfaces(ax3)
        plt.show()

    def plot_jint(self,
                  t_: Sequence | None = None,  # 时刻序列
                  ):
        """局部体积电流密度、过电位、交换电流密度-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        jJ, iI = ['j', 'i'] if self.xUnit else ['J', 'I']
        jintneg__ = self.interpolate(f'{jJ}intneg_', t_=t_, x_=self.xneg_)   # 呈时间序列的负极局部体积电流密度场
        jintpos__ = self.interpolate(f'{jJ}intpos_', t_=t_, x_=self.xpos_)   # 呈时间序列的正极局部体积电流密度场
        ηintneg__ = self.interpolate('ηintneg_', t_=t_, x_=self.xneg_)*1e3  # 呈时间序列的负极固相表面过电位场 [mV]
        ηintpos__ = self.interpolate('ηintpos_', t_=t_, x_=self.xpos_)*1e3  # 呈时间序列的正极固相表面过电位场 [mV]
        i0intneg__ = self.interpolate(f'{iI}0intneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极交换电流密度场
        i0intpos__ = self.interpolate(f'{iI}0intpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极交换电流密度场

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
            y_ = *jintneg_, *([nan]*self.Nsep), *jintpos_
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(rf'{self.jSign}$_{{int}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        self.plot_interfaces(ax1)
        ax1.legend(bbox_to_anchor=[1, 1])

        ax2.set_title('Field of lithium (de-)intercalation overpotential', fontsize=12)
        for n, (ηintneg_, ηintpos_, t) in enumerate(zip(ηintneg__, ηintpos__, t_)):
            x_ = self.xPlot_
            y_ = *ηintneg_, *([nan]*self.Nsep), *ηintpos_
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(rf'${{\it η}}_{{int}}$ ({self.xSign}, {self.tSign}) [mV]')
        self.plot_interfaces(ax2)

        ax3.set_title('Field of lithium (de-)intercalation exchange current density', fontsize=12)
        for n, (i0intneg_, i0intpos_, t) in enumerate(zip(i0intneg__, i0intpos__, t_)):
            x_ = self.xPlot_
            y_ = *i0intneg_, *([nan]*self.Nsep), *i0intpos_
            ax3.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(rf'{self.i0Sign}({self.xSign}, {self.tSign}) [{ self.i0Unit}]')
        self.plot_interfaces(ax3)
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
            y_ = *jDLneg_, *([nan]*self.Nsep), *jDLpos_
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'Double-layer local volumetric\ncurrent density {self.jSign}$_{{DL}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        self.plot_interfaces(ax1)
        ax1.legend(bbox_to_anchor=[1, 1])

        t_ = self.data['t']
        jDLneg__ = self.interpolate(f'{jJ}DLneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的双电层效应负极局部体积电流密度场
        jDLpos__ = self.interpolate(f'{jJ}DLpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的双电层效应正极局部体积电流密度场
        if jJ=='j':
            IDLneg_ = jDLneg__.sum(axis=1)*(self.Δxneg*self.A)
            IDLpos_ = jDLpos__.sum(axis=1)*(self.Δxpos*self.A)
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

    def plot_i(self,
               t_: Sequence | None = None,  # 时刻序列
               ):
        """固液相电流密度-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        isneg__ = self.interpolate('isneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极固相电流密度场 [V]
        ispos__ = self.interpolate('ispos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极固相电流密度场 [V]
        ie__ = self.interpolate('ie_', t_=t_, x_=self.x_)  # 呈时间序列的电解液电流密度场 [V]
        i_ = self.interpolate('I', t_=t_)/self.A       # 呈时间序列的总电流密度 [A/m^2]
        Nneg, Nsep, Npos = self.Nneg, self.Nsep, self.Npos

        fig = plt.figure(figsize=[10, 7])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.set_position([.1, .59, .75, 0.375])
        ax2.set_position([.1, .08, .75, 0.375])

        for n, (isneg_, ispos_, i, t) in enumerate(zip(isneg__, ispos__, i_, t_)):
            x_ = hstack([0, self.xPlot_[:Nneg],
                           self.xInterfacesPlot_[Nneg], self.xPlot_[Nneg:Nneg+Nsep], self.xInterfacesPlot_[Nneg+Nsep],
                           self.xPlot_[-Npos:], self.xInterfacesPlot_[-1]])
            y_ = hstack([i, isneg_, 0, zeros(self.Nsep), 0, ispos_, i])
            ax1.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax1.set_ylabel(rf'Solid-phase current density ${{\it i}}_{{s}}$({self.xSign}, {self.tSign}) [A/m$^2$]')
        self.plot_interfaces(ax1)
        ax1.legend(bbox_to_anchor=[1, 1])

        for n, (ie_, i, t) in enumerate(zip(ie__, i_, t_)):
            x_ = [0, *self.xPlot_, self.xInterfacesPlot_[-1]]
            y_ = [0, *ie_, 0]
            ax2.plot(x_, y_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax2.set_ylabel(rf'Liquid-phase current density ${{\it i}}_{{e}}$({self.xSign}, {self.tSign}) [A/m$^2$]')
        self.plot_interfaces(ax2)
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
            x = xR if LPmodel else (xR*self.Lneg)
        elif 2<=xR<=3:
            reg = 'pos'
            x = xR if LPmodel else ((xR - 2)*self.Lpos + self.Lneg + self.Lsep)
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
                    'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax.legend(bbox_to_anchor=[1, 1])
        ax.set_ylabel(rf'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.xSign[1:-1]}={x if LPmodel else x*1e6:g}\;{self.xUnit} }}$ [{self.cUnit or '–'}]')
        ax.set_xlabel(rf'Radial location {self.rSign} [{self.rUnit or '–'}]')
        ax.set_xlim(X_[0], X_[-1])
        ax.grid(axis='y', linestyle='--')
        plt.show()


    def plot_jLP(self,
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
            ax1.plot(self.xPlot_[:self.Nneg], jLP_, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'Lithium plating local volumetric\ncurrent density {self.jSign}$_{{LP}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (ηLPneg_, t) in enumerate(zip(ηLPneg__, t_)):
            ax2.plot(self.xPlot_[:self.Nneg], ηLPneg_*1e3, 'o-', color=DFNP2D.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(rf'Lithium plating overpotential ${{\it η}}_{{LP}}$({self.xSign}, {self.tSign}) [mV]')
        for ax in [ax1, ax2]:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit}]')
            ax.set_xlim(0, self.xInterfacesPlot_[self.Nneg])
            ax.grid(axis='y', linestyle='--')

        plt.show()

    def plot_ηLP(self,
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
            ratio_ = jLP__.sum(axis=1)*self.Δxneg*self.A/I_  # 析锂电流与总电流之比
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

    def plot_OCV(self):
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

    def plot_interfaces(self, ax):
        ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')  # 横坐标标签
        ax.set_ylim(ax.get_ylim())  # 固定纵坐标上下限
        ax.set_xlim(self.xInterfacesPlot_[[0, -1]])  # 横坐标上下限
        ax.vlines(self.xInterfacesPlot_[self.Nneg], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  ls='--', color=[.5, .5, .5],
                  alpha=0.5)  # 负极-隔膜界面
        ax.vlines(self.xInterfacesPlot_[self.Nneg + self.Nsep], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  ls='--', color=[.5, .5, .5],
                  alpha=0.5)  # 隔膜-正极界面
        ax.grid(axis='y', linestyle='--')  # 纵坐标网格线

    @staticmethod
    def get_color(s_: Sequence | int, n: int):
        """返回viridis颜色"""
        if isinstance(s_, Iterable):
            N = len(s_)
        elif isscalar(s_):
            N = int(s_)
        color_ = plt.get_cmap('viridis')(int(linspace(0, 255, N)[n]))[:3]  # (3,)
        return color_

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

    @staticmethod
    def solve_4θ(UOCPneg: Callable,
                 UOCPpos: Callable,
                 Qcell: float | int,
                 Qneg: float | int,
                 Qpos: float | int,
                 Umin: float  | int,
                 Umax: float | int, ):
        # 已知Qcell、Qneg、Qpos，求4个边界嵌锂状态
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
        Nr, Nneg, Nsep, Npos = self.Nr, self.Nneg, self.Nsep, self.Npos  # 读取：负极、隔膜、正极网格数
        assert csneg__.shape==(Nr, Nneg), f'负极固相颗粒内部锂离子浓度csneg__.shape应为({Nr}, {Nneg})'
        assert csneg__.shape==(Nr, Nneg), f'正极固相颗粒内部锂离子浓度cspos__.shape应为({Nr}, {Npos})'
        assert ce_.shape==(self.Ne,), f'电解液锂离子浓度ce_.shape应为({self.Ne},)'
        assert ((0<=csneg__) & (csneg__<=self.csmaxneg)).all(), 'csneg__取值范围应为(0, csmaxneg) [mol/m^3]'
        assert ((0<=cspos__) & (cspos__<=self.csmaxpos)).all(), 'cspos__取值范围应为(0, csmaxpos) [mol/m^3]'
        assert (0<ce_).all(), 'ce_取值应大于0 [mol/m^3]'
        # 更新K__矩阵的参数相关值
        if decouple_cs:=self.decouple_cs:
            pass
        else:
            self.update_K__idxcsnegsurf_idxjintneg_(self.aneg, self.Dsneg)
            self.update_K__idxcspossurf_idxjintpos_(self.apos, self.Dspos)
        self.update_K__idxφsneg_idxjneg_(σeffneg := self.σeffneg)
        self.update_K__idxφspos_idxjpos_(σeffpos := self.σeffpos)
        κeffWest_ = κeffEast_ = self.κeff_
        self.update_K__idxφe_idxφe_(κeffWest_, κeffEast_)
        self.update_K__idxηintneg_idxjneg_(RSEIneg := self.RSEIneg, aeffneg := self.aeffneg)
        self.update_K__idxηintpos_idxjpos_(RSEIpos := self.RSEIpos, aeffpos := self.aeffpos)
        RSEI2aeffneg = RSEIneg/aeffneg
        RSEI2aeffpos = RSEIpos/aeffpos

        # 原索引
        idxcsnegsurf_ = self.idxcsnegsurf_
        idxcspossurf_ = self.idxcspossurf_
        idxφsneg_ = self.idxφsneg_
        idxφspos_ = self.idxφspos_
        idxφe_ = self.idxφe_
        idxjintneg_ = self.idxjintneg_
        idxjintpos_ = self.idxjintpos_
        idxi0intneg_ = self.idxi0intneg_
        idxi0intpos_ = self.idxi0intpos_
        idxηintneg_ = self.idxηintneg_
        idxηintpos_ = self.idxηintpos_
        # 拼接一致性初始化待求解的变量
        idx_ = concatenate([
            idxcsnegsurf_, idxcspossurf_,
            idxφsneg_, idxφspos_, idxφe_,
            idxjintneg_, idxjintpos_,
            idxi0intneg_, idxi0intpos_,
            idxηintneg_, idxηintpos_,])
        Kinit__ = self.K__[ix_(idx_, idx_)]  # 提取K__矩阵
        Ninit = Kinit__.shape[0]                # Kinit__矩阵的行列数
        start = 0
        def assign(idxOld_) -> ndarray:
            """对矩阵Kinit__重新安排索引"""
            nonlocal start
            N = len(idxOld_)
            idxNew_ = arange(start, start + N)
            start += N
            return idxNew_
        idxcsnegsurf_ = assign(idxcsnegsurf_)
        idxcspossurf_ = assign(idxcspossurf_)
        idxφsneg_ = assign(idxφsneg_)
        idxφspos_ = assign(idxφspos_)
        idxφe_ = assign(idxφe_)
        idxjintneg_ = assign(idxjintneg_)
        idxjintpos_ = assign(idxjintpos_)
        idxi0intneg_ = assign(idxi0intneg_)
        idxi0intpos_ = assign(idxi0intpos_)
        idxηintneg_ = assign(idxηintneg_)
        idxηintpos_ = assign(idxηintpos_)

        # 读取方法
        solve_jint_ = DFNP2D.solve_jint_
        solve_djintdηint_ = DFNP2D.solve_djintdηint_
        solve_djintdi0int_ = DFNP2D.solve_djintdi0int_
        solve_i0int_ = DFNP2D.solve_i0int_
        solve_di0intdcssurf_ = DFNP2D.solve_di0intdcssurf_
        solve_UOCPneg_, solve_UOCPpos_ = self.solve_UOCPneg_, self.solve_UOCPpos_              # 读取：负极、正极开路电位函数 [V]
        solve_dUOCPdθsneg_, solve_dUOCPdθspos_ = self.solve_dUOCPdθsneg_, self.solve_dUOCPdθspos_  # 读取：负极、正极开路电位对嵌锂状态的偏导数函数 [V/–]

        Δxneg, Δxpos, ΔxWest_, ΔxEast_, Δx_ = self.Δxneg, self.Δxpos, self.ΔxWest_, self.ΔxEast_, self.Δx_  # 读取：网格尺寸
        F, T = DFNP2D.F, self.T
        F2RT = F/(2*DFNP2D.R*T)
        csmaxneg, csmaxpos = self.csmaxneg, self.csmaxpos
        κDeffWest_ = κDeffEast_ = self.κDeff_
        DeeffWest_ = DeeffEast_ = self.Deeff_
        ceneg_, cepos_ = ce_[:Nneg], ce_[-Npos:]

        if i0intnegUnknown := (self._i0intneg is None):
            kneg = self.kneg          # 读取：负极主反应速率常数
        else:
            i0intneg = self.i0intneg  # 读取：负极主反应交换电流密度 [A/m^2]
        if i0intposUnknown := (self._i0intpos is None):
            kpos = self.kpos          # 读取：正极主反应速率常数
        else:
            i0intpos = self.i0intpos  # 读取：正极主反应交换电流密度 [A/m^2]

        coeffs_ = self.coeffs_
        # 外推表面浓度
        csnegsurfExpl_ = coeffs_.dot(csneg__[-3:])
        cspossurfExpl_ = coeffs_.dot(cspos__[-3:])

        ## 对Kinit__的右端项bKinit_赋值 ##
        bKinit_ = zeros(Ninit)  # 右端项
        if decouple_cs:
            # 强制表面浓度约束：认为 csnegsurf_、cspossurf_ 是外推得到的已知值
            bKinit_[idxcsnegsurf_] = csnegsurfExpl_
            bKinit_[idxcspossurf_] = cspossurfExpl_
        else:
            # 用颗粒扩散边界条件：方程关联jint、cssurf及靠近颗粒表面的3个内部节点浓度
            K__ = self.K__
            csneg_ = csneg__.ravel('F')
            cspos_ = cspos__.ravel('F')
            bKinit_[idxcsnegsurf_] = -(
                  K__[self.idxcsnegsurf_, self.idxcsneg_[Nr-3::Nr]]*csneg_[Nr-3::Nr]
                + K__[self.idxcsnegsurf_, self.idxcsneg_[Nr-2::Nr]]*csneg_[Nr-2::Nr]
                + K__[self.idxcsnegsurf_, self.idxcsneg_[Nr-1::Nr]]*csneg_[Nr-1::Nr])
            bKinit_[idxcspossurf_] =  -(
                  K__[self.idxcspossurf_, self.idxcspos_[Nr-3::Nr]] * cspos_[Nr-3::Nr]
                + K__[self.idxcspossurf_, self.idxcspos_[Nr-2::Nr]] * cspos_[Nr-2::Nr]
                + K__[self.idxcspossurf_, self.idxcspos_[Nr-1::Nr]] * cspos_[Nr-1::Nr])
        # 固相电流边界条件
        i = I/self.A  # 电极电流密度 [A/m^2]
        bKinit_[idxφsneg_[0]]  = -Δxneg*i/σeffneg
        bKinit_[idxφspos_[-1]] =  Δxpos*i/σeffpos
        # 电解液电势方程的电解液锂离子浓度项
        bKinit_[idxφe_[0]] = κDeffEast_[0]*(ce_[1] - ce_[0])/ΔxEast_[0]/(0.5*(ce_[1] + ce_[0]))
        bKinit_[idxφe_[-1]] = -κDeffWest_[-1]*(ce_[-1] - ce_[-2])/ΔxWest_[-1]/(0.5*(ce_[-1] + ce_[-2]))
        bKinit_[idxφe_[1:-1]] = ( κDeffEast_[1:-1]*(ce_[2:]  - ce_[1:-1])/ΔxEast_[1:-1]/(0.5*(ce_[2:]   + ce_[1:-1]))
                                - κDeffWest_[1:-1]*(ce_[1:-1] - ce_[:-2])/ΔxWest_[1:-1]/(0.5*(ce_[1:-1] + ce_[:-2])))
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = DeeffWest_[nE]*Δx_[nW], DeeffEast_[nW]*Δx_[nE]
            cInterface = (a*ce_[nE] + b*ce_[nW])/(a + b)
            bKinit_[idxφe_[nW]] = (κDeffEast_[nW]*(cInterface - ce_[nW])/(0.5*Δx_[nW])/cInterface
                                 - κDeffWest_[nW]*(ce_[nW] - ce_[nW - 1])/ΔxWest_[nW]/(0.5*(ce_[nW] + ce_[nW - 1])))
            bKinit_[idxφe_[nE]] = (κDeffEast_[nE]*(ce_[nE + 1] - ce_[nE])/ΔxEast_[nE]/(0.5*(ce_[nE + 1] + ce_[nE]))
                                 - κDeffWest_[nE]*(ce_[nE] - cInterface)/(0.5*Δx_[nE])/cInterface)

        ## Newton迭代初值 ##
        X_ = zeros(Ninit)
        X_[idxcsnegsurf_] = csnegsurfExpl_
        X_[idxcspossurf_] = cspossurfExpl_
        X_[idxφe_] = 0
        jintneg =  i/self.Lneg  # 初值：主反应平均局部体积电流密度 [A/m^3]
        jintpos = -i/self.Lpos
        X_[idxjintneg_] = jintneg
        X_[idxjintpos_] = jintpos
        i0intneg_ = solve_i0int_(kneg, csmaxneg, csnegsurfExpl_, ce_[:Nneg])  if i0intnegUnknown else i0intneg
        i0intpos_ = solve_i0int_(kpos, csmaxpos, cspossurfExpl_, ce_[-Npos:]) if i0intposUnknown else i0intpos
        if i0intnegUnknown:
            X_[idxi0intneg_] = i0intneg_
        if i0intposUnknown:
            X_[idxi0intpos_] = i0intpos_
        X_[idxηintneg_] = arcsinh(jintneg/(2*aeffneg*i0intneg_))/F2RT
        X_[idxηintpos_] = arcsinh(jintpos/(2*aeffpos*i0intpos_))/F2RT
        X_[idxφsneg_] = X_[idxηintneg_] + RSEI2aeffneg*jintneg + solve_UOCPneg_(X_[idxcsnegsurf_]/csmaxneg)
        X_[idxφspos_] = X_[idxηintpos_] + RSEI2aeffpos*jintpos + solve_UOCPpos_(X_[idxcspossurf_]/csmaxpos)

        # Newton迭代
        J__ = Kinit__.copy()
        for nNewton in range(1, 101):
            F_ = Kinit__.dot(X_) - bKinit_  # (Ninit,) F残差向量

            csnegsurf_, cspossurf_ = X_[idxcsnegsurf_], X_[idxcspossurf_]
            i0intneg_ = X_[idxi0intneg_] if i0intnegUnknown else i0intneg
            i0intpos_ = X_[idxi0intpos_] if i0intposUnknown else i0intpos
            ηintneg_, ηintpos_ = X_[idxηintneg_], X_[idxηintpos_]

            # F向量非线性部分
            F_[idxjintneg_] -= solve_jint_(T, aeffneg, i0intneg_, ηintneg_)  # F向量jintneg部分
            F_[idxjintpos_] -= solve_jint_(T, aeffpos, i0intpos_, ηintpos_)  # F向量jintpos部分
            if i0intnegUnknown:
                F_[idxi0intneg_] -= solve_i0int_(kneg, csmaxneg, csnegsurf_, ceneg_)  # F向量i0intneg部分
            if i0intposUnknown:
                F_[idxi0intpos_] -= solve_i0int_(kpos, csmaxpos, cspossurf_, cepos_)  # F向量i0intpos部分
            F_[idxηintneg_] += solve_UOCPneg_(csnegsurf_/csmaxneg)  # F向量ηintneg非线性部分
            F_[idxηintpos_] += solve_UOCPpos_(cspossurf_/csmaxpos)  # F向量ηintpos非线性部分
            # 更新Jacobi矩阵
            J__[idxjintneg_, idxηintneg_] = -solve_djintdηint_(T, aeffneg, i0intneg_, ηintneg_)  # ∂Fjintneg/∂ηintneg
            J__[idxjintpos_, idxηintpos_] = -solve_djintdηint_(T, aeffpos, i0intpos_, ηintpos_)  # ∂Fjintpos/∂ηintpos
            if i0intnegUnknown:
                J__[idxjintneg_, idxi0intneg_] = -solve_djintdi0int_(T, aeffneg, ηintneg_)   # ∂Fjintneg/∂i0intneg
                J__[idxi0intneg_, idxcsnegsurf_] = -solve_di0intdcssurf_(kneg, csmaxneg, csnegsurf_, ceneg_, i0intneg_)  # ∂Fi0intneg/∂csnegsurf
            if i0intposUnknown:
                J__[idxjintpos_, idxi0intpos_] = -solve_djintdi0int_(T, aeffpos, ηintpos_)   # ∂Fjintpos/∂i0intpos
                J__[idxi0intpos_, idxcspossurf_] = -solve_di0intdcssurf_(kpos, csmaxpos, cspossurf_, cepos_, i0intpos_)  # ∂Fi0intpos/∂cspossurf
            J__[idxηintneg_, idxcsnegsurf_] = solve_dUOCPdθsneg_(csnegsurf_/csmaxneg) / csmaxneg  # ∂Fηintneg/∂csnegsurf
            J__[idxηintpos_, idxcspossurf_] = solve_dUOCPdθspos_(cspossurf_/csmaxpos) / csmaxpos  # ∂Fηintpos/∂cspossurf

            ΔX_ = solve(J__, F_)
            X_ -= ΔX_

            if abs(ΔX_).max()<1e-6:
                break
        else:
            raise DFNP2D.Error(f'一致性初始化失败，Newton迭代{nNewton = }次，不收敛')

        # 初始化状态
        self.I = I
        self.csneg__[:] = csneg__
        self.cspos__[:] = cspos__
        self.csnegsurf_[:] = X_[idxcsnegsurf_]
        self.cspossurf_[:] = X_[idxcspossurf_]
        self.ce_[:] = ce_
        self.φsneg_[:] = φsneg_ = X_[idxφsneg_]
        self.φspos_[:] = φspos_ = X_[idxφspos_]
        self.φe_[:] = X_[idxφe_]
        self.jintneg_[:] = jintneg_ = X_[idxjintneg_]
        self.jintpos_[:] = jintpos_ = X_[idxjintpos_]
        self.jDLneg_[:] = 0.
        self.jDLpos_[:] = 0.
        self.i0intneg_[:] = X_[idxi0intneg_] if i0intnegUnknown else full(Nneg, i0intneg)
        self.i0intpos_[:] = X_[idxi0intpos_] if i0intposUnknown else full(Npos, i0intpos)
        self.ηintneg_[:] = X_[idxηintneg_]
        self.ηintpos_[:] = X_[idxηintpos_]

        if self.lithiumPlating:
            self.jLP_[:] = 0.
            self.i0LP_[:] = self.i0LP if self._i0LP else DFNP2D.solve_i0LP_(self.kLP, self.ceneg_)

        self.jneg_[:] = jintneg_
        self.jpos_[:] = jintpos_
        self.ηLPneg_[:] = φsneg_ - self.φeneg_ - RSEI2aeffneg * jintneg_
        self.ηLPpos_[:] = φspos_ - self.φepos_ - RSEI2aeffpos * jintpos_

        if self.verbose:
            print(f'一致性初始化完成。Newton迭代{nNewton = }。Consistent initial conditions are solved! ')

if __name__=='__main__':
    cell = DFNP2D(
        Δt=10, SOC0=0.1,
        Nneg=9, Nsep=8, Npos=7, Nr=6,
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
        # decouple_cs=False,
        )

    I = cell.Qcell
    cell.count_lithium()
    thermalModel = True
    cell.CC(-I, 2300, thermalModel).CC(I, 2000, thermalModel).CC(0, 3700, thermalModel)
    cell.count_lithium()

    plt.close('all')

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
    cell.plot_jLP(arange(1700, 2301, 100))
    cell.plot_ηLP()
    cell.plot_OCV()
    cell.plot_dUOCPdθs()
    '''
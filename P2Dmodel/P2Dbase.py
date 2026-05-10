#%%
import time, pathlib
from math import exp
from typing import Sequence, Callable
from collections.abc import Iterable
from collections import namedtuple
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.linalg.lapack import dgbsv
from scipy.optimize import root
from numpy import ndarray, nan, \
    array, arange, zeros, eye, full, linspace, stack, hstack, concatenate, meshgrid, \
    cbrt, ix_, asfortranarray, isnan, isscalar, savez

from P2Dmodel.OCP import NMC111, Graphite
from P2Dmodel.tools import Interpolate1D, set_matplotlib, DiagonalSliceRavel, F, R


class P2Dbase(ABC):
    """抽象类：锂离子电池准二维模型 Pseudo-two-Dimensional model"""

    ## 常数 ##
    F = F  # 法拉第Faraday常数 [C/mol]
    R = R  # 理想气体常数 [J/(mol·K)]

    __slots__ = (
        # P2D模型通用参数名
        'θminneg', 'θmaxneg', 'θminpos', 'θmaxpos',
        'UOCPneg', 'UOCPpos', 'solve_dUOCPdθsneg_', 'solve_dUOCPdθspos_',
        'dUOCPdTneg', 'dUOCPdTpos',
        '_kneg', '_kpos', '_kLP', '_Dsneg', '_Dspos',
        'σneg', 'σpos', 'RSEIneg', 'RSEIpos',
        'CDLneg', 'CDLpos', 'l',
        '_i0intneg', '_i0intpos', '_i0LP',
        # 网格参数
        'Δt', 'Nneg', 'Nsep', 'Npos', 'Ne', 'Nr',
        'x_', 'Δx_', 'ΔxWest_', 'ΔxEast_', 'Δxneg', 'Δxpos',
        'rneg_', 'rpos_', 'Δrneg_', 'Δrpos_', 'Vr_',
        # 热参数
        'EDsneg', 'EDspos', 'Ekneg', 'Ekpos', 'Eκ', 'EDe', 'EkLP',
        'Tref', 'hA', 'Cth', 'Tamb',
        # 模式
        'lithiumPlating', 'doubleLayerEffect',
        'timeDiscretization', 'radialDiscretization',
        'decouple_cs', 'constants', 'complete', 'verbose',
        # 状态量
        'T', 'I', 't', 'QLP',
        'φsneg_', 'φspos_', 'φe_',
        'ηintneg_', 'ηintpos_', 'ηLPneg_', 'ηLPpos_',
        'nNewton',
        'bK_', 'ravelK_',
        'data',
        # 恒定量
        'datanames_', 'banded_experience_of_J__',
        'bandKcsneg__', 'bandKcspos__',
        'e__', 'coeffsExpl_', 'sK',
        'coeffs_csneg_', 'coeffs_cspos_',
        'coeff_csnegsurf_csnegsurf',
        'coeff_cspossurf_cspossurf',
        # 作图恒定量
        'tSign', 'tUnit',
        'xSign', 'xUnit', 'rSign', 'rUnit',
        'cSign', 'cUnit', 'jSign', 'jUnit', 'i0Sign', 'i0Unit',
        )


    # 类型注解 ##
    Qcell: float; Qneg: float; Qpos: float  # 理论可用容量、负极、正极容量 [Ah]
    _kneg: float; _kpos: float    # 负极、正极主反应速率常数
    _kLP: float                   # 负极析锂反应速率常数
    _Dsneg: float; _Dspos: float  # 负极、正极固相扩散系数
    _i0intneg: float | None       # 负极交换电流密度
    _i0intpos: float | None       # 正极交换电流密度

    def __init__(self,
            Lneg: float = 1.,
            Lsep: float = 1.,
            Lpos: float = 1.,   # 负极、隔膜、正极厚度 [m]/[–]
            Rsneg: float = 1.,
            Rspos: float = 1.,  # 负极、正极固相颗粒半径 [m]/[–]
            T0: float = 298.15,         # 初始温度 [K]
            SOC0: float = 0.2,          # 初始荷电状态 [–]
            θminneg: float = .0370744,  # SOC=0%的负极嵌锂状态 [–]
            θmaxneg: float = .8775600,  # SOC=100%的负极嵌锂状态 [–]
            θminpos: float = .0746557,  # SOC=100%的正极嵌锂状态 [–]
            θmaxpos: float = .9589741,  # SOC=0%的正极嵌锂状态 [–]
            Δt: float | int = 10.,      # 时间步长 [s]
            Nneg: int = 10,             # 负极区域网格数
            Nsep: int = 10,             # 隔膜区域网格数
            Npos: int = 10,             # 正极区域网格数
            Nr: int = 10,               # 球形固相颗粒径向网格数
            UOCPneg: Callable = Graphite().Graphite_COMSOL,  # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位UOCPneg_ [V]
            UOCPpos: Callable = NMC111().NMC111_COMSOL,      # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位UOCPpos_ [V]
            dUOCPdθsneg: Callable | None = None,  # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位对嵌锂状态的导数 [V/–]
            dUOCPdθspos: Callable | None = None,  # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位对嵌锂状态的导数 [V/–]
            dUOCPdTneg: Callable | float = 0.,    # 函数：输入负极嵌锂状态θsneg_ [–]，输出负极开路电位的熵热系数 [V/K]
            dUOCPdTpos: Callable | float = 0.,    # 函数：输入正极嵌锂状态θspos_ [–]，输出正极开路电位的熵热系数 [V/K]
            EDsneg: float = 30e3,  # 负极锂离子扩散系数的比活化能 [J/mol]
            EDspos: float = 30e3,  # 正极锂离子扩散系数的比活化能 [J/mol]
            Ekneg: float = 68e3,   # 负极主反应速率常数的比活化能 [J/mol]
            Ekpos: float = 50e3,   # 正极主反应速率常数的比活化能 [J/mol]
            Eκ: float = 4e3,       # 电解液锂离子电导率的比活化能 [J/mol]
            EDe: float = 16.5e3,   # 电解液锂离子扩散系数的比活化能 [J/mol]
            EkLP: float = 3.53e4,  # 负极析锂反应速率常数的比活化能 [J/mol]
            Tref: float = 298.15,  # 参考温度 [K]
            hA: float = 1.5,       # 表面传热系数与传热面积之积 [W/K]
            Cth: float = 600.,     # 热容 [J/K]
            Tamb: float = 298.15,  # 环境温度 [K]
            lithiumPlating: bool = False,      # 是否考虑析锂反应
            doubleLayerEffect: bool = True,    # 是否考虑电极颗粒双电层效应
            timeDiscretization: str = 'CN',    # 时间离散格式 'CN'/'backward'
            radialDiscretization: str = 'EV',  # 球形颗粒径向离散方法 等体积'EV'/等间隔'EI'
            decouple_cs: bool = True,  # 是否解耦固相锂离子浓度的求解，设置decouple_cs==True可加速，几乎无代价
            constants: bool = False,   # 是否使用常量参数缓存，设置constants==True可加速，但应在考虑参数为恒定值的情况下
            complete: bool = True,     # 是否确保功能完备，设置complete==False可加速，但省略不必要的计算和存储
            verbose: bool = True,      # 是否显示初始化、运行进度
            ):
        # 通用参数
        assert T0>0, f'初始温度{T0 = }，应大于0 [K]'
        assert 0<=SOC0<=1, f'初始荷电状态{SOC0 = }，取值范围应为[0, 1]'
        # 4边界嵌锂状态参数
        assert 0<θminneg<θmaxneg<1, f'负极最小、最大嵌锂状态{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1'
        assert 0<θminpos<θmaxpos<1, f'正极最小、最大嵌锂状态{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1'
        self.θminneg = θminneg  # SOC=0%的负极嵌锂状态
        self.θmaxneg = θmaxneg  # SOC=100%的负极嵌锂状态
        self.θminpos = θminpos  # SOC=100%的正极嵌锂状态
        self.θmaxpos = θmaxpos  # SOC=0%的正极嵌锂状态
        # 网格参数
        self.Δt = Δt; assert Δt>0, f'时间步长{Δt = }，应大于0 [s]'
        self.Nneg = Nneg; assert isinstance(Nneg, int) and Nneg>=3, f'负极区域网格数{Nneg = }，应为不小于3的正整数'
        self.Nsep = Nsep; assert isinstance(Nsep, int) and Nsep>=3, f'隔膜区域网格数{Nsep = }，应为不小于3的正整数'
        self.Npos = Npos; assert isinstance(Npos, int) and Npos>=3, f'正极区域网格数{Npos = }，应为不小于3的正整数'
        self.Nr = Nr;     assert isinstance(Nr, int) and Nr>=3, f'球形固相颗粒半径方向网格数{Nr = }，应为不小于3的正整数'
        self.Ne = Ne = Nneg + Nsep + Npos  # 电解液网格总数
        # 函数
        self.UOCPneg = UOCPneg; assert callable(UOCPneg), '函数UOCPneg，输入负极嵌锂状态θsneg_ [–]，输出正极开路电位UOCPneg_ [V]'
        self.UOCPpos = UOCPpos; assert callable(UOCPpos), '函数UOCPpos，输入正极嵌锂状态θspos_ [–]，输出负极开路电位UOCPpos_ [V]'
        assert callable(dUOCPdθsneg) or (dUOCPdθsneg is None), '负极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入负极嵌锂状态θsneg_ [–]）'
        self.solve_dUOCPdθsneg_ = P2Dbase.generate_solve_dUOCPdθs_(UOCPneg) if (dUOCPdθsneg is None) else dUOCPdθsneg
        assert callable(dUOCPdθspos) or (dUOCPdθspos is None), '正极开路电位对嵌锂状态的导数 [V/–]，None或函数（输入正极嵌锂状态θspos_ [–]）'
        self.solve_dUOCPdθspos_ = P2Dbase.generate_solve_dUOCPdθs_(UOCPpos) if (dUOCPdθspos is None) else dUOCPdθspos
        self.dUOCPdTneg = dUOCPdTneg; assert callable(dUOCPdTneg) or isinstance(dUOCPdTneg, (int, float)), '负极开路电位的熵热系数dUOCPdTneg [V/K]，标量或函数（输入负极嵌锂状态θsneg_ [–]）'
        self.dUOCPdTpos = dUOCPdTpos; assert callable(dUOCPdTpos) or isinstance(dUOCPdTpos, (int, float)), '正极开路电位的熵热系数dUOCPdTpos [V/K]，标量或函数（输入正极嵌锂状态θspos_ [–]）'
        # 比活化能
        self.EDsneg = EDsneg; assert EDsneg>=0, f'负极锂离子扩散系数的比活化能{EDsneg = }，应大于或等于0 [J/mol]'
        self.EDspos = EDspos; assert EDspos>=0, f'正极锂离子扩散系数的比活化能{EDspos = }，应大于或等于0 [J/mol]'
        self.Ekneg = Ekneg; assert Ekneg>=0,    f'负极主反应速率常数的比活化能{Ekneg = }，应大于或等于0 [J/mol]'
        self.Ekpos = Ekpos; assert Ekpos>=0,    f'正极主反应速率常数的比活化能{Ekpos = }，应大于或等于0 [J/mol]'
        self.Eκ = Eκ;     assert Eκ>=0,   f'电解液锂离子电导率的比活化能{Eκ = }，应大于或等于0 [J/mol]'
        self.EDe = EDe;   assert EDe>=0,  f'电解液锂离子扩散系数的比活化能{EDe = }，应大于或等于0 [J/mol]'
        self.EkLP = EkLP; assert EkLP>=0, f'负极析锂反应速率常数的比活化能{EkLP = }，应大于或等于0 [J/mol]'
        self.Tref = Tref; assert Tref>0, f'参考温度{Tref = }，应大于0 [K]'
        # 热参数
        self.hA = hA; assert hA>0, f'表面传热系数与传热面积之积{hA = }，应大于0 [W/K]'
        self.Cth = Cth; assert Cth>0, f'电池热容{Cth = }，应大于0 [J/K]'
        self.Tamb = Tamb; assert Tamb>0, f'环境温度{Tamb = }，应大于0 [K]'
        # 模式
        self.lithiumPlating = lithiumPlating        # 是否考虑析锂反应
        self.doubleLayerEffect = doubleLayerEffect  # 是否考虑双电层效应
        self.timeDiscretization = timeDiscretization;     assert timeDiscretization in {'backward', 'CN'}, f'时间离散格式{timeDiscretization = }，应为 "CN"（Crank-Nicolson） 或 "backward"（后向差分格式）'
        self.radialDiscretization = radialDiscretization; assert radialDiscretization in {'EV', 'EI'}, f'球形颗粒径向离散方法{radialDiscretization = }，应为 "EV"（等体积）/"EI"（等间隔）'
        self.decouple_cs = decouple_cs  # 是否解耦固相锂离子浓度的求解
        self.constants = constants      # 是否使用常量参数缓存
        self.complete = complete        # 是否确保功能完备
        self.verbose = verbose          # 是否显示初始化、运行进度
        # P2D通用状态量
        self.T = T0    # 温度
        self.I = 0.    # 电流 [A]
        self.t = 0.    # 时刻 [s]
        self.QLP = 0.  # 累计析锂量 [Ah]
        θsneg = θminneg + SOC0*(θmaxneg - θminneg)  # 初始负极嵌锂状态
        θspos = θmaxpos + SOC0*(θminpos - θmaxpos)  # 初始正极嵌锂状态
        self.φsneg_ = full(Nneg, UOCPneg := self.solve_UOCPneg_(θsneg))  # 初始化：负极固相电势场 [V]
        self.φspos_ = full(Npos, UOCPpos := self.solve_UOCPpos_(θspos))  # 初始化：正极固相电势场 [V]
        self.φe_ = zeros(Ne)                # 初始化：电解液电势场 [V]
        self.ηintneg_ = zeros(Nneg)
        self.ηintpos_ = zeros(Npos)         # 初始化：负极、正极主反应过电位场 [V]
        self.ηLPneg_ = full(Nneg, UOCPneg)
        self.ηLPpos_ = full(Npos, UOCPpos)  # 初始化：负极、正极析锂反应过电位场 [V]
        self.nNewton = 0
        # 恒定量
        self.banded_experience_of_J__ = None  # Jacobi矩阵带状化经验
        self.Δxneg = Δxneg = Lneg/Nneg  # 负极网格厚度
        Δxsep      = Lsep/Nsep          # 隔膜网格厚度
        self.Δxpos = Δxpos = Lpos/Npos  # 正极网格厚度
        self.x_ = concatenate([
            linspace(0, Lneg, Nneg + 1)[:-1] + Δxneg*0.5,
            linspace(Lneg, Lneg + Lsep, Nsep + 1)[:-1] + Δxsep*0.5,
            linspace(Lneg + Lsep, Lneg + Lsep + Lpos, Npos + 1)[:-1] + Δxpos*0.5,])   # (Ne,) 全区域各控制体中心坐标
        self.Δx_ = concatenate([full(Nneg, Δxneg), full(Nsep, Δxsep), full(Npos, Δxpos)])  # (Ne,) 全区域控制体厚度
        self.ΔxWest_ = hstack([
            full(Nneg, Δxneg),
            (Δxneg + Δxsep)*0.5, full(Nsep - 1, Δxsep),
            (Δxsep + Δxpos)*0.5, full(Npos - 1, Δxpos)])  # (Ne,) 当前控制体中心到左侧控制体中心的距离
        self.ΔxEast_ = hstack([
            full(Nneg - 1, Δxneg), (Δxneg + Δxsep)*0.5,
            full(Nsep - 1, Δxsep), (Δxsep + Δxpos)*0.5,
            full(Npos, Δxpos)])  # (Ne,) 当前控制体中心到右侧控制体中心的距离

        (self.rneg_,        # (Nr,) 负极球壳控制体中心坐标 [m]/[-]
         self.Δrneg_,       # (Nr,) 负极球壳控制体厚度 [m]/[-]
         self.Vr_,          # 从中心到表面球壳体积分数序列 [-]
         self.bandKcsneg__, # (3, Nr) 负极固相浓度三对角矩阵的带 [m^-2]/[–]
         ) = P2Dbase.generate_r_Δr_bandKcs__(Nr, Rsneg, radialDiscretization)
        if Rsneg==Rspos:
            self.rpos_ = self.rneg_
            self.Δrpos_ = self.Δrneg_
            self.bandKcspos__ = self.bandKcsneg__
        else:
            (self.rpos_,
             self.Δrpos_, _,
             self.bandKcspos__,
             ) = P2Dbase.generate_r_Δr_bandKcs__(Nr, Rspos, radialDiscretization)
        rneg_ = self.rneg_
        r_3, r_2, r_1 = rneg_[-3:]
        Δr_3, Δr_2, Δr_1 = Δrneg_3_1_ = Rsneg - rneg_[-3:]
        self.coeffsExpl_ = coeffsExpl_ = array([
            Δr_1*Δr_2/((r_3 - r_1)*(r_3 - r_2)),
            Δr_1*Δr_3/((r_2 - r_1)*(r_2 - r_3)),
            Δr_2*Δr_3/((r_1 - r_2)*(r_1 - r_3))
            ])  # (3,) 用于由3个颗粒内部节点浓度外推表面浓度的无量纲系数 [–]，计算结果不依赖正负极，仅取决于Nr和radialDiscretization

        if decouple_cs:
            self.e__ = zeros((Nr, 1))  # (Nr, 1) 非零末元为1的向量
            self.e__[-1] = 1
        else:
            self.coeffs_csneg_ = coeffs_csneg_ = coeffsExpl_/-Δrneg_3_1_  # (3,) [1/m]/[–]
            self.coeff_csnegsurf_csnegsurf = coeff_csnegsurf_csnegsurf = (1/Δrneg_3_1_).sum()
            self.coeffs_cspos_ = coeffs_csneg_
            if Rsneg==Rspos:
                self.coeffs_cspos_ = coeffs_csneg_
                self.coeff_cspossurf_cspossurf = coeff_csnegsurf_csnegsurf
            else:
                Δrpos_3_1_ = Rspos - self.rpos_[-3:]
                self.coeffs_cspos_ = coeffsExpl_/-Δrpos_3_1_  # (3,) [1/m]/[–]
                self.coeff_cspossurf_cspossurf = (1/Δrpos_3_1_).sum()

        # 需保存状态量的名称
        self.datanames_ = [
            't', 'I', 'U',  # 时刻 [s] 电流 [A] 端电压 [V]
            'ηLPneg_', 'ηLPpos_', ]  # 负极、正极析锂反应过电位场 [V]
        if complete:
            self.datanames_.extend([
                'φsneg_', 'φspos_', 'φe_',  # 负极、正极固相电势场 [V]、电解液电势场 [V]
                'ηintneg_', 'ηintpos_',  # 负极、正极主反应过电位场 [V]
                'θsneg', 'θspos', 'SOC',  # 负极、正极嵌锂状态、全电池荷电状态 [–]
                'T', 'Qgen',  # 温度 [K]、产热量 [W]
                'nNewton', ])  # Newton迭代次数
            set_matplotlib()  # matplotlib作图设置
            # matplotlib作图变量单位
            self.tSign, self.tUnit = r'${\it t}$', 's'            # 时间t符号、单位
            self.xSign, self.xUnit = r'${\it x}$', 'μm'           # 电极厚度方向坐标x符号、单位
            self.rSign, self.rUnit = r'${\it r}$', 'μm'           # 径向坐标r符号、单位
            self.cSign, self.cUnit = r'${\it c}$', 'mol/m$^3$'    # 锂离子浓度c符号、单位
            self.jSign, self.jUnit = r'${\it j}$', 'A/m$^3$'      # 局部体积电流密度j符号、单位
            self.i0Sign, self.i0Unit = r'${\it i}_0$', 'A/m$^2$'  # 交换电流密度i0符号、单位
        self.data = {name: [] for name in self.datanames_}  # 字典：存储呈时间序列的运行数据
        # 因变量线性矩阵K__
        self.ravelK_ = None  # (NK*NK,) 因变量线性矩阵K__展平视图
        self.bK_ = None      # (NK,) 常数项向量 K__ @ X_ = bK_
        self.sK = None       # 切片索引集

    def _generate_K__bK_and_slices(self):
        """生成时域因变量矩阵K__、常数项向量bK_及切片索引，并对K__赋常系数、几何网格相关参数"""
        # 读取网格数
        Nneg, Npos, Ne, Nr = self.Nneg, self.Npos, self.Ne, self.Nr
        # 读取模式
        lithiumPlating = self.lithiumPlating
        doubleLayerEffect = self.doubleLayerEffect
        decouple_cs = self.decouple_cs

        NK = 0  # 全局索引游标
        def allocate(n: int) -> slice:
            # 分配切片
            nonlocal NK
            s = slice(NK, NK + n)
            NK += n
            return s
        slices_ = {
            's_csneg': (s_csneg := allocate(0 if decouple_cs else Nr*Nneg)),  # 索引：负极固相内部浓度 先排颗粒径向r，再排厚度方向x
            's_cspos': (s_cspos := allocate(0 if decouple_cs else Nr*Npos)),  # 索引：正极固相内部浓度
            's_csnegsurf': (s_csnegsurf := allocate(Nneg)),  # 索引：正极固相表面浓度
            's_cspossurf': (s_cspossurf := allocate(Npos)),  # 索引：正极固相表面浓度
            's_ce':     (s_ce :=  allocate(Ne)),             # 索引：电解液浓度
            's_φsneg':  (s_φsneg := allocate(Nneg)),         # 索引：负极固相电势
            's_φspos':  (s_φspos := allocate(Npos)),         # 索引：正极固相电势
            's_φe':     (s_φe := allocate(Ne)),              # 索引：电解液电势
            's_jintneg': (s_jintneg := allocate(Nneg)),      # 索引：负极主反应局部体积电流密度
            's_jintpos': (s_jintpos := allocate(Npos)),      # 索引：正极主反应局部体积电流密度
            's_jDLneg':  (s_jDLneg := allocate(Nneg if doubleLayerEffect else 0)),  # 索引：负极双电层局部体积电流密度
            's_jDLpos':  (s_jDLpos := allocate(Npos if doubleLayerEffect else 0)),  # 索引：正极双电层局部体积电流密度
            's_i0intneg': (s_i0intneg := allocate(0 if self._i0intneg else Nneg)),  # 索引：负极主反应交换电流密度
            's_i0intpos': (s_i0intpos := allocate(0 if self._i0intpos else Npos)),  # 索引：正极主反应交换电流密度
            's_ηintneg':  (s_ηintneg := allocate(Nneg)),  # 索引：负极主反应过电位
            's_ηintpos':  (s_ηintpos := allocate(Npos)),  # 索引：正极主反应过电位
            }
        if lithiumPlating:
            slices_.update({
                's_jLP': (s_jLP := allocate(Nneg)),  # 索引：负极析锂反应局部体积电流密度
                's_ηLP': (s_ηLP := allocate(Nneg)),  # 索引：负极析锂反应过电位
                })
        slices_.update({
            's_ceneg': (s_ceneg := slice(start := s_ce.start, start + Nneg)),  # 索引：负极电解液浓度
            's_cepos': (s_cepos := slice((stop := s_ce.stop) - Npos, stop)),   # 索引：正极电解液浓度
            's_φeneg': (s_φeneg := slice(start := s_φe.start, start + Nneg)),  # 索引：负极电解液电势
            's_φepos': (s_φepos := slice((stop := s_φe.stop) - Npos, stop)),   # 索引：正极电解液电势
            's_c': slice(0, s_ce.stop),                    # 索引：所有浓度量
            's_φ': slice(s_φsneg.start, s_φe.stop),        # 索引：所有电势量
            's_j': slice(s_jintneg.start, s_jDLpos.stop),  # 索引：主要局部体积电流密度量
            })

        # K__矩阵子块对角元对应K__.ravel()的索引
        dsr = DiagonalSliceRavel(NK)
        if decouple_cs:
            pass
        else:
            step = NK*Nr + 1
            startneg = (s_csneg.start + Nr - 1) * NK + s_jintneg.start
            stopneg = startneg + Nneg*step
            startpos = (s_cspos.start + Nr - 1) * NK + s_jintpos.start
            stoppos = startpos + Npos*step
            slices_.update({
                'sr_csneg_csneg'   : dsr(s_csneg, s_csneg),
                'sr_csneg_csneg_l' : dsr(s_csneg, s_csneg, -1),
                'sr_csneg_csneg_u' : dsr(s_csneg, s_csneg,  1),
                'sr_cspos_cspos'   : dsr(s_cspos, s_cspos),
                'sr_cspos_cspos_l' : dsr(s_cspos, s_cspos, -1),
                'sr_cspos_cspos_u' : dsr(s_cspos, s_cspos,  1),
                'sr_csnegEnd_jintneg' : slice(startneg, stopneg, step),
                'sr_csposEnd_jintpos' : slice(startpos, stoppos, step),
                })
        slices_.update({
            'sr_csnegsurf_jintneg' : dsr(s_csnegsurf, s_jintneg),
            'sr_cspossurf_jintpos' : dsr(s_cspossurf, s_jintpos),
            'sr_ce_ce'   : dsr(s_ce, s_ce),
            'sr_ce_ce_l' : dsr(s_ce, s_ce, -1),
            'sr_ce_ce_u' : dsr(s_ce, s_ce,  1),
            'sr_ceneg_jintneg' : dsr(s_ceneg, s_jintneg),
            'sr_ceneg_jDLneg'  : dsr(s_ceneg, s_jDLneg),
            'sr_cepos_jintpos' : dsr(s_cepos, s_jintpos),
            'sr_cepos_jDLpos'  : dsr(s_cepos, s_jDLpos),
            'sr_φsneg_jintneg' : dsr(s_φsneg, s_jintneg),
            'sr_φsneg_jDLneg'  : dsr(s_φsneg, s_jDLneg),
            'sr_φspos_jintpos' : dsr(s_φspos, s_jintpos),
            'sr_φspos_jDLpos'  : dsr(s_φspos, s_jDLpos),
            'sr_φe_ce'   : dsr(s_φe, s_ce),
            'sr_φe_ce_l' : dsr(s_φe, s_ce, -1),
            'sr_φe_ce_u' : dsr(s_φe, s_ce,  1),
            'sr_φe_φe'   : dsr(s_φe, s_φe),
            'sr_φe_φe_l' : dsr(s_φe, s_φe, -1),
            'sr_φe_φe_u' : dsr(s_φe, s_φe,  1),
            'sr_jintneg_i0intneg' : dsr(s_jintneg, s_i0intneg),
            'sr_jintneg_ηintneg'  : dsr(s_jintneg, s_ηintneg),
            'sr_jintpos_i0intpos' : dsr(s_jintpos, s_i0intpos),
            'sr_jintpos_ηintpos'  : dsr(s_jintpos, s_ηintpos),
            'sr_jDLneg_φsneg'   : dsr(s_jDLneg, s_φsneg),
            'sr_jDLneg_φeneg'   : dsr(s_jDLneg, s_φeneg),
            'sr_jDLneg_jintneg' : dsr(s_jDLneg, s_jintneg),
            'sr_jDLneg_jDLneg'  : dsr(s_jDLneg, s_jDLneg),
            'sr_jDLpos_φspos'   : dsr(s_jDLpos, s_φspos),
            'sr_jDLpos_φepos'   : dsr(s_jDLpos, s_φepos),
            'sr_jDLpos_jintpos' : dsr(s_jDLpos, s_jintpos),
            'sr_jDLpos_jDLpos'  : dsr(s_jDLpos, s_jDLpos),
            'sr_i0intneg_csnegsurf' : dsr(s_i0intneg, s_csnegsurf),
            'sr_i0intneg_ceneg'     : dsr(s_i0intneg, s_ceneg),
            'sr_i0intpos_cspossurf' : dsr(s_i0intpos, s_cspossurf),
            'sr_i0intpos_cepos'     : dsr(s_i0intpos, s_cepos),
            'sr_ηintneg_csnegsurf' : dsr(s_ηintneg, s_csnegsurf),
            'sr_ηintneg_jintneg'   : dsr(s_ηintneg, s_jintneg),
            'sr_ηintneg_jDLneg'    : dsr(s_ηintneg, s_jDLneg),
            'sr_ηintpos_cspossurf' : dsr(s_ηintpos, s_cspossurf),
            'sr_ηintpos_jintpos'   : dsr(s_ηintpos, s_jintpos),
            'sr_ηintpos_jDLpos'    : dsr(s_ηintpos, s_jDLpos),
            })
        if lithiumPlating:
            slices_.update({
                'sr_ceneg_jLP'   : dsr(s_ceneg, s_jLP),
                'sr_φsneg_jLP'   : dsr(s_φsneg, s_jLP),
                'sr_jDLneg_jLP'  : dsr(s_jDLneg, s_jLP),
                'sr_ηintneg_jLP' : dsr(s_ηintneg, s_jLP),
                'sr_jLP_ceneg'   : dsr(s_jLP, s_ceneg),
                'sr_jLP_ηLP'     : dsr(s_jLP, s_ηLP),
                'sr_ηLP_jintneg' : dsr(s_ηLP, s_jintneg),
                'sr_ηLP_jDLneg'  : dsr(s_ηLP, s_jDLneg),
                'sr_ηLP_jLP'     : dsr(s_ηLP, s_jLP),
                })
        self.sK = namedtuple('SlicesK', slices_.keys())(**slices_)
        del slices_
        self.ravelK_ = ravelK_ = eye(NK).ravel()  # (NK*NK,) 因变量线性矩阵K__展平视图
        self.bK_ = zeros(NK)                      # (NK,) 常数项向量 K__ @ X_ = bK_

        # 对K__矩阵赋恒定值（几何参数、常系数）

        # 负极、正极固相表面浓度cssurf行（此处为几何参数Rsneg、Rspos相关，据其物理意义，Rsneg、Rspos在电池运行过程中不应变化）
        if decouple_cs:
            pass
        else:
            step = NK + Nr
            start0neg = s_csnegsurf.start * NK + s_csneg.start + Nr
            start0pos = s_cspossurf.start * NK + s_cspos.start + Nr
            stepNneg = step*Nneg
            stepNpos = step*Npos
            c_ = self.coeffs_csneg_
            ravelK_[slice(start := start0neg - 3, start + stepNneg, step)] = c_[-3]
            ravelK_[slice(start := start0neg - 2, start + stepNneg, step)] = c_[-2]
            ravelK_[slice(start := start0neg - 1, start + stepNneg, step)] = c_[-1]
            ravelK_[dsr(s_csnegsurf, s_csnegsurf)] = self.coeff_csnegsurf_csnegsurf
            c_ = self.coeffs_cspos_
            ravelK_[slice(start := start0pos - 3, start + stepNpos, step)] = c_[-3]
            ravelK_[slice(start := start0pos - 2, start + stepNpos, step)] = c_[-2]
            ravelK_[slice(start := start0pos - 1, start + stepNpos, step)] = c_[-1]
            ravelK_[dsr(s_cspossurf, s_cspossurf)] = self.coeff_cspossurf_cspossurf

        # 负极、正极固相电势φs行φs列
        for s_φs, Nreg in zip((s_φsneg, s_φspos), (Nneg, Npos)):
            ravelK_[dsr(s_φs, s_φs)] = [-1] + [-2]*(Nreg - 2) + [-1]  # 主对角线
            ravelK_[dsr(s_φs, s_φs, -1)] = \
            ravelK_[dsr(s_φs, s_φs,  1)] = 1  # 上下对角线

        # 电解液电势φe行j列
        ravelK_[dsr(s_φeneg, s_jintneg)] = Δxneg = self.Δxneg  # jintneg列
        ravelK_[dsr(s_φepos, s_jintpos)] = Δxpos = self.Δxpos  # jintpos列
        if doubleLayerEffect:
            ravelK_[dsr(s_φeneg, s_jDLneg)] = Δxneg  # jDLneg列
            ravelK_[dsr(s_φepos, s_jDLpos)] = Δxpos  # jDLpos列

        # 负极、正极主反应过电位ηint行
        ravelK_[dsr(s_ηintneg, s_φeneg)] = \
        ravelK_[dsr(s_ηintpos, s_φepos)] = 1   # φe列
        ravelK_[dsr(s_ηintneg, s_φsneg)] = \
        ravelK_[dsr(s_ηintpos, s_φspos)] = -1  # φs列

        # 析锂补充
        if lithiumPlating:
            ravelK_[dsr(s_φeneg, s_jLP)] = Δxneg  # φeneg行jLP列
            ravelK_[dsr(s_ηLP, s_φsneg)] = -1     # ηLP行φsneg列
            ravelK_[dsr(s_ηLP, s_φeneg)] = 1      # ηLP行φeneg列

        if self.verbose:
            print(f'时域因变量线性矩阵 {ravelK_.base.shape = }')

    def _update_K__bK_ce_ce_j(self,
            DeeffWest_: ndarray, DeeffEast_: ndarray,
            εe_: ndarray,
            Kcej: float,
            Δt: float,
            old_ce_,
            old_jneg_,
            old_jpos_,
            ):
        # 更新K__矩阵ce行ce列
        # 更新K__矩阵ce行j列
        # 更新bK_向量ce行
        Nneg, Nsep = self.Nneg, self.Nsep
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        timeDiscretization = self.timeDiscretization
        ravelK_ = self.ravelK_
        sK = self.sK
        s_ce = sK.s_ce
        dl_ = ravelK_[sK.sr_ce_ce_l]  # (Ne-1,) ce列下对角线
        du_ = ravelK_[sK.sr_ce_ce_u]  # (Ne-1,) ce列上对角线
        d_  = ravelK_[sr_ce_ce := sK.sr_ce_ce]  # (Ne,) ce列主对角线
        dl_[:] = -DeeffWest_[1:]  / ΔxWest_[1:]
        du_[:] = -DeeffEast_[:-1] / ΔxEast_[:-1]
        d_[:]  = -(hstack([0, dl_]) + hstack([du_, 0]))
        NK1 = self.bK_.size + 1
        start0 = sr_ce_ce.start
        for nW, nE in ((Nneg - 1, Nneg), (Nneg + Nsep - 1, Nneg + Nsep)):
            # 修正负极-隔膜界面、隔膜-正极界面
            nrW = start0 + nW*NK1
            nrE = start0 + nE*NK1
            a, c = -DeeffWest_[nW]/ΔxWest_[nW], -2*DeeffEast_[nW]*DeeffWest_[nE]/(DeeffEast_[nW]*Δx_[nE] + DeeffWest_[nE]*Δx_[nW])
            ravelK_[nrW-1:nrW+2] = a, -(a + c), c  # 界面左侧控制体
            a, c = c, -DeeffEast_[nE]/ΔxEast_[nE]
            ravelK_[nrE-1:nrE+2] = a, -(a + c), c  # 界面右侧控制体
        Δt2Δx_ = Δt/Δx_
        dl_ *= Δt2Δx_[1:]   # ce列下对角线
        du_ *= Δt2Δx_[:-1]  # ce列上对角线
        d_  *= Δt2Δx_       # ce列主对角线
        if timeDiscretization=='CN':
            dl_ *= .5
            du_ *= .5
            d_  *= .5
            Bce__ = -ravelK_.base[s_ce, s_ce]  # (Ne, Ne)
            Bce__.ravel()[::self.Ne+1] += εe_  # 对角元+εe_
        d_ += εe_  # 对角元+εe_

        # 更新K__矩阵ce行j列
        if timeDiscretization=='CN':
            Kcej *= .5
        ravelK_[sK.sr_ceneg_jintneg] = Kcej  # jintneg列
        ravelK_[sK.sr_cepos_jintpos] = Kcej  # jintneg列、jintpos列
        if self.doubleLayerEffect:
            ravelK_[sK.sr_ceneg_jDLneg] = Kcej  # jDLneg列
            ravelK_[sK.sr_cepos_jDLpos] = Kcej  # jDLpos列
        if self.lithiumPlating:
            ravelK_[sK.sr_ceneg_jLP] = Kcej  # jLP列

        bK_ = self.bK_
        match timeDiscretization:
            case 'CN':
                bK_[s_ce] = Bce__.dot(old_ce_)
                bK_[sK.s_ceneg] -= Kcej * old_jneg_
                bK_[sK.s_cepos] -= Kcej * old_jpos_
            case 'backward':
                bK_[s_ce] = εe_ * old_ce_

    def _update_K__φsneg_jneg(self, σeffneg):
        # 更新K__矩阵φsneg行jneg列
        ravelK_ = self.ravelK_
        sK = self.sK
        Δxneg = self.Δxneg
        a = -Δxneg*Δxneg/σeffneg
        ravelK_[sK.sr_φsneg_jintneg] = a     # jintneg列
        if self.doubleLayerEffect:
            ravelK_[sK.sr_φsneg_jDLneg] = a  # jDLneg列
        if self.lithiumPlating:
            ravelK_[sK.sr_φsneg_jLP] = a     # jLP列

    def _update_K__φspos_jpos(self, σeffpos):
        # 更新K__矩阵φspos行jpos列
        ravelK_ = self.ravelK_
        sK = self.sK
        Δxpos = self.Δxpos
        a = -Δxpos*Δxpos/σeffpos
        ravelK_[sK.sr_φspos_jintpos] = a     # jintpos列
        if self.doubleLayerEffect:
            ravelK_[sK.sr_φspos_jDLpos] = a  # jDLpos列

    def _update_K__φe_φe(self, κeffWest_, κeffEast_):
        # 更新K__矩阵φe行φe列
        ΔxEast_, ΔxWest_, Δx_ = self.ΔxEast_, self.ΔxWest_, self.Δx_
        Nneg, Nsep = self.Nneg, self.Nsep
        ravelK_ = self.ravelK_
        sK = self.sK
        dl_ = ravelK_[sK.sr_φe_φe_l]
        du_ = ravelK_[sK.sr_φe_φe_u]
        d_  = ravelK_[sr_φe_φe := sK.sr_φe_φe]
        dl_[:] = κeffWest_[1:] /ΔxWest_[1:]              # (Ne-1,) φe列下对角线
        du_[:] = κeffEast_[:-1]/ΔxEast_[:-1]             # (Ne-1,) φe列上对角线
        d_[:]  = -(hstack([0, dl_]) + hstack([du_, 0]))  # (Ne,) φe列主对角线
        d_[0] -= κeffWest_[0]/(0.5*Δx_[0])               # 首元占优
        NK1 = self.bK_.size + 1
        start0 = sr_φe_φe.start
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            nrW = start0 + nW*NK1
            nrE = start0 + nE*NK1
            a, c = κeffWest_[nW]/ΔxWest_[nW], 2*κeffEast_[nW]*κeffWest_[nE]/(κeffEast_[nW]*Δx_[nE] + κeffWest_[nE]*Δx_[nW])
            ravelK_[nrW-1:nrW+2] = a, -(a + c), c  # 界面左侧控制体
            a, c = c, κeffEast_[nE]/ΔxEast_[nE]
            ravelK_[nrE-1:nrE+2] = a, -(a + c), c  # 界面右侧控制体

    def _update_K__bK_jDL_φs_φe_j(self,
                                  aeffneg, aeffpos,
                                  RSEIneg, RSEIpos,
                                  CDLneg, CDLpos,
                                  Δt):
        # 更新K__矩阵jDL行φs、φe、j列
        # 更新bK_向量jDL行
        CDLneg  = aeffneg*CDLneg
        CDLpos  = aeffpos*CDLpos
        RSEIneg = RSEIneg/aeffneg
        RSEIpos = RSEIpos/aeffpos
        ravelK_ = self.ravelK_
        bK_ = self.bK_
        sK = self.sK
        s_jDLneg = sK.s_jDLneg
        s_jDLpos = sK.s_jDLpos
        data = self.data
        data_t_ = data['t']
        Nt = len(data_t_)  # 存储数据时刻数
        t_1 = data_t_[-1]  # 上一时刻 [s]
        t_2 = data_t_[-2] if Nt>1 else None  # 上上一时刻
        t_3 = data_t_[-3] if Nt>2 else None  # 上上上一时刻
        t = t_1 + Δt  # 当前时刻 [s]
        c = 1/Δt
        if Nt>1: c += 1/(t - t_2)
        if Nt>2: c += 1/(t - t_3)
        CDLneg2Δt = CDLneg*c
        CDLpos2Δt = CDLpos*c
        # 负极双电层局部体积电流密度jDLneg行
        ravelK_[sK.sr_jDLneg_φsneg] = -CDLneg2Δt  # φsneg列
        ravelK_[sK.sr_jDLneg_φeneg] =  CDLneg2Δt  # φeneg列
        ravelK_[sK.sr_jDLneg_jintneg] = a = CDLneg2Δt*RSEIneg  # jintneg列
        ravelK_[sK.sr_jDLneg_jDLneg]  = 1 + a     # jDLneg列
        if self.lithiumPlating:
            ravelK_[sK.sr_jDLneg_jLP] = a  # jLP列
        # 正极双电层局部体积电流密度jDLpos行
        ravelK_[sK.sr_jDLpos_φspos] = -CDLpos2Δt  # φspos列
        ravelK_[sK.sr_jDLpos_φepos] = CDLpos2Δt   # φepos列
        ravelK_[sK.sr_jDLpos_jintpos] = a = CDLpos2Δt*RSEIpos  # jintpos列
        ravelK_[sK.sr_jDLpos_jDLpos] = 1 + a      # jDLpos列

        Δφseneg_1_ = (ηLPneg__ := data['ηLPneg_'])[-1]  # 上一时刻负极固液相电势场之差
        Δφsepos_1_ = (ηLPpos__ := data['ηLPpos_'])[-1]  # 上一时刻负极、正极固液相电势场之差
        Δφseneg_2_ = ηLPneg__[-2] if Nt>1 else None    # 上上时刻
        Δφsepos_2_ = ηLPpos__[-2] if Nt>1 else None     # 上上时刻
        Δφseneg_3_ = ηLPneg__[-3] if Nt>2 else None    # 上上上时刻
        Δφsepos_3_ = ηLPpos__[-3] if Nt>2 else None     # 上上上时刻
        if Nt>2:
            A = (t - t_2)*(t - t_3)/-Δt/(t_1 - t_2)/(t_1 - t_3)
            B = Δt*(t - t_3)/(t_2 - t)/(t_2 - t_1)/(t_2 - t_3)
            C = Δt*(t - t_2)/(t_3 - t)/(t_3 - t_1)/(t_3 - t_2)
            bK_[s_jDLneg] = CDLneg * (A*Δφseneg_1_ + B*Δφseneg_2_ + C*Δφseneg_3_)
            bK_[s_jDLpos] = CDLpos * (A*Δφsepos_1_ + B*Δφsepos_2_ + C*Δφsepos_3_)
        elif Nt==2:
            A = (t - t_2)/(-Δt*(t_1 - t_2))
            B = Δt/((t_2 - t)*(t_2 - t_1))
            bK_[s_jDLneg] = CDLneg * (A*Δφseneg_1_ + B*Δφseneg_2_)
            bK_[s_jDLpos] = CDLpos * (A*Δφsepos_1_ + B*Δφsepos_2_)
        else:
            bK_[s_jDLneg] = -CDLneg2Δt * Δφseneg_1_
            bK_[s_jDLpos] = -CDLpos2Δt * Δφsepos_1_

    def _update_K__ηintneg_jneg(self, RSEIneg, aeffneg):
        # 更新K__矩阵ηintneg行jneg列
        ravelK_ = self.ravelK_
        sK = self.sK
        a = RSEIneg/aeffneg
        ravelK_[sK.sr_ηintneg_jintneg] = a     # jintneg列
        if self.doubleLayerEffect:
            ravelK_[sK.sr_ηintneg_jDLneg] = a  # jDLneg列
        if self.lithiumPlating:
            ravelK_[sK.sr_ηintneg_jLP] = a     # jLP列

    def _update_K__ηintpos_jpos(self, RSEIpos, aeffpos):
        # 更新K__矩阵ηintpos行jpos列
        ravelK_ = self.ravelK_
        sK = self.sK
        a = RSEIpos/aeffpos
        ravelK_[sK.sr_ηintpos_jintpos] = a     # jintpos列
        if self.doubleLayerEffect:
            ravelK_[sK.sr_ηintpos_jDLpos] = a  # jDLpos列

    def _update_K__ηLP_jneg(self, RSEIneg, aeffneg):
        # 更新K__矩阵ηLP行jneg列
        ravelK_ = self.ravelK_
        sK = self.sK
        a = RSEIneg/aeffneg
        ravelK_[sK.sr_ηLP_jintneg] = \
        ravelK_[sK.sr_ηLP_jLP] = a  # jintneg、jLP列
        if self.doubleLayerEffect:
            ravelK_[sK.sr_ηLP_jDLneg] = a  # jDLneg列

    def CC(self,
           I: float | int = 0,           # 电流 [A]，放电为正，充电为负
           duration: float | int = 1,    # 充放电运行时间 [s]
           thermalModel: bool = False,   # 是否计算电池产热
           Umax: float | None = None,    # 最高电压 [V]
           Umin: float | None = None,    # 最低电压 [V]
           SOCmax: float | None = None,  # 最大SOC
           SOCmin: float | None = None,  # 最小SOC
           minΔt: float = 0.1,  # 最小时间步长 [s]
           ):
        """恒流充放电"""
        assert minΔt<=self.Δt, f'最小时间步长 {minΔt = }s，应小于或等于Δt = {self.Δt}s'
        if verbose := self.verbose:
            startTime = time.time()  # 开始时间戳 [s]
            info = f"电流{I = :.2f}A{'放电' if I>0 else ('充电' if I<0 else '静置')}"

        # 读取模式
        lithiumPlating = self.lithiumPlating
        constants = self.constants

        # 读取方法
        record_data = self.record_data
        _stepping = self._stepping

        t_ = self.data['t']  # 读取：已计算时刻序列
        if self.t==0:
            record_data()  # 记录初始时刻数据
        tStart = t_[-1]    # 开始时刻 [s]
        tStop = tStart + duration  # 终止时刻 [s]
        self.I = I                 # 电流 [A]

        # 生成矩阵K__矩阵、bK_向量、初始化纯电化学参数相关值
        if self.ravelK_ is None:
            self._generate_K__bK_and_slices()
            self.update_K__with_pure_electrochemical_parameters()

        while True:
            ## 持续时间步进...

            # 检查终止条件
            if self.t>=tStop:
                if verbose:
                    print(f'\n达到运行时长{duration}s，停止{info}')
                break
            if I<0 and (Umax is not None) and self.U>=Umax:
                if verbose:
                    print(f'\n电压U达到{Umax = }V，停止{info}')
                break
            if I<0 and (SOCmax is not None) and self.SOC>=SOCmax:
                if verbose:
                    print(f'\nSOC达到{SOCmax = }，停止{info}')
                break
            if I>0 and (Umin is not None) and self.U<=Umin:
                if verbose:
                    print(f'\n电压U达到{Umin = }V，停止{info}')
                break
            if I>0 and (SOCmin is not None) and self.SOC<=SOCmin:
                if verbose:
                    print(f'\nSOC达到{SOCmin = }，停止{info}')
                break

            # 选择时间步长
            remainingTime = tStop - self.t  # 剩余时长 [s]
            if remainingTime<(self.Δt + minΔt):
                Δt = remainingTime  # 使用剩余时长作为最后时间步长
            else:
                Δt = self.Δt  # 使用默认时间步长 [s]

            # 更新K__矩阵纯电化学参数相关值
            if constants:
                pass
            else:
                self.update_K__with_pure_electrochemical_parameters()

            while True:
                # 试探步进
                try:
                    self.nNewton = _stepping(Δt)
                    break  # 步进成功，无报错，跳出
                except P2Dbase.Error as message:
                    if Δt==minΔt:
                        raise
                    ΔtNew = max(minΔt, Δt*0.5)
                    if verbose:
                        print(f'异常：{message}，缩小时间步长{Δt}s -> {ΔtNew}s', )
                    Δt = ΔtNew
                    continue

            self.t = t_[-1] + Δt  # 更新：时刻 [s]

            if lithiumPlating:
                self.QLP += -self.ILP*Δt/3600  # 更新：累计析锂量 [Ah]

            if thermalModel:
                dTdt = (self.Qgen + self.hA*(self.Tamb - self.T))/self.Cth
                self.T += dTdt*Δt  # 更新：温度 [K]

            record_data()  # 记录运行数据

            if verbose:
                # 显示进度
                finishedProportion = (self.t - tStart)/duration  # 已完成的比例
                finishedProgresses = int(25*finishedProportion)  # 已完成的进度条长度
                unfinishedProgresses = 25 - finishedProgresses   # 未完成的进度条长度
                finishedBar = '▓'*finishedProgresses        # 已完成的进度条
                unfinishedBar = '-'*unfinishedProgresses    # 未完成的进度条
                percentage = finishedProportion*100         # 已完成进度的百分比
                timeStamp = time.time() - startTime         # 累计耗时
                U = self.U
                SOC = self.SOC
                print(f'|{finishedBar}{unfinishedBar}|已完成{percentage:.0f}%，耗时{timeStamp:.1f}s，'
                      f't={self.t:g}s-->{tStop:g}s，{info}，'
                      f'电压{U = :.3f}V, {SOC = :.3f}，温度{self.T - 273.15:.1f}°C，'
                      f'析锂过电位{self.ηLPneg_[-1]*1000:.0f}mV，'
                      f'{self.nNewton}次Newton迭代'
                      f'\r', end='')
        return self

    def record_data(self):
        """记录数据"""
        data = self.data
        for name in self.datanames_:
            value = getattr(self, name)
            if isscalar(value):
                pass
            else:
                value = value.copy()
            data[name].append(value)

    @property
    def kneg(self):
        """负极主反应速率常数"""
        return self.Arrhenius(self._kneg, self.Ekneg)
    @kneg.setter
    def kneg(self, kneg):
        self._kneg = kneg

    @property
    def kpos(self):
        """正极主反应速率常数"""
        return self.Arrhenius(self._kpos, self.Ekpos,)
    @kpos.setter
    def kpos(self, kpos):
        self._kpos = kpos

    @property
    def kLP(self):
        """负极析锂反应速率常数"""
        return self.Arrhenius(self._kLP, self.EkLP)
    @kLP.setter
    def kLP(self, kLP):
        self._kLP = kLP

    @property
    def Dsneg(self):
        """负极固相扩散系数"""
        return self.Arrhenius(self._Dsneg, self.EDsneg)
    @Dsneg.setter
    def Dsneg(self, Dsneg):
        self._Dsneg = Dsneg

    @property
    def Dspos(self):
        """正极固相扩散系数"""
        return self.Arrhenius(self._Dspos, self.EDspos)
    @Dspos.setter
    def Dspos(self, Dspos):
        self._Dspos = Dspos

    @property
    def xneg_(self):
        """负极区域控制体中心坐标"""
        return self.x_[:self.Nneg]

    @property
    def xpos_(self):
        """正极区域控制体中心坐标"""
        return self.x_[-self.Npos:]

    @property
    def xInterfaces_(self):
        """(Ne+1,) 各控制体界面的坐标（包括负极-集流体界面、正极集流体界面）"""
        return hstack([0, self.Δx_.cumsum()])

    @property
    def OCV(self):
        """开路电压 [V]"""
        OCPpos = self.solve_UOCPpos_(self.θspos)
        OCPneg = self.solve_UOCPneg_(self.θsneg)
        return OCPpos - OCPneg

    @property
    def SOC(self):
        return (self.θsneg - self.θminneg)/(self.θmaxneg - self.θminneg)

    @property
    def φeneg_(self):
        """(Nneg,) 负极区域电解液电势 [V]"""
        return self.φe_[:self.Nneg]

    @property
    def φepos_(self):
        """(Npos,) 正极区域电解液电势 [V]"""
        return self.φe_[-self.Npos:]

    Qohme: float
    Qohmneg: float; Qohmpos: float
    Qrxnneg: float; Qrxnpos: float
    Qrevneg: float; Qrevpos: float
    @property
    def Qgen(self):
        """总产热量 [W]"""
        return (  self.Qohme + self.Qohmneg + self.Qohmpos  # 总欧姆热 [W]
                + self.Qrxnneg + self.Qrxnpos   # 总反应热 [W]
                + self.Qrevneg + self.Qrevpos)  # 总可逆热 [W]

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

    @staticmethod
    def generate_r_Δr_bandKcs__(Nr, Rs, radialDiscretization='EV'):
        """生成颗粒径向r相关坐标、(Nr, Nr)矩阵Kcs__的带bandKcs__"""
        Rs3 = 1 if Rs==1 else (Rs*Rs*Rs)
        if radialDiscretization=='EV':
            # 等体积划分球壳网格
            a = 4.1887902047863905  # 4π/3
            V = a*Rs3  # 颗粒体积 [m^3]/[–]
            ΔV = V/Nr  # 球壳控制体体积 [m^3]/[–]
            rO_ = cbrt(ΔV*arange(1, Nr + 1)/a)  # (Nr,) 球壳外界面坐标 [m]/[–]
            rI_ = hstack([0, rO_[:-1]])         # (Nr,) 球壳内界面坐标 [m]/[–]
            r_ = (rI_ + rO_)*0.5  # (Nr,) 球壳控制体中心坐标 [m]/[–]
            Δr_ = rO_ - rI_       # (Nr,) 球壳网格厚度序列 [m]/[–]
            Vr_ = full(Nr, 1/Nr)  # (Nr,) 球壳体积分数序列 [–]
        elif radialDiscretization=='EI':
            # 等间隔划分球壳网格
            Δr_ = full(Nr, Δr := Rs/Nr) # (Nr,) 球壳网格厚度序列 [m]/[–]
            hΔr = Δr*0.5
            r_ = arange(Nr)*Δr + hΔr    # (Nr,) 球壳控制体中心坐标 [m]/[–]
            rO_ = r_ + hΔr
            rI_ = r_ - hΔr
            Vr_ = (rO_*rO_*rO_ - rI_*rI_*rI_)/Rs3  # (Nr,) 球壳体积分数序列 [–]

        Nr1 = Nr + 1
        dr_ = r_[1:] - r_[:-1]  # (Nr-1,)
        rO2_ = rO_*rO_          # (Nr,)
        rI2_ = rI_*rI_          # (Nr,)
        Kcs__ = zeros((Nr, Nr))
        Kcs__ravel_ = Kcs__.ravel()
        Kcs__ravel_[Nr::Nr1] = a_ = -rI2_[1:] /dr_  # (Nr-1,) 下对角线
        Kcs__ravel_[1::Nr1]  = c_ = -rO2_[:-1]/dr_  # (Nr-1,) 上对角线
        Kcs__ravel_[::Nr1] = -(hstack([0., a_]) + hstack([c_, 0.]))  # (Nr, ) 主对角线
        Kcs__ /= ((rO2_*rO_ - rI2_*rI_)/3).reshape(-1, 1)
        bandKcs__ = zeros((3, Nr))
        bandKcs__[0, 1:]  = Kcs__ravel_[1::Nr1]   # (Nr-1,) 上对角线
        bandKcs__[1, :]   = Kcs__ravel_[::Nr1]    # (Nr,) 主对角线
        bandKcs__[2, :-1] = Kcs__ravel_[Nr::Nr1]  # (Nr-1,) 下对角线

        return r_, Δr_, Vr_, bandKcs__

    @staticmethod
    def solve_banded_matrix(
            A__: ndarray,                  # (N, N) 矩阵
            b_: ndarray,                   # (N,) A__ @ X_ = b_
            idxReorder_: Sequence[int],    # (N,) 索引：重排A__，使带状化
            idxRecover_: Sequence[int],    # (N,) 索引：恢复排序
            l: int,  # 下带宽
            u: int,  # 上带宽
            ):
        """解带状化矩阵"""
        N = A__.shape[0]  # 因变量总数
        A__ = A__[ix_(idxReorder_, idxReorder_)]  # 重新排列矩阵A__，使之带状化
        ab__ = zeros((2*l + u + 1, N),
                     dtype=A__.dtype, order='F')  # 适合dgbsv
        band__ = ab__[l:, :]   # (u + l + 1, N) 矩阵A__的带
        diag = A__.diagonal
        for row, offset in enumerate(range(u, -l - 1, -1)):
            d_ = diag(offset)  # 提取矩阵Areorder__的带
            start = max(0, offset)
            end   = min(N, N + offset)
            band__[row, start:end] = d_
        X_ = dgbsv(l, u, ab__, asfortranarray(b_[idxReorder_]), True, True)[2]
        # X_ = solve_banded((l, u), band__, asfortranarray(b_[idxReorder_]), True, True, False)
        return X_[idxRecover_]

    @staticmethod
    def banded_experience(A__: ndarray) -> dict[str, ndarray | int]:
        """带状化经验"""
        idxReorder_ = reverse_cuthill_mckee(csr_matrix(A__))  # 索引：重排A__，使之带状化
        idxRecover_ = idxReorder_.argsort()                   # 索引：恢复A__，使之还原
        A__ = A__[ix_(idxReorder_, idxReorder_)]  # 带状化矩阵
        ## 辨识带状矩阵的上、下带宽 ##
        N = A__.shape[0]
        diag = A__.diagonal
        for offset in range(-N + 1, 0):
            # 从矩阵左下角开始遍历对角线
            if any(diag(offset)!=0):
                l = abs(offset)  # 下带宽
                break
        for offset in range(N - 1, 0, -1):
            # 从矩阵右上角开始遍历对角线
            if any(diag(offset)!=0):
                u = offset       # 上带宽
                break
        return {'idxReorder_': idxReorder_, 'idxRecover_': idxRecover_, 'l': l, 'u': u}

    def Arrhenius(self,
                  X: float | None,  # 参考温度下的参数值
                  E: float,          # 活化能 [J/mol]
                  ):
        """Arrhenius温度修正"""
        if self.constants or self.T==self.Tref or X is None or E==0:
            return X
        return X * exp(E/P2Dbase.R*(1/self.Tref - 1/self.T))

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
        dUOCPdθs_ = (
              UOCP_[:-2]  * (θs_[1:-1] - θs_[2:]) / (θs_[:-2] - θs_[1:-1]) / (θs_[:-2] - θs_[2:])
            + UOCP_[1:-1] * (1/(θs_[1:-1] - θs_[:-2]) + 1/(θs_[1:-1] - θs_[2:]))
            + UOCP_[2:]   * (θs_[1:-1] - θs_[:-2]) / (θs_[2:] - θs_[:-2]) / (θs_[2:] - θs_[1:-1]) )  # 内部点开路电位导数
        dUOCPdθs0 = (
              UOCP_[0] * (1/(θs_[0] - θs_[1]) + 1/(θs_[0] - θs_[2]))
            + UOCP_[1] * (θs_[0] - θs_[2]) / (θs_[1] - θs_[0]) / (θs_[1] - θs_[2])
            + UOCP_[2] * (θs_[0] - θs_[1]) / (θs_[2] - θs_[0]) / (θs_[2] - θs_[1]) )  # 左界点开路电位导数
        dUOCPdθsEnd = (
              UOCP_[-3] * (θs_[-1] - θs_[-2]) / (θs_[-3] - θs_[-2]) / (θs_[-3] - θs_[-1])
            + UOCP_[-2] * (θs_[-1] - θs_[-3]) / (θs_[-2] - θs_[-3]) / (θs_[-2] - θs_[-1])
            + UOCP_[-1] * (1/(θs_[-1] - θs_[-3]) + 1/(θs_[-1] - θs_[-2]))  )    # 右界点开路电位导数
        return Interpolate1D(θs_, hstack([dUOCPdθs0, dUOCPdθs_, dUOCPdθsEnd]))  # 插值函数
        # return interp1d(θs_, hstack([dUOCPdθs0, dUOCPdθs_, dUOCPdθsEnd]), bounds_error=False, fill_value='extrapolate')

    @staticmethod
    def get_color(s_: Sequence | int, n: int, cmap='viridis'):
        """返回viridis颜色"""
        if isinstance(s_, Iterable):
            N = len(s_)
        elif isscalar(s_):
            N = int(s_)
        color_ = plt.get_cmap(cmap)(int(linspace(0, 255, N)[n]))[:3]  # (3,)
        return color_

    @staticmethod
    def solve_4θ(UOCPneg: Callable,
                 UOCPpos: Callable,
                 Qcell: float | int,
                 Qneg: float | int,
                 Qpos: float | int,
                 Umin: float  | int,
                 Umax: float | int, ):
        """已知Qcell、Qneg、Qpos，求4个边界嵌锂状态θminneg、θmaxneg、θminpos、θmaxpos"""
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

    class Error(Exception):
        """P2D模型专属异常类"""
        def __init__(self, information: str, *args):
            super().__init__(information, *args)

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
        ax1.set_ylim([2.4, 4.5])
        ax1.set_yticks(arange(2.4, 4.5 + 1e-6, 0.3))

        ax2.plot(t_, I_, 'r-')
        ax2.set_ylabel('Current ${\it I}({\it t})$ [A]')

        duration = t_[-1] - t_[0]
        xlim_ = [t_[0] - duration*0.02, t_[-1] + duration*0.02]
        for ax in (ax1, ax2):
            ax.set_xlabel(r'Time $\it t$ [s]')
            ax.set_xlim(xlim_)
            ax.grid(axis='y', linestyle='--', color=[.5, .5, .5])
            ax.minorticks_on()
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

        ax2.plot(t_, Qgen_, 'k-')
        ax2.set_ylabel(r'Heat generation rate ${\it Q}_{gen}({\it t})$ [W]')

        duration = t_[-1] - t_[0]
        xlim_ = [t_[0] - duration*0.02, t_[-1] + duration*0.02]
        for ax in (ax1, ax2):
            ax.set_xlabel(r'Time $\it t$ [s]')
            ax.set_xlim(xlim_)
            ax.grid(axis='y', linestyle='--', color=[.5, .5, .5])
            ax.minorticks_on()

        plt.show()

    def plot_SOC(self,
                 t_: Sequence | None = None,  # 时刻序列
                 ):
        """SOC-时间"""
        if t_ is None:
            t_ = self.data['t']
        SOC_ = self.interpolate('SOC', t_)      # 呈时间序列的全电池荷电状态
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
        csnegcent___ = self.interpolate(f'{cθ}sneg__', t_=t_, x_=self.xneg_, r_=[0])  # 呈时间序列的负极固相颗粒中心锂离子浓度
        csposcent___ = self.interpolate(f'{cθ}spos__', t_=t_, x_=self.xpos_, r_=[0])  # 呈时间序列的正极固相颗粒中心锂离子浓度
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
            y_ = *csnegcent__.ravel(), *[nan]*self.Nsep, *csposcent__.ravel()
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{{self.rSign[1:-1]}=0}}$ [{self.cUnit or '–'}]')
        ax1.legend(bbox_to_anchor=(1, 1))

        ax2.set_title('Lithium concentration at electrode particle surface', fontsize=12)
        for n, (csnegsurf_, cspossurf_, t) in enumerate(zip(csnegsurf__, cspossurf__, t_)):
            x_ = self.xPlot_
            y_ = *csnegsurf_, *[nan]*self.Nsep, *cspossurf_
            ax2.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(f'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.rSign[1:-1]} = {"{\\it R}_{s,reg}" if self.rUnit else 1} }}$ [{self.cUnit or '–'}]')

        ax3.set_title('Lithium-ion concentration in electrolyte', fontsize=12)
        for n, (ce_, t) in enumerate(zip(ce__, t_)):
            x_ = [0, *self.xPlot_, self.xInterfacesPlot_[-1]]
            y_ = [ce_[0], *ce_, ce_[-1]]
            ax3.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax3.set_ylabel(rf'{self.cSign}$_e$({self.xSign}, {self.tSign}) [{self.cUnit or '–'}]')

        self.plot_interfaces(ax1, ax2, ax3)
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
                A, σeffneg = getattr(self, 'A'), getattr(self, 'σeffneg')
                y_ = 1e3*hstack([φsneg_[0] + I/A/σeffneg*0.5*self.Δxneg, φsneg_, φsneg_[-1]])
            else:
                σneg = getattr(self, 'σneg')
                y_ = 1e3*hstack([φsneg_[0] + I/σneg*0.5*self.Δxneg, φsneg_, φsneg_[-1]])
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(r'Electrode potential ${\it φ}_{s,neg}$' + f'({self.xSign}, {self.tSign}) [mV]')
        ax1.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
        ax1.set_xlim(x_[0], x_[-1])
        ax1.grid(axis='y', linestyle='--')

        x_ = [self.xInterfacesPlot_[Nneg+Nsep], *self.xPlot_[-Npos:], self.xInterfacesPlot_[-1]]
        for n, (φspos_, I, t) in enumerate(zip(φspos__, I_, t_)):
            if self.xUnit:
                A, σeffpos = getattr(self, 'A'), getattr(self, 'σeffpos')
                y_ = 1e3*hstack([φspos_[0], φspos_, φspos_[-1] - I/A/σeffpos*0.5*self.Δxpos])
            else:
                σpos = getattr(self, 'σpos')
                y_ = 1e3*hstack([φspos_[0], φspos_, φspos_[-1] - I/σpos*0.5*self.Δxpos])
            ax2.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(ax1.get_ylabel().replace('neg', 'pos'))
        ax2.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')
        ax2.set_xlim(x_[0], x_[-1])
        ax2.grid(axis='y', linestyle='--')
        ax2.legend(bbox_to_anchor=[1, 1])

        for n, (φe_, t) in enumerate(zip(φe__, t_)):
            ax3.plot([0, *self.xPlot_, self.xInterfacesPlot_[-1]],
                     hstack([φe_[0], φe_, φe_[-1]]), 'o-', color=P2Dbase.get_color(t_, n),
                     label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(r'Electrolyte potential ${\it φ}_e$' + f'({self.xSign}, {self.tSign}) [mV]')

        self.plot_interfaces(ax3)
        plt.show()

    def plot_jint_i0int_ηint(self,
                             t_: Sequence | None = None,  # 时刻序列
                             ):
        """主反应局部体积电流密度、交换电流密度、过电位-空间、时间"""
        if t_ is None:
            t_ = self.data['t']
        jJ, iI = ['j', 'i'] if self.xUnit else ['J', 'I']
        jintneg__ = self.interpolate(f'{jJ}intneg_', t_=t_, x_=self.xneg_)    # 呈时间序列的负极局部体积电流密度场
        jintpos__ = self.interpolate(f'{jJ}intpos_', t_=t_, x_=self.xpos_)    # 呈时间序列的正极局部体积电流密度场
        i0intneg__ = self.interpolate(f'{iI}0intneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的负极交换电流密度场
        i0intpos__ = self.interpolate(f'{iI}0intpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的正极交换电流密度场
        ηintneg__ = self.interpolate('ηintneg_', t_=t_, x_=self.xneg_)*1e3    # 呈时间序列的负极固相表面过电位场 [mV]
        ηintpos__ = self.interpolate('ηintpos_', t_=t_, x_=self.xpos_)*1e3    # 呈时间序列的正极固相表面过电位场 [mV]

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
            y_ = *jintneg_, *[nan]*self.Nsep, *jintpos_
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(rf'{self.jSign}$_{{int}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        ax1.legend(bbox_to_anchor=[1, 1])

        ax2.set_title('Field of lithium (de-)intercalation exchange current density', fontsize=12)
        for n, (i0intneg_, i0intpos_, t) in enumerate(zip(i0intneg__, i0intpos__, t_)):
            x_ = self.xPlot_
            y_ = *i0intneg_, *[nan]*self.Nsep, *i0intpos_
            ax2.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(rf'{self.i0Sign}({self.xSign}, {self.tSign}) [{self.i0Unit}]')

        ax3.set_title('Field of lithium (de-)intercalation overpotential', fontsize=12)
        for n, (ηintneg_, ηintpos_, t) in enumerate(zip(ηintneg__, ηintpos__, t_)):
            x_ = self.xPlot_
            y_ = *ηintneg_, *[nan]*self.Nsep, *ηintpos_
            ax3.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax3.set_ylabel(rf'${{\it η}}_{{int}}$ ({self.xSign}, {self.tSign}) [mV]')

        self.plot_interfaces(ax1, ax2, ax3)
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
            y_ = *jDLneg_, *[nan]*self.Nsep, *jDLpos_
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'Double-layer local volumetric\ncurrent density {self.jSign}$_{{DL}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        self.plot_interfaces(ax1)
        ax1.legend(bbox_to_anchor=[1, 1])

        t_ = self.data['t']
        jDLneg__ = self.interpolate(f'{jJ}DLneg_', t_=t_, x_=self.xneg_)  # 呈时间序列的双电层效应负极局部体积电流密度场
        jDLpos__ = self.interpolate(f'{jJ}DLpos_', t_=t_, x_=self.xpos_)  # 呈时间序列的双电层效应正极局部体积电流密度场
        if jJ=='j':
            A = getattr(self, 'A')
            IDLneg_ = jDLneg__.sum(axis=1)*(self.Δxneg*A)
            IDLpos_ = jDLpos__.sum(axis=1)*(self.Δxpos*A)
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
            x = xR if LPmodel else (xR*getattr(self, 'Lneg'))
        elif 2<=xR<=3:
            reg = 'pos'
            x = xR if LPmodel else ((xR - 2)*getattr(self, 'Lpos') + getattr(self, 'Lneg') + getattr(self, 'Lsep'))
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
                    'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax.legend(bbox_to_anchor=[1, 1])
        ax.set_ylabel(rf'{self.cSign}$_s$({self.xSign}, {self.rSign}, {self.tSign})|$_{{ {self.xSign[1:-1]}={x if LPmodel else x*1e6:g}\;{self.xUnit} }}$ [{self.cUnit or '–'}]')
        ax.set_xlabel(rf'Radial location {self.rSign} [{self.rUnit or '–'}]')
        ax.set_xlim(X_[0], X_[-1])
        ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_jLP_ηLP(self,
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
            ax1.plot(self.xPlot_[:self.Nneg], jLP_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax1.set_ylabel(f'Lithium plating local volumetric\ncurrent density {self.jSign}$_{{LP}}$({self.xSign}, {self.tSign}) [{self.jUnit}]')
        ax1.legend(loc='upper left', bbox_to_anchor=[1, 1])

        for n, (ηLPneg_, t) in enumerate(zip(ηLPneg__, t_)):
            ax2.plot(self.xPlot_[:self.Nneg], ηLPneg_*1e3, 'o-', color=P2Dbase.get_color(t_, n), label=rf'{self.tSign} = {t:g} s')
        ax2.set_ylabel(rf'Lithium plating overpotential ${{\it η}}_{{LP}}$({self.xSign}, {self.tSign}) [mV]')
        for ax in [ax1, ax2]:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit}]')
            ax.set_xlim(0, self.xInterfacesPlot_[self.Nneg])
            ax.grid(axis='y', linestyle='--')

        plt.show()

    def plot_LP(self,
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
            A = getattr(self, 'A')
            ratio_ = jLP__.sum(axis=1)*self.Δxneg*A/I_  # 析锂电流与总电流之比
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

    def plot_OCV_OCP(self):
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

    def plot_nNewton(self):
        """Newton迭代次数nNewton-t曲线"""
        t_ = self.data['t']
        nNewton_ = self.data['nNewton']

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        ax.plot(t_, nNewton_, 'x-k')
        duration = max(t_) - min(t_)
        ax.set_xlim([t_[0] - duration*0.02, t_[-1] + duration*0.02])
        ax.set_xlabel(f'Time {self.tSign} [s]')
        ax.set_ylabel('Number of Newton iterations')
        ax.grid(axis='y', linestyle='--')
        plt.show()

    def plot_interfaces(self, *axes_):
        for ax in axes_:
            ax.set_xlabel(rf'Location {self.xSign} [{self.xUnit or '–'}]')  # 横坐标标签
            ax.set_ylim(ax.get_ylim())                   # 固定纵坐标上下限
            ax.set_xlim(self.xInterfacesPlot_[[0, -1]])  # 横坐标上下限
            kw = {'ymin': ax.get_ylim()[0], 'ymax': ax.get_ylim()[1],
                  'ls': '--', 'color': [.5]*3, 'alpha': 0.5}
            ax.vlines(self.xInterfacesPlot_[self.Nneg], **kw)              # 负极-隔膜界面
            ax.vlines(self.xInterfacesPlot_[self.Nneg + self.Nsep], **kw)  # 隔膜-正极界面
            ax.grid(axis='y', linestyle='--')  # 纵坐标网格线
    
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
    def s2idx(s: slice) -> ndarray:
        """将切片索引s转化为数组索引idx"""
        return arange(s.start, s.stop, s.step)

    @abstractmethod
    def _stepping(self, Δt):
        """时间步进：Newton法迭代因变量"""
        pass

    @property
    @abstractmethod
    def U(self):
        """正负极端电压 [V]"""
        pass

    @property
    @abstractmethod
    def θsneg(self):
        """负极嵌锂状态"""
        pass

    @property
    @abstractmethod
    def θspos(self):
        """正极嵌锂状态"""
        pass

    @property
    @abstractmethod
    def ILP(self):
        """析锂反应电流 [A]"""
        pass

    @property
    @abstractmethod
    def xPlot_(self):
        """全区域控制体中心的坐标（用于作图） """
        pass

    @property
    @abstractmethod
    def xInterfacesPlot_(self):
        """各控制体交界面的坐标（用于作图） """
        pass

    @abstractmethod
    def update_K__with_pure_electrochemical_parameters(self):
        # 对K__矩阵赋纯电化学参数相关值
        pass

if __name__=='__main__':
    pass
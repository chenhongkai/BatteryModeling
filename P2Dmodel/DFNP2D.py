#%%
import warnings
from typing import Sequence

import matplotlib.pyplot as plt
from scipy.linalg.lapack import dgtsv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from numpy import ndarray, array, arange, zeros, zeros_like, full, hstack, concatenate, tile, \
    exp, sqrt, sinh, cosh, arcsinh, outer, \
    ix_, isnan
from numpy.linalg import solve

from P2Dmodel.P2Dbase import P2Dbase
from P2Dmodel.tools import triband_to_dense


class DFNP2D(P2Dbase):
    """锂离子电池经典准二维模型 Doyle-Fuller-Newman Pseudo-two-Dimensional model"""

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
        P2Dbase.__init__(self, SOC0=SOC0,
                         θminneg=θminneg, θmaxneg=θmaxneg,
                         θminpos=θminpos, θmaxpos=θmaxpos, **kwargs)
        Nneg, Nsep, Npos, Ne, Nr = self.Nneg, self.Nsep, self.Npos, self.Ne, self.Nr  # 读取：网格数
        lithiumPlating, radialDiscretization, decouple_cs, complete, verbose = (
            self.lithiumPlating, self.radialDiscretization, self.decouple_cs, self.complete, self.verbose)  # 读取：模式
        # 若不考虑双电层效应，正负极双电层电容赋0
        if not self.doubleLayerEffect:
            self.CDLneg = self.CDLpos = 0
        if verbose and (εsneg + εeneg)>1:
            warnings.warn(f'负极固相体积分数εsneg与负极电解液体积分数εeneg之和大于1，{εsneg + εeneg = } > 1')
        if verbose and (εspos + εepos)>1:
            warnings.warn(f'正极固相体积分数εspos与正极电解液体积分数εepos之和大于1，{εspos + εepos = } > 1')
        # 状态量
        csneg = csmaxneg*(θminneg + SOC0*(θmaxneg - θminneg))  # 初始负极固相锂离子浓度 [mol/m^3]
        cspos = csmaxpos*(θmaxpos + SOC0*(θminpos - θmaxpos))  # 初始正极固相锂离子浓度 [mol/m^3]
        self.csneg__ = full((Nr, Nneg), csneg)  # 初始化：负极固相颗粒锂离子浓度场 [mol/m^3]
        self.cspos__ = full((Nr, Npos), cspos)  # 初始化：正极固相颗粒锂离子浓度场 [mol/m^3]
        self.csnegsurf_ = full(Nneg, csneg)     # 初始化：负极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.cspossurf_ = full(Npos, cspos)     # 初始化：正极固相颗粒表面锂离子浓度场 [mol/m^3]
        self.ce_ = full(Ne, ce0)                # 初始化：电解液锂离子浓度场 [mol/m^3]
        self.jintneg_, self.jintpos_ = zeros(Nneg), zeros(Npos)  # 初始化：负极、正极主反应局部体积电流密度场 [A/m^3]
        self.jDLneg_, self.jDLpos_   = zeros(Nneg), zeros(Npos)  # 初始化：负极、正极双电层效应局部体积电流密度场 [A/m^3]
        i0intneg = self.i0intneg if self._i0intneg else DFNP2D.solve_i0int_(self.kneg, csmaxneg, csneg, ce0)
        i0intpos = self.i0intpos if self._i0intpos else DFNP2D.solve_i0int_(self.kpos, csmaxpos, cspos, ce0)
        self.i0intneg_, self.i0intpos_ = full(Nneg, i0intneg), full(Npos, i0intpos)  # 初始化：负极、正极主反应交换电流密度场 [A/m^2]
        self.jneg_, self.jpos_ = zeros(Nneg), zeros(Npos)                            # 初始化：负极、正极总局部体积电流密度 [A/m^3]
        if lithiumPlating:
            self.jLP_  = zeros(Nneg)      # 初始化：负极析锂局部体积电流密度场 [A/m^3]
            i0LP = self.i0LP if self._i0LP else DFNP2D.solve_i0LP_(self.kLP, ce0)
            self.i0LP_ = full(Nneg, i0LP)  # 初始化：负极析锂反应交换电流密度 [A/m^2]
        # 恒定量
        self.x_, self.Δx_, self.ΔxWest_, self.ΔxEast_ = P2Dbase.generate_x_related_coordinates(Nneg, Nsep, Npos, Lneg, Lsep, Lpos)
        self.rneg_, self.Δrneg_, self.Vr_ = P2Dbase.generate_r_related_coordinates(Nr, Rsneg, radialDiscretization)
        self.rpos_, self.Δrpos_, _        = P2Dbase.generate_r_related_coordinates(Nr, Rspos, radialDiscretization)
        # 需记录的数据名称
        if complete:
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
        self.data = {dataname: [] for dataname in self.datanames_}  # 初始化：存储呈时间序列的运行数据
        N = self.generate_indices_of_dependent_variables()  # 因变量索引
        self.K__ = K__ = zeros((N, N))                      # 因变量线性矩阵
        # 对K__矩阵赋参数相关值
        if decouple_cs:
            pass
        else:
            self.update_K__idxcsnegsurf_idxjintneg_(self.aneg, self.Dsneg)
            self.update_K__idxcspossurf_idxjintpos_(self.apos, self.Dspos)
        self.update_K__idxφsneg_idxjneg_(self.σeffneg)
        self.update_K__idxφspos_idxjpos_(self.σeffpos)
        self.update_K__idxφe_idxφe_(κeff_ := self.κeff_, κeff_)
        self.update_K__idxηintneg_idxjneg_(self.RSEIneg, self.aeffneg)
        self.update_K__idxηintpos_idxjpos_(self.RSEIpos, self.aeffpos)
        if lithiumPlating:
            self.update_K__idxηLP_idxjneg_(self.RSEIneg, self.aeffneg)
        # 对K__矩阵赋固定值（x方向网格相关值，常数值）
        self.assign_K__with_constants()
        # 对K__额外赋固定值（此处为参数Rsneg、Rspos相关，据其物理意义，这两参数在电池运行过程中不应变化）
        # 负极、正极固相表面浓度cssurf行
        for i, (idxcs_, idxcssurf_, r_, Rs) in enumerate(zip(
                [self.idxcsneg_, self.idxcspos_],
                [self.idxcsnegsurf_, self.idxcspossurf_],
                [self.rneg_, self.rpos_],
                [Rsneg, Rspos],)):
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
                K__[idxcssurf_, idxcs_[Nr-3::Nr]] = a1*a2/(-a3*(r_3 - r_1)*(r_3 - r_2))
                K__[idxcssurf_, idxcs_[Nr-2::Nr]] = a1*a3/(-a2*(r_2 - r_1)*(r_2 - r_3))
                K__[idxcssurf_, idxcs_[Nr-1::Nr]] = a2*a3/(-a1*(r_1 - r_2)*(r_1 - r_3))
                K__[idxcssurf_, idxcssurf_] = 1/a1 + 1/a2 + 1/a3

        self.bandKcsneg__ = bandKcsneg__ = zeros((3, Nr))  # 负极固相浓度矩阵的带 [m^-2]，仅与Rsneg、Nr相关
        self.bandKcspos__ = bandKcspos__ = zeros((3, Nr))  # 正极固相浓度矩阵的带 [m^-2]，仅与Rspos、Nr相关
        idx_ = arange(Nr)
        idxm_ = idx_[1:-1]
        for band__, r_, Δr_ in zip([bandKcsneg__, bandKcspos__],
                                   [self.rneg_, self.rpos_],
                                   [self.Δrneg_, self.Δrpos_]):
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

        if verbose and type(self) is DFNP2D:
            print(self)
            print('经典P2D模型(DFNP2D)初始化完成!')

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
        idxjLP_ = self.idxjLP_
        idxηLP_ = self.idxηLP_
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
        T, F, R= self.T, P2Dbase.F, P2Dbase.R  # 读取：温度 [K]、法拉第常数 [C/mol]、摩尔气体常数 [J/(mol·K)]
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
        return φsposCollector - φsnegCollector

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
        return 2*aeff*i0int_*sinh(P2Dbase.F/(2*P2Dbase.R*T)*ηint_)

    @staticmethod
    def solve_djintdi0int_(T, aeff, ηint_) -> ndarray:
        """求解主反应局部体积电流密度jint对交换电流密度i0int的偏导数 [A/m^3 / A/m^2]"""
        return 2*aeff*sinh(P2Dbase.F/(2*P2Dbase.R*T)*ηint_)

    @staticmethod
    def solve_djintdηint_(T, aeff, i0int_, ηint_) -> ndarray:
        """求解主反应局部体积电流密度jint对过电位ηint的偏导数 [A/m^3 / V]"""
        FRT = P2Dbase.F/(P2Dbase.R*T)
        return FRT*aeff*i0int_*cosh(FRT*0.5*ηint_)

    @staticmethod
    def solve_i0int_(k, csmax, cssurf_, ce_) -> ndarray:
        """求主反应交换电流密度 [A/m^2]"""
        return P2Dbase.F * k * sqrt(ce_*(csmax - cssurf_)*cssurf_)

    @staticmethod
    def solve_di0intdce_(ce_, i0int_):
        """求解主反应交换电流密度i0int对电解液锂离子浓度ce的偏导数 [A/m2 / mol/m^3]"""
        return 0.5*i0int_/ce_

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

    @staticmethod
    def solve_djLPdηLP_(T, aeffneg, i0LP_, ηLP_):
        """求解析锂反应局部体积电流密度jLP对析锂过电位ηLP的偏导数 [A/m^3 / V]"""
        FRT = P2Dbase.F/(P2Dbase.R*T)
        a, b = 0.3*FRT, -0.7*FRT
        djLPdηLP_ = aeffneg * i0LP_ * (a*exp(a*ηLP_) - b*exp(b*ηLP_))
        djLPdηLP_[ηLP_>=0] = 0
        return djLPdηLP_

    @staticmethod
    def solve_i0LP_(kLP, ceneg_) -> ndarray:
        """由液相浓度场求析锂反应交换电流密度 [A/m^2]"""
        return P2Dbase.F * kLP * ceneg_**0.3

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
            x_ = (0, *self.xPlot_[:Nneg], self.xInterfacesPlot_[Nneg],
                  *self.xPlot_[Nneg:Nneg+Nsep], self.xInterfacesPlot_[Nneg+Nsep],
                  *self.xPlot_[-Npos:], self.xInterfacesPlot_[-1])
            y_ = i, *isneg_, 0, *[0]*Nsep, 0, *ispos_, i
            ax1.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'$\it t$ = {t:g} s')
        ax1.set_ylabel(rf'Solid-phase current density ${{\it i}}_{{s}}$({self.xSign}, {self.tSign}) [A/m$^2$]')
        ax1.legend(bbox_to_anchor=[1, 1])

        for n, (ie_, i, t) in enumerate(zip(ie__, i_, t_)):
            x_ = 0, *self.xPlot_, self.xInterfacesPlot_[-1]
            y_ = 0, *ie_, 0
            ax2.plot(x_, y_, 'o-', color=P2Dbase.get_color(t_, n), label=rf'$\it t$ = {t:g} s')
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
        F, T = P2Dbase.F, self.T
        F2RT = F/(2*P2Dbase.R*T)
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
    cell.plot_jint_i0int_ηint(arange(0, 2001, 200))
    cell.plot_jDL(arange(0, 2001, 200))
    cell.plot_csr(range(0, 2001, 200), 1)
    cell.plot_jLP_ηLP(arange(1700, 2301, 100))
    cell.plot_LP()
    cell.plot_OCV_OCP()
    cell.plot_dUOCPdθs()
    cell.plot_i(arange(0, 2001, 200))
    '''
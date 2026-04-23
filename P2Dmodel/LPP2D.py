#%%
import numpy as np
from scipy.linalg.lapack import dgtsv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from P2Dmodel import DFNP2D, triband_to_dense


class LPP2D(DFNP2D):
    """锂离子电池集总参数准二维模型 Lumped-Parameter Pseudo-two-Dimension model"""

    def __init__(self,
            Qcell: float = 20.,     # 电池理论可用容量 [Ah]
            Qneg: float = 26.,      # 负极容量 [Ah]
            Qpos: float = 23.,      # 正极容量 [Ah]
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
            De: float = 0.2,        # 集总离子扩散率/电导率之比 [A/S]
            κD: float = 4.39e-4,    # 电解液集总扩散离子电导率系数 [V/K]
            kneg: float = 32.,      # 负极集总反应速率常数 [A]
            kpos: float = 42.,      # 正极集总反应速率常数 [A]
            kLP: float = 3.607e-6,  # 负极集总析锂反应速率常数 [A]
            RSEIneg: float = 6.91e-5,  # 负极集总SEI膜内阻 [Ω]
            RSEIpos: float = 2e-5,     # 正极集总SEI膜内阻 [Ω]
            CDLneg: float = 144.691,   # 负极集总双电层电容 [F]
            CDLpos: float = 19.971,    # 正极集总双电层电容 [F]
            l: float = 1e-13,          # 等效电感 [H]
            I0intneg: float | None = None,  # 负极主反应集总交换电流密度 [A]
            I0intpos: float | None = None,  # 正极主反应集总交换电流密度 [A]
            I0LP: float | None = None,      # 负极析锂反应集总交换电流密度 [A]
            Umin: float = 2.8,      # SOC=100%开路电压 [V]
            Umax: float  = 4.2,     # SOC=0%开路电压 [V]
            θminneg: float = None,  # SOC=0%的负极嵌锂状态 [–]，默认需要由Qcell、Qneg、Qpos计算4个边界嵌锂状态
            θmaxneg: float = None,  # SOC=100%的负极嵌锂状态 [–]
            θminpos: float = None,  # SOC=100%的正极嵌锂状态 [–]
            θmaxpos: float = None,  # SOC=0%的正极嵌锂状态 [–]
            SOC0: float = 0.5,   # 初始荷电状态 [–]
            T0: float = 298.15,  # 初始温度 [K]
            **kwargs,):
        DFNP2D.__init__(self, fullyInitialize=False, **kwargs)
        # 6容量参数
        self.Qcell = Qcell; assert Qcell>0, f'电池理论可用容量{Qcell = }，应大于0 [Ah]'
        self.Qneg = Qneg; assert Qneg>0, f'负极容量{Qneg = }，应大于0 [Ah]'
        self.Qpos = Qpos; assert Qpos>0, f'正极容量{Qpos = }，应大于0 [Ah]'
        if Qcell>=Qneg or Qcell>=Qpos: raise DFNP2D.Error(f'电池理论可用容量{Qcell = }应小于负极容量{Qneg = }和正极容量{Qpos = }',
                                                          {'Qcell': Qcell, 'Qneg': Qneg, 'Qpos': Qpos})
        self.qeneg = qeneg; assert qeneg>0, f'负极电解液锂离子电荷量{qeneg = }，应大于0 [C]'
        self.qesep = qesep; assert qesep>0, f'隔膜电解液锂离子电荷量{qesep = }，应大于0 [C]'
        self.qepos = qepos; assert qepos>0, f'正极电解液锂离子电荷量{qepos = }，应大于0 [C]'
        # 9输运参数
        self.σneg = σneg; assert σneg>0, f'负极集总固相电导率{σneg = }，应大于0 [S]'
        self.σpos = σpos; assert σpos>0, f'正极集总固相电导率{σpos = }，应大于0 [S]'
        self.κneg = κneg; assert κneg>0, f'负极电解液集总离子电导率{κneg = }，应大于0 [S]'
        self.κsep = κsep; assert κsep>0, f'隔膜电解液集总离子电导率{κsep = }，应大于0 [S]'
        self.κpos = κpos; assert κpos>0, f'正极电解液集总离子电导率{κpos = }，应大于0 [S]'
        self.Dsneg = Dsneg; assert Dsneg>0, f'负极集总固相锂离子扩散系数{Dsneg = }，应大于0 [1/s]'
        self.Dspos = Dspos; assert Dspos>0, f'正极集总固相锂离子扩散系数{Dspos = }，应大于0 [1/s]'
        self.De = De; assert De>0, f'集总离子扩散率/电导率之比{De = }，应大于0 [A/S]'
        self.κD = κD; assert κD>0, f'集总扩散电解液离子电导率系数{κD = }，应大于0 [V/K]'
        # 5动力学参数
        self.kneg = kneg; assert kneg>0, f'负极集总主反应速率常数{kneg = }，应大于0 [A]'
        self.kpos = kpos; assert kpos>0, f'正极集总主反应速率常数{kpos = }，应大于0 [A]'
        self.kLP = kLP;   assert kLP>0, f'负极集总析锂反应速率常数{kLP = }，应大于0 [A]'
        self.RSEIneg = RSEIneg; assert RSEIneg>=0, f'负极集总SEI膜电阻{RSEIneg = }，应大于或等于0 [Ω]'
        self.RSEIpos = RSEIpos; assert RSEIpos>=0, f'正极集总SEI膜电阻{RSEIpos = }，应大于或等于0 [Ω]'
        # 3电抗参数
        self.CDLneg = CDLneg; assert CDLneg>=0, f'负极集总双电层电容{CDLneg = }，应大于或等于0 [F]'
        self.CDLpos = CDLpos; assert CDLpos>=0, f'正极集总双电层电容{CDLpos = }，应大于或等于0 [F]'
        self.l = l;           assert l>=0, f'等效电感{l = }，应大于或等于0 [H]'
        # 交换电流密度
        self.I0intneg = self.i0intneg = I0intneg; assert (I0intneg is None) or (I0intneg>0), f'负极主反应集总交换电流密度{I0intneg = }，应大于0 [A]'
        self.I0intpos = self.i0intpos = I0intpos; assert (I0intpos is None) or (I0intpos>0), f'正极主反应集总交换电流密度{I0intpos = }，应大于0 [A]'
        self.I0LP = self.i0LP = I0LP;             assert (I0LP is None)  or (I0LP>0), f'负极析锂反应集总交换电流密度{I0LP = }，应大于0 [A]'
        # 运行电压
        assert Umax>Umin>0, f'{Umax = }，{Umin = }，应满足Umax > Umin > 0 [V]'
        self.Umin = Umin  # 100%SOC开路电压 [V]
        self.Umax = Umax  # 0%SOC开路电压 [V]
        # 4边界嵌锂状态参数
        if all(v is not None for v in (θminneg, θmaxneg, θminpos, θmaxpos)):
            assert 0<θminneg<θmaxneg<1, f'负极最小、最大嵌锂状态{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1'
            assert 0<θminpos<θmaxpos<1, f'正极最小、最大嵌锂状态{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1'
            if self.verbose:
                print('4个边界嵌锂状态均非None，使用给定的4个边界嵌锂状态，忽略4等式关系，并重新计算负极、正极容量Qneg、Qpos')
            self.Qneg = self.Qcell/(θmaxneg - θminneg)
            self.Qpos = self.Qcell/(θmaxpos - θminpos)
        else:
            # 使用4等式由Qcell、Qneg、Qpos计算4θ
            θminneg, θmaxneg, θminpos, θmaxpos, ΔFmax = DFNP2D.solve_4θ(
                self.solve_Uneg_, self.solve_Upos_,
                self.Qcell, self.Qneg, self.Qpos,
                self.Umin, self.Umax)
            if self.verbose:
                print(f'由Qcell、Qneg、Qpos求4个边界嵌锂状态，F函数最大绝对误差{ΔFmax}')
            tempdict = {'θminneg': θminneg, 'θmaxneg': θmaxneg, 'θminpos': θminpos, 'θmaxpos': θmaxpos, 'ΔFmax': ΔFmax}
            if not ΔFmax<1e-5:
                raise DFNP2D.Error(f'求4个边界嵌锂状态，不收敛，无解！F函数最大绝对误差{ΔFmax = }', tempdict)
            if not 0<θminneg<θmaxneg<1:
                raise DFNP2D.Error(f'负极嵌锂状态{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1', tempdict)
            if not 0<θminpos<θmaxpos<1:
                raise DFNP2D.Error(f'正极嵌锂状态{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1', tempdict)
        self.θminneg = θminneg  # SOC=0%的负极嵌锂状态
        self.θmaxneg = θmaxneg  # SOC=100%的负极嵌锂状态
        self.θmaxpos = θmaxpos  # SOC=0%的正极嵌锂状态
        self.θminpos = θminpos  # SOC=100%的正极嵌锂状态
        # 作图变量单位
        self.xUnit = ''  # 横坐标x单位
        self.rUnit = ''  # 径向标r单位
        self.θUnit = self.cUnit = ''        # 浓度单位
        self.JUnit = self.jUnit = 'A'       # 局部体积电流密度单位
        self.I0Unit = self.i0Unit = 'A'     # 交换电流密度单位
        self.xSign = r'$\overline{\it x}$'  # 横坐标x符号
        self.rSign = r'$\overline{\it r}$'  # 横坐标r符号
        self.θSign = self.cSign = r'${\it θ}$'        # 浓度符号
        self.JSign = self.jSign = r'${\it J}$'        # 局部体积电流密度符号
        self.I0Sign = self.i0Sign = r'${\it I}_{0}$'  # 交换电流密度符号
        (
        # 状态量
        self.θsneg__, self.θspos__,        # 负极、正极无量纲固相锂离子浓度场 [–]
        self.θe_,                          # 电解液无量纲锂离子浓度场 [–]
        self.Jintneg_, self.Jintpos_,      # 负极、正极集总主反应局部体积电流密度场 [A]
        self.JDLneg_, self.JDLpos_,        # 负极、正极集总双电层效应局部体积电流密度场 [A]
        self.JLP_,                         # 负极析锂局部体积电流密度场 [A/m^3]
        # 恒定量
        self.r_,         # (Nr,) 球形固相颗粒半径方向控制体中心的坐标 [–]
        self.Δr_,        # (Nr,) 球形固相颗粒球壳控制体厚度 [–]
        self.bandKθs__,  # (3, Nr) 固相锂离子浓度矩阵的带
        # 索引集总因变量
        self.idxθsneg_, self.idxθspos_, self.idxθsnegsurf_, self.idxθspossurf_, self.idxθe_,
        self.idxJintneg_, self.idxJintpos_, self.idxJDLneg_, self.idxJDLpos_, self.idxJLP_,
        self.idxI0intneg_, self.idxI0intpos_,
        self.idxθ_, self.idxJ_,
        ) = (None,)*25
        # 初始化
        if type(self) is LPP2D:
            self.initialize(
                SOC0=SOC0,  # 初始荷电状态 [–]
                T0=T0)      # 初始温度 [K]

    def initialize(self,
            SOC0: int | float = 0.,    # 初始荷电状态 [–]
            T0: int | float = 298.15,  # 初始温度 [K]
            ):
        """初始化"""
        if self.verbose and type(self) is LPP2D:
            print(f'集总参数P2D模型初始化...')
        assert 0<=SOC0<=1, f'初始荷电状态{SOC0 = }，取值范围应为[0, 1]'
        self.T = T0; assert T0>0, f'初始温度{T0 = }，应大于0 [K]'
        self.I = 0.  # 初始化：电流 [A]
        self.t = 0.  # 初始化：时刻 [s]
        Nneg, Nsep, Npos, Ne, Nr = self.Nneg, self.Nsep, self.Npos, self.Ne, self.Nr  # 读取：网格数
        # 恒定量
        self.Δxneg = Δxneg = 1/Nneg  # 负极网格厚度 [–]
        self.Δxsep = Δxsep = 1/Nsep  # 隔膜网格厚度 [–]
        self.Δxpos = Δxpos = 1/Npos  # 正极网格厚度 [–]
        self.x_ = np.block([
            np.linspace(0, 1, Nneg + 1)[:-1] + Δxneg/2,
            np.linspace(1, 2, Nsep + 1)[:-1] + Δxsep/2,
            np.linspace(2, 3, Npos + 1)[:-1] + Δxpos/2,])  # (Ne,) 各控制体中心坐标 [–]
        self.generate_x_related_coordinates()
        if not self.doubleLayerEffect:
            self.CDLneg = self.CDLpos = 0  # 若不考虑双电层效应，正负极双电层电容赋0
        # 固相锂离子浓度矩阵
        match self.radialDiscretization:
            case 'EI':  # 等间隔划分球壳网格
                Δr = 1/Nr  # 球壳网格厚度 [–]
                self.r_ = (Δr*np.arange(1, Nr+1) + Δr*np.arange(0, Nr))/2  # 颗粒半径方向控制体中心的坐标 [–]
                self.Δr_ = np.full(Nr, Δr)                                 # 颗粒球壳网格厚度序列 [–]
            case 'EV':  # 等体积划分球壳网格
                V = 4/3*np.pi  # 颗粒体积 [–]
                ΔV = V/Nr      # 球壳控制体体积 [–]
                rW_ = (ΔV*np.arange(0, Nr)/(4/3*np.pi))**(1/3)     # Nr维向量：球壳内界面坐标 [–]
                rE_ = (ΔV*np.arange(1, Nr + 1)/(4/3*np.pi))**(1/3) # Nr维向量：球壳外界面坐标 [–]
                self.r_ = (rW_ + rE_)/2
                self.Δr_ = rE_ - rW_
        r_  = self.rneg_  = self.rpos_  = self.r_
        Δr_ = self.Δrneg_ = self.Δrpos_ = self.Δr_
        self.Vr_ = (r_ + Δr_/2)**3 - (r_ - Δr_/2)**3  # 从中心到边缘负极固相颗粒球壳体积分数序列 [–]
        self.initialize_linear_matrix()
        # 状态量
        θneg = self.θminneg + SOC0*(self.θmaxneg - self.θminneg)  # 初始化：负极嵌锂状态 [–]
        θpos = self.θmaxpos + SOC0*(self.θminpos - self.θmaxpos)  # 初始化：正极嵌锂状态 [–]
        self.θsneg__ = np.full((Nr, Nneg), θneg)    # 初始化：负极固相颗粒无量纲锂离子浓度场 [–]
        self.θspos__ = np.full((Nr, Npos), θpos)    # 初始化：正极固相颗粒无量纲锂离子浓度场 [–]
        self.θe_ = np.ones(self.Ne)     # 初始化：电解液无量纲锂离子浓度场 [–]
        self.Jintneg_ = np.zeros(Nneg)  # 初始化：负极主反应集总局部体积电流 [A]
        self.Jintpos_ = np.zeros(Npos)  # 初始化：正极主反应集总局部体积电流 [A]
        self.φsneg_ = self.Uneg_        # 初始化：负极固相电势场 [V]
        self.φspos_ = self.Upos_        # 初始化：正极固相电势场 [V]
        self.φe_ = np.zeros(Ne)         # 初始化：电解液电势场 [V]
        self.JDLneg_ = np.zeros(Nneg)   # 初始化：负极双电层效应集总局部体积电流场 [A]
        self.JDLpos_ = np.zeros(Npos)   # 初始化：负极双电层效应集总局部体积电流场 [A]
        self.JLP_ = np.zeros(Nneg)      # 初始化：负极析锂反应集总局部体积电流 [A]
        self.QLP = 0.                   # 初始化：累计析锂量 [Ah]
        # 需记录的数据名称
        self.datanames_ = ['U', 'I', 't',        # 端电压 [V]、电流 [A]、时刻 [s]
                           'ηLPneg_', 'ηLPpos_',  # 负极、正极表面析锂反应过电位场 [V]
                           ]
        if self.complete:
            self.datanames_.extend([
                'θsneg__', 'θspos__',        # 负极、正极固相无量纲锂离子浓度场 [–]
                'θsnegsurf_', 'θspossurf_',  # 负极、正极表面无量纲锂离子浓度场 [–]
                'θe_',                       # 电解液无量纲锂离子浓度场 [–]
                'φsneg_', 'φspos_',          # 负极、正极固相电势场 [V]
                'φe_',                       # 电解液电势场 [V]
                'Jintneg_', 'Jintpos_',      # 负极、正极主反应集总局部体积电流密度场 [A]
                'JDLpos_', 'JDLneg_',        # 负极、正极双电层效应集总局部体积电流密度场 [A]
                'I0intneg_', 'I0intpos_',    # 负极、正极主反应集总交换电流密度场 [A]
                'ηintneg_', 'ηintpos_',      # 负极、正极主反应过电位场 [V]
                'JLP_',                      # 负极析锂集总局部体积电流密度场 [A]
                'θneg', 'θpos', 'SOC',       # 负极、正极嵌锂状态、全电池荷电状态 [–]
                'T', 'Qgen',])               # 温度 [K]、产热量 [W]
        self.data = {name: [] for name in self.datanames_}  # 字典：存储呈时间序列的运行数据
        if self.verbose and type(self) is LPP2D:
            print(self)
            print(f'集总参数P2D模型初始化完成!')
        return self

    def initialize_linear_matrix(self):
        """初始化因变量线性矩阵K__"""
        N = self.generate_indices_of_dependent_variables()
        self.K__ = K__ = np.zeros([N, N])  # 因变量线性矩阵
        if self.verbose:
            print(f'初始化因变量线性矩阵 K__.shape = {K__.shape}')
        # 覆盖对应集总因变量索引
        idxθsneg_ = self.idxθsneg_ = self.idxcsneg_
        idxθspos_ = self.idxθspos_ = self.idxcspos_
        idxθsnegsurf_ = self.idxθsnegsurf_ = self.idxcsnegsurf_
        idxθspossurf_ = self.idxθspossurf_ = self.idxcspossurf_
        self.idxθe_ = self.idxce_
        self.idxθ_ =self.idxc_
        self.idxJintneg_ = self.idxjintneg_
        self.idxJintpos_ = self.idxjintpos_
        self.idxJLP_ = self.idxjLP_
        self.idxJDLneg_ = self.idxjDLneg_
        self.idxJDLpos_ = self.idxjDLpos_
        self.idxJ_ = self.idxj_
        self.idxI0intneg_ = self.idxi0intneg_
        self.idxI0intpos_ = self.idxi0intpos_
        
        ## 对K__矩阵赋参数相关值 ##
        if decouple_cs := self.decouple_cs:
            pass
        else:
            self.update_K__idxθsnegsurf_idxJintneg_(self.Qneg, self.Dsneg)
            self.update_K__idxθspossurf_idxJintpos_(self.Qpos, self.Dspos)
        self.update_K__idxφsneg_idxJneg_(self.σneg)
        self.update_K__idxφspos_idxJpos_(self.σpos)
        self.update_K__idxφe_idxφe_(κ_:=self.κ_, κ_)
        self.update_K__idxηintneg_idxJneg_(self.RSEIneg)
        self.update_K__idxηintpos_idxJpos_(self.RSEIpos)
        if self.lithiumPlating:
            self.update_K__idxηLP_idxJneg_(self.RSEIneg)

        ## 对K__矩阵赋固定值 ##
        DFNP2D.assign_K__with_constants(self)
        # 集总参数模型需额外赋固定值（原始模型的此处为参数Rsneg、Rspos相关的值）
        Nneg, Npos, Nr = self.Nneg, self.Npos, self.Nr  # 读取：网格数
        r_, Δr_ = self.r_, self.Δr_                     # 读取：颗粒网格坐标 [–]
        r_3, r_2, r_1 = r_[-3:]
        a3, a2, a1 = 1 - r_[-3:]
        self.coeffs_ = np.array([
            a1*a2/((r_3 - r_1)*(r_3 - r_2)),
            a1*a3/((r_2 - r_1)*(r_2 - r_3)),
            a2*a3/((r_1 - r_2)*(r_1 - r_3))])  # 用于由3个颗粒内部节点浓度外推表面浓度的系数
        # 负极、正极固相表面浓度θsnegsurf行、θspossurf行
        if decouple_cs:
            K__[idxθsnegsurf_, idxθsnegsurf_] = \
            K__[idxθspossurf_, idxθspossurf_] = 1
        else:
            K__[idxθsnegsurf_, idxθsneg_[Nr-3::Nr]] = \
            K__[idxθspossurf_, idxθspos_[Nr-3::Nr]] = a1*a2 / (-a3*(r_3 - r_1)*(r_3 - r_2))
            K__[idxθsnegsurf_, idxθsneg_[Nr-2::Nr]] = \
            K__[idxθspossurf_, idxθspos_[Nr-2::Nr]] = a1*a3 / (-a2*(r_2 - r_1)*(r_2 - r_3))
            K__[idxθsnegsurf_, idxθsneg_[Nr-1::Nr]] = \
            K__[idxθspossurf_, idxθspos_[Nr-1::Nr]] = a2*a3 / (-a1*(r_1 - r_2)*(r_1 - r_3))
            K__[idxθsnegsurf_, idxθsnegsurf_] = \
            K__[idxθspossurf_, idxθspossurf_] = 1/a1 + 1/a2 + 1/a3

        self.bandKθs__ = bandKθs__ = np.zeros((3, Nr))  # 固相浓度三对角矩阵的带
        Kθs__ = np.zeros((Nr, Nr))  # 固相浓度矩阵
        idx_ = np.arange(Nr)
        idxm_ = idx_[1:-1]
        a = (r_[0] + Δr_[0]/2)**2/(r_[1] - r_[0])
        Kθs__[0, :2] = a, -a  # 首行
        a = (r_[-1] - Δr_[-1]/2)**2/(r_[-1] - r_[-2])
        Kθs__[-1, -2:] = -a, a  # 末行
        Kθs__[idxm_, idx_[:-2]] = a_ = -(r_[1:-1] - Δr_[1:-1]/2)**2/(r_[1:-1] - r_[:-2])  # 下对角线
        Kθs__[idxm_, idx_[2:]]  = c_ = -(r_[1:-1] + Δr_[1:-1]/2)**2/(r_[2:] - r_[1:-1])   # 上对角线
        Kθs__[idxm_, idx_[1:-1]] = -(a_ + c_)                                             # 主对角线
        Kθs__ /= (((r_ + Δr_/2)**3 - (r_ - Δr_/2)**3)/3).reshape(-1, 1)
        diag = Kθs__.diagonal
        bandKθs__[0, 1:]  = diag(1)   # 上对角线
        bandKθs__[1, :]   = diag(0)   # 主对角线
        bandKθs__[2, :-1] = diag(-1)  # 下对角线

    def update_K__idxθsnegsurf_idxJintneg_(self, Qneg, Dsneg):
        # 更新K__矩阵θsnegsurf行Jintneg列
        self.K__[self.idxθsnegsurf_, self.idxJintneg_] = 1/(10800*Qneg*Dsneg)

    def update_K__idxθspossurf_idxJintpos_(self, Qpos, Dspos):
        # 更新K__矩阵θspossurf行Jintpos列
        self.K__[self.idxθspossurf_, self.idxJintpos_] = 1/(10800*Qpos*Dspos)

    def update_K__idxφsneg_idxJneg_(self, σneg):
        # 更新K__矩阵φsneg行Jneg列
        DFNP2D.update_K__idxφsneg_idxjneg_(self, σneg)

    def update_K__idxφspos_idxJpos_(self, σpos):
        # 更新K__矩阵φspos行Jpos列
        DFNP2D.update_K__idxφspos_idxjpos_(self, σpos)

    def update_K__idxηintneg_idxJneg_(self, RSEIneg):
        # 更新K__矩阵ηintneg行Jneg列
        DFNP2D.update_K__idxηintneg_idxjneg_(self, RSEIneg, 1)

    def update_K__idxηintpos_idxJpos_(self, RSEIpos):
        # 更新K__矩阵ηintpos行Jpos列
        DFNP2D.update_K__idxηintpos_idxjpos_(self, RSEIpos, 1)

    def update_K__idxηLP_idxJneg_(self, RSEIneg):
        # 更新K__矩阵ηLP行Jneg列
        DFNP2D.update_K__idxηLP_idxjneg_(self, RSEIneg, 1)

    def count_lithium(self):
        """统计锂电荷量"""
        qsneg = self.θneg*self.Qneg       # 负极固相锂电荷量 [Ah]
        qspos = self.θpos*self.Qpos       # 正极固相锂电荷量 [Ah]
        qe = (self.θe_*self.Δx_*self.qe_).sum()/3600  # 电解液锂电荷量 [Ah]
        print(f'合计锂电荷总量 {qsneg + qspos + qe + self.QLP: .6f} Ah = '
              f'负极嵌锂{qsneg:.6f} Ah + 正极嵌锂{qspos:.6f} Ah + 电解液锂{qe:.6f} Ah + 负极析锂 {self.QLP:.6f} Ah')

    def step(self, Δt):
        """时间步进：Newton法迭代所有因变量"""
        idxθsneg_ = self.idxθsneg_
        idxθspos_ = self.idxθspos_
        idxθsnegsurf_ = self.idxθsnegsurf_
        idxθspossurf_ = self.idxθspossurf_
        idxθe_ = self.idxθe_
        idxφsneg_ = self.idxφsneg_
        idxφspos_ = self.idxφspos_
        idxφe_ = self.idxφe_
        idxJintneg_ = self.idxJintneg_
        idxJintpos_ = self.idxJintpos_
        idxJDLneg_ = self.idxJDLneg_
        idxJDLpos_ = self.idxJDLpos_
        idxI0intneg_ = self.idxI0intneg_
        idxI0intpos_ = self.idxI0intpos_
        idxηintneg_ = self.idxηintneg_
        idxηintpos_ = self.idxηintpos_
        idxηLP_ = self.idxηLP_
        idxJLP_ = self.idxJLP_
        idxφ_ = self.idxφ_
        idxθ_ = self.idxθ_
        idxJ_ = self.idxJ_

        # 读取方法
        solve_banded_matrix = DFNP2D.solve_banded_matrix
        solve_Jint_ = LPP2D.solve_Jint_
        solve_dJintdηint_ = LPP2D.solve_dJintdηint_
        solve_dJintdI0int_ = LPP2D.solve_dJintdI0int_
        solve_I0int_ = LPP2D.solve_I0int_
        solve_dI0intdθssurf_ = LPP2D.solve_dI0intdθssurf_
        solve_dI0intdθe_ = LPP2D.solve_dI0intdθe_
        solve_Uneg_, solve_Upos_ = self.solve_Uneg_, self.solve_Upos_              # 读取：负极、正极开路电位函数 [V]
        solve_dUdθneg_, solve_dUdθpos_ = self.solve_dUdθneg_, self.solve_dUdθpos_  # 读取：负极、正极开路电位对嵌锂状态的偏导数函数 [V/–]

        data = self.data  # 读取：运行数据字典
        Nneg, Nsep, Npos, Ne, Nr = self.Nneg, self.Nsep, self.Npos, self.Ne, self.Nr      # 读取：网格数
        Δxneg, Δxpos, Δx_, Δr_, x_ = self.Δxneg, self.Δxpos, self.Δx_, self.Δr_, self.x_  # 读取：网格尺寸 [–]
        ΔxWest_, ΔxEast_ = self.ΔxWest_, self.ΔxEast_  # 读取：网格距离 [–]
        lithiumPlating = self.lithiumPlating           # 是否考虑析锂反应
        timeDiscretization = self.timeDiscretization   # 时间离散格式
        decouple_cs = self.decouple_cs  # 是否解耦固相锂离子浓度的求解
        verbose = self.verbose

        σneg, σpos = self.σneg, self.σpos  # 读取：负极、正极集总固相电导率 [S]
        RSEIneg, RSEIpos = self.RSEIneg, self.RSEIpos  # 读取：负极、正极集总SEI膜内阻 [Ω]
        CDLneg, CDLpos = self.CDLneg, self.CDLpos      # 读取：负极、正极集总双电层电容 [F]
        Dsneg, Dspos = self.Dsneg, self.Dspos          # 读取：负极、正极集总固相扩散系数 [1/s]
        Qneg, Qpos = self.Qneg, self.Qpos              # 读取：负极、正极容量 [Ah]

        qe_ = self.qe_  # (Ne,) 各控制体电解液锂离子电荷量 [C]
        κ_ = self.κ_    # (Ne,) 各控制体电解液集总电导率 [S]
        T, F, R = self.T, DFNP2D.F, DFNP2D.R   # 读取：温度 [K]、法拉第常数 [C/mol]、理想气体常数 [J/(mol·K)]
        Deκ_ = self.Deκ_  # # (Ne,) 各控制体电解液集总扩散系数 [A]
        F2RT = F/(2*R*T)  # 常数 [1/V]
        I = self.I        # 电流 [A]
        if I0intnegUnknown := (self._I0intneg is None):
            kneg = self.kneg          # 读取：负极集总主反应速率常数 [A]
        else:
            I0intneg = self.I0intneg  # 读取：负极集总主反应交换电流密度 [A]
        if I0intposUnknown := (self._I0intpos is None):
            kpos = self.kpos          # 读取：正极集总主反应速率常数 [A]
        else:
            I0intpos = self.I0intpos  # 读取：正极集总主反应交换电流密度 [A]
        if lithiumPlating:
            solve_JLP_ = LPP2D.solve_JLP_
            solve_dJLPdθe_ = LPP2D.solve_dJLPdθe_
            solve_dJLPdηLP_ = LPP2D.solve_dJLPdηLP_
            solve_I0LP_ = LPP2D.solve_I0LP_
            if I0LPUnknown := self._I0LP is None:
                kLP = self.kLP    # 读取：负极析锂反应速率常数 [A]
            else:
                I0LP = self.I0LP  # 读取：负极析锂反应交换电流密度 [A]

        if self.constants:
            pass
        else:
            # 更新K__矩阵的参数相关值
            if decouple_cs:
                pass
            else:
                self.update_K__idxθsnegsurf_idxJintneg_(Qneg, Dsneg)
                self.update_K__idxθspossurf_idxJintpos_(Qpos, Dspos)
            self.update_K__idxφsneg_idxJneg_(σneg)
            self.update_K__idxφspos_idxJpos_(σpos)
            self.update_K__idxφe_idxφe_(κ_, κ_)
            self.update_K__idxηintneg_idxJneg_(RSEIneg)
            self.update_K__idxηintpos_idxJpos_(RSEIpos)
            if lithiumPlating:
                self.update_K__idxηLP_idxJneg_(RSEIneg)

        κDκT_ = (self.κD*T)*κ_  # (Ne,) 各控制体电解液集总扩散离子电导率 [A]

        K__ = self.K__                # 读取：因变量线性矩阵
        bK_ = np.zeros(K__.shape[0])  # K__矩阵b向量，F_ = K__ @ X_ - bK_

        bandKθs__ = self.bandKθs__
        bandKθsneg__ = (Δt*Dsneg) * bandKθs__  # (3, Nr)
        bandKθspos__ = (Δt*Dspos) * bandKθs__  # (3, Nr)
        Kθs_Jintneg = Δt / (10800*Qneg) / ((1 - (1 - Δr_[-1])**3)/3)
        Kθs_Jintpos = Δt / (10800*Qpos) / ((1 - (1 - Δr_[-1])**3)/3)
        if timeDiscretization=='CN':
            bandKθsneg__ *= .5
            bandKθspos__ *= .5
            Kθs_Jintneg *= .5
            Kθs_Jintpos *= .5
            bandBθsneg__ = -bandKθsneg__  # (3, Nr)
            bandBθspos__ = -bandKθspos__  # (3, Nr)
            bandBθsneg__[1] += 1  # 对角元+1
            bandBθspos__[1] += 1  # 对角元+1
        bandKθsneg__[1] += 1  # 对角元+1
        bandKθspos__[1] += 1  # 对角元+1

        ## 对K__矩阵赋值 ##
        if decouple_cs:
            # 历史浓度影响的浓度分量
            match timeDiscretization:
                case 'backward':
                    RHSneg__ = self.θsneg__  # (Nr, Nneg)
                    RHSpos__ = self.θspos__  # (Nr, Npos)
                case 'CN':
                    RHSneg__ = triband_to_dense(bandBθsneg__) @ self.θsneg__  # (Nr, Nneg)
                    RHSpos__ = triband_to_dense(bandBθspos__) @ self.θspos__  # (Nr, Npos)
            e__ = self.e__
            RHSneg__ = np.concatenate([RHSneg__, e__], axis=1)  # (Nr, Nneg+1)
            RHSpos__ = np.concatenate([RHSpos__, e__], axis=1)  # (Nr, Npos+1)
            Sneg__ = dgtsv(bandKθsneg__[2, :-1], bandKθsneg__[1], bandKθsneg__[0, 1:], RHSneg__, True, True, True, True)[3]  # (Nr, Nneg+1)
            Spos__ = dgtsv(bandKθspos__[2, :-1], bandKθspos__[1], bandKθspos__[0, 1:], RHSpos__, True, True, True, True)[3]  # (Nr, Npos+1)
            θsnegI__ = Sneg__[:, :-1]  # (Nr, Nneg) 内部锂离子浓度的历史影响分量
            θsposI__ = Spos__[:, :-1]  # (Nr, Npos)
            γneg_ = Sneg__[:, -1] * -Kθs_Jintneg  # (Nr,)
            γpos_ = Spos__[:, -1] * -Kθs_Jintpos  # (Nr,)
            # 3点2次多项式外推颗粒表面锂离子浓度的历史影响分量
            # backward: θssurf_ = α_ + Jint_*β
            # CN:       θssurf_ = α_ + (Jint_ + Jintold)*β
            coeffs_ = self.coeffs_
            αneg_ = coeffs_.dot(θsnegI__[-3:])  # (Nneg,)
            αpos_ = coeffs_.dot(θsposI__[-3:])  # (Npos,)
            βneg = coeffs_.dot(γneg_[-3:])
            βpos = coeffs_.dot(γpos_[-3:])
            # 负极、正极固相表面浓度θssurf行
            K__[idxθsnegsurf_, idxJintneg_] = -βneg
            K__[idxθspossurf_, idxJintpos_] = -βpos
        else:
            # 负极、正极固相内部浓度θsneg行、θspos行
            for band__, idxθs_, Nreg in zip(
                    [bandKθsneg__, bandKθspos__], [idxθsneg_, idxθspos_], [Nneg, Npos]):
                idx__ = idxθs_.reshape(Nreg, Nr)
                K__[idx__[:, :-1].ravel(), idx__[:, 1:].ravel()] = np.tile(band__[0, 1:], Nreg)   # 上对角线
                K__[idxθs_, idxθs_]                              = np.tile(band__[1], Nreg)       # 主对角线
                K__[idx__[:, 1:].ravel(), idx__[:, :-1].ravel()] = np.tile(band__[2, :-1], Nreg)  # 下对角线
            K__[idxθsneg_[Nr-1::Nr], idxJintneg_] = Kθs_Jintneg  # Jintneg列
            K__[idxθspos_[Nr-1::Nr], idxJintpos_] = Kθs_Jintpos  # Jintpos列
        # 电解液浓度θe行θe列
        a = Deκ_[0]/ΔxEast_[0]
        K__[idxθe_[0], idxθe_[:2]] = [a, -a]  # θe列首行
        a = Deκ_[-1]/ΔxWest_[-1]
        K__[idxθe_[-1], idxθe_[-2:]] = [-a, a]  # θe列末行
        K__[idxθe_[1:-1], idxθe_[:-2]] = a_ = -Deκ_[1:-1]/ΔxWest_[1:-1]  # θe列下对角线
        K__[idxθe_[1:-1], idxθe_[2:]]  = c_ = -Deκ_[1:-1]/ΔxEast_[1:-1]  # θe列上对角线
        K__[idxθe_[1:-1], idxθe_[1:-1]] = -(a_ + c_)  # θe列主对角线
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, c = -Deκ_[nW]/ΔxWest_[nW], -2*Deκ_[nW]*Deκ_[nE]/(Deκ_[nW]*Δx_[nE] + Deκ_[nE]*Δx_[nW])
            K__[idxθe_[nW], idxθe_[nW - 1:nW + 2]] = [a, -(a + c), c]  # 界面左侧控制体
            a, c = c, -Deκ_[nE]/ΔxEast_[nE]
            K__[idxθe_[nE], idxθe_[nE - 1:nE + 2]] = [a, -(a + c), c]  # 界面右侧控制体
        Δt2Δx_ = Δt/Δx_
        K__[idxθe_[1:], idxθe_[:-1]] *= Δt2Δx_[1:]   # θe列下对角线
        K__[idxθe_[:-1], idxθe_[1:]] *= Δt2Δx_[:-1]  # θe列上对角线
        K__[idxθe_, idxθe_]          *= Δt2Δx_  # θe列主对角线
        if timeDiscretization=='CN':
            K__[idxθe_[1:], idxθe_[:-1]] *= .5
            K__[idxθe_[:-1], idxθe_[1:]] *= .5
            K__[idxθe_, idxθe_] *= .5
            start = idxθe_[0]
            end = idxθe_[-1] + 1
            Kθe__ = -K__[start:end, start:end]
            Kθe__.ravel()[::Ne+1] += qe_  # 对角元+qe_
        K__[idxθe_, idxθe_] += qe_
        # 电解液浓度θe行J列
        Kθe_j = -Δt
        if timeDiscretization=='CN':
            Kθe_j *= .5
        idxθeneg_, idxθepos_ = idxθe_[:Nneg], idxθe_[-Npos:]
        K__[idxθeneg_, idxJintneg_] = Kθe_j  # Jintneg列
        K__[idxθepos_, idxJintpos_] = Kθe_j  # Jintpos列
        if JDLnegUnknown := (idxJDLneg_.size > 0):
            K__[idxθeneg_, idxJDLneg_] = Kθe_j  # JDLneg列
        if JDLposUnknown := (idxJDLpos_.size > 0):
            K__[idxθepos_, idxJDLpos_] = Kθe_j  # JDLpos列
        if lithiumPlating:
            K__[idxθeneg_, idxJLP_] = Kθe_j  # JLP列

        # 集总双电层电流密度JDL行
        if JDLnegUnknown or JDLposUnknown:
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
            if JDLnegUnknown:
                C2Δtneg = CDLneg*c
                # 负极双电层局部体积电流密度JDLneg行
                K__[idxJDLneg_, idxφe_[:Nneg]] = C2Δtneg           # φe负极列
                K__[idxJDLneg_, idxφsneg_] = -C2Δtneg              # φsneg列
                K__[idxJDLneg_, idxJintneg_] = a = C2Δtneg*RSEIneg # Jintneg列
                K__[idxJDLneg_, idxJDLneg_] = 1 + a                # JDLneg列
                if lithiumPlating:
                    K__[idxJDLneg_, idxJLP_] = a                   # JLP列
            if JDLposUnknown:
                C2Δtpos = CDLpos*c
                # 正极双电层局部体积电流密度JDLpos行
                K__[idxJDLpos_, idxφe_[-Npos:]] = C2Δtpos          # φe正极列
                K__[idxJDLpos_, idxφspos_] = -C2Δtpos              # φspos列
                K__[idxJDLpos_, idxJintpos_] = a = C2Δtpos*RSEIpos # Jintpos列
                K__[idxJDLpos_, idxJDLpos_] = 1 + a                # JDLpos列

        # b向量（常数值、固液相浓度场旧值）
        bK_[idxφsneg_[0]]  = -Δxneg*I/σneg
        bK_[idxφspos_[-1]] =  Δxpos*I/σpos
        match timeDiscretization:
            case 'backward':
                if decouple_cs:
                    bK_[idxθsnegsurf_] = αneg_
                    bK_[idxθspossurf_] = αpos_
                else:
                    bK_[idxθsneg_] = self.θsneg__.ravel('F')
                    bK_[idxθspos_] = self.θspos__.ravel('F')
                bK_[idxθe_] = qe_*self.θe_
            case 'CN':
                if decouple_cs:
                    bK_[idxθsnegsurf_] = αneg_ + βneg*self.Jintneg_
                    bK_[idxθspossurf_] = αpos_ + βpos*self.Jintpos_
                else:
                    bK_[idxθsneg_] = (triband_to_dense(bandBθsneg__) @ self.θsneg__).ravel('F')
                    bK_[idxθspos_] = (triband_to_dense(bandBθspos__) @ self.θspos__).ravel('F')
                    bK_[idxθsneg_[Nr-1::Nr]] -= Kθs_Jintneg * self.Jintneg_
                    bK_[idxθspos_[Nr-1::Nr]] -= Kθs_Jintpos * self.Jintpos_
                bK_[idxθe_] = Kθe__.dot(self.θe_)
                bK_[idxθeneg_] -= Kθe_j * self.Jneg_
                bK_[idxθepos_] -= Kθe_j * self.Jpos_

        if JDLnegUnknown or JDLposUnknown:
            # 上一时刻负极、正极固液相电势场之差
            Δφseneg_1_ = data['ηLPneg_'][-1]
            Δφsepos_1_ = data['ηLPpos_'][-1]
            # 上上时刻
            Δφseneg_2_ = data['ηLPneg_'][-2] if (JDLnegUnknown and Nt>1) else None
            Δφsepos_2_ = data['ηLPpos_'][-2] if (JDLposUnknown and Nt>1) else None
            # 上上上时刻
            Δφseneg_3_ = data['ηLPneg_'][-3] if (JDLnegUnknown and Nt>2) else None
            Δφsepos_3_ = data['ηLPpos_'][-3] if (JDLposUnknown and Nt>2) else None
            if Nt>2:
                A = (t - t_2)*(t - t_3)/-Δt/(t_1 - t_2)/(t_1 - t_3)
                B = Δt*(t - t_3)/(t_2 - t)/(t_2 - t_1)/(t_2 - t_3)
                C = Δt*(t - t_2)/(t_3 - t)/(t_3 - t_1)/(t_3 - t_2)
                if JDLnegUnknown:
                    bK_[idxJDLneg_] = CDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_ + C*Δφseneg_3_)
                if JDLposUnknown:
                    bK_[idxJDLpos_] = CDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_ + C*Δφsepos_3_)
            elif Nt==2:
                A = (t - t_2)/(-Δt*(t_1 - t_2))
                B = Δt/((t_2 - t)*(t_2 - t_1))
                if JDLnegUnknown:
                    bK_[idxJDLneg_] = CDLneg*(A*Δφseneg_1_ + B*Δφseneg_2_)
                if JDLposUnknown:
                    bK_[idxJDLpos_] = CDLpos*(A*Δφsepos_1_ + B*Δφsepos_2_)
            else:
                if JDLnegUnknown:
                    bK_[idxJDLneg_] = -C2Δtneg*Δφseneg_1_
                if JDLposUnknown:
                    bK_[idxJDLpos_] = -C2Δtpos*Δφsepos_1_

        # 初始化解向量
        X_ = np.zeros_like(bK_)
        if decouple_cs:
            pass
        else:
            X_[idxθsneg_] = self.θsneg__.ravel('F')
            X_[idxθspos_] = self.θspos__.ravel('F')
        X_[idxθsnegsurf_] = self.θsnegsurf_
        X_[idxθspossurf_] = self.θspossurf_
        X_[idxθe_] = self.θe_
        if I0intnegUnknown:
            X_[idxI0intneg_] = self.I0intneg_
        if I0intposUnknown:
            X_[idxI0intpos_] = self.I0intpos_
        if I==data['I'][-1]:
            # 恒电流
            X_[idxφsneg_] = self.φsneg_
            X_[idxφspos_] = self.φspos_
            X_[idxφe_] = self.φe_
            X_[idxJintneg_] = self.Jintneg_
            X_[idxJintpos_] = self.Jintpos_
            X_[idxηintneg_] = self.ηintneg_
            X_[idxηintpos_] = self.ηintpos_
        else:
            # 变电流瞬间
            X_[idxφe_] = 0
            Jintneg = I
            Jintpos = -I
            X_[idxJintneg_] = Jintneg
            X_[idxJintpos_] = Jintpos
            I0intneg_ = X_[idxI0intneg_] if I0intnegUnknown else I0intneg
            I0intpos_ = X_[idxI0intpos_] if I0intposUnknown else I0intpos
            X_[idxηintneg_] = np.arcsinh(Jintneg/(2*I0intneg_))/F2RT
            X_[idxηintpos_] = np.arcsinh(Jintpos/(2*I0intpos_))/F2RT
            X_[idxφsneg_] = X_[idxηintneg_] + RSEIneg*Jintneg + solve_Uneg_(X_[idxθsnegsurf_])
            X_[idxφspos_] = X_[idxηintpos_] + RSEIpos*Jintpos + solve_Upos_(X_[idxθspossurf_])

        if lithiumPlating:
            X_[idxηLP_] = X_[idxφsneg_] - X_[idxφe_[:Nneg]] - RSEIneg*X_[idxJintneg_]

        J__ = K__.copy()  # 初始化Jacobi矩阵
        ΔFφe_ = np.zeros(Ne)
        for nNewton in range(1, 201):
            ## Newton迭代
            F_ = K__.dot(X_) - bK_  # F残差向量的线性部分

            # 提取解
            θsnegsurf_, θspossurf_ = X_[idxθsnegsurf_], X_[idxθspossurf_]
            θe_ = X_[idxθe_]
            θeneg_, θepos_ = θe_[:Nneg], θe_[-Npos:]
            I0intneg_ = X_[idxI0intneg_] if I0intnegUnknown else I0intneg
            I0intpos_ = X_[idxI0intpos_] if I0intposUnknown else I0intpos
            ηintneg_, ηintpos_ = X_[idxηintneg_], X_[idxηintpos_]

            # F向量非线性部分
            ΔFφe_[0]  = -κDκT_[0]  * (θe_[1] - θe_[0]  )/ΔxEast_[0] / (0.5*(θe_[1] + θe_[0]))
            ΔFφe_[-1] =  κDκT_[-1] * (θe_[-1] - θe_[-2])/ΔxWest_[-1] / (0.5*(θe_[-1] + θe_[-2]))
            ΔFφe_[1:-1] = -κDκT_[1:-1]*( (θe_[2:] - θe_[1:-1] )/ΔxEast_[1:-1] / (0.5*(θe_[2:] + θe_[1:-1]))
                                        -(θe_[1:-1] - θe_[:-2])/ΔxWest_[1:-1] / (0.5*(θe_[1:-1] + θe_[:-2])) )
            for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
                # 修正负极-隔膜界面、隔膜-正极界面
                a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
                θinterface = (a*θe_[nE] + b*θe_[nW])/(a + b)
                ΔFφe_[nW] = -( κDκT_[nW] * (θinterface - θe_[nW])/(0.5*Δx_[nW]) / θinterface
                              -κDκT_[nW] * (θe_[nW] - θe_[nW-1])/ΔxWest_[nW]  / (0.5*(θe_[nW] + θe_[nW-1])) )
                ΔFφe_[nE] = -( κDκT_[nE] * (θe_[nE+1] - θe_[nE] )/ΔxEast_[nE] / (0.5*(θe_[nE+1] + θe_[nE]))
                              -κDκT_[nE] * (θe_[nE] - θinterface)/(0.5*Δx_[nE]) / θinterface )
            F_[idxφe_] += ΔFφe_
            F_[idxJintneg_] -= solve_Jint_(T, I0intneg_, ηintneg_)  # F向量Jintneg部分
            F_[idxJintpos_] -= solve_Jint_(T, I0intpos_, ηintpos_)  # F向量Jintpos部分
            if I0intnegUnknown:
                F_[idxI0intneg_] -= solve_I0int_(kneg, θsnegsurf_, θeneg_)  # F向量I0intneg部分
            if I0intposUnknown:
                F_[idxI0intpos_] -= solve_I0int_(kpos, θspossurf_, θepos_)  # F向量I0intpos部分
            F_[idxηintneg_] += solve_Uneg_(θsnegsurf_)  # F向量ηintneg非线性部分
            F_[idxηintpos_] += solve_Upos_(θspossurf_)  # F向量ηintpos非线性部分
            if lithiumPlating:
                ηLP_ = X_[idxηLP_]
                I0LP_ = solve_I0LP_(kLP, θeneg_) if I0LPUnknown else I0LP  # 负极析锂反应的交换电流场 [A]
                F_[idxJLP_] -= solve_JLP_(T, I0LP_, ηLP_)   # F向量JLP部分
            # 更新Jacobi矩阵非线性部分
            # φe行θe列
            a = κDκT_[0] / (0.5*(θe_[1] + θe_[0]) * ΔxEast_[0])
            aa = a * (θe_[1] - θe_[0]) / (θe_[1] + θe_[0])
            J__[idxφe_[0], idxθe_[:2]] = [aa + a, aa - a]      # θe首行起始2列

            a = κDκT_[-1] / (0.5*(θe_[-1] + θe_[-2]) * ΔxWest_[-1])
            aa = a * (θe_[-1] - θe_[-2]) / (θe_[-1] + θe_[-2])
            J__[idxφe_[-1], idxθe_[-2:]] = [-aa - a, -aa + a]  # θe末行末尾2列

            a_ = κDκT_[1:-1] / (0.5*(θe_[1:-1] + θe_[:-2]) * ΔxWest_[1:-1])
            aa_ = a_ * (θe_[1:-1] - θe_[:-2]) / (θe_[1:-1] + θe_[:-2])
            c_ = κDκT_[1:-1] / (0.5*(θe_[1:-1] + θe_[2:]) * ΔxEast_[1:-1])
            cc_ = c_ * (θe_[2:] - θe_[1:-1]) / (θe_[2:] + θe_[1:-1])
            J__[idxφe_[1:-1], idxθe_[:-2]] = - aa_ - a_            # 下对角线
            J__[idxφe_[1:-1], idxθe_[2:]] = cc_ - c_               # 上对角线
            J__[idxφe_[1:-1], idxθe_[1:-1]] = cc_ + c_ - aa_ + a_  # 主对角线
            for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
                # 修正负极-隔膜界面、隔膜-正极界面
                num = κDκT_[nW]*κ_[nE] - κDκT_[nE]*κ_[nW]
                den1 = κ_[nE]*Δx_[nW] + κ_[nW]*Δx_[nE]
                den2 = κ_[nE]*θe_[nE]*Δx_[nW] + κ_[nW]*θe_[nW]*Δx_[nE]
                product = den1*den2

                a = 2*κDκT_[nW] / ((θe_[nW] + θe_[nW-1]) * Δx_[nW])
                aa = a * (θe_[nW] - θe_[nW-1]) / (θe_[nW] + θe_[nW-1])
                c = 2*Δx_[nE]*κ_[nW]*num / product
                cc = c * (θe_[nE] - θe_[nW])*κ_[nE]*Δx_[nW] / den2
                d = 2*κDκT_[nW]*κ_[nE] / den2
                dd = d * (θe_[nE] - θe_[nW])*κ_[nE]*Δx_[nW] / den2
                J__[idxφe_[nW], idxθe_[nW-1:nW+2]] = [
                    -a - aa,
                    -c - cc/Δx_[nW]*Δx_[nE]/κ_[nE]*κ_[nW] + d + dd/Δx_[nW]*Δx_[nE]/κ_[nE]*κ_[nW] + a - aa,
                    c - cc - d + dd]  # 界面左侧控制体

                a = 2*κ_[nE]*Δx_[nW]*num / product
                aa = a * (θe_[nE] - θe_[nW])*κ_[nW]*Δx_[nE] / den2
                c = 2*κDκT_[nE]*κ_[nW] / den2
                cc = c * (θe_[nE] - θe_[nW])*κ_[nW]*Δx_[nE] / den2
                d = 2*κDκT_[nE] / ((θe_[nE] + θe_[nE+1]) * Δx_[nE])
                dd = d * (θe_[nE] - θe_[nE+1]) / (θe_[nE] + θe_[nE+1])
                J__[idxφe_[nE], idxθe_[nE-1:nE+2]] = [
                    -a - aa - c - cc,
                     a - aa/Δx_[nE]*Δx_[nW]/κ_[nW]*κ_[nE] + c - cc/Δx_[nE]*Δx_[nW]/κ_[nW]*κ_[nE] + d - dd,
                    -d - dd]  # # 界面右侧

            J__[idxJintneg_, idxηintneg_]  = -solve_dJintdηint_(T, I0intneg_, ηintneg_)  # ∂FJintneg/∂ηintneg
            J__[idxJintpos_, idxηintpos_]  = -solve_dJintdηint_(T, I0intpos_, ηintpos_)  # ∂FJintpos/∂ηintpos
            if I0intnegUnknown:
                J__[idxJintneg_, idxI0intneg_] = -solve_dJintdI0int_(T, ηintneg_)        # ∂FJintneg/∂I0intneg
                J__[idxI0intneg_, idxθe_[:Nneg]] = -solve_dI0intdθe_(θeneg_, I0intneg_)  # ∂FI0intneg/∂θe
                J__[idxI0intneg_, idxθsnegsurf_] = -solve_dI0intdθssurf_(kneg, θsnegsurf_, θeneg_, I0intneg_)  # ∂FI0intneg/∂θsnegsurf
            if I0intposUnknown:
                J__[idxJintpos_, idxI0intpos_] = -solve_dJintdI0int_(T, ηintpos_)         # ∂FJintpos/∂I0intpos
                J__[idxI0intpos_, idxθe_[-Npos:]] = -solve_dI0intdθe_(θepos_, I0intpos_)  # ∂FI0intpos/∂θe
                J__[idxI0intpos_, idxθspossurf_] = -solve_dI0intdθssurf_(kpos, θspossurf_, θepos_, I0intpos_)  # ∂FI0intpos/∂θspossurf
            J__[idxηintneg_, idxθsnegsurf_] = solve_dUdθneg_(θsnegsurf_)  # ∂Fηintneg/∂θsnegsurf
            J__[idxηintpos_, idxθspossurf_] = solve_dUdθpos_(θspossurf_)  # ∂Fηintpos/∂θspossurf
            if lithiumPlating:
                J__[idxJLP_, idxθe_[:Nneg]] = -solve_dJLPdθe_(T, θeneg_, I0LP_, ηLP_)  # ∂FJLP/∂ce
                J__[idxJLP_, idxηLP_] = -solve_dJLPdηLP_(T, I0LP_, ηLP_)               # ∂FJLP/∂ce

            if self.bandwidthsJ_ is None and any(data['I']):
                if verbose:
                    print('辨识重排因变量Jacobi矩阵的带宽 -> ', end='')
                self.idxJreordered_ = idxJreordered_ = reverse_cuthill_mckee(csr_matrix(J__))
                self.idxJrecovered_ = np.argsort(idxJreordered_)
                self.bandwidthsJ_ = DFNP2D.identify_bandwidths(J__[np.ix_(idxJreordered_, idxJreordered_)])
                if verbose:
                    print(f'上带宽{self.bandwidthsJ_['upper']}，下带宽{self.bandwidthsJ_['lower']}')

            # Newton迭代新解向量
            if bandwidthsJ_ := self.bandwidthsJ_:
                # 带状化求解
                ΔX_ = solve_banded_matrix(J__, F_,
                    self.idxJreordered_, self.idxJrecovered_, bandwidthsJ_)
            else:
                # 直接求解
                ΔX_ = np.linalg.solve(J__, F_)

            X_ -= ΔX_

            if np.isnan(X_).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现nan')
            if (X_[idxθe_]<=0).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θe<=0')
            θsnegsurf_ = X_[idxθsnegsurf_]
            if (θsnegsurf_<=0).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θsnegsurf<=0')
            if (θsnegsurf_>=1).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θsnegsurf>=1')
            θspossurf_ = X_[idxθspossurf_]
            if (θspossurf_<=0).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θspossurf<=0')
            if (θspossurf_>=1).any():
                raise DFNP2D.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θspossurf>=1')

            ΔX_ = abs(ΔX_)
            maxΔφ = ΔX_[idxφ_].max()  # 新旧电势场最大绝对误差
            maxΔθ = ΔX_[idxθ_].max()  # 新旧浓度场最大绝对误差
            maxΔj = ΔX_[idxJ_].max()  # 新旧局部体积电流密度场最大绝对误差
            # print(f'新旧局部体积电流场最大绝对误差{maxΔj:.6f} A，'
            #       f'新旧浓度场最大绝对误差{maxΔθ:.6f}，'
            #       f'新旧电势场最大绝对误差{maxΔφ*1e3:.6f} mV' )
            if maxΔj/(abs(I)+0.001)<1e-3 and maxΔθ<1e-3 and maxΔφ<1e-3:
                break
        else:
            if verbose:
                print(f'时刻t = {self.t}s，Newton迭代达到最大次数{nNewton}，'
                      f'{maxΔφ*1e3 = :.6f} V，'
                      f'{maxΔθ = :.4f}，'
                      f'{maxΔj = :.3f} A')

        Jintneg_ = X_[idxJintneg_]
        Jintpos_ = X_[idxJintpos_]
        if decouple_cs:
            outer = np.outer
            match timeDiscretization:
                case 'CN':
                    self.θsneg__ = θsnegI__ + outer(γneg_, Jintneg_ + self.Jintneg_)
                    self.θspos__ = θsposI__ + outer(γpos_, Jintpos_ + self.Jintpos_)
                case 'backward':
                    self.θsneg__ = θsnegI__ + outer(γneg_, Jintneg_)
                    self.θspos__ = θsposI__ + outer(γpos_, Jintpos_)
        else:
            self.θsneg__ = X_[idxθsneg_].reshape(Nr, Nneg, order='F')
            self.θspos__ = X_[idxθspos_].reshape(Nr, Npos, order='F')
        self.θe_ = X_[idxθe_]
        self.φsneg_ = X_[idxφsneg_]
        self.φspos_ = X_[idxφspos_]
        self.φe_ = X_[idxφe_]
        self.Jintneg_ = Jintneg_
        self.Jintpos_ = Jintpos_
        if JDLnegUnknown:
            self.JDLneg_ = X_[idxJDLneg_]
        if JDLposUnknown:
            self.JDLpos_ = X_[idxJDLpos_]
        if lithiumPlating:
            self.JLP_ = X_[idxJLP_]
        return nNewton  # 返回Newton迭代次数

    @property
    def I0intneg(self):
        """负极主反应交换电流 [A]"""
        if self.constants or (T := self.T)==self.Tref:
            return self._I0intneg
        else:
            return self.Arrhenius(_I0intneg, self.Ekneg, T) if (_I0intneg:=self._I0intneg) else _I0intneg
    @I0intneg.setter
    def I0intneg(self, I0intneg):
        self._I0intneg = I0intneg

    @property
    def I0intpos(self):
        """正极主反应交换电流 [A]"""
        if self.constants or (T := self.T)==self.Tref:
            return self._I0intpos
        else:
            return self.Arrhenius(_I0intpos, self.Ekpos, T) if (_I0intpos:=self._I0intpos) else _I0intpos
    @I0intpos.setter
    def I0intpos(self, I0intpos):
        self._I0intpos = I0intpos

    @property
    def I0LP(self):
        """负极析锂反应交换电流 [A]"""
        if self.constants or (T := self.T)==self.Tref:
            return self._I0LP
        else:
            return self.Arrhenius(_I0LP, self.EkLP, T) if (_I0LP:=self._I0LP) else _I0LP
    @I0LP.setter
    def I0LP(self, I0LP):
        self._I0LP = I0LP

    @property
    def κneg(self):
        """负极电解液集总离子电导率 [S]"""
        if self.constants or (T := self.T)==self.Tref:
            return self._κneg
        else:
            return self.Arrhenius(self._κneg, self.Eκ, T)
    @κneg.setter
    def κneg(self, κneg):
        self._κneg = κneg

    @property
    def κsep(self):
        """隔膜电解液集总离子电导率 [S]"""
        if self.constants or (T := self.T)==self.Tref:
            return self._κsep
        else:
            return self.Arrhenius(self._κsep, self.Eκ, T)
    @κsep.setter
    def κsep(self, κsep):
        self._κsep = κsep

    @property
    def κpos(self):
        """正极电解液集总离子电导率 [S]"""
        if self.constants or (T := self.T)==self.Tref:
            return self._κpos
        else:
            return self.Arrhenius(self._κpos, self.Eκ, T)
    @κpos.setter
    def κpos(self, κpos):
        self._κpos = κpos

    @property
    def κ_(self):
        """(Ne,) 各控制体集总电解液离子电导率 [S]"""
        full = np.full
        return np.concatenate([
            full(self.Nneg, self.κneg),
            full(self.Nsep, self.κsep),
            full(self.Npos, self.κpos)])

    @property
    def De(self):
        """集总电解液离子扩散率/电导率之比 [A/S]"""
        return self._De
    @De.setter
    def De(self, De):
        self._De = De

    @property
    def Deκ_(self):
        """(Ne,) 各控制体集总电解液锂离子扩散系数 [A]"""
        De3κ_ = self.De * np.array([self._κneg, self._κsep, self._κpos])
        if self.constants or (T := self.T)==self.Tref:
            pass
        else:
            De3κ_ = self.Arrhenius(De3κ_, self.EDe, T)
        return np.concatenate([
            np.full(self.Nneg, De3κ_[0]),
            np.full(self.Nsep, De3κ_[1]),
            np.full(self.Npos, De3κ_[2])])

    @property
    def Qcell(self):
        """读取全电池理论可用容量"""
        return self._Qcell
    @Qcell.setter
    def Qcell(self, Qcell):
        """赋值全电池理论可用容量"""
        self._Qcell = Qcell

    @property
    def Qneg(self):
        """负极容量"""
        return self._Qneg
    @Qneg.setter
    def Qneg(self, Qneg):
        """赋值负极容量"""
        self._Qneg = Qneg

    @property
    def Qpos(self):
        """正极容量"""
        return self._Qpos
    @Qpos.setter
    def Qpos(self, Qpos):
        """赋值正极容量"""
        self._Qpos = Qpos

    @property
    def qe_(self):
        full = np.full
        return np.concatenate([
            full(self.Nneg, self.qeneg),
            full(self.Nsep, self.qesep),
            full(self.Npos, self.qepos),
            ])  # (Ne,) 各控制体电解液锂离子电荷量 [C]

    @property
    def U(self):
        """正负极端电压 [V]"""
        a = 0.5*self.I
        φsposCollector = self.φspos_[-1] - a*self.Δxpos/self.σpos
        φsnegCollector = self.φsneg_[0]  + a*self.Δxneg/self.σneg
        return φsposCollector - φsnegCollector

    @property
    def θsnegsurf_(self):
        """(Nneg,) 负极固相表面无量纲锂离子浓度场 [–]"""
        if self.decouple_cs:
            return self.coeffs_.dot(self.θsneg__[-3:])
        else:
            Nr = self.Nr
            idxθsnegsurf_, idxθsneg_ = self.idxθsnegsurf_, self.idxθsneg_
            θsneg_ = self.θsneg__.ravel('F')
            K__ = self.K__
            return -(K__[idxθsnegsurf_, idxθsneg_[Nr-3::Nr]] * θsneg_[Nr-3::Nr]
                    + K__[idxθsnegsurf_, idxθsneg_[Nr-2::Nr]] * θsneg_[Nr-2::Nr]
                    + K__[idxθsnegsurf_, idxθsneg_[Nr-1::Nr]] * θsneg_[Nr-1::Nr]
                    + K__[idxθsnegsurf_, self.idxJintneg_] * self.Jintneg_)/K__[idxθsnegsurf_, idxθsnegsurf_]

    @property
    def θspossurf_(self):
        """(Npos,) 正极固相表面无量纲锂离子浓度场 [–]"""
        if self.decouple_cs:
            return self.coeffs_.dot(self.θspos__[-3:])
        else:
            Nr = self.Nr
            idxθspossurf_, idxθspos_ = self.idxθspossurf_, self.idxθspos_
            θspos_ = self.θspos__.ravel('F')
            K__ = self.K__
            return -(K__[idxθspossurf_, idxθspos_[Nr-3::Nr]] * θspos_[Nr-3::Nr]
                   + K__[idxθspossurf_, idxθspos_[Nr-2::Nr]] * θspos_[Nr-2::Nr]
                   + K__[idxθspossurf_, idxθspos_[Nr-1::Nr]] * θspos_[Nr-1::Nr]
                   + K__[idxθspossurf_, self.idxJintpos_] * self.Jintpos_)/K__[idxθspossurf_, idxθspossurf_]

    @property
    def θeneg_(self):
        """(Nneg,) 负极区域电解液无量纲锂离子浓度 [–]"""
        return self.θe_[:self.Nneg]

    @property
    def θesep_(self):
        """(Nsep,) 隔膜区域电解液无量纲锂离子浓度 [–]"""
        return self.θe_[self.Nneg:-self.Npos]

    @property
    def θepos_(self):
        """(Npos,) 正极区域电解液无量纲锂离子浓度 [–]"""
        return self.θe_[-self.Npos:]

    @property
    def θeInterfaces_(self):
        """(Ne+1,) 各控制体界面的无量纲锂离子浓度 [–]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_ = self.Δx_
        θe_ = self.θe_
        κ_ = self.κ_
        θeInterfaces_ = np.hstack([θe_[0], (θe_[:-1] + θe_[1:])/2, θe_[-1]])  # 各控制体界面的锂离子浓度
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            θeInterfaces_[nW + 1] = (a*θe_[nE] + b*θe_[nW])/(a + b)
        return θeInterfaces_

    @property
    def φeInterfaces_(self):
        """(Ne+1,) 各控制体界面的电解液电势 [V]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        Δx_, ΔxWest_, ΔxEast_ = self.Δx_, self.ΔxWest_, self.ΔxEast_
        φe_, θe_ = self.φe_, self.θe_
        θeInterfaces_ = self.θeInterfaces_
        θeWest_ = θeInterfaces_[:-1]  # 各控制体左界面的电解液锂离子浓度
        θeEast_ = θeInterfaces_[1:]  # 各控制体右界面的电解液锂离子浓度
        gradθeWest_ = np.hstack([0, (θe_[1:] - θe_[:-1])/ΔxWest_[1:]])  # 各控制体左界面的锂离子浓度梯度
        gradθeEast_ = np.hstack([(θe_[1:] - θe_[:-1])/ΔxEast_[:-1], 0])  # 各控制体右界面的锂离子浓度梯度
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、修正隔膜-正极界面
            gradθeEast_[nW] = (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW])
            gradθeWest_[nE] = (θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE])
        gradlnθeWest_ = gradθeWest_/θeWest_  # 各控制体左界面的对数锂离子浓度梯度 [ln –/–]
        gradlnθeEast_ = gradθeEast_/θeEast_  # 各控制体右界面的对数锂离子浓度梯度 [ln –/–]
        φeInterfaces_ = np.hstack([φe_[0], (φe_[:-1] + φe_[1:])/2, φe_[-1]])
        κ_ = self.κ_
        κDκT_ = (self.κD*self.T) * κ_
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = κ_[nW]*Δx_[nE], κ_[nE]*Δx_[nW]
            c = 0.5*Δx_[nW]*Δx_[nE]
            φeInterfaces_[nE] = (  a*φe_[nW] + b*φe_[nE]
                                 + c*κDκT_[nW]*gradlnθeEast_[nW]
                                 - c*κDκT_[nE]*gradlnθeWest_[nE]
                                 )/(a + b)
        return φeInterfaces_

    @property
    def Jneg_(self):
        """负极总反应局部体积电流密度场 [V]"""
        Jneg_ = self.Jintneg_.copy()
        if self.idxJDLneg_.size: Jneg_ += self.JDLneg_
        if    self.idxJLP_.size: Jneg_ += self.JLP_
        return Jneg_

    @property
    def Jpos_(self):
        """正极总反应局部体积电流密度场 [V]"""
        Jpos_ = self.Jintpos_.copy()
        if self.idxJDLpos_.size: Jpos_ += self.JDLpos_
        return Jpos_

    @staticmethod
    def solve_Jint_(T, I0int_, ηint_) -> np.ndarray:
        """求解主反应局部体积电流密度Jint [A]"""
        return 2*I0int_*np.sinh(DFNP2D.F/(2*DFNP2D.R*T) * ηint_)

    @property
    def dJintdI0intneg_(self):
        """负极主反应局部体积电流密度Jintneg对交换电流密度I0intneg的偏导数 [A/A]"""
        return LPP2D.solve_dJintdI0int_(self.T, self.ηintneg_)

    @property
    def dJintdI0intpos_(self):
        """正极主反应局部体积电流密度Jintpos对交换电流密度I0pos的偏导数 [A/A]"""
        return LPP2D.solve_dJintdI0int_(self.T, self.ηintpos_)

    @staticmethod
    def solve_dJintdI0int_(T, ηint_) -> np.ndarray:
        """求解主反应局部体积电流密度Jint对交换电流密度I0int的偏导数 [A/A]"""
        return 2*np.sinh(DFNP2D.F/(2*DFNP2D.R*T) * ηint_)

    @property
    def dJintdηintneg_(self):
        """负极主反应局部体积电流密度Jintneg对过电位ηintneg的偏导数 [A/V]"""
        return LPP2D.solve_dJintdηint_(self.T, self.I0intneg_, self.ηintneg_)

    @property
    def dJintdηintpos_(self):
        """正极主反应局部体积电流密度Jintpos对过电位ηintpos的偏导数 [A/V]"""
        return LPP2D.solve_dJintdηint_(self.T, self.I0intpos_, self.ηintpos_)

    @staticmethod
    def solve_dJintdηint_(T, I0int_, ηint_) -> np.ndarray:
        """求解主反应局部体积电流密度Jint对过电位ηint的偏导数 [A/V]"""
        FRT = DFNP2D.F / (DFNP2D.R*T)
        return FRT*I0int_*np.cosh(FRT*0.5*ηint_)

    @property
    def I0intneg_(self):
        """负极主反应交换电流密度场 [A]"""
        if self._I0intneg:
            return np.full(self.Nneg, self.I0intneg)
        else:
            return LPP2D.solve_I0int_(self.kneg, self.θsnegsurf_, self.θeneg_)

    @property
    def I0intpos_(self):
        """正极主反应交换电流密度场 [A]"""
        if self._I0intpos:
            return np.full(self.Npos, self.I0intpos)
        else:
            return LPP2D.solve_I0int_(self.kpos, self.θspossurf_, self.θepos_)

    @staticmethod
    def solve_I0int_(k, θssurf_, θe_) -> np.ndarray:
        """由固液相浓度场求主反应交换电流密度I0int [A]"""
        return k * np.sqrt(θe_*(1 - θssurf_)*θssurf_)

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

    @staticmethod
    def solve_dI0intdθe_(θe_, I0int_):
        """求解主反应交换电流密度I0int对电解液无量纲锂离子浓度θe的偏导数  [A/-]"""
        return 0.5*I0int_/θe_

    @property
    def dI0intdθsnegsurf_(self):
        """负极主反应交换电流密度I0intneg对电极表面浓度的偏导数 [A/-]"""
        return 0  if self._I0intneg\
            else self.solve_dI0intdθssurf_(self.kneg,  self.θsnegsurf_, self.θeneg_, self.I0intneg_)

    @property
    def dI0intdθspossurf_(self):
        """正极主反应交换电流密度I0intpos对电极表面浓度的偏导数 [A/-]"""
        return 0 if self._I0intpos\
            else self.solve_dI0intdθssurf_(self.kpos,  self.θspossurf_, self.θepos_, self.I0intpos_)

    @staticmethod
    def solve_dI0intdθssurf_(k, θssurf_, θe_, I0int_):
        """求解主反应交换电流密度I0int对固相颗粒表面无量纲锂离子浓度θssurf的偏导数 [A/-]"""
        return k*k * θe_*(0.5 - θssurf_)/I0int_

    @property
    def ηLPneg_(self):
        """负极析锂反应过电位场 [V]"""
        return self.φsneg_ - self.φeneg_ - self.RSEIneg*self.Jneg_

    @property
    def ηLPpos_(self):
        """正极析锂反应过电位场 [V]"""
        return self.φspos_ - self.φepos_ - self.RSEIpos*self.Jpos_

    @staticmethod
    def solve_JLP_(T, I0LP_, ηLP_) -> np.ndarray:
        """求解析锂反应局部体积电流密度JLP [A]"""
        FRT = DFNP2D.F/DFNP2D.R/T
        a, b = 0.3*FRT, -0.7*FRT
        JLP_ = I0LP_*(np.exp(a*ηLP_) - np.exp(b*ηLP_))
        JLP_[ηLP_>=0] = 0
        return JLP_

    @property
    def dJLPdθe_(self):
        """析锂反应局部体积电流密度JLP对电解液浓度θe的偏导数 [A/-]"""
        return 0 if self._I0LP \
            else LPP2D.solve_dJLPdθe_(self.T, self.θeneg_, self.I0LP_, self.ηLPneg_)

    @staticmethod
    def solve_dJLPdθe_(T, θeneg_, I0LP_, ηLP_):
        """析锂反应局部体积电流密度JLP对电解液锂离子浓度θe的偏导数"""
        FRT = DFNP2D.F/DFNP2D.R/T
        a, b = 0.3*FRT, -0.7*FRT
        dJLPdI0LP_ = np.exp(a*ηLP_) - np.exp(b*ηLP_)
        dI0LPdθe_ = 0.3*I0LP_/θeneg_
        dJLPdθe_ = dJLPdI0LP_*dI0LPdθe_
        dJLPdθe_[ηLP_>=0] = 0
        return dJLPdθe_

    @property
    def dJLPdηLP_(self):
        """析锂反应局部体积电流密度JLP对析锂过电位ηLP的偏导数 [A/V]"""
        return LPP2D.solve_dJLPdηLP_(self.T, self.I0LP_, self.ηLPneg_)

    @staticmethod
    def solve_dJLPdηLP_(T, I0LP_, ηLP_):
        """求解析锂反应局部体积电流密度JLP对析锂过电位ηLP的偏导数 [A/V]"""
        FRT = DFNP2D.F / (DFNP2D.R*T)
        a, b = 0.3*FRT, -0.7*FRT
        exp = np.exp
        dJLPdηLP_ = I0LP_*(a*exp(a*ηLP_) - b*exp(b*ηLP_))
        dJLPdηLP_[ηLP_>=0] = 0
        return dJLPdηLP_

    @property
    def I0LP_(self):
        """负极析锂反应交换电流密度场 [A/m^2]"""
        if self._I0LP:
            return np.full(self.Nneg, self.I0LP)
        else:
            return LPP2D.solve_I0LP_(self.kLP, self.θeneg_)

    @staticmethod
    def solve_I0LP_(kLP, θeneg_) -> np.ndarray:
        """求解析锂反应交换电流密度I0LP [A]"""
        return kLP * θeneg_**0.3

    @property
    def gradθe_(self):
        """(Ne,) 电解液锂离子浓度场的梯度∂θe/∂x [–/–]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        x_, Δx_, ΔxWest_, ΔxEast_ = self.x_, self.Δx_, self.ΔxWest_, self.ΔxEast_
        θe_ = self.θe_
        θeInterfaces_ = self.θeInterfaces_
        θeWest_ = θeInterfaces_[:-1]  # 各控制体左界面的电解液锂离子浓度
        θeEast_ = θeInterfaces_[1:]   # 各控制体右界面的电解液锂离子浓度
        gradθe_ = np.hstack([
            (0 + (θe_[1] - θe_[0])/(x_[1] - x_[0]))/2,  # 负极首个控制体
            (θe_[2:] - θe_[:-2])/(x_[2:] - x_[:-2]),  # 内部控制体
            ((θe_[-1] - θe_[-2])/(x_[-1] - x_[-2]) + 0)/2])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradθe_[nW] = ((θe_[nW] - θe_[nW - 1])/ΔxWest_[nW] + (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW]))/2  # 界面左侧控制体
            gradθe_[nE] = ((θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE]) + (θe_[nE + 1] - θe_[nE])/ΔxEast_[nE])/2  # 界面右侧控制体
        return gradθe_

    @property
    def gradlnθe_(self):
        """对数电解液锂离子浓度场的梯度 [(ln –)/–]"""
        return self.gradθe_/self.θe_

    @property
    def gradφsneg_(self):
        """负极固相电势场的梯度 [V/m]"""
        φsneg_ = self.φsneg_
        Δxneg = self.Δxneg
        gradφsneg_ = np.hstack([
            (-self.I/self.σneg + (φsneg_[1] - φsneg_[0])/Δxneg)/2, # 负极首个控制体
            (φsneg_[2:] - φsneg_[:-2])/(2*Δxneg),      # 负极内部控制体
            ((φsneg_[-1] - φsneg_[-2])/Δxneg + 0)/2])  # 负极末尾控制体
        return gradφsneg_

    @property
    def gradφspos_(self):
        """正极固相电势场的梯度 [V/m]"""
        φspos_ = self.φspos_
        Δxpos = self.Δxpos
        gradφspos_ = np.hstack([
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
        gradφe_ = np.hstack([
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
        return self.JDLneg_.mean()

    @property
    def IDLpos(self):
        """正极双电层电流 [A]"""
        return self.JDLpos_.mean()

    @property
    def ILP(self):
        """析锂电流 [A]"""
        return self.JLP_.mean()

    @property
    def Qohme(self):
        """电解液欧姆热 [W]"""
        gradφe_ = self.gradφe_
        return (self.κ_*(gradφe_ + self.κD*self.T*self.gradlnθe_)*gradφe_*self.Δx_).sum()

    @property
    def Qohmneg(self):
        """负极固相欧姆热 [W]"""
        return self.σneg*(self.gradφsneg_**2).mean()

    @property
    def Qohmpos(self):
        """正极固相欧姆热 [W]"""
        return self.σpos*(self.gradφspos_**2).mean()

    @property
    def Qrxnneg(self):
        """负极反应热 [W]"""
        return (self.Jintneg_*self.ηintneg_).mean()

    @property
    def Qrxnpos(self):
        """正极反应热 [W]"""
        return (self.Jintpos_*self.ηintpos_).mean()

    @property
    def Qrevneg(self):
        """负极可逆热 [W]"""
        return 0. if self.dUdTnegsurf_==0 else (self.Jintneg_*self.T*self.dUdTnegsurf_).mean()

    @property
    def Qrevpos(self):
        """正极可逆热 [W]"""
        return 0. if self.dUdTpossurf_==0 else (self.Jintpos_*self.T*self.dUdTpossurf_).mean()

    @property
    def dUdTnegsurf_(self):
        """负极熵热系数场 [V/K]"""
        return self.solve_dUdTneg_(self.θsnegsurf_) if callable(self.solve_dUdTneg_) else self.solve_dUdTneg_

    @property
    def dUdTpossurf_(self):
        """正极熵热系数场 [V/K]"""
        return self.solve_dUdTpos_(self.θspossurf_) if callable(self.solve_dUdTpos_) else self.solve_dUdTpos_

    @property
    def Uneg_(self):
        """负极表面电位场 [V]"""
        Uneg_ = self.solve_Uneg_(self.θsnegsurf_)
        if self.T==self.Tref:
            return Uneg_
        else:
            return Uneg_ + (self.T - self.Tref)*self.dUdTnegsurf_

    @property
    def Upos_(self):
        """正极表面电位场 [V]"""
        Upos_ = self.solve_Upos_(self.θspossurf_)
        if self.T==self.Tref:
            return Upos_
        else:
            return Upos_ + (self.T - self.Tref)*self.dUdTpossurf_

    @property
    def dUdθnegsurf_(self):
        """负极电位对负极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUdθneg_(self.θsnegsurf_)

    @property
    def dUdθpossurf_(self):
        """正极电位对正极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUdθpos_(self.θspossurf_)

    @property
    def θneg(self):
        """负极嵌锂状态"""
        return self.Vr_.dot(self.θsneg__).mean()

    @property
    def θpos(self):
        """正极嵌锂状态"""
        return self.Vr_.dot(self.θspos__).mean()

    @property
    def xPlot_(self):
        """全区域控制体中心的坐标坐标（用于作图） [–]"""
        return self.x_

    @property
    def xInterfacesPlot_(self):
        """各控制体交界面的坐标（用于作图） [–]"""
        return self.xInterfaces_

    def plot_θ(self, *arg, **kwargs):
        """作图：浓度场"""
        self.plot_c(*arg, **kwargs)

    def plot_Jint(self, *arg, **kwargs):
        """作图：主反应局部体积电流密度、过电位、交换电流密度"""
        self.plot_jint(*arg, **kwargs)

    def plot_JDL(self, *arg, **kwargs):
        """作图：双电层效应局部体积电流密度、电流"""
        self.plot_jDL(*arg, **kwargs)

    def plot_θsr(self, *arg, **kwargs):
        """作图：双电层效应局部体积电流密度、电流"""
        self.plot_csr(*arg, **kwargs)

    def plot_JLP(self, *arg, **kwargs):
        """作图：双电层效应局部体积电流密度、电流"""
        self.plot_jLP(*arg, **kwargs)

    def initialize_consistent(self,
            θsneg__: np.ndarray,
            θspos__: np.ndarray,
            θe_: np.ndarray,
            I: float | int = 0):
        """一致性初始化
        已知：θsneg__、θspos__、θe_、I
        求解：θsnegsurf_、θspossurf__、φsneg_、φspos_、φe_、Jintneg_、Jintpos、I0intneg_、I0intpos_、ηintneg_、ηintpos_
        令：JDLneg_ = JDLpos_ = JLP_ = 0
        """
        Nr, Nneg, Nsep, Npos = self.Nr, self.Nneg, self.Nsep, self.Npos  # 读取：负极、隔膜、正极网格数
        assert θsneg__.shape==(Nr, Nneg), f'负极固相颗粒内部无量纲锂离子浓度θsneg__.shape应为({Nr}, {Nneg})'
        assert θsneg__.shape==(Nr, Nneg), f'正极固相颗粒内部无量纲锂离子浓度θspos__.shape应为({Nr}, {Npos})'
        assert θe_.shape==(self.Ne,), f'电解液无量纲锂离子浓度θe_.shape应为({self.Ne},)'
        assert ((0<=θsneg__) & (θsneg__<=1)).all(), 'θsneg__取值范围应为(0, 1)'
        assert ((0<=θspos__) & (θspos__<=1)).all(), 'θspos__取值范围应为(0, 1)'
        assert (0<θe_).all(), 'θe_取值应大于0'
        # 更新K__矩阵的参数相关值
        if decouple_cs:=self.decouple_cs:
            pass
        else:
            self.update_K__idxθsnegsurf_idxJintneg_(self.Qneg, self.Dsneg)
            self.update_K__idxθspossurf_idxJintpos_(self.Qpos, self.Dspos)
        self.update_K__idxφsneg_idxJneg_(σneg := self.σneg)
        self.update_K__idxφspos_idxJpos_(σpos := self.σpos)
        self.update_K__idxφe_idxφe_(κ_ := self.κ_, κ_)
        self.update_K__idxηintneg_idxJneg_(RSEIneg := self.RSEIneg)
        self.update_K__idxηintpos_idxJpos_(RSEIpos := self.RSEIpos)

        # 原索引
        idxθsnegsurf_ = self.idxθsnegsurf_
        idxθspossurf_ = self.idxθspossurf_
        idxφsneg_ = self.idxφsneg_
        idxφspos_ = self.idxφspos_
        idxφe_ = self.idxφe_
        idxJintneg_ = self.idxJintneg_
        idxJintpos_ = self.idxJintpos_
        idxI0intneg_ = self.idxI0intneg_
        idxI0intpos_ = self.idxI0intpos_
        idxηintneg_ = self.idxηintneg_
        idxηintpos_ = self.idxηintpos_
        # 拼接一致性初始化待求解的变量
        idx_ = np.concatenate([
            idxθsnegsurf_, idxθspossurf_,
            idxφsneg_, idxφspos_, idxφe_,
            idxJintneg_, idxJintpos_,
            idxI0intneg_, idxI0intpos_,
            idxηintneg_, idxηintpos_,])
        Kinit__ = self.K__[np.ix_(idx_, idx_)]  # 提取K__矩阵
        Ninit = Kinit__.shape[0]                # Kinit__矩阵的行列数
        start = 0
        def assign(idxOld_) -> np.ndarray:
            """对矩阵Kinit__重新安排索引"""
            nonlocal start
            N = len(idxOld_)
            idxNew_ = np.arange(start, start + N)
            start += N
            return idxNew_
        idxθsnegsurf_ = assign(idxθsnegsurf_)
        idxθspossurf_ = assign(idxθspossurf_)
        idxφsneg_ = assign(idxφsneg_)
        idxφspos_ = assign(idxφspos_)
        idxφe_ = assign(idxφe_)
        idxJintneg_ = assign(idxJintneg_)
        idxJintpos_ = assign(idxJintpos_)
        idxI0intneg_ = assign(idxI0intneg_)
        idxI0intpos_ = assign(idxI0intpos_)
        idxηintneg_ = assign(idxηintneg_)
        idxηintpos_ = assign(idxηintpos_)

        solve_Jint_ = LPP2D.solve_Jint_
        solve_dJintdI0int_ = LPP2D.solve_dJintdI0int_
        solve_dJintdηint_  = LPP2D.solve_dJintdηint_
        solve_I0int_ = LPP2D.solve_I0int_
        solve_dI0intdθssurf_ = LPP2D.solve_dI0intdθssurf_
        solve_Uneg_, solve_Upos_ = self.solve_Uneg_, self.solve_Upos_              # 读取：负极、正极开路电位函数 [V]
        solve_dUdθneg_, solve_dUdθpos_ = self.solve_dUdθneg_, self.solve_dUdθpos_  # 读取：负极、正极开路电位对嵌锂状态的偏导数函数 [V/–]

        Δxneg, Δxpos, ΔxWest_, ΔxEast_, Δx_ = self.Δxneg, self.Δxpos, self.ΔxWest_, self.ΔxEast_, self.Δx_  # 读取：网格尺寸
        T = self.T  # 温度
        F2RT = DFNP2D.F/(2*DFNP2D.R*T)
        κDκT_ = self.κD*T * κ_

        if I0intnegUnknown := (self._I0intneg is None):
            kneg = self.kneg          # 读取：负极集总主反应速率常数 [A]
        else:
            I0intneg = self.I0intneg  # 读取：负极集总主反应交换电流密度 [A]
        if I0intposUnknown := (self._I0intpos is None):
            kpos = self.kpos          # 读取：正极集总主反应速率常数 [A]
        else:
            I0intpos = self.I0intpos  # 读取：正极集总主反应交换电流密度 [A]

        coeffs_ = self.coeffs_
        # 外推表面浓度
        θsnegsurfExpl_ = coeffs_.dot(θsneg__[-3:])
        θspossurfExpl_ = coeffs_.dot(θspos__[-3:])

        ## 对Kinit__的右端项bKinit_赋值 ##
        bKinit_ = np.zeros(Ninit)  # 右端项
        if decouple_cs:
            # 强制表面浓度约束：认为 θsnegsurf_、θspossurf_ 是外推得到的已知值
            bKinit_[idxθsnegsurf_] = θsnegsurfExpl_
            bKinit_[idxθspossurf_] = θspossurfExpl_
        else:
            # 用颗粒扩散边界条件 关联Jint、θssurf以及靠近颗粒表面的3个内部节点浓度
            K__ = self.K__
            θsneg_ = θsneg__.ravel('F')
            θspos_ = θspos__.ravel('F')
            idxθsneg_, idxθspos_ = self.idxθsneg_, self.idxθspos_
            bKinit_[idxθsnegsurf_] = -(
                  K__[self.idxθsnegsurf_, idxθsneg_[Nr-3::Nr]]*θsneg_[Nr-3::Nr]
                + K__[self.idxθsnegsurf_, idxθsneg_[Nr-2::Nr]]*θsneg_[Nr-2::Nr]
                + K__[self.idxθsnegsurf_, idxθsneg_[Nr-1::Nr]]*θsneg_[Nr-1::Nr])
            bKinit_[idxθspossurf_] =  -(
                  K__[self.idxθspossurf_, idxθspos_[Nr-3::Nr]] * θspos_[Nr-3::Nr]
                + K__[self.idxθspossurf_, idxθspos_[Nr-2::Nr]] * θspos_[Nr-2::Nr]
                + K__[self.idxθspossurf_, idxθspos_[Nr-1::Nr]] * θspos_[Nr-1::Nr])
        # 固相电流边界条件
        bKinit_[idxφsneg_[0]]  = -Δxneg*I/σneg
        bKinit_[idxφspos_[-1]] =  Δxpos*I/σpos
        # 电解液电势方程的电解液锂离子浓度项
        bKinit_[idxφe_[0]] = κDκT_[0]*(θe_[1] - θe_[0])/ΔxEast_[0]/(0.5*(θe_[1] + θe_[0]))
        bKinit_[idxφe_[-1]] = -κDκT_[-1]*(θe_[-1] - θe_[-2])/ΔxWest_[-1]/(0.5*(θe_[-1] + θe_[-2]))
        bKinit_[idxφe_[1:-1]] = κDκT_[1:-1]*((θe_[2:] - θe_[1:-1])/ΔxEast_[1:-1]/(0.5*(θe_[2:] + θe_[1:-1]))
                                           - (θe_[1:-1] - θe_[:-2])/ΔxWest_[1:-1]/(0.5*(θe_[1:-1] + θe_[:-2])))
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            θinterface = (a*θe_[nE] + b*θe_[nW])/(a + b)
            bKinit_[idxφe_[nW]] = (κDκT_[nW]*(θinterface - θe_[nW])/(0.5*Δx_[nW])/θinterface
                                 - κDκT_[nW]*(θe_[nW] - θe_[nW - 1])/ΔxWest_[nW]/(0.5*(θe_[nW] + θe_[nW - 1])))
            bKinit_[idxφe_[nE]] = (κDκT_[nE]*(θe_[nE + 1] - θe_[nE])/ΔxEast_[nE]/(0.5*(θe_[nE + 1] + θe_[nE]))
                                 - κDκT_[nE]*(θe_[nE] - θinterface)/(0.5*Δx_[nE])/θinterface)
        ## Newton迭代初值 ##
        X_ = np.zeros(Ninit)
        X_[idxθsnegsurf_] = θsnegsurfExpl_
        X_[idxθspossurf_] = θspossurfExpl_
        X_[idxφe_] = 0
        Jintneg = I
        Jintpos = -I
        X_[idxJintneg_] = Jintneg
        X_[idxJintpos_] = Jintpos
        I0intneg_ = solve_I0int_(kneg, θsnegsurfExpl_, θe_[:Nneg])  if I0intnegUnknown else I0intneg
        I0intpos_ = solve_I0int_(kpos, θspossurfExpl_, θe_[-Npos:]) if I0intposUnknown else I0intpos
        if I0intnegUnknown:
            X_[idxI0intneg_] = I0intneg_
        if I0intposUnknown:
            X_[idxI0intpos_] = I0intpos_
        X_[idxηintneg_] = np.arcsinh(Jintneg/(2*I0intneg_))/F2RT
        X_[idxηintpos_] = np.arcsinh(Jintpos/(2*I0intpos_))/F2RT
        X_[idxφsneg_] = X_[idxηintneg_] + RSEIneg*Jintneg + self.solve_Uneg_(X_[idxθsnegsurf_])
        X_[idxφspos_] = X_[idxηintpos_] + RSEIpos*Jintpos + self.solve_Upos_(X_[idxθspossurf_])

        # Newton迭代
        J__ = Kinit__.copy()
        θeneg_, θepos_ = θe_[:Nneg], θe_[-Npos:]
        for nNewton in range(1, 51):
            F_ = Kinit__.dot(X_) - bKinit_  # (Ninit,) F残差向量
            # 提取解
            θsnegsurf_, θspossurf_ = X_[idxθsnegsurf_], X_[idxθspossurf_]
            I0intneg_ = X_[idxI0intneg_] if I0intnegUnknown else I0intneg
            I0intpos_ = X_[idxI0intpos_] if I0intposUnknown else I0intpos
            ηintneg_, ηintpos_ = X_[idxηintneg_], X_[idxηintpos_]
            # F向量非线性部分
            F_[idxJintneg_] -= solve_Jint_(T, I0intneg_, ηintneg_)  # F向量Jintneg部分
            F_[idxJintpos_] -= solve_Jint_(T, I0intpos_, ηintpos_)  # F向量Jintpos部分
            if I0intnegUnknown:
                F_[idxI0intneg_] -= solve_I0int_(kneg, θsnegsurf_, θeneg_)   # F向量I0intneg部分
            if I0intposUnknown:
                F_[idxI0intpos_] -= solve_I0int_(kpos, θspossurf_, θe_[-Npos:])  # F向量I0intpos部分
            F_[idxηintneg_] += solve_Uneg_(θsnegsurf_)  # F向量ηintneg非线性部分
            F_[idxηintpos_] += solve_Upos_(θspossurf_)  # F向量ηintpos非线性部分
            # 更新Jacobi矩阵
            J__[idxJintneg_, idxηintneg_] = -solve_dJintdηint_(T, I0intneg_, ηintneg_)  # ∂FJintneg/∂ηintneg
            J__[idxJintpos_, idxηintpos_] = -solve_dJintdηint_(T, I0intpos_, ηintpos_)  # ∂FJintpos/∂ηintpos
            if I0intnegUnknown:
                J__[idxJintneg_, idxI0intneg_] = -solve_dJintdI0int_(T, ηintneg_)  # ∂FJintneg/∂I0intneg
                J__[idxI0intneg_, idxθsnegsurf_] = -solve_dI0intdθssurf_(T, θsnegsurf_, θeneg_, I0intneg_)  # ∂FI0intneg/∂θsnegsurf
            if I0intposUnknown:
                J__[idxJintpos_, idxI0intpos_] = -solve_dJintdI0int_(T, ηintpos_)  # ∂FJintpos/∂I0intpos
                J__[idxI0intpos_, idxθspossurf_] = -solve_dI0intdθssurf_(T, θspossurf_, θepos_, I0intpos_)  # ∂FI0intpos/∂θspossurf
            J__[idxηintneg_, idxθsnegsurf_] = solve_dUdθneg_(θsnegsurf_)  # ∂Fηintneg/∂θsnegsurf
            J__[idxηintpos_, idxθspossurf_] = solve_dUdθpos_(θspossurf_)  # ∂Fηintpos/∂θspossurf

            ΔX_ = np.linalg.solve(J__, F_)
            X_ -= ΔX_

            if abs(ΔX_).max()<1e-6:
                break
        else:
            raise DFNP2D.Error(f'一致性初始化失败，Newton迭代{nNewton = }次，不收敛，{abs(ΔX_).max() = }')

        # 初始化状态
        self.I = I
        self.θsneg__ = θsneg__
        self.θspos__ = θspos__
        self.θe_ = θe_
        self.φsneg_ = X_[idxφsneg_]
        self.φspos_ = X_[idxφspos_]
        self.φe_ = X_[idxφe_]
        self.Jintneg_ = X_[idxJintneg_]
        self.Jintpos_ = X_[idxJintpos_]
        self.JDLneg_ = np.zeros(Nneg)
        self.JDLpos_ = np.zeros(Npos)
        self.JLP_ = np.zeros(Nneg)
        if self.verbose:
            print(f'一致性初始化完成。Newton迭代{nNewton = }。Consistent initial conditions are solved! ')
        return

if __name__=='__main__':
    cell = LPP2D(
        Δt=10, SOC0=0.2,
        Nneg=8, Nsep=7, Npos=6, Nr=9,
        # CDLneg=0, CDLpos=0,
        # I0intneg=21, I0intpos=25,
        # I0LP=0.1,
        Qcell=9,
        Qneg=13, Qpos=12,
        lithiumPlating=True,
        doubleLayerEffect=False,
        # timeDiscretization='backward',
        # radialDiscretization='EI',
        # verbose=False,
        # complete=False,
        # constants=True,
        # decouple_cs=False,
        )
    cell.count_lithium()
    cell.CC(-5, 2000)#.CC(0, 300).CC(10, 500).CC(5, 1000)

    cell.count_lithium()

    '''
    cell.plot_UI()
    cell.plot_TQgen()
    cell.plot_SOC()
    cell.plot_θ(np.arange(0, 2001, 200))
    cell.plot_φ(np.arange(0, 2001, 200))
    cell.plot_Jint(np.arange(0, 2001, 200))
    cell.plot_JDL(np.arange(0, 2001, 200))
    cell.plot_θsr(range(0, 2001, 200), 1)
    cell.plot_JLP(np.arange(0, 2001, 200))
    cell.plot_ηLP()
    cell.plot_OCV()
    cell.plot_dUdθ()
    '''
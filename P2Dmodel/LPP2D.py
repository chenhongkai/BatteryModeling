#%%
from scipy.linalg.lapack import dgtsv
from numpy import ndarray,\
    array, zeros, zeros_like, ones, full, hstack, concatenate, tile, \
    exp, sqrt, sinh, cosh, arcsinh, outer, \
    ix_, isnan
from numpy.linalg import solve

from P2Dmodel.P2Dbase import P2Dbase
from P2Dmodel.OCP import NMC111, Graphite
from P2Dmodel.tools import triband_to_dense, DiagonalSliceRavel


class LPP2D(P2Dbase):
    """锂离子电池集总参数准二维模型 Lumped-Parameter Pseudo-two-Dimension model"""

    __slots__ = (
        # LPP2D专有参数名
        'Qcell', 'Qneg', 'Qpos', 'qeneg', 'qesep', 'qepos',
        '_κneg', '_κsep', '_κpos', 'De', 'κD',
        '_I0intneg', '_I0intpos', '_I0LP',
        # LPP2D专有状态量
        'θsneg__', 'θspos__', 'θsnegsurf_', 'θspossurf_', 'θe_',
        'Jintneg_', 'Jintpos_', 'JDLneg_', 'JDLpos_',
        'I0intneg_', 'I0intpos_',
        'Jneg_', 'Jpos_',
        'JLP_',
        # LPP2D专有恒定量
        'r_', 'Δr_',
        'bandKθs__',
        )

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
            RSEIneg: float = 6.91e-5,  # 负极集总SEI膜内阻 [Ω]
            RSEIpos: float = 2e-5,     # 正极集总SEI膜内阻 [Ω]
            kneg: float = 32.,      # 负极集总反应速率常数 [A]
            kpos: float = 42.,      # 正极集总反应速率常数 [A]
            kLP: float = 3.607e-6,  # 负极集总析锂反应速率常数 [A]
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
            **kwargs,):
        self.Qcell = Qcell; assert Qcell>0, f'电池理论可用容量{Qcell = }，应大于0 [Ah]'
        # 4边界嵌锂状态参数、负极、正极容量Qneg Qpos
        if all(v is not None for v in (θminneg, θmaxneg, θminpos, θmaxpos)):
            # 4θ均非None，使用给定4θ，忽略4等式关系，并重新计算Qneg、Qpos')
            assert 0<θminneg<θmaxneg<1, f'负极最小、最大嵌锂状态{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1'
            assert 0<θminpos<θmaxpos<1, f'正极最小、最大嵌锂状态{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1'
            self.Qneg = Qcell/(θmaxneg - θminneg)
            self.Qpos = Qcell/(θmaxpos - θminpos)
        else:
            # 使用4等式由Qcell、Qneg、Qpos计算4θ
            if Qcell>=Qneg or Qcell>=Qpos:
                raise P2Dbase.Error(f'电池理论可用容量{Qcell = }应小于负极容量{Qneg = }与正极容量{Qpos = }', {'Qcell': Qcell, 'Qneg': Qneg, 'Qpos': Qpos})
            self.Qneg = Qneg; assert Qneg>0, f'负极容量{Qneg = }，应大于0 [Ah]'
            self.Qpos = Qpos; assert Qpos>0, f'正极容量{Qpos = }，应大于0 [Ah]'
            UOCPneg = kwargs.pop('UOCPneg', None)  # 从 kwargs 获取 UOCPneg
            UOCPpos = kwargs.pop('UOCPpos', None)  # 从 kwargs 获取 UOCPpos
            if UOCPneg is None:
                UOCPneg = Graphite().Graphite_COMSOL
            else:
                assert callable(UOCPneg), '函数UOCPneg，输入负极嵌锂状态θsneg_ [–]，输出正极开路电位UOCPneg_ [V]'
            if UOCPpos is None:
                UOCPpos = NMC111().NMC111_COMSOL
            else:
                assert callable(UOCPpos), '函数UOCPpos，输入正极嵌锂状态θspos_ [–]，输出负极开路电位UOCPpos_ [V]'
            assert Umax>Umin>0, f'运行电压{Umax = }，{Umin = }，应满足Umax > Umin > 0 [V]'
            θminneg, θmaxneg, θminpos, θmaxpos, ΔFmax = P2Dbase.solve_4θ(
                UOCPneg, UOCPpos, Qcell, Qneg, Qpos,
                Umin, Umax)
            tempdict = {'θminneg': θminneg, 'θmaxneg': θmaxneg, 'θminpos': θminpos, 'θmaxpos': θmaxpos, 'ΔFmax': ΔFmax}
            if not ΔFmax<1e-5:
                raise P2Dbase.Error(f'求4个边界嵌锂状态，不收敛，无解！F函数最大绝对误差{ΔFmax = }', tempdict)
            if not 0<θminneg<θmaxneg<1:
                raise P2Dbase.Error(f'{θminneg = }，{θmaxneg = }，应满足0<θminneg<θmaxneg<1', tempdict)
            if not 0<θminpos<θmaxpos<1:
                raise P2Dbase.Error(f'{θminpos = }，{θmaxpos = }，应满足0<θminpos<θmaxpos<1', tempdict)
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
        self.De = De; assert De>0, f'集总离子扩散率/电导率之比{De = }，应大于0 [A/S]'
        self.κD = κD; assert κD>0, f'集总扩散电解液离子电导率系数{κD = }，应大于0 [V/K]'
        self.RSEIneg = RSEIneg; assert RSEIneg>=0, f'负极集总SEI膜电阻{RSEIneg = }，应大于或等于0 [Ω]'
        self.RSEIpos = RSEIpos; assert RSEIpos>=0, f'正极集总SEI膜电阻{RSEIpos = }，应大于或等于0 [Ω]'
        # 5动力学参数
        self.kneg = kneg; assert kneg>0, f'负极集总主反应速率常数{kneg = }，应大于0 [A]'
        self.kpos = kpos; assert kpos>0, f'正极集总主反应速率常数{kpos = }，应大于0 [A]'
        self.kLP = kLP;   assert kLP>0, f'负极集总析锂反应速率常数{kLP = }，应大于0 [A]'
        # 3电抗参数
        self.CDLneg = CDLneg; assert CDLneg>=0, f'负极集总双电层电容{CDLneg = }，应大于或等于0 [F]'
        self.CDLpos = CDLpos; assert CDLpos>=0, f'正极集总双电层电容{CDLpos = }，应大于或等于0 [F]'
        self.l = l;           assert l>=0, f'等效电感{l = }，应大于或等于0 [H]'
        # 3交换电流密度
        self._I0intneg = self._i0intneg = I0intneg; assert (I0intneg is None) or (I0intneg>0), f'负极主反应集总交换电流密度{I0intneg = }，应大于0 [A]'
        self._I0intpos = self._i0intpos = I0intpos; assert (I0intpos is None) or (I0intpos>0), f'正极主反应集总交换电流密度{I0intpos = }，应大于0 [A]'
        self._I0LP = self._i0LP = I0LP;             assert (I0LP is None)  or (I0LP>0), f'负极析锂反应集总交换电流密度{I0LP = }，应大于0 [A]'
        # P2D通用参数
        P2Dbase.__init__(self,
                         Lneg=1, Lsep=1, Lpos=1,
                         Rsneg=1, Rspos=1,
                         SOC0=SOC0,
                         UOCPneg=UOCPneg, UOCPpos=UOCPpos,
                         θminneg=θminneg, θmaxneg=θmaxneg,
                         θminpos=θminpos, θmaxpos=θmaxpos, **kwargs)
        # LPP2D专有状态量
        Nneg, Npos, Nr = self.Nneg, self.Npos, self.Nr  # 读取：网格数
        θsneg = θminneg + SOC0*(θmaxneg - θminneg)  # 初始负极嵌锂状态 [–]
        θspos = θmaxpos - SOC0*(θmaxpos - θminpos)  # 初始正极嵌锂状态 [–]
        self.θsneg__ = full((Nr, Nneg), θsneg)
        self.θspos__ = full((Nr, Npos), θspos)  # 初始化：负极、正极固相内部无量纲锂离子浓度场 [–]
        self.θsnegsurf_ = full(Nneg, θsneg)
        self.θspossurf_ = full(Npos, θspos)  # 初始化：负极、正极固相表面无量纲锂离子浓度场 [–]
        self.θe_ = ones(self.Ne)             # 初始化：电解液无量纲锂离子浓度场 [–]
        self.Jintneg_ = zeros(Nneg)
        self.Jintpos_ = zeros(Npos)  # 初始化：负极、正极集总主反应局部体积电流密度场 [A]
        self.JDLneg_ = zeros(Nneg)
        self.JDLpos_ = zeros(Npos)   # 初始化：负极、正极集总双电层效应局部体积电流密度场 [A]
        I0intneg = self.I0intneg if self._I0intneg else LPP2D.solve_I0int_(self.kneg, θsneg, 1)
        I0intpos = self.I0intpos if self._I0intpos else LPP2D.solve_I0int_(self.kpos, θspos, 1)
        self.I0intneg_ = full(Nneg, I0intneg)
        self.I0intpos_ = full(Npos, I0intpos)  # 初始化：负极、正极主反应集总交换电流密度场 [A]
        self.Jneg_ = zeros(Nneg)
        self.Jpos_ = zeros(Npos)     # 初始化：负极、正极总局部体积电流密度场 [A]
        if lithiumPlating := self.lithiumPlating:
            self.JLP_ = zeros(Nneg)  # 初始化：负极析锂反应集总局部体积电流密度场 [A]
        # 恒定量
        self.Δr_ = self.Δrneg_
        self.r_  = self.rneg_
        self.bandKθs__ = self.bandKcsneg__  # (3, Nr) 固相浓度三对角矩阵的带
        del self.Δrneg_, self.Δrpos_, self.bandKcsneg__, self.bandKcspos__
        if self.complete:
            # 作图变量单位
            self.xSign, self.xUnit = r'$\overline{\it x}$', ''  # 电极厚度方向坐标x符号、单位
            self.rSign, self.rUnit = r'$\overline{\it r}$', ''  # 颗粒径向坐标r符号、单位
            self.cSign, self.cUnit = r'${\it θ}$', ''   # 锂离子浓度θ符号、单位
            self.jSign, self.jUnit = r'${\it J}$', 'A'  # 集总局部体积电流密度J符号、单位
            self.i0Sign, self.i0Unit = r'${\it I}_{0}$', 'A'  # 集总交换电流密度I0符号、单位
            # 需记录的数据名称
            extra_datanames_ = [
                'θsneg__', 'θspos__',        # 负极、正极固相无量纲锂离子浓度场 [–]
                'θsnegsurf_', 'θspossurf_',  # 负极、正极表面无量纲锂离子浓度场 [–]
                'θe_',                       # 电解液无量纲锂离子浓度场 [–]
                'Jintneg_', 'Jintpos_',      # 负极、正极主反应集总局部体积电流密度场 [A]
                'JDLpos_', 'JDLneg_',        # 负极、正极双电层效应集总局部体积电流密度场 [A]
                'I0intneg_', 'I0intpos_',]   # 负极、正极主反应集总交换电流密度场 [A]
            if lithiumPlating:
                extra_datanames_.append('JLP_')  # 负极析锂集总局部体积电流密度场 [A]
            self.datanames_.extend(extra_datanames_)
            self.data.update({name: [] for name in extra_datanames_})

        if self.verbose and type(self) is LPP2D:
            print(self)
            print(f'集总参数P2D模型(LPP2D)初始化完成!')

    def update_K__with_pure_electrochemical_parameters(self):
        # 对K__矩阵赋纯电化学参数相关值
        if self.decouple_cs:
            pass
        else:
            self._update_K__θsnegsurf_Jintneg_when_coupling_cs(self.Qneg, self.Dsneg)
            self._update_K__θspossurf_Jintpos_when_coupling_cs(self.Qpos, self.Dspos)
        self._update_K__φsneg_Jneg(self.σneg)
        self._update_K__φspos_Jpos(self.σpos)
        self._update_K__φe_φe(κ_ := self.κ_, κ_)
        self._update_K__ηintneg_Jneg(self.RSEIneg)
        self._update_K__ηintpos_Jpos(self.RSEIpos)
        if self.lithiumPlating:
            self._update_K__ηLP_Jneg(self.RSEIneg)

    def _update_K__θsnegsurf_Jintneg_when_coupling_cs(self, Qneg, Dsneg):
        # 更新K__矩阵θsnegsurf行Jintneg列
        self.ravelK_[self.sK.sr_csnegsurf_jintneg] = 1/(10800*Qneg*Dsneg)

    def _update_K__θspossurf_Jintpos_when_coupling_cs(self, Qpos, Dspos):
        # 更新K__矩阵θspossurf行Jintpos列
        self.ravelK_[self.sK.sr_cspossurf_jintpos] = 1/(10800*Qpos*Dspos)

    # 更新K__矩阵φsneg行Jneg列
    _update_K__φsneg_Jneg = P2Dbase._update_K__φsneg_jneg

    # 更新K__矩阵φspos行Jpos列
    _update_K__φspos_Jpos = P2Dbase._update_K__φspos_jpos

    def _update_K__bK_θe_θe_J(self,
                              Deκ_: ndarray,
                              qe_: ndarray,
                              Δt: float, old_θe_, old_Jneg_, old_Jpos_,):

        # 更新K__矩阵θe行θe列
        # 更新K__矩阵θe行J列
        # 更新bK_向量θe行
        KθeJ = -Δt
        P2Dbase._update_K__bK_ce_ce_j(self,
            Deκ_, Deκ_, qe_, KθeJ, Δt, old_θe_, old_Jneg_, old_Jpos_)

    def _update_bK_φsneg0_φsposEnd(self, σneg, σpos, I):
        # 更新bK_向量φsneg行首元、φspos行末元
        bK_ = self.bK_
        sK = self.sK
        bK_[sK.s_φsneg.start]    = -self.Δxneg*I/σneg
        bK_[sK.s_φspos.stop - 1] =  self.Δxpos*I/σpos

    def _update_K__bK_JDL_φs_φe_J(self, RSEIneg, RSEIpos, CDLneg, CDLpos, Δt):
        # 更新K__矩阵JDL行φs、φe、J列
        # 更新bK_向量JDL行
        P2Dbase._update_K__bK_jDL_φs_φe_j(self, 1, 1, RSEIneg, RSEIpos, CDLneg, CDLpos, Δt)

    def _update_K__ηintneg_Jneg(self, RSEIneg):
        # 更新K__矩阵ηintneg行Jneg列
        self._update_K__ηintneg_jneg(RSEIneg, 1)

    def _update_K__ηintpos_Jpos(self, RSEIpos):
        # 更新K__矩阵ηintpos行Jpos列
        self._update_K__ηintpos_jpos(RSEIpos, 1)

    def _update_K__ηLP_Jneg(self, RSEIneg):
        # 更新K__矩阵ηLP行Jneg列
        self._update_K__ηLP_jneg(RSEIneg, 1)

    def _stepping(self, Δt):
        """时间步进：Newton法迭代所有因变量"""
        # 读取模式
        lithiumPlating = self.lithiumPlating
        timeDiscretization = self.timeDiscretization
        decouple_cs = self.decouple_cs
        doubleLayerEffect = self.doubleLayerEffect

        ravelK_ = self.ravelK_  # 读取：因变量线性矩阵K__展平视图
        bK_ = self.bK_          # 读取：常数项向量，F_ = K__ @ X_ - bK_
        K__ = ravelK_.base

        # 读取索引
        sK = self.sK
        s_θsneg = sK.s_csneg
        s_θspos = sK.s_cspos
        s_θsnegsurf = sK.s_csnegsurf
        s_θspossurf = sK.s_cspossurf
        s_θe = sK.s_ce
        s_θeneg = sK.s_ceneg
        s_θepos = sK.s_cepos
        s_φsneg = sK.s_φsneg
        s_φspos = sK.s_φspos
        s_φe = sK.s_φe
        s_Jintneg = sK.s_jintneg
        s_Jintpos = sK.s_jintpos
        s_JDLneg = sK.s_jDLneg
        s_JDLpos = sK.s_jDLpos
        s_I0intneg = sK.s_i0intneg
        s_I0intpos = sK.s_i0intpos
        s_ηintneg = sK.s_ηintneg
        s_ηintpos = sK.s_ηintpos
        s_θ = sK.s_c
        s_φ = sK.s_φ
        s_J = sK.s_j
        if lithiumPlating:
            s_JLP = sK.s_jLP

        # 读取方法
        solve_banded_matrix = P2Dbase.solve_banded_matrix
        solve_Jint_ = LPP2D.solve_Jint_
        solve_dJintdηint_ = LPP2D.solve_dJintdηint_
        solve_dJintdI0int_ = LPP2D.solve_dJintdI0int_
        solve_I0int_ = LPP2D.solve_I0int_
        solve_dI0intdθssurf_ = LPP2D.solve_dI0intdθssurf_
        solve_dI0intdθe_ = LPP2D.solve_dI0intdθe_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_
        solve_dUOCPdθsneg_ = self.solve_dUOCPdθsneg_
        solve_dUOCPdθspos_ = self.solve_dUOCPdθspos_
        if lithiumPlating:
            solve_JLP_ = LPP2D.solve_JLP_
            solve_dJLPdθe_ = LPP2D.solve_dJLPdθe_
            solve_dJLPdηLP_ = LPP2D.solve_dJLPdηLP_
            solve_I0LP_ = LPP2D.solve_I0LP_

        # 读取参数
        Nneg, Nsep, Npos, Nr = self.Nneg, self.Nsep, self.Npos, self.Nr  # 读取：网格数
        Δx_, Δr_ = self.Δx_, self.Δr_                  # 读取：网格尺寸 [–]
        ΔxWest_, ΔxEast_ = self.ΔxWest_, self.ΔxEast_  # 读取：网格距离 [–]
        RSEIneg, RSEIpos = self.RSEIneg, self.RSEIpos  # 读取：负极、正极集总SEI膜内阻 [Ω]
        Dsneg, Dspos = self.Dsneg, self.Dspos          # 读取：负极、正极集总固相扩散系数 [1/s]
        Qneg, Qpos = self.Qneg, self.Qpos              # 读取：负极、正极容量 [Ah]
        κ_ = self.κ_
        if I0intnegUnknown := (self._I0intneg is None):
            kneg = self.kneg          # 读取：负极集总主反应速率常数 [A]
        else:
            I0intneg = self.I0intneg  # 读取：负极集总主反应交换电流密度 [A]
        if I0intposUnknown := (self._I0intpos is None):
            kpos = self.kpos          # 读取：正极集总主反应速率常数 [A]
        else:
            I0intpos = self.I0intpos  # 读取：正极集总主反应交换电流密度 [A]
        if lithiumPlating:
            if I0LPUnknown := (self._I0LP is None):
                kLP = self.kLP    # 读取：负极析锂反应速率常数 [A]
            else:
                I0LP = self.I0LP  # 读取：负极析锂反应交换电流密度 [A]

        # 读取状态
        I = self.I  # 电流 [A]
        T = self.T  # 温度 [K]
        F2RT = 0.5*P2Dbase.F/P2Dbase.R*T  # 常数 [1/V]
        κDκT_ = (self.κD * T) * κ_
        data = self.data  # 运行数据字典

        KθsJintneg = Δt/(10800*Qneg)/((1 - (1 - Δr_[-1])**3)/3)
        KθsJintpos = Δt/(10800*Qpos)/((1 - (1 - Δr_[-1])**3)/3)
        bandKθs__ = self.bandKθs__
        bandKθsneg__ = (Δt*Dsneg) * bandKθs__  # (3, Nr)
        bandKθspos__ = (Δt*Dspos) * bandKθs__  # (3, Nr)

        if timeDiscretization=='CN':
            bandKθsneg__ *= .5
            bandKθspos__ *= .5
            KθsJintneg *= .5
            KθsJintpos *= .5
            bandBθsneg__ = -bandKθsneg__  # (3, Nr)
            bandBθspos__ = -bandKθspos__  # (3, Nr)
            bandBθsneg__[1] += 1  # 对角元+1
            bandBθspos__[1] += 1  # 对角元+1
        bandKθsneg__[1] += 1  # 对角元+1
        bandKθspos__[1] += 1  # 对角元+1

        ## 更新K__矩阵、bK_向量时变值 ##
        if decouple_cs:
            # 历史浓度影响的浓度分量
            match timeDiscretization:
                case 'backward':
                    RHSneg__ = self.θsneg__  # (Nr, Nneg)
                    RHSpos__ = self.θspos__  # (Nr, Npos)
                case 'CN':
                    RHSneg__ = triband_to_dense(bandBθsneg__) @ self.θsneg__  # (Nr, Nneg)
                    RHSpos__ = triband_to_dense(bandBθspos__) @ self.θspos__  # (Nr, Npos)
            e__ = self.e__  # (1, Nr)
            RHSneg__ = concatenate([RHSneg__, e__], axis=1)  # (Nr, Nneg+1)
            RHSpos__ = concatenate([RHSpos__, e__], axis=1)  # (Nr, Npos+1)
            Sneg__ = dgtsv(bandKθsneg__[2, :-1], bandKθsneg__[1], bandKθsneg__[0, 1:], RHSneg__, True, True, True, True)[3]  # (Nr, Nneg+1)
            Spos__ = dgtsv(bandKθspos__[2, :-1], bandKθspos__[1], bandKθspos__[0, 1:], RHSpos__, True, True, True, True)[3]  # (Nr, Npos+1)
            θsnegI__ = Sneg__[:, :-1]  # (Nr, Nneg) 内部锂离子浓度的历史影响分量
            θsposI__ = Spos__[:, :-1]  # (Nr, Npos)
            γneg_ = Sneg__[:, -1] * -KθsJintneg  # (Nr,)
            γpos_ = Spos__[:, -1] * -KθsJintpos  # (Nr,)
            # 3点2次多项式外推颗粒表面锂离子浓度的历史影响分量
            # backward: θssurf_ = α_ + Jint_*β
            # CN:       θssurf_ = α_ + (Jint_ + Jintold)*β
            c_ = self.coeffsExpl_  # (3,)
            αneg_ = c_.dot(θsnegI__[-3:])  # (Nneg,)
            αpos_ = c_.dot(θsposI__[-3:])  # (Npos,)
            βneg = c_.dot(γneg_[-3:])
            βpos = c_.dot(γpos_[-3:])
            # 负极、正极固相表面浓度θssurf行
            ravelK_[sK.sr_csnegsurf_jintneg] = -βneg
            ravelK_[sK.sr_cspossurf_jintpos] = -βpos
        else:
            # 负极、正极固相内部浓度θsneg行、θspos行
            for Nreg, band__, sr_θs_θs_u, sr_θs_θs, sr_θs_θs_l, sr_θsEnd_Jint, Kθs_Jint in zip(
                    (Nneg, Npos),
                    (bandKθsneg__, bandKθspos__),
                    (sK.sr_csneg_csneg_u,    sK.sr_cspos_cspos_u),
                    (sK.sr_csneg_csneg,      sK.sr_cspos_cspos),
                    (sK.sr_csneg_csneg_l,    sK.sr_cspos_cspos_l),
                    (sK.sr_csnegEnd_jintneg, sK.sr_csposEnd_jintpos),
                    (KθsJintneg, KθsJintpos)):
                ravelK_[sr_θs_θs_u] = tile(hstack([band__[0, 1:], 0.]), Nreg)[:-1]   # 上对角线
                ravelK_[sr_θs_θs]   = tile(band__[1], Nreg)                          # 主对角线
                ravelK_[sr_θs_θs_l] = tile(hstack([band__[2, :-1], 0.]), Nreg)[:-1]  # 下对角线
                ravelK_[sr_θsEnd_Jint] = Kθs_Jint  # Jint列

        match timeDiscretization:
            case 'CN':
                if decouple_cs:
                    bK_[s_θsnegsurf] = αneg_ + βneg*self.Jintneg_
                    bK_[s_θspossurf] = αpos_ + βpos*self.Jintpos_
                else:
                    bK_[s_θsneg] = (triband_to_dense(bandBθsneg__) @ self.θsneg__).ravel('F')
                    bK_[s_θspos] = (triband_to_dense(bandBθspos__) @ self.θspos__).ravel('F')
                    bK_[s_θsneg][Nr - 1::Nr] -= KθsJintneg*self.Jintneg_
                    bK_[s_θspos][Nr - 1::Nr] -= KθsJintpos*self.Jintpos_
            case 'backward':
                if decouple_cs:
                    bK_[s_θsnegsurf] = αneg_
                    bK_[s_θspossurf] = αpos_
                else:
                    bK_[s_θsneg] = self.θsneg__.ravel('F')
                    bK_[s_θspos] = self.θspos__.ravel('F')


        # 电解液浓度θe行θe、J列
        self._update_K__bK_θe_θe_J(
            self.Deκ_, self.qe_, Δt,
            self.θe_, self.Jneg_, self.Jpos_)

        # 固相电流边界条件
        self._update_bK_φsneg0_φsposEnd(self.σneg, self.σpos, I)

        # 集总双电层电流密度JDL行
        if doubleLayerEffect:
            self._update_K__bK_JDL_φs_φe_J(RSEIneg, RSEIpos, self.CDLneg, self.CDLpos, Δt)

        # 索引解
        X_ = zeros_like(bK_)
        if decouple_cs:
            pass
        else:
            θsneg_ = X_[s_θsneg]
            θspos_ = X_[s_θspos]
        θsnegsurf_ = X_[s_θsnegsurf]
        θspossurf_ = X_[s_θspossurf]
        θe_ = X_[s_θe]
        θeneg_ = X_[s_θeneg]
        θepos_ = X_[s_θepos]
        φsneg_ = X_[s_φsneg]
        φspos_ = X_[s_φspos]
        φe_    = X_[s_φe]
        φeneg_ = X_[sK.s_φeneg]
        φepos_ = X_[sK.s_φepos]
        Jintneg_ = X_[s_Jintneg]
        Jintpos_ = X_[s_Jintpos]
        if doubleLayerEffect:
            JDLneg_ = X_[s_JDLneg]
            JDLpos_ = X_[s_JDLpos]
        I0intneg_ = X_[s_I0intneg] if I0intnegUnknown else I0intneg
        I0intpos_ = X_[s_I0intpos] if I0intposUnknown else I0intpos
        ηintneg_ = X_[s_ηintneg]
        ηintpos_ = X_[s_ηintpos]
        if lithiumPlating:
            JLP_ = X_[s_JLP]
            ηLP_ = X_[sK.s_ηLP]

        # 解向量赋初值
        if decouple_cs:
            pass
        else:
            θsneg_[:] = self.θsneg__.ravel('F')
            θspos_[:] = self.θspos__.ravel('F')
        θsnegsurf_[:] = self.θsnegsurf_
        θspossurf_[:] = self.θspossurf_
        θe_[:] = self.θe_
        if I0intnegUnknown:
            I0intneg_[:] = self.I0intneg_
        if I0intposUnknown:
            I0intpos_[:] = self.I0intpos_
        if I==data['I'][-1]:
            # 恒电流
            φsneg_[:] = self.φsneg_
            φspos_[:] = self.φspos_
            φe_[:] = self.φe_
            Jintneg_[:] = self.Jintneg_
            Jintpos_[:] = self.Jintpos_
            ηintneg_[:] = self.ηintneg_
            ηintpos_[:] = self.ηintpos_
        else:
            # 变电流瞬间
            Jintneg_[:] = Jintneg =  I
            Jintpos_[:] = Jintpos = -I
            ηintneg_[:] = arcsinh(Jintneg/(2 * I0intneg_)) / F2RT
            ηintpos_[:] = arcsinh(Jintpos/(2 * I0intpos_)) / F2RT
            φsneg_[:] = ηintneg_ + RSEIneg*Jintneg + solve_UOCPneg_(θsnegsurf_)
            φspos_[:] = ηintpos_ + RSEIpos*Jintpos + solve_UOCPpos_(θspossurf_)
        if lithiumPlating:
            ηLP_[:] = φsneg_ - φeneg_ - RSEIneg*Jintneg_

        # 初始化Jacobi矩阵
        J__ = K__.copy()
        ravelJ_ = J__.ravel()      # (N*N,) Jacobi矩阵展平视图
        ravelJ_φe_θe_l_ = ravelJ_[sK.sr_φe_ce_l]
        ravelJ_φe_θe_u_ = ravelJ_[sK.sr_φe_ce_u]
        ravelJ_φe_θe_   = ravelJ_[sr_φe_ce := sK.sr_φe_ce]
        start0φece = sr_φe_ce.start
        NJ1 = bK_.size + 1
        ravelJ_Jintneg_ηintneg_ = ravelJ_[sK.sr_jintneg_ηintneg]
        ravelJ_Jintpos_ηintpos_ = ravelJ_[sK.sr_jintpos_ηintpos]
        ravelJ_ηintneg_θsnegsurf_ = ravelJ_[sK.sr_ηintneg_csnegsurf]
        ravelJ_ηintpos_θspossurf_ = ravelJ_[sK.sr_ηintpos_cspossurf]
        if I0intnegUnknown:
            ravelJ_Jintneg_I0intneg_ = ravelJ_[sK.sr_jintneg_i0intneg]
            ravelJ_I0intneg_θsnegsurf_ = ravelJ_[sK.sr_i0intneg_csnegsurf]
            ravelJ_I0intneg_θeneg_ = ravelJ_[sK.sr_i0intneg_ceneg]
        if I0intposUnknown:
            ravelJ_Jintpos_I0intpos_ = ravelJ_[sK.sr_jintpos_i0intpos]
            ravelJ_I0intpos_θspossurf_ = ravelJ_[sK.sr_i0intpos_cspossurf]
            ravelJ_I0intpos_θepos_     = ravelJ_[sK.sr_i0intpos_cepos]
        if lithiumPlating:
            ravelJ_JLP_θeneg_ = ravelJ_[sK.sr_jLP_ceneg]
            ravelJ_JLP_ηLP_ = ravelJ_[sK.sr_jLP_ηLP]

        # 预计算
        κDκT2ΔxWest_ = κDκT_[1:]  / ΔxWest_[1:]   # (Ne-1,)
        κDκT2ΔxEast_ = κDκT_[:-1] / ΔxEast_[:-1]  # (Ne-1,)

        for nNewton in range(1, 201):
            ## Newton迭代
            F_ = K__.dot(X_) - bK_  # F残差向量的线性部分

            # F向量非线性部分
            θeM_ = 0.5*(θe_[1:] + θe_[:-1])    # (Ne-1,) 相邻浓度均值
            q_ = (θe_[1:] - θe_[:-1]) / θeM_   # (Ne-1,)
            a_ = κDκT2ΔxWest_ * q_  # (Ne-1,)
            c_ = κDκT2ΔxEast_ * q_  # (Ne-1,)
            ΔFφe_ = hstack([0, a_]) - hstack([c_, 0])  # (Ne,)
            for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
                # 修正负极-隔膜界面、隔膜-正极界面
                a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
                θinterface = (a*θe_[nE] + b*θe_[nW])/(a + b)
                ΔFφe_[nW] = ( κDκT_[nW] * (θe_[nW] - θe_[nW-1])/ΔxWest_[nW]  / (0.5*(θe_[nW] + θe_[nW-1]))
                            - κDκT_[nW] * (θinterface - θe_[nW])/(0.5*Δx_[nW]) / θinterface)
                ΔFφe_[nE] = ( κDκT_[nE] * (θe_[nE] - θinterface)/(0.5*Δx_[nE]) / θinterface
                            - κDκT_[nE] * (θe_[nE+1] - θe_[nE] )/ΔxEast_[nE] / (0.5*(θe_[nE+1] + θe_[nE])))
            F_[s_φe] += ΔFφe_
            F_[s_Jintneg] -= solve_Jint_(T, I0intneg_, ηintneg_)  # F向量Jintneg部分
            F_[s_Jintpos] -= solve_Jint_(T, I0intpos_, ηintpos_)  # F向量Jintpos部分
            if I0intnegUnknown:
                F_[s_I0intneg] -= solve_I0int_(kneg, θsnegsurf_, θeneg_)  # F向量I0intneg部分
            if I0intposUnknown:
                F_[s_I0intpos] -= solve_I0int_(kpos, θspossurf_, θepos_)  # F向量I0intpos部分
            F_[s_ηintneg] += solve_UOCPneg_(θsnegsurf_)  # F向量ηintneg非线性部分
            F_[s_ηintpos] += solve_UOCPpos_(θspossurf_)  # F向量ηintpos非线性部分
            if lithiumPlating:
                I0LP_ = solve_I0LP_(kLP, θeneg_) if I0LPUnknown else I0LP  # 负极析锂反应的交换电流场 [A]
                F_[s_JLP] -= solve_JLP_(T, I0LP_, ηLP_)   # F向量JLP部分

            # 更新Jacobi矩阵非线性部分
            # φe行θe列
            q_ *= 0.5
            a_ = κDκT2ΔxWest_ / θeM_
            aa_ = a_ * q_
            c_ = κDκT2ΔxEast_ / θeM_
            cc_ = c_ * q_
            ravelJ_φe_θe_l_[:] = -aa_ - a_   # 下对角线
            ravelJ_φe_θe_u_[:] = cc_ - c_    # 上对角线
            ravelJ_φe_θe_[:]   = hstack([0, a_ - aa_]) + hstack([cc_ + c_, 0])  # 主对角线
            for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
                # 修正负极-隔膜界面、隔膜-正极界面
                nrW = start0φece + nW*NJ1
                nrE = start0φece + nE*NJ1
                κDκTWκE = κDκT_[nW]*κ_[nE]
                κDκTEκW = κDκT_[nE]*κ_[nW]
                num = κDκTWκE - κDκTEκW
                κEΔxW = κ_[nE]*Δx_[nW]
                κWΔxE = κ_[nW]*Δx_[nE]
                den1 = κEΔxW + κWΔxE
                den2 = κEΔxW * θe_[nE] + κWΔxE * θe_[nW]
                quotient = num / (den1*den2)
                ΔθeEW = θe_[nE] - θe_[nW]
                SθeW = θe_[nW] + θe_[nW-1]
                SθeE = θe_[nE] + θe_[nE+1]

                a  = 2 * κDκT_[nW] / (SθeW * Δx_[nW])
                aa = a * (θe_[nW] - θe_[nW-1]) / SθeW
                c  = 2 * κWΔxE * quotient
                coeff = ΔθeEW * κEΔxW / den2
                cc = c * coeff
                d  = 2 * κDκTWκE / den2
                dd = d * coeff
                p  = κWΔxE/κEΔxW
                ravelJ_[nrW-1:nrW+2] = -a - aa, -c - cc*p + d + dd*p + a - aa, c - cc - d + dd  # 界面左侧控制体

                a  = 2 * κEΔxW * quotient
                coeff = ΔθeEW * κWΔxE / den2
                aa = a * coeff
                c  = 2 * κDκTEκW / den2
                cc = c * coeff
                d  = 2 * κDκT_[nE] / (SθeE * Δx_[nE])
                dd = d * (θe_[nE] - θe_[nE+1]) / SθeE
                p  = κEΔxW/κWΔxE
                ravelJ_[nrE-1:nrE+2] = -a - aa - c - cc, a - aa*p + c - cc*p + d - dd, -d - dd  # # 界面右侧控制体

            ravelJ_Jintneg_ηintneg_[:]  = -solve_dJintdηint_(T, I0intneg_, ηintneg_)  # ∂FJintneg/∂ηintneg
            ravelJ_Jintpos_ηintpos_[:]  = -solve_dJintdηint_(T, I0intpos_, ηintpos_)  # ∂FJintpos/∂ηintpos
            if I0intnegUnknown:
                ravelJ_Jintneg_I0intneg_[:]   = -solve_dJintdI0int_(T, ηintneg_)      # ∂FJintneg/∂I0intneg
                ravelJ_I0intneg_θsnegsurf_[:] = -solve_dI0intdθssurf_(kneg, θsnegsurf_, θeneg_, I0intneg_)  # ∂FI0intneg/∂θsnegsurf
                ravelJ_I0intneg_θeneg_[:]     = -solve_dI0intdθe_(θeneg_, I0intneg_)  # ∂FI0intneg/∂θe
            if I0intposUnknown:
                ravelJ_Jintpos_I0intpos_[:]   = -solve_dJintdI0int_(T, ηintpos_)      # ∂FJintpos/∂I0intpos
                ravelJ_I0intpos_θspossurf_[:] = -solve_dI0intdθssurf_(kpos, θspossurf_, θepos_, I0intpos_)  # ∂FI0intpos/∂θspossurf
                ravelJ_I0intpos_θepos_[:]     = -solve_dI0intdθe_(θepos_, I0intpos_)  # ∂FI0intpos/∂θe
            ravelJ_ηintneg_θsnegsurf_[:] = solve_dUOCPdθsneg_(θsnegsurf_)  # ∂Fηintneg/∂θsnegsurf
            ravelJ_ηintpos_θspossurf_[:] = solve_dUOCPdθspos_(θspossurf_)  # ∂Fηintpos/∂θspossurf
            if lithiumPlating:
                ravelJ_JLP_θeneg_[:] = -solve_dJLPdθe_(T, θeneg_, I0LP_, ηLP_)  # ∂FJLP/∂ce
                ravelJ_JLP_ηLP_[:]   = -solve_dJLPdηLP_(T, I0LP_, ηLP_)         # ∂FJLP/∂ce

            if (self.banded_experience_of_J__ is None) and any(data['I']):
                self.banded_experience_of_J__ = expe = P2Dbase.banded_experience(J__)
                if self.verbose:
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
                raise P2Dbase.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现nan')
            if (θe_<=0).any():
                raise P2Dbase.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θe<=0')
            if (θsnegsurf_<=0).any():
                raise P2Dbase.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θsnegsurf<=0')
            if (θsnegsurf_>=1).any():
                raise P2Dbase.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θsnegsurf>=1')
            if (θspossurf_<=0).any():
                raise P2Dbase.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θspossurf<=0')
            if (θspossurf_>=1).any():
                raise P2Dbase.Error(f'时刻t = {self.t}s，时间步长{Δt = }s，Newton迭代出现θspossurf>=1')

            ΔX_ = abs(ΔX_)
            maxΔθ = ΔX_[s_θ].max()  # 新旧浓度场最大绝对误差
            maxΔφ = ΔX_[s_φ].max()  # 新旧电势场最大绝对误差
            maxΔJ = ΔX_[s_J].max()  # 新旧局部体积电流密度场最大绝对误差
            if maxΔθ<1e-3 and maxΔφ<1e-3 and maxΔJ/(abs(I)+0.001)<1e-3:
                break
        else:
            if self.verbose:
                print(f'时刻t = {self.t}s，Newton迭代达到最大次数{nNewton}，'
                      f'{maxΔθ = :.4f}，{maxΔφ = :.6f} V，{maxΔJ = :.3f} A')

        # Newton迭代收敛，更新状态量
        if decouple_cs:
            match timeDiscretization:
                case 'CN':
                    self.θsneg__[:] = θsnegI__ + outer(γneg_, Jintneg_ + self.Jintneg_)
                    self.θspos__[:] = θsposI__ + outer(γpos_, Jintpos_ + self.Jintpos_)
                case 'backward':
                    self.θsneg__[:] = θsnegI__ + outer(γneg_, Jintneg_)
                    self.θspos__[:] = θsposI__ + outer(γpos_, Jintpos_)
        else:
            self.θsneg__[:] = θsneg_.reshape(Nr, Nneg, order='F')
            self.θspos__[:] = θspos_.reshape(Nr, Npos, order='F')
        self.θsnegsurf_[:] = self.solve_θsnegsurf_(Qneg, Dsneg, self.θsneg__, Jintneg_)
        self.θspossurf_[:] = self.solve_θspossurf_(Qpos, Dspos, self.θspos__, Jintpos_)
        # self.θsnegsurf_[:] = θsnegsurf_
        # self.θspossurf_[:] = θspossurf_
        # a = max(abs(self.solve_θsnegsurf_(self.θsneg__, Jintneg_) - X_[s_θsnegsurf]).max(),
        #         abs(self.solve_θspossurf_(self.θspos__, Jintpos_) - X_[s_θspossurf]).max()  )
        # if a>1e-12: print(a)
        self.θe_[:] = θe_
        self.φsneg_[:] = φsneg_
        self.φspos_[:] = φspos_
        self.φe_[:] = φe_
        self.Jintneg_[:] = Jintneg_
        self.Jintpos_[:] = Jintpos_
        self.Jneg_[:] = Jintneg_
        self.Jpos_[:] = Jintpos_
        if doubleLayerEffect:
            self.JDLneg_[:] = JDLneg_
            self.JDLpos_[:] = JDLpos_
            self.Jneg_ += JDLneg_
            self.Jpos_ += JDLpos_
        self.I0intneg_[:] = I0intneg_
        self.I0intpos_[:] = I0intpos_
        self.ηintneg_[:] = ηintneg_
        self.ηintpos_[:] = ηintpos_
        if lithiumPlating:
            self.JLP_[:] = JLP_
            self.Jneg_ += JLP_

        self.ηLPneg_[:] = φsneg_ - φeneg_ - RSEIneg*self.Jneg_
        self.ηLPpos_[:] = φspos_ - φepos_ - RSEIpos*self.Jpos_

        return nNewton  # 返回Newton迭代次数

    def count_lithium(self):
        """统计锂电荷量"""
        qsneg = self.θsneg*self.Qneg  # 负极固相锂电荷量 [Ah]
        qspos = self.θspos*self.Qpos  # 正极固相锂电荷量 [Ah]
        qe = (self.θe_*self.Δx_*self.qe_).sum()/3600  # 电解液锂电荷量 [Ah]
        qtot = qsneg + qspos + qe + self.QLP
        print(f'合计锂电荷总量 {qtot: .6f} Ah = '
              f'负极嵌锂{qsneg:.6f} Ah + 正极嵌锂{qspos:.6f} Ah'
              f' + 电解液锂{qe:.6f} Ah'
              f' + 负极析锂{self.QLP:.6f} Ah')

    @property
    def I0intneg(self):
        """负极主反应交换电流 [A]"""
        return self.Arrhenius(self._I0intneg, self.Ekneg)
    @I0intneg.setter
    def I0intneg(self, I0intneg):
        self._I0intneg = I0intneg

    @property
    def I0intpos(self):
        """正极主反应交换电流 [A]"""
        return self.Arrhenius(self._I0intpos, self.Ekpos)
    @I0intpos.setter
    def I0intpos(self, I0intpos):
        self._I0intpos = I0intpos

    @property
    def I0LP(self):
        """负极析锂反应交换电流 [A]"""
        return self.Arrhenius(self._I0LP, self.EkLP)
    @I0LP.setter
    def I0LP(self, I0LP):
        self._I0LP = I0LP

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
        """(Ne,) 各控制体集总电解液锂离子扩散系数 [A]"""
        De3κ_ = self.De * array([self._κneg, self._κsep, self._κpos])
        De3κ_ = self.Arrhenius(De3κ_, self.EDe)
        return concatenate([
            full(self.Nneg, De3κ_[0]),
            full(self.Nsep, De3κ_[1]),
            full(self.Npos, De3κ_[2])])

    @property
    def qe_(self):
        """(Ne,) 各控制体电解液锂离子电荷量 [C]"""
        return concatenate([
            full(self.Nneg, self.qeneg),
            full(self.Nsep, self.qesep),
            full(self.Npos, self.qepos),])

    @property
    def U(self):
        """正负极端电压 [V]"""
        a = 0.5*self.I
        φsposCollector = self.φspos_[-1] - a*self.Δxpos/self.σpos
        φsnegCollector = self.φsneg_[0]  + a*self.Δxneg/self.σneg
        return φsposCollector - φsnegCollector

    def solve_θsnegsurf_(self, Qneg, Dsneg, θsneg__, Jintneg_):
        """求解负极固相表面无量纲锂离子浓度场θsnegsurf_ [–]"""
        if self.decouple_cs:
            return self.coeffsExpl_.dot(θsneg__[-3:])
        else:
            Nr = self.Nr
            θsneg_ = θsneg__.ravel('F')
            c_ = self.coeffs_csneg_
            return -(  c_[-3] * θsneg_[Nr-3::Nr]
                     + c_[-2] * θsneg_[Nr-2::Nr]
                     + c_[-1] * θsneg_[Nr-1::Nr]
                     + 1/(10800*Qneg*Dsneg) * Jintneg_
                     )/self.coeff_csnegsurf_csnegsurf

    def solve_θspossurf_(self, Qpos, Dspos, θspos__, Jintpos_):
        """求解正极固相表面无量纲锂离子浓度场θspossurf_ [–]"""
        if self.decouple_cs:
            return self.coeffsExpl_.dot(θspos__[-3:])
        else:
            Nr = self.Nr
            θspos_ = θspos__.ravel('F')
            c_ = self.coeffs_cspos_
            return -(  c_[-3] * θspos_[Nr-3::Nr]
                     + c_[-2] * θspos_[Nr-2::Nr]
                     + c_[-1] * θspos_[Nr-1::Nr]
                     + 1/(10800*Qpos*Dspos) * Jintpos_
                     )/self.coeff_cspossurf_cspossurf

    @property
    def θeneg_(self):
        """(Nneg,) 负极区域电解液无量纲锂离子浓度 [–]"""
        return self.θe_[:self.Nneg]

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
        θeInterfaces_ = hstack([θe_[0], (θe_[:-1] + θe_[1:])/2, θe_[-1]])  # 各控制体界面的锂离子浓度
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
        gradθeWest_ = hstack([0, (θe_[1:] - θe_[:-1])/ΔxWest_[1:]])  # 各控制体左界面的锂离子浓度梯度
        gradθeEast_ = hstack([(θe_[1:] - θe_[:-1])/ΔxEast_[:-1], 0])  # 各控制体右界面的锂离子浓度梯度
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、修正隔膜-正极界面
            gradθeEast_[nW] = (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW])
            gradθeWest_[nE] = (θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE])
        gradlnθeWest_ = gradθeWest_/θeWest_  # 各控制体左界面的对数锂离子浓度梯度 [ln –/–]
        gradlnθeEast_ = gradθeEast_/θeEast_  # 各控制体右界面的对数锂离子浓度梯度 [ln –/–]
        φeInterfaces_ = hstack([φe_[0], (φe_[:-1] + φe_[1:])/2, φe_[-1]])
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

    @staticmethod
    def solve_Jint_(T, I0int_, ηint_) -> ndarray:
        """求解主反应局部体积电流密度Jint [A]"""
        return 2*I0int_*sinh(P2Dbase.F/(2*P2Dbase.R*T)*ηint_)

    @staticmethod
    def solve_dJintdI0int_(T, ηint_) -> ndarray:
        """求解主反应局部体积电流密度Jint对交换电流密度I0int的偏导数 [A/A]"""
        return 2*sinh(P2Dbase.F/(2*P2Dbase.R*T)*ηint_)

    @staticmethod
    def solve_dJintdηint_(T, I0int_, ηint_) -> ndarray:
        """求解主反应局部体积电流密度Jint对过电位ηint的偏导数 [A/V]"""
        FRT = P2Dbase.F/(P2Dbase.R*T)
        return FRT*I0int_*cosh(FRT*0.5*ηint_)

    @staticmethod
    def solve_I0int_(k, θssurf_, θe_) -> ndarray:
        """由固液相浓度场求主反应交换电流密度I0int [A]"""
        return k * sqrt(θe_*(1 - θssurf_)*θssurf_)

    @staticmethod
    def solve_dI0intdθe_(θe_, I0int_):
        """求解主反应交换电流密度I0int对电解液无量纲锂离子浓度θe的偏导数  [A/-]"""
        return 0.5*I0int_/θe_

    @staticmethod
    def solve_dI0intdθssurf_(k, θssurf_, θe_, I0int_):
        """求解主反应交换电流密度I0int对固相颗粒表面无量纲锂离子浓度θssurf的偏导数 [A/-]"""
        return k*k * θe_*(0.5 - θssurf_)/I0int_

    @staticmethod
    def solve_JLP_(T, I0LP_, ηLP_) -> ndarray:
        """求解析锂反应局部体积电流密度JLP [A]"""
        FRT = P2Dbase.F/P2Dbase.R/T
        a, b = 0.3*FRT, -0.7*FRT
        JLP_ = I0LP_*(exp(a*ηLP_) - exp(b*ηLP_))
        JLP_[ηLP_>=0] = 0
        return JLP_

    @staticmethod
    def solve_dJLPdθe_(T, θeneg_, I0LP_, ηLP_):
        """析锂反应局部体积电流密度JLP对电解液锂离子浓度θe的偏导数"""
        FRT = P2Dbase.F/P2Dbase.R/T
        a, b = 0.3*FRT, -0.7*FRT
        dJLPdI0LP_ = exp(a*ηLP_) - exp(b*ηLP_)
        dI0LPdθe_ = 0.3*I0LP_/θeneg_
        dJLPdθe_ = dJLPdI0LP_*dI0LPdθe_
        dJLPdθe_[ηLP_>=0] = 0
        return dJLPdθe_

    @staticmethod
    def solve_dJLPdηLP_(T, I0LP_, ηLP_):
        """求解析锂反应局部体积电流密度JLP对析锂过电位ηLP的偏导数 [A/V]"""
        FRT = P2Dbase.F/(P2Dbase.R*T)
        a, b = 0.3*FRT, -0.7*FRT
        dJLPdηLP_ = I0LP_*(a*exp(a*ηLP_) - b*exp(b*ηLP_))
        dJLPdηLP_[ηLP_>=0] = 0
        return dJLPdηLP_

    @property
    def I0LP_(self):
        """(Nneg,) 析锂反应交换电流密度场[A]"""
        return full(self.Nneg, self.I0LP) if self._I0LP else\
            LPP2D.solve_I0LP_(self.kLP, self.θeneg_)

    @staticmethod
    def solve_I0LP_(kLP, θeneg_) -> ndarray:
        """求解析锂反应交换电流密度I0LP [A]"""
        return kLP * θeneg_**0.3

    @property
    def gradlnθe_(self):
        """对数电解液锂离子浓度场的梯度 [(ln –)/–]"""
        Nneg, Nsep = self.Nneg, self.Nsep
        x_, Δx_, ΔxWest_, ΔxEast_ = self.x_, self.Δx_, self.ΔxWest_, self.ΔxEast_
        θe_ = self.θe_
        θeInterfaces_ = self.θeInterfaces_
        θeWest_ = θeInterfaces_[:-1]  # 各控制体左界面的电解液锂离子浓度
        θeEast_ = θeInterfaces_[1:]   # 各控制体右界面的电解液锂离子浓度
        gradθe_ = hstack([
            (θe_[1] - θe_[0])/(x_[1] - x_[0]) * 0.5,       # 负极首个控制体
            (θe_[2:] - θe_[:-2])/(x_[2:] - x_[:-2]),       # 内部控制体
            (θe_[-1] - θe_[-2])/(x_[-1] - x_[-2]) * 0.5])  # 正极末尾控制体
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            gradθe_[nW] = ((θe_[nW] - θe_[nW-1])  /ΔxWest_[nW]  + (θeEast_[nW] - θe_[nW])/(0.5*Δx_[nW])) * 0.5  # 界面左侧控制体
            gradθe_[nE] = ((θe_[nE] - θeWest_[nE])/(0.5*Δx_[nE]) + (θe_[nE+1] - θe_[nE]) / ΔxEast_[nE]) * 0.5   # 界面右侧控制体
        return gradθe_/θe_

    @property
    def gradφsneg_(self):
        """负极固相电势场的梯度 [V/m]"""
        φsneg_ = self.φsneg_
        Δxneg = self.Δxneg
        gradφsneg_ = hstack([
            (-self.I/self.σneg + (φsneg_[1] - φsneg_[0])/Δxneg) * 0.5, # 负极首个控制体
            (φsneg_[2:] - φsneg_[:-2])/(2*Δxneg),      # 负极内部控制体
            (φsneg_[-1] - φsneg_[-2])/Δxneg * 0.5])  # 负极末尾控制体
        return gradφsneg_

    @property
    def gradφspos_(self):
        """正极固相电势场的梯度 [V/m]"""
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
        dUOCPdTnegsurf_ = self.dUOCPdTneg(self.θsnegsurf_) if callable(self.dUOCPdTneg) else self.dUOCPdTneg
        return (self.T*dUOCPdTnegsurf_*self.Jintneg_).mean()

    @property
    def Qrevpos(self):
        """正极可逆热 [W]"""
        dUOCPdTpossurf_ = self.dUOCPdTpos(self.θspossurf_) if callable(self.dUOCPdTpos) else self.dUOCPdTpos
        return (self.T*dUOCPdTpossurf_*self.Jintpos_).mean()

    @property
    def θsneg(self):
        """负极嵌锂状态"""
        return self.Vr_.dot(self.θsneg__).mean()

    @property
    def θspos(self):
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

    plot_θ = P2Dbase.plot_c  # 作图：浓度场
    plot_Jint_I0int_ηint = P2Dbase.plot_jint_i0int_ηint  # 作图：主反应局部体积电流密度、过电位、交换电流密度
    plot_JDL = P2Dbase.plot_jDL          # 作图：双电层效应局部体积电流密度、电流
    plot_θsr = P2Dbase.plot_csr          # 作图：固相颗粒径向锂离子浓度场
    plot_JLP_ηLP = P2Dbase.plot_jLP_ηLP  # 作图：负极析锂局部体积电流密度-空间、时间

    def initialize_consistent(self,
            θsneg__: ndarray,
            θspos__: ndarray,
            θe_: ndarray,
            I: float | int = 0):
        """一致性初始化
        已知：θsneg__、θspos__、θe_、I
        求解：θsnegsurf_、θspossurf__、φsneg_、φspos_、φe_、Jintneg_、Jintpos、I0intneg_、I0intpos_、ηintneg_、ηintpos_
        令：JDLneg_ = JDLpos_ = JLP_ = 0
        """
        Nr, Nneg, Nsep, Npos, Ne = self.Nr, self.Nneg, self.Nsep, self.Npos, self.Ne  # 读取：网格数
        assert θsneg__.shape==(Nr, Nneg), f'负极固相颗粒内部无量纲锂离子浓度θsneg__.shape应为({Nr}, {Nneg})'
        assert θsneg__.shape==(Nr, Nneg), f'正极固相颗粒内部无量纲锂离子浓度θspos__.shape应为({Nr}, {Npos})'
        assert θe_.shape==(self.Ne,), f'电解液无量纲锂离子浓度θe_.shape应为({self.Ne},)'
        assert ((0<=θsneg__) & (θsneg__<=1)).all(), 'θsneg__取值范围应为(0, 1)'
        assert ((0<=θspos__) & (θspos__<=1)).all(), 'θspos__取值范围应为(0, 1)'
        assert (0<θe_).all(), 'θe_取值应大于0'
        θeneg_, θepos_ = θe_[:Nneg], θe_[-Npos:]

        if self.ravelK_ is None:
            self._generate_K__bK_and_slices()

        # 更新K__矩阵的参数相关值
        self.update_K__with_pure_electrochemical_parameters()

        # 生成索引
        sK = self.sK  # 原索引
        s2idx = P2Dbase.s2idx
        idx_ = concatenate([
            s2idx(sK.s_csnegsurf), s2idx(sK.s_cspossurf),
            s2idx(sK.s_φsneg),     s2idx(sK.s_φspos), s2idx(sK.s_φe),
            s2idx(sK.s_jintneg),   s2idx(sK.s_jintpos),
            s2idx(sK.s_i0intneg),  s2idx(sK.s_i0intpos),
            s2idx(sK.s_ηintneg),   s2idx(sK.s_ηintpos),])
        Kinit__ = self.ravelK_.base[ix_(idx_, idx_)]  # 提取K__矩阵
        NKinit = Kinit__.shape[0]  # Kinit__矩阵的行列数
        bKinit_ = zeros(NKinit)    # 常数项向量
        start = 0
        def reassign(s_old: slice) -> slice:
            """对矩阵Kinit__重新安排切片索引"""
            nonlocal start
            N = s_old.stop - s_old.start
            s_new = slice(start, start + N)
            start += N
            return s_new
        s_θsnegsurf = reassign(sK.s_csnegsurf)
        s_θspossurf = reassign(sK.s_cspossurf)
        s_φsneg = reassign(sK.s_φsneg)
        s_φspos = reassign(sK.s_φspos)
        s_φe = reassign(sK.s_φe)
        s_Jintneg = reassign(sK.s_jintneg)
        s_Jintpos = reassign(sK.s_jintpos)
        s_I0intneg = reassign(sK.s_i0intneg)
        s_I0intpos = reassign(sK.s_i0intpos)
        s_ηintneg = reassign(sK.s_ηintneg)
        s_ηintpos = reassign(sK.s_ηintpos)

        # 读取方法
        solve_Jint_ = LPP2D.solve_Jint_
        solve_dJintdI0int_ = LPP2D.solve_dJintdI0int_
        solve_dJintdηint_  = LPP2D.solve_dJintdηint_
        solve_I0int_ = LPP2D.solve_I0int_
        solve_dI0intdθssurf_ = LPP2D.solve_dI0intdθssurf_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_
        solve_dUOCPdθsneg_ = self.solve_dUOCPdθsneg_
        solve_dUOCPdθspos_ = self.solve_dUOCPdθspos_

        # 读取参数
        ΔxWest_, ΔxEast_, Δx_ = self.ΔxWest_, self.ΔxEast_, self.Δx_
        T = self.T   # 温度
        F2RT = P2Dbase.F/(2*P2Dbase.R*T)
        κ_ = self.κ_
        κDκT_ = self.κD * T * κ_
        RSEIneg = self.RSEIneg
        RSEIpos = self.RSEIpos
        if I0intnegUnknown := (self._I0intneg is None):
            kneg = self.kneg          # 读取：负极集总主反应速率常数 [A]
        else:
            I0intneg = self.I0intneg  # 读取：负极集总主反应交换电流密度 [A]
        if I0intposUnknown := (self._I0intpos is None):
            kpos = self.kpos          # 读取：正极集总主反应速率常数 [A]
        else:
            I0intpos = self.I0intpos  # 读取：正极集总主反应交换电流密度 [A]

        # 外推表面浓度
        c_ = self.coeffsExpl_
        θsnegsurfExpl_ = c_.dot(θsneg__[-3:])
        θspossurfExpl_ = c_.dot(θspos__[-3:])

        ## 对Kinit__的右端项bKinit_赋值 ##
        if self.decouple_cs:
            # 强制表面浓度约束：认为 θsnegsurf_、θspossurf_ 是外推得到的已知值
            bKinit_[s_θsnegsurf] = θsnegsurfExpl_
            bKinit_[s_θspossurf] = θspossurfExpl_
        else:
            # 用颗粒扩散边界条件 关联Jint、θssurf以及靠近颗粒表面的3个内部节点浓度
            θsneg_ = θsneg__.ravel('F')
            θspos_ = θspos__.ravel('F')
            c_ = self.coeffs_csneg_
            bKinit_[s_θsnegsurf] = -(
                      c_[-3] * θsneg_[Nr-3::Nr]
                    + c_[-2] * θsneg_[Nr-2::Nr]
                    + c_[-1] * θsneg_[Nr-1::Nr])
            c_ = self.coeffs_cspos_
            bKinit_[s_θspossurf] =  -(
                      c_[-3] * θspos_[Nr-3::Nr]
                    + c_[-2] * θspos_[Nr-2::Nr]
                    + c_[-1] * θspos_[Nr-1::Nr])
        # 固相电流边界条件
        bKinit_[s_φsneg.start]    = -self.Δxneg*I/self.σneg
        bKinit_[s_φspos.stop - 1] =  self.Δxpos*I/self.σpos
        # 电解液电势方程的电解液锂离子浓度项
        q_ = 2*(θe_[1:] - θe_[:-1])/(θe_[1:] + θe_[:-1])  # (Ne-1,)
        c_ = κDκT_[:-1]*q_ / ΔxEast_[:-1]  # (Ne-1,)
        a_ = κDκT_[1:] *q_ / ΔxWest_[1:]   # (Ne-1,)
        bKinit_φe_ = bKinit_[s_φe]
        bKinit_φe_[:] = hstack([c_, 0]) - hstack([0, a_])
        for (nW, nE) in ([Nneg - 1, Nneg], [Nneg + Nsep - 1, Nneg + Nsep]):
            # 修正负极-隔膜界面、隔膜-正极界面
            a, b = κ_[nE]*Δx_[nW], κ_[nW]*Δx_[nE]
            θinterface = (a*θe_[nE] + b*θe_[nW])/(a + b)
            bKinit_φe_[nW] = (κDκT_[nW] * (θinterface - θe_[nW])  / (0.5*Δx_[nW]) / θinterface
                            - κDκT_[nW] * (θe_[nW] - θe_[nW-1])   / ΔxWest_[nW]   / (0.5*(θe_[nW] + θe_[nW-1])))
            bKinit_φe_[nE] = (κDκT_[nE] * (θe_[nE+1] - θe_[nE])   / ΔxEast_[nE]   / (0.5*(θe_[nE+1] + θe_[nE]))
                            - κDκT_[nE] * (θe_[nE] - θinterface)  / (0.5*Δx_[nE]) / θinterface)

        ## Newton迭代初值 ##
        X_ = zeros(NKinit)
        # 索引解向量
        θsnegsurf_ = X_[s_θsnegsurf]
        θspossurf_ = X_[s_θspossurf]
        φsneg_ = X_[s_φsneg]
        φspos_ = X_[s_φspos]
        φe_ = X_[s_φe]
        Jintneg_ = X_[s_Jintneg]
        Jintpos_ = X_[s_Jintpos]
        I0intneg_ = X_[s_I0intneg] if I0intnegUnknown else I0intneg
        I0intpos_ = X_[s_I0intpos] if I0intposUnknown else I0intpos
        ηintneg_ = X_[s_ηintneg]
        ηintpos_ = X_[s_ηintpos]

        # 初始化解向量
        θsnegsurf_[:] = θsnegsurfExpl_
        θspossurf_[:] = θspossurfExpl_
        Jintneg_[:] = Jintneg = I
        Jintpos_[:] = Jintpos = -I
        if I0intnegUnknown:
            I0intneg_[:] = solve_I0int_(kneg, θsnegsurfExpl_, θeneg_)
        if I0intposUnknown:
            I0intpos_[:] = solve_I0int_(kpos, θspossurfExpl_, θepos_)
        ηintneg_[:] = arcsinh(Jintneg/(2*I0intneg_))/F2RT
        ηintpos_[:] = arcsinh(Jintpos/(2*I0intpos_))/F2RT
        φsneg_[:] = ηintneg_ + RSEIneg*Jintneg + solve_UOCPneg_(θsnegsurfExpl_)
        φspos_[:] = ηintpos_ + RSEIpos*Jintpos + solve_UOCPpos_(θspossurfExpl_)

        # Newton迭代
        J__ = Kinit__.copy()
        ravelJ_ = J__.ravel()
        dsr = DiagonalSliceRavel(NKinit)
        ravelJ_Jintneg_ηintneg_ = ravelJ_[dsr(s_Jintneg, s_ηintneg)]
        ravelJ_Jintpos_ηintpos_ = ravelJ_[dsr(s_Jintpos, s_ηintpos)]
        ravelJ_ηintneg_θsnegsurf_ = ravelJ_[dsr(s_ηintneg, s_θsnegsurf)]
        ravelJ_ηintpos_θspossurf_ = ravelJ_[dsr(s_ηintpos, s_θspossurf)]
        if I0intnegUnknown:
            ravelJ_Jintneg_I0intneg_ = ravelJ_[dsr(s_Jintneg, s_I0intneg)]
            ravelJ_I0intneg_θsnegsurf_ = ravelJ_[dsr(s_I0intneg, s_θsnegsurf)]
        if I0intposUnknown:
            ravelJ_Jintpos_I0intpos_ = ravelJ_[dsr(s_Jintpos, s_I0intpos)]
            ravelJ_I0intpos_θspossurf_ = ravelJ_[dsr(s_I0intpos, s_θspossurf)]

        for nNewton in range(1, 101):
            F_ = Kinit__.dot(X_) - bKinit_  # (Ninit,) F残差向量

            # F向量非线性部分
            F_[s_Jintneg] -= solve_Jint_(T, I0intneg_, ηintneg_)  # F向量Jintneg部分
            F_[s_Jintpos] -= solve_Jint_(T, I0intpos_, ηintpos_)  # F向量Jintpos部分
            if I0intnegUnknown:
                F_[s_I0intneg] -= solve_I0int_(kneg, θsnegsurf_, θeneg_)   # F向量I0intneg部分
            if I0intposUnknown:
                F_[s_I0intpos] -= solve_I0int_(kpos, θspossurf_, θepos_)  # F向量I0intpos部分
            F_[s_ηintneg] += solve_UOCPneg_(θsnegsurf_)  # F向量ηintneg非线性部分
            F_[s_ηintpos] += solve_UOCPpos_(θspossurf_)  # F向量ηintpos非线性部分
            # 更新Jacobi矩阵
            ravelJ_Jintneg_ηintneg_[:] = -solve_dJintdηint_(T, I0intneg_, ηintneg_)  # ∂FJintneg/∂ηintneg
            ravelJ_Jintpos_ηintpos_[:] = -solve_dJintdηint_(T, I0intpos_, ηintpos_)  # ∂FJintpos/∂ηintpos
            if I0intnegUnknown:
                ravelJ_Jintneg_I0intneg_[:]  = -solve_dJintdI0int_(T, ηintneg_)   # ∂FJintneg/∂I0intneg
                ravelJ_I0intneg_θsnegsurf_[:] = -solve_dI0intdθssurf_(T, θsnegsurf_, θeneg_, I0intneg_)  # ∂FI0intneg/∂θsnegsurf
            if I0intposUnknown:
                ravelJ_Jintpos_I0intpos_[:]   = -solve_dJintdI0int_(T, ηintpos_)  # ∂FJintpos/∂I0intpos
                ravelJ_I0intpos_θspossurf_[:] = -solve_dI0intdθssurf_(T, θspossurf_, θepos_, I0intpos_)  # ∂FI0intpos/∂θspossurf
            ravelJ_ηintneg_θsnegsurf_[:] = solve_dUOCPdθsneg_(θsnegsurf_)  # ∂Fηintneg/∂θsnegsurf
            ravelJ_ηintpos_θspossurf_[:] = solve_dUOCPdθspos_(θspossurf_)  # ∂Fηintpos/∂θspossurf

            ΔX_ = solve(J__, F_)

            X_ -= ΔX_

            if (maxΔX := abs(ΔX_).max())<1e-5:
                break
        else:
            raise P2Dbase.Error(f'一致性初始化失败，Newton迭代{nNewton = }次，不收敛，{maxΔX = }')

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
        self.Jintneg_[:] = Jintneg_
        self.Jintpos_[:] = Jintpos_
        self.JDLneg_[:] = 0
        self.JDLpos_[:] = 0
        self.I0intneg_[:] = I0intneg_
        self.I0intpos_[:] = I0intpos_
        self.ηintneg_[:] = ηintneg_
        self.ηintpos_[:] = ηintpos_

        if self.lithiumPlating:
            self.JLP_[:] = 0

        self.Jneg_[:] = Jintneg_
        self.Jpos_[:] = Jintpos_
        self.ηLPneg_[:] = φsneg_ - self.φeneg_ - RSEIneg*Jintneg_
        self.ηLPpos_[:] = φspos_ - self.φepos_ - RSEIpos*Jintpos_

        if self.verbose:
            print(f'一致性初始化完成。Newton迭代{nNewton = }。Consistent initial conditions are solved! ')


if __name__=='__main__':
    import numpy as np
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
    cell.CC(-10, 2000).CC(0, 300).CC(15, 800).CC(8, 600)
    cell.count_lithium()

    '''
    cell.plot_UI()
    cell.plot_TQgen()
    cell.plot_SOC()
    cell.plot_θ(np.arange(0, 2001, 200))
    cell.plot_φ(np.arange(0, 2001, 200))
    cell.plot_Jint_I0int_ηint(np.arange(0, 2001, 200))
    cell.plot_JDL(np.arange(0, 2001, 200))
    cell.plot_θsr(np.arange(0, 2001, 200), 1)
    cell.plot_JLP_ηLP(np.arange(0, 2001, 200))
    cell.plot_LP()
    cell.plot_OCV_OCP()
    cell.plot_dUOCPdθs()
    cell.plot_nNewton()
    '''
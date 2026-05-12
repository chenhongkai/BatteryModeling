#%%
from numpy import ndarray,\
    array, zeros, zeros_like, ones, empty, full, hstack, concatenate, stack, \
    exp, sqrt, cos, sin, sinh, cosh, arcsinh, outer, \
    ix_, isnan
from numpy.linalg import solve


from P2Dmodel.P2Dbase import P2Dbase
from P2Dmodel.OCP import NMC111, Graphite
from P2Dmodel.tools import DiagonalSliceRavel


class LPJTFP2D(P2Dbase):
    """锂离子电池集总参数准二维模型 Lumped-Parameter Pseudo-two-Dimension model"""

    __slots__ = (
        # 专有参数名
        'Qcell', 'Qneg', 'Qpos', 'qeneg', 'qesep', 'qepos',
        '_κneg', '_κsep', '_κpos', 'De', 'κD',
        '_I0intneg', '_I0intpos', '_I0LP',
        # 专有时域状态量
        'θsneg__', 'θspos__', 'θsnegsurf_', 'θspossurf_', 'θe_',
        'Jintneg_', 'Jintpos_', 'JDLneg_', 'JDLpos_',
        'I0intneg_', 'I0intpos_',
        'Jneg_', 'Jpos_',
        'JLP_',
        # 专有频域状态量
        'REθsnegsurf__', 'IMθsnegsurf__', 'REθspossurf__', 'IMθspossurf__', 'REθe__', 'IMθe__',
        'REJintneg__',   'IMJintneg__',   'REJintpos__',   'IMJintpos__',
        'REJDLneg__',    'IMJDLneg__',    'REJDLpos__',    'IMJDLpos__',
        'REI0intneg__',  'IMI0intneg__',  'REI0intpos__',  'IMI0intpos__',
        'REJLP__',       'IMJLP__',
        # 专有恒定量
        'r_', 'Δr_',
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
        Nneg, Npos, Ne, Nr = self.Nneg, self.Npos, self.Ne, self.Nr  # 读取：网格数
        θsneg = θminneg + SOC0*(θmaxneg - θminneg)  # 初始负极嵌锂状态 [–]
        θspos = θmaxpos - SOC0*(θmaxpos - θminpos)  # 初始正极嵌锂状态 [–]
        self.θsneg__ = full((Nr, Nneg), θsneg)
        self.θspos__ = full((Nr, Npos), θspos)  # 初始化：负极、正极固相内部无量纲锂离子浓度场 [–]
        self.θsnegsurf_ = full(Nneg, θsneg)
        self.θspossurf_ = full(Npos, θspos)  # 初始化：负极、正极固相表面无量纲锂离子浓度场 [–]
        self.θe_ = ones(Ne)                  # 初始化：电解液无量纲锂离子浓度场 [–]
        self.Jintneg_ = zeros(Nneg)
        self.Jintpos_ = zeros(Npos)  # 初始化：负极、正极集总主反应局部体积电流密度场 [A]
        self.JDLneg_ = zeros(Nneg)
        self.JDLpos_ = zeros(Npos)   # 初始化：负极、正极集总双电层效应局部体积电流密度场 [A]
        I0intneg = self.I0intneg if self._I0intneg else LPJTFP2D.solve_I0int_(self.kneg, θsneg, 1)
        I0intpos = self.I0intpos if self._I0intpos else LPJTFP2D.solve_I0int_(self.kpos, θspos, 1)
        self.I0intneg_ = full(Nneg, I0intneg)
        self.I0intpos_ = full(Npos, I0intpos)  # 初始化：负极、正极主反应集总交换电流密度场 [A]
        self.Jneg_ = zeros(Nneg)
        self.Jpos_ = zeros(Npos)     # 初始化：负极、正极总局部体积电流密度场 [A]
        if lithiumPlating := self.lithiumPlating:
            self.JLP_ = zeros(Nneg)  # 初始化：负极析锂反应集总局部体积电流密度场 [A]
        # 恒定量
        self.Δr_ = self.Δrneg_
        self.r_  = self.rneg_
        if self.complete:
            # 状态量
            Nf = self.f_.size
            self.REθsnegsurf__, self.IMθsnegsurf__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 负极固相表面浓度实部、虚部
            self.REθspossurf__, self.IMθspossurf__ = empty((Nf, Npos)), empty((Nf, Npos))  # 正极固相表面浓度实部、虚部
            self.REθe__, self.IMθe__ = empty((Nf, Ne)), empty((Nf, Ne))                    # 电解液锂离子浓度实部、虚部
            self.REJintneg__, self.IMJintneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))      # 负极主反应局部体积电流密度实部、虚部
            self.REJintpos__, self.IMJintpos__ = empty((Nf, Npos)), empty((Nf, Npos))      # 正极主反应局部体积电流密度实部、虚部
            self.REJDLneg__, self.IMJDLneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))        # 负极双电层局部体积电流密度实部、虚部
            self.REJDLpos__, self.IMJDLpos__ = empty((Nf, Npos)), empty((Nf, Npos))        # 正极双电层局部体积电流密度实部、虚部
            self.REI0intneg__, self.IMI0intneg__ = empty((Nf, Nneg)), empty((Nf, Nneg))    # 负极交换电流密度实部、虚部
            self.REI0intpos__, self.IMI0intpos__ = empty((Nf, Npos)), empty((Nf, Npos))    # 正极交换电流密度实部、虚部
            if lithiumPlating:
                self.REJLP__, self.IMJLP__ = empty((Nf, Nneg)), empty((Nf, Nneg))  # 析锂反应局部体积电流密度实部、虚部

            extra_datanames_ = [            # 需记录的数据名称
                'θsneg__', 'θspos__',        # 负极、正极固相无量纲锂离子浓度场 [–]
                'θsnegsurf_', 'θspossurf_',  # 负极、正极表面无量纲锂离子浓度场 [–]
                'θe_',                       # 电解液无量纲锂离子浓度场 [–]
                'Jintneg_', 'Jintpos_',      # 负极、正极主反应集总局部体积电流密度场 [A]
                'JDLpos_', 'JDLneg_',        # 负极、正极双电层效应集总局部体积电流密度场 [A]
                'I0intneg_', 'I0intpos_',]   # 负极、正极主反应集总交换电流密度场 [A]
            if lithiumPlating:
                extra_datanames_.append('JLP_')  # 负极析锂集总局部体积电流密度场 [A]
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

        if self.verbose and type(self) is LPJTFP2D:
            print(self)
            print('集总参数时频联合P2D模型(LPJTFP2D)初始化完成!')

    def update_K__with_pure_electrochemical_parameters(self):
        # 对K__矩阵赋纯电化学参数相关值
        if self.decouple:
            pass
        else:
            self._update_K__θsnegsurf_Jintneg_when_coupling(self.Qneg, self.Dsneg)
            self._update_K__θspossurf_Jintpos_when_coupling(self.Qpos, self.Dspos)
        self._update_K__φsneg_Jneg(self.σneg)
        self._update_K__φspos_Jpos(self.σpos)
        self._update_K__φe_φe(κ_ := self.κ_, κ_)
        self._update_K__ηintneg_Jneg(self.RSEIneg)
        self._update_K__ηintpos_Jpos(self.RSEIpos)
        if self.lithiumPlating:
            self._update_K__ηLP_Jneg(self.RSEIneg)
    
    def _update_K__bK_θsnegsurf_Jintneg_when_decoupling(self, Dsneg, Qneg, Δt, old_θsneg__, old_jintneg_):
        # 更新K__矩阵θsnegsurf行Jintneg列
        # 更新bK_向量θsnegsurf行
        Kθsjintneg = self._solve_KθsJint(Qneg, Δt)
        θsnegI__, γneg_ = P2Dbase._update_K__bK_csnegsurf_jintneg_when_decoupling(
            self, Dsneg, Kθsjintneg, Δt, old_θsneg__, old_jintneg_)
        return θsnegI__, γneg_

    def _update_K__bK_θspossurf_Jintpos_when_decoupling(self, Dspos, Qpos, Δt, old_θspos__, old_Jintpos_):
        # 更新K__矩阵θspossurf行Jintpos列
        # 更新bK_向量θspossurf行
        KθsJintpos = self._solve_KθsJint(Qpos, Δt)
        θsposI__, γpos_ = P2Dbase._update_K__bK_cspossurf_jintpos_when_decoupling(
            self, Dspos, KθsJintpos, Δt, old_θspos__, old_Jintpos_)
        return θsposI__, γpos_

    def _update_K__bK_θsneg_θsneg_Jintneg_when_coupling(self, Dsneg, Qneg, Δt, old_θsneg__, old_Jintneg_):
        # 更新K__矩阵θsneg行θsneg列
        # 更新K__矩阵θsneg末尾球壳控制体行Jintneg列
        # 更新bK_向量θsneg行
        KθsJintneg = self._solve_KθsJint(Qneg, Δt)
        P2Dbase._update_K__bK_csneg_csneg_jintneg_when_coupling(
            self, Dsneg, Δt, KθsJintneg, old_θsneg__, old_Jintneg_)

    def _update_K__bK_θspos_θspos_Jintpos_when_coupling(self, Dspos, Qpos, Δt, old_θspos__, old_Jintpos_):
        # 更新K__矩阵θspos行θspos列
        # 更新K__矩阵θspos末尾球壳控制体行Jintpos列
        # 更新bK_向量θspos行
        KθsJintpos = self._solve_KθsJint(Qpos, Δt)
        P2Dbase._update_K__bK_cspos_cspos_jintpos_when_coupling(
            self, Dspos, Δt, KθsJintpos, old_θspos__, old_Jintpos_)

    def _solve_KθsJint(self,
                       Qreg: float,  # 电极容量 [Ah]
                       Δt: float,  # 时间步长 [s]
                       ) -> float:
        Δ = 1 - self.Δr_[-1]
        return Δt/(10800*Qreg)/((1 - Δ*Δ*Δ)/3)
    
    def _update_K__θsnegsurf_Jintneg_when_coupling(self, Qneg, Dsneg):
        # 更新K__矩阵θsnegsurf行Jintneg列
        self.ravelK_[self.sK.sr_csnegsurf_jintneg] = 1/(10800*Qneg*Dsneg)

    def _update_K__θspossurf_Jintpos_when_coupling(self, Qpos, Dspos):
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
        # 更新bK_向量φsneg行首元、φspos行末元（固相电流边界条件）
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
        decouple = self.decouple
        doubleLayerEffect = self.doubleLayerEffect

        ravelK_ = self.ravelK_  # 读取：因变量线性矩阵K__展平视图
        bK_ = self.bK_          # 读取：常数项向量，F_ = K__ @ X_ - bK_
        K__ = ravelK_.base

        # 读取索引
        sK = self.sK
        s_φe = sK.s_φe
        s_Jintneg = sK.s_jintneg
        s_Jintpos = sK.s_jintpos
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
        solve_Jint_ = LPJTFP2D.solve_Jint_
        solve_dJintdηint_ = LPJTFP2D.solve_dJintdηint_
        solve_dJintdI0int_ = LPJTFP2D.solve_dJintdI0int_
        solve_I0int_ = LPJTFP2D.solve_I0int_
        solve_dI0intdθssurf_ = LPJTFP2D.solve_dI0intdθssurf_
        solve_dI0intdθe_ = LPJTFP2D.solve_dI0intdθe_
        solve_UOCPneg_ = self.solve_UOCPneg_
        solve_UOCPpos_ = self.solve_UOCPpos_
        solve_dUOCPdθsneg_ = self.solve_dUOCPdθsneg_
        solve_dUOCPdθspos_ = self.solve_dUOCPdθspos_
        if lithiumPlating:
            solve_JLP_ = LPJTFP2D.solve_JLP_
            solve_dJLPdθe_ = LPJTFP2D.solve_dJLPdθe_
            solve_dJLPdηLP_ = LPJTFP2D.solve_dJLPdηLP_
            solve_I0LP_ = LPJTFP2D.solve_I0LP_

        # 读取参数
        Nneg, Nsep, Npos, Nr = self.Nneg, self.Nsep, self.Npos, self.Nr  # 读取：网格数
        Δx_ = self.Δx_                                 # 读取：网格尺寸 [–]
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

        if decouple:
            # 固相表面浓度θssurf行Jint列
            # 历史固相浓度影响分量θsI__、系数向量γ_
            θsnegI__, γneg_ = self._update_K__bK_θsnegsurf_Jintneg_when_decoupling(Dsneg, Qneg, Δt, self.θsneg__, self.Jintneg_)
            θsposI__, γpos_ = self._update_K__bK_θspossurf_Jintpos_when_decoupling(Dspos, Qpos, Δt, self.θspos__, self.Jintpos_)
        else:
            # 固相内部浓度cs行cs行、cs末尾球壳控制体行jint列
            self._update_K__bK_θsneg_θsneg_Jintneg_when_coupling(Dsneg, Qneg, Δt, self.θsneg__, self.Jintneg_)
            self._update_K__bK_θspos_θspos_Jintpos_when_coupling(Dspos, Qpos, Δt, self.θspos__, self.Jintpos_)

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
        if decouple:
            pass
        else:
            θsneg_ = X_[sK.s_csneg]
            θspos_ = X_[sK.s_cspos]
        θsnegsurf_ = X_[sK.s_csnegsurf]
        θspossurf_ = X_[sK.s_cspossurf]
        θe_ = X_[sK.s_ce]
        θeneg_ = X_[sK.s_ceneg]
        θepos_ = X_[sK.s_cepos]
        φsneg_ = X_[sK.s_φsneg]
        φspos_ = X_[sK.s_φspos]
        φe_    = X_[s_φe]
        Jintneg_ = X_[s_Jintneg]
        Jintpos_ = X_[s_Jintpos]
        if doubleLayerEffect:
            JDLneg_ = X_[sK.s_jDLneg]
            JDLpos_ = X_[sK.s_jDLpos]
        I0intneg_ = X_[s_I0intneg] if I0intnegUnknown else I0intneg
        I0intpos_ = X_[s_I0intpos] if I0intposUnknown else I0intpos
        ηintneg_ = X_[s_ηintneg]
        ηintpos_ = X_[s_ηintpos]
        if lithiumPlating:
            φeneg_ = X_[sK.s_φeneg]
            JLP_ = X_[s_JLP]
            ηLP_ = X_[sK.s_ηLP]

        # 解向量赋初值
        if decouple:
            pass
        else:
            θsneg_[:] = self.θsneg__.ravel('F')
            θspos_[:] = self.θspos__.ravel('F')
        θsnegsurf_[:] = self.θsnegsurf_
        θspossurf_[:] = self.θspossurf_
        θe_[:] = self.θe_
        # if doubleLayerEffect:
        #     JDLneg_[:] = self.JDLneg_
        #     JDLpos_[:] = self.JDLpos_
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
        ravelJ_ = J__.ravel()  # (N*N,) Jacobi矩阵展平视图
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
                ravelJ_[nrE-1:nrE+2] = -a - aa - c - cc, a - aa*p + c - cc*p + d - dd, -d - dd  # 界面右侧控制体

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
                raise P2Dbase.Error('nan')
            if (θe_<=0).any():
                raise P2Dbase.Error('θe<=0')
            if (θsnegsurf_<=0).any():
                raise P2Dbase.Error('θsnegsurf<=0')
            if (θsnegsurf_>=1).any():
                raise P2Dbase.Error('θsnegsurf>=1')
            if (θspossurf_<=0).any():
                raise P2Dbase.Error('θspossurf<=0')
            if (θspossurf_>=1).any():
                raise P2Dbase.Error('θspossurf>=1')

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
        if decouple:
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
        if self.decouple:
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
        if self.decouple:
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

    @property
    def dJintdI0intneg_(self):
        """负极主反应局部体积电流密度Jintneg对交换电流密度I0intneg的偏导数 [A/A]"""
        return LPJTFP2D.solve_dJintdI0int_(self.T, self.ηintneg_)

    @property
    def dJintdI0intpos_(self):
        """正极主反应局部体积电流密度Jintpos对交换电流密度I0pos的偏导数 [A/A]"""
        return LPJTFP2D.solve_dJintdI0int_(self.T, self.ηintpos_)

    @staticmethod
    def solve_dJintdI0int_(T, ηint_) -> ndarray:
        """求解主反应局部体积电流密度Jint对交换电流密度I0int的偏导数 [A/A]"""
        return 2*sinh(P2Dbase.F/(2*P2Dbase.R*T)*ηint_)

    @property
    def dJintdηintneg_(self):
        """负极主反应局部体积电流密度Jintneg对过电位ηintneg的偏导数 [A/V]"""
        return LPJTFP2D.solve_dJintdηint_(self.T, self.I0intneg_, self.ηintneg_)

    @property
    def dJintdηintpos_(self):
        """正极主反应局部体积电流密度Jintpos对过电位ηintpos的偏导数 [A/V]"""
        return LPJTFP2D.solve_dJintdηint_(self.T, self.I0intpos_, self.ηintpos_)

    @staticmethod
    def solve_dJintdηint_(T, I0int_, ηint_) -> ndarray:
        """求解主反应局部体积电流密度Jint对过电位ηint的偏导数 [A/V]"""
        FRT = P2Dbase.F/(P2Dbase.R*T)
        return FRT*I0int_*cosh(FRT*0.5*ηint_)

    @staticmethod
    def solve_I0int_(k, θssurf_, θe_) -> ndarray:
        """由固液相浓度场求主反应交换电流密度I0int [A]"""
        return k * sqrt(θe_*(1 - θssurf_)*θssurf_)

    @property
    def dI0intdθsnegsurf_(self):
        """负极主反应交换电流密度I0intneg对电极表面浓度的偏导数 [A/-]"""
        return 0  if self._I0intneg\
            else LPJTFP2D.solve_dI0intdθssurf_(self.kneg, self.θsnegsurf_, self.θeneg_, self.I0intneg_)

    @property
    def dI0intdθspossurf_(self):
        """正极主反应交换电流密度I0intpos对电极表面嵌锂状态的偏导数 [A/-]"""
        return 0 if self._I0intpos\
            else LPJTFP2D.solve_dI0intdθssurf_(self.kpos, self.θspossurf_, self.θepos_, self.I0intpos_)

    @staticmethod
    def solve_dI0intdθssurf_(k, θssurf_, θe_, I0int_):
        """求解主反应交换电流密度I0int对固相颗粒表面无量纲锂离子浓度θssurf的偏导数 [A/-]"""
        return k*k * θe_*(0.5 - θssurf_)/I0int_

    @property
    def dI0intdθeneg_(self):
        """负极主反应交换电流密度I0int对电解液浓度θe的偏导数 [A/-]"""
        return 0 if self._I0intneg \
            else LPJTFP2D.solve_dI0intdθe_(self.θeneg_, self.I0intneg_)

    @property
    def dI0intdθepos_(self):
        """正极主反应交换电流密度I0int对电解液浓度θe的偏导数 [A/-]"""
        return 0 if self._I0intpos \
            else LPJTFP2D.solve_dI0intdθe_(self.θepos_, self.I0intpos_)

    @staticmethod
    def solve_dI0intdθe_(θe_, I0int_):
        """求解主反应交换电流密度I0int对电解液无量纲锂离子浓度θe的偏导数  [A/-]"""
        return 0.5*I0int_/θe_

    @staticmethod
    def solve_JLP_(T, I0LP_, ηLP_) -> ndarray:
        """求解析锂反应局部体积电流密度JLP [A]"""
        FRT = P2Dbase.F/P2Dbase.R/T
        a, b = 0.3*FRT, -0.7*FRT
        JLP_ = I0LP_*(exp(a*ηLP_) - exp(b*ηLP_))
        JLP_[ηLP_>=0] = 0
        return JLP_

    @property
    def dJLPdθe_(self):
        """析锂反应局部体积电流密度JLP对电解液浓度θe的偏导数 [A/-]"""
        return 0 if self._I0LP \
            else LPJTFP2D.solve_dJLPdθe_(self.T, self.θeneg_, self.I0LP_, self.ηLPneg_)

    @staticmethod
    def solve_dJLPdθe_(T, θeneg_, I0LP_, ηLP_):
        """析锂反应局部体积电流密度JLP对电解液锂离子浓度θe的偏导数 [A/-]"""
        FRT = P2Dbase.F/P2Dbase.R/T
        a, b = 0.3*FRT, -0.7*FRT
        dJLPdI0LP_ = exp(a*ηLP_) - exp(b*ηLP_)
        dI0LPdθe_ = 0.3*I0LP_/θeneg_
        dJLPdθe_ = dJLPdI0LP_*dI0LPdθe_
        dJLPdθe_[ηLP_>=0] = 0
        return dJLPdθe_

    @property
    def dJLPdηLP_(self):
        """析锂反应局部体积电流密度JLP对析锂过电位ηLP的偏导数 [A/V]"""
        return LPJTFP2D.solve_dJLPdηLP_(self.T, self.I0LP_, self.ηLPneg_)

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
            LPJTFP2D.solve_I0LP_(self.kLP, self.θeneg_)

    @staticmethod
    def solve_I0LP_(kLP, θeneg_) -> ndarray:
        """求解析锂反应交换电流密度I0LP [A]"""
        return kLP * θeneg_**0.3

    @property
    def ηLPneg_(self):
        """负极析锂反应过电位场 [V]"""
        return self.φsneg_ - self.φeneg_ - self.RSEIneg*self.Jneg_

    @property
    def ηLPpos_(self):
        """正极析锂反应过电位场 [V]"""
        return self.φspos_ - self.φepos_ - self.RSEIpos*self.Jpos_

    @property
    def dUOCPdθsnegsurf_(self):
        """负极电位对负极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUOCPdθsneg_(self.θsnegsurf_)

    @property
    def dUOCPdθspossurf_(self):
        """正极电位对正极表面嵌锂状态的导数 [V/–]"""
        return self.solve_dUOCPdθspos_(self.θspossurf_)

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
        gradφsneg_ = self.gradφsneg_
        return self.σneg*(gradφsneg_*gradφsneg_).mean()

    @property
    def Qohmpos(self):
        """正极固相欧姆热 [W]"""
        gradφspos_ = self.gradφspos_
        return self.σpos*(gradφspos_*gradφspos_).mean()

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

    def initialize_consistent(self,
            θsneg__: ndarray,
            θspos__: ndarray,
            θe_: ndarray,
            I: float | int = 0):
        # 一致性初始化
        # 已知：θsneg__、θspos__、θe_、I
        # 求解：θsnegsurf_、θspossurf__、φsneg_、φspos_、φe_、Jintneg_、Jintpos、I0intneg_、I0intpos_、ηintneg_、ηintpos_
        # 令：JDLneg_ = JDLpos_ = JLP_ = 0
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
        solve_Jint_ = LPJTFP2D.solve_Jint_
        solve_dJintdI0int_ = LPJTFP2D.solve_dJintdI0int_
        solve_dJintdηint_  = LPJTFP2D.solve_dJintdηint_
        solve_I0int_ = LPJTFP2D.solve_I0int_
        solve_dI0intdθssurf_ = LPJTFP2D.solve_dI0intdθssurf_
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
        if self.decouple:
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

        for nNewton in range(1, 201):
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

        if self.verbose:
            print(f'一致性初始化完成。Newton迭代{nNewton = }。Consistent initial conditions are solved! ')

    def _generate_Kf__bKf_and_slices(self):
        # 生成频域因变量矩阵Kf__、常数项向量bKf_及切片索引，并对Kf__赋常系数、几何网格相关参数
        P2Dbase._generate_Kf__bKf_and_slices(self)
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
        self._update_Kf__REθe_REθe_and_IMθe_IMθe(self.Deκ_)
        self._update_Kf__REφsneg_REJneg_and_IMφsneg_IMJneg(σneg := self.σneg)
        self._update_Kf__REφspos_REJpos_and_IMφspos_IMJpos(σpos := self.σpos)
        self._update_bKf_REφsneg0_and_REφsposEnd(σneg, σpos)
        self._update_Kf__REφe_REφe_and_IMφe_IMφe(κ_ := self.κ_, κ_)
        self._update_Kf__REηintneg_REJneg_and_IMηintneg_IMJneg(self.RSEIneg)
        self._update_Kf__REηintpos_REJpos_and_IMηintpos_IMJpos(self.RSEIpos)
        if self.lithiumPlating:
            self._update_Kf__REηLP_REJneg_and_IMηLP_IMJneg(self.RSEIneg)

    def _update_Kf__REθe_REθe_and_IMθe_IMθe(self, Deκ_):
        # 更新Kf__矩阵REθe行REθe列、IMθe行IMθe列
        P2Dbase._update_Kf__REce_REce_and_IMce_IMce(self, Deκ_, Deκ_)

    def _update_Kf__REφsneg_REJneg_and_IMφsneg_IMJneg(self, σneg):
        # 更新Kf__矩阵REφsneg行REJneg列、IMφsneg行IMJneg列
        P2Dbase._update_Kf__REφsneg_REjneg_and_IMφsneg_IMjneg(self, σneg)

    def _update_Kf__REφspos_REJpos_and_IMφspos_IMJpos(self, σpos):
        # 更新Kf__矩阵REφspos行REJpos列、IMφspos行IMJpos列
        P2Dbase._update_Kf__REφspos_REjpos_and_IMφspos_IMjpos(self, σpos)

    def _update_bKf_REφsneg0_and_REφsposEnd(self, σneg, σpos):
        bKf_ = self.bKf_
        ΔIAC = self.ΔIAC
        sKf = self.sKf
        # 更新bKf_向量REφsneg首元
        bKf_[sKf.s_REφsneg.start]   = -self.Δxneg*ΔIAC/σneg
        # 更新bKf_向量REφspos末元
        bKf_[sKf.s_REφspos.stop - 1] = self.Δxpos*ΔIAC/σpos

    def _update_Kf__REηintneg_REJneg_and_IMηintneg_IMJneg(self, RSEIneg):
        # 更新Kf__矩阵REηintneg行REJneg列、IMηintneg行IMJneg列
        P2Dbase._update_Kf__REηintneg_REjneg_and_IMηintneg_IMjneg(self, RSEIneg, 1)

    def _update_Kf__REηintpos_REJpos_and_IMηintpos_IMJpos(self, RSEIpos):
        # 更新Kf__矩阵REηintpos行REJpos列、IMηintpos行IMJpos列
        P2Dbase._update_Kf__REηintpos_REjpos_and_IMηintpos_IMjpos(self, RSEIpos, 1)

    def _update_Kf__REηLP_REJneg_and_IMηLP_IMJneg(self, RSEIneg):
        # 更新Kf__矩阵REηLP行REJneg列、IMηLP行IMJneg列
        P2Dbase._update_Kf__REηLP_REjneg_and_IMηLP_IMjneg(self, RSEIneg, 1)

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
        κDκT_ = (self.κD * self.T) * κ_
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
    def solve_REθs__IMθs__(
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

    def solve_frequency_dependent_variables(self):
        """求解频率相关变量"""
        ω_ = self.ω_
        solve_Kθssurf___ = LPJTFP2D.solve_Kθssurf___
        frequency_dependent_variables = {
            'ωqeΔx__'  : outer(ω_, self.qe_*self.Δx_),  # (len(f_), Ne) 各频率各控制体的ω*qe*Δx值
            'ωCDLneg_' : (ωCDLneg_ := ω_*self.CDLneg),  # (Ne,)
            'ωCDLpos_' : (ωCDLpos_ := ω_*self.CDLpos),  # (Ne,)
            'ωCDLRSEIneg_' : ωCDLneg_*self.RSEIneg,     # (Ne,)
            'ωCDLRSEIpos_' : ωCDLpos_*self.RSEIpos,     # (Ne,)
            'minusKθsnegsurf___' : -solve_Kθssurf___(ω_, self.Qneg, self.Dsneg),   # (Nf, 2, 2) 负极各频率Kθssurf__矩阵
            'minusKθspossurf___' : -solve_Kθssurf___(ω_, self.Qpos, self.Dspos),}  # (Nf, 2, 2) 正极各频率Kθssurf__矩阵
        return frequency_dependent_variables

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
        F2RT = 0.5*P2Dbase.F/P2Dbase.R/self.T
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
        print(f'电解液实部浓度方程 REθe 最大误差{maxError: 8e} [C]')

        LHS__ = outer(ω_, qe_) * REθe__
        RHS__ = Deκ_*(gradIMθeEast__ - gradIMθeWest__)/Δx_ + hstack([IMJintneg__ + IMJDLneg__ + IMJLP__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液虚部浓度方程 IMθe 最大误差{maxError: 8e} [C]')

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
        print(f'负极固相电势方程 REφsneg IMφsneg 最大误差{maxError: 8e} [A]')

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
        print(f'正极固相电势方程 REφspos IMφspos 最大误差{maxError: 8e} [A]')

        term1__ = κ_*(gradREφeEast__ - gradREφeWest__)/Δx_
        term2__ = (κDκT_*(gradREθeEast__/θeEast_ - REθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradREθeWest__/θeWest_ - REθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([REJintneg__ + REJDLneg__ + REJLP__ , zeros([Nf, Nsep]), REJintpos__ + REJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势实部方程 REφe 最大误差{maxError: 8e} [A]')

        term1__ = κ_*(gradIMφeEast__ - gradIMφeWest__)/Δx_
        term2__ = (κDκT_*(gradIMθeEast__/θeEast_ - IMθeEast__/θeEast_**2*gradθeEast_) -
                   κDκT_*(gradIMθeWest__/θeWest_ - IMθeWest__/θeWest_**2*gradθeWest_))/Δx_
        LHS__ = term1__ - term2__
        RHS__ = -hstack([IMJintneg__ + IMJDLneg__ + IMJLP__, zeros([Nf, Nsep]), IMJintpos__ + IMJDLpos__])
        maxError = abs(LHS__ - RHS__).max()
        print(f'电解液电势虚部方程 IMφe 最大误差{maxError: 8e} [A]')

        maxError = max([
            abs(REJintneg__ - 2*(REI0intneg__*sinh(F2RT*ηintneg_) + REηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(IMJintneg__ - 2*(IMI0intneg__*sinh(F2RT*ηintneg_) + IMηintneg__*F2RT*I0intneg_*cosh(F2RT*ηintneg_))).max(),
            abs(REJintpos__ - 2*(REI0intpos__*sinh(F2RT*ηintpos_) + REηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(),
            abs(IMJintpos__ - 2*(IMI0intpos__*sinh(F2RT*ηintpos_) + IMηintpos__*F2RT*I0intpos_*cosh(F2RT*ηintpos_))).max(), ])
        print(f'主反应BV动力学方程 REJint IMJint 最大误差{maxError: 8e} [A]')

        dI0intdθeneg_, dI0intdθepos_ = self.dI0intdθeneg_, self.dI0intdθepos_
        dI0intdθsnegsurf_, dI0intdθspossurf_ = self.dI0intdθsnegsurf_, self.dI0intdθspossurf_
        maxError = max([
            abs(REI0intneg__ - (dI0intdθeneg_*REθe__[:, :Nneg] + dI0intdθsnegsurf_*REθsnegsurf__)).max(),
            abs(IMI0intneg__ - (dI0intdθeneg_*IMθe__[:, :Nneg] + dI0intdθsnegsurf_*IMθsnegsurf__)).max(),
            abs(REI0intpos__ - (dI0intdθepos_*REθe__[:, -Npos:] + dI0intdθspossurf_*REθspossurf__)).max(),
            abs(IMI0intpos__ - (dI0intdθepos_*IMθe__[:, -Npos:] + dI0intdθspossurf_*IMθspossurf__)).max(), ])
        print(f'交换电流密度方程 REI0int IMI0int 最大误差{maxError: 8e} [A]')

        dUOCPdθsnegsurf_, dUOCPdθspossurf_ = self.dUOCPdθsnegsurf_, self.dUOCPdθspossurf_
        maxError = max([
            abs(REηintneg__ - (REφsneg__ - REφe__[:, :Nneg] - dUOCPdθsnegsurf_*REθsnegsurf__ - RSEIneg*(REJintneg__ + REJDLneg__ + REJLP__))).max(),
            abs(IMηintneg__ - (IMφsneg__ - IMφe__[:, :Nneg] - dUOCPdθsnegsurf_*IMθsnegsurf__ - RSEIneg*(IMJintneg__ + IMJDLneg__ + IMJLP__))).max(),
            abs(REηintpos__ - (REφspos__ - REφe__[:, -Npos:] - dUOCPdθspossurf_*REθspossurf__ - RSEIpos*(REJintpos__ + REJDLpos__))).max(),
            abs(IMηintpos__ - (IMφspos__ - IMφe__[:, -Npos:] - dUOCPdθspossurf_*IMθspossurf__ - RSEIpos*(IMJintpos__ + IMJDLpos__))).max(), ])
        print(f'主反应过电位方程 REηint IMηint 最大误差{maxError: 8e} [V]')

        if self.lithiumPlating:
            dJLPdθe_, dJLPdηLP_ = self.dJLPdθe_, self.dJLPdηLP_
            maxError = max([
                abs(REJLP__ - (dJLPdθe_*REθe__[:, :Nneg] + dJLPdηLP_*REηLP__) ).max(),
                abs(IMJLP__ - (dJLPdθe_*IMθe__[:, :Nneg] + dJLPdηLP_*IMηLP__) ).max(),])
            print(f'析锂BV动力学方程 REJLP IMJLP 最大误差{maxError: 8e} [A]')

            maxError = max([
                abs(REηLP__ - (REφsneg__ - REφe__[:, :Nneg] - RSEIneg*(REJintneg__ + REJDLneg__ + REJLP__))).max(),
                abs(IMηLP__ - (IMφsneg__ - IMφe__[:, :Nneg] - RSEIneg*(IMJintneg__ + IMJDLneg__ + IMJLP__))).max(), ])
            print(f'析锂过电位方程 REηLP IMηLP 最大误差{maxError: 8e} [V]')

    plot_θ = P2Dbase.plot_c              # 作图：浓度场-空间、时间
    plot_Jint_I0int_ηint = P2Dbase.plot_jint_i0int_ηint  # 作图：主反应局部体积电流密度、过电位、交换电流密度-空间、时间
    plot_JDL = P2Dbase.plot_jDL          # 作图：双电层效应局部体积电流密度、电流
    plot_θsr = P2Dbase.plot_csr          # 作图：固相颗粒径向锂离子浓度场-空间、时间
    plot_JLP_ηLP = P2Dbase.plot_jLP_ηLP  # 作图：负极析锂局部体积电流密度-空间、时间
    plot_REθssurf_IMEθssurf = P2Dbase.plot_REcssurf_IMcssurf
    plot_REθe_IMθe = P2Dbase.plot_REce_IMce
    plot_REJint_IMJint = P2Dbase.plot_REjint_IMjint
    plot_REJDL_IMJDL = P2Dbase.plot_REjDL_IMjDL
    plot_REI0int_IMI0int = P2Dbase.plot_REi0int_IMi0int

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
        # decouple=False,
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
    cell.plot_θ(np.arange(0, 2001, 200))
    cell.plot_φ(np.arange(0, 2001, 200))
    cell.plot_Jint_I0int_ηint(np.arange(0, 2001, 200))
    cell.plot_JDL(np.arange(0, 2001, 200))
    cell.plot_csr(np.arange(0, 2001, 200), 1)
    cell.plot_JLP_ηLP(np.arange(1000, 1601, 100))
    cell.plot_LP()
    cell.plot_OCV_OCP()
    cell.plot_nNewton()

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

import math
import numpy as np

from P2Dmodel import DFNP2D

F = DFNP2D.F  # 法拉第常数 [C/mol]
R = DFNP2D.R  # 理想气体常数 [J/(mol·K)]

class LumpedParameters:
    """23集总参数取值"""
    def __init__(self,
            Qnom: float | int = 1., # 电池标称容量 [Ah]
            ):
        assert Qnom>0, '电池标称容量应大于0 [Ah]'
        Qnom_in_C = Qnom*3600  # 电池标称容量 [C]
        self.bounds__ = {
            'SOC0': (0.01, 0.8),
            'Qcell': np.array([0.75, 1.1])*Qnom,
            'Qneg': np.array([1.01, 1.6])*Qnom,
            'Qpos': np.array([1.01, 1.6])*Qnom,
            'qeneg': np.array([1e-2, 1])*Qnom_in_C,
            'qesep': np.array([1e-3, 1])*Qnom_in_C,
            'qepos': np.array([1e-2, 1])*Qnom_in_C,
            'κneg': np.array([0.00075, 0.08])*Qnom_in_C,
            'κsep': np.array([0.004, 0.08])*Qnom_in_C,
            'κpos': np.array([0.00075, 0.08])*Qnom_in_C,
            'σneg': np.array([0.01, 170])*Qnom_in_C,
            'σpos': np.array([0.01, 170])*Qnom_in_C,
            'Dsneg': (5e-6, 0.63),
            'Dspos': (5e-6, 0.63),
            'De': (2.56e-3, 2.22e-1),
            'κD': np.array([0.5, 9.])*2*R/F,
            'kneg': np.array([1e-6, 5e-2])*Qnom_in_C,
            'kpos': np.array([1e-6, 5e-2])*Qnom_in_C,
            'RSEIneg': np.array([0.09, 330])/Qnom_in_C,
            'RSEIpos': np.array([0.09, 330])/Qnom_in_C,
            'CDLneg': np.array([1e-6, 1e-2])*Qnom_in_C,
            'CDLpos': np.array([1e-6, 1e-2])*Qnom_in_C,
            'l': np.array([1e-13, 1e-11])*Qnom_in_C, }

        for name, bound_ in self.bounds__.items():
            self.bounds__[name] = (float(bound_[0]), float(bound_[1]))

    def print(self):
        nominalSet_ = self.nominalSet_
        for n, name in enumerate(nominalSet_, start=1):
            print(f'{n: <5}{name: 10}'
                  f'下限{self.bounds__[name][0]: <15g}'
                  f'上限{self.bounds__[name][1]: <15g}'
                  f'标称值{nominalSet_[name]: 3g}')

    @property
    def names_(self):
        # 参数名序列
        return np.array(list(self.bounds__.keys()))

    @property
    def nominalSet_(self) -> dict:
        # 标称参数集
        return {name: self.denormalize(name, 0.5) for name in self.bounds__}

    def normalize(self,
                  name: str,
                  value: float | int,) -> float:
        # 参数值归一化
        assert name in self.bounds__, f'参数{name}不包含于{self.bounds__.keys()}'
        value = float(value)
        lb, ub = self.bounds__[name]
        # assert lb<=value<=ub, f'参数{name}的取值范围为[{lb}, {ub}]，当前输入取值{value}'
        if abs(ub/lb)>10:
            normvalue = (math.log10(value) - math.log10(lb))/(math.log10(ub) - math.log10(lb))
        else:
            normvalue = (value - lb)/(ub - lb)
        return normvalue

    def denormalize(self,
                    name: str,
                    normvalue: float | int) -> float:
        # 归一化参数值去归一化
        assert name in self.bounds__, f'参数{name}不包含于{self.bounds__.keys()}'
        # normvalue = float(normvalue)
        lb, ub = self.bounds__[name]
        if abs(ub/lb)>10:
            value = 10**(math.log10(lb) + normvalue*(math.log10(ub) - math.log10(lb)))
        else:
            value = lb + normvalue*(ub - lb)
        return value

    def Normalize(self, parameterSet_: dict):
        # 批量归一化
        return {name: self.normalize(name, value) for name, value in parameterSet_.items()}

    def Denormalize(self, parameterSet_: dict):
        # 批量去归一化
        return {name: self.denormalize(name, normvalue) for name, normvalue in parameterSet_.items()}

    @staticmethod
    def sign(name):
        match name:
            case 'Qcell' | 'Qneg' | 'Qpos':
                sign = rf'${{\overline {{\it Q}} }}_{{\mathrm {{ {name[1:]} }} }}$'
            case 'θminneg' | 'θmaxneg' | 'θminpos' | 'θmaxpos':
                sign = rf'${{\it θ}}_{{\mathrm{{ {name[1:4]},{name[-3:]} }} }}$'
            case 'σneg' | 'σpos' | 'κneg' | 'κsep' | 'κpos':
                sign = rf'${{\overline {{\it {name[0]} }} }}_{{\mathrm{{ {name[-3:]} }} }}$'
            case 'Dsneg' | 'Dspos' | 'qeneg' | 'qesep' | 'qepos':
                sign = rf'${{\overline {{\it {name[0]} }} }}_{{\mathrm{{ {name[1]},{name[-3:]} }} }}$'
            case 'Kqeneg' | 'Kqepos':
                sign = rf'${{\overline {{\it K}} }}_{{\mathrm{{q_{{e,{name[-3:]} }} }} }}$'
            case 'Kκneg' | 'Kκpos':
                sign = rf'${{\overline {{\it K}} }}_{{\mathrm{{κ,{name[-3:]} }} }}$'
            case 'kneg' | 'kpos':
                sign = rf'${{\overline {{\it k}} }}_{{\mathrm{{{name[-3:]} }} }}$'
            case 'RSEIneg' | 'RSEIpos':
                sign = rf'${{\overline {{\it R}} }}_{{\mathrm{{SEI,{name[-3:]} }} }}$'
            case 'CDLneg' | 'CDLpos':
                sign = rf'${{\overline {{\it C}} }}_{{\mathrm{{DL,{name[-3:]} }} }}$'
            case 'κD':
                sign = r'${\overline {\it κ} }_{\mathrm{D}}$'
            case 'De':
                sign = r'${\overline {\it D} }_{\mathrm{e}}$'
            case 'l':
                sign = r'${\it l}$'
            case 'SOC0':
                sign = r'${\it SOC}_{\mathrm{0}}$'
            case 'I0intneg' | 'I0intpos':
                sign = rf'${{\it I}}_{{\mathrm{{0,int,{name[-3:]} }} }}$'
            case 'T':
                sign = r'${\it T}$'
            case _:
                raise ValueError(f'未定义参数{name}')
        return sign

    @staticmethod
    def unit(name):
        match name:
            case 'Qcell' | 'Qneg' | 'Qpos':
                unit = '$Ah$'
            case 'θminneg' | 'θmaxneg' | 'θminpos' | 'θmaxpos' |\
                 'SOC0' |\
                 'Kqeneg' | 'Kqepos' | 'Kκneg' | 'Kκpos':
                unit = ''
            case 'σneg' | 'σpos' | 'κneg' | 'κsep' | 'κpos':
                unit = '$S$'
            case 'Dsneg' | 'Dspos':
                unit = '$s^{-1}$'
            case 'qeneg' | 'qesep' | 'qepos':
                unit = '$C$'
            case 'kneg' | 'kpos' | 'I0intneg' | 'I0intpos':
                unit = '$A$'
            case 'RSEIneg' | 'RSEIpos':
                unit = '$Ω$'
            case 'CDLneg' | 'CDLpos':
                unit = '$F$'
            case 'κD':
                unit = '$V/K$'
            case 'De':
                unit = '$A/S$'
            case 'l':
                unit = '$H$'
            case 'T':
                unit = '$K$'
            case _:
                raise ValueError(f'未定义参数{name}')
        return unit

    @staticmethod
    def value(name, value):
        match name:
            case 'Qcell' | 'Qneg' | 'Qpos' | 'T':
                string = f'{value:.2f}'
            case 'θminneg' | 'θmaxneg' | 'θminpos' | 'θmaxpos' | 'SOC0' |\
                 'Kqeneg' | 'Kqepos' | 'Kκneg' | 'Kκpos':
                string = f'{value:.3f}'
            case 'σneg' | 'σpos' | 'κneg' | 'κsep' | 'κpos' |\
                 'Dsneg' |'Dspos' |\
                 'qeneg' | 'qesep' | 'qepos' |\
                 'kneg' | 'kpos'| 'I0intneg' | 'I0intpos' |\
                 'RSEIneg' | 'RSEIpos' |\
                 'CDLneg' | 'CDLpos' |\
                 'κD' | 'De' | 'l':
                base, expo = f'{value:.2e}'.split('e')
                string = rf'${float(base):.2f}\;×\;10^{{ {int(expo)} }}$'
            case _:
                raise ValueError(f'未定义参数{name}')
        return string

    @classmethod
    def value_unit(cls, name, value):
        return cls.value(name, value) + r'$\;$' + cls.unit(name)

class EnhancedLumpedParameters(LumpedParameters):
    """强集总参数取值"""
    def __init__(self,
                 Qnom: float | int = 18.,  # 电池标称容量 [Ah]
                 ):
        LumpedParameters.__init__(self, Qnom=Qnom,)
        del self.bounds__['κneg'], self.bounds__['κpos'],\
            self.bounds__['qeneg'], self.bounds__['qepos']
        self.bounds__ |= {
            'Kκneg': (0.08, 1),
            'Kκpos': (0.08, 1),
            'Kqeneg': (0.5, 4),
            'Kqepos': (0.5, 4),}

class ConservativeLumpedParameters(LumpedParameters):
    """保守集总参数模型（25参数，含4边界嵌锂状态，不含正负极容量Qneg、Qpos）的参数取值"""
    def __init__(self, Qnom):
        LumpedParameters.__init__(self, Qnom=Qnom)
        del self.bounds__['Qneg'], self.bounds__['Qpos']
        self.bounds__ |= {
            'θminneg': (0.001, 0.44),   # SOC=0%的负极嵌锂状态取值范围
            'θmaxneg': (0.60, 0.99),    # SOC=100%的负极嵌锂状态取值范围
            'θminpos': (0.001, 0.44),   # SOC=100%的正极嵌锂状态取值范围
            'θmaxpos': (0.60, 0.99),}   # SOC=0%的正极嵌锂状态取值范围


def transform37to23(
        A, Lneg, Lsep, Lpos, εsneg, εspos,
        εeneg, εesep, εepos, Rsneg, Rspos,
        bneg, bsep, bpos,
        Dsneg, Dspos, De,
        σneg, σpos, κ,
        tplus, TDF,
        kneg, kpos, RSEIneg, RSEIpos,
        csmaxneg, csmaxpos, ce0,
        CDLneg, CDLpos, l,
        θminneg, θmaxneg, θminpos, θmaxpos, SOC0,
        ):
    Qcellneg = A*Lneg*εsneg*csmaxneg*(θmaxneg - θminneg)*F/3600
    Qcellpos = A*Lpos*εspos*csmaxpos*(θmaxpos - θminpos)*F/3600
    print(f'正极容量{Qcellpos:.4f}Ah，负极容量{Qcellneg:.4f}Ah，使用负极容量')
    aneg = 3*εsneg/Rsneg
    apos = 3*εspos/Rspos
    return {
        'Qcell' : Qcellneg,  # 全电池理论可用容量 [Ah]
        'Qneg': A*Lneg*εsneg*csmaxneg*F/3600,  # 负极容量 [Ah]
        'Qpos': A*Lpos*εspos*csmaxpos*F/3600,  # 正极容量 [Ah]
        'σneg' : A*σneg*εsneg**bneg/Lneg,  # 负极集总固相电导率 [S]
        'σpos' : A*σpos*εspos**bpos/Lpos,  # 正极集总固相电导率 [S]
        'κneg' : A*κ*εeneg**bneg/Lneg,  # 负极集总液相离子电导率 [S]
        'κsep' : A*κ*εesep**bsep/Lsep,  # 隔膜集总液相离子电导率 [S]
        'κpos' : A*κ*εepos**bpos/Lpos,  # 正极集总液相离子电导率 [S]
        'Dsneg' : Dsneg/Rsneg**2,  # 负极集总固相锂离子扩散系数 [1/s]
        'Dspos' : Dspos/Rspos**2,  # 正极集总固相锂离子扩散系数 [1/s]
        'qeneg' : F*εeneg*ce0*A*Lneg/(1 - tplus),  # 负极液相锂离子电荷量 [C]
        'qesep' : F*εesep*ce0*A*Lsep/(1 - tplus),  # 隔膜液相锂离子电荷量 [C]
        'qepos' : F*εepos*ce0*A*Lpos/(1 - tplus),  # 正极液相锂离子电荷量 [C]
        'kneg' : F*aneg*A*Lneg*kneg*ce0**0.5*csmaxneg,  # 负极集总反应速率常数 [A]
        'kpos' : F*apos*A*Lpos*kpos*ce0**0.5*csmaxpos,  # 正极集总反应速率常数 [A]
        'RSEIneg' : RSEIneg/(aneg*A*Lneg),  # 负极集总SEI膜内阻 [Ω]
        'RSEIpos' : RSEIpos/(apos*A*Lpos),  # 负极集总SEI膜内阻 [Ω]
        'κD' : 2*R/F*(1 - tplus)*TDF,   # 液相离子电导系数 [–]
        'De' : F*De*ce0/κ/(1 - tplus),    # 液相集总扩散系数 [V]
        'CDLneg' : aneg*A*Lneg*CDLneg,  # 负极集总双电层电容 [F]
        'CDLpos' : apos*A*Lpos*CDLpos,  # 正极集总双电层电容 [F]
        'l' : l,
        'SOC0': SOC0, }

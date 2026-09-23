#%%
import pathlib
from typing import Sequence, Callable
from functools import partial
from math import sqrt

from numba import njit
import numpy as np
from numpy import ndarray, array, concatenate, interp, logspace
from numpy import abs as np_abs
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

np.seterr(divide='ignore', over='ignore', invalid='ignore')

from P2Dmodel import LPJTFP2D, LumpedParameters, set_matplotlib, Interpolate1D
set_matplotlib()

with np.load(pathlib.Path(__file__).parent.joinpath('example_UOCPneg_and_UOCPpos.npz'),
             allow_pickle=True) as npz:
    UOCPneg = Interpolate1D(npz['θsneg_'], npz['UOCPneg_'])
    UOCPpos = Interpolate1D(npz['θspos_'], npz['UOCPpos_'])




class IdentificationBase(LumpedParameters):
    __slots__ = (
        'verbose',
        'thermalModel',
        'targets_', 'weighting', 'w_',
        'objective',
        'kwargs',
        'experiences_',
        'record',
        'pVC_',)

    def __init__(self,
                 verbose: bool = True,    # 是否提示
                 Qnom: float | int = 20,  # 标称容量 [Ah]
                 thermalModel: bool = False,    # 是否开启热模型
                 targets_: tuple[str] = ('UDC', 'Zreal', 'Zimag'),  # 拟合目标 'UDC', 'Zreal', 'Zimag', 'Z'
                 weighting: str = 'balanced',   # 加权策略 'balanced'：大规模采样估计权重；'adaptive'：自适应权重：'given'：强制给定权重
                 w_: ndarray | None = None,     # 权重
                 objective: str = 'RMSE',       # 目标类型 'RMSE'/'MAE'/'MSE'
                 f_: ndarray = logspace(3, 0, 16),  # 频率序列 [Hz]
                 tC: float | int = 25,  # 给定温度 [°C]
                 Nreg = 10,  # 负极、隔膜、正极区域网格数
                 Nr = 10,    # 颗粒网格数
                 UOCPneg: Callable = UOCPneg,  # 负极开路电位函数 [V]
                 UOCPpos: Callable = UOCPpos,  # 正极开路电位函数 [V]
                 Umax: float = 4.2,  # 最大运行电压 [V]
                 Umin: float = 2.8,  # 最小运行电压 [V]
                 ):
        LumpedParameters.__init__(self, Qnom, thermalModel)
        self.verbose = verbose
        self.thermalModel = thermalModel
        self.targets_ = targets_
        assert len(targets_)>=1 and all(target in ('UDC', 'Zreal', 'Zimag', 'Z') for target in targets_), f'存在非法拟合目标，{targets_ = }'
        if len(targets_)==1:
            self.weighting = 'given'
            self.w_ = array([1.])
            if verbose:
                print('拟合目标仅1个，强制给定权重1')
        else:
            self.weighting = weighting
            assert weighting in ('balanced', 'adaptive', 'given'), f'{weighting = }，非法'
            if weighting in ('balanced', 'adaptive'):
                self.w_ = None
            elif weighting in ('given',):
                assert isinstance(w_, (ndarray, tuple, list))
                self.w_ = w_ = array(w_); assert w_.size==len(targets_), f'{w_ = }，权重数目应等于拟合目标数目'
        self.objective = objective; assert objective in ('RMSE', 'MAE', 'MSE'), f'未定义{objective = }'

        assert tC>=0, f'给定温度{tC = }，应大于或等于0 [K]'
        self.kwargs = kwargs = {
            'f_': f_,
            'T0': tC + 273.15, 'Tref': tC + 273.15,
            'Nneg': Nreg, 'Nsep': Nreg, 'Npos': Nreg, 'Nr': Nr,
            'UOCPneg': UOCPneg, 'UOCPpos': UOCPpos,
            'Umax': Umax, 'Umin': Umin,
            'dUOCPdθsneg': LPJTFP2D.generate_solve_dUOCPdθs_(UOCPneg),
            'dUOCPdθspos': LPJTFP2D.generate_solve_dUOCPdθs_(UOCPpos),
            'complete': False, 'verbose': False,
            }
        if thermalModel:
            del kwargs['T0']
            kwargs['Tref'] = 298.15

        self.experiences_ = {
            key: None for key in [
                'banded_experience_of_J__',
                'banded_experience_of_Kf__',
                'ravelKf_', 'bKf_',
                'ravelK_', 'bK_',
                'sK', 'sKf']}  # P2D计算经验
        self.pVC_: dict = None     # 虚拟电池实际值参数集
        self.record: dict = None   # 辨识记录

        if verbose:
            print(self)

    def __str__(self):
        names_ = self.names_
        targets_ = self.targets_
        weighting = self.weighting
        w_ = self.w_
        objective = self.objective
        f_ = self.kwargs['f_']
        string = f'{names_.size}参数：{names_}\n'
        if self.thermalModel:
            string += '考虑6个活化能、热阻Rth、热容Cth，作为参数\n'
        string += (
            f'拟合目标 {targets_ = }\n'
            f'加权策略 {weighting = }\n'
            f'权重 {w_ = }\n'
            f'目标类型 {objective = }\n'
            f'频段 {f_.min():g}-{f_.max():g} Hz，共{f_.size}频点\n')
        if not self.thermalModel:
            string += f'给定电池温度tC = {self.kwargs['T0'] - 273.15} °C\n'
        return string

    def create_experiences_(self) -> None:
        """创造经验"""
        cell = LPJTFP2D(**self.kwargs)
        cell.CC(-1e-3, cell.Δt*3).EIS()
        for key in self.experiences_:
            self.experiences_[key] = getattr(cell, key)
        # breakpoint()
        del cell
        if self.verbose:
            print('已创造经验！Experiences have been created!')

    def plot_iteration(self):
        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        record = self.record
        x_ = range(1, len(record['yMean_']) + 1)
        ax.plot(x_, record['yMean_'], '-b', label='Current mean')
        ax.plot(x_, record['yCurrentOptimal_'], '-r', label='Current optimal')
        ax.plot(x_, record['yGlobalOptimal_'], '-k', label='Global optimal', alpha=0.7)
        ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        ax.set_yscale('log')
        plt.show()

    @staticmethod
    def MSE_on_interval(t_, ysim_, ymea_, ta, tb):
        """在区间 [ta, tb] 上按时间积分意义计算 MSE"""
        if ta>=tb:
            raise ValueError("ta must be smaller than tb")
        logic_ = (ta<t_) & (t_<tb)
        tseg_ = concatenate([[ta],  t_[logic_], [tb]])
        Δy_ = ysim_ - ymea_
        Δyseg_ = np.concatenate([
            [interp(ta, t_, Δy_)],
            Δy_[logic_],
            [interp(tb, t_, Δy_)],])
        dt_ = np.diff(tseg_)
        Δy0_ = Δyseg_[:-1]
        Δy1_ = Δyseg_[1:]
        MSE = (dt_/3*(Δy0_*Δy0_ + Δy0_*Δy1_ + Δy1_*Δy1_)).sum()/(tb - ta)
        return MSE

    @staticmethod
    def RMSE_on_interval(t_, ysim_, ymea_, ta, tb):
        """在区间 [ta, tb] 上按时间积分意义计算 RMSE"""
        return sqrt(IdentificationBase.MSE_on_interval(t_, ysim_, ymea_, ta, tb))

    @staticmethod
    def MAE_on_interval(t_, ysim_, ymea_, ta, tb):
        """在区间 [ta, tb] 上按时间积分意义计算 MAE"""
        if ta>=tb:
            raise ValueError("ta must be smaller than tb")
        logic_ = (ta<t_) & (t_<tb)
        tseg_ = np.concatenate([[ta], t_[logic_], [tb]])
        Δy_ = ysim_ - ymea_
        Δyseg_ = np.concatenate([
            [interp(ta, t_, Δy_)],
            Δy_[logic_],
            [interp(tb, t_, Δy_)], ])
        dt_ = np.diff(tseg_)
        Δy0_ = Δyseg_[:-1]
        Δy1_ = Δyseg_[1:]

        same_sign_ = (Δy0_*Δy1_)>=0

        MAE_segment_ = np.empty_like(dt_, dtype=float)

        # 不跨零：|Δy| 在该小段内也是线性变化，积分为梯形面积
        MAE_segment_[same_sign_] = dt_[same_sign_]/2*(abs(Δy0_[same_sign_]) + abs(Δy1_[same_sign_]))

        # 跨零：|Δy| 形成两个三角形，解析积分如下
        cross_zero_ = ~same_sign_
        MAE_segment_[cross_zero_] = (dt_[cross_zero_]/2
            *(Δy0_[cross_zero_]*Δy0_[cross_zero_] + Δy1_[cross_zero_]*Δy1_[cross_zero_])
            /(np.abs(Δy0_[cross_zero_]) + np.abs(Δy1_[cross_zero_]))
        )

        MAE = MAE_segment_.sum()/(tb - ta)
        return MAE

    @staticmethod
    def solve_objective(ΔY__: ndarray, objective: str) -> float:
        # 求解目标
        if objective=='RMSE':
            value = sqrt((ΔY__*ΔY__).mean())
        elif objective=='MAE':
            value = np_abs(ΔY__).mean()
        elif objective=='MSE':
            value = (ΔY__*ΔY__).mean()
        return value

    @property
    def f_(self):
        return self.kwargs['f_']


if __name__ == '__main__':
    task = IdentificationBase(
        targets_=('UDC', 'Z'),
        weighting='adaptive'
    )
    task.create_experiences_()
    task.experiences_

#%%
import pathlib
from typing import Sequence, Callable
from functools import partial
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from numpy import concatenate, interp, diff

np.seterr(divide='ignore', over='ignore', invalid='ignore')

from P2Dmodel import LPJTFP2D, LumpedParameters, set_matplotlib
set_matplotlib()

with np.load(pathlib.Path(__file__).parent.joinpath('example_UOCPneg_and_UOCPpos.npz'),
             allow_pickle=True) as npz:
    UOCPneg, UOCPpos = npz['UOCPneg'].item(), npz['UOCPpos'].item()


class IdentificationBase(LumpedParameters):
    __slots__ = (
        'activation_energy',
        'tC', 'f_', 'T', 'N',
        'n_jobs', 'batch_size',
        'algorithm', 'objective',
        'verbose',
        'pVC_',
        'record',
        'kwargs',
        'experiences_',
        )

    def __init__(self,
            Qnom: float | int = 20,  # 标称容量 [Ah]
            activation_energy: bool = False,  # 是否认为活化能也是参数
            tC: float | int = 25,    # 温度 [°C]
            Nreg = 10,  # 负极、隔膜、正极区域网格数
            Nr = 10,    # 颗粒网格数
            UOCPneg: Callable = UOCPneg,  # 负极开路电位函数 [V]
            UOCPpos: Callable = UOCPpos,  # 正极开路电位函数 [V]
            Umax: float = 4.2,  # 最大运行电压 [V]
            Umin: float = 2.8,  # 最小运行电压 [V]
            f_: Sequence[float] = np.logspace(np.log10(400), np.log10(4), 17),  # 频率序列 [Hz]
            T: int = 1000,           # 迭代次数
            N: int = 500,            # 种群规模
            n_jobs: int = -1,        # joblib并行执行CPU核数
            batch_size: int = 1,     # joblib并行执行batch_size
            algorithm: str = 'STA',  # 优化算法
            objective: str = 'RMSE', # 最小化目标
            verbose = True,          # 是否提示
            ):
        LumpedParameters.__init__(self, Qnom, activation_energy)
        self.activation_energy = activation_energy
        self.tC = tC; assert tC>=0, f'温度{tC = }，应大于或等于0 [K]'
        self.f_ = f_ = np.array(f_)
        self.T = T
        self.N = N
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.objective = objective
        self.verbose = verbose
        self.pVC_ = None     # 虚拟电池实际值参数集
        self.record = None   # 辨识记录
        self.kwargs = kwargs = {
            'T0': tC + 273.15, 'Tref': tC + 273.15,
            'Nneg': Nreg, 'Nsep': Nreg, 'Npos': Nreg, 'Nr': Nr,
            'doubleLayerEffect': True, 'lithiumPlating': False,
            'complete': False, 'verbose': False, 'constants': True,
            'f_': f_,
            'UOCPneg': UOCPneg, 'UOCPpos': UOCPpos,
            'Umax': Umax, 'Umin': Umin,
            'dUOCPdθsneg': LPJTFP2D.generate_solve_dUOCPdθs_(UOCPneg),
            'dUOCPdθspos': LPJTFP2D.generate_solve_dUOCPdθs_(UOCPpos),}
        if activation_energy:
            del kwargs['T0']
            kwargs['Tref'] = 298.15
            kwargs['constants'] = False

        self.experiences_ = {
            key: None for key in [
                'banded_experience_of_J__',
                'banded_experience_of_Kf__',
                'ravelKf_', 'bKf_',
                'ravelK_', 'bK_',
                'sK', 'sKf']}

        if verbose:
            print(f'频段{f_.min():g}-{f_.max():g}Hz，共{f_.size}频点\n'
                  f'电池温度{tC = } °C\n'
                  f'迭代次数{T = }，个体数{N = }\n'
                  f'joblib利用核数 {n_jobs = }，{batch_size = }\n'
                  f'优化算法 {algorithm = }，目标类型 {objective = }'
                  )

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


if __name__ == '__main__':
    task = IdentificationBase()
    task.create_experiences_()


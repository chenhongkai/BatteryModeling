#%%
import time, joblib, pathlib
from typing import Sequence, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.stats import qmc
np.seterr(divide='ignore', over='ignore', invalid='ignore')

import Optimization
from P2Dmodel import LPJTFP2D, LumpedParameters, set_matplotlib
set_matplotlib()

with np.load(pathlib.Path(__file__).parent.joinpath('example_UnegUpos.npz'), allow_pickle=True) as npz:
    Uneg, Upos = npz['Uneg'].item(), npz['Upos'].item()

class Identification(LumpedParameters):
    def __init__(self,
            Qnom: float | int = 20,  # 标称容量 [Ah]
            IC: float | int = 1,     # 电流倍率 [C-rate]
            tC: float | int = 25,    # 温度 [°C]
            TVT_: tuple[float] = (.7, .15, .15),  # 训练集、验证集、测试集数据比例
            onset: float | int = 0,               # 充电开始时刻 [s]
            duration: float | int = 1000,         # 持续时间 [s]
            Δt: int = 10,           # 时间步长 [s]
            ΔtUDC: int = 10,        # 端电压UDC测量时间间隔 [s]
            ΔtEIS: int = 50,        # EIS测量时间间隔 [s]
            Umax: float = 4.2,      # 最大运行电压 [V]
            Umin: float = 2.8,      # 最小运行电压 [V]
            Uneg: Callable = Uneg,  # 负极开路电位函数 [V]
            Upos: Callable = Upos,  # 正极开路电位函数 [V]
            f_: Sequence[float] = np.logspace(np.log10(400), np.log10(4), 17),  # 频率序列 [Hz]
            T: int = 1000,     # 迭代次数
            N: int = 200,      # 种群规模
            n_jobs: int = -1,  # joblib并行执行核数
            algorithm: str = 'STA',  # 优化算法
            objective: str = 'RMSE', # 最小化目标
            verbose = True,          # 是否提示
            ):
        self.Qnom = Qnom
        LumpedParameters.__init__(self, self.Qnom)
        self.IC = IC
        self.tC = tC
        self.TVT_ = TVT_ = np.array(TVT_)
        assert (0<=TVT_).all() and (TVT_<1).all() and sum(TVT_)==1 and len(TVT_)==3, f'训练集、验证集、测试集数据比例TVT_应满足len(TVT_)==3，sum(TVT_)==1，且各元素取值范围为[0, 1)，当前{TVT_ = }'
        self.onset = onset
        self.duration = duration
        self.Δt = Δt
        self.ΔtUDC = ΔtUDC
        self.ΔtEIS = ΔtEIS
        assert ΔtEIS % ΔtUDC == 0, f'ΔtEIS应可整除ΔtUDC，当前{ΔtEIS = }，{ΔtUDC = }'
        assert ΔtUDC % Δt == 0, f'ΔtUDC应可整除Δt，当前{ΔtUDC = }，{Δt = }'
        self.f_ = np.array(f_)
        self.T, self.N = T, N
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.objective = objective
        self.I = -abs(self.IC*self.Qnom)  # 充电电流 [A]
        tEnd = onset + duration           # 终止时刻 [s]
        self.tUDCmea_ = np.arange(onset,  tEnd + 1e-6, ΔtUDC)  # 电压直流分量测量时刻
        self.tZmea_   = np.arange(onset + (0 if onset else ΔtEIS), tEnd + 1e-6, ΔtEIS)  # 阻抗测量时刻
        self.UDCmea_ = None  # (len(tUDCmea_),) 电压直流分量测量值
        self.Zmea__ = None   # (len(tZmea_), len(f_)) 阻抗测量值
        self.pVC_ = None     # 虚拟电池实际值参数集
        self.record = None  # 辨识记录
        self.kwargs = {
            'Δt': self.Δt, 'T0': self.tC + 273.15, 'Tref': self.tC + 273.15,
            'doubleLayerEffect': True, 'lithiumPlating': False,
            'complete': False, 'verbose': False, 'constants':True,
            'f_': self.f_,
            'Uneg': Uneg, 'Upos': Upos,
            'Umax': Umax, 'Umin': Umin,
            'dUdθneg': LPJTFP2D.generate_solve_dUdθ_(Uneg),
            'dUdθpos': LPJTFP2D.generate_solve_dUdθ_(Upos),}
        self.banded_experience = {key: None for key in
            ['bandwidthsJ_',
             'bandwidthsKf_',
             'idxJreordered_',
             'idxJrecovered_',
             'idxKfReordered_',
             'idxKfRecovered_']}
        self.verbose = verbose
        if verbose:
            end = self.onset + self.TVT_[0]*self.duration
            logic_tUDC_ = self.tUDCmea_ <= end
            logic_tZ_ = self.tZmea_ <= end
            print(f'Training    UDC points: {logic_tUDC_.sum()}，DEIS points: {logic_tZ_.sum()}')
            start = self.onset + self.TVT_[0]*self.duration
            end = self.onset + sum(self.TVT_[:2])*self.duration
            logic_tUDC_ = (start < self.tUDCmea_) & (self.tUDCmea_<=end)
            logic_tZ_ = (start < self.tZmea_) & (self.tZmea_ <= end)
            print(f'Validation  UDC points: {logic_tUDC_.sum()}, DEIS points: {logic_tZ_.sum()}')
            start = self.onset + sum(self.TVT_[:2])*self.duration
            logic_tUDC_ = start < self.tUDCmea_
            logic_tZ_ = start < self.tZmea_
            print(f'Test        UDC points: {logic_tUDC_.sum()}, DEIS points: {logic_tZ_.sum()}')

    def identify(self,
                 pnormfixed_: dict[str, float],
                 targets_: tuple[str] = ('UDC', 'Zreal', 'Zimag'),
                 Nsample: int = 50000,
                 states0: dict[str, float] | None = None,
                 hyperparameters_Optimizer: dict | None= None,
                 pcandidates__: list[dict[str, float]] | None = None,
                 ) -> dict:
        timeStart = time.time()
        assert all(name in self.names_ for name in pnormfixed_), "固定参数集存在非法参数"
        namesfixed_ = tuple(pnormfixed_.keys())  # 固定参数
        namesoptimized_ = [str(name) for name in self.names_ if name not in pnormfixed_]  # 待优化参数
        D = len(namesoptimized_)
        assert D>=1, f'待优化参数数目{D = }，应不少于1'
        IC, onset, duration = self.IC, self.onset, self.duration
        TVT_ = self.TVT_
        ΔtEIS, ΔtUDC, Δt = self.ΔtEIS, self.ΔtUDC, self.Δt
        n_jobs, algorithm, objective = self.n_jobs, self.algorithm, self.objective
        if verbose := self.verbose:
            print(f'固定{len(namesfixed_)}参数 {namesfixed_ = }\n'
                  f'待优化{D}参数 {namesoptimized_ = }\n'
                  f'电流倍率 {IC = }，充电时长 {duration = }s\n'
                  f'训练验证测试数据比例 {TVT_ = }\n'
                  f'阻抗测量间隔 {ΔtEIS = } s，电压测量间隔 {ΔtUDC = } s，时间步长 {Δt = } s\n'
                  f'拟合目标 {targets_ = }\n'
                  f'初始状态数据类型 {type(states0) = }\n'
                  f'joblib利用核数 {n_jobs = }\n'
                  f'优化算法 {algorithm = }，目标类型 {objective = }\n'
                  f'时段{onset}-{onset + duration}s，频段{self.f_.min():g}~{self.f_.max():g}Hz，共{len(self.f_)}频点\n')

        self.create_banded_experience()  # 创造带状化经验
        compute_cell = self.compute_cell
        compute_costs_ = self.compute_costs

        def function_costs_(x_: Sequence[float]) -> np.ndarray:
            """定义最小化目标函数"""
            pnormoptimized_ = {name: x for name, x in zip(namesoptimized_, x_)}
            pnorm_ = pnormfixed_ | pnormoptimized_
            cell = compute_cell(pnorm_, targets_=targets_, states0=states0)
            costs_ = compute_costs_(cell, targets_=targets_, dataset='training')
            return costs_

        def weight(Y__, w_) -> np.ndarray:
            # 拟合目标加权求和
            y_ = Y__.dot(w_)
            return y_

        if len(targets_)>1:
            X__ = qmc.LatinHypercube(d=D).random(n=Nsample)  # (Nsample, D)
            if verbose:
                print(f'{D}维空间采样{len(X__)}点，估计权重。', end='')
            Y__ = np.array(joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(joblib.delayed(function_costs_)(x_) for x_ in X__))
            if verbose:
                print(f'采样耗时{time.time() - timeStart:.1f}s，', end='')
            logic_ = Y__.min(axis=1)<100_0000
            X__ = X__[logic_]
            Y__ = Y__[logic_]
            if verbose:
                print(f'剔除{Nsample - len(Y__)}异常点')
            Ymax_ = Y__.max(axis=0)
            if verbose:
                print(f'各目标最大值 ymax_ = {np.array2string(Ymax_, formatter={'float_kind': '{:0.4e}'.format})}')
            σ_ = Y__.std(axis=0)
            if verbose:
                print(f'各目标标准差 σ_ = {np.array2string(σ_, formatter={'float_kind': '{:0.4e}'.format})}')
            w_ = 1/σ_
            if verbose:
                print(f'基于目标标准差倒数的权重 w_ = {np.array2string(w_, formatter={'float_kind': '{:.6f}'.format})}')
            X__ = X__[[weight(Y__, w_).argmin()]]  # (1, D) 保留大规模采样当中最优的1点
            del Y__
        else:
            if verbose:
                print(f'拟合目标仅1个，免于采样估计权重，赋权重1')
            w_ = np.array([1.])
            X__ = None

        if pcandidates__:
            # 给定候选解
            pnormcandidates__ = [self.Normalize(pcandidate_) for pcandidate_ in pcandidates__]
            Xcandidates__ = np.array([[pnormcandidate_[name] for name in namesoptimized_]
                                      for pnormcandidate_ in pnormcandidates__])
            if X__ is not None:
                X__ = np.vstack([X__,
                                 Xcandidates__])
            else:
                X__ = Xcandidates__

        if X__ is not None:
            print(f'给定{len(X__)}候选解！ Preset {len(X__)} candidate solution(s)!')
        else:
            print(f'无给定候选解！ Preset no candidate solution!')

        """优化辨识"""
        Optimizer = getattr(Optimization, self.algorithm)
        def function(x_: Sequence[float]) -> float:
            return weight(function_costs_(x_), w_).item()
        assert hyperparameters_Optimizer is None or isinstance(hyperparameters_Optimizer, dict), \
            f'{hyperparameters_Optimizer = }，应为dict或None'
        if hyperparameters_Optimizer is None:
            hyperparameters_Optimizer = {}
        optimizer = Optimizer(
            function=function,
            bounds__=[[0, 1]]*D,
            T=self.T, N=self.N,
            **hyperparameters_Optimizer,
            )
        optimizer.n_jobs = self.n_jobs
        X__, y_ = optimizer.minimize(X__=X__)

        print('验证、测试...')
        def function(x_: Sequence[float]) -> LPJTFP2D:
            pnormoptimized_ = {name: x for name, x in zip(namesoptimized_, x_)}
            pnorm_ = pnormfixed_ | pnormoptimized_
            cell = compute_cell(pnorm_, states0=states0)
            return cell
        cells_ = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(function)(x_) for x_ in X__)
        p__ = []  # list[dict] N个体的实际参数值
        costsTraining__   = []
        costsValidation__ = []
        costsTest__       = []
        for x_, cell in zip(X__, cells_):
            pnorm_ = pnormfixed_ | {name: x for name, x in zip(namesoptimized_, x_)}  # 合并固定、待优化参数
            p_ = self.Denormalize(pnorm_)
            p__.append(p_)
            costsTraining__.append(  tuple(compute_costs_(cell, dataset='training')))
            costsValidation__.append(tuple(compute_costs_(cell, dataset='validation')))
            costsTest__.append(      tuple(compute_costs_(cell, dataset='test')))
        else:
            dtype = [(target, float) for target in ('UDC', 'Zreal', 'Zimag')]
            costsTraining__   = np.array(costsTraining__,   dtype=dtype)  # (N, 3)
            costsValidation__ = np.array(costsValidation__, dtype=dtype)  # (N, 3)
            costsTest__       = np.array(costsTest__,       dtype=dtype)  # (N, 3)
            print('完成验证、测试...')

        self.record = {prop: getattr(self, prop) for prop in
                       ['Qnom', 'IC', 'tC', 'TVT_', 'onset', 'duration', 'Δt', 'ΔtUDC', 'ΔtEIS',
                        'f_', 'T', 'N',
                        'algorithm', 'objective',
                        'kwargs', 'I', 'pVC_',]}
        self.record |= {
            'tUDCmea_': self.tUDCmea_,
            'UDCmea_': self.UDCmea_,
            'tZmea_': self.tZmea_,
            'Zmea__': self.Zmea__,
            'pnormfixed_': pnormfixed_,
            'targets_': tuple(targets_),
            'states0': states0,
            'hyperparameters_Optimizer': hyperparameters_Optimizer,
            'p__': p__,
            'w_' : w_,
            'costsTraining__' : costsTraining__,
            'costsValidation__' : costsValidation__,
            'costsTest__' : costsTest__,
            'xGlobalOptimal_' : optimizer.xGlobalOptimal_,
            'yGlobalOptimal' : optimizer.yGlobalOptimal,
            'yCurrentOptimal_' : optimizer.yCurrentOptimal_,
            'yGlobalOptimal_' : optimizer.yGlobalOptimal_,
            'yMean_': optimizer.yMean_,
            'Nfunctions': optimizer.Nfunctions,
            }
        return self.record

    def create_banded_experience(self) -> None:
        """创造带状化经验"""
        cell = LPJTFP2D(**self.kwargs)
        cell.CC(-0.001, self.Δt*3).EIS()
        for key in self.banded_experience:
            self.banded_experience[key] = getattr(cell, key)
        del cell
        if self.verbose:
            print('已创造带状化经验！Banded experience has been created!')

    def compute_virtual_cell(self, pnorm_: dict[str, float]) -> LPJTFP2D:
        """计算虚拟电池"""
        assert set(pnorm_.keys())==set(self.names_), f'检查归一化参数字典pnorm_键，应符合{self.names_}'
        if self.verbose:
            print('计算虚拟电池…… Computing a virtual cell...')
        self.create_banded_experience()
        cell = self.compute_cell(pnorm_)
        self.UDCmea_ = cell.interpolate('U', t_=self.tUDCmea_)
        self.Zmea__  = cell.interpolate('Z_', t_=self.tZmea_, f_=self.f_)
        self.pVC_ = self.Denormalize(pnorm_)
        if self.verbose:
            print('完成计算虚拟电池。A virtual cell has been computed.')
        return cell

    def receive_measured_data(self,
            tUDCmea_: np.ndarray,  # 端电压直流分量测量时刻序列 [s]
            UDCmea_: np.ndarray,   # 端电压直流分量测量值 [V]
            tZmea_: np.ndarray,    # 阻抗测量时刻序列 [s]
            fZmea_: np.ndarray,    # 阻抗测量频率序列 [Hz]
            Zmea_: np.ndarray,   # 复阻抗测量值序列 [Ω]
            ) -> None:
        """接收并处理端电压UDC和DEIS测量数据"""
        onset, duration = self.onset, self.duration
        assert all((tUDCmea_[1:] - tUDCmea_[:-1])>0), 'tUDCmea_应为严格递增序列'
        assert tUDCmea_[0]<=onset and tUDCmea_[-1]>=(onset + duration), 'tUDCmea_范围应覆盖[onset, onset + duration]'
        assert len(tUDCmea_)==len(UDCmea_), 'len(tUDCmea_)应等于len(UDCmea_)'
        assert all((tZmea_[1:] - tZmea_[:-1])>0), 'tZmea_应为严格递增序列'
        assert tZmea_[0]<=max(onset, 50) and tZmea_[-1]>=(onset + duration), 'tUDCmea_范围应覆盖[max(onset, 50), onset + duration]'
        assert len(tZmea_)==len(fZmea_)==len(Zmea_), 'len(tUDCmea_), len(UDCmea_), len(Zmea_)三者应相等'
        uniquefZmea_ = np.unique(fZmea_)
        for f in self.f_:
            assert any(abs(f - uniquefZmea_)<1e-4), f'辨识设定频率{f}不包含于DEIS测量数据'

        self.UDCmea_ = interp1d(tUDCmea_, UDCmea_,
                                bounds_error=False, fill_value='extrapolate')(self.tUDCmea_)
        Z__ = []
        for f in self.f_:
            logic_ = abs(f - fZmea_)<1e-4
            Z_ = interp1d(tZmea_[logic_], Zmea_[logic_], bounds_error=False, fill_value='extrapolate')(self.tZmea_)
            Z__.append(Z_)
        self.Zmea__ = np.array(Z__).T  # (len(tZmea_), len(f_)) DEIS测量值
        if self.verbose:
            print('端电压和DEIS测量数据已处理。Voltage and DEIS measurements have been processed.')

    def compute_cell(self,
            pnorm_: dict[str, float],  # 归一化23参数集
            targets_: tuple[str] = ('UDC', 'Zreal', 'Zimag'),
            states0: dict[str, np.ndarray] | None = None,
            ) -> LPJTFP2D | dict[str, float]:
        # 计算电池
        try:
            p_ = self.Denormalize(pnorm_)
            cell = LPJTFP2D(**(self.kwargs | p_))  # 实例化
            I, ΔtEIS  = self.I, self.ΔtEIS
            for key, value in self.banded_experience.items():
                setattr(cell, key, value)  # 带状化经验
            if states0:
                cell.initialize_consistent(
                    θsneg__=states0['θsneg__'], θspos__=states0['θspos__'],
                    θe_=states0['θe_'], I=I)
                cell.EIS()
            tEnd = self.duration  # 充电终止时刻
            if tuple(targets_)==('UDC',):
                cell.CC(I, timeInterval=tEnd)
            else:
                while True:
                    timeInterval = min(ΔtEIS, tEnd - cell.t)
                    cell.CC(I, timeInterval=timeInterval)
                    if timeInterval==ΔtEIS:
                        cell.EIS()
                    if cell.t >= tEnd:
                        break
        except LPJTFP2D.Error as message:
            if 'cell' not in locals():
                cell = message.args[-1]
        return cell

    def compute_costs(self,
                      cell: LPJTFP2D | dict[str, float],
                      dataset: str = 'training',  # 训练/验证/测试
                      targets_: tuple[str] = ('UDC', 'Zreal', 'Zimag'),
                      ) -> np.ndarray:
        onset, duration = self.onset, self.duration
        if isinstance(cell, dict):
            # 若cell是字典，应直接惩罚
            if 'Qcell' in cell:
                Qcell, Qneg, Qpos = cell['Qcell'], cell['Qneg'], cell['Qpos']
                penalty = max(Qcell - Qneg, 0) + max(Qcell - Qpos, 0)
            elif 'ΔFmax' in cell:
                ΔFmax = cell['ΔFmax']
                θminneg = cell['θminneg']
                θmaxneg = cell['θmaxneg']
                θminpos = cell['θminpos']
                θmaxpos = cell['θmaxpos']
                penalty = (  max(ΔFmax - 1e-5, 0)
                           + max(θminneg - 1, 0) + max(-θminneg, 0)
                           + max(θmaxneg - 1, 0) + max(-θmaxneg, 0)
                           + max(θminpos - 1, 0) + max(-θminpos, 0)
                           + max(θmaxpos - 1, 0) + max(-θmaxpos, 0)
                           + max(θminpos - θmaxpos, 0) + max(θminneg - θmaxneg, 0) )
            else:
                raise ValueError('检查cell.keys()')
            penalty += 100_0000
            return np.full(len(targets_), penalty)
        elif cell.t<(duration - 1e-3):
            # 若cell模拟时间未达到duration，应直接惩罚
            penalty = duration - cell.t + 100_0000
            return np.full(len(targets_), penalty)

        TVT_ = self.TVT_
        tUDCmea_ = self.tUDCmea_
        tZmea_ = self.tZmea_
        objective = self.objective
        match dataset:
            case 'training':
                end = onset + TVT_[0]*duration
                logic_tUDC_ = tUDCmea_ <= end
                logic_tZ_ = tZmea_<=end
            case 'validation':
                start = onset + TVT_[0]*duration
                end = onset + sum(TVT_[:2])*duration
                logic_tUDC_ = (start<tUDCmea_) & (tUDCmea_<=end)
                logic_tZ_ = (start < tZmea_) & (tZmea_ <= end)
            case 'test':
                start = onset + sum(TVT_[:2])*duration
                logic_tUDC_ = start < tUDCmea_
                logic_tZ_ = start < tZmea_
            case _:
                raise ValueError(f'无 "{dataset}" 数据集')

        UDCsim_ = cell.interpolate('U', t_=tUDCmea_[logic_tUDC_] - onset)
        ΔU_ = UDCsim_ - self.UDCmea_[logic_tUDC_]

        match objective:
            case 'RMSE':
                costs_ = {'UDC': (ΔU_**2).mean()**0.5}
            case 'MSE':
                costs_ = {'UDC': (ΔU_**2).mean()}
            case 'MAE':
                costs_ = {'UDC': abs(ΔU_).mean()}
            case _:
                raise ValueError(f'未定义objective "{self.objective}"')

        if tuple(targets_)==('UDC',):
            pass
        else:
            Zmea__ = self.Zmea__[logic_tZ_]
            Zsim__ = cell.interpolate('Z_', t_=tZmea_[logic_tZ_] - onset, f_=self.f_)
            ΔZreal__ = Zsim__.real - Zmea__.real
            ΔZimag__ = Zsim__.imag - Zmea__.imag
            match objective:
                case 'RMSE':
                    costs_ |= {'Zreal': (ΔZreal__**2).mean()**0.5,
                               'Zimag': (ΔZimag__**2).mean()**0.5,}
                case 'MSE':
                    costs_ |= {'Zreal': (ΔZreal__**2).mean(),
                               'Zimag': (ΔZimag__**2).mean(),}
                case 'MAE':
                    costs_ |= {'Zreal': abs(ΔZreal__).mean(),
                               'Zimag': abs(ΔZimag__).mean(),}
                case _:
                    raise ValueError(f'未定义objective{objective}')

        return np.array([costs_[target] for target in targets_])

    def plot_iteration(self):
        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        x_ = range(1, len(self.record['yMean_']) + 1)
        ax.plot(x_, self.record['yMean_'], '-b', label='Current mean')
        ax.plot(x_, self.record['yCurrentOptimal_'], '-r', label='Current optimal')
        ax.plot(x_, self.record['yGlobalOptimal_'], '-k', label='Global optimal', alpha=0.7)
        ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        ax.set_yscale('log')
        plt.show()

    def plot_comparison(self, cell):
        fig = plt.figure(figsize=[14, 7])
        ax = fig.add_subplot(121)
        ax.plot(self.tUDCmea_,  self.UDCmea_,   'ok', label='Measured')
        ax.plot(cell.data['t'], cell.data['U'], '-r', label='Simulated')
        ax.legend()
        ax.set_ylabel(r'Terminal voltage ${\it U}_{DC}$ [V]')
        ax.set_xlabel('Time [s]')

        ax = fig.add_subplot(122)
        t_ = self.tZmea_
        for n, t in enumerate(t_):
            color = LPJTFP2D.get_color(t_, n)
            Z_ = self.Zmea__[t_==t][0]
            ax.plot(Z_.real*1000, -Z_.imag*1000, 'o--', color=color)
            Z_ = np.array(cell.data['Z'])[t==np.array(cell.data['tZ'])]
            ax.plot(Z_.real*1000, -Z_.imag*1000, '^-', color=color)
        h_ = [ax.plot([np.nan], [np.nan], ['o--', '^-'][n], color='k')[0] for n in range(2)]
        ax.legend(h_, ['Measured', 'Simulated'])
        ax.set_ylabel(r'Imaginary part of impedance $-{\it Z}″$ [mΩ]')
        ax.set_xlabel(r'Real part of impedance ${\it Z}′$ [mΩ]')

        fig.tight_layout()
        plt.show()

    def plot_DEISt(self,):
        """Nyquist图"""
        fig = plt.figure('Z measurement',figsize=(18, 6))
        ax = fig.add_subplot(131)
        color_ = []
        for n, Z_ in enumerate(self.Zmea__):
            color = LPJTFP2D.get_color(self.Zmea__, n)
            ax.plot(Z_.real*1e3, -Z_.imag*1e3, '-o', color=color)
            color_.append(color)
        ax.set_xlabel(r'Real part of dynamic impedance ${\it Z}′$ [mΩ]')
        ax.set_ylabel(r'Imaginary part of dynamic impedance $-{\it Z}″$ [mΩ]')
        ax.set_xticks(np.arange(ax.set_xlim()[0], ax.set_xlim()[1] + 1e-6, 0.5))
        ax.set_yticks(np.arange(ax.set_ylim()[0], ax.set_ylim()[1] + 1e-6, 0.1))
        ax.grid(ls='--', color=[.5]*3)
        colormap = mpl.colors.ListedColormap(color_)
        norm = mpl.colors.BoundaryNorm(self.tZmea_, colormap.N)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax,)
        cbar.set_ticks(cbar.get_ticks(), minor=False)
        # cbar.set_ticklabels([rf'{tick:.1f}' for tick in cbar.get_ticks()], minor=False)
        cbar.set_label(r'Time ${\it t}$ [s]')

        ax = fig.add_subplot(132)
        for n, f in enumerate(self.f_):
            color = LPJTFP2D.get_color(self.f_, n)
            ax.plot(self.tZmea_, self.Zmea__.real[:, f==self.f_].ravel()*1e3, '-o', color=color)
        ax.set_ylabel(r'Real part of dynamic impedance ${\it Z}′$ [mΩ]')
        ax.set_xlabel(r'Time ${\it t}$ [s]')
        ax.vlines(self.onset + self.duration*self.TVT_[0],       ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.vlines(self.onset + self.duration*sum(self.TVT_[:2]), ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.grid(axis='y', ls='--', color=[.5]*3)

        ax = fig.add_subplot(133)
        for n, f in enumerate(self.f_):
            color = LPJTFP2D.get_color(self.f_, n)
            ax.plot(self.tZmea_, -self.Zmea__.imag[:, f==self.f_].ravel()*1e3, '-o', color=color)
        ax.set_ylabel(r'Imaginary part of dynamic impedance $-{\it Z}″$ [mΩ]')
        ax.set_xlabel(r'Time ${\it t}$ [s]')
        ax.vlines(self.onset + self.duration*self.TVT_[0], ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.vlines(self.onset + self.duration*sum(self.TVT_[:2]), ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.grid(axis='y', ls='--', color=[.5]*3)

        fig.tight_layout()
        plt.show()

    def plot_Ut(self,):
        fig = plt.figure('UDC measurement', figsize=[6, 6*0.8])
        ax = fig.add_subplot(111)
        ax.set_position([.1, .11, .88, 0.87])
        ax.plot(self.tUDCmea_, self.UDCmea_, 'x-', label=r'Measured ${\it U}_{DC}$')
        ax.set_xlim(self.onset - 50, self.onset + self.duration + 50)
        ax.set_yticks(np.arange(2.8, 4.4, 0.2))
        ax.set_ylim(2.8, 4.4)
        ax.set_ylabel(r'Terminal voltage [V]')
        ax.set_xlabel(r'Time ${\it t}$ [s]')
        ax.vlines(self.onset + self.duration*self.TVT_[0], ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.vlines(self.onset + self.duration*sum(self.TVT_[:2]), ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.minorticks_on()
        ax.legend(loc='upper left')
        ax.grid(axis='y', ls='--', color=[.5]*3)
        plt.show()

    @staticmethod
    def trapezoid_on_interval(x_, y_, a, b):
        """在区间(a, b)按 x_, y_ 插值求数值积分"""
        ya = np.interp(a, x_, y_)
        yb = np.interp(b, x_, y_)
        logic_ = (a<x_) & (x_<b)
        x_ = np.hstack([a,  x_[logic_], b])
        y_ = np.hstack([ya, y_[logic_], yb])
        return trapezoid(y_, x_)

if __name__ == '__main__':
    task = Identification()
    task.create_banded_experience()

    # cell.CC(1, 20)

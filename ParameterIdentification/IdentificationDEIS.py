#%%
import time, joblib
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import ndarray, array, array2string, log10
from scipy.interpolate import interp1d
from scipy.stats import qmc


np.seterr(divide='ignore', over='ignore', invalid='ignore')

from ParameterIdentification.IdentificationBase import IdentificationBase
import optimization
from P2Dmodel import LPJTFP2D, set_matplotlib, get_color, Interpolate1D
set_matplotlib()


class IdentificationDEIS(IdentificationBase):
    __slots__ = (
        'IC', 'TVT_',
        'onset', 'duration',
        'I',
        'tUDCmea_', 'tZmea_', 'tTmea_',
        'UDCmea_', 'Zmea__',  'Tmea_',
        'interpT',
        )

    def __init__(self,
            IC: float | int = 1,     # 电流倍率 [C-rate]
            TVT_: tuple[float] = (.7, .15, .15),  # 训练集、验证集、测试集时长比例
            onset: float | int = 0,               # 充电开始时刻 [s]
            duration: float | int = 1000,         # 持续时间 [s]
            EISonset: float | int = 150,  # 有效EIS开始时刻 [s]
            Δt: int = 10,       # 时间步长 [s]
            ΔtUDC: int = 10,    # 端电压UDC测量时间间隔 [s]
            ΔtEIS: int = 50,    # EIS测量时间间隔 [s]
            **kwargs
            ):
        IdentificationBase.__init__(self, **kwargs)
        self.IC = IC; assert IC>=0, f'电流倍率{IC = }，应大于或等于0'
        self.TVT_ = TVT_ = array(TVT_)
        assert ((0<=TVT_).all() and (TVT_<=1).all()
                and sum(TVT_)==1
                and len(TVT_)==3), f'训练集、验证集、测试集数据比例TVT_应满足len(TVT_)==3，sum(TVT_)==1，且各元素取值范围为[0, 1]，当前{TVT_ = }'
        self.onset = onset;       assert onset>=0, f'充电开始时刻{onset = }，应大于或等于0 [s]'
        self.duration = duration; assert duration>0, f'持续时间{duration = }，应大于0 [s]'
        assert EISonset>=0, f'有效EIS开始时刻{EISonset = }，应大于0 [s]'
        assert Δt>0, f'时间步长{Δt = }，应大于0 [s]'
        assert ΔtUDC>0, f'端电压UDC测量时间间隔{ΔtUDC = }，应大于0 [s]'
        assert ΔtEIS>0, f'EIS测量时间间隔{ΔtEIS = }，应大于0 [s]'
        assert ΔtEIS % ΔtUDC == 0, f'ΔtEIS应可整除ΔtUDC，当前{ΔtEIS = }，{ΔtUDC = }'
        assert ΔtUDC % Δt == 0,    f'ΔtUDC应可整除Δt，当前{ΔtUDC = }，{Δt = }'
        self.I = -abs(IC*self.Qnom)  # 充电电流 [A]
        tEnd = onset + duration      # 终止时刻 [s]
        self.tUDCmea_ = tUDCmea_ = np.arange(onset, tEnd + 1e-6, ΔtUDC)  # 电压直流分量UDC测量时刻序列
        tZmea_ = np.arange(onset, tEnd + 1e-6, ΔtEIS)
        self.tZmea_ = tZmea_ = tZmea_[tZmea_>=EISonset]  # EIS测量时刻序列
        self.UDCmea_ = None  # (len(tUDCmea_),) 电压直流分量测量值
        self.Zmea__ = None   # (len(tZmea_), len(f_)) 阻抗测量值
        self.tTmea_ = np.arange(onset, tEnd + 1e-6, Δt)  # 温度测量时刻序列
        self.Tmea_ = None
        self.interpT = None
        self.kwargs.update({'Δt': Δt})
        if verbose := self.verbose:
            print(
                f'电流倍率 {IC = }\n'
                f'训练/验证/测试数据时长比例 {TVT_ = }\n'
                f'时段{onset}-{onset + duration}s，充电时长 {duration = } s\n'
                f'EIS测量间隔 {ΔtEIS = } s，电压测量间隔 {ΔtUDC = } s，时间步长 {Δt = } s\n'
                f'有效EIS开始时刻 {EISonset = } s'
                )

        end = onset + TVT_[0]*duration
        logic_tUDC_ = tUDCmea_ <= end
        logic_tZ_ = tZmea_ <= end
        NtUDC = logic_tUDC_.sum(); assert NtUDC>0, f'训练集电压测量点{NtUDC = }，应大于0'
        NtZ = logic_tZ_.sum();     assert NtZ>0,   f'训练集DEIS测量点{NtZ = }，应大于0'
        if verbose:
            print(f'训练集UDC数据点数目: {NtUDC}，DEIS数据点数目: {NtZ}')

        start = onset + TVT_[0]*duration
        end = onset + sum(TVT_[:2])*duration
        logic_tUDC_ = (start < tUDCmea_) & (tUDCmea_<=end)
        logic_tZ_ = (start < tZmea_) & (tZmea_ <= end)
        if TVT_[1]:
            NtUDC = logic_tUDC_.sum(); assert NtUDC>0, f'验证集电压测量点{NtUDC = }，应大于0'
            NtZ = logic_tZ_.sum();     assert NtZ>0,   f'验证集DEIS测量点{NtZ = }，应大于0'
            if verbose:
                print(f'验证集UDC数据点数目: {NtUDC}, DEIS数据点数目: {NtZ}')

        start = onset + sum(TVT_[:2])*duration
        logic_tUDC_ = start < tUDCmea_
        logic_tZ_ = start < tZmea_
        if TVT_[2]:
            NtUDC = logic_tUDC_.sum(); assert NtUDC>0, f'测试集电压测量点{NtUDC = }，应大于0'
            NtZ = logic_tZ_.sum();     assert NtZ>0,   f'测试集DEIS测量点{NtZ = }，应大于0'
            if verbose:
                print(f'测试集UDC数据点数目: {NtUDC}, DEIS数据点数目: {NtZ}')

    def identify(self,
            pfixed_: dict[str, float],
            T: int = 1000,                   # 迭代次数
            N: int = 500,                    # 个体数
            algorithm: str = 'STA',          # 优化算法
            n_jobs: int = -1,                # joblib并行执行CPU核数
            batch_size: int | str = 'auto',  # joblib并行执行batch_size
            hyperparameters_Optimizer: dict | None = None,  # 优化算法超参数
            states0: dict[str, np.ndarray] | None = None,
            pcandidates__: list[dict[str, float]] | None = None,
            Nsample: int = 5_0000,  # 估计权重、筛选可行初始解的采样规模
            ) -> dict:
        timeStart = time.time()
        assert all(name in self.names_ for name in pfixed_), "固定参数集存在非法参数"
        if states0 is not None:
            requiredStateNames_ = {'θsneg__', 'θspos__', 'θe_'}
            missingStateNames_ = requiredStateNames_ - states0.keys()
            if missingStateNames_:
                raise ValueError(f'states0缺少状态量：{sorted(missingStateNames_)}')
        namesfixed_ = tuple(pfixed_.keys())  # 固定参数
        namesoptimized_ = tuple([
            str(name) for name in self.names_
            if name not in pfixed_])  # 待优化参数
        D = len(namesoptimized_)
        assert D>=1, f'待优化参数数目{D = }，应至少为1'
        if verbose := self.verbose:
            print(
                f'固定{len(namesfixed_)}参数：{namesfixed_ = }\n'
                f'待优化{D}参数：{namesoptimized_ = }\n'
                f'迭代次数 {T = }，个体数 {N = }\n'
                f'优化算法 {algorithm = }\n'
                f'joblib利用核数 {n_jobs = }，{batch_size = }\n'
                f'初始状态数据类型 {type(states0) = }\n'
                )

        self.create_experiences_()  # 创造带状化经验
        compute_cell = self.compute_cell
        compute_costs_ = self.compute_costs_
        pnormfixed_ = self.Normalize(pfixed_)

        def function_costs_(x_: ndarray) -> ndarray:
            pnormoptimized_ = {name: x for name, x in zip(namesoptimized_, x_)}
            pnorm_ = pnormfixed_ | pnormoptimized_
            cell   = compute_cell(pnorm_, states0)
            costs_ = compute_costs_(cell, 'training')
            return costs_

        def compute_costs_parallel_(X__: ndarray) -> ndarray:
            return array(joblib.Parallel(
                n_jobs=n_jobs, backend="loky", batch_size=batch_size,
                )(joblib.delayed(function_costs_)(x_) for x_ in X__))

        def feasible_(Y__: ndarray) -> ndarray:
            return np.all(Y__ < 100_0000, axis=1)

        # 给定候选解应先检查可行性
        Xcandidates__ = np.empty((0, D))
        Ycandidates__ = np.empty((0, len(self.targets_)))
        if pcandidates__:
            pnormcandidates__ = [self.Normalize(pcandidate_) for pcandidate_ in pcandidates__]
            Xcandidates__ = array([[pnormcandidate_[name] for name in namesoptimized_]
                                   for pnormcandidate_ in pnormcandidates__])
            Ycandidates__ = compute_costs_parallel_(Xcandidates__)
            logicCandidates_ = feasible_(Ycandidates__)
            Xcandidates__ = Xcandidates__[logicCandidates_]
            Ycandidates__ = Ycandidates__[logicCandidates_]
            print(f'给定{len(pcandidates__)}个候选解，其中{len(Xcandidates__)}个可行！')

        Xlhs__ = np.empty((0, D))  # 已经获得的可行LHS样本
        Ylhs__ = np.empty((0, len(self.targets_)))
        # 确定优化目标函数
        if len(self.targets_)==1:
            def objective_from_costs_(costs_: ndarray) -> float:
                return costs_.item()
        else:
            match self.weighting:
                case 'balanced':
                    """"大规模采样估计权重"""
                    Xlhs__ = qmc.LatinHypercube(d=D).random(n=Nsample)  # (Nsample, D)
                    if verbose:
                        print(f'{D}维空间采样{Nsample}点，估计权重。Estimating weights...', end='')
                    Ylhs__ = compute_costs_parallel_(Xlhs__)
                    if verbose:
                        print(f'采样耗时{time.time() - timeStart:.1f}s，', end='')
                    logic_ = feasible_(Ylhs__)
                    Xlhs__ = Xlhs__[logic_]
                    Ylhs__ = Ylhs__[logic_]
                    if verbose:
                        print(f'剔除{Nsample - len(Ylhs__)}异常点')
                    if len(Ylhs__)==0:
                        raise RuntimeError('LHS采样未得到可行解，无法估计权重和初始化种群')
                    Ymax_ = Ylhs__.max(axis=0)
                    if verbose:
                        print(f'各目标最大值 ymax_ = {array2string(Ymax_, formatter={'float_kind': '{:0.4e}'.format})}')
                    σ_ = Ylhs__.std(axis=0)
                    if verbose:
                        print(f'各目标标准差 σ_ = {array2string(σ_, formatter={'float_kind': '{:0.4e}'.format})}')
                    self.w_ = w_ = 1/σ_
                    if verbose:
                        print(f'基于目标标准差倒数的权重 w_ = {array2string(w_, formatter={'float_kind': '{:.6f}'.format})}')
                    del logic_
                    def objective_from_costs_(costs_: ndarray) -> float:
                        return costs_.dot(w_).item()
                case 'given':
                    self.w_ = w_ = array(self.w_)
                    if verbose:
                        print(f'强制给定权重 w_ = {array2string(w_, formatter={'float_kind': '{:.6f}'.format})}')
                    def objective_from_costs_(costs_: ndarray) -> float:
                        return costs_.dot(w_).item()
                case 'adaptive':
                    self.w_ = w_ = None
                    if verbose:
                        print('采用自适应权重！')
                    def objective_from_costs_(costs_: ndarray) -> float:
                        # 自适应权重，目标值大，自动赋大权重，目标值小，自动赋小权重
                        threshold = 1 - 1e-8
                        logic_ = costs_ >= threshold
                        if any(logic_):
                            return float(1e6 + (costs_[logic_] - threshold).sum())
                        else:
                            w_ = 1/log10(costs_ + 1e-8)
                            w_ /= w_.sum()
                            return costs_.dot(w_).item()

        def function(x_: ndarray) -> float:
            return objective_from_costs_(function_costs_(x_))

        # balanced复用估计权重时的可行LHS样本；其他情况专门进行可行性LHS采样
        NlhsRequired = max(N - len(Xcandidates__), 0)
        while len(Xlhs__)<NlhsRequired:
            samplingStart = time.time()
            if verbose:
                print(f'{D}维空间采样{N}点，筛选可行初始解...', end='')
            Xsample__ = qmc.LatinHypercube(d=D).random(n=N)
            Ysample__ = compute_costs_parallel_(Xsample__)
            logic_ = feasible_(Ysample__)
            Xsample__ = Xsample__[logic_]
            Ysample__ = Ysample__[logic_]
            if verbose:
                print(f'耗时{time.time() - samplingStart:.1f}s，'
                      f'获得{len(Xsample__)}个可行解，剔除{N - len(Xsample__)}个异常点')
            if len(Xsample__)==0:
                print('本次LHS采样未得到可行解，再次采样')
                continue
            Xlhs__ = np.vstack([Xlhs__, Xsample__])
            Ylhs__ = np.vstack([Ylhs__, Ysample__])

        Xpool__ = np.vstack([Xcandidates__, Xlhs__])
        Ypool__ = np.vstack([Ycandidates__, Ylhs__])
        ypool_ = array([objective_from_costs_(costs_) for costs_ in Ypool__])
        idxBest = int(ypool_.argmin())

        # 可行的给定候选解优先保留；全池最优解一定保留，其余从可行池中无放回随机抽取
        if len(Xcandidates__)<N:
            idxSelected_ = list(range(len(Xcandidates__)))  # 前面的都是给定的候选解
        else:
            idxSelected_ = []
        if idxBest in idxSelected_:
            idxSelected_.remove(idxBest)
        idxSelected_.insert(0, idxBest)
        idxRemaining_ = np.setdiff1d(np.arange(len(Xpool__)), idxSelected_)  # 剩余解
        Nrandom = N - len(idxSelected_)
        if Nrandom:
            idxSelected_.extend(np.random.choice(idxRemaining_, Nrandom, replace=False).tolist())
        X__ = Xpool__[idxSelected_]
        print(f'从{len(Xpool__)}个可行解中选取{len(X__)}个初始解，最优解固定保留！')

        """优化辨识"""
        Optimizer = getattr(optimization, algorithm)

        assert hyperparameters_Optimizer is None or isinstance(hyperparameters_Optimizer, dict), \
            f'{hyperparameters_Optimizer = }，应为dict或None'
        if hyperparameters_Optimizer is None:
            hyperparameters_Optimizer = {}
        optimizer = Optimizer(
            function=function,
            bounds__=[[0, 1]]*D,
            T=T, N=N,
            n_jobs=n_jobs,
            batch_size=batch_size,
            reuse_parallel=True,
            **hyperparameters_Optimizer,
            )
        X__, y_ = optimizer.minimize(X__=X__)

        print('验证、测试...')
        def function(x_: Sequence[float]) -> LPJTFP2D:
            pnormoptimized_ = {name: x for name, x in zip(namesoptimized_, x_)}
            pnorm_ = pnormfixed_ | pnormoptimized_
            cell = compute_cell(pnorm_, states0=states0)
            return cell
        cells_ = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(function)(x_) for x_ in X__)

        tUDCmea_ = self.tUDCmea_
        tZmea_ = self.tZmea_
        TVT_ = self.TVT_
        onset, duration = self.onset, self.duration
        Denormalize = self.Denormalize
        p__: list[dict] = []    # 合格个体实际参数值
        UDCsim__ = []           # 合格个体模拟电压序列
        Zsim___ = []            # 合格个体模拟DEIS序列
        costsTraining__   = []  # 合格个体训练集拟合目标值
        costsValidation__ = []  # 合格个体验证集拟合目标值
        costsTest__       = []  # 合格个体测试集拟合目标值
        for n, (x_, cell) in enumerate(zip(X__, cells_)):
            if isinstance(cell, LPJTFP2D) and cell.t>=(duration - 1e-3):
                pnorm_ = pnormfixed_ | {name: x for name, x in zip(namesoptimized_, x_)}  # 合并固定、待优化参数
                p_ = Denormalize(pnorm_)
                p__.append(p_)
                UDCsim__.append(cell('U', t_=tUDCmea_ - onset))
                if any('Z' in target for target in self.targets_):
                    Zsim___.append( cell('Z_', t_=tZmea_ - onset, f_=self.f_))
                costsTraining__.append(      tuple(compute_costs_(cell, 'training',)))
                if TVT_[1]:
                    costsValidation__.append(tuple(compute_costs_(cell, 'validation',)))
                if TVT_[2]:
                    costsTest__.append(      tuple(compute_costs_(cell, 'test',)))
        UDCsim__ = array(UDCsim__)
        if any('Z' in target for target in self.targets_):
            Zsim___  = array(Zsim___)
        dtype = [(target, float) for target in self.targets_]
        costsTraining__       = array(costsTraining__,   dtype=dtype)  # (len(p__), len(targets_))
        if TVT_[1]:
            costsValidation__ = array(costsValidation__, dtype=dtype)  # (len(p__), len(targets_))
        if TVT_[2]:
            costsTest__       = array(costsTest__,       dtype=dtype)  # (len(p__), len(targets_))
        print(f'完成验证、测试，从{N}个体保留{len(p__)}合格个体')

        self.record = record = {
            properti: getattr(self, properti) for properti in
            ['Qnom', 'thermalModel',
             'targets_', 'weighting', 'w_', 'objective',
             'IC', 'TVT_', 'onset', 'duration', 'f_',
             'I', 'pVC_',
             'tUDCmea_', 'tZmea_', 'tTmea_',
             'UDCmea_', 'Zmea__', 'Tmea_',
             ]}
        record['kwargs'] = self.kwargs.copy()
        del record['kwargs']['dUOCPdθsneg']
        del record['kwargs']['dUOCPdθspos']
        record.update({
            'p__': p__,
            'UDCsim__': UDCsim__,
            'Zsim___': Zsim___,
            'costsTraining__':   costsTraining__,
            'costsValidation__': costsValidation__,
            'costsTest__':       costsTest__,
            'pfixed_': pfixed_,
            'T': T, 'N': N,
            'algorithm': algorithm,
            'hyperparameters_Optimizer': hyperparameters_Optimizer,
            'states0': states0,
            'xGlobalOptimal_' : optimizer.xGlobalOptimal_,
            'yGlobalOptimal' : optimizer.yGlobalOptimal,
            'yCurrentOptimal_' : optimizer.yCurrentOptimal_,
            'yGlobalOptimal_' : optimizer.yGlobalOptimal_,
            'yMean_': optimizer.yMean_,
            'Nfunctions': optimizer.Nfunctions,
            'timeconsumed': (time.time() - timeStart)/3600,
            })
        print(f'辨识耗时{record['timeconsumed']:.2f}h')
        return record

    def compute_virtual_cell(self, pnorm_: dict[str, float]) -> LPJTFP2D:
        """计算虚拟电池"""
        assert set(pnorm_.keys())==set(self.names_), f'检查归一化参数字典pnorm_键，应符合{self.names_}'
        if self.verbose:
            print('计算虚拟电池…… Computing a virtual cell...')
        self.create_experiences_()
        cell = self.compute_cell(pnorm_)
        self.UDCmea_ = cell('U',  t_=self.tUDCmea_ - self.onset)
        self.Zmea__  = cell('Z_', t_=self.tZmea_ - self.onset, f_=self.f_)
        self.pVC_ = self.Denormalize(pnorm_)
        if self.verbose:
            print('完成计算虚拟电池。A virtual cell has been computed.')
        return cell

    def receive_measured_data(self,
            tUDCmea_: ndarray,  # 端电压直流分量测量时刻序列 [s]
            UDCmea_: ndarray,   # 端电压直流分量测量值 [V]
            tZmea_: ndarray,    # 阻抗测量时刻序列 [s]
            fZmea_: ndarray,    # 阻抗测量频率序列 [Hz]
            Zmea_: ndarray,   # 复阻抗测量值序列 [Ω]
            tTmea_: ndarray | None = None,  # 温度测量时刻序列 [s]
            Tmea_: ndarray | None = None,   # 温度测量值 [K]
            ) -> None:
        """接收并处理端电压UDC和DEIS测量数据"""
        onset, duration = self.onset, self.duration
        if self.thermalModel:
            assert (tTmea_ is not None) and (Tmea_ is not None), (
                'thermalModel=True时，receive_measured_data()必须传入tTmea_和Tmea_温度曲线')
        assert all((tUDCmea_[1:] - tUDCmea_[:-1])>0), 'tUDCmea_应为严格递增序列'
        assert tUDCmea_[0]<=onset and tUDCmea_[-1]>=(onset + duration), \
            (f'tUDCmea_范围应覆盖[onset, onset + duration]，i.e., [{onset}, {onset + duration}]，'
             f'当前{tUDCmea_[0] = :.1f}，{tUDCmea_[-1] = :.1f}')
        assert len(tUDCmea_)==len(UDCmea_), 'len(tUDCmea_)应等于len(UDCmea_)'
        assert all((tZmea_[1:] - tZmea_[:-1])>0), 'tZmea_应为严格递增序列'
        assert tZmea_[0]<=max(50, onset) and tZmea_[-1]>=(onset + duration), \
            (f'tZmea_范围应覆盖[{max(50, onset)}, {onset + duration}]，'
             f'当前{tZmea_[0] = :.1f}，{tZmea_[-1] = :.1f}')
        assert len(tZmea_)==len(fZmea_)==len(Zmea_), 'len(tUDCmea_), len(UDCmea_), len(Zmea_)三者应相等'
        uniquefZmea_ = np.unique(fZmea_)
        for f in self.f_:
            assert any(abs(f - uniquefZmea_)<1e-4), f'辨识设定频率{f}不包含于DEIS测量数据'

        self.UDCmea_ = interp1d(tUDCmea_, UDCmea_,
                                bounds_error=False,
                                fill_value='extrapolate')(self.tUDCmea_)
        Z__ = []
        for f in self.f_:
            logic_ = abs(f - fZmea_)<1e-4
            Z_ = interp1d(tZmea_[logic_], Zmea_[logic_],
                          bounds_error=False, fill_value='extrapolate')(self.tZmea_)
            Z__.append(Z_)
        self.Zmea__ = array(Z__).T  # (len(tZmea_), len(f_)) DEIS测量值

        if (tTmea_ is not None) and (Tmea_ is not None):
            self.Tmea_ = interp1d(tTmea_, Tmea_,
                                  bounds_error=False,
                                  fill_value='extrapolate')(self.tTmea_)
            if self.thermalModel:
                self.interpT = Interpolate1D(self.tTmea_ - self.onset, self.Tmea_)  # 插值函数 温度[T]-时间 [s]

        if self.verbose:
            print('端电压和DEIS测量数据已处理。Voltage and DEIS measurements have been processed.')

    def compute_cell(self,
            pnorm_: dict[str, float], # 归一化参数集
            states0: dict[str, np.ndarray] | None = None,
            ) -> LPJTFP2D | dict[str, float]:
        # 计算电池
        try:
            p_ = self.Denormalize(pnorm_)
            cell = LPJTFP2D(**(self.kwargs | p_))  # 实例化
            I = self.I  # 充电电流
            exps_ = self.experiences_ # 经验
            # 时域因变量计算经验
            cell.banded_experience_of_J__  = exps_['banded_experience_of_J__'].copy()
            cell.bK_ = exps_['bK_'].copy()
            cell.ravelK_ = exps_['ravelK_'].reshape(cell.bK_.size, -1).copy().ravel()
            cell.sK = exps_['sK']

            if states0:
                cell.initialize_consistent(
                    θsneg__=states0['θsneg__'],
                    θspos__=states0['θspos__'],
                    θe_=states0['θe_'], I=I)

            tEnd = self.duration  # 充电终止时刻
            tZmea_ = self.tZmea_
            interpT = self.interpT
            thermalModel = self.thermalModel
            if tuple(self.targets_)==('UDC',):
                cell.CC(I, tEnd, thermalModel, Ts=interpT)
            else:
                # 频域因变量计算经验
                cell.banded_experience_of_Kf__ = exps_['banded_experience_of_Kf__']
                cell.bKf_ = exps_['bKf_'].copy()
                cell.ravelKf_ = exps_['ravelKf_'].reshape(cell.bKf_.size, -1).copy().ravel()
                cell.sKf = exps_['sKf']

                cell.CC(I, tEnd, thermalModel, tEIS_=tZmea_ - self.onset, Ts=interpT)
        except LPJTFP2D.Error as message:
            if 'cell' not in locals():  # 实例化不成功，返回字典
                cell = message.args[-1]
        return cell

    def compute_costs_(self,
            cell: LPJTFP2D | dict[str, float],
            dataset: str = 'training',  # 训练/验证/测试
            ) -> np.ndarray:
        onset, duration = self.onset, self.duration
        targets_ = self.targets_
        if isinstance(cell, dict):
            # 若cell是字典，应直接惩罚
            if 'constraintViolation' in cell:
                penalty = cell['constraintViolation']
            else:
                raise ValueError('检查cell.keys()')
            penalty += 100_0000
            return np.full(len(targets_), penalty)
        elif cell.t<(duration - 1e-3):
            # 若cell模拟时间未达到duration，应直接惩罚
            penalty = (duration - cell.t)/duration + 100_0000
            return np.full(len(targets_), penalty)

        TVT_ = self.TVT_
        tUDCmea_ = self.tUDCmea_
        tZmea_ = self.tZmea_
        match dataset:
            case 'training':
                end   = onset + TVT_[0]*duration
                logic_tUDC_ = tUDCmea_ <= end
                logic_tZ_ = tZmea_<=end
            case 'validation':
                start = onset + TVT_[0]*duration
                end   = onset + sum(TVT_[:2])*duration
                logic_tUDC_ = (start<tUDCmea_) & (tUDCmea_<=end)
                logic_tZ_ = (start < tZmea_) & (tZmea_ <= end)
            case 'test':
                start = onset + sum(TVT_[:2])*duration
                logic_tUDC_ = start < tUDCmea_
                logic_tZ_ = start < tZmea_
            case _:
                raise ValueError(f'无定义 {dataset = }')

        solve_objective = IdentificationBase.solve_objective
        costs_ = {}
        objective = self.objective
        if 'UDC' in targets_:
            UDCsim_ = cell('U', t_=tUDCmea_[logic_tUDC_] - onset)
            ΔU_ = UDCsim_ - self.UDCmea_[logic_tUDC_]
            costs_['UDC'] = solve_objective(ΔU_, objective)
        
        if any(('Z' in target) for target in targets_):
            Zsim__ = array(cell.data['Z_'])[logic_tZ_]
            ΔZ__ = Zsim__ - self.Zmea__[logic_tZ_]  # complex
            if 'Zreal' in targets_:
                ΔZreal__ = ΔZ__.real
                costs_['Zreal'] = solve_objective(ΔZreal__, objective)
            if 'Zimag' in targets_:
                ΔZimag__ = ΔZ__.imag
                costs_['Zimag'] = solve_objective(ΔZimag__, objective)
            if 'Z' in targets_:
                ΔZabs__ = abs(ΔZ__)
                costs_['Z'] = solve_objective(ΔZabs__, objective)

        return array([costs_[target] for target in targets_])

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
            color = get_color(t_, n)
            Z_ = self.Zmea__[t_==t][0]
            ax.plot(Z_.real*1000, -Z_.imag*1000, 'o--', color=color)
            Z_ = array(cell.data['Z'])[t==array(cell.data['tZ'])]
            ax.plot(Z_.real*1000, -Z_.imag*1000, '^-', color=color)
        h_ = [ax.plot([np.nan], [np.nan], ['o--', '^-'][n], color='k')[0] for n in range(2)]
        ax.legend(h_, ['Measured', 'Simulated'])
        ax.set_ylabel(r'Imaginary part of impedance $-{\it Z}″$ [mΩ]')
        ax.set_xlabel(r'Real part of impedance ${\it Z}′$ [mΩ]')

        fig.tight_layout()
        plt.show()

    def plot_DEIS(self,):
        """Nyquist图"""
        fig = plt.figure('Z measurement',figsize=(18, 6))
        ax = fig.add_subplot(131)
        color_ = []
        for n, Z_ in enumerate(self.Zmea__):
            color = get_color(self.Zmea__, n)
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

        f_ = self.f_
        ax = fig.add_subplot(132)
        for n, f in enumerate(f_):
            color = get_color(f_, n)
            ax.plot(self.tZmea_, self.Zmea__.real[:, f==f_].ravel()*1e3, '-o', color=color)
        ax.set_ylabel(r'Real part of dynamic impedance ${\it Z}′$ [mΩ]')
        ax.set_xlabel(r'Time ${\it t}$ [s]')
        ax.vlines(self.onset + self.duration*self.TVT_[0],       ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.vlines(self.onset + self.duration*sum(self.TVT_[:2]), ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.grid(axis='y', ls='--', color=[.5]*3)

        ax = fig.add_subplot(133)
        for n, f in enumerate(f_):
            color = get_color(f_, n)
            ax.plot(self.tZmea_, -self.Zmea__.imag[:, f==f_].ravel()*1e3, '-o', color=color)
        ax.set_ylabel(r'Imaginary part of dynamic impedance $-{\it Z}″$ [mΩ]')
        ax.set_xlabel(r'Time ${\it t}$ [s]')
        ax.vlines(self.onset + self.duration*self.TVT_[0], ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.vlines(self.onset + self.duration*sum(self.TVT_[:2]), ax.get_ylim()[0], ax.get_ylim()[1], ls='--', color=[.5]*3)
        ax.grid(axis='y', ls='--', color=[.5]*3)

        fig.tight_layout()
        plt.show()

    def plot_UDC(self, ):
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


if __name__ == '__main__':
    task = IdentificationDEIS(TVT_=[1, 0, 0], ΔtEIS=50, thermalModel=True)
    task.create_experiences_()

    task.interpT = Interpolate1D([0, 1000,], [298.15, 309.15])

    cell = task.compute_cell({name: .5 for name in task.names_})
    cell.plot_Nyquist()

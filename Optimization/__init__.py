#%%
from typing import Callable
import os, joblib, time, math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from numpy.random import uniform
import scipy
import matplotlib
matplotlib.use('Qt5Agg')  # TkAgg/Qt5Agg
import matplotlib.pyplot as plt
fontname = 'Times New Roman'  # 字体
plt.rcParams['font.serif'] = [fontname]       # 衬线字体
plt.rcParams['font.sans-serif'] = [fontname]  # 无衬线字体
plt.rcParams['font.size'] = 18                # 字号
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
plt.rcParams['mathtext.default'] = 'regular'  # 默认样式：正体、不加粗
plt.rcParams['mathtext.rm'] = 'STIXGeneral:regular'      # 正体、不加粗
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'       # 斜体、不加粗
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'  # 斜体、加粗


class Optimizer:
    def __init__(self,
            function: Callable,  # 目标函数
            bounds__,      # 决策变量边界
            T: int = 1000,  # 迭代次数
            N: int = 200,  # 种群规模
            ):
        assert callable(function), '目标函数function应为可调用对象'
        bounds__ = np.array(bounds__)  # 决策变量边界
        assert bounds__.ndim==2 and bounds__.shape[1]==2, '决策变量边界bounds__应可转化成满足.shape[1]==2的2维np.ndarray'
        assert (bounds__[:, 0]<bounds__[:, 1]).all(), '决策变量边界bounds__的下限应小于上限'
        assert isinstance(T, int) and T>0, '最大迭代次数T应为正整数'
        assert isinstance(N, int) and N>0, '种群规模N应为正整数'
        self.function = function  # 目标函数
        self.bounds__ = bounds__  # 决策变量边界
        self.T = T                # 迭代次数
        self.N = N                # 种群规模
        self.D = len(bounds__)    # 决策变量维数
        self.xGlobalOptimal_ = None         # 全局最优解
        self.yGlobalOptimal = float('inf')  # 全局最优目标函数值
        self.yGlobalOptimal_ = []   # 每一代的全局最优目标函数值
        self.yCurrentOptimal_ = []  # 每一代种群的最优目标函数值
        self.yMean_ = []            # 每一代种群的平均目标函数值
        self.n_jobs = 1             # joblib并行执行数目
        self.boundaryHandlingMethod = 'reflect'  # 边界处理方法 reflect / clip
        self.sampleMethod = 'LHS'  # 采样方法 'LHS'/'BCM'
        self.Nfunctions = 0        # function的执行次数
        self.timeStart = time.time()
        self.timePast = time.time()

    def initialize(self, X__: np.ndarray | None = None):
        """初始化种群"""
        sampler = getattr(self, f'sampler{self.sampleMethod}')
        if X__ is None:
            X__ = sampler(self.N)  # 随机初始化种群 (N, D)
        else:
            assert isinstance(X__, np.ndarray) and X__.ndim==2 and X__.shape[1]==self.D, \
            '初始种群X__应为None，或shape为(?, D)的ndarray'
            X__ = X__[:self.N]   # 取前N个体
            if len(X__)<self.N:  # 补足N个体
                X__ = np.vstack([X__,
                                 sampler(self.N - len(X__))])
        self.yGlobalOptimal_  = []  # 每代的全局最优目标函数值
        self.yCurrentOptimal_ = []  # 每代种群的最优目标函数值
        self.yMean_ = []            # 每代种群的平均目标函数值
        return X__

    def samplerLHS(self, N):
        """拉丁超立方采样（Latin hypercube sampling）"""
        sampler = scipy.stats.qmc.LatinHypercube(d=self.D)  # 拉丁超立方采样器
        X__ = sampler.random(n=N)  # (N, D)采样
        for d, bound_ in enumerate(self.bounds__):
            X__[:, d] = bound_[0] + (bound_[1] - bound_[0])*X__[:, d]
        return X__

    def samplerBCM(self, N):
        """伯努利混沌映射（Bernoulli chaotic mapping）"""
        λ = 0.4
        X__ = np.empty([N, self.D])  # (N, D)采样
        for d in range(self.D):
            z = math.e*(d+1) % 1
            z_ = [z]
            for _ in range(N - 1):
                if z<=(1 - λ):
                    z = z/(1 - λ)
                else:
                    z = (z - (1 - λ))/λ
                z_.append(z)
            X__[:, d] = z_
        for d, bound_ in enumerate(self.bounds__):
            X__[:, d] = bound_[0] + (bound_[1] - bound_[0])*X__[:, d]
        return X__

    def boundaryHandling(self,
                         Xnew__: np.ndarray,  # 新种群
                         Xold__: np.ndarray,  # 原种群
                         ):
        """边界条件"""
        if self.boundaryHandlingMethod=='reflect':
            # 随机反射
            N, _ = Xnew__.shape
            lb__ = np.tile(self.bounds__[:, 0], [N, 1])  # (N, D)
            ub__ = np.tile(self.bounds__[:, 1], [N, 1])  # (N, D)
            logic__ = Xnew__>ub__
            Xnew__[logic__] = uniform(Xold__[logic__], ub__[logic__])
            logic__ = Xnew__<lb__
            Xnew__[logic__] = uniform(lb__[logic__], Xold__[logic__])
        elif self.boundaryHandlingMethod=='clip':
            # 裁剪
            Xnew__ = np.clip(Xnew__, self.bounds__[:, 0], self.bounds__[:, 1])  # 边界处理
        else:
            raise ValueError(f'未定义边界处理方法"{self.boundaryHandlingMethod}"')
        return Xnew__

    def record(self, X__: np.ndarray, y_: np.ndarray, t: int | None = None):
        """更新全局最优、记录迭代过程的目标函数值"""
        i = y_.argmin()
        if y_[i]<self.yGlobalOptimal:
            self.yGlobalOptimal = y_[i]
            self.xGlobalOptimal_ = X__[i].copy()
        self.yGlobalOptimal_.append(self.yGlobalOptimal)  # 记录：当代全局最优目标函数值
        self.yCurrentOptimal_.append(y_[i])  # 记录：当代最优目标函数值
        self.yMean_.append(y_.mean())        # 记录：当代平均目标函数值

        timePast = self.timePast
        timeNow = time.time()
        self.timePast = timeNow
        if t:
            Δt = timeNow - timePast
            Nfunctions = self.Nfunctions
            ΔtTotal = (timeNow - self.timeStart)/3600
            print(f"迭代{t}/{self.T} "
                  f"当代最优目标{y_[i]:.4f}，全局最优目标{self.yGlobalOptimal:.4f}，"
                  f"效率{Δt:.2f}秒/代，{ΔtTotal*3600/Nfunctions:.4f}秒/个体，累计执行目标函数{Nfunctions}次，"
                  f"已耗时{ΔtTotal:.2f}小时，"
                  f"剩余{(self.T - t)/t*ΔtTotal:.2f}小时", end='\r')

    def batchObjective(self,
                       X__: np.ndarray,  # (N, D) 种群
                       ) -> np.ndarray:
        """计算种群各个体的目标函数值"""
        if self.n_jobs==1:
            y_ = [self.function(x_) for x_ in X__]  # 所有个体的目标函数值
        else:
            y_ = joblib.Parallel(n_jobs=self.n_jobs, backend='loky', batch_size=1)(
                 joblib.delayed(self.function)(x_) for x_ in X__)
        y_ = np.array(y_)
        self.Nfunctions += len(y_)
        return y_

    def plot(self):
        """作图：Objective-Iteration"""
        fig = plt.figure(self.__class__.__name__, figsize=[10, 7])
        ax = fig.add_subplot(111)
        x_ = range(1, len(self.yGlobalOptimal_) + 1)
        ax.plot(x_, self.yMean_, '-b', label='Current mean')
        ax.plot(x_, self.yCurrentOptimal_, '-r', label='Current optimal')
        ax.plot(x_, self.yGlobalOptimal_, '-k', label='Global optimal', alpha=0.7)
        ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        ax.set_yscale('log')
        plt.show()

    @staticmethod
    def timer():
        """.minimize方法的装饰器"""
        def timer(minimize: Callable):
            def wrapper(self, *args, **kwargs):
                self.timeStart = time.time()
                self.Nfunctions = 0  # 执行次数置零
                result = minimize(self, *args, **kwargs)  # 执行函数
                print(f'达到最大迭代次数{self.T}，总耗时{(time.time() - self.timeStart)/3600:.2f} h，目标函数执行总数{self.Nfunctions}')
                print(f'最优目标函数值：{self.yGlobalOptimal}')
                print(f'最优解：{self.xGlobalOptimal_}')
                return result
            return wrapper
        return timer

    def resample(self, X__: np.ndarray):
        """重采样生成新种群"""
        Xmin_, Xmax_ = X__.min(axis=0), X__.max(axis=0)      # 种群D个维度的最小值、最大值
        lb_, ub_ = self.bounds__[:, 0], self.bounds__[:, 1]  # 种群D个维度的下界、上界
        N, D = X__.shape
        X__ = np.empty(X__.shape)  # 初始化
        # 计算左右未探索区的宽度
        leftWidths_ = np.maximum(Xmin_ - lb_, 0)
        rightWidths_ = np.maximum(ub_ - Xmax_, 0)
        totalWidths_ = leftWidths_ + rightWidths_
        for d in range(D):
            if totalWidths_[d]<1e-10:
                X__[:, d] = np.random.uniform(lb_[d], ub_[d], N)
            else:
                # 按宽度比例分配采样数量
                Nleft = int(N*leftWidths_[d]/totalWidths_[d])
                Nright = N - Nleft
                # 在左右区域分别采样
                Xleft_ = np.random.uniform(lb_[d], Xmin_[d], Nleft)
                Xright_ = np.random.uniform(Xmax_[d], ub_[d], Nright)
                X__[:, d] = np.concatenate([Xleft_, Xright_])
                np.random.shuffle(X__[:, d])  # 打乱顺序
        return X__


from .StateTransitionAlgorithm import STA
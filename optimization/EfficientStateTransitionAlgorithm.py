#%%
from math import ceil, floor

from numpy import ndarray, arange, zeros, ones, empty, sqrt, argmin, where, minimum, concatenate
from numpy.random import random, randint, randn, uniform

from optimization import Optimizer

# References:
# Xiaojun Zhou, et. al. Efficient State Transition Algorithm With Guaranteed Optimality, 2026. https://doi.org/10.1109/TSMC.2026.3688256
# https://github.com/tiezhongyu2005/ESTA
# 周晓君, 等. 智能优化状态转移算法. 科学出版社, 2022.


class ESTA(Optimizer):
    """高效状态转移算法 Efficient State Transition Algorithm"""
    def __init__(self,
            SE: int = 1,           # 采样强度
            β: float = 2.,         # 固定平移因子
            δ: float = 2.,         # 固定轴向因子
            ε: float = 1e-8,       # 解的精度
            pRest: float = 0.9,    # 恢复概率
            pRisk: float = 0.3,    # 冒险概率
            archiveSize: int = 100, # 各个体历史最优档案的容量
            **kwargs):
        Optimizer.__init__(self, **kwargs)
        self.NArchive_ = None
        self.XArchive___ = None
        self.SE = SE; assert isinstance(SE, int) and SE>0, '采样强度SE应为正整数'
        self.β = β; assert β>0, '平移因子β应为正数'
        self.δ = δ; assert δ>0, '轴向因子δ应为正数'
        self.ε = ε; assert 0<ε<0.01, '解的精度ε应属于(0, 0.01)'
        self.pRest = pRest; assert 0<pRest<=1, '恢复概率pRest应属于(0, 1]'
        self.pRisk = pRisk; assert 0<=pRisk<1, '冒险概率pRisk应属于[0, 1)'
        self.archiveSize = archiveSize; assert isinstance(archiveSize, int) and archiveSize>0, '历史档案容量archiveSize应为正整数'
        self.α_ = 2*ones(self.N)
        self.γ_ = 2*ones(self.N)

    @Optimizer.timer()
    def minimize(self, X__: ndarray | None = None):
        # 读取方法
        expansion = self.expansion
        rotation = self.rotation
        axesion = self.axesion
        translation = self.translation
        boundaryHandling = self.boundaryHandling
        batchObjective = self.batchObjective

        X__ = self.initialize(X__)  # (N, D) 初始化种群
        y_ = batchObjective(X__)    # (N,) 初始种群目标函数值
        N, D = self.N, self.D
        Xopt__ = X__.copy()  # (N, D) 个体历史最优解
        yopt_  = y_.copy()   # (N,) 个体历史最优目标函数值
        # 各个体独立维护历史最优档案，最新状态位于索引0
        Xarchive___ = empty((N, self.archiveSize, D), dtype=X__.dtype)
        Xarchive___[...] = X__[:, None, :]
        Narchive_ = ones(N, dtype=int)
        self.XArchive___ = Xarchive___
        self.NArchive_ = Narchive_
        # 读取参数
        pRisk = self.pRisk
        pRest = self.pRest
        self.α_ = 2*ones(N)
        self.γ_ = 2*ones(N)

        idxN_ = arange(N)

        for t in range(1, self.T + 1):
            # 迭代
            XoptOld__ = Xopt__.copy()

            for transform, useArchive in (
                    (rotation, False),
                    (expansion, False),
                    (axesion, False),
                    (translation, True),
                    ):
                # 按作者GitHub源码顺序依次应用RT、ET、AT、TT；预测式平移变换每轮独立执行
                if useArchive:
                    Xnew___ = transform(X__, Xarchive___, Narchive_)
                else:
                    Xnew___ = transform(X__)
                Xnew___ = boundaryHandling(Xnew___, X__[:, None, :])  # (N, K, D) 边界处理
                K = Xnew___.shape[1]                    # 每个体本次变换生成的候选解数量
                Xnew__ = Xnew___.reshape(N*K, D)        # (N*K, D)
                ynew_ = batchObjective(Xnew__)          # (N*K,) 候选解的目标函数值
                ynew__ = ynew_.reshape(N, K)            # (N, K)
                idxynewBest_ = argmin(ynew__, axis=1)    # (N,) 索引：N个体的最优候选解
                ynewBest_ = ynew__[idxN_, idxynewBest_]  # (N,) N个体的最优候选解目标函数值
                XnewBest__ = Xnew___[idxN_, idxynewBest_, :]  # (N, D) N个体的最优候选解

                # 更新N个体的历史最优解
                logicOpt_ = ynewBest_ < yopt_  # (N,) 索引：最优候选解小于历史最优解的个体
                Xopt__[logicOpt_] = XnewBest__[logicOpt_]
                yopt_[logicOpt_] = ynewBest_[logicOpt_]
                if logicOpt_.any():
                    self.updateArchive(Xopt__, logicOpt_, Xarchive___, Narchive_)

                logicN_ = ynewBest_ < y_  # (N,) 索引：最优候选解小于当前解的个体
                Nupdated = logicN_.sum()  # 需更新当前解的个体数
                if Nupdated:
                    # 若存在个体找到更优的解，更新对应个体的当前解
                    X__[logicN_] = XnewBest__[logicN_]
                    y_[logicN_] = ynewBest_[logicN_]

                Nnonupdated = N - Nupdated  # 不更新的个体数
                if Nnonupdated:
                    # 以概率pRisk接受一个较差解
                    mask_accept_ = zeros(N, dtype=bool)
                    mask_accept_[~logicN_] = random(Nnonupdated) < pRisk
                    X__[mask_accept_] = XnewBest__[mask_accept_]
                    y_[mask_accept_] = ynewBest_[mask_accept_]
                    # print(f'迭代{t}/{self.T}以概率{self.pRisk}接受一个较差解')

            # 以概率pRest恢复历史最优解
            logic_ = random(N)<pRest
            X__[logic_] = Xopt__[logic_]
            y_[logic_] = yopt_[logic_]
            # print(f'迭代{t}/{self.T}以概率{self.pRest}恢复历史最优解')

            self.updateParameter(Xopt__, XoptOld__)
            self.record(Xopt__, yopt_, t)  # 记录
        return Xopt__, yopt_

    def updateParameter(self,
                        Xopt__,     # (N, D) 当前一轮的个体历史最优解
                        XoptOld__,  # (N, D) 上一轮的个体历史最优解
                        ):
        """根据相邻两轮历史最优解的最大坐标变化更新各个体的α和γ"""
        Δx_ = abs(Xopt__ - XoptOld__).max(axis=1)
        factor_ = Δx_.copy()
        logic_ = Δx_>1
        factor_[logic_] = minimum(2, Δx_[logic_])
        logic_ = (Δx_>0.01) & (Δx_<=1)
        factor_[logic_] = 1
        factor_[Δx_<=self.ε] = self.ε
        self.α_[...] = factor_
        self.γ_[...] = factor_

    def rotation(self,
                 X__, # (N, D)
                 ):
        """旋转变换"""
        N, D = X__.shape
        SE = self.SE
        eps = 1e-8

        # # 原始旋转变换
        # R____ = uniform(-1, 1, [N, SE, D, D])  # 随机旋转矩阵
        # X_col____ = X__[:, None, :, None]
        # RX____ = matmul(R____, X_col____)
        # RX___ = RX____[..., 0]
        # norm___ = norm(X__, axis=1, keepdims=True)[:, :, None]
        # scale___ = self.α/D/(norm___ + eps)
        # Xbase___ = X__[:, None, :]
        # return Xbase___ + scale___ * RX___

        # 快速旋转变换 《智能优化状态转移算法》4.3.1 式4.8
        R___ = uniform(-1, 1, (N, SE, 1))  # [-1, 1]随机变量
        u___ = uniform(-1, 1, (N, SE, D))  # [-1, 1]随机向量
        norm___ = sqrt((u___ * u___).sum(axis=2, keepdims=True))  # (N, SE, 1)
        u___ /= (norm___ + eps)
        u___ *= R___
        u___ *= self.α_[:, None, None]
        Xbase___ = X__[:, None, :]  # (N, 1, D)
        return Xbase___ + u___      # (N, SE, D)

    def translation(self,
                    X__,          # (N, D) 当前状态
                    Xarchive___,  # (N, archiveSize, D) 各个体历史最优档案
                    Narchive_,    # (N,) 各个体历史最优档案的有效长度
                    ):
        """基于一阶、二阶混合预测模型的平移变换"""
        N, D = X__.shape
        SE1 = floor(self.SE/2)
        SE2 = ceil(self.SE/2)
        Xbase___ = X__[:, None, :]  # (N, 1, D)

        # 一阶预测模型：X + βR(X - Xhistory)
        Xhistory1___ = self.sampleArchive(Xarchive___, Narchive_, SE1)
        Xnew1___ = Xbase___ + self.β*uniform(-1, 1, (N, SE1, 1))*(Xbase___ - Xhistory1___)

        # 二阶预测模型：X + βR(Xhistory1 - Xhistory2)，历史索引独立有放回抽样
        Xhistory21___ = self.sampleArchive(Xarchive___, Narchive_, SE2)
        Xhistory22___ = self.sampleArchive(Xarchive___, Narchive_, SE2)
        Xnew2___ = Xbase___ + self.β*uniform(-1, 1, (N, SE2, 1))*(Xhistory21___ - Xhistory22___)
        return concatenate((Xnew1___, Xnew2___), axis=1)  # (N, SE, D)

    @staticmethod
    def sampleArchive(XArchive___, NArchive_, SE: int):
        """从各个体自己的历史最优档案中独立有放回抽样"""
        N = len(NArchive_)
        idx__ = (random((N, SE))*NArchive_[:, None]).astype(int)
        return XArchive___[arange(N)[:, None], idx__, :]

    def updateArchive(self,
                      Xopt__,        # (N, D) 各个体历史最优解
                      logicN_,       # (N,) 本次找到新历史最优解的个体
                      XArchive___,   # (N, archiveSize, D)
                      NArchive_,     # (N,) 各个体档案有效长度
                      ):
        """把新的历史最优解插到档案首位，并只保留最近archiveSize个"""
        archiveSize = self.archiveSize
        for n in where(logicN_)[0]:
            Xopt_ = Xopt__[n]
            if (Xopt_ == XArchive___[n, 0]).all():
                continue
            Narchive = NArchive_[n]
            Nmove = min(Narchive, archiveSize - 1)
            if Nmove:
                XArchive___[n, 1:Nmove + 1] = XArchive___[n, :Nmove].copy()
            XArchive___[n, 0] = Xopt_
            if Narchive<archiveSize:
                NArchive_[n] = Narchive + 1

    def expansion(self, X__: ndarray,  # (N, D)
                  ):
        """原伸缩变换与零点附近不退化的新伸缩变换"""
        N, D = X__.shape
        hSE = self.SE/2
        SEold = floor(hSE)
        SEnew = ceil(hSE)
        Xbase___ = X__[:, None, :]  # (N, 1, D)
        γ___ = self.γ_[:, None, None]
        Xnew1___ = Xbase___ + γ___*randn(N, SEold, D)*Xbase___  # 原伸缩变换
        Xnew2___ = Xbase___ + γ___*randn(N, SEnew, D)           # 新伸缩变换：uk为D维全1向量
        return concatenate((Xnew1___, Xnew2___), axis=1)  # (N, SE, D)

    def axesion(self, X__: ndarray,  # (N, D)
                ):
        """原轴向变换与零点附近不退化的新轴向变换"""
        N, D = X__.shape
        hSE = self.SE/2
        SEold = floor(hSE)
        SEnew = ceil(hSE)
        Nidx__ = arange(N)[:, None]  # (N, 1)

        # 原轴向变换
        d1__ = randint(0, D, (N, SEold))
        Xnew1___ = empty((N, SEold, D), dtype=X__.dtype)
        Xnew1___[...] = X__[:, None, :]
        SEidx__ = arange(SEold)[None, :]  # (1, SEold)
        Xnew1___[Nidx__, SEidx__, d1__] += (
            self.δ * randn(N, SEold) * Xnew1___[Nidx__, SEidx__, d1__]
        )

        # 新轴向变换：uk为D维全1向量，使选中坐标采用加性扰动
        d2__ = randint(0, D, (N, SEnew))
        Xnew2___ = empty((N, SEnew, D), dtype=X__.dtype)
        Xnew2___[...] = X__[:, None, :]
        SEidx__ = arange(SEnew)[None, :]  # (1, SEnew)
        Xnew2___[Nidx__, SEidx__, d2__] += self.δ*randn(N, SEnew)
        return concatenate((Xnew1___, Xnew2___), axis=1)  # (N, SE, D)

if __name__ == "__main__":
    from BaselineFunctions import BaselineFunctions
    function = BaselineFunctions.F9
    bounds__ = [[-5, 5]]*23

    optimizer = ESTA(
        function=function,
        bounds__=bounds__,
        N=33,
        T=66,
        SE=5,
        n_jobs=1,
        reuse_parallel=False,
    )
    X__, y_ = optimizer.minimize()
    optimizer.plot()
    
    #%%
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # X__ = array([[1., 1], [0.1, 0.1], [0, 0]])
    # # X___ = optimizer.rotation(X__)
    # # X___ = optimizer.expansion(X__)
    # # X___ = optimizer.axesion(X__)
    #
    # XArchive___ = X__[:, None, :]
    # NArchive_ = array([1, 1, 1])
    # X___ = optimizer.translation(X__, XArchive___, NArchive_)
    # n = 2  # 个体
    # plt.plot(X___[n, :, 0], X___[n, :, 1], 'o' )
    # plt.plot(X__[n, 0], X__[n, 1], '^', ms=10)
    # plt.axis('equal')
    # plt.show()

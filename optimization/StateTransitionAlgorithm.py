#%%
from numpy import ndarray, arange, zeros, empty, sqrt, argmin, where
from numpy.linalg import norm
from numpy.random import random, randint, randn, uniform

from optimization import Optimizer

# References:
# Xiaojun Zhou, et. al. State transition algorithm, 2013. https://doi.org/10.3934/jimo.2012.8.1039
# 周晓君, 等. 智能优化状态转移算法. 科学出版社, 2022.


class STA(Optimizer):
    """状态转移算法 State Transition Algorithm"""
    def __init__(self,
            SE: int = 1,           # 采样强度
            αmax: float = 1., αmin: float = 1e-4,  # 旋转因子最大值、最小值
            βmax: float = 1., βmin: float = 1e-4,  # 平移因子最大值、最小值
            γmax: float = 1., γmin: float = 1e-4,  # 伸缩因子最大值、最小值
            δmax: float = 1., δmin: float = 1e-4,  # 轴向因子最大值、最小值
            fc: float = 2.,         # 变换因子的指数衰减底数
            pRest: float = 0.9,     # 恢复概率
            pRisk: float = 0.3,     # 冒险概率
            **kwargs):
        Optimizer.__init__(self, **kwargs)
        self.SE = SE; assert isinstance(SE, int) and SE>0, '采样强度SE应为正整数'
        self.αmax, self.αmin = αmax, αmin; assert αmax>αmin>0, '旋转因子最大值αmax和最小值αmin应为正数，且αmax>αmin'
        self.βmax, self.βmin = βmax, βmin; assert βmax>βmin>0, '平移因子最大值βmax和最小值βmin应为正数，且βmax>βmin'
        self.γmax, self.γmin = γmax, γmin; assert γmax>γmin>0, '伸缩因子最大值γmax和最小值γmin应为正数，且γmax>γmin'
        self.δmax, self.δmin = δmax, δmin; assert δmax>δmin>0, '轴向因子最大值δmax和最小值δmin应为正数，且δmax>δmin'
        self.fc = fc; assert fc>1, '变换因子的指数衰减底数fc应大于1'
        self.pRest = pRest; assert 0<pRest<=1, '恢复概率pRest应属于(0, 1]'
        self.pRisk = pRisk; assert 0<=pRisk<1, '冒险概率pRisk应属于[0, 1)'
        self.α, self.β, self.γ, self.δ = αmax, βmax, γmax, δmax

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
        N, SE, D = self.N, self.SE, self.D
        Xopt__ = X__.copy()  # (N, D) 个体历史最优解
        yopt_  = y_.copy()   # (N,) 个体历史最优目标函数值
        # 读取参数
        pRisk = self.pRisk
        pRest = self.pRest
        self.α, self.β, self.γ, self.δ = self.αmax, self.βmax, self.γmax, self.δmax

        idxN_ = arange(N)

        for t in range(1, self.T + 1):
            # 迭代
            if self.α<self.αmin:
                self.α = self.αmax
            if self.β<self.βmin:
                self.β = self.βmax
            if self.γ<self.γmin:
                self.γ = self.γmax
            if self.δ<self.δmin:
                self.δ = self.δmax

            for transform in (expansion, rotation, axesion):
                # 依次应用状态转移算子
                Xnew___ = transform(X__)                              # (N, SE, D) 采样N个体邻域域内的候选解
                Xnew___ = boundaryHandling(Xnew___, X__[:, None, :])  # (N, SE, D) 边界处理
                Xnew__ = Xnew___.reshape(N*SE, D)      # (N*SE, D)
                ynew_ = batchObjective(Xnew__)         # (N*SE,) 候选解的目标函数值
                ynew__ = ynew_.reshape(N, SE)          # (N, SE)
                idxynewBest_ = argmin(ynew__, axis=1)    # (N,) 索引：N个体的最优候选解
                ynewBest_ = ynew__[idxN_, idxynewBest_]  # (N,) N个体的最优候选解目标函数值
                XnewBest__ = Xnew___[idxN_, idxynewBest_, :]  # (N, D) N个体的最优候选解

                # 更新N个体的历史最优解
                logicN_ = ynewBest_ < yopt_  # (N,) 索引：最优候选解小于历史最优解的个体
                Xopt__[logicN_] = XnewBest__[logicN_]
                yopt_[logicN_] = ynewBest_[logicN_]

                logicN_ = ynewBest_ < y_  # (N,) 索引：最优候选解小于当前解的个体
                Nupdated = logicN_.sum()  # 需更新当前解的个体数
                if Nupdated:
                    # 若存在个体找到更优的解，更新对应个体的当前解，并执行平移变换
                    Xold__ = X__[logicN_]  # (Nupdated, D) 记录：当前解旧值
                    # 更新个体的当前解
                    Xupdated__ = XnewBest__[logicN_]
                    X__[logicN_] = Xupdated__
                    y_[logicN_] = ynewBest_[logicN_]
                    # 执行平移变换
                    Xnew___ = translation(Xupdated__, Xold__)  # (Nupdated, SE, D)
                    Xnew___ = boundaryHandling(Xnew___, Xupdated__[:, None, :])  # (Nupdated, SE, D) 边界处理
                    Xnew__ = Xnew___.reshape(Nupdated*SE, D)   # (Nupdated*SE, D)
                    ynew_ = batchObjective(Xnew__)             # (Nupdated*SE,) 候选解的目标函数值
                    ynew__ = ynew_.reshape(Nupdated, SE)
                    idxynewBest_ = argmin(ynew__, axis=1)           # (Nupdated,) 索引：Nupdated个体的最优候选解
                    idxNupdated_ = arange(Nupdated)
                    ymin_ = ynew__[idxNupdated_, idxynewBest_]        # (Nupdated,) Nupdated个体的最优候选解目标函数值
                    Xymin__ = Xnew___[idxNupdated_, idxynewBest_, :]  # (Nupdated, D) Nupdated个体的最优候选解

                    # 更新Nupdated个体的历史最优解
                    idx_ = where(logicN_)[0]              # (Nupdated,)
                    logicNupdated_ = ymin_ < yopt_[idx_]  # (Nupdated,)
                    sub_idx_ = idx_[logicNupdated_]
                    Xopt__[sub_idx_] = Xymin__[logicNupdated_]
                    yopt_[sub_idx_]  = ymin_[logicNupdated_]

                    # 更新Nupdated个体的当前解
                    logicNupdated_ = ymin_< y_[idx_]
                    sub_idx_ = idx_[logicNupdated_]
                    X__[sub_idx_] = Xymin__[logicNupdated_]
                    y_[sub_idx_] = ymin_[logicNupdated_]

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

            self.α /= self.fc
            self.β /= self.fc
            self.γ /= self.fc
            self.δ /= self.fc
            self.record(Xopt__, yopt_, t)  # 记录
        return Xopt__, yopt_

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

        # 快速旋转变换 4.3.1 式4.8
        R___ = uniform(-1, 1, (N, SE, 1))  # [-1, 1]随机变量
        u___ = uniform(-1, 1, (N, SE, D))  # [-1, 1]随机向量
        norm___ = sqrt((u___ * u___).sum(axis=2, keepdims=True))  # (N, SE, 1)
        u___ /= (norm___ + eps)
        u___ *= R___
        u___ *= self.α
        Xbase___ = X__[:, None, :]  # (N, 1, D)
        return Xbase___ + u___      # (N, SE, D)

    def translation(self,
                    X__,     # (N, D)
                    Xold__,  # (N, D)
                    ):
        """平移变换"""
        N, D = X__.shape
        SE = self.SE
        eps = 1e-8
        ΔX__ = X__ - Xold__  # (N, D)
        norm__ = norm(ΔX__, axis=1, keepdims=True)  # (N, 1)
        direction__ = ΔX__/(norm__ + eps)                     # (N, D)
        Xbase___ = X__[:, None, :]                            # (N, 1, D)
        direction___ = direction__[:, None, :]                # (N, 1, D)
        return Xbase___ + self.β * random((N, SE, 1))*direction___  # (N, SE, D)

    def expansion(self, X__: ndarray,  # (N, D)
                  ):
        """伸缩变换"""
        N, D = X__.shape
        SE = self.SE
        R___ = randn(N, SE, D)  # (N, SE, D) 高斯扰动
        Xbase___ = X__[:, None, :]        # (N, 1, D)
        return Xbase___ + self.γ * R___ * Xbase___  # (N, SE, D)

    def axesion(self, X__: ndarray,  # (N, D)
                ):
        """轴向变换"""
        N, D = X__.shape
        SE = self.SE
        d__ = randint(0, D, (N, SE)) # (N, SE) 每个个体选择一个维度
        Xnew___ = empty((N, SE, D), dtype=X__.dtype)
        Xnew___[...] = X__[:, None, :]
        # 构造索引
        Nidx__ = arange(N)[:, None]    # (N, 1)
        SEidx__ = arange(SE)[None, :]  # (1, SE)
        Xnew___[Nidx__, SEidx__, d__] += (
            self.δ * randn(N, SE) * Xnew___[Nidx__, SEidx__, d__]
        )
        return Xnew___  # (N, SE, D)

if __name__ == "__main__":
    from BaselineFunctions import BaselineFunctions
    function = BaselineFunctions.F9
    bounds__ = [[-5, 5]]*2

    optimizer = STA(
        function=function,
        bounds__=bounds__,
        N=200,
        T=1000,
        SE=5,
        n_jobs=-1,
        reuse_parallel=False,
    )
    X__, y_ = optimizer.minimize()
    optimizer.plot()

    #%%
    # import matplotlib.pyplot as plt
    # plt.close('all')
    # X__ = array([[1., 1], [0.1, 0.1], [0, 0]])
    # Xold__ = array([[0.5, 0.5], [-0.1, 0], [0.4, 1]])
    # # X___ = optimizer.rotation(X__)
    # # X___ = optimizer.expansion(X__)
    # # X___ = optimizer.axesion(X__)
    #
    # X___ = optimizer.translation(X__, Xold__)
    # n = 2  # 个体
    # plt.plot(X___[n, :, 0], X___[n, :, 1], 'o' )
    # plt.plot(X__[n, 0], X__[n, 1], '^', ms=10)
    # plt.axis('equal')
    # plt.show()


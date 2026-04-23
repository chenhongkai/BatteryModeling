#%%
import numpy as np
import matplotlib.pyplot as plt


def u(x, a, k, m):
    return k*(abs(x) - a)**m if abs(x)>a else 0

class BaselineFunctions:
    def __init__(self,):
        pass

    @staticmethod
    def F1(x_: np.ndarray):  # [-100, 100]
        return (x_**2).sum()

    @staticmethod
    def F2(x_: np.ndarray):  # [-10, 10]
        x_ = abs(x_)
        return x_.sum() + x_.prod()

    @staticmethod
    def F3(x_: np.ndarray):  # [-100, 100]
        y = 0
        for d, _ in enumerate(x_):
            y += x_[:(d + 1)].sum()**2
        return y

    @staticmethod
    def F4(x_: np.ndarray):  # [-100, 100]
        return max(abs(x_))

    @staticmethod
    def F5(x_: np.ndarray):  # [-30, 30]
        y = 0
        for d, _ in enumerate(x_[:-1]):
            y += 100*(x_[d+1] - x_[d]**2)**2 + (x_[d] - 1)**2
        return y

    @staticmethod
    def F6(x_: np.ndarray):  # [-100, 100]
        return ((x_ + 0.5)**2).sum()

    @staticmethod
    def F7(x_: np.ndarray):  # [-1.28, 1.28]
        return (np.arange(1, len(x_) + 1) * x_**4).sum() + np.random.rand()

    @staticmethod
    def F8(x_: np.ndarray):  # [-500, 500]
        return (-x_ * np.sin(abs(x_)**0.5)).sum()

    @staticmethod
    def F9(x_: np.ndarray):  # [-5.12, 5.12]
        return ((x_ - .5)**2 - 10*np.cos(2*np.pi*(x_ - 0.5)) + 10).sum()

    @staticmethod
    def F10(x_: np.ndarray):  # [-32, 32]
        return -20*np.exp(-0.2 * ((x_**2).mean())**0.5) - np.exp(np.cos(2*np.pi*x_).mean()) + 20 + np.exp(1)

    @staticmethod
    def F11(x_: np.ndarray):  # [-600, 600]
        return (x_**2).sum()/4000 + np.cos(x_/np.arange(1, len(x_) + 1)**0.5).prod() + 1


def plot3D(function,        # 函数
           bounds__,        # 每个维度的上下界
           resolution=100,  # 单个维度网格数
           ):
    x0_ = np.linspace(bounds__[0][0], bounds__[0][1], resolution)
    x1_ = np.linspace(bounds__[1][0], bounds__[1][1], resolution)
    grid__ = np.zeros([x0_.size, x1_.size])  # 网格
    for i, x0 in enumerate(x0_):
        for j, x1 in enumerate(x1_):
            grid__[i, j] = function(np.array([x0, x1]))
    ax = plt.axes(projection='3d')
    x0__, x1__ = np.meshgrid(x0_, x1_)
    ax.plot_surface(x0__, x1__, grid__, cmap='rainbow')   # 曲面
    ax.contour(x0__, x1__, grid__, offset=0, cmap='jet')  # 等高线，要设置offset，为z轴最小值
    plt.show()

if __name__ == '__main__':
    bounds__ = [[-5.12, 5.12],
                [-5.12, 5.12]]
    plot3D(BaselineFunctions.F9, bounds__)
import pathlib
from typing import Sequence
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from P2Dmodel.tools import set_matplotlib


set_matplotlib()
path = pathlib.Path(__file__).parent  # 获得当前模块所在路径


class OCPbase(ABC):
    """开路电位抽象类"""

    path_OCP_from_COMSOL = path.joinpath('OCP_from_COMSOL.xlsx')
    path_OCP_from_LiionDB = path.joinpath('OCP_from_LiionDB.xlsx')
    kwargs_interp1d = {'bounds_error': False,
                       'fill_value': 'extrapolate'}

    @property
    def sources_(self) -> tuple[str]:
        """tuple：数据源名称字符串"""
        return tuple([method for method in dir(self) if self.__class__.__name__ in method])

    @property
    def defaultSource(self) -> str:
        """默认COMSOL数据源"""
        return self.__class__.__name__ + '_COMSOL'

    def UOCP(self,
             θs_: Sequence[float],      # 嵌锂状态序列
             source: str | None= None,  # 数据源
             ) -> np.ndarray:
        """输入电极嵌锂状态θ_，输出开路电位"""
        if source is None:
            source = self.defaultSource
        UOCP_ = getattr(self, source)(θs_)  # 开路电位序列
        return UOCP_

    def plot(self,
             θs_: Sequence[float] = np.arange(0, 1 + 1e-6, 0.001),
             sources_: Sequence[str] | None = None,
             ):
        """可视化实验数据：不同数据源的UOCP-θs曲线"""
        if sources_ is None:
            sources_ = self.sources_

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        for source in sources_:
            assert source in self.sources_
            ax.plot(θs_, self.UOCP(θs_, source), label=source)

        ax.set_xlabel(r'Degree-of-lithiation ${\it θ}_{s}$ [–]')
        ax.set_ylabel(r'Open-circuit potential ${\it U}_{OCP}$ [V]')

        ax.set_ylim(0, 5)
        ax.set_yticks(np.arange(0, 5 + 1e-6, 0.5))

        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1 + 1e-6, 0.1))

        ax.legend(ncol=2, loc='best')
        ax.grid(axis='y', ls='--', color=[.5]*3)
        plt.show()
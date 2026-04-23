import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from P2Dmodel import set_matplotlib; set_matplotlib()


path = pathlib.Path(__file__).parent  # 获得当前模块所在路径

class OpenCircuitPotential:
    """开路电位类的父类"""

    path_OCP_from_COMSOL  = path.joinpath('OCP_from_COMSOL.xlsx')
    path_OCP_from_LiionDB = path.joinpath('OCP_from_LiionDB.xlsx')

    @property
    def sources_(self) -> tuple[str]:
        """tuple：数据源名称字符串"""
        return tuple([method for method in dir(self) if self.__class__.__name__ in method])

    @property
    def defaultSource(self) -> str:
        """默认COMSOL数据源"""
        return self.__class__.__name__ + '_COMSOL'

    def OCP(self,
            θ_: Sequence[float],  # 嵌锂状态序列
            source=None,          # 数据源
            ) -> np.ndarray:
        """输入电极嵌锂状态θ_，输出开路电位"""
        if source is None:
            source = self.defaultSource
        OCP_ = getattr(self, source)(θ_)  # 开路电位序列
        return OCP_

    def plot(self,
             θ_: Sequence[float] | float | int = np.arange(0, 1 + 1e-6, 0.001),
             sources_: Sequence[str] | None = None,
             ):
        """可视化实验数据：不同数据源的开路电位-电极嵌锂状态曲线"""
        if sources_ is None:
            sources_ = self.sources_

        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(111)
        for source in sources_:
            assert source in self.sources_
            ax.plot(θ_, self.OCP(θ_, source), label=source)
        ax.set_xlabel(r'Degree of lithiation ${\it θ}$ [–]')
        ax.set_ylabel(r'Open-circuit potential ${\it U}_{OCP}$ [V]')
        ax.set_ylim(0, 5)
        ax.set_yticks(np.arange(0, 5 + 1e-6, 0.5))
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1 + 1e-6, 0.1))
        ax.legend(ncol=2, loc='best')
        ax.grid(axis='y', ls='--', color=[.5]*3)
        plt.show()

from .Graphite import Graphite
from .LFP import LFP
from .NCA import NCA
from .NMC import NMC
from .NMC111 import NMC111
from .NMC532 import NMC532
from .NMC622 import NMC622
from .NMC811 import NMC811
from .LMO import LMO
from .PseudoNegativeElectrode import PseudoNegativeElectrode
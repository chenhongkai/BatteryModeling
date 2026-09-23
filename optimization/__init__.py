#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
matplotlib.use('Qt5Agg')  # TkAgg/Qt5Agg
import matplotlib.pyplot as plt
fontname = 'Times New Roman'  # 字体
plt.rcParams['font.serif'] = [fontname]       # 衬线字体
plt.rcParams['font.sans-serif'] = [fontname]  # 无衬线字体
plt.rcParams['font.size'] = 18  # 字号
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['axes.unicode_minus'] = False               # 正常显示负号
plt.rcParams['mathtext.default'] = 'regular'             # 默认样式：正体、不加粗
plt.rcParams['mathtext.rm'] = 'STIXGeneral:regular'      # 正体、不加粗
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'       # 斜体、不加粗
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'  # 斜体、加粗

# 优化算法基类
from .Optimizer import Optimizer

from .StateTransitionAlgorithm import STA
from .EfficientStateTransitionAlgorithm import ESTA

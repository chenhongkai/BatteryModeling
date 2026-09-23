# 锂离子电池时频联合电化学模型 (Joint time-frequency electrochemical models of lithium-ion batteries)

`P2Dmodel` 是一个用于锂离子电池电化学建模与仿真的研究型 Python 模块，包含经典 DFN/P2D 模型和集总参数 P2D 模型。它既能计算恒流工况下的时域响应，也能在充电过程中指定多个时刻计算动态电化学阻抗谱（DEIS）。

该模块主要服务于：

- 锂离子电池恒流充放电仿真；
- 电压、浓度、电势、电流密度和温度等状态量计算；
- 静态 EIS 与充电过程中的 DEIS 计算；
- 电化学参数辨识；
- 集总参数模型与经典 DFN/P2D 模型的对比研究。

## 主要功能

### 时域仿真

通过 `cell.CC()` 进行恒流仿真。模型使用 Newton 迭代求解离散控制方程，并在 Newton 迭代失败时自动缩小时间步长。

项目中的电流符号约定为：

- 充电电流为负；
- 放电电流为正。

### EIS 和 DEIS

- `cell.EIS()`：计算模型当前状态下的电化学阻抗谱；
- `cell.CC(..., tEIS_=[...])`：在恒流过程的指定时刻计算阻抗谱，得到随电池状态变化的 DEIS。

每个 DEIS 时刻都会在当前电化学状态附近进行频域线性化，并计算频率数组 `cell.f_` 对应的复阻抗 `Z_`。

### 电化学-热耦合

模型支持集总热平衡方程、Arrhenius 温度修正以及热阻、热容参数。启用热模型后，可计算电池温度和产热量随时间的变化。

### 参数化与参数辨识支持

`P2Dmodel.tools` 提供集总参数的范围、标称值、归一化/反归一化和插值工具。`LPJTFP2D` 是参数辨识中主要使用的快速模型。

## 模型与目录结构

```text
Battery/
├─ P2Dmodel/
│  ├─ P2Dbase.py       # 公共基类：CC、EIS、数据记录、绘图和数值求解工具
│  ├─ LPJTFP2D.py      # 集总参数时频联合 P2D 模型
│  ├─ DFNJTFP2D.py     # 经典物理参数 DFN/P2D 时频联合模型
│  ├─ ELPJTFP2D.py     # 增强集总参数模型
│  ├─ tools.py         # 参数集合、插值、矩阵和绘图辅助工具
│  ├─ OCP/             # 石墨、LFP、NMC、NCA、LMO 等材料的 OCP 数据和函数
│  └─ __init__.py      # P2Dmodel 的公开导出接口
└─ ParameterIdentification/  # SEIS/DEIS 参数辨识
```

主要模型的适用场景如下：

| 类 | 用途 |
|---|---|
| `LPJTFP2D` | 集总参数时域/频域联合仿真；适合参数辨识和大量重复计算 |
| `DFNJTFP2D` | 使用原始物理参数的经典 DFN/P2D 仿真 |
| `ELPJTFP2D` | 在 `LPJTFP2D` 基础上扩展电解液相关参数化方式 |

## Python 版本与依赖

建议使用 **Python 3.12 或更高版本**。当前开发环境使用 Python 3.13。

核心依赖：

```text
numpy
scipy
numba
pandas
matplotlib
openpyxl
```

可按需安装：

```powershell
pip install numpy scipy numba pandas matplotlib openpyxl
```

可选依赖：

- `PyQt5`：使用 Matplotlib 的 Qt 图形窗口时需要；
- `joblib`：参数辨识脚本中的并行计算会使用。

仓库目前没有统一的依赖锁定文件。建议从 `Battery` 根目录运行脚本，以保证 `P2Dmodel` 可以被正常导入。首次运行涉及 Numba 的函数时会进行编译，因此通常比后续运行慢。

## 最小运行示例

下面的示例使用较小网格完成 60 s 恒流充电，并在 `0 s`、`30 s` 和 `60 s` 计算 DEIS：

```python
import numpy as np

from P2Dmodel import LPJTFP2D, LumpedParameters


Qnom = 20  # 标称容量 [Ah]
parameters = LumpedParameters(Qnom=Qnom).nominalSet_

cell = LPJTFP2D(
    **parameters,
    f_=np.logspace(3, 0, 13),  # 1000 Hz 至 1 Hz，共13个频点
    Nneg=5,
    Nsep=5,
    Npos=5,
    Nr=5,
    Δt=5,
    verbose=False,
)

# 充电电流为负：这里进行 0.5C 恒流充电
cell.CC(
    I=-0.5 * Qnom,
    duration=60,
    tEIS_=[0, 30, 60],
)

t_ = np.asarray(cell.data['t'])
U_ = np.asarray(cell.data['U'])
tEIS_ = np.asarray(cell.data['tEIS'])
Z_ = np.asarray(cell.data['Z_'])

print('末时刻:', t_[-1], 's')
print('末时刻电压:', U_[-1], 'V')
print('DEIS时刻:', tEIS_, 's')
print('阻抗数组形状:', Z_.shape)  # (3, 13)
print('末时刻阻抗实部:', Z_[-1].real)
print('末时刻阻抗虚部:', Z_[-1].imag)

# 需要图形界面；无显示环境中可注释掉
cell.plot_UI()
cell.plot_Nyquist()
```

网格数量和频点数量仅为快速示例设置。正式计算前应根据精度需求进行网格无关性和频率范围检查。

## 主要输出

模型结果保存在 `cell.data` 中。最常用的输出包括：

| 数据 | 含义 |
|---|---|
| `cell.data['t']` | 时域计算时刻 `[s]` |
| `cell.data['U']` | 电池端电压 `[V]` |
| `cell.data['tEIS']` | 进行 EIS/DEIS 计算的时刻 `[s]` |
| `cell.data['Z_']` | 各 EIS 时刻、各频率对应的复阻抗 `[Ω]` |
| `cell.f_` | EIS 频率 `[Hz]` |
| `cell.data['SOC']` | 荷电状态，完整记录模式下提供 |
| `cell.data['T']` | 电池温度，完整记录模式下提供 |
| `cell.data['Qgen']` | 电池产热量，完整记录模式下提供 |

在完整记录模式下，还可获得：

- 固相表面浓度和固相内部浓度；
- 电解液浓度；
- 固相电势和液相电势；
- 主反应电流密度和双电层电流密度；
- 交换电流密度与主反应过电位；
- 正负极分解阻抗及频域场变量。

常用绘图接口包括：

```python
cell.plot_UI()       # 电流和端电压
cell.plot_Nyquist()  # Nyquist 图
cell.plot_Z()        # 阻抗随频率变化
```

`Z_` 使用复数保存，其中 `Z_.real` 为阻抗实部，`Z_.imag` 为阻抗虚部。Nyquist 图通常绘制 `Z'` 与 `-Z''`。

## 使用注意事项

- 这是研究型代码库，不是已经发布到 PyPI 的独立软件包；部分脚本依赖本机数据文件和工作目录。
- 修改模型控制方程时，应同时核对时域 Newton 方程、频域 EIS 方程和 `checkEIS()` 检验结果。

## 联系方式

电子邮箱：[gdchenhongkai@outlook.com]

如需交流模型使用、复现或开发问题，请保持礼貌沟通。用户文档仍在持续完善中。

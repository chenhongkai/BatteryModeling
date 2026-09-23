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

cell.plot_UI()       # 电流和端电压
cell.plot_Nyquist()  # Nyquist 图
cell.plot_Z()        # 阻抗随频率变化

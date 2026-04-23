from P2Dmodel import LPJTFP2D


class ELPJTFP2D(LPJTFP2D):
    """锂离子电池强化集总参数时频联合准二维模型（Enhanced Lumped-Parameter Joint Time-Frequency Pseudo-two-Dimension model）"""
    def __init__(self,
            Kκneg: float | int = .5,   # 负极与隔膜集总电解液离子电导率之比 [–]
            Kκpos: float | int = .5,   # 正极与隔膜集总电解液离子电导率之比 [–]
            κsep: float = 854.24,      # 隔膜集总电解液离子电导率 [S]
            Kqeneg: float | int = 1.,  # 负极与隔膜电解液锂离子电荷量之比 [–]
            Kqepos: float | int = 1.,  # 正极与隔膜电解液锂离子电荷量之比 [–]
            qesep: float = 16229.,     # 隔膜电解液锂离子电荷量 [C]
            **kwargs):
        self.Kκneg = Kκneg; assert Kκneg>0, f'负极与隔膜集总电解液离子电导率之比{Kκneg = }，应大于0'
        self.Kκpos = Kκpos; assert Kκpos>0, f'正极与隔膜集总电解液离子电导率之比{Kκpos = }，应大于0'
        self.Kqeneg = Kqeneg; assert Kqeneg>0, f'负极与隔膜电解液锂离子电荷量之比{Kqeneg = }，应大于0'
        self.Kqepos = Kqepos; assert Kqepos>0, f'正极与隔膜电解液锂离子电荷量之比{Kqepos = }，应大于0'
        κneg = Kκneg*κsep     # 负极集总电解液离子电导率 [S]
        κpos = Kκpos*κsep     # 正极集总电解液离子电导率 [S]
        qeneg = Kqeneg*qesep  # 负极电解液锂离子电荷量 [C]
        qepos = Kqepos*qesep  # 正极电解液锂离子电荷量 [C]
        LPJTFP2D.__init__(self,
            κneg=κneg, κpos=κpos, κsep=κsep,
            qeneg=qeneg, qepos=qepos, qesep=qesep,
            **kwargs)


if __name__=='__main__':
    import numpy as np
    cell = ELPJTFP2D(
        SOC0=0.1,
        f_=np.logspace(4, -1, 26),
        )

    cell.CC(-20, 400).EIS()
    cell.CC(20, 400).EIS()

    cell.count_lithium()

    '''
    cell.plot_Z()
    cell.plot_Nyquist()
    cell.plot_REφeIMφe()
    cell.plot_REφsIMφs()
    cell.plot_REceIMce()
    cell.plot_REcsSurfIMcsSurf()
    cell.plot_REjIMj()
    cell.plot_REjDLIMjDL()
    cell.plot_REηIMη()
    cell.plot_REi0IMi0()
    '''

#%%
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class PseudoNegativeElectrode(OCPbase):

    def __init__(self):
        # COMSOL 高电位负极
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='PseudoNegativeElectrode')
        self.PseudoNegativeElectrode_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

if __name__=='__main__':
    import numpy as np
    neg = PseudoNegativeElectrode()
    neg.plot(np.arange(0.001, 1 + 1e-6, 0.001))


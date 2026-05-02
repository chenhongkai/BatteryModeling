#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class LMO(OCPbase):
    def __init__(self):

        # COMSOL LMO
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='LMO')
        self.LMO_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

if __name__=='__main__':
    electrode = LMO()
    electrode.plot(np.arange(0.001, 1 + 1e-6, 0.001))

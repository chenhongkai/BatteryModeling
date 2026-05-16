#%%
import numpy as np
import pandas as pd

from P2Dmodel.OCP.OCPbase import OCPbase


class LMO(OCPbase):
    def __init__(self):

        # COMSOL LMO
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='LMO')
        self.LMO_COMSOL = OCPbase.interp1d(table['θs'], table['UOCP'])

        del table

if __name__=='__main__':
    electrode = LMO()
    electrode.plot(np.arange(0.001, 1 + 1e-6, 0.001))

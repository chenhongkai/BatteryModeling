#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP import OpenCircuitPotential


class LMO(OpenCircuitPotential):
    def __init__(self):

        # COMSOL LMO
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_COMSOL, sheet_name='LMO')
        self.LMO_COMSOL = interp1d(table['θ'], table['OCP'],
            bounds_error=False,
            fill_value='extrapolate')

        del table

if __name__=='__main__':
    electrode = LMO()
    electrode.plot(np.arange(0.001, 1 + 1e-6, 0.001))

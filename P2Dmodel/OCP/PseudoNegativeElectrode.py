#%%
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP import OpenCircuitPotential


class PseudoNegativeElectrode(OpenCircuitPotential):
    def __init__(self):

        # COMSOL 高电位负极
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_COMSOL, sheet_name='PseudoNegativeElectrode')
        self.PseudoNegativeElectrode_COMSOL = interp1d(table['θ'], table['OCP'],
            bounds_error=False,
            fill_value='extrapolate')

        del table

if __name__=='__main__':
    import numpy as np
    neg = PseudoNegativeElectrode()
    neg.plot(np.arange(0.001, 1 + 1e-6, 0.001))


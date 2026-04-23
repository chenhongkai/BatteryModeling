#%%
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP import OpenCircuitPotential


class NMC111(OpenCircuitPotential):
    def __init__(self):

        # COMSOL NMC111
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_COMSOL, sheet_name='NMC111')
        self.NMC111_COMSOL = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # Lithiation (discharge) OCV measured at C/10 5 percent SOC steps, with 5 hour rest and with qOCV method at C/100
        # https://doi.org/10.1149%2F2.0321816jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC111_52_Schmalstieg2018')
        self.NMC111_52_Schmalstieg2018 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # C/25 pseudo OCV with MX3 cathode
        # https://doi.org/10.1149%2F2.062204jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC111_563_Wu2012')
        self.NMC111_563_Wu2012 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # 283 K
        # https://doi.org/10.1149%2F2.0661913jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC111_678_Ko2019')
        self.NMC111_678_Ko2019 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # 298 K
        # https://doi.org/10.1149%2F2.0661913jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC111_679_Ko2019')
        self.NMC111_679_Ko2019 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # 313 K
        # https://doi.org/10.1149%2F2.0661913jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC111_681_Ko2019')
        self.NMC111_681_Ko2019 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        del table

if __name__=='__main__':
    pos = NMC111()
    pos.plot()
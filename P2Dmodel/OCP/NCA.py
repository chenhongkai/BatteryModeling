#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP import OpenCircuitPotential


class NCA(OpenCircuitPotential):
    def __init__(self):
        OpenCircuitPotential.__init__(self)

        # COMSOL NCA
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_COMSOL, sheet_name='NCA')
        self.NCA_COMSOL = interp1d(table['θ'], table['OCP'],
            bounds_error=False,
            fill_value='extrapolate')

        # C/100 pseudo OCV, array from digitized plot.
        # https://doi.org/10.1149%2F1.3129656
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NCA_460_Albertus2009')
        self.NCA_460_Albertus2009 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # Model fitted data from Fig 6 (e)
        # https://doi.org/10.1016%2Fj.electacta.2007.09.018
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NCA_593_Abraham2008')
        self.NCA_593_Abraham2008 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # C/50 rate
        # https://doi.org/10.18154/RWTH-2019-00249
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NCA_716_Hust2019')
        self.NCA_716_Hust2019 = interp1d(table['θ'], table['OCP'],
            bounds_error=False,
            fill_value='extrapolate')

        del table

    @staticmethod
    def NCA_422_Kim2011(θ_):
        # Fit provided in manuscript, C/100 quasiOCV fitted from ref24 Albertus2009.
        # https://doi.org/10.1149%2F1.3597614
        OCP_ = 1.638*θ_**10 - 2.222*θ_**9 + 15.056*θ_**8 - 23.488*θ_**7 + 81.246*θ_**6 - 344.566*θ_**5\
            + 621.3475*θ_**4 - 554.774*θ_**3 + 264.427*θ_**2 - 66.3691*θ_\
            + 11.8058 - 0.61386*np.exp(5.8201*θ_**136.4)
        return OCP_

    @staticmethod
    def NCA_601_Dees2008(θ_):
        # Model fitted data from previous paper Abraham2008
        # [295, 298]
        # https://doi.org/10.1149%2F1.2939211
        OCP_ = 0.7869*θ_**2 - 2.0565*θ_ + 4.8091
        return OCP_


if __name__=='__main__':
    electrode = NCA()
    electrode.plot()




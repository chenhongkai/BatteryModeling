#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP import OpenCircuitPotential


class NMC(OpenCircuitPotential):
    def __init__(self):

        # C/40 pseudo OCV, array from manuscript appendices.
        # https://doi.org/10.3390%2Fbatteries5030062
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC_531_Liebig2019')
        self.NMC_531_Liebig2019 = interp1d(table['θ'], table['OCP'],
            bounds_error=False,
            fill_value='extrapolate')

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC_668_Birkl2015')
        self.NMC_668_Birkl2015 = interp1d(table['θ'], table['OCP'],
            bounds_error=False,
            fill_value='extrapolate')

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC_669_Birkl2015')
        self.NMC_669_Birkl2015 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC_670_Birkl2015')
        NMC_670_Birkl2015 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # Dufour2019 thesis fig 2.8, delithiation at C/10 rate
        # https://doi.org/10.1016%2Fj.electacta.2018.03.196
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC_853_Dufour2018')
        self.NMC_853_Dufour2018 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        # Dufour2019 thesis fig 2.8, lithiation at C/10 rate
        # https://doi.org/10.1016%2Fj.electacta.2018.03.196
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_LiionDB, sheet_name='NMC_854_Dufour2018')
        self.NMC_854_Dufour2018 = interp1d(table['θ'], table['OCP'],
            bounds_error=False, fill_value='extrapolate')

        del table

    @staticmethod
    def NMC_400_Smith2006(θ_):
        # Fit provided in manuscript, inverse optimized to fit USA FreedomBattery comsol.
        # https://doi.org/10.1016%2Fj.jpowsour.2006.03.050
        # Equations taken directly from paper table 1
        A = 85.681
        B = -357.70
        C = 613.89
        D = -555.65
        E = 281.06
        F = -76.648
        G = -0.30987
        H = 5.657
        I = 115.0
        J = 13.1983
        OCP_ = A*θ_**6 + B*θ_**5 + C*θ_**4 + D*θ_**3 + E*θ_**2 + F*θ_ + G*np.exp(H*θ_**I) + J
        return OCP_

    @staticmethod
    def NMC_826_Ferraro2020(θ_):
        # Fitted from Wu et al., ref 94, C/25 pseudoOCV
        # https://doi.org/10.1149%2F1945-7111%2Fab632b
        # Equations taken directly from paper
        U1 = -2960.98*θ_**7 + 14272.3*θ_**6 - 29127*θ_**5
        U2 = 32600*θ_**4 - 21599.4*θ_**3
        U3 = 8471.45*θ_**2 - 1823.91*θ_ + 170.967 - np.exp(250*(θ_ - 1))
        OCP_ = U1 + U2 + U3
        return OCP_

if __name__=='__main__':
    electrode = NMC()
    electrode.plot(np.arange(0., 1 + 1e-6, 0.001))




#%%
import numpy as np
import pandas as pd

from P2Dmodel.OCP.OCPbase import OCPbase


class NMC(OCPbase):
    def __init__(self):
        interp1d = OCPbase.interp1d

        # C/40 pseudo OCV, array from manuscript appendices.
        # https://doi.org/10.3390%2Fbatteries5030062
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC_531_Liebig2019')
        self.NMC_531_Liebig2019 = interp1d(table['θs'], table['UOCP'])

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC_668_Birkl2015')
        self.NMC_668_Birkl2015 = interp1d(table['θs'], table['UOCP'])

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC_669_Birkl2015')
        self.NMC_669_Birkl2015 = interp1d(table['θs'], table['UOCP'])

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC_670_Birkl2015')
        self.NMC_670_Birkl2015 = interp1d(table['θs'], table['UOCP'])

        # Dufour2019 thesis fig 2.8, delithiation at C/10 rate
        # https://doi.org/10.1016%2Fj.electacta.2018.03.196
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC_853_Dufour2018')
        self.NMC_853_Dufour2018 = interp1d(table['θs'], table['UOCP'])

        # Dufour2019 thesis fig 2.8, lithiation at C/10 rate
        # https://doi.org/10.1016%2Fj.electacta.2018.03.196
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC_854_Dufour2018')
        self.NMC_854_Dufour2018 = interp1d(table['θs'], table['UOCP'])

        del table

    @staticmethod
    def NMC_400_Smith2006(θs_: np.ndarray) -> np.ndarray:
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
        UOCP_ = A*θs_**6 + B*θs_**5 + C*θs_**4 + D*θs_**3 + E*θs_**2 + F*θs_ + G*np.exp(H*θs_**I) + J
        return UOCP_

    @staticmethod
    def NMC_826_Ferraro2020(θs_: np.ndarray) -> np.ndarray:
        # Fitted from Wu et al., ref 94, C/25 pseudoOCV
        # https://doi.org/10.1149%2F1945-7111%2Fab632b
        # Equations taken directly from paper
        U1_ = -2960.98*θs_**7 + 14272.3*θs_**6 - 29127*θs_**5
        U2_ = 32600*θs_**4 - 21599.4*θs_**3
        U3_ = 8471.45*θs_**2 - 1823.91*θs_ + 170.967 - np.exp(250*(θs_ - 1))
        UOCP_ = U1_ + U2_ + U3_
        return UOCP_

if __name__=='__main__':
    electrode = NMC()
    electrode.plot(np.arange(0., 1 + 1e-6, 0.001))




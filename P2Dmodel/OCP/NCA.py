#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class NCA(OCPbase):
    def __init__(self):

        # COMSOL NCA
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='NCA')
        self.NCA_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/100 pseudo OCV, array from digitized plot.
        # https://doi.org/10.1149%2F1.3129656
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NCA_460_Albertus2009')
        self.NCA_460_Albertus2009 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Model fitted data from Fig 6 (e)
        # https://doi.org/10.1016%2Fj.electacta.2007.09.018
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NCA_593_Abraham2008')
        self.NCA_593_Abraham2008 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/50 rate
        # https://doi.org/10.18154/RWTH-2019-00249
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NCA_716_Hust2019')
        self.NCA_716_Hust2019 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

    @staticmethod
    def NCA_422_Kim2011(θs_: np.ndarray) -> np.ndarray:
        # Fit provided in manuscript, C/100 quasiOCV fitted from ref24 Albertus2009.
        # https://doi.org/10.1149%2F1.3597614
        UOCP_ = 1.638*θs_**10 - 2.222*θs_**9 + 15.056*θs_**8 - 23.488*θs_**7 + 81.246*θs_**6 - 344.566*θs_**5 \
               + 621.3475*θs_**4 - 554.774*θs_**3 + 264.427*θs_**2 - 66.3691*θs_ \
               + 11.8058 - 0.61386*np.exp(5.8201*θs_**136.4)
        return UOCP_

    @staticmethod
    def NCA_601_Dees2008(θs_: np.ndarray) -> np.ndarray:
        # Model fitted data from previous paper Abraham2008
        # [295, 298]
        # https://doi.org/10.1149%2F1.2939211
        UOCP_ = 0.7869*θs_**2 - 2.0565*θs_ + 4.8091
        return UOCP_


if __name__=='__main__':
    electrode = NCA()
    electrode.plot()




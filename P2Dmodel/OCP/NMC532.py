#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class NMC532(OCPbase):
    def __init__(self):

        # COMSOL NMC532
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='NMC532')
        self.NMC532_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

    @staticmethod
    def NMC532_569_Verma2017(θs_: np.ndarray) -> np.ndarray:
        # Delitiation OCV Function from paper.
        # https://doi.org/10.1149%2F2.1701713jes
        UOCP_ = 4851.489856*θs_**9 - 30525.40243*θs_**8 + 83781.31058*θs_**7 - 131496.4647*θs_**6 + 129923.6915*θs_**5 \
               - 83737.64466*θs_**4 + 35194.22436*θs_**3 - 9301.7411*θs_**2 \
               + 1401.288467*θs_ - 87.11977804 - 0.0003*np.exp(7.657*(θs_**115))
        return UOCP_

    @staticmethod
    def NMC532_570_Verma2017(θs_: np.ndarray) -> np.ndarray:
        # Lithiation OCV Function from paper.
        # https://doi.org/10.1149%2F2.1701713jes
        UOCP_ = 5744.862289*θs_**9 - 35520.41099*θs_**8 + 95714.29862*θs_**7 - 147364.5514*θs_**6 \
               + 142718.3782*θs_**5 - 90095.81521*θs_**4 + 37061.41195*θs_**3 - 9578.599274*θs_**2 \
               + 1409.309503*θs_ - 85.31153081 - 0.0003*np.exp(7.657*(θs_**115))
        return UOCP_

if __name__=='__main__':
    pos = NMC532()
    pos.plot(np.arange(0.0, 1 + 1e-6, 0.001))




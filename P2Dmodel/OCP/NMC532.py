#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP import OpenCircuitPotential


class NMC532(OpenCircuitPotential):
    def __init__(self):

        # COMSOL NMC532
        table = pd.read_excel(OpenCircuitPotential.path_OCP_from_COMSOL, sheet_name='NMC532')
        self.NMC532_COMSOL = interp1d(table['θ'], table['OCP'],
            bounds_error=False,
            fill_value='extrapolate')

        del table

    @staticmethod
    def NMC532_569_Verma2017(θ_):
        # Delitiation OCV Function from paper.
        # https://doi.org/10.1149%2F2.1701713jes
        OCP_ = 4851.489856*θ_**9 - 30525.40243*θ_**8 + 83781.31058*θ_**7 - 131496.4647*θ_**6 + 129923.6915*θ_**5\
               - 83737.64466*θ_**4 + 35194.22436*θ_**3 - 9301.7411*θ_**2\
               + 1401.288467*θ_ - 87.11977804 - 0.0003*np.exp(7.657*(θ_**115))
        return OCP_

    @staticmethod
    def NMC532_570_Verma2017(θ_):
        # Lithiation OCV Function from paper.
        # https://doi.org/10.1149%2F2.1701713jes
        OCP_ = 5744.862289*θ_**9 - 35520.41099*θ_**8 + 95714.29862*θ_**7 - 147364.5514*θ_**6 \
               + 142718.3782*θ_**5 - 90095.81521*θ_**4 + 37061.41195*θ_**3 - 9578.599274*θ_**2 \
               + 1409.309503*θ_ - 85.31153081 - 0.0003*np.exp(7.657*(θ_**115))
        return OCP_

if __name__=='__main__':
    electrode = NMC532()
    electrode.plot(np.arange(0.0, 1 + 1e-6, 0.001))




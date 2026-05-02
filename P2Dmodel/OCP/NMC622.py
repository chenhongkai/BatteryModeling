#%%
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class NMC622(OCPbase):
    def __init__(self):

        # COMSOL NMC622
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='NMC622')
        self.NMC622_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/50 quasi-OCV formation cycle experimental data, thickest electrode 304um.
        # https://doi.org/10.1021%2Facs.jpclett.8b02229
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC622_338_Gao2018')
        self.NMC622_338_Gao2018 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Pulses is applied to the python at a fixed
        # current of C/10 during 12 min,
        # followed by a 4 h relaxation time.
        # https://doi.org/10.1016%2Fj.electacta.2020.137428
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC622_968_Chaouachi2021')
        self.NMC622_968_Chaouachi2021 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table


if __name__=='__main__':
    pos = NMC622()
    pos.plot()




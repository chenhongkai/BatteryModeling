#%%
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class NMC111(OCPbase):
    def __init__(self):

        # COMSOL NMC111
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='NMC111')
        self.NMC111_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Lithiation (discharge) OCV measured at C/10 5 percent SOC steps, with 5-hour rest and with qOCV method at C/100
        # https://doi.org/10.1149%2F2.0321816jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC111_52_Schmalstieg2018')
        self.NMC111_52_Schmalstieg2018 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/25 pseudo OCV with MX3 cathode
        # https://doi.org/10.1149%2F2.062204jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC111_563_Wu2012')
        self.NMC111_563_Wu2012 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # 283 K
        # https://doi.org/10.1149%2F2.0661913jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC111_678_Ko2019')
        self.NMC111_678_Ko2019 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # 298 K
        # https://doi.org/10.1149%2F2.0661913jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC111_679_Ko2019')
        self.NMC111_679_Ko2019 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # 313 K
        # https://doi.org/10.1149%2F2.0661913jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC111_681_Ko2019')
        self.NMC111_681_Ko2019 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

if __name__=='__main__':
    pos = NMC111()
    pos.plot()
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class NMC811(OCPbase):
    def __init__(self):

        # COMSOL NMC811
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='NMC811')
        self.NMC811_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Fit provided in manuscript
        # https://doi.org/10.1149%2F1945-7111%2Fab9050
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC811_129_Chen2020')
        self.NMC811_129_Chen2020 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/30 qOCV charge, 3 electrode full python data
        # https://doi.org/10.1016%2Fj.jpowsour.2018.11.043
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='NMC811_626_Sturm2019')
        self.NMC811_626_Sturm2019 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

if __name__=='__main__':
    pos = NMC811()
    pos.plot()




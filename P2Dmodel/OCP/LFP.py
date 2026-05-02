#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class LFP(OCPbase):
    def __init__(self):

        # COMSOL LFP
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='LFP')
        self.LFP_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Assumed pulse GITT charge OCP curves. No further information provided.
        # Web plot digitized data. Hysteresis factor not included.
        # https://doi.org/10.1149%2F2.064209jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='LFP_377_Prada2012')
        self.LFP_377_Prada2012 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Assumed pulse GITT discharge OCP curves. No further information provided.
        # Web plot digitized data. Hysteresis factor not included.
        # https://doi.org/10.1149%2F2.064209jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='LFP_378_Prada2012')
        self.LFP_378_Prada2012 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Sparse GITT data points
        # https://doi.org/10.1149%2F1.1785012
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='LFP_512_Srinivasan2004a')
        self.LFP_512_Srinivasan2004a = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

    @staticmethod
    def LFP_313_Kashkooli2016(θs_: np.ndarray) -> np.ndarray:
        # Fit provided in manuscript, C/50 quasiOCV measurement vs Li
        # https://doi.org/10.1016%2Fj.jpowsour.2015.12.134
        UOCP_ = 3.382 + 0.00470*θs_ + 1.627*np.exp(-81.163*θs_**1.0138) \
               + 7.6445e-8*np.exp(25.36*θs_**2.469) - 8.4410E-8*np.exp(25.262*θs_**2.478)
        return UOCP_

    @staticmethod
    def LFP_325_Farkhondeh2014(θs_: np.ndarray) -> np.ndarray:
        # Fit provided in manuscript, C/50 charge quasiOCV measurement vs Li, note hysteresis effects.
        # https://doi.org/10.1149%2F2.094401jes
        UOCP_ = 3.451 + -8.8e-3*θs_ + 0.6678*np.exp(-81.002*θs_**1.1776) \
               + 2.3738e-9*np.exp(25.222*θs_**3.4801) + -2.6367e-9*np.exp(25.12*θs_**3.4879)
        return UOCP_

    @staticmethod
    def LFP_326_Farkhondeh2014(θs_: np.ndarray) -> np.ndarray:
        # Fit provided in manuscript, C/50 discharge quasiOCV measurement vs Li, note hysteresis effects.
        # https://doi.org/10.1149%2F2.094401jes
        UOCP_ = 3.4227 + -2.0269e-2*θs_ + 0.5087*np.exp(-81.163*θs_**1.0138) \
               + 7.6445e-8*np.exp(25.361*θs_**3.2983) + -8.441e-8*np.exp(25.262*θs_**3.3111)
        return UOCP_

    @staticmethod
    def LFP_371_Delacourt2011(θs_: np.ndarray) -> np.ndarray:
        # Fit provided in manuscript, C/100 quasiOCV measurement vs Li.
        # https://doi.org/10.1016%2Fj.electacta.2011.03.030
        x_ = 1 - θs_
        UOCP_ = 3.4323 + -0.8428*np.exp(-80.2493*x_**1.3198) + -3.2474e-6*np.exp(20.2645*x_**3.8003)\
             + 3.2482E-6*np.exp(20.2646*x_**3.7995)
        return UOCP_

    @staticmethod
    def LFP_511_Srinivasan2004a(θs_: np.ndarray) -> np.ndarray:
        # Sparse GITT ignoring first phase, assumes initial coreshell on LFP, fit provided in manuscript.
        # https://doi.org/10.1149%2F1.1785012
        UOCP_ = 3.114559 + 4.438792*np.arctan(-71.7352*θs_ + 70.85337) - 4.240252*np.arctan(-68.5605*θs_ + 67.730082)
        return UOCP_

    @staticmethod
    def LFP_830_Thorat2011(θs_: np.ndarray) -> np.ndarray:
        # C/30 pseudo OCV on discharge
        # https://doi.org/10.1149%2F2.001111jes
        U1_ = 2.567462 + 57.69*(1 - np.tanh(100*θs_ + 2.9163927))
        U2_ = 0.442953*np.arctan(-65.41928*θs_ + 64.89741)
        U3_ = 0.097237*np.arctan(-160.9058*θs_ + 154.59)
        UOCP_ = U1_ + U2_ + U3_
        return UOCP_

if __name__=='__main__':
    pos = LFP()
    pos.plot()
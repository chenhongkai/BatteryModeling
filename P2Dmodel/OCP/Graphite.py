#%%
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from P2Dmodel.OCP.OCPbase import OCPbase


class Graphite(OCPbase):
    def __init__(self):

        # COMSOL Graphite
        table = pd.read_excel(OCPbase.path_OCP_from_COMSOL, sheet_name='Graphite')
        self.Graphite_COMSOL = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Data points extracted with WebPlotDigitizer,
        # functional form fit with optimization process as discussed in maintext.
        # To measure the OCV, the coin cells were discharged to a minimum voltage using a CC-CV discharge, followed by a stepwise charge process.
        # In each step, the OCV was identified after a break of 5 h. The measurement was performed at 23◦C.
        # https://doi.org/10.1149%2F2.0551509jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_6_Ecker2015')
        self.Graphite_6_Ecker2015 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # OCV measured at C/10 5 percent SOC steps,
        # with 5-hour rest and with qOCV method at C/100,
        # charge and discharge results were similar
        # https://doi.org/10.1149%2F2.0321816jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_53_Schmalstieg2018')
        self.Graphite_53_Schmalstieg2018 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Assumed pulse GITT charge OCP curves.
        # No further information provided.
        # Web plot digitized data. Hysteresis factor not included.
        # https://doi.org/10.1149%2F2.064209jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_381_Prada2012')
        self.Graphite_381_Prada2012 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Assumed pulse GITT discharge OCP curves.
        # No further information provided.
        # Web plot digitized data. Hysteresis factor not included.
        # https://doi.org/10.1149%2F2.064209jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_382_Prada2012')
        self.Graphite_382_Prada2012 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/40 pseudo OCV, array from manuscript appendices.
        # https://doi.org/10.3390%2Fbatteries5030062
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_532_Liebig2019')
        self.Graphite_532_Liebig2019 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_671_Birkl2015')
        self.Graphite_671_Birkl2015 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_672_Birkl2015')
        self.Graphite_672_Birkl2015 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Kokam NMC cells, GITT procedure given in table II,
        # averaged for charge and discharge
        # https://doi.org/10.1149%2F2.0331512jes
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_673_Birkl2015')
        self.Graphite_673_Birkl2015 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Half-python quasi-OCV on charge
        # https://doi.org/10.1016%2Fj.electacta.2012.04.050
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_708_Li2012')
        self.Graphite_708_Li2012 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/50 rate
        # https://doi.org/10.18154/RWTH-2019-00249
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_717_Hust2019')
        self.Graphite_717_Hust2019 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

            # Dufour2019 thesis fig 2.8, lithiation at C/10 rate
        # https://doi.org/10.1016%2Fj.electacta.2018.03.196
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_851_Dufour2018')
        self.Graphite_851_Dufour2018 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Dufour2019 thesis fig 2.8, delithiation at C/10 rate
        # https://doi.org/10.1016%2Fj.electacta.2018.03.196
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_852_Dufour2018')
        self.Graphite_852_Dufour2018 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # C/70 pseudo ocv
        # https://doi.org/10.1149%2F1.2817888
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_967_Kumaresan2008')
        self.Graphite_967_Kumaresan2008 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        # Pulses is applied to the python at a fixed current of C/10 during 12 min,
        # followed by a 4 h relaxation time.
        # https://doi.org/10.1016%2Fj.electacta.2020.137428
        table = pd.read_excel(OCPbase.path_OCP_from_LiionDB, sheet_name='Graphite_969_Chaouachi2021')
        self.Graphite_969_Chaouachi2021 = interp1d(table['θs'], table['UOCP'], **OCPbase.kwargs_interp1d)

        del table

    @staticmethod
    def Graphite_312_Doyle1996(θs_: np.ndarray) -> np.ndarray:
        # Fit provided in manuscript
        # https://doi.org/10.1149%2F1.1836921
        # FITTING:
        # Equations taken directly from paper appendix B
        U1_ = -0.16 + 1.32*np.exp(-3.0*θs_)
        U2_ = 10*np.exp(-2000.0*θs_)
        UOCP_ = U1_ + U2_
        return UOCP_

    @staticmethod
    def Graphite_401_Smith2006(θs_: np.ndarray) -> np.ndarray:
        # Fit provided in manuscript, inverse optimized to fit USA FreedomBattery comsol.
        # https://doi.org/10.1016%2Fj.jpowsour.2006.03.050
        # FITTING:
        # Equations taken directly from paper table 1
        A = 8.00229
        B = 5.0647
        C = -12.578
        D = -8.6322E-4
        E = 2.1765E-5
        F = -0.46016
        G = 15.0
        H = 0.06
        I = -0.55364
        J = -2.4326
        K = -0.92
        UOCP_ = A + B*θs_ + C*θs_**0.5 + D*θs_**-1 + E*θs_**1.5 + F*np.exp(G*(H - θs_)) + I*np.exp(J*(θs_ + K))
        return UOCP_

    @staticmethod
    def Graphite_423_Kim2011(θs_: np.ndarray) -> np.ndarray:
        # C/25 quasiOCV fitted from ref26 Srinivasan2004.
        # https://doi.org/10.1149%2F1.3597614
        A0 = 0.124
        B0 = 1.5
        C0 = -70

        A1 = -0.0351
        B1 = -0.286
        C1 = 0.083

        A2 = -0.0045
        B2 = -0.9
        C2 = 0.119

        A3 = -0.035
        B3 = -0.99
        C3 = 0.05

        A4 = -0.0147
        B4 = -0.5
        C4 = 0.034

        A5 = -0.102
        B5 = -0.194
        C5 = 0.142

        A6 = -0.022
        B6 = -0.98
        C6 = 0.0164

        A7 = -0.011
        B7 = -0.124
        C7 = 0.0226

        A8 = 0.0155
        B8 = -0.105
        C8 = 0.029

        UOCP_ = A0 + B0*np.exp(C0*θs_) + A1*np.tanh((θs_ + B1)/C1) + A2*np.tanh((θs_ + B2)/C2) + A3*np.tanh(
            (θs_ + B3)/C3) + A4*np.tanh((θs_ + B4)/C4) + A5*np.tanh((θs_ + B5)/C5) + A6*np.tanh((θs_ + B6)/C6) + A7*np.tanh(
            (θs_ + B7)/C7) + A8*np.tanh((θs_ + B8)/C8)
        return UOCP_

    @staticmethod
    def Graphite_484_Doyle2003(θs_: np.ndarray) -> np.ndarray:
        # C/60 pseudo OCV, fit provided in manuscript.
        # https://doi.org/10.1149%2F1.1569478
        # Equations taken directly from paper appendices
        c1 = 8.002296379
        c2 = 5.064722977
        c3 = -12.57808059
        c4 = 8.632208755E-4
        c5 = 2.176468281E-5
        c6 = -0.4601573522
        c7 = 0.5536351675
        c8 = -2.432630003
        OCP_ = c1 + c2*θs_ + c3*θs_**0.5 - c4*θs_**(-1) + c5*θs_**1.5 + c6*np.exp(15*(0.06 - θs_)) - c7*np.exp(c8*(θs_ - 0.92))
        return OCP_

    @staticmethod
    def Graphite_523_Srinivasan2004b(θs_: np.ndarray) -> np.ndarray:
        # C/25 pseudo OCV, fit provided in manuscript.
        # https://doi.org/10.1149%2F1.1785013
        # Equations taken directly from  manuscript
        UOCP_ = 0.124 + 1.5*np.exp(-70*θs_) - 0.0351*np.tanh((θs_ - 0.286)/0.083) - 0.0045*np.tanh(
            (θs_ - 0.9)/0.119) - 0.035*np.tanh((θs_ - 0.99)/0.05) - 0.0147*np.tanh((θs_ - 0.5)/0.034) - 0.102*np.tanh(
            (θs_ - 0.194)/0.142) - 0.022*np.tanh((θs_ - 0.98)/0.0164) - 0.011*np.tanh((θs_ - 0.124)/0.0226) + 0.0155*np.tanh(
            (θs_ - 0.105)/0.029)
        return UOCP_
    
    @staticmethod
    def Graphite_719_Tang2019(θs_: np.ndarray) -> np.ndarray:
        # GITT and C/50 qOCV compared with Ramadass2003
        # https://doi.org/10.1016%2Fj.ssi.2019.115083
        # Equations taken directly from paper
        UOCP_ = (0.722 + 0.1387*θs_ + 0.029*θs_**0.5 - 0.0172/θs_ + 0.0019/θs_**1.5
                + 0.2808*np.exp(0.9 - 15*θs_) - 0.7984*np.exp(0.4465*θs_ - 0.4108))
        return UOCP_

    @staticmethod
    def Graphite_749_Arora1999(θs_: np.ndarray) -> np.ndarray:
        # MCMB material OCV from reference 4
        # https://doi.org/10.1149%2F1.1392512
        # Equations taken directly from paper
        UOCP_ = 0.7222 + 0.13868*θs_ + 0.028952*θs_**0.5 - 0.017189*(1/θs_) + 0.0019144*(1/(θs_**1.5)) \
               + 0.28082*np.exp(15*(0.06 - θs_)) - 0.79844*np.exp(0.44649*(θs_ - 0.92))
        return UOCP_

    @staticmethod
    def Graphite_785_Jiang2016(θs_: np.ndarray) -> np.ndarray:
        # Taken from ref 56, Fang2014
        # https://doi.org/10.1038%2Fsrep32639
        # Equations taken directly from paper
        U1_ = 0.13966 + 0.68920*np.exp(-49.20361*θs_)
        U2_ = 0.41903*np.exp(-254.40067*θs_) - np.exp(49.97886*θs_ - 43.37888)
        U3_ = - 0.028221*np.arctan(22.523*θs_ - 3.65328) - 0.01308*np.arctan(28.34801*θs_ - 13.43960)
        UOCP_ = U1_ + U2_ + U3_
        return UOCP_

if __name__=='__main__':
    neg = Graphite()
    neg.plot(np.arange(0.01, 1 + 1e-6, 0.001))


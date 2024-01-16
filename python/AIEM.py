"""
Author: Fei Zhao
Create: 2023-08-19

Description:
    Advanced Integration Equation Model (AIEM)
    ref: Chen, K. S., et al. (2003). "Emission of rough surfaces calculated by the integral equation method with comparison to three-dimensional moment method simulations." IEEE Transactions on Geoscience and Remote Sensing 41(1): 90-101.
"""
import numpy as np
import matlab.engine
import scipy.constants as C
from QuasiSpecularSurface import QuasiSpecularSurface

class AIEM(QuasiSpecularSurface):
    """
    Advanced Integration Equation Model (AIEM)
    """
    matlab_eng = None
    is_matlab_started = False

    def __init__(self, epsilon_r, kdel, kcor, corr_fun='exp', x=1.5, shadow_flag=True):
        """
        Set parameters for characterizing the rough surface
        INPUT:
            epsilon_r: relative dielectric constant
            kdel: k * delta, normalized RMS height
            kcor: k * corr_len, normalized correlation length
            corr_fun: correlation function: 'gauss', 'exp', 'x-power', 'x-exp' (default 'exp')
            x: coefficient (>1) needed for 'x-power' and 'x-exp(onential)' correl. fnc. (default 1.5)
            shadow_flag: True (default) or False for including shadow effect
        OUTPUT:
            a rough surface instance
        """
        super(AIEM, self).__init__(epsilon_r, kdel)
        self.kcor = kcor
        self.corr_fun = corr_fun
        self.x = x
        self.shadow_flag = shadow_flag

        # startup the matlab only once
        if self.is_matlab_started is False:
            self.matlab_eng = matlab.engine.start_matlab()
            self.is_matlab_started = True
            self.matlab_eng.cd(r'E:/exps/lunar_scattering/lunar_regolith_scattering/mores/matlab', nargout=0)
    
    def Mue_noncoh_R(self, geom, is_down=True):
        """
        Mueller matrix for upward diffuse scattering from rough surface in forward scattering alignment (FSA) convention.
        INPUT:
            geom (tuple): observation angles (theta_s, phi_s, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                            Note that theta_s and theta_i belong to [0, 90] defined in surface scattering coordinate
                            theta_s is the angle between z and ks, while theta_i is the angle between z and -ki
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            R: 4x4 real Mueller matrix
        """
        # return np.zeros((4, 4)) # test for the speed of no using matlab
    
        theta_s, phi_s, theta_i, phi_i = geom
        f = 1.0e9
        angle_s = matlab.double([theta_s, phi_s])
        angle_i = matlab.double([theta_i, phi_i])
        epsilon_1 = 1.0
        epsilon_2 = 1.0 * self.epsilon_r if is_down is True else 1.0/self.epsilon_r
        Lambda = C.speed_of_light / f
        k = 2.0 * np.pi / Lambda
        sp = matlab.double([self.kdel/k, self.kcor/k])
        SpR = self.corr_fun
        x = 1.0 * self.x
        shadow_flag = self.shadow_flag

        # print("AIEM parameters=f:{}, angle_s:{}, angle_i:{}, epsilon_1:{}, epsilon_2:{}, sp:{}, SpR:{}, x:{}, shadow_flag:{}".format(f, 
        #     angle_s, angle_i, epsilon_1, epsilon_2, sp, SpR, x, shadow_flag))

        # AIEM
        # R = self.matlab_eng.Fun_R_rough_AIEMs_noRvplusRh(f, angle_s, angle_i, 
        #             epsilon_1, epsilon_2, sp, SpR, x, shadow_flag, nargout=1)
        # IEM
        R = self.matlab_eng.Fun_R_rough_IIEMs(f, angle_s, angle_i, 
            epsilon_1, epsilon_2, sp, SpR, x, shadow_flag, nargout=1)
        R = np.array(R)

        return R
        
    def Mue_noncoh_T(self, geom, is_down=True):
        """
        Mueller matrix for downward diffuse scattering from rough surface in forward scattering alignment (FSA) convention.
        INPUT:
            geom (tuple): observation angles (theta_t, phi_t, theta_i, phi_i) in degree
                            theta_s and phi_s are scattering angles, and theta_i and phi_i are incidence angles
                            Note that theta_t and theta_i belong to [0, 90] defined in surface scattering coordinate
                            theta_t is the angle between -z and ks, while theta_i is the angle between z and -ki
            isdown: True (default) for downward incident and False for upward incident
        OUTPUT:
            T: 4x4 real Mueller matrix
        """
        return np.zeros((4, 4))
# Fuction for calculating the viscous accretion disk evolution under the
# self-similar solution (no extra disk removal)

import numpy as np
import inputFile as pars
from commonFunctions import (
    midplaneTemp,kepAngVel,soundSpeed,gasScaleHeight,viscosity
)

#########################################

def viscAccDisc_grid(t,r):

    Mdot0_r = np.ones((len(r)))*pars.Mdot0

    gamma = 1.5 - pars.tempExp
    Omega = kepAngVel(pars.Rout)
    Temp = midplaneTemp(pars.Rout)
    Cs = soundSpeed(Temp)
    Hg = gasScaleHeight(Cs,Omega)
    nu_Rout = viscosity(Omega,Hg)
    t_s = 1.0/(3.0*(2.0-gamma)**2) * pars.Rout**2/nu_Rout
    T_1 = t/t_s + 1

    if np.isscalar(T_1):
        Mdot = Mdot0_r * T_1**( -(5/2-gamma)/(2-gamma) )
        sigma = Mdot/(3*np.pi*nu_Rout*(r/pars.Rout)**gamma) * np.exp(-(r/pars.Rout)**(2-gamma)/T_1)
    else:
        Mdot = Mdot0_r * T_1[:, np.newaxis]**( -(5/2-gamma)/(2-gamma) )
        sigma = Mdot/(3*np.pi*nu_Rout*(r[np.newaxis, :]/pars.Rout)**gamma) * np.exp(-(r[np.newaxis, :]/pars.Rout)**(2-gamma)/T_1[:, np.newaxis])

    return Mdot,sigma
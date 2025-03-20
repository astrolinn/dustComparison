# File containing common functions
######################################################

import numpy as np
from astropy import constants as c
import inputFile as pars

######################################################

# Constants in cgs units

G = c.G.cgs.value
au = c.au.cgs.value
sigma_sb = c.sigma_sb.cgs.value
kB = c.k_B.cgs.value
mH = c.u.cgs.value

#####################################################

# Keplerian angular velocity

def kepAngVel(r):
    Omega = np.sqrt(G*pars.Mstar/r**3)
    return Omega

# Midplane temperature

def midplaneTemp(r):
    if pars.tempProfile == "CG97":
        T = pars.Tconst * (r/au)**(-pars.tempExp)
    elif pars.tempProfile == "passIrr":
        Lstar = 4*np.pi*pars.Rstar**2*sigma_sb*pars.Tstar**4
        T = ( 1/2*0.05*Lstar/(4*np.pi*r**2*sigma_sb) )**(1/4)
    else:
        raise ValueError("Must choose temperature profile")
    return T

# Sound speed

def soundSpeed(T):
    Cs = np.sqrt(kB*T/(pars.mu*mH))
    return Cs

# Scale height of gas

def gasScaleHeight(Cs,Omega):
    H = Cs/Omega
    return H

# Viscosity

def viscosity(Omega,H):
    nu = pars.alpha*Omega*H**2
    return nu

# Stokes number

def st_number(r,temp,sigmaGas,size):
    Omega = (G*pars.Mstar/r**3)**(1/2)
    Cs = (kB*temp/(pars.mu*mH))**(1/2)
    Hg = Cs/Omega[None,:]
    rho_2D = sigmaGas/(np.sqrt(2*np.pi)*Hg)
    st_2D = size*pars.rhop/(Hg*rho_2D)
    return st_2D
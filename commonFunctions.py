# File containing common functions
######################################################

import numpy as np
from astropy import constants as c
from inputFile import (
    Mstar,tempProfile,Tconst,tempExp,Rstar,Tstar,mu,alpha,alphaTurb
)

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
    Omega = np.sqrt(G*Mstar/r**3)
    return Omega

# Midplane temperature

def midplaneTemp(r):
    if tempProfile == "CG97":
        T = Tconst * (r/au)**(-tempExp)
    elif tempProfile == "passIrr":
        Lstar = 4*np.pi*Rstar**2*sigma_sb*Tstar**4
        T = ( 1/2*0.05*Lstar/(4*np.pi*r**2*sigma_sb) )**(1/4)
    else:
        raise ValueError("Must choose temperature profile")
    return T

# Sound speed

def soundSpeed(T):
    Cs = np.sqrt(kB*T/(mu*mH))
    return Cs

# Scale height of gas

def gasScaleHeight(Cs,Omega):
    H = Cs/Omega
    return H

# Viscosity

def viscosity(Omega,H):
    nu = alpha*Omega*H**2
    return nu
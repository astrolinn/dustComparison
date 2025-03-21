# File containing common functions
######################################################

import numpy as np
from scipy.interpolate import interp1d
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
    """
    Calculates the radial midplane temperature structure
    Two options, decided in inputFile
    """
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
    """
    Calculates the Shakura & Sunyaev viscosity
    """
    nu = pars.alpha*Omega*H**2
    return nu

# Stokes number

def st_number(r,temp,sigmaGas,size):
    """
    Calculates the Stokes number along the entire
    time and semimajor axis grid
    """
    Omega = kepAngVel(r)
    Cs = soundSpeed(temp)
    Hg = gasScaleHeight(Cs, Omega)
    rho_2D = sigmaGas/(np.sqrt(2*np.pi)*Hg)
    st_2D = size*pars.rhop/(Hg*rho_2D)
    return st_2D

# Pressure gradient

def calc_dlnPdlnr(r, temp, sigma_gas):
    """
    Calculates the radial pressure gradient along the
    entire time and semimajor axis grid
    """
    Omega = kepAngVel(r)
    Cs = soundSpeed(temp)
    H = gasScaleHeight(Cs, Omega)
    dlnsigma_gasdlnr = np.gradient(np.log(sigma_gas), axis=1) / np.gradient(np.log(r))
    dlntempdlnr = np.gradient(np.log(temp), axis=1) / np.gradient(np.log(r))
    dlnHdlnr = np.gradient(np.log(H), axis=1) / np.gradient(np.log(r))
    dlnPdlnr = dlnsigma_gasdlnr + dlntempdlnr - dlnHdlnr
    return dlnPdlnr

# Pebble scale height (assumes monodisperse St)

def pebbleScaleHeight(r, temp, St):
    Omega = kepAngVel(r)
    Cs = soundSpeed(temp)
    H = gasScaleHeight(Cs, Omega)
    return H * np.sqrt(pars.alphaTurb/(pars.alphaTurb+St))

# Midplane density

def midplaneDensity(sigma, H):
    return sigma / (np.sqrt(2*np.pi) * H)

# Time interpolation

def interp_t(tnew, told, array_old):
    """
    Performs linear interpolation along axis=0 to obtain
    an array onto a new time-array
    """
    array_new = interp1d(told, array_old, kind='linear', axis=0, fill_value='extrapolate')(tnew)
    return array_new


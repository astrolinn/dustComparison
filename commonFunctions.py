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
m_p = c.m_p.cgs.value

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
    Cs = np.sqrt(kB*T/(pars.mu*m_p))
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
    time and semimajor axis grid in the same way
    as TwoPopPy
    """
    Omega = kepAngVel(r)
    Cs = soundSpeed(temp)
    Hg = gasScaleHeight(Cs, Omega)
    rhoGas = sigmaGas/(np.sqrt(2*np.pi)*Hg)
    mfp = pars.mu * m_p / (rhoGas * pars.sig_h)
    epstein = np.pi / 2. * size * pars.rhop / sigmaGas
    stokesI = np.pi * 2. / 9. * size**2. * pars.rhop / (mfp * sigmaGas)
    return np.where((size > 9. / 4. * mfp), stokesI, epstein)

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

# Planetary Gap profiles (from the DustPy documentation)

def duffell2020(r, a, q, h, alpha0):
    """
    Function calculates the planetary gap profile according Duffell (2020).

    Parameters
    ----------
    r : array-like, (Nr,)
        Radial grid
    a : float
        Semi-major axis of planet
    q : float
        Planet-star mass ratio
    h : float
        Aspect ratio at planet location
    alpha0 : float
        Unperturbed alpha viscosity parameter

    Returns
    -------
    f : array-like, (Nr,)
        Pertubation of surface density due to planet
    """

    # Mach number
    M = 1./h

    # Add small value to avoid division by zero
    qp = q + 1.e-100

    # qtilde from equation (18) has shape (Nr,)
    D = 7*M**1.5/alpha0**0.25
    qtilde = qp/(1+D**3*((r/a)**(1./6.)-1)**6)**(1./3.)

    # delta from equation (9)
    # Note: there is a typo in the original publication
    # (q/qw)**3 is added in both cases
    qnl = 1.04/M**3
    qw = 34. * qnl * np.sqrt(alpha0*M)
    delta = np.where(qtilde > qnl, np.sqrt(qnl/qtilde), 1.) + (qtilde/qw)**3

    # Gap shape
    ret = 1. / (1. + 0.45/(3.*np.pi) * qtilde**2 * M**5 * delta/alpha0)

    return ret


def kanagawa2017(r, a, q, h, alpha0):
    """
    Function calculates the planetary gap profile according
    Kanagawa et al. (2017).

    Parameters
    ----------
    r : array-like, (Nr,)
        Radial grid
    a : float
        Semi-major axis of planet
    q : float
        Planet-star mass ratio
    h : float
        Aspect ratio at planet location
    alpha0 : float
        Unperturbed alpha viscosity parameter

    Returns
    -------
    f : array-like, (Nr,)
        Pertubation of surface density due to planet
    """

    # Distance to planet (normalized)
    dist = np.abs(r-a)/a

    # Add small value to avoid division by zero
    qp = q + 1.e-100

    K = qp**2 / (h**5 * alpha0)
    Kp = qp**2 / (h**3 * alpha0)
    Kp4 = Kp**(0.25)
    SigMin = 1. / (1 + 0.04*K)
    SigGap = 4 / Kp4 * dist - 0.32
    dr1 = (0.25*SigMin + 0.08) * Kp**0.25
    dr2 = 0.33 * Kp**0.25
    # Gap edges
    ret = np.where((dr1 < dist) & (dist < dr2), SigGap, 1.)
    # Gap center
    ret = np.where(dist < dr1, SigMin, ret)

    return ret

def planet_profile(profile, r, a, q, hr, alpha0):
    """
    Function calculates the planetary gap profile according to
    the chosen profile.

    Parameters
    ----------
    profile : str
        Either "Duffell" or "Kanagawa
    ----------
    r : array-like, (Nr,)
        Radial grid
    a : float
        Semi-major axis of planet
    q : float
        Planet-star mass ratio
    hr : float
        Aspect ratio as a function of r
    alpha0 : float
        Unperturbed alpha viscosity parameter

    Returns
    -------
    f : array-like, (Nr,)
        Pertubation of surface density due to planet
    """

    hrp = interp1d(r, hr)(a)

    if profile=="Duffell":
        return duffell2020(r, a, q, hrp, alpha0)
    elif profile=="Kanagawa":
        return kanagawa2017(r, a, q, hrp, alpha0)
    else:
        raise ValueError("Profile must be either 'Duffell' or 'Kanagawa'")

def refine_radial_local(ri, r0, num=3):
    """
    Function refines the radial grid locally bysplitting grid cells
    recursively at a specific location.

    Parameters
    ----------
    ri : array-like, (Nr,)
        Radial grid cell interfaces
    r0 : float
        Radial location to be refined
    num : int, optional, default: 3
        Number of refinement steps

    Returns
    -------
    ri_fine : array-like, (Nr+,)
        Refined radial grid cell interfaces
    """
    # Break recursion
    if num == 0:
        return ri

    # Closest index to location
    i0 = np.abs(ri-r0).argmin()
    i0 = np.argmax(ri > r0)-1
    # Boundary indices of refinement region
    il = np.maximum(0, i0-num+1)
    ir = np.minimum(i0+num, ri.shape[0]-1)

    # Left and right unmodified regions
    ril = ri[:il]
    rir = ri[ir:]

    # Initialize refined region
    N = ir-il
    rim = np.empty(2*N)

    # Refined grid boundary is geometric mean
    for i in range(0, N):
        j = il+i
        rim[2*i] = ri[j]
        rim[2*i+1] = np.sqrt(ri[j]*ri[j+1])

    # New refined grid
    ri_fine = np.hstack((ril, rim, rir))

    # Next level of recursion
    return refine_radial_local(ri_fine, r0, num=num-1)
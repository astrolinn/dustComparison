# Simulate the growth of a single planet via pebble accretion, 
# growth is not stopped at the pebble isolation mass, plot the results
# Dust evolution codes:
# - TwoPopPy2
# - TriPod
# - DustPy
#
# For TriPod & DustPy:
#   Polydisperse and Monodisperse pebble accretion
#   2 choices for representative St in monodisperse case:
#   - Density-weighted average
#   - Peak of density-vs-St distribution

import os
import numpy as np
from scipy.special import i0,i1
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy import constants as c
import inputFile as pars
from commonFunctions import (
    st_number, kepAngVel, soundSpeed, 
    gasScaleHeight, calc_dlnPdlnr, interp_t, 
    pebbleScaleHeight, midplaneDensity
)
import sys
sys.path.insert(0, os.getcwd())

year = 365.25*24*3600
au = c.au.cgs.value
kB = c.k_B.cgs.value
mH = c.u.cgs.value
G = c.G.cgs.value
ME = c.M_earth.cgs.value

path = "pebbledata"
exists = os.path.exists(path)
if not exists:
    os.mkdir(path)

################################
### Load dust evolution data ###

# DustPy
r_dp=np.load('files_dp/r.npy')
t_dp=np.load('files_dp/tdustev.npy')
temp_2D_dp=np.load('files_dp/temp_2D.npy')
sigma_g_2D_dp=np.load('files_dp/sigma_gas_2D.npy')
rho_d_3D_dp=np.load('files_dp/rho_dust_3D.npy')
rho_d_2D_dp = rho_d_3D_dp.sum(-1)
sigma_d_3D_dp=np.load('files_dp/sigma_dust_3D.npy')
sigma_d_2D_dp = sigma_d_3D_dp.sum(-1)
st_3D_dp=np.load('files_dp/st_3D.npy')

# TwoPopPy2
r_tp2=np.load('files_tp/r_tp.npy')
t_tp2=np.load('files_tp/time_tp.npy')
temp_2D_tp2=np.load('files_tp/temp_tp.npy')
sigma_g_2D_tp2=np.load('files_tp/sigma_gas_tp.npy')
sigma_d_2D_tp2=np.load('files_tp/sigma_dust_tp.npy')
size_2D_tp2=np.load('files_tp/size_tp.npy')

# TriPod
r_trpd=np.load('files_trpd/r_trpd.npy')
t_trpd=np.load('files_trpd/t_trpd.npy')
temp_2D_trpd=np.load('files_trpd/temp_2D.npy')
sigma_g_2D_trpd=np.load('files_trpd/sigma_gas_2D.npy')
rho_d_3D_trpd=np.load('files_trpd/rho_recon.npy')
rho_d_2D_trpd = rho_d_3D_trpd.sum(-1)
sigma_d_3D_trpd=np.load('files_trpd/Sigma_recon.npy')
sigma_d_2D_trpd = sigma_d_3D_trpd.sum(-1)
st_3D_trpd=np.load('files_trpd/St_recon.npy')
st_2D_trpd_max=np.load('files_trpd/st_3D.npy')[:,:,-1]

####################
### Calculate St ###

# Convert size to St for TwoPopPy2
st_2D_tp2 = st_number(r_tp2,temp_2D_tp2,sigma_g_2D_tp2,size_2D_tp2)

# Obtain representative St for DustPy and tripod

# Density-weighted average
st_2D_dp_aver = np.sum(st_3D_dp * sigma_d_3D_dp, axis=2) / np.sum(sigma_d_3D_dp, axis=2)
st_2D_trpd_aver = np.sum(st_3D_trpd * sigma_d_3D_trpd, axis=2) / np.sum(sigma_d_3D_trpd, axis=2)
# Peak of size distribution
ind = np.argmax(sigma_d_3D_dp, axis=2)
st_2D_dp_peak = np.take_along_axis(st_3D_dp, ind[..., np.newaxis], axis=2).squeeze(axis=2)

################################################
### Calculate midplane density for TwoPopPy2 ###

Hpebb_tp2 = pebbleScaleHeight(r_tp2, temp_2D_tp2, st_2D_tp2)
rho_d_2D_tp2 = midplaneDensity(sigma_d_2D_tp2, Hpebb_tp2)

##########################################################
### Function for calculating the pebble isolatino mass ###

def calc_Miso(a, temp, dlnPdlnr):
    """
    Calculates the pebble isolation mass at semimajor
    axis a
    """
    alpha3 = 0.001
    Omega = kepAngVel(a)
    Cs = soundSpeed(temp)
    H = gasScaleHeight(Cs, Omega)
    ffit = (H / a / 0.05)**3 * (0.34 * (np.log(alpha3) / np.log(pars.alphaTurb))**4 + 0.66) * (1 - (dlnPdlnr + 2.5) / 6)
    return 25 * c.M_earth.cgs.value * ffit

#######################################################
### Function for monodisperse pebble accretion rate ###

def pebbAccRate_mono(a, temp, m, St, sigma_dust, rho_dust, dlnPdlnr):
    """
    Calculates the pebble accretion rate using eq.35
    of Lyra et al. (2023) for the case of monodisperse
    pebble accretion
    The standard 2D limit (eq.30 of Lyra et al. 2023)
    is used as well to avoid numerical issues
    """
    chi = 0.4
    gamma = 0.65
    Omega = kepAngVel(a)
    Cs = soundSpeed(temp)
    H = gasScaleHeight(Cs, Omega)
    Deltav = -0.5 * H / a * dlnPdlnr * Cs
    rH = (m / (3 * pars.Mstar))**(1/3) * a
    tp = c.G.cgs.value * m / (Deltav + Omega * rH)**3
    Mt = Deltav**3 / (c.G.cgs.value * Omega)
    Hd = pebbleScaleHeight(a, temp, St)
    tau_f = St / Omega
    M_HB = Mt / (8 * St)
    if (m < M_HB):
        # Bondi regime
        R_B = c.G.cgs.value * m / Deltav**2
        t_B = R_B / Deltav
        Racc_hat = (4 * tau_f / t_B)**0.5 * R_B
    else:
        # Hill regime
        Racc_hat = (St / 0.1)**(1/3) * rH
    Racc = Racc_hat * np.exp(-chi * (tau_f / tp)**gamma)
    deltav = Deltav + Omega * Racc
    eps = (Racc / (2 * Hd))**2
    if (eps < pars.lim3d_pebbAcc):
        I0 = i0(eps)
        I1 = i1(eps)
        Mdot_core = np.pi * Racc**2 * rho_dust * deltav * np.exp(-eps) * (I0+I1)
    else:
        Mdot_core = 2 * Racc * sigma_dust * deltav
    return Mdot_core

#######################################################
### Function for polydisperse pebble accretion rate ###

def pebbAccRate_poly(a, temp, m, St_poly, sigma_dust_poly, rho_dust_poly, dlnPdlnr):
    """
    Calculates the pebble accretion rate using eq.35
    of Lyra et al. (2023) for the case of polydisperse
    pebble accretion
    The standard 2D limit (eq.30 of Lyra et al. 2023)
    is used as well to avoid numerical issues
    DustPy produces dust fluids with St>>1, to avoid
    numerical issues we remove the high tail of the
    stokes distribution
    """
    chi = 0.4
    gamma = 0.65
    Omega = kepAngVel(a)
    Cs = soundSpeed(temp)
    H = gasScaleHeight(Cs, Omega)
    Deltav = -0.5 * H / a * dlnPdlnr * Cs
    rH = (m / (3 * pars.Mstar))**(1/3) * a
    tp = c.G.cgs.value * m / (Deltav + Omega * rH)**3
    Mt = Deltav**3 / (c.G.cgs.value * Omega)
    indmax = np.argmax(np.where(np.cumsum(rho_dust_poly, -1) / np.sum(rho_dust_poly, -1) < pars.fracmax_pebbAcc, St_poly, 0.0)) + 1
    Mdot_core = np.zeros((indmax))
    for i in range(indmax):
        Hd = pebbleScaleHeight(a, temp, St_poly[i])
        tau_f = St_poly[i] / Omega
        M_HB = Mt / (8 * St_poly[i])
        if (m < M_HB):
            # Bondi regime
            R_B = c.G.cgs.value * m / Deltav**2
            t_B = R_B / Deltav
            Racc_hat = (4 * tau_f / t_B)**0.5 * R_B
        else:
            # Hill regime
            Racc_hat = (St_poly[i] / 0.1)**(1/3) * rH
        Racc = Racc_hat * np.exp(-chi * (tau_f / tp)**gamma)
        deltav = Deltav + Omega * Racc
        eps = (Racc / (2 * Hd))**2
        if (eps < pars.lim3d_pebbAcc):
            I0 = i0(eps)
            I1 = i1(eps)
            Mdot_core[i] = np.pi * Racc**2 * rho_dust_poly[i] * deltav * np.exp(-eps) * (I0+I1)
        else:
            Mdot_core[i] = 2 * Racc * sigma_dust_poly[i] * deltav
    Mdot_core_tot = sum(Mdot_core)
    return Mdot_core_tot

##############################################################
### Function for calculating the growth of a single planet ###

def pebbAcc(t, tdustev, St, sigma_dust, rho_dust, rdustev, temp, sigma_gas):
    """
    Solves for the mass growth of a single planet via
    pebble accretion
    """
    mPlanet = np.zeros((len(t)))
    mPlanet[0] = pars.Mcore
    # Find closest index in rdustev to Rcore, use that as
    # the planet's semimajor axis
    ir = np.abs(rdustev - pars.Rcore).argmin()
    a = rdustev[ir]
    # Interpolate to obtain the dust evolution arrays on
    # the time-array t for pebble accretion
    sigma_gas = interp_t(t, tdustev, sigma_gas)
    temp = interp_t(t, tdustev, temp)
    sigma_dust = interp_t(t, tdustev, sigma_dust)
    rho_dust = interp_t(t, tdustev, rho_dust)
    St = interp_t(t, tdustev, St)
    dlnPdlnr = calc_dlnPdlnr(rdustev, temp, sigma_gas)
    # Update planetary masses using simple Euler method 
    for it in range(1, len(t)):
        m = mPlanet[it-1]
        Miso = calc_Miso(a, temp[it,ir], dlnPdlnr[it,ir])
        if St.ndim == 2:
            # Monodisperse pebble accretion
            Mdot = pebbAccRate_mono(a, temp[it,ir], m, St[it,ir], sigma_dust[it,ir], rho_dust[it,ir], dlnPdlnr[it,ir])
            mnew = m + Mdot * pars.dt_pebbAcc
        elif St.ndim == 3:
            # Polydisperse pebble accretion
            Mdot = pebbAccRate_poly(a, temp[it,ir], m, St[it,ir,:], sigma_dust[it,ir,:], rho_dust[it,ir,:], dlnPdlnr[it,ir])
            mnew = m + Mdot * pars.dt_pebbAcc
        else:
            print("Wrong array dimensions in pebble accretion")
        mPlanet[it] = mnew
    return mPlanet

###########################################################
### Compute the growth of a planet via pebble accretion ###

# Set up time-array for pebble accretion
# For now do until end of time-array for DustPy
tend = t_dp[-1]
num_points = int((tend - pars.tcore) / pars.dt_pebbAcc) + 1
t = np.linspace(pars.tcore, tend, num_points)

# TwoPopPy2
m_tp2 = pebbAcc(t, t_tp2, st_2D_tp2, sigma_d_2D_tp2, rho_d_2D_tp2, r_tp2, temp_2D_tp2, sigma_g_2D_tp2)

# DustPy polydisperse
m_dp_poly = pebbAcc(t, t_dp, st_3D_dp, sigma_d_3D_dp, rho_d_3D_dp, r_dp, temp_2D_dp, sigma_g_2D_dp)

# DustPy monodisperse - density weighted average St
m_dp_mono_aver = pebbAcc(t, t_dp, st_2D_dp_aver, sigma_d_2D_dp, rho_d_2D_dp, r_dp, temp_2D_dp, sigma_g_2D_dp)

# DustPy monodisperse - peak St
m_dp_mono_peak = pebbAcc(t, t_dp, st_2D_dp_peak, sigma_d_2D_dp, rho_d_2D_dp, r_dp, temp_2D_dp, sigma_g_2D_dp)

# TriPod polydisperse
m_trpd_poly = pebbAcc(t, t_trpd, st_3D_trpd, sigma_d_3D_trpd, rho_d_3D_trpd, r_trpd, temp_2D_trpd, sigma_g_2D_trpd)

# TriPod monodisperse - density weighted average St
m_trpd_mono_aver = pebbAcc(t, t_trpd, st_2D_trpd_aver, sigma_d_2D_trpd, rho_d_2D_trpd, r_trpd, temp_2D_trpd, sigma_g_2D_trpd)

# TriPod monodisperse - max St
m_trpd_mono_max = pebbAcc(t, t_trpd, st_2D_trpd_max, sigma_d_2D_trpd, rho_d_2D_trpd, r_trpd, temp_2D_trpd, sigma_g_2D_trpd)


###################################
### Plot planetary growth track ###

plt.plot(t/(1e6*year), m_tp2/ME, color='red', label='tp2')
plt.plot(t/(1e6*year), m_dp_poly/ME, color='black', label='dp-poly')
plt.plot(t/(1e6*year), m_dp_mono_aver/ME, '--', color='black', label='dp-mono-aver')
plt.plot(t/(1e6*year), m_dp_mono_peak/ME, '-.', color='black', label='dp-mono-peak')
plt.plot(t/(1e6*year), m_trpd_poly/ME, color='blue', label='trpd-poly')
plt.plot(t/(1e6*year), m_trpd_mono_aver/ME, '--', color='blue', label='trpd-mono-aver')
plt.plot(t/(1e6*year), m_trpd_mono_max/ME, ':', color='blue', label='trpd-mono-peak')
plt.legend()
plt.xlim([pars.tcore/(year*1e6), pars.tend/(year*1e6)])
plt.yscale('log')
plt.xlabel('Time [Myr]')
plt.ylabel('Planetary mass [Mearth]')
#plt.show()

###################################
### Save planetary growth track ###

dist = pars.Rcore/au

np.save('pebbledata/tpebble.npy',t)
np.save(f'pebbledata/noMiso_mp_tp2_{dist:.1f}au.npy',m_tp2)
np.save(f'pebbledata/noMiso_mp_dp_poly_{dist:.1f}au.npy',m_dp_poly)
np.save(f'pebbledata/noMiso_mp_dp_mono_aver_{dist:.1f}au.npy',m_dp_mono_aver)
np.save(f'pebbledata/noMiso_mp_dp_mono_peak_{dist:.1f}au.npy',m_dp_mono_peak)
np.save(f'pebbledata/noMiso_mp_trpd_poly_{dist:.1f}au.npy',m_trpd_poly)
np.save(f'pebbledata/noMiso_mp_trpd_mono_aver_{dist:.1f}au.npy',m_trpd_mono_aver)
np.save(f'pebbledata/noMiso_mp_trpd_mono_max_{dist:.1f}au.npy',m_trpd_mono_max)

### Save some extra data ###
'''
ir = np.abs(r_tp2 - pars.Rcore).argmin()
np.save('pebbledata/t_tp2.npy',t_tp2)
np.save(f'pebbledata/st_tp2_{dist:.1f}au.npy',st_2D_tp2[:,ir])
np.save(f'pebbledata/sigma_d_tp2_{dist:.1f}au.npy',sigma_d_2D_tp2[:,ir])

ir = np.abs(r_dp - pars.Rcore).argmin()
np.save('pebbledata/t_dp.npy',t_dp)
np.save(f'pebbledata/st_peak_{dist:.1f}au.npy',st_2D_dp_peak[:,ir])
np.save(f'pebbledata/st_aver_{dist:.1f}au.npy',st_2D_dp_aver[:,ir])
np.save(f'pebbledata/sigma_d_dp_{dist:.1f}au.npy',sigma_d_2D_dp[:,ir])

ir = np.abs(r_trpd - pars.Rcore).argmin()
np.save('pebbledata/t_trpd.npy',t_trpd)
np.save(f'pebbledata/st_max_trpd_{dist:.1f}au.npy',st_2D_trpd_max[:,ir])
np.save(f'pebbledata/st_aver_trpd_{dist:.1f}au.npy',st_2D_trpd_aver[:,ir])
np.save(f'pebbledata/sigma_d_trpd_{dist:.1f}au.npy',sigma_d_2D_trpd[:,ir])
'''

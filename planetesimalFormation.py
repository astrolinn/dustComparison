'''
Calculate where and when the criteria for planetesimal formation
via the streaming instability are met
Dust evolution codes:
- two-pop-py2
- TriPoDPy
- DustPy
2 choices for representative St in DustPy & TriPoDPy:
- aver: Density-weighted average St
- peak: Peak of density-vs-St distribution
SI criteria:
- YG05: Youdin & Goodman (2005) - criteria for linear unstratified SI
- LI24: Lim et al. (2024) - 3D with forced turbulence, valid for limited St and alphaTurb
- LI25: Lim et al. (2025) - 2D without turbulence, valid in large St regime

Last modified: Jan 22, 2026
Author: Linn Eriksson
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy import constants as c
import sys
sys.path.insert(0, os.getcwd())

import inputFile as pars
from commonFunctions import (
    kepAngVel, soundSpeed, gasScaleHeight, st_number,
    pebbleScaleHeight, midplaneDensity 
)

year = 365.25*24*3600
au = c.au.cgs.value
kB = c.k_B.cgs.value
mH = c.u.cgs.value
G = c.G.cgs.value

path = "planetesimaldata"
exists = os.path.exists(path)
if not exists:
    os.mkdir(path)

################################
### Load dust evolution data ###

# DustPy
r_dp = np.load('files_dp/r.npy')
t_dp = np.load('files_dp/tdustev.npy')
sigma_g_2D_dp = np.load('files_dp/sigma_gas_2D.npy')
sigma_d_3D_dp = np.load('files_dp/sigma_dust_3D.npy')
sigma_d_2D_dp = sigma_d_3D_dp.sum(-1)
rho_g_2D_dp = np.load('files_dp/rho_gas_2D.npy')
rho_d_2D_dp = np.load('files_dp/rho_dust_3D.npy').sum(-1)
st_3D_dp = np.load('files_dp/st_3D.npy')

# TriPoDPy
r_trpd = np.load('files_trpd/r_trpd.npy')
t_trpd = np.load('files_trpd/t_trpd.npy')
sigma_g_2D_trpd = np.load('files_trpd/sigma_gas_2D.npy')
sigma_d_3D_trpd = np.load('files_trpd/Sigma_recon.npy')
sigma_d_2D_trpd = sigma_d_3D_trpd.sum(-1)
rho_g_2D_trpd = np.load('files_trpd/rho_gas_2D.npy')
rho_d_2D_trpd = np.load('files_trpd/rho_recon.npy').sum(-1)
st_3D_trpd = np.load('files_trpd/St_recon.npy')
st_2D_trpd_peak = np.load('files_trpd/st_3D.npy')[:,:,-1]

# two-pop-py2
r_tp2 = np.load('files_tp/r_tp.npy')
t_tp2 = np.load('files_tp/time_tp.npy')
temp_2D_tp2 = np.load('files_tp/temp_tp.npy')
sigma_g_2D_tp2 = np.load('files_tp/sigma_gas_tp.npy')
sigma_d_2D_tp2 = np.load('files_tp/sigma_dust_tp.npy')
size_2D_tp2 = np.load('files_tp/size_tp.npy')

####################
### Calculate St ###

# Convert from size to St for two-pop-py
st_2D_tp2 = st_number(r_tp2, temp_2D_tp2, sigma_g_2D_tp2, size_2D_tp2)

# Obtain representative St for DustPy and TriPoDPy

# Density-weighted average
st_2D_dp_aver = np.sum(st_3D_dp * sigma_d_3D_dp, axis=2) / np.sum(sigma_d_3D_dp, axis=2)
st_2D_trpd_aver = np.sum(st_3D_trpd * sigma_d_3D_trpd, axis=2) / np.sum(sigma_d_3D_trpd, axis=2)

# Peak of size distribution
ind = np.argmax(sigma_d_3D_dp, axis=2)
st_2D_dp_peak = np.take_along_axis(st_3D_dp, ind[..., np.newaxis], axis=2).squeeze(axis=2)

#####################
### Calculate rho ###

# Calculate rho_dust and rho_gas for two-pop-py
Omega = kepAngVel(r_tp2)
Cs = soundSpeed(temp_2D_tp2)
H = gasScaleHeight(Cs, Omega)
rho_g_2D_tp2 = midplaneDensity(sigma_g_2D_tp2, H)
Hpebb_tp2 = pebbleScaleHeight(r_tp2, temp_2D_tp2, st_2D_tp2)
rho_d_2D_tp2 = midplaneDensity(sigma_d_2D_tp2, Hpebb_tp2)

##############################
### Define the SI crtieria ###

def SI_Lim2024(St, alpha):
    """
    Calculates the critical dust-to-gas midplane
    density ratio for the streaming instability
    to develop according to eq.19 of Lim et al.
    (2024)
    """
    Z_crit = 10**( 0.15 * np.log10(alpha)**2 - 0.24 * np.log10(St) * np.log10(alpha)
        - 1.48 * np.log10(St) + 1.18 * np.log10(alpha) )
    return Z_crit

def SI_Lim2025(St):
    """
    Calculates the critical dust-to-gas surface
    density ratio for the streaming instability
    to develop according to eq.15 of Lim et al.
    (2025)
    """
    Z_crit = 10**( 0.10 * np.log10(St)**2 + 0.07 * np.log10(St) - 2.36 )
    return Z_crit

###################################################
### Function for checking if planetesimals form ###

def planForm(Z, Z_crit):
    """
    Check if the chosen SI criteria is reached
    """
    pf = np.array((Z > Z_crit), dtype=int)
    return pf

def planForm_YG05(rho_g, rho_d):
    """
    Check if epsilon>=1 is reached
    Criteria from linear stability analysis of 
    Youdin & Goodman (2005)
    """
    pf = np.array((rho_d/rho_g >= 1), dtype=int)
    return pf

################################################
### Check where planetesimals form
### The output are 2d arrays of shape (t,r) that
### contain ones (where the criteria are met)
### and zeros (where the criteria are not met)

# two-pop-py2
pf_tp2_SI24 = planForm(sigma_d_2D_tp2/sigma_g_2D_tp2, SI_Lim2024(st_2D_tp2, pars.alphaTurb))
pf_tp2_SI25 = planForm(sigma_d_2D_tp2/sigma_g_2D_tp2, SI_Lim2025(st_2D_tp2))
pf_tp2_YG05 = planForm_YG05(rho_g_2D_tp2, rho_d_2D_tp2)

# DustPy
pf_dp_YG05 = planForm_YG05(rho_g_2D_dp, rho_d_2D_dp)

pf_dp_aver_SI24 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2024(st_2D_dp_aver, pars.alphaTurb))
pf_dp_aver_SI25 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2025(st_2D_dp_aver))

pf_dp_peak_SI24 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2024(st_2D_dp_peak, pars.alphaTurb))
pf_dp_peak_SI25 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2025(st_2D_dp_peak))

# TriPoDPy
pf_trpd_YG05 = planForm_YG05(rho_g_2D_trpd, rho_d_2D_trpd)

pf_trpd_aver_SI24 = planForm(sigma_d_2D_trpd/sigma_g_2D_trpd, SI_Lim2024(st_2D_trpd_aver, pars.alphaTurb))
pf_trpd_aver_SI25 = planForm(sigma_d_2D_trpd/sigma_g_2D_trpd, SI_Lim2025(st_2D_trpd_aver))

pf_trpd_peak_SI24 = planForm(sigma_d_2D_trpd/sigma_g_2D_trpd, SI_Lim2024(st_2D_trpd_peak, pars.alphaTurb))
pf_trpd_peak_SI25 = planForm(sigma_d_2D_trpd/sigma_g_2D_trpd, SI_Lim2025(st_2D_trpd_peak))

#################
### Save data ###

np.save('planetesimaldata/t_tp2.npy',t_tp2)
np.save('planetesimaldata/r_tp2.npy',r_tp2)
np.save('planetesimaldata/t_dp.npy',t_dp)
np.save('planetesimaldata/r_dp.npy',r_dp)
np.save('planetesimaldata/t_trpd.npy',t_trpd)
np.save('planetesimaldata/r_trpd.npy',r_trpd)

np.save('planetesimaldata/pf_tp2_YG05.npy',pf_tp2_YG05)
np.save('planetesimaldata/pf_tp2_SI24.npy',pf_tp2_SI24)
np.save('planetesimaldata/pf_tp2_SI25.npy',pf_tp2_SI25)

np.save('planetesimaldata/pf_dp_YG05.npy',pf_dp_YG05)
np.save('planetesimaldata/pf_dp_aver_SI24.npy',pf_dp_aver_SI24)
np.save('planetesimaldata/pf_dp_peak_SI24.npy',pf_dp_peak_SI24)
np.save('planetesimaldata/pf_dp_aver_SI25.npy',pf_dp_aver_SI25)
np.save('planetesimaldata/pf_dp_peak_SI25.npy',pf_dp_peak_SI25)

np.save('planetesimaldata/pf_trpd_YG05.npy',pf_trpd_YG05)
np.save('planetesimaldata/pf_trpd_aver_SI24.npy',pf_trpd_aver_SI24)
np.save('planetesimaldata/pf_trpd_peak_SI24.npy',pf_trpd_peak_SI24)
np.save('planetesimaldata/pf_trpd_aver_SI25.npy',pf_trpd_aver_SI25)
np.save('planetesimaldata/pf_trpd_peak_SI25.npy',pf_trpd_peak_SI25)

#######################
### Some extra data ###

np.save('planetesimaldata/st_tp2.npy',st_2D_tp2)
np.save('planetesimaldata/st_dp_aver.npy',st_2D_dp_aver)
np.save('planetesimaldata/st_dp_peak.npy',st_2D_dp_peak)
np.save('planetesimaldata/st_trpd_aver.npy',st_2D_trpd_aver)
np.save('planetesimaldata/st_trpd_peak.npy',st_2D_trpd_peak)

np.save('planetesimaldata/Z_tp2.npy',sigma_d_2D_tp2/sigma_g_2D_tp2)
np.save('planetesimaldata/Z_dp.npy',sigma_d_2D_dp/sigma_g_2D_dp)
np.save('planetesimaldata/Z_trpd.npy',sigma_d_2D_trpd/sigma_g_2D_trpd)

#####################
### Outdated Plot ###
'''
# Contour lines 
plt.contour(r_tp2/au, t_tp2/(1e6*year), pf_tp2_SI25, levels=[0], colors='black', linewidths=1.4)
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_aver_SI25, levels=[0], colors='red', linewidths=1.4)
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_peak_SI25, levels=[0], colors='green', linewidths=1.4)
plt.contour(r_trpd/au, t_trpd/(1e6*year), pf_trpd_aver_SI25, levels=[0], colors='blue', linewidths=1.4)
plt.contour(r_trpd/au, t_trpd/(1e6*year), pf_trpd_peak_SI25, levels=[0], colors='orange', linewidths=1.4)
plt.contour(r_tp2/au, t_tp2/(1e6*year), pf_tp2_SI24, levels=[0], colors='black', linewidths=0.7)
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_aver_SI24, levels=[0], colors='red', linewidths=0.7)
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_peak_SI24, levels=[0], colors='green', linewidths=0.7)
plt.contour(r_trpd/au, t_trpd/(1e6*year), pf_trpd_aver_SI24, levels=[0], colors='blue', linewidths=0.7)
plt.contour(r_trpd/au, t_trpd/(1e6*year), pf_trpd_peak_SI24, levels=[0], colors='orange', linewidths=0.7)
# Create proxy artists
line1 = mlines.Line2D([], [], color='black', linewidth=1.4, label='tp2-SI25')
line2 = mlines.Line2D([], [], color='red', linewidth=1.4, label='dp-aver-SI25')
line3 = mlines.Line2D([], [], color='green', linewidth=1.4, label='dp-peak-SI25')
line5 = mlines.Line2D([], [], color='blue', linewidth=1.4, label='trpd-aver-SI25')
line6 = mlines.Line2D([], [], color='orange', linewidth=1.4, label='trpd-peak-SI25')
line7 = mlines.Line2D([], [], color='black', linewidth=0.7, label='tp2-SI24')
line8 = mlines.Line2D([], [], color='red', linewidth=0.7, label='dp-aver-SI24')
line9 = mlines.Line2D([], [], color='green', linewidth=0.7, label='dp-peak-SI24')
line11 = mlines.Line2D([], [], color='blue', linewidth=0.7, label='trpd-aver-SI24')
line12 = mlines.Line2D([], [], color='orange', linewidth=0.7, label='trpd-peak-SI24')
# Text
plt.legend(handles=[line1, line2, line3, line5, line6, line7, line8, line9, line11, line12], frameon=False, fontsize=11)
plt.xlabel('Semimajor axis [au]')
plt.ylabel('Time [Myr]')
plt.title('Planetesimal formation regions')
plt.xlim([1,200])
plt.ylim([0,3])
#plt.show()
'''


# Calculate where and when the criteria for planetesimal formation
# via the streaming instability are met, plot the results
# Dust evolution codes:
# - TwoPopPy2
# - DustPy
#   2 choices for representative St in DustPy:
#   - Density-weighted average
#   - Peak of density-vs-St distribution
# SI criteria:
# - Lim et al. (2024a)
# - Lim et al. (2024b)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from astropy import constants as c
import inputFile as pars
from commonFunctions import (
    st_number
)

year = 365.25*24*3600
au = c.au.cgs.value
kB = c.k_B.cgs.value
mH = c.u.cgs.value
G = c.G.cgs.value

################################
### Load dust evolution data ###

# DustPy
r_dp=np.load('files_dp/r.npy')
t_dp=np.load('files_dp/tdustev.npy')
sigma_g_2D_dp=np.load('files_dp/sigma_gas_2D.npy')
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

####################
### Calculate St ###

# Convert size to St for TwoPopPy2
st_2D_tp2 = st_number(r_tp2,temp_2D_tp2,sigma_g_2D_tp2,size_2D_tp2)

# Obtain representative St for DustPy

# Density-weighted average
st_2D_dp_aver = np.sum(st_3D_dp * sigma_d_3D_dp, axis=2) / np.sum(sigma_d_3D_dp, axis=2)
# Peak of size distribution
ind = np.argmax(sigma_d_3D_dp, axis=2)
st_2D_dp_peak = np.take_along_axis(st_3D_dp, ind[..., np.newaxis], axis=2).squeeze(axis=2)

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

######################################
### Check where planetesimals form ###

pf_tp2_SI24 = planForm(sigma_d_2D_tp2/sigma_g_2D_tp2, SI_Lim2024(st_2D_tp2, pars.alphaTurb))
pf_tp2_SI25 = planForm(sigma_d_2D_tp2/sigma_g_2D_tp2, SI_Lim2025(st_2D_tp2))

pf_dp_aver_SI24 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2024(st_2D_dp_aver, pars.alphaTurb))
pf_dp_aver_SI25 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2025(st_2D_dp_aver))

pf_dp_peak_SI24 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2024(st_2D_dp_peak, pars.alphaTurb))
pf_dp_peak_SI25 = planForm(sigma_d_2D_dp/sigma_g_2D_dp, SI_Lim2025(st_2D_dp_peak))

############
### Plot ###

# Contour lines 
plt.contour(r_tp2/au, t_tp2/(1e6*year), pf_tp2_SI25, levels=[0], colors='black', linewidths=1.2)
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_aver_SI25, levels=[0], colors='black', linewidths=1.2, linestyles='dashed')
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_peak_SI25, levels=[0], colors='black', linewidths=1.4, linestyles='dotted')
plt.contour(r_tp2/au, t_tp2/(1e6*year), pf_tp2_SI24, levels=[0], colors='red', linewidths=1.2)
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_aver_SI24, levels=[0], colors='red', linewidths=1.2, linestyles='dashed')
plt.contour(r_dp/au, t_dp/(1e6*year), pf_dp_peak_SI24, levels=[0], colors='red', linewidths=1.4, linestyles='dotted')
# Create proxy artists
line1 = mlines.Line2D([], [], color='black', linewidth=1.2, label='tp2-SI25')
line2 = mlines.Line2D([], [], color='black', linewidth=1.2, linestyle='dashed', label='dp-aver-SI25')
line3 = mlines.Line2D([], [], color='black', linewidth=1.4, linestyle='dotted', label='dp-peak-SI25')
line4 = mlines.Line2D([], [], color='red', linewidth=1.2, label='tp2-SI24')
line5 = mlines.Line2D([], [], color='red', linewidth=1.2, linestyle='dashed', label='dp-aver-SI24')
line6 = mlines.Line2D([], [], color='red', linewidth=1.4, linestyle='dotted', label='dp-aver-SI24')
# Text
plt.legend(handles=[line1, line2, line3, line4, line5, line6], frameon=False, fontsize=11)
plt.xlabel('Semimajor axis [au]')
plt.ylabel('Time [Myr]')
plt.title('Planetesimal formation regions')
plt.xlim([1,200])
plt.ylim([0,3])
plt.show()


import os
import numpy as np
from astropy import constants as c
from twopoppy2 import twopoppy2 as tp2

import inputFile as pars
from commonFunctions import (
    midplaneTemp
)
from viscAccDisc import viscAccDisc_grid

path = "files_tp"
exists = os.path.exists(path)
if not exists:
    os.mkdir(path)

#####################################################

mH = c.u.cgs.value

#####################################################

# Set up semimajor axis and time grid

ri = np.geomspace(pars.Rin_dust, pars.Redge_dust, pars.Rnr_dust+1)
ri_outer = np.concatenate([np.arange(pars.Rcoarse_int*(ri[-1]//pars.Rcoarse_int+1.), pars.Rgrid_out, pars.Rcoarse_int), [pars.Rgrid_out]])
ri = np.concatenate([ri, ri_outer])
t = np.linspace(pars.tstart, pars.tend, int(np.floor(pars.tend/pars.dt_dust)))

#####################################################

# Initialize twopoppy2
m = tp2.Twopoppy(grid=tp2.Grid(ri))
m.M_star = pars.Mstar
m.R_star = pars.Rstar
m.T_star = pars.Tstar
m.T_gas = midplaneTemp(m.r)
m.alpha_diff=pars.alphaTurb
m.alpha_gas=pars.alpha
m.alpha_turb=pars.alphaTurb
m.mu = pars.mu
m.rho_s = pars.rhop
Mdot_gas_0,sigma_gas_0 = viscAccDisc_grid(t[0],m.r)
m.sigma_g = sigma_gas_0
m.allowDriftingParticles = pars.allowDriftingParticles
m._a_0 = pars.dustMinSize
m._a_1 = pars.a_0 # Initial size (bad name choice...)
m._floor = 1e-25
m.sigma_d = pars.Z * sigma_gas_0
m.v_frag = pars.vfrag
m.snapshots = t
m.initialize()

# Run twopoppy2
m.run()

# Get results
r = m.r
time = m.data['time'][:,0]
temp = m.data['T_gas']
sigma_gas = m.data['sigma_g']
sigma_dust = m.data['sigma_d']
size = m.data['a_1']

# Save data arrays
np.save('files_tp/r_tp.npy',m.r)
np.save('files_tp/time_tp.npy',time)
np.save('files_tp/temp_tp.npy',temp)
np.save('files_tp/sigma_gas_tp.npy',sigma_gas)
np.save('files_tp/sigma_dust_tp.npy',sigma_dust)
np.save('files_tp/size_tp.npy',size)

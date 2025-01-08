import sys
import numpy as np
from astropy import constants as c
import inputFile as pars
from commonFunctions import (
    midplaneTemp
)
from viscAccDisc import viscAccDisc_grid

#### CHANGE THIS TO YOUR PATH ###
sys.path.append('/nobackup/leerikss/twopoppy2')
from twopoppy2 import twopoppy2 as tp2

#####################################################

mH = c.u.cgs.value
Z = 0.01

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
m.sigma_d = Z * sigma_gas_0
m.snapshots = t
m.v_frag = pars.vfrag
m.initialize()

# Run twopoppy2
m.run()

# Get results
r = m.r
time = m.data['time'][:,0]
sigma_gas = m.data['sigma_g']
sigma_dust = m.data['sigma_d']
size = m.data['a_1']

# Save data arrays
np.save('r_tp.npy',m.r)
np.save('time_tp.npy',time)
np.save('sigma_gas_tp.npy',sigma_gas)
np.save('sigma_dust_tp.npy',sigma_dust)
np.save('size_tp.npy',size)
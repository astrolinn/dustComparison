import sys
import numpy as np
from astropy import constants as c
from inputFile import (
    Mstar,Rstar,Tstar,rhop,alphaTurb,mu,alpha,
    Rin_dust,Redge_dust,Rnr_dust,tstart,tend,dt_dust,
    vfrag
)
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

ri = np.logspace(np.log10(Rin_dust),np.log10(Redge_dust),Rnr_dust)
t = np.linspace(tstart,tend,int(np.floor(tend/dt_dust)))

#####################################################

# Initialize twopoppy2
m = tp2.Twopoppy(grid=tp2.Grid(ri))
m.M_star = Mstar
m.R_star = Rstar
m.T_star = Tstar
m.T_gas = midplaneTemp(m.r)
m.alpha_diff=alphaTurb
m.alpha_gas=alpha
m.alpha_turb=alphaTurb
m.mu = mu
m.rho_s = rhop
Mdot_gas_0,sigma_gas_0 = viscAccDisc_grid(t[0],m.r)
m.sigma_g = sigma_gas_0
m.sigma_d = Z * sigma_gas_0
m.snapshots = t
m.v_frag = vfrag
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
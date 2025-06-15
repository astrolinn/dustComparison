import numpy as np
from astropy import constants as c
import dustpy
import tripod
from tripod import std

import inputFile as pars
from commonFunctions import  midplaneTemp
from viscAccDisc import viscAccDisc_grid

#####################################################

mH = c.u.cgs.value

#####################################################

# Set up semimajor axis and time grid

ri = np.geomspace(pars.Rin_dust, pars.Redge_dust, pars.Rnr_dust+1)
ri_outer = np.concatenate([np.arange(pars.Rcoarse_int*(ri[-1]//pars.Rcoarse_int+1.), pars.Rgrid_out, pars.Rcoarse_int), [pars.Rgrid_out]])
ri = np.concatenate([ri, ri_outer])
t = np.linspace(pars.tstart, pars.tend, int(np.floor(pars.tend/pars.dt_dust)))

#####################################################

sim = tripod.Simulation()

# Grid Configuration
sim.ini.dust.rhoMonomer = pars.rhop
sim.ini.grid.mmin = 4./3. * np.pi * sim.ini.dust.rhoMonomer * pars.dustMinSize**3
sim.grid.ri = ri
### Stellar Parameters
sim.ini.star.M = pars.Mstar
sim.ini.star.R = pars.Rstar
sim.ini.star.T = pars.Tstar
# Gas Parameters
sim.ini.gas.alpha = pars.alpha
sim.ini.gas.mu = pars.mu*mH
sim.ini.gas.SigmaExp = -pars.sigmaExp
sim.ini.gas.SigmaRc = pars.Rout
### Dust Parameters
sim.ini.dust.aIniMax = pars.a_0
sim.ini.dust.allowDriftingParticles = pars.allowDriftingParticles
sim.ini.dust.d2gRatio = pars.Z
sim.ini.dust.vFrag = pars.vfrag
### Initialize
sim.initialize()
### Different dust diffusivity
sim.dust.delta.rad[...] = pars.alphaTurb
sim.dust.delta.rad.updater = None
sim.dust.delta.turb[...] = pars.alphaTurb
sim.dust.delta.turb.updater = None
sim.dust.delta.vert[...] = pars.alphaTurb
sim.dust.delta.vert.updater = None
sim.dust.update()
### Set the initial surface densities and temperature structure
sim.gas.T[...] = midplaneTemp(sim.grid.r)
sim.gas.T.updater = None
sim.gas.update()
Mdot_gas_0, sigma_gas_0 = viscAccDisc_grid(t[0], sim.grid.r)
sim.gas.Sigma[...] = sigma_gas_0
sim.gas.update()
sim.dust.Sigma[...] = std.dust.Sigma_initial(sim)
sim.dust.update()
### Update and finalize all fields
sim.update()
### Time between saved snapshots
sim.t.snapshots = t
### Save statement
sim.writer.datadir = "data"
sim.writer.overwrite = True
### Run TriPod
sim.update()
sim.run()
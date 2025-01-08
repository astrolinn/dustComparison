from commonFunctions import  midplaneTemp
from dustpy import Simulation
from astropy import constants as c
from dustpy import std
import inputFile as pars
import matplotlib.pyplot as plt
import numpy as np
from viscAccDisc import viscAccDisc_grid

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


### Initialize dustPy
sim = Simulation()

### Grid Configuration
sim.ini.dust.rhoMonomer = pars.rhop
sim.ini.grid.Nmbpd = 7 # Default
sim.ini.grid.mmin = 4./3. * np.pi * sim.ini.dust.rhoMonomer * pars.dustMinSize**3
sim.ini.grid.mmax = 4./3. * np.pi * sim.ini.dust.rhoMonomer * pars.dustMaxSize**3
sim.grid.ri = ri
sim.makegrids()
### Stellar Parameters
sim.ini.star.M = pars.Mstar
sim.ini.star.R = pars.Rstar
sim.ini.star.T = pars.Tstar
# Gas Parameters
sim.ini.gas.alpha = pars.alpha
sim.ini.gas.gamma = 1.0 # Adiabatic index, set to 1 for isothermal
sim.ini.gas.mu = pars.mu*mH
sim.ini.gas.SigmaExp = -pars.sigmaExp
sim.ini.gas.SigmaRc = pars.Rout
### Dust Parameters
sim.ini.dust.aIniMax = pars.a_0
sim.ini.dust.allowDriftingParticles = pars.allowDriftingParticles
sim.ini.dust.d2gRatio = Z
sim.ini.dust.vfrag = pars.vfrag
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
sim.dust.Sigma[...] = std.dust.MRN_distribution(sim)
sim.dust.update()
### Update and finalize all fields
sim.update()
sim.integrator._finalize()
### Time between saved snapshots
sim.t.snapshots = t
### Lots of things that are not saved
sim.dust.kernel.save = False
sim.dust.v.rel.azi.save = False
sim.dust.v.rel.rad.save = False
sim.dust.v.rel.brown.save = False
sim.dust.v.rel.turb.save = False
sim.dust.v.rel.vert.save = False
sim.dust.Fi.adv.save = False
sim.dust.Fi.diff.save = False
sim.dust.coagulation.A.save = False
sim.dust.coagulation.eps.save = False
sim.dust.coagulation.phi.save = False
sim.dust.coagulation.lf_ind.save = False
sim.dust.coagulation.rm_ind.save = False
sim.dust.coagulation.stick.save = False
sim.dust.coagulation.stick_ind.save = False
sim.dust.p.stick.save = False
sim.dust.p.frag.save = False
sim.dust.v.rel.tot.save = False
### Save statement
sim.writer.datadir = "data"
sim.writer.overwrite = True
### Run dustPy
sim.update()
sim.run()
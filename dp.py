import numpy as np
from astropy import constants as c
from inputFile import (
    Mstar,Rstar,Tstar,rhop,alphaTurb,mu,alpha,sigmaExp,
    a_0,Rin_dust,Redge_dust,Rnr_dust,tstart,tend,dt_dust,
    vfrag,Rout,dustMinSize,dustMaxSize,allowDriftingParticles
)
from commonFunctions import (
    midplaneTemp
)
from viscAccDisc import viscAccDisc_grid

#####################################################

mH = c.u.cgs.value
Z = 0.01

#####################################################

# Set up semimajor axis and time grid

ri = np.logspace(np.log10(Rin_dust),np.log10(Redge_dust),Rnr_dust)
t = np.linspace(tstart,tend,int(np.floor(tend/dt_dust)))

#####################################################

import dustpy
from dustpy import hdf5writer
wrtr = hdf5writer()

### Initialize dustPy
sim = dustpy.Simulation()
### Stellar Parameters
sim.ini.star.M = Mstar
sim.ini.star.R = Rstar
sim.ini.star.T = Tstar
### Grid Configuration
sim.ini.grid.Nmbpd = 7 # Default
sim.ini.grid.mmin = 4./3. * np.pi * sim.ini.dust.rhoMonomer * dustMinSize**3
sim.ini.grid.mmax = 4./3. * np.pi * sim.ini.dust.rhoMonomer * dustMaxSize**3
sim.ini.grid.Nr = Rnr_dust
sim.ini.grid.rmin = Rin_dust
sim.ini.grid.rmax = Redge_dust
# Gas Parameters
sim.ini.gas.alpha = alpha
sim.ini.gas.gamma = 1.0 # Adiabatic index, set to 1 for isothermal
sim.ini.gas.mu = mu*mH
sim.ini.gas.SigmaExp = -sigmaExp
sim.ini.gas.SigmaRc = Rout
### Dust Parameters
sim.ini.dust.aIniMax = a_0
sim.ini.dust.allowDriftingParticles = allowDriftingParticles
sim.ini.dust.d2gRatio = Z
sim.ini.dust.rhoMonomer = rhop
sim.ini.dust.vfrag = vfrag
### Initialize
sim.initialize()
### Set the initial surface densities and temperature structure
Sigma_old = sim.gas.Sigma.copy()
Mdot_gas_0,sigma_gas_0 = viscAccDisc_grid(t[0],sim.grid.r)
sim.gas.Sigma = sigma_gas_0
sim.dust.Sigma *= (sim.gas.Sigma/Sigma_old)[:,None]
sim.gas.gamma = 1.0 # Adiabatic index, set to 1 for isothermal
sim.gas.T = midplaneTemp(sim.grid.r)
sim.gas.T.updater = None
### Time between saved snapshots
sim.t.snapshots = t
### Different dust diffusivity
sim.dust.delta.rad = alphaTurb
sim.dust.delta.rad.updater = None
sim.dust.delta.turb = alphaTurb
sim.dust.delta.turb.updater = None
sim.dust.delta.vert = alphaTurb
sim.dust.delta.vert.updater = None
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
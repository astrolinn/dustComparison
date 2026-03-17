import numpy as np
import astropy.constants as c
import astropy.units as u
from dustpy import Simulation
from dustpy import std

import inputFile as pars
from commonFunctions import  midplaneTemp, refine_radial_local, planet_profile

#####################################################

mH = c.u.cgs.value
year = 365.25*24*3600

#####################################################

# Set up semimajor axis and time grid

if(pars.refine_grid and pars.include_planet):
    ri = np.geomspace(pars.Rin_dust, pars.Redge_dust, pars.Rnr_dust+1)
    ri_outer = np.concatenate([np.arange(pars.Rcoarse_int*(ri[-1]//pars.Rcoarse_int+1.), pars.Rgrid_out, pars.Rcoarse_int), [pars.Rgrid_out]])
    ri = np.concatenate([ri, ri_outer])
    ri = refine_radial_local(ri, pars.Rp, num=5)
    t = np.linspace(pars.tstart, pars.tend, int(np.floor(pars.tend/pars.dt_dust)))
else:
    ri = np.geomspace(pars.Rin_dust, pars.Redge_dust, pars.Rnr_dust+1)
    #ri_outer = np.concatenate([np.arange(pars.Rcoarse_int*(ri[-1]//pars.Rcoarse_int+1.), pars.Rgrid_out, pars.Rcoarse_int), [pars.Rgrid_out]])
    #ri = np.concatenate([ri, ri_outer])
    t = np.geomspace(100*year, 2e5*year, 201)
    t = np.insert(t, 0, 0.)
    #t = np.linspace(pars.tstart, pars.tend, int(np.floor(pars.tend/pars.dt_dust)))

############ Setup temperature and semimajor axis array #############

M_star = 1.*c.M_sun.cgs.value
T_star = 4397. 
R_star = 3.096*c.R_sun.cgs.value
L_star = 4.*np.pi*c.sigma_sb.cgs.value*T_star**4.*R_star**2.

pSig  = -1.
rchar = 50*c.au.cgs.value
Mdot  = (4e-8*u.M_sun/u.year).cgs.value
alpha = 5e-3

T_c    = (0.5 * 0.05 * L_star / (4.*np.pi * rchar**2 * c.sigma_sb.cgs.value))**0.25
cs_c   = np.sqrt(c.k_B.cgs.value*T_c/(2.34*c.m_p.cgs.value))
nu_c   = alpha * cs_c**2/np.sqrt(c.G.cgs.value*M_star/rchar**3)

T_au    = (0.5 * 0.05 * L_star / (4.*np.pi * c.au.cgs.value**2 * c.sigma_sb.cgs.value))**0.25
Sig_1au = Mdot/(3*np.pi*nu_c*(c.au.cgs.value/rchar)**(-pSig)) * np.exp(-(c.au.cgs.value/rchar)**(2+pSig))
print(T_au, Sig_1au)

const_T = T_au / (1*c.au.cgs.value**(-pars.tempExp))
const_Sig = Sig_1au / (1*c.au.cgs.value**(-pars.sigmaExp))
print(const_T, const_Sig)

#####################################################

### Initialize dustPy
sim = Simulation()

### Grid Configuration
sim.ini.dust.rhoMonomer = pars.rhop
sim.ini.grid.Nmbpd = pars.Nmbpd
sim.ini.grid.mmin = 4./3. * np.pi * sim.ini.dust.rhoMonomer * pars.dustMinSize**3
sim.ini.grid.mmax = 4./3. * np.pi * sim.ini.dust.rhoMonomer * pars.dustMaxSize**3
sim.grid.ri = ri
sim.makegrids()
### Stellar Parameters
sim.ini.star.M = M_star
sim.ini.star.R = R_star
sim.ini.star.T = T_star
# Gas Parameters
sim.ini.gas.alpha = 0
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
T_ic = const_T * sim.grid.r**(-pars.tempExp)
sim.gas.T[...] = T_ic
sim.gas.T.updater = None
sim.gas.update()
Sig_ic = const_Sig * sim.grid.r**(-pars.sigmaExp)
sim.gas.Sigma[...] = Sig_ic
sim.gas.update()
sim.dust.Sigma[...] = std.dust.MRN_distribution(sim)
sim.dust.update()
### Turn of gas evolution
sim.gas.nu[:] = 0.
sim.gas.nu.updater = None
sim.gas.v.rad = 0.0
sim.gas.v.rad.updater = None
sim.gas.v.visc = 0.0
sim.gas.v.visc.updater = None
### Setup planet if included
if pars.include_planet:
    sim.gas.alpha /= planet_profile(pars.profile, sim.grid.r, pars.Rp, pars.Mp_Mstar, sim.gas.Hp/sim.grid.r, pars.alpha)
    sim.gas.alpha.updater = None
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

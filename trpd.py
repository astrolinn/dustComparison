import numpy as np
from astropy import constants as c
import os
import dustpy
import tripod
from tripod import std
from tripod import hdf5writer
from tripod.utils.size_distribution import get_q
wrtr = hdf5writer()

import inputFile as pars
from commonFunctions import  midplaneTemp
from viscAccDisc import viscAccDisc_grid

path = "files_trpd"
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
# sim.run()

#####################################################

### Save the data ###

# Standard output
data = wrtr.read.all()
np.save('files_trpd/r_trpd.npy',data.grid.r[0,:])
np.save('files_trpd/t_trpd.npy',data.t)
np.save('files_trpd/sigma_gas_2D.npy',data.gas.Sigma)
np.save('files_trpd/sigma_dust_3D.npy',data.dust.Sigma)
np.save('files_trpd/st_3D.npy',data.dust.St) 
np.save('files_trpd/size_3D.npy',data.dust.a)
np.save('files_trpd/temp_2D.npy',data.gas.T)
np.save('files_trpd/rho_gas_2D.npy',data.gas.rho)
np.save('files_trpd/rho_dust_3D.npy',data.dust.rho)
np.save('files_trpd/vrad_dust_3D.npy',data.dust.v.rad)
np.save('files_trpd/vrad_dust_3D.npy',data.dust.v.rad)

# Reconstruct the size distribution in exactly the same was as DustPy
log_mmin = np.log10(4./3. * np.pi * pars.rhop * pars.dustMinSize**3)
log_mmax = np.log10(4./3. * np.pi * pars.rhop * pars.dustMaxSize**3)
decades = np.ceil(log_mmax - log_mmin)
Nm = int(decades * pars.Nmbpd) + 1
m = np.logspace(log_mmin, log_mmax, num=Nm, base=10)
A = np.mean(m[1:]/m[:-1])
mi = np.append(2./(A+1.)*m, A*2./(A+1.)*m[-1])

ai = (mi/(4./3. * np.pi * pars.rhop))**(1/3)
a  = (m/(4./3. * np.pi * pars.rhop))**(1/3)

q = get_q(data.dust.Sigma, data.dust.s.min, data.dust.s.max)

mmax = 4./3.*np.pi*pars.rhop*data.dust.s.max**3.
qmass = (-q+4.)/3.
q4_mask = data.dust.Sigma[...,0][:,:,None]!=data.dust.Sigma[...,1][:,:,None]

# Calculate size distribution where q!=4
Sigma_recon  = np.where(np.logical_and(q4_mask, mi[None,None,1:]<=mmax[:,:,None]), 
                       (mi[None,None,1:]**qmass[:,:,None] - mi[None,None,:-1]**qmass[:,:,None]), 0)
Sigma_recon  = np.where(np.logical_and(q4_mask, np.logical_and(mi[None,None,:-1]<mmax[:,:,None], mi[None,None,1:]>mmax[:,:,None])), 
                                       (mmax[:,:,None]**qmass[:,:,None] - mi[None,None,:-1]**qmass[:,:,None]), Sigma_recon)
Sigma_recon *= np.where(q4_mask, 
                        (data.dust.Sigma.sum(-1)[:,:,None]/np.ma.masked_where(~q4_mask, (mmax[:,:,None]**(qmass[:,:,None]) - mi.min()**(qmass[:,:,None])))), 1)

# Calculate size distribution where q==4
Sigma_recon  = np.where(np.logical_and(~q4_mask, mi[None,None,1:]<=mmax[:,:,None]), 
                        np.log(mi[None,None,1:]/mi[None,None,:-1]), Sigma_recon)
Sigma_recon  = np.where(np.logical_and(~q4_mask, np.logical_and(mi[None,None,:-1]<mmax[:,:,None], 
                        mi[None,None,1:]>mmax[:,:,None])), np.log(mmax[:,:,None]/mi[None,None,:-1]), Sigma_recon)
Sigma_recon *= np.where(~q4_mask, 
                        (np.ones_like(Sigma_recon)*data.dust.Sigma.sum(-1)[:,:,None]/np.log(mmax[:,:,None]/mi.min())), 1)

a_3d = np.ones_like(data.gas.mfp)[..., None] * a
condition = a_3d < 2.25 * data.gas.mfp[:,:,None]
St_recon = np.empty_like(a_3d)
Sigma_3d = np.broadcast_to(data.gas.Sigma[:, :, None], a_3d.shape)
mfp_3d   = np.broadcast_to(data.gas.mfp[:, :, None], a_3d.shape)
St_recon[condition] = 0.5 * np.pi * a_3d[condition] * pars.rhop / Sigma_3d[condition]
St_recon[~condition] = 2.0/9.0 * np.pi * a_3d[~condition]**2 * pars.rhop / (mfp_3d[~condition] * Sigma_3d[~condition])

H_recon = data.gas.Hp[:,:,None] / np.sqrt(1.0 + St_recon / data.dust.delta.vert[:,:,None])
rho_recon = Sigma_recon / (np.sqrt(2 * np.pi) * H_recon)

np.save('files_trpd/Sigma_recon.npy', Sigma_recon)
np.save('files_trpd/St_recon.npy', St_recon)
np.save('files_trpd/rho_recon.npy', rho_recon)

'''
#####################################################
# Some reconstruction of the size distribution with provided functions
from tripod.utils.read_data import read_data
data_1 = read_data(data="data", Na=100)
Sigma_recon_1 = data_1.dust.Sigma_recon
St_recon_1 = data_1.dust.St_recon
H_recon_1 = data.gas.Hp[:,:,None] / np.sqrt(1.0 + St_recon_1 / data.dust.delta.vert[:,:,None])
rho_recon_1 = Sigma_recon_1 / (np.sqrt(2 * np.pi) * H_recon_1)

np.save('files_trpd/Sigma_recon_1.npy', Sigma_recon_1)
np.save('files_trpd/St_recon_1.npy', St_recon_1)
np.save('files_trpd/rho_recon_1.npy', rho_recon_1)
#####################################################
'''


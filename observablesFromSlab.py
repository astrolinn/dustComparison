# Dipsys required -> get from https://github.com/NIcolas-Kaufmann/dipsy
import dipsy 
from dustpy import hdf5writer
import numpy as np


opac = dipsy.Opacity('default_opacities_smooth.npz')
#Wavelengths to observe at (in cm) Nlam
lam_obs = [0.089, 0.1, 0.31]


writer = hdf5writer()
datadirs = ["data"]
sim_type = ["tripod"]

for i in range(len(datadirs)):
    writer.datadir = datadirs[i]
    path = datadirs[i]+"/"

    if(sim_type[i] == "tripod"):
        # Read data using dipsy
        data = dipsy.dipsy_functions.read_tripod_data(path)
        obs = dipsy.get_all_observables(data,opac,lam_obs,amax=True,scattering=True)
    elif(sim_type[i] == "dustpy"):
        # Read data using dustpy
        data = dipsy.dipsy_functions.read_dustpy_data(path)
        obs = dipsy.get_all_observables(data,opac,lam_obs,amax=False,scattering=True)
    else:
        raise ValueError("Unknown simulation type")
    
    # save the data calculated fron the slab model
    np.savez_compressed(f"{datadirs[i]}/observables.npz",
                        time = data.time, 
                        Intes = data.I_nu, # radial intensity profiles (Nt xNr x Nlam)
                        flux = obs.flux,   # fluxes at different wavelengths (Nt x Nlam)
                        Rmm = obs.rf,      # flux radius 60% flux (Nt x Nlam)
                        rCO = obs.r_CO     # CO radius -< as defined in Trapman et al. (2023) (Nt))
    )

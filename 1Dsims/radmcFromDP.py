from dustpy import hdf5writer
import numpy as np
from dustpylib.radtrans import radmc3d

import re
import os
import glob

from dustpy import hdf5writer
import numpy as np
from dustpylib.radtrans import radmc3d
import dustpy.constants as c
from astropy import units as u

def extract_number(f):
    s = re.findall("\d+$",f[:-5])
    return (int(s[0]) if s else -1,f)

datadirs = ["data"]
radmcdirs = ["files_radmc_dp"]

writer = hdf5writer()

for i in range(len(datadirs)):
    writer.datadir = datadirs[i]

    ''' ##################################### '''
    ''' Determine Last Data File in Directory '''
    ''' ##################################### '''
    path = datadirs[i]+"/"
    dirs = np.array(glob.glob(path+"*"))
    dirs[:] = [s[len(path):] for s in dirs]
    lastfile = max(dirs,key=extract_number)
    Nlast = int(re.findall(r'\d+', lastfile[:-5])[0])
    print(Nlast)
    ''' ##################################### '''

    data = writer.read.output(Nlast)
    data.star.M = data.star.M[0]
    data.star.R = data.star.R[0]
    data.star.L = data.star.L[0]
    data.star.T = data.star.T[0]

    a    = data.dust.a
    amax = 1.5*np.where(np.cumsum(data.dust.Sigma, axis=-1)<0.999*np.sum(data.dust.Sigma, axis=-1)[:,None]*np.ones_like(a), a, 0, axis=1)[:-1].max()

    rt = radmc3d.Model(data)
    rt.phii_grid = np.array([0., 2.*np.pi])
    rt.lam_grid  = np.geomspace(1e-5,1,150)
    rt.ai_grid   = np.geomspace(rt.a_dust_.min(), amax, 17)
    
    rt.radmc3d_options["nphot"] = 10_000_000
    rt.radmc3d_options["nphot_scat"] = 1_000_000
    rt.radmc3d_options["nphot_spec"] = 10_000
    
    rt.radmc3d_options["mc_scat_maxtauabs"] = 5.
    rt.radmc3d_options["dust_2daniso_nphi"] = 60
    rt.datadir = radmcdirs[i]
    
    rt.write_files()

    
    

from dustpy import hdf5writer
import numpy as np
from dustpylib.radtrans import radmc3d
from tripod.utils import get_size_distribution
import dustpy.std.dust as dp_dust_f

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

# Convert dustpy data to format suitable for radmc3d i.e. Dustpy like
def convert_to_dustpy(data):
    a, a_i , sig_da = get_size_distribution(data.dust.Sigma.sum(-1),data.dust.s.max,q = data.dust.qrec,agrid_min = data.dust.s.min.min(),na=100)

    del data.dust.a
    data.dust.a = a * np.ones_like(sig_da)
    del data.dust.Sigma
    data.dust.Sigma = sig_da
    _rho_s = data.dust.rhos[0,0]
    _fill = data.dust.fill[0,0]
    del data.dust.rhos
    data.dust.rhos =  _rho_s * np.ones_like(sig_da)
    del data.dust.fill
    data.dust.fill = _fill * np.ones_like(sig_da)
    del data.dust.St
    data.dust.St = dp_dust_f.St_Epstein_StokesI(data)
    del data.dust.H
    data.dust.H = dp_dust_f.H(data)
    del data.dust.rho
    data.dust.rho = dp_dust_f.rho_midplane(data)

    return data

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

    data = convert_to_dustpy(data)

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

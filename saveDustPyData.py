#########################################
### Save data from dustPy as numpy arrays

import numpy as np
import os
import re
import dustpy
from dustpy import hdf5writer
wrtr = hdf5writer()

path = "files_dp"
exists = os.path.exists(path)
if not exists:
    os.mkdir(path)

########################################
# Find nr of saved snapshots from dustPy
def findnrsave():
    directory = 'data/'
    files = os.listdir(directory)
    pattern = re.compile(r'data(\d+)\.hdf5')
    indices = []
    for filename in files:
        match = pattern.match(filename)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        max_index = max(indices)
    return max_index
########################################

lent = findnrsave()+1
data = wrtr.read.output(0)
nrfluids = len(data.dust.Sigma[0,:])
r = data.grid.r
tdustev = np.zeros((lent))
sigma_gas_2D = np.zeros((lent,len(r)))
sigma_dust_3D = np.zeros((lent,len(r),nrfluids))
rho_gas_2D = np.zeros((lent,len(r)))
rho_dust_3D = np.zeros((lent,len(r),nrfluids))
vrad_dust_3D = np.zeros((lent,len(r),nrfluids))
temp_2D = np.zeros((lent,len(r)))
st_3D = np.zeros((lent,len(r),nrfluids))
size_3D = np.zeros((lent,len(r),nrfluids))

for i in range(lent):
    data = wrtr.read.output(i)
    tdustev[i] = data.t
    sigma_gas_2D[i,:] = data.gas.Sigma
    sigma_dust_3D[i,:,:] = data.dust.Sigma
    rho_gas_2D[i,:] = data.gas.rho
    rho_dust_3D[i,:,:] = data.dust.rho
    vrad_dust_3D[i,:,:] = data.dust.v.rad
    temp_2D[i,:] = data.gas.T
    st_3D[i,:,:] = data.dust.St
    size_3D[i,:,:] = data.dust.a

np.save('files_dp/r.npy',r)
np.save('files_dp/tdustev.npy',tdustev)
np.save('files_dp/sigma_gas_2D.npy',sigma_gas_2D)
np.save('files_dp/sigma_dust_3D.npy',sigma_dust_3D)
np.save('files_dp/rho_gas_2D.npy',rho_gas_2D)
np.save('files_dp/rho_dust_3D.npy',rho_dust_3D)
np.save('files_dp/vrad_dust_3D.npy',vrad_dust_3D)
np.save('files_dp/temp_2D.npy',temp_2D)
np.save('files_dp/st_3D.npy',st_3D)
np.save('files_dp/size_3D.npy',size_3D)
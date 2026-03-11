# 1D radial simulations
Scripts for setting up, running and post-processing the 1D radial simulations of Eriksson et al. (2026). 

### Codes
- DustPy
- TriPoDPy
- two-pop-py
---
## Setting up and running the dust evolution codes
All input variables that can be modified are found in the script inputFile.py
#### DustPy
- Install the DustPy code: https://github.com/stammler/dustpy
- Run the script dp.py (the nominal simulation takes one to a few days)
- If needed, the simulation can be restarted by running the script restartDustpy.py
- Save the necessary data by running the script saveDustPyData.py, which saves data as numpy arrays and stores them inside the directory files_dp
#### TriPoDPy (python implementation of TriPoD)
- Install the TriPoDPy code: https://github.com/tripod-code/tripodpy (need correct version of dustpylib)
- Run the script trpd.py (the nominal simulation takes a few minutes), necessary data is stored as numpy arrays and stores inside the directory files_trpd
#### two-pop-py
- Install the two-pop-py code, make sure to use our fork: https://github.com/astrolinn/twopoppy2_fork (note: twopoppy2 does not work with the newest python version, use version 3.11 or older)
- Run the script tp.py (the nominal simulation takes a few seconds), necessary data is stored as numpy arrays and stores inside the directory files_tp
---
## Planetesimal formation
Requires that data from the 3 dust evolution codes exist inside the directory (files_dp, files_trpd, files_tp)
The script 
- Run the script planetesimalFormation.py, data is stores as numpy arrays inside the directory planetesimaldata
- 



Required modules: numpy, astropy, os


### Caluclating Observables
#### Radio image  
Run `create_radmc.py` to create the radmc models for the DustPy, TriPoDPy and twopoppy simulations, then run radmc3d (`radmc3d mctherm && radmc3d image lambdarange 890 3100 nlam 2 sizeau 100 npixx 512 npixy 512`). Lastly, make the plots with the `radio_im.py` script.
##### Requirements
  - dustpylib
  - radmc3d
#### Fluxes and radii
The script `observables.py` creates the plots for the radii and fluxes
##### Requirements
  - dipsy (available at: https://github.com/NIcolas-Kaufmann/dipsy)

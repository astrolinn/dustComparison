This repository contains all scripts used for initializing, 
running and post-processing the simulations in the dust evolution 
comparison study by Eriksson et al. (2026). 

Required modules:
  numpy,
  astropy,
  os,
  dustpy,
  twopoppy2_fork (https://github.com/astrolinn/twopoppy2_fork)

To set up twopoppy2_fork: 
clone the git repo and execute
  pip install -e .
inside the local repository
 
NOTE: twopoppy2 does not work with the newest python version, 
use version 3.11 or older

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

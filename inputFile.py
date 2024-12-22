# Contains all input parameters for the simulation
# Note: cgs units

import numpy as np
from astropy import constants as c
#########################################

year = 365.25*24*3600
au = c.au.cgs.value
ME = c.M_earth.cgs.value
Msolar = c.M_sun.cgs.value

#########################################

# Time-grid
tstart = 0          # Start point of time-grid
tend = 3e6*year     # End point of time-grid, also end of planet formation
dt_dust = 3e3*year  # Frequency at which outputs are saved from dust evolution

# Stellar parameters
Mstar= 1.0 * c.M_sun.cgs.value
Rstar  = 3.096 * c.R_sun.cgs.value
Tstar = 4397

# Protoplanetary disk parameters
Mdot0 = 1.0e-8 * Msolar/year
                    # Initial disk accretion rate (determines initial
                    # Gas surface density)
Rout = 50.0*au      # Critical cut-off radius of surface density
Z = 0.01
tempProfile = "passIrr"
                    # Choose between CG97 or passIrr
Tconst = 150.0      # Temperature at 1au in model CG97
tempExp = 0.5       # dlnT/dlnR
sigmaExp = 1.0      # dlnSigma/dlnR at 1au
HExp = 1.25         # dlnH/dlnR
alpha = 5.0e-3      # alpha governing viscous evolution
alphaTurb = 0.0001
mu = 2.34           # Mean molecular weight
sig_h = 2e-15       # Collisional cross-section of hydrogen atom

# Dust evolution parameters
dustMinSize = 5e-5  # Minimum dust size on dust-grid in dustPy
dustMaxSize = 100   # Maximum dust size on dust-grid in dustPy
allowDriftingParticles = True
                    # Do or don't allow initially drifting particles in dustpy
Rin_dust = 1.0*au   # Inner edge of semimajor axis grid used for dust ev.
Redge_dust = 200*au # Outer edge of semimajor axis grid used for dust ev.
Rnr_dust = 150      # Nr of grid-points on semimajor axis grid used for dust ev.
Rcoarse_int = 50*au # Coarse radial grid spacing in outer disk
Rgrid_out = 1000*au # Outer edge of grid
a_0 = 1.0e-4        # Initial size of dust grains
vfrag = 100.0       # Fragmentation velocity
rhop = 1.0          # Internal density of dust grains

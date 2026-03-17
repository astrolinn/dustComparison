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
tstart = 100          # Start point of time-grid
tend = 2e5*year     # End point of time-grid, also end of planet formation
dt_dust = 2.5e2*year  # Frequency at which outputs are saved from dust evolution

# Stellar parameters
Mstar= 1.0*c.M_sun.cgs.value
Rstar  = 3.096*c.R_sun.cgs.value
Tstar = 4397

# Protoplanetary disk parameters
Mdot0 = 4.e-8*Msolar/year
                    # Initial disk accretion rate (determines initial
                    # gas surface density)
Rout = 50.0*au      # Critical cut-off radius of surface density
Z = 0.01            # Initial dust-to-gas ratio
tempProfile = "passIrr"
                    # Choose between CG97 or passIrr
Tconst = 150.0      # Temperature at 1au in model CG97
tempExp = 0.5       # dlnT/dlnR
sigmaExp = 1.0      # dlnSigma/dlnR at 1au
HExp = 1.25         # dlnH/dlnR
alpha = 5.0e-3      # alpha governing viscous evolution
alphaTurb = 0.001  # alpha governing turbulent diffusion
mu = 2.34           # Mean molecular weight
sig_h = 2e-15       # Collisional cross-section of hydrogen atom

# planet gap parameters
include_planet = False
profile        = 'Duffell' # either "Duffell" or "Kanagawa" using the dustpylib routines
refine_grid    = True      # whether to refine the grid around the planet, using the dustpylib routine
Mp_Mstar       = 1e-3
Rp             = 10.*au

# Dust evolution parameters
dustMinSize = 5.3028708485114374e-05  # Minimum dust size on dust-grid in dustPy and TriPod
dustMaxSize = 47.562712261767125    # Maximum dust size on dust-grid in dustPy
allowDriftingParticles = False
                    # Do or don't allow initially drifting particles
Nmbpd = 7           # Number of dust fluids per decade in dustpy
Rin_dust = 5.0*au   # Inner edge of semimajor axis grid used for dust ev.
Redge_dust = 50*au # Outer edge of semimajor axis grid used for dust ev.
Rnr_dust = 128      # Nr of grid-points on semimajor axis grid used for dust ev.
Rcoarse_int = 50*au # Coarse radial grid spacing in outer disk
Rgrid_out = 1000*au # Outer edge of grid
a_0 = 1e-4          # Initial size of dust grains
vfrag = 100.0       # Fragmentation velocity
rhop = 1.0          # Internal density of dust grains

# Pebble accretion parameters
Mcore = 0.1*ME      # Initial core mass
Rcore = 5*au        # Semimajor axis of core
tcore = 1e5*year    # Time of core formation, start of pebble accretion
dt_pebbAcc = 5e2*year
                    # Time-step used in planet growth calculation
fracmax_pebbAcc = 0.999   
                    # Remove high end of St tail in pebble accretion
lim3d_pebbAcc = 10  # If (Racc/(2*Hp))**2 > lim3d, switch to using the
                    # original eq for pebble accretion in the 3d limit

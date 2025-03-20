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

Run the dustPy simulation using: python dp.py

Run the twopoppy2 simulation using: python tp.py

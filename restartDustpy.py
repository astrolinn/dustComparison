# How to restart a DustPy simulation

from dustpy import readdump
sim_restart = readdump("data_dp/frame.dmp")
sim_restart.run()

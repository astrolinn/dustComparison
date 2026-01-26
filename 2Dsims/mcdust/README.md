These are the setup files of the two mcdust runs show in Eriksson+2026.

Steps to reproduce the mcdust simulations

1) Clone the mcdust repository

git clone git@github.com:vicky1997/mcdust.git

2) copy the directories alpha1.e-4 and alpha1.e-3 to mcdust/setups/

cp -r alpha1.e-3 alpha1.e-4 /foo/mcdust/setups

3) load the necessary modules (GCC, HDF5-serial) as per your cluster/computer rules.

make SETUP_FILE=alpha1.e-4

4) set the number of threads you want to use for your program

export OMP_NUM_THREADS=num_threads

5) Run the program

./alpha1.e-4 setups/alpha1.e-4/setup.par

To run the alpha1.e-3 simulation. Replace the occurences of alpha1.e-4 with alpha.1e-3 in steps 3 and 5.

The documentation of mcdust can be accessed with:

[https://mcdust.readthedocs.io/](https://mcdust.readthedocs.io/)

The github repository for mcdust can be accessed with

[https://github.com/vicky1997/mcdust](https://github.com/vicky1997/mcdust)

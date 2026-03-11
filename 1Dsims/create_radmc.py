# %%
import shutil
import numpy as np 
from dustpy import hdf5writer
import dustpy.constants as c
from dustpylib.radtrans import radmc3d
from dustpylib.radtrans import slab
from tripod.utils import get_size_distribution
import dustpy.std.dust as dp_dust_f
import matplotlib.pyplot as plt
import shutil
from copy import deepcopy



# Set up which data to read and whic snapshots -> expect Dustpy, tripod and Two-Pop-Py to be in the subfolders "data_dp", "data_trpd" and "files_tp" respectively, with the same naming convention for the files.
sim_dir = "/path/to/simulations/"
# Snapshots to be converted to radmc models ->  corresponds to 2 Myr with the way the simulations are set up, but can be changed to any snapshot that is available in the data folders.
# after running this script run:
# radmc3d mctherm && radmc3d image lambdarange 890 3100 nlam 2 sizeau 100 npixx 512 npixy 512
#to crate the images
it = 666
it_tp = it//333 -1
output_dir = sim_dir

# %%
def plot_model(model, spec=-1):
    """
    Function plots the RADMC-3D model.
    
    Parameters
    ----------
    model : namespace
        The RADMC-3D model data
    spec : integer, optional, default : -1
        Particle species to be plotted. If -1, the total
        dust densities are plotted.
    """
    width = 6.
    height = width/1.3

    if spec==-1:
        rho = np.maximum(np.hstack((model.rho[:, :1, 0, :].sum(-1), model.rho[:, :, 0, :].sum(-1))), 1.e-100)
        T = np.hstack((model.T[:, :1, :, 0], model.T[:, :, :, 0])).mean(-1)
        a_rho = "total"
        a_T = "$a$ = {:.2e} cm".format(model.grid.a[0])
    else:
        rho = np.maximum(np.hstack((model.rho[:, :1, 0, spec], model.rho[:, :, 0, spec])), 1.e-100)
        T = np.hstack((model.T[:, :1, :, spec], model.T[:, :, :, spec])).mean(-1)
        a_rho = "$a$ = {:.2e} cm".format(model.grid.a[spec])
        a_T = "$a$ = {:.2e} cm".format(model.grid.a[spec])
                         
    rho = np.hstack((rho, np.flip(rho[:, 1:, ...], 1)))
    T = np.hstack((T, np.flip(T[:, 1:, ...], 1)))

    theta = np.hstack((model.grid.theta[0]-(model.grid.theta[1]-model.grid.theta[0]), model.grid.theta))
    theta = np.hstack((theta, theta[1:]+np.pi))
    lev_rho_max = np.ceil(np.log10(rho.max()))
    levels_rho = np.arange(lev_rho_max-6, lev_rho_max+1, 1.)

    fig, ax = plt.subplots(ncols=2, figsize=(2*width, height), subplot_kw={"projection": "polar"})

    p0 = ax[0].contourf(theta, model.grid.r/c.au, np.log10(rho), levels=levels_rho, extend="both", cmap="viridis")
    ax[0].contourf(-theta, model.grid.r/c.au, np.log10(rho), levels=levels_rho, extend="both", cmap="viridis")
    ax[0].set_theta_zero_location("N")
    ax[0].set_rlim(0)
    ax[0].set_rscale("symlog")
    ax[0].set_rgrids([1., 10., 100.], angle=-45)
    ax[0].tick_params(axis='y', colors='white')
    ax[0].set_xticks([])
    ax[0].plot(0., 0., "*", color="C3", markeredgewidth=0, markersize=12)
    cbar0 = plt.colorbar(p0)
    cbar0.set_ticks(cbar0.get_ticks())
    cbar0.set_ticklabels(["$10^{{{:d}}}$".format(int(t)) for t in cbar0.get_ticks()])
    cbar0.set_label(r"$\rho_\mathrm{dust}$ [g/cm³]")
    ax[0].set_title("Dust density, {}".format(a_rho))

    p1 = ax[1].contourf(theta, model.grid.r/c.au, T, extend="both", cmap="coolwarm")
    ax[1].contourf(-theta, model.grid.r/c.au, T, extend="both", cmap="coolwarm")
    ax[1].set_theta_zero_location("N")
    ax[1].set_rlim(0)
    ax[1].set_rscale("symlog")
    ax[1].set_rgrids([1., 10., 100.], angle=-45)
    ax[1].tick_params(axis='y', colors='black')
    ax[1].set_xticks([])
    ax[1].plot(0., 0., "*", color="C3", markeredgewidth=0, markersize=12)
    cbar1 = plt.colorbar(p1)
    cbar1.set_ticks(cbar1.get_ticks())
    cbar1.set_label(r"$T$ [K]")
    ax[1].set_title("Dust temperature, {}".format(a_T))

    fig.tight_layout()

# %%
from collections import namedtuple
writer = hdf5writer()


writer.datadir = sim_dir + "/data_trpd/"
data_trpd = writer.read.output(it)

writer.datadir = sim_dir + "/data_dp/"
data_dp = writer.read.output(it)

def read_tp(path,amin=5e-5):
    r = np.load(path + 'r_tp.npy')
    shape_tot = (3,len(r))
    sig_d = np.load(path + 'sigma_dust_tp.npy').reshape(shape_tot)
    sig_g = np.load(path + 'sigma_gas_tp.npy').reshape(shape_tot)
    amax = np.load(path + 'size_tp.npy').reshape(shape_tot)
    T = np.load(path + 'temp_tp.npy').reshape(shape_tot)
    q = np.load(path + 'q_tp.npy').reshape(shape_tot)


    df_dp = namedtuple('DustpyResult', [
        'r', 'a_max', 'a', 'a_mean', 'sig_d', 'sig_g', "q",'time', 'T', 'M_disk'
    ])

    return df_dp(
        r=r, 
        a_max=amax, 
        a=None, 
        a_mean=np.zeros_like(sig_g), 
        sig_d=sig_d, 
        sig_g=sig_g, 
        time=np.arange(1, 4) * c.year * 1e6, 
        T=T, 
        q = q,
        M_disk = np.arange(1, 4) * c.year * 1e6
    )

data_tp = read_tp(sim_dir + "/files_tp/")
data_tp_frame = deepcopy(data_trpd)


# %%
def convert_trpd_to_dustpy(data):

    a, a_i , sig_da = get_size_distribution(data.dust.Sigma.sum(-1),data.dust.s.max,q = abs(data.dust.qrec),agrid_min = data.dust.s.min.min(),na=100)

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

def convert_tp_to_dustpy(data,frame,it):

    a, a_i , sig_da = get_size_distribution(data.sig_d[it,:],data.a_max[it,:],q = data.q[it,:],agrid_min = frame.dust.s.min.min(),na=100)

    del frame.dust.a
    frame.dust.a = a * np.ones_like(sig_da)
    del frame.dust.Sigma
    frame.dust.Sigma = sig_da
    _rho_s = frame.dust.rhos[0,0]
    _fill = frame.dust.fill[0,0]
    del frame.dust.rhos
    frame.dust.rhos =  _rho_s * np.ones_like(sig_da)
    del frame.dust.fill
    frame.dust.fill = _fill * np.ones_like(sig_da)
    del frame.dust.St
    frame.dust.St = dp_dust_f.St_Epstein_StokesI(frame)
    del frame.dust.H
    frame.dust.H = dp_dust_f.H(frame)
    del frame.dust.rho
    frame.dust.rho = dp_dust_f.rho_midplane(frame)
    return frame

# %%
input_dp = data_dp
input_trpd = convert_trpd_to_dustpy(data_trpd)
input_tp = convert_tp_to_dustpy(data_tp, data_tp_frame,it_tp)


# %%

for inpt,dir in zip([input_dp, input_trpd, input_tp], ["dp","trpd","tp"]):
    rt = radmc3d.Model(inpt)
    rt.phii_grid = np.array([0., 2.*np.pi])
    rt.ai_grid = np.geomspace(1e-5,1., 17)
    rt.radmc3d_options["nphot"] = 10_000_000
    rt.radmc3d_options["nphot_scat"] = 1_000_000
    rt.radmc3d_options["nphot_spec"] = 10_000
    rt.radmc3d_options["dust_2daniso_nphi"] = 60
    rt.radmc3d_options["mc_scat_maxtauabs"] = 5.
    rt.datadir = output_dir + f"/radmc3d_{dir}/"
    rt.write_files()
    model = radmc3d.read_model(datadir=rt.datadir)
    plot_model(model, spec=-1)
    plt.suptitle(f"Model from {dir}")
    plt.show()

    



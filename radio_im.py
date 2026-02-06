# %%
import numpy as np 
import dustpy.constants as c
from dustpylib.radtrans import radmc3d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Simulation direcory you chose with create_radmc.py
sim_dir = "/project/ag-birnstiel/N.Kaufmann/linn_paper/new_nodrift/param_adpF_obsdata/Nom"

im_list = []
for im_source in ["dp","trpd","tp"]:
    im_dir =  sim_dir + f"/radmc3d_{im_source}/"
    image_radio = radmc3d.read_image(im_dir+"image.out")
    im_list.append(image_radio)



ax_fsize = 16
f, axs = plt.subplots(ncols=3, figsize=(18,6))
Imax = 0.5*im_list[-1]["I"].max()
mag = np.floor(np.log10(Imax))
logmax = np.ceil(Imax * 10**(-mag))
ma_range = 3
title = ["DustPy", "TriPoDPy", "two-pop-py"]
for nc in range(3):
    levels = np.logspace(np.log10(logmax * 10**(mag-ma_range)), np.log10(logmax * 10**mag), 100)
    axs[nc].set_aspect(1)
    x, y = im_list[nc]["x"], im_list[nc]["y"]
    p =axs[nc].contourf(x/c.au, y/c.au, im_list[nc]["I"][..., 0].T, cmap="inferno", levels=levels, norm=LogNorm(vmin=logmax*10**(mag-ma_range), vmax=logmax * 10**mag), extend="both")
    axs[nc].set_title(title[nc],fontsize=ax_fsize)
    axs[nc].set_xlabel(r"$X\,\left[\mathrm{au}\right]$",fontsize=ax_fsize)
    if nc==0:
        axs[nc].set_ylabel(r"$Y\,\left[\mathrm{au}\right]$",fontsize=ax_fsize)
        axs[nc].text(0.05, 0.95, r"$\lambda = {:.2f}\,\mathrm{{mm}}$".format(im_list[nc]["lambda"][0]*10.), color="white", fontsize=ax_fsize, transform=axs[nc].transAxes, verticalalignment='top')
#f.subplots_adjust(bottom=0.2)
pos0 = axs[0].get_position()
pos2 = axs[-1].get_position()
left, right = pos0.x0, pos2.x1
cbar_ax = f.add_axes([left, -0.04, right-left, 0.05])
cb = f.colorbar(p, cax=cbar_ax, orientation='horizontal')
cb.set_ticks(np.logspace(mag-ma_range, mag, ma_range+1)[1:])
cb.set_label(r"Intensity [erg/s/cm²/Hz/ster]",fontsize=ax_fsize)
#plt.tight_layout()
f.savefig("Radmc_fluxes_1Mj.png")

# %%

import matplotlib.cm as cm
import matplotlib.colors as colors

ax_fsize = 16
f, axs = plt.subplots(ncols=3, figsize=(18,6),sharey=True)
Imax = 0.5*im_list[-1]["I"].max()
mag = np.floor(np.log10(Imax))
logmax = np.ceil(Imax * 10**(-mag))
ma_range = 3
title = ["DustPy", "TriPoDPy", "two-pop-py"]
for nc in range(3)[:1]:
    levels = np.logspace(np.log10(logmax * 10**(mag-ma_range)), np.log10(logmax * 10**mag), 100)
    x, y = im_list[nc]["x"], im_list[nc]["y"]
    p1 =axs[nc].contourf(x/c.au, y/c.au, im_list[nc]["I"][..., 0].T, cmap="inferno", levels=levels, norm=LogNorm(vmin=logmax*10**(mag-ma_range), vmax=logmax * 10**mag), extend="both")
    axs[nc].set_title(title[nc],fontsize=ax_fsize)
    axs[nc].set_xlabel(r"$X\,\left[\mathrm{au}\right]$",fontsize=ax_fsize)
    if nc==0:
        axs[nc].set_ylabel(r"$Y\,\left[\mathrm{au}\right]$",fontsize=ax_fsize)
        axs[nc].text(0.05, 0.95, r"$\lambda = {:.2f}\,\mathrm{{mm}}$".format(im_list[nc]["lambda"][0]*10.), color="white", fontsize=ax_fsize, transform=axs[nc].transAxes, verticalalignment='top')


for nc in range(3)[1:]:
    residuals = im_list[nc]["I"][..., 0]/ im_list[0]["I"][..., 0]
    p =axs[nc].pcolormesh(x/c.au, y/c.au, residuals.T, norm=colors.LogNorm(vmax= 1e1, vmin=1e-1), cmap="bwr")
    axs[nc].set_title(title[nc],fontsize=ax_fsize)
    axs[nc].set_xlabel(r"$X\,\left[\mathrm{au}\right]$",fontsize=ax_fsize)

#f.subplots_adjust(bottom=0.2)
pos0 = axs[0].get_position()
pos2 = axs[-1].get_position()
left, right = pos0.x0, pos2.x1
cbar_ax = f.add_axes([left, -0.15, right-left, 0.1])
cb = f.colorbar(p, cax=cbar_ax, orientation='horizontal')
cb.set_label("Flux ratio to DustPy model",fontsize=ax_fsize)
pos0 = axs[0].get_position()
cbar2_ax = f.add_axes([pos0.x0 - 0.145, pos0.y0, 0.02, pos0.y1 - pos0.y0])
cb2 = f.colorbar(p1, cax=cbar2_ax, orientation='vertical')
cb2.ax.yaxis.set_ticks_position('left')
cb2.ax.yaxis.set_label_position('left')
cb2.set_ticks(np.logspace(mag-ma_range, mag, ma_range+1)[1:])
cb2.set_label(r"Intensity [erg/s/cm²/Hz/ster]",fontsize=ax_fsize)
plt.tight_layout()
plt.savefig("relative_fluxes_1Mj.png")



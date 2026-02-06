import numpy as np 
# the repo is public ->https://github.com/birnstiel/dipsy.git
import dipsy 
import matplotlib.pyplot as plt
import os 
import dustpy.constants as c
from collections import namedtuple
opac = dipsy.Opacity('default_opacities_smooth.npz')
lam_obs = [0.089, 0.1, 0.31]


#path where the simulation data is saved -> paths may vary depending on setup
#each sim is at mainpath + simulations_top[i] -> which contains the data directories for dustpy, tripod and two-pop models
mainpath = "./new_nodrift/param_adpF_obsdata"


# Simulation names
simulations_top = ["Nom", 
               "MS_0.05", "MS_0.1", "MS_0.5", "MS_0.9", "MS_1.3",
               "M0_8e-10", "M0_8e-9", "M0_8e-8",
               "Rc_10", "Rc_25", "Rc_100", "Rc_250",
               "Z_1e-4", "Z_1e-3", "Z_5e-3", "Z_5e-2", "Z_1e-1",
               "at_1e-5", "at_5e-5", "at_5e-4", "at_1e-3",] 

simulations_bottom = [
               "vf_1e1", "vf_5e1",  "vf_5e2", "vf_1e3", "vf_2.5e3",
               "rp_0.1", "rp_0.25", "rp_4", "rp_10", "../param_adpT_obsdata/Nom", "../../PlanetGap_obs/Mplanet/Mp01Mjup","../../PlanetGap_obs/Mplanet/Mp03Mjup","../../PlanetGap_obs/Mplanet/Mp1Mjup","../../PlanetGap_obs/Mplanet/Mp2Mjup",
               "../../PlanetGap_obs/alpha/000_1","../../PlanetGap_obs/alpha/000_5","../../PlanetGap_obs/alpha/005_0","../../PlanetGap_obs/alpha/010_0",
               "../../PlanetGap_obs/vfrag/0010","../../PlanetGap_obs/vfrag/0050","../../PlanetGap_obs/vfrag/0500"]




times = np.arange(1,4)*1e6*c.year

def read_dp(path,amin=5e-5):
    r = np.load(path + 'r.npy')
    shape_dust = np.load(path + 'sigma_dust_1.npy').shape
    shape_tot = (3,shape_dust[0],shape_dust[1])
    sig_d = np.zeros(shape_tot)
    sig_g = np.zeros((3,shape_dust[0]))
    T = np.zeros((3,shape_dust[0]))
    for i,k in enumerate(range(1,4)):
        sig_d[i,...] = np.load(path + f'sigma_dust_{k}.npy')
        sig_g[i,:] = np.load(path + f'sigma_gas_{k}.npy')
        T[i,:]= np.load(path + f'temp_{k}.npy')

    size = np.load(path + 'size_1.npy')[0,:]

    df_dp = namedtuple('DustpyResult', [
        'r', 'a_max', 'a', 'a_mean', 'sig_d', 'sig_da', 'sig_g', 'time', 'T', 'M_disk'
    ])
    i_min = np.argmin(np.abs(size - amin))

    return df_dp(
        r=r, 
        a_max=np.zeros_like(sig_g), 
        a=size[i_min:], 
        a_mean=np.zeros_like(sig_g), 
        sig_d=sig_d.sum(-1), 
        sig_da=sig_d[:,:,i_min:], 
        sig_g=sig_g, 
        time=np.arange(1, 4) * c.year * 1e6, 
        T=T, 
        M_disk = np.arange(1, 4) * c.year * 1e6
    )

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



obs_tri = []
obs_dp = []
obs_tp2 = []
for i, sim in enumerate(simulations_top + simulations_bottom):
    print(f"Processing simulation {i+1}/{len(simulations_top + simulations_bottom)}: {sim}")
    if sim == "a0_1e-5":
        continue
    data_tri = dipsy.dipsy_functions.read_tripod_data(f"{mainpath}/{sim}/data_trpd/")
    obs = dipsy.dipsy_functions.get_all_observables(data_tri,opac=opac,lam=lam_obs)
    obs.alpha = np.log(obs.flux[:,2]/obs.flux[:,0])/np.log(lam_obs[2]/lam_obs[0])
    
    obs_tri.append(obs)
    data_tp = dipsy.read_dustpy_data(f"{mainpath}/{sim}/data_dp/")
    obs = dipsy.dipsy_functions.get_all_observables(data_tp, opac,lam_obs,amax=False,scattering=True)
    obs.alpha = np.log(obs.flux[:,2]/obs.flux[:,0])/np.log(lam_obs[2]/lam_obs[0])
    obs_dp.append(obs)
    data_tp2 = read_tp(f"{mainpath}/{sim}/files_tp/")
    obs = dipsy.dipsy_functions.get_all_observables(data_tp2, opac,lam_obs,amax=True,scattering=True)
    obs.alpha = np.log(obs.flux[:,2]/obs.flux[:,0])/np.log(lam_obs[2]/lam_obs[0])
    obs_tp2.append(obs)



# %%
import matplotlib
import matplotlib as mpl
###############
# Style change.
mpl.rc("axes", labelsize="x-large")
mpl.rc("xtick", labelsize="x-large")
mpl.rc("ytick", labelsize="x-large")

# Set up figure.
params = {
    'image.origin': 'lower',
    'image.interpolation': 'none',
    'image.cmap': 'viridis',
    'axes.grid': False,
    'savefig.dpi': 200,  # to adjust notebook inline plot size
    'axes.labelsize': 12, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'font.size': 12, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': [5.25, 6.92],
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'figure.autolayout':False,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'xtick.minor.visible':False,
    'ytick.minor.visible':True,
    'xtick.minor.top':False,
    'ytick.minor.right':True,
    'xtick.top': False,
    'ytick.right': True}
matplotlib.rcParams.update(params)
##############



# Simulation names
simulations_top = ["Nom", 
               "MS_0.05", "MS_0.1", "MS_0.5", "MS_0.9", "MS_1.3",
               "M0_8e-10", "M0_8e-9", "M0_8e-8",
               "Rc_10", "Rc_25", "Rc_100", "Rc_250",
               "Z_1e-4", "Z_1e-3", "Z_5e-3", "Z_5e-2", "Z_1e-1",
               "at_1e-5", "at_5e-5", "at_5e-4", "at_1e-3",] 

simulations_bottom = [
               "vf_1e1", "vf_5e1",  "vf_5e2", "vf_1e3", "vf_2.5e3",
               "rp_0.1", "rp_0.25", "rp_4", "rp_10", "../param_adpT_obsdata/Nom", "../../PlanetGap_obs/Mplanet/Mp01Mjup","../../PlanetGap_obs/Mplanet/Mp03Mjup","../../PlanetGap_obs/Mplanet/Mp1Mjup","../../PlanetGap_obs/Mplanet/Mp2Mjup",
               "../../PlanetGap_obs/alpha/000_1","../../PlanetGap_obs/alpha/000_5","../../PlanetGap_obs/alpha/005_0","../../PlanetGap_obs/alpha/010_0",
               "../../PlanetGap_obs/vfrag/0010","../../PlanetGap_obs/vfrag/0050","../../PlanetGap_obs/vfrag/0500"]

textlabels_top = [
    "Nom", 
    r"$0.05M_{\ast}$", r"$0.1M_{\ast}$", r"$0.5M_{\ast}$", r"$0.9M_{\ast}$", r"$1.3M_{\ast}$",
    r"$0.02\dot{M}_0$", r"$0.2\dot{M}_0$", r"2$\dot{M}_0$",
    r"$0.2r_{\rm c}$", r"$0.5r_{\rm c}$", r"$2r_{\rm c}$", r"$5r_{\rm c}$",
    r"$0.01Z$", r"$0.1Z$", r"$0.5Z$", r"$5Z$", r"$10Z$",
    r"$0.1\alpha_t$", r"$0.5\alpha_t$", r"$5\alpha_t$", r"$10\alpha_t$"
]

textlabels_bottom = [
    r"$0.1v_{\rm frag}$", r"$0.5v_{\rm frag}$", r"$5v_{\rm frag}$", r"$10v_{\rm frag}$", r"$25v_{\rm frag}$",
    r"$0.1\rho_{\bullet}$", r"$0.25\rho_{\bullet}$", r"$4\rho_{\bullet}$", r"$10\rho_{\bullet}$","adp=F",
    r"$0.1M_{\rm jup}$",r"$0.3M_{\rm jup}$",r"$1M_{\rm jup}$",r"$2M_{\rm jup}$",
    r"$1M_{\rm jup}, 0.1\alpha_t$", r"$1M_{\rm jup}, 0.5\alpha_t$",r"$1M_{\rm jup}, 5\alpha_t$", r"$1M_{\rm jup}, 10\alpha_t$",
    r"$1M_{\rm jup}, 0.1v_{\rm frag}$", r"$1M_{\rm jup}, 0.5v_{\rm frag}$",r"$1M_{\rm jup}, 5v_{\rm frag}$"
]


# %%
# Number of simulations to loop over
num_sims_top = len(simulations_top)
num_sims_bottom = len(simulations_bottom)
print(num_sims_top,num_sims_bottom,num_sims_bottom+num_sims_top,len(obs_dp),len(obs_tp2),len(obs_tri))

# Initialize density array with NaNs
masses_twopop_top = np.full((num_sims_top, 3), np.nan)  # Fill with NaN by default
masses_tripod_top = np.full((num_sims_top, 3), np.nan)  # Fill with NaN by default
masses_twopop_bottom = np.full((num_sims_bottom, 3), np.nan)  # Fill with NaN by default
masses_tripod_bottom = np.full((num_sims_bottom, 3), np.nan)  # Fill with NaN by default

# Dust mass ratios
for i in range(num_sims_top):
    try:

    
        # Compute density ratios
        for j in range(3):
            # Check if the time index exists in the arrays
            masses_twopop_top[i, j] = obs_tp2[i].flux[j,1] / obs_dp[i].flux[j,1]
            masses_tripod_top[i, j] =  obs_tri[i].flux[j,1] / obs_dp[i].flux[j,1]

    except FileNotFoundError:
        print(f"Warning: Missing data for {simulations_top[i]}")
        continue  # Skip this simulation if files are missing
# Dust mass ratios
for k,i in enumerate(range(num_sims_top, num_sims_top + num_sims_bottom)):
    try:
        # Compute density ratios
        for j in range(3):
            # Check if the time index exists in the arrays
            masses_twopop_bottom[k, j] = obs_tp2[i].flux[j,1] / obs_dp[i].flux[j,1]
            masses_tripod_bottom[k, j] =  obs_tri[i].flux[j,1] / obs_dp[i].flux[j,1]

    except FileNotFoundError:
        print(f"Warning: Missing data for {simulations_top[i]}")
        continue  # Skip this simulation if files are missing


# %%
#########################

# Time labels and corresponding colors (sunset scheme)
time_labels = ["1 Myr", "2 Myr", "3 Myr"]
colors = ['#ffcc66', '#ff6600', '#cc0000']  # Light Yellow M-b^f^r Orange M-b^f^r Deep Red
colors_trpd = ['#99ffcc', '#3399ff', '#003366']  # Light Mint → Sky Blue → Deep Navy
markers = ['o', 's', '^']  # Different markers for extra distinction

###################################################
# Create figure
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.7), sharey=True)
plt.subplots_adjust(hspace=0.2)

#####################
### First subplot ###
#####################

ax = axes[0]

# Add gray band from y=0.5 to y=2
ax.axhspan(0.5, 2, color='gray', alpha=0.3, zorder=0)

# Scatter plot
for j in range(3):  # Loop over time steps (1Myr, 2Myr, 3Myr)
    ax.scatter(range(len(simulations_top)), masses_twopop_top[:, j], color=colors[j],
               marker=markers[j]) #, label=time_labels[j])
    ax.scatter(range(len(simulations_top)), masses_tripod_top[:, j], edgecolor=colors_trpd[j],
               facecolor='none', marker=markers[j])

# Set axis limits
ax.set_ylim(0.1, 7)
ax.set_xlim(-0.5,len(simulations_top)-0.5)

# Set x-axis
ax.set_xticks(range(len(simulations_top)))
ax.set_xticklabels(textlabels_top, rotation=90)

# Set y-axis (log scale)
ax.set_yscale('log')
ax.set_ylabel(r"$\frac{F_{mm}}{F_{mm}(DustPy)}$")

# First legend for time steps
for i in range(3):
    ax.scatter(-1, -1, color='black', marker=markers[i], label=time_labels[i])
time_legend = ax.legend(frameon=False, fontsize=11, loc='upper right')  # Adjust location>
ax.add_artist(time_legend)  # Ensure it remains on the plot

# Create handles for second legend
filled_marker = plt.Line2D([], [], color='#ff6600', marker='o', linestyle='None', markersize=6, label="TwoPop")
empty_marker = plt.Line2D([], [], color='#3399ff', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label="TriPod")
# Second legend for drifting particles (placed at top-left)
ax.legend(handles=[filled_marker, empty_marker], frameon=False, fontsize=11, loc='center', bbox_to_anchor=(0.82,0.877))

######################
### Second subplot ###
######################

ax = axes[1]

# Add gray band from y=0.5 to y=2
ax.axhspan(0.5, 2, color='gray', alpha=0.3, zorder=0)

# Scatter plot
for j in range(3):  # Loop over time steps (1Myr, 2Myr, 3Myr)
    ax.scatter(range(len(simulations_bottom)), masses_twopop_bottom[:, j], color=colors[j],
               marker=markers[j]) #, label=time_labels_bottom)
    ax.scatter(range(len(simulations_bottom)), masses_tripod_bottom[:, j], edgecolor=colors_trpd[j],
               facecolor='none', marker=markers[j])

# Set axis limits
ax.set_ylim(0.1, 7)
ax.set_xlim(-0.5,len(simulations_bottom)-0.5)

# Set x-axis
ax.set_xticks(range(len(simulations_bottom)))
ax.set_xticklabels(textlabels_bottom, rotation=90)

# Set y-axis (log scale)
ax.set_yscale('log')
ax.set_ylabel(r"$\frac{F_{mm}}{F_{mm}(DustPy)}$")

# Split to show planet
ax.plot([9.5,9.5],[0.1,7],':',color='black',linewidth=1.5)
ax.text(9.55,5,r"$\rightarrow$ planetary gap")

#######

plt.tight_layout()
plt.savefig(f"./F_mm.pdf")

# %%
# Number of simulations to loop over
num_sims_top = len(simulations_top)
num_sims_bottom = len(simulations_bottom)

# Initialize density array with NaNs
masses_twopop_top = np.full((num_sims_top, 3), np.nan)  # Fill with NaN by default
masses_tripod_top = np.full((num_sims_top, 3), np.nan)  # Fill with NaN by default
masses_twopop_bottom = np.full((num_sims_bottom, 3), np.nan)  # Fill with NaN by default
masses_tripod_bottom = np.full((num_sims_bottom, 3), np.nan)  # Fill with NaN by default

# Dust mass ratios
for i in range(num_sims_top):
    try:

    
        # Compute density ratios
        for j in range(3):
            # Check if the time index exists in the arrays
            masses_twopop_top[i, j] = obs_tp2[i].rf[j,1] / obs_dp[i].rf[j,1]
            masses_tripod_top[i, j] =  obs_tri[i].rf[j,1] / obs_dp[i].rf[j,1]

    except FileNotFoundError:
        print(f"Warning: Missing data for {simulations_top[i]}")
        continue  # Skip this simulation if files are missing
# Dust mass ratios
for k,i in enumerate(range(num_sims_top, num_sims_top + num_sims_bottom)):
    try:
        # Compute density ratios
        for j in range(3):
            # Check if the time index exists in the arrays
            masses_twopop_bottom[k, j] = obs_tp2[i].rf[j,1] / obs_dp[i].rf[j,1]
            masses_tripod_bottom[k, j] =  obs_tri[i].rf[j,1] / obs_dp[i].rf[j,1]

    except FileNotFoundError:
        print(f"Warning: Missing data for {simulations_top[i]}")
        continue  # Skip this simulation if files are missing


# %%
#########################

# Time labels and corresponding colors (sunset scheme)
time_labels = ["1 Myr", "2 Myr", "3 Myr"]
colors = ['#ffcc66', '#ff6600', '#cc0000']  # Light Yellow M-b^f^r Orange M-b^f^r Deep Red
colors_trpd = ['#99ffcc', '#3399ff', '#003366']  # Light Mint → Sky Blue → Deep Navy
markers = ['o', 's', '^']  # Different markers for extra distinction

###################################################
# Create figure
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(11, 7.7), sharey=True)
plt.subplots_adjust(hspace=0.2)

#####################
### First subplot ###
#####################

ax = axes[0]

# Add gray band from y=0.5 to y=2
ax.axhspan(0.5, 2, color='gray', alpha=0.3, zorder=0)

# Scatter plot
for j in range(3):  # Loop over time steps (1Myr, 2Myr, 3Myr)
    ax.scatter(range(len(simulations_top)), masses_twopop_top[:, j], color=colors[j],
               marker=markers[j]) #, label=time_labels[j])
    ax.scatter(range(len(simulations_top)), masses_tripod_top[:, j], edgecolor=colors_trpd[j],
               facecolor='none', marker=markers[j])

# Set axis limits
ax.set_ylim(0.1, 7)
ax.set_xlim(-0.5,len(simulations_top)-0.5)

# Set x-axis
ax.set_xticks(range(len(simulations_top)))
ax.set_xticklabels(textlabels_top, rotation=90)

# Set y-axis (log scale)
ax.set_yscale('log')
ax.set_ylabel(r"$\frac{R_{mm} (68\%)}{R_{mm}(DustPy)}$")

# First legend for time steps
for i in range(3):
    ax.scatter(-1, -1, color='black', marker=markers[i], label=time_labels[i])
time_legend = ax.legend(frameon=False, fontsize=11, loc='upper right')  # Adjust location>
ax.add_artist(time_legend)  # Ensure it remains on the plot

# Create handles for second legend
filled_marker = plt.Line2D([], [], color='#ff6600', marker='o', linestyle='None', markersize=6, label="TwoPop")
empty_marker = plt.Line2D([], [], color='#3399ff', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label="TriPod")
# Second legend for drifting particles (placed at top-left)
ax.legend(handles=[filled_marker, empty_marker], frameon=False, fontsize=11, loc='center', bbox_to_anchor=(0.82,0.877))

######################
### Second subplot ###
######################

ax = axes[1]

# Add gray band from y=0.5 to y=2
ax.axhspan(0.5, 2, color='gray', alpha=0.3, zorder=0)

# Scatter plot
for j in range(3):  # Loop over time steps (1Myr, 2Myr, 3Myr)
    ax.scatter(range(len(simulations_bottom)), masses_twopop_bottom[:, j], color=colors[j],
               marker=markers[j]) #, label=time_labels_bottom)
    ax.scatter(range(len(simulations_bottom)), masses_tripod_bottom[:, j], edgecolor=colors_trpd[j],
               facecolor='none', marker=markers[j])

# Set axis limits
ax.set_ylim(0.1, 7)
ax.set_xlim(-0.5,len(simulations_bottom)-0.5)

# Set x-axis
ax.set_xticks(range(len(simulations_bottom)))
ax.set_xticklabels(textlabels_bottom, rotation=90)

# Split to show planet
ax.plot([9.5,9.5],[0.1,7],':',color='black',linewidth=1.5)
ax.text(9.55,5,r"$\rightarrow$ planetary gap")

# Set y-axis (log scale)
ax.set_yscale('log')
ax.set_ylabel(r"$\frac{R_{mm}(68\%)}{R_{mm}(DustPy)}$")

#######

plt.tight_layout()
plt.savefig(f"./R_mm.pdf")

# %%
# Number of simulations to loop over
num_sims_top = len(simulations_top)
num_sims_bottom = len(simulations_bottom)

# Initialize density array with NaNs
masses_twopop_top = np.full((num_sims_top, 3), np.nan)  # Fill with NaN by default
masses_tripod_top = np.full((num_sims_top, 3), np.nan)  # Fill with NaN by default
masses_twopop_bottom = np.full((num_sims_bottom, 3), np.nan)  # Fill with NaN by default
masses_tripod_bottom = np.full((num_sims_bottom, 3), np.nan)  # Fill with NaN by default
masses_dp_top = np.full((num_sims_top, 3), np.nan)  # Fill with NaN by default
masses_dp_bottom = np.full((num_sims_bottom, 3), np.nan)  # Fill with NaN by default

# Dust mass ratios
for i in range(num_sims_top):
    try:

    
        # Compute density ratios
        for j in range(3):
            # Check if the time index exists in the arrays
            masses_twopop_top[i, j] = obs_tp2[i].alpha[j]
            masses_tripod_top[i, j] =  obs_tri[i].alpha[j]
            masses_dp_top[i, j] = obs_dp[i].alpha[j]
    except FileNotFoundError:
        print(f"Warning: Missing data for {simulations_top[i]}")
        continue  # Skip this simulation if files are missing
# Dust mass ratios
for k,i in enumerate(range(num_sims_top, num_sims_top + num_sims_bottom)):
    try:
        # Compute density ratios
        for j in range(3):
            # Check if the time index exists in the arrays
            masses_twopop_bottom[k, j] = obs_tp2[i].alpha[j]
            masses_tripod_bottom[k, j] = obs_tri[i].alpha[j]
            masses_dp_bottom[k, j] = obs_dp[i].alpha[j]

    except FileNotFoundError:
        print(f"Warning: Missing data for {simulations_top[i]}")
        continue  # Skip this simulation if files are missing


# %%
#########################

# Time labels and corresponding colors (sunset scheme)
time_labels = ["1 Myr", "2 Myr", "3 Myr"]
colors = ['#ffcc66', '#ff6600', '#cc0000']  # Light Yellow M-b^f^r Orange M-b^f^r Deep Red
colors_trpd = ['#99ffcc', '#3399ff', '#003366']  # Light Mint → Sky Blue → Deep Navy
markers = ['o', 's', '^']  # Different markers for extra distinction

###################################################
# Create figure
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharey=True)
plt.subplots_adjust(hspace=0.27)

#####################
### First subplot ###
#####################

ax = axes[0]
ax.set_ylim(-4.5, -3)

# Add gray band from y=0.5 to y=2
time = [2]

# Scatter plot
for j in time:  # Loop over time steps (1Myr, 2Myr, 3Myr)
    ax.scatter(range(len(simulations_top)), masses_twopop_top[:, j], color=colors[j],
               marker=markers[j]) #, label=time_labels[j])
    ax.scatter(range(len(simulations_top)), masses_tripod_top[:, j], color=colors_trpd[j], marker=markers[j])
    ax.scatter(range(len(simulations_top)), masses_dp_top[:, j], color='black',
               marker=markers[j], facecolor='none', s=30)
    
ax.set_xlim(-0.5,len(simulations_bottom)-0.5)
# Set axis limits
# Set x-axis
ax.set_xticks(range(len(simulations_top)))
ax.set_xticklabels(textlabels_top, rotation=90)

# Set y-axis (log scale)
ax.set_ylabel(r"$\frac{\alpha_{0.89-3.1}}{\alpha(DustPy)}$")

# First legend for time steps
for i in range(3):
    ax.scatter(-1, -1, color='black', marker=markers[i], label=time_labels[i])
time_legend = ax.legend(frameon=False, fontsize=11, loc='upper right')  # Adjust location>
ax.add_artist(time_legend)  # Ensure it remains on the plot

# Create handles for second legend
filled_marker = plt.Line2D([], [], color='#ff6600', marker='o', linestyle='None', markersize=6, label="TwoPop")
empty_marker = plt.Line2D([], [], color='#3399ff', marker='o', linestyle='None', markersize=6, label="TriPod")
# Second legend for drifting particles (placed at top-left)
ax.legend(handles=[filled_marker, empty_marker], loc='upper left', frameon=False, fontsize=11)

######################
### Second subplot ###
######################

ax = axes[1]
ax.set_ylim(-4.5, -3)
# Add gray band from y=0.5 to y=2

# Scatter plot
for j in range(2,3):  # Loop over time steps (1Myr, 2Myr, 3Myr)
    ax.scatter(range(len(simulations_bottom)), masses_twopop_bottom[:, j], color=colors[j],
               marker=markers[j]) #, label=time_labels_bottom)
    ax.scatter(range(len(simulations_bottom)), masses_tripod_bottom[:, j], color=colors_trpd[j], marker=markers[j])
    ax.scatter(range(len(simulations_top)), masses_dp_top[:, j], color='black',
               marker=markers[j], facecolor='none', s=30)
# Set axis limits
ax.set_xlim(-0.5,len(simulations_bottom)-0.5)

# Set x-axis
ax.set_xticks(range(len(simulations_bottom)))
ax.set_xticklabels(textlabels_bottom, rotation=90)

# Set y-axis (log scale)
ax.set_ylabel(r"$\frac{\alpha_{0.89-3.1}}{\alpha(DustPy)}$")

#######

plt.tight_layout()
plt.savefig(f"./alpha_mm.pdf")





import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib import colors, rc
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean.cm as cmo


strat_stair = True
figure3 = True
potential_vorticity = True


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/dwbc-proj')
raw_path = base_path / 'data/raw'
processed_path = base_path / 'data/processed'
figure_path = base_path / 'figures/presentation'


logging.info('Setting plotting defaults')
SMALL_SIZE = 28
MEDIUM_SIZE = 32
BIGGER_SIZE = 36
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

cm = 1 / 2.54
dpi = 192
px = 1 / 50

text_width = 5.5  # in inches
    
if strat_stair:
    logging.info('Plotting stratification slices')

    da_dbdz = xr.open_dataarray(processed_path / 'dbdz_slice.nc')
    X, Z = da_dbdz['XC'] * 1e-3, -da_dbdz['Zl'] * 1e-3
    
    fig = plt.figure(figsize=(500 * px, 500 * px))

    gs = gridspec.GridSpec(2, 1,
                           width_ratios=[1],
                           height_ratios=[15, 1]
                           )

    ax1 = fig.add_subplot(gs[0])

    cbax = plt.subplot(gs[1:])
    
    fig.suptitle('Stratification', fontweight='bold')

    ax1.set_ylabel('Depth (km)')
    ax1.set_xlabel('Longitude (km)')

    cmo.matter.set_bad('grey')
    cax = ax1.pcolormesh(X, Z, da_dbdz.isel(YC=1), shading='nearest', cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)

    ax1.invert_yaxis()

    ax1.set_xlim(0, 300)

    ax1.axvline(90, c='magenta', lw=4)

    cb = plt.colorbar(cax, cax=cbax, orientation='horizontal', label='$\partial_z$b (m$\,$s$^{-2})$')
    cb.formatter.set_useMathText(True)
    
    yticks = [0, 1, 2, 3, 4]
    ax1.set_yticks(yticks)

    fig.tight_layout()

    fig.savefig(figure_path / 'Figure2.pdf', dpi=dpi)


if potential_vorticity:
    logging.info('Plotting potential vorticity')
    
    cmo.curl.set_bad('grey')
    Qcmap = cmo.curl
    Qlim = 4e-11

    da_Q_on_rho = xr.open_dataarray(processed_path / 'Q_on_rho.nc')
    X_bigQ, Y_bigQ = da_Q_on_rho['XC'] * 1e-3, da_Q_on_rho['YC'] * 1e-3
    C_bigQ = da_Q_on_rho.values.squeeze()
    
    da_Q_slice = xr.open_dataarray(processed_path / 'Q_slice.nc')
    X, Z = da_Q_slice['XC'] * 1e-3, -da_Q_slice['Z'] * 1e-3
    
    fig = plt.figure(figsize=(600 * px, 800 * px))

    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1, 0.1]
                           )

    big_Q_ax = fig.add_subplot(gs[:2, 0])

    slice_ax3 = fig.add_subplot(gs[1, 1])
    slice_ax1 = fig.add_subplot(gs[0, 1], sharex=slice_ax3)

    #plt.setp(slice_ax2.get_xticklabels(), visible=False)
    plt.setp(slice_ax1.get_xticklabels(), visible=False)

    slice_cbax = fig.add_subplot(gs[2, :])
    #big_cbax = fig.add_subplot(gs[0:, 0])

    big_Q_ax.set_title('$\sigma = 28.04$')
    slice_ax1.set_title('250 km South')
    slice_ax3.set_title('500 km South')

    big_Q_ax.set_title('(a)', loc='left')
    slice_ax1.set_title('(b)', loc='left')
    slice_ax3.set_title('(c)', loc='left')


    big_Q_ax.set_xlabel('Longitude (km)')
    slice_ax3.set_xlabel('Longitude (km)')

    big_Q_ax.set_ylabel('Latitude (km)')
    slice_ax1.set_ylabel('Depth (km)')
    slice_ax3.set_ylabel('Depth (km)')
    
    yticks = [0, 1, 2, 3, 4]
    slice_ax1.set_yticks(yticks)
    slice_ax3.set_yticks(yticks)

    big_Q_cax = big_Q_ax.pcolormesh(X_bigQ, Y_bigQ, C_bigQ, cmap=Qcmap, shading='nearest',
                                    vmin=-Qlim, vmax=Qlim, rasterized=True)

    slice_ax1.pcolormesh(X, Z, da_Q_slice.isel(YC=1), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)
    slice_ax3.pcolormesh(X, Z, da_Q_slice.isel(YC=2), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)


    lw = 4
    big_Q_ax.axhline(0, label='Equator', lw=4, ls='-', c='k')
    big_Q_ax.axhline(-250, label='250 km South', lw=4, ls=':', c='k')
    big_Q_ax.axhline(-500, label='500 km South', lw=4, ls='-.', c='k')
    big_Q_ax.scatter(90, -250, label='Profile point', marker='o', c='magenta', linewidths=4)

    slice_ax1.axvline(90, c='magenta', lw=4)

    slice_ax1.set_xlim(0, 300)
    slice_ax3.set_xlim(0, 300)

    big_Q_ax.set_aspect('equal')
    big_Q_ax.set_ylim(-1800, 500)

    slice_ax1.invert_yaxis()
    slice_ax3.invert_yaxis()

    slice_cb = plt.colorbar(big_Q_cax, cax=slice_cbax, orientation='horizontal', label='Q (s$^{-3}$)')
    slice_cb.formatter.set_useMathText(True)

    #big_Q_ax.legend(loc='upper right')
    fig.suptitle('Potential vorticity', fontweight='bold')
    
    fig.tight_layout()

    fig.savefig(figure_path / 'Figure4.pdf', dpi=dpi)
    

if figure3:
    logging.info('Plotting overturning mechanisms')
    
    ds_overturning = xr.open_dataset(processed_path / 'overturning.nc')
    #rho_3, zetay_3 = xr.open_dataarray('rho3.nc'), xr.open_dataarray('zeta_y.nc')
    da_zeta_y_slice = xr.open_dataarray(processed_path / 'zeta_y_slice.nc')
    
    fig, ax1 = plt.subplots(1, 1, figsize=(500 * px, 500 * px))
    ax2 = ax1.twiny()

    lw=4
    # Right hand panel with staircase and zeta_y
    ax1.plot(ds_overturning['rho'] - 1000, -ds_overturning['Z'] * 1e-3, ls='-', lw=lw, c='k', label='$\\sigma(z)$')
    ax2.plot(ds_overturning['zeta_y'], -ds_overturning['Zl'] * 1e-3, ls='-', lw=lw, c='grey', label='$\\zeta_y(z)$')
    ax2.axvline(0, ls=':', lw=lw, c='grey')

    ax1.set_xlim(1027.9 - 1000, 1028.15 - 1000)
    ax1.set_ylim((ds_overturning['Depth'] - 16)* 1e-3, 1500 * 1e-3)

    ax1.set_ylabel('Depth (km)')
    ax1.set_xlabel('$\\sigma$ (kg$\,$m$^{-3}$)')
    ax2.set_xlabel('$\\zeta_y$ (s$^{-1}$)', labelpad=10)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    fig.suptitle('Staircases & overturning', fontweight='bold')

    # Figure stuff
    fig.legend(loc='upper right', bbox_to_anchor=(0.42, 0.33))
    #fig.suptitle('Staircases & overturning', weight='bold', y=0.97)

    fig.tight_layout()

    fig.savefig(figure_path / 'Figure3.pdf', dpi=dpi)
    
logging.info('Plotting complete')


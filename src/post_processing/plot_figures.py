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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib import rc
from matplotlib import font_manager as fm
import cmocean.cm as cmo
import dask
#from dask.distributed import Client, LocalCluster
#from dask_jobqueue import SLURMCluster


figure1 = False
figure2 = False
thesiscover = False
figure3 = False
figure4 = False

figure2and3NS = False
figure4NS = False

figure2and3_2km = False
figure4_2km = False

anim_figure2 = False
anim_figure3_bot = False
anim_figure4 = False

paper = True
thesis = False

assert paper != thesis

logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/dwbc-proj')
raw_path = base_path / 'data/raw'
processed_path = base_path / 'data/processed'
log_path = base_path / "src/.tmp/slurm-out"
dask_worker_path = base_path / "src/.tmp/"
env_path = base_path / "dwbc-proj/bin/activate"
figure_path = base_path / "figures"

if paper:
    SMALL_SIZE = 8
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 8
    rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    text_width = 5.5  # in inches

    cm = 1/2.54
    dpi = 300

elif thesis:
    # fonts
    fpath = Path('/home/n01/n01/fwg/.local/share/fonts/PTSans-Regular.ttf')
    assert fpath.exists()
    font_prop = fm.FontProperties(fname=fpath)
    plt.rcParams['font.family'] = font_prop.get_family()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]

    # font size
    mpl.use("pgf")
    plt.rc('xtick', labelsize='8')
    plt.rc('ytick', labelsize='8')
    plt.rc('text', usetex=False)
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams["text.latex.preamble"] = "\\usepackage{euler} \\usepackage{paratype}  \\usepackage{mathfont} \\mathfont[digits]{PT Sans}"
    plt.rcParams["pgf.preamble"] = plt.rcParams["text.latex.preamble"]
    plt.rc('text', usetex=False)



    # output
    dpi = 600
    text_width = 6




if figure1:
    logging.info('Plotting initial and boundary conditions')
    
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    # Open the datasats
    ds_init = xr.open_dataset(processed_path / f'init.zarr', engine="zarr")
    ds_bathymetry = xr.open_dataset(raw_path / 'GEBCO-bathymetry-data/gebco_2021_n30.0_s-30.0_w-85.0_e-10.0.nc')
    ds_bathymetry = ds_bathymetry.coarsen(lon=5, lat=5, boundary='trim').mean()
    ds_climatological_gamma_n = xr.open_dataset(processed_path / 'climatological_gamman.nc', decode_times=False)

    # Set up the canvas
    pad = 35
    fig = plt.figure(figsize=(text_width, 4))#9 * cm))

    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1/16, 1, 1/16]
                           )

    ax1 = fig.add_subplot(gs[:2, 0], projection=ccrs.PlateCarree())

    # Plot the bathymetry
    cax_bathy = ax1.pcolormesh(ds_bathymetry['lon'],
                               ds_bathymetry['lat'],
                               -ds_bathymetry['elevation'],
                               shading='nearest',
                               rasterized=True,
                               cmap=cmo.deep,
                               vmin=0
                              )

    # Add some land
    ax1.add_feature(cfeature.NaturalEarthFeature('physical',
                                                 'land',
                                                 '110m',
                                                 edgecolor='face',
                                                 facecolor='grey'
                                                ))
    
    y0 = ds_climatological_gamma_n['lat'].min()
    ywid = ds_climatological_gamma_n['lat'].max() - y0
    x0 = ds_climatological_gamma_n['lon'].min()
    xwid = ds_climatological_gamma_n['lon'].max() - x0
    ax1.add_patch(Rectangle((x0, y0), xwid, ywid, ec='red', fc='none'))

    # Axes limits, labels and features
    ax1.axhline(0, c='k', ls='--')

    ax1.set_ylim(-12, 30)
    ax1.set_xlim(-85, -25)

    ax1.set_xticks(np.arange(-85, -24, 10), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(-10, 31, 10), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    axtopL = fig.add_subplot(gs[0, 0])
    axtopL.axis("off")

    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    axtopL.set_title('The Tropical Atlantic', pad=pad)
    axtopL.set_title('(a)', loc='left', pad=pad)

    # Colorbars
    cbax1 = fig.add_subplot(gs[2, 0])
    cb1 = plt.colorbar(cax_bathy, cax=cbax1, label='Depth (m)', orientation='horizontal')

    # Initial condition plots
    axtopR = fig.add_subplot(gs[0, 1])
    axtopR.axis("off")
    ax2 = fig.add_subplot(gs[0:2, 1])
    cbax = fig.add_subplot(gs[2, 1])

    
    axtopR.set_title('Initial conditions', pad=pad)
    axtopR.set_title('(b)', loc='left', pad=pad)
    ax2.set_xlabel('Longitude (km)')
    ax2.set_ylabel('Depth (m)')

    cmo.tempo_r.set_bad('grey')
    cax = ax2.pcolormesh(ds_init['XC'] * 1e-3,
                         -ds_init['Z'],
                         ds_init['V_init'] * 1e2,
                         vmin=-20,
                         vmax=0,
                         shading='nearest',
                         cmap=cmo.tempo_r,
                         rasterized=True
                        )

    ax2.invert_yaxis()

    cb = plt.colorbar(cax, cax=cbax, label='Meridional velocity (cm$\,$s$^{-1}$)',
                      orientation='horizontal')

    cb.formatter.set_useMathText(True)

    axins = ax2.twiny()

    ln_id, = axins.plot(ds_init['rho_init'] - 1000, -ds_init['Z'], c='k', label='idealised')
    ln_clim, = axins.plot(ds_climatological_gamma_n['mn_gamma_n'],
                          -ds_climatological_gamma_n['depth'],
                          label='climatalogical',
                          c='k',
                          ls='--'
                         )

    axins.set_xlabel('$\\gamma^n$ (kg$\,$m$^{-3}$)',
                     labelpad=3,
                     loc='center',
                     usetex=True)
    axins.set_xlim(20,29)
    axins.set_xticks(range(22, 29))

    ax2.legend([ln_id, ln_clim], ['Idealised', 'Climatological'], loc='lower center')
    ax2.set_ylim(-ds_init['Z'][-1] ,0)

    fig.tight_layout()
    fig.savefig(figure_path / 'Figure1.pdf', dpi=dpi)
    fig.show()

    
if figure2:
    logging.info('Plotting stratification slices')

    da_dbdz = xr.open_dataarray(processed_path / f'standard/dbdz_slice.zarr', engine="zarr")
    X, Z = da_dbdz['XC'] * 1e-3, -da_dbdz['Zl']
    
    fig = plt.figure(figsize=(text_width, 3.25))#8.5 * cm))

    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[1, 1, 1],
                           height_ratios=[15, 1]
                           )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    cbax = plt.subplot(gs[3:])

    ax1.set_title('Equator')
    ax2.set_title('250 km South')
    ax3.set_title('500 km South')

    ax1.set_title('(a)', loc='left')
    ax2.set_title('(b)', loc='left')
    ax3.set_title('(c)', loc='left')

    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('Longitude (km)')
    ax2.set_xlabel('Longitude (km)')
    ax3.set_xlabel('Longitude (km)')

    cmo.matter.set_bad('grey')
    cax = ax1.pcolormesh(X, Z, da_dbdz.isel(YC=0), shading='nearest', cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)
    ax2.pcolormesh(X, Z, da_dbdz.isel(YC=1), shading='nearest', cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)
    ax3.pcolormesh(X, Z, da_dbdz.isel(YC=2), shading='nearest', cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)

    ax1.invert_yaxis()

    ax1.set_xlim(0, 300)
    ax2.set_xlim(0, 300)
    ax3.set_xlim(0, 300)

    ax2.axvline(90, c='magenta')

    cb = plt.colorbar(cax, cax=cbax, orientation='horizontal')
    cb.formatter.set_useMathText(True)
    cb.set_label("$\partial_z b$ (s$^{-2})$", usetex=True)
    yticks = [0, 1000, 2000, 3000, 4000]
    ax1.set_yticks(yticks)

    fig.tight_layout()

    fig.savefig(figure_path / 'Figure2.pdf', dpi=dpi)
    

# if figure4 and thesiscover:
#     logging.info('Plotting thesis cover image')
    
#     fig, ax = plt.subplots(frameon=True, figsize=(12, 12))

#     ax.pcolormesh(X_bigQ, Y_bigQ, C_bigQ, cmap=Qcmap, shading='nearest',vmin=-Qlim, vmax=Qlim, rasterized=True)
#     ax.set_aspect('equal')
#     ax.axis('off')
#     ax.set_xlim(50, 600)
#     fig.tight_layout()
#     fig.savefig(figure_path / 'ThesisCover.png', dpi=600)
    

if figure3:
    logging.info('Plotting overturning mechanisms')
    
    ds_overturning = xr.open_dataset(processed_path / f'standard/overturning.zarr', engine="zarr")
    #rho_3, zetay_3 = xr.open_dataarray('rho3.zarr'), xr.open_dataarray('zeta_y.zarr')
    da_zeta_y_slice = xr.open_dataarray(processed_path / f'standard/zeta_y_slice.zarr', engine="zarr")
    da_dbdz = xr.open_dataarray(processed_path / f'standard/dbdz_slice.zarr', engine="zarr").sel(YC=-250e3, method='nearest')
    
    da_tm = xr.open_dataarray(processed_path / 'toy_model/toy_strat_data.zarr', engine='zarr')
    
    fig = plt.figure(figsize=(text_width, 6)) #2 * 8.5 * cm))

    width_ratios = [1, 1, 1, 1, 1, 1]
    height_ratios = [1, 1/16, 0.6, 1/16]
    
    gst = gridspec.GridSpec(4, 6,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios
                           )
    
    gsm = gridspec.GridSpec(4, 6,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios
                           )
    
    gsb = gridspec.GridSpec(4, 6,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios
                           )

    gst.update(wspace=4, hspace=1.1)
    gsm.update(wspace=4, hspace=2.25)
    gsb.update(wspace=0.5, hspace=0.5)

    ax1 = fig.add_subplot(gsm[:2, 3:])
    ax_overturn = fig.add_subplot(gst[0, :3])
    cbax = fig.add_subplot(gsm[1, :3])
    ax2 = ax1.twiny()

    axtm0 = fig.add_subplot(gsb[2, :2])
    axtm1 = fig.add_subplot(gsb[2, 2:4])
    axtm2 = fig.add_subplot(gsb[2, 4:6])
    
    cbaxtm = fig.add_subplot(gsb[3, :])

    # Right hand panel with staircase and zeta_y
    ln1 = ax1.plot(ds_overturning['rho'] - 1000, -ds_overturning['Z'], ls='-', c='k', label='$\\gamma^n(z)$')
    ln2 = ax2.plot(ds_overturning['zeta_y'], -ds_overturning['Zl'], ls='-', c='tab:orange', label='$\\xi_y(z)$')
    ax2.axvline(0, ls=':', c='grey')

    ax1.set_xlim(1027.9 - 1000, 1028.15 - 1000)
    ax1.set_ylim((ds_overturning['Depth'] - 16), 1500)

    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('$\\gamma^n$ (kg$\,$m$^{-3}$)', usetex=True)
    ax2.set_xlabel('$\\xi_y$ (s$^{-1}$)', usetex=True)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    ax1.set_title('(b)', loc='left')
    ax1.set_title('Staircases & overturning')

    # Left hand panel with zeta_y
    cmo.balance.set_bad('grey')
    zylim = 2.5e-3
    cax = ax_overturn.pcolormesh(da_zeta_y_slice['XG'] * 1e-3,
                                 -da_zeta_y_slice['Zl'],
                                 da_zeta_y_slice,
                                 shading='nearest',
                                 cmap=cmo.balance,
                                 vmin=-zylim, vmax=zylim,
                                 rasterized=True)

    ax_overturn.contour(da_dbdz['XC'] * 1e-3,
                        -da_dbdz['Zl'],
                        da_dbdz,
                        levels=[2e-6], colors='k', linewidths=1.25,
                        vmax=2.1e-6)
    
    ax_overturn.invert_yaxis()
    ax_overturn.axvline(90, c='magenta')
    ax_overturn.set_xlim(0, 300)

    ax_overturn.set_xlabel('Longitude (km)')
    ax_overturn.set_ylabel('Depth (m)')
    ax_overturn.set_title('Zonal overturning')
    ax_overturn.set_title('(a)', loc='left')

    ax_overturn.set_ylim(3500, 1500)
    ax_overturn.set_xlim(20, 180)
    # Colorbar
    cb = plt.colorbar(cax, cax=cbax,
                      orientation='horizontal')
    cb.set_label("$\\xi_y$ (s$^{-1}$)", usetex=True)
    cb.formatter.set_useMathText(True)
    cb.formatter.set_powerlimits((0, 0))

    # Toy model plots
    axtm0.set_title('(c)', loc='left')
    axtm1.set_title('(d)', loc='left')
    axtm2.set_title('(e)', loc='left')
    
    axtm0.set_title('0 days')
    axtm1.set_title('14 days')
    axtm2.set_title('28 days')
    
    axtm1.set_xlabel('Longitude (km)')
    axtm0.set_ylabel('Depth (m)')

    axtm1.set_yticklabels([])
    axtm2.set_yticklabels([])
    
    axtm0.set_xlim(-50, 50)
    axtm1.set_xlim(-50, 50)
    axtm2.set_xlim(-50, 50)
    
    axtm0.set_ylim(600, 0)
    axtm1.set_ylim(600, 0)
    axtm2.set_ylim(600, 0)
    
    psi = np.exp(-da_tm['X'] ** 2 / 2 / (25e3) ** 2) * np.sin(2 * np.pi / 200 * da_tm['Z'])
    
    cax = axtm0.pcolormesh(da_tm['X'] * 1e-3,
                          -da_tm['Z'],
                          da_tm.isel(time=0),
                          shading='nearest',
                          cmap=cmo.matter,
                          vmin=0, vmax=4e-6,
                          rasterized=True)
    
    axtm0.contour(da_tm['X'] * 1e-3,
                  -da_tm['Z'],
                  psi.transpose(),
                  cmap=cmo.balance,
                  levels=[-0.8, -0.4, 0, 0.4, 0.8],
                  vmin=-1, vmax=1)

    axtm1.pcolormesh(da_tm['X'] * 1e-3,
                     -da_tm['Z'],
                     da_tm.isel(time=2),
                     shading='nearest',
                     cmap=cmo.matter,
                     vmin=0, vmax=4e-6,
                     rasterized=True)
    
    axtm2.pcolormesh(da_tm['X'] * 1e-3,
                     -da_tm['Z'],
                     da_tm.isel(time=4),
                     shading='nearest',
                     cmap=cmo.matter,
                     vmin=0, vmax=4e-6,
                     rasterized=True)
    

    
    cbtm = fig.colorbar(cax, cax=cbaxtm,
                        orientation='horizontal')
    cbtm.set_label("$\\partial_z b$ (s$^{-2}$)", usetex=True)
    cbtm.formatter.set_useMathText(True)
    cbtm.formatter.set_powerlimits((0, 0))
    
    # Figure stuff
    plt.rc('text', usetex=True)
    ax1.legend(loc='lower left', handles=ln1 + ln2)
    plt.rc('text', usetex=False)
    #fig.tight_layout()
    
    fig.savefig(figure_path / 'Figure3.pdf', dpi=dpi)
    
logging.info('Plotting complete')

if anim_figure2:
    logging.info('Plotting stratification slice movie')

    da_path = processed_path / 'standard/animations/stratification_slices.zarr'
    da_dbdz = xr.open_dataarray(da_path, engine="zarr")
    X, Z = da_dbdz['XC'] * 1e-3, -da_dbdz['Zl']
    
    @dask.delayed()
    def plot_frame(nt, da_dbdz):
        fig = plt.figure(figsize=(text_width, 3.25))#8.5 * cm))

        gs = gridspec.GridSpec(2, 3,
                            width_ratios=[1, 1, 1],
                            height_ratios=[15, 1]
                            )

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

        cbax = plt.subplot(gs[1, :])

        ax1.set_title('Equator')
        ax2.set_title('250 km South')
        ax3.set_title('500 km South')

        ax1.set_title('(a)', loc='left')
        ax2.set_title('(b)', loc='left')
        ax3.set_title('(c)', loc='left')

        tdays = float(da_dbdz['time'].isel(time=nt).values) * 1e-9 / 24 / 60 / 60
        fig.suptitle('t = {:.2f} days'.format(tdays))

        ax1.set_ylabel('Depth (m)')
        ax1.set_xlabel('Longitude (km)')
        ax2.set_xlabel('Longitude (km)')
        ax3.set_xlabel('Longitude (km)')

        cmo.matter.set_bad('grey')
        cax = ax1.pcolormesh(X, Z, da_dbdz.isel(YC=0, time=nt),
                            shading='nearest',
                            cmap=cmo.matter,
                            vmin=0,
                            vmax=7.5e-6,
                            rasterized=True)
        
        ax2.pcolormesh(X, Z, da_dbdz.isel(YC=1, time=nt),
                    shading='nearest',
                    cmap=cmo.matter,
                    vmin=0,
                    vmax=7.5e-6,
                    rasterized=True)
        
        ax3.pcolormesh(X, Z, da_dbdz.isel(YC=2, time=nt),
                    shading='nearest',
                    cmap=cmo.matter,
                    vmin=0,
                    vmax=7.5e-6,
                    rasterized=True)

        ax1.invert_yaxis()

        ax1.set_xlim(0, 300)
        ax2.set_xlim(0, 300)
        ax3.set_xlim(0, 300)

        ax2.axvline(90, c='magenta')

        cb = plt.colorbar(cax, cax=cbax, orientation='horizontal')
        cb.formatter.set_useMathText(True)
        cb.set_label("$\partial_z$b (s$^{-2})$", usetex=True)
        yticks = [0, 1000, 2000, 3000, 4000]
        ax1.set_yticks(yticks)

        fig.tight_layout()

        fig_name = 'dbdz{:03d}.png'.format(nt)
        fig.savefig(figure_path / 'strat_slice_frames2' / fig_name, dpi=dpi)
        plt.close(fig)

    logging.info('Initialising the dask cluster')
    # Set up the dask cluster
    # scluster = SLURMCluster(queue='standard',
    #                         project="n01-siAMOC",
    #                         job_cpu=128,
    #                         log_directory=log_path,
    #                         local_directory=dask_worker_path,
    #                         cores=64,
    #                         processes=8,  # Can change this
    #                         memory="256 GiB",
    #                         header_skip= ['#SBATCH --mem='],  
    #                         walltime="00:20:00",
    #                         death_timeout=60,
    #                         interface='hsn0',
    #                         job_extra=['--qos=short', '--reservation=shortqos'],
    #                         env_extra=["export PATH=$PATH:/work/n01/n01/fwg/local-sites/texlive/2022/bin/x86_64-linux",
    #                                 'module load cray-python',
    #                                 'source {}'.format(str(env_path.absolute()))]
    #                     )

    # njobs = 1
    # client = Client(scluster)
    # scluster.scale(jobs=njobs)
        
    # logging.info('Creating list of tasks')
    # collection = [plot_frame(nt, da_dbdz) for nt in range(da_dbdz.sizes['time'])]
    [plot_frame(nt, da_dbdz).compute() for nt in range(da_dbdz.sizes['time'])]
    # logging.info('Computing tasks')
    # dask.compute(collection)

    # ffmpeg -framerate 15 -i toy_model_frames/toy_dbdz%03d.png -pix_fmt yuv420p Figure3_lower.mp4

if anim_figure3_bot:
    da = xr.open_dataset(processed_path / "toy_model/toy_strat_data_full.zarr",
                         engine="zarr")["db_dz"]
    
    
    def plot_frame(nt):
        tdays = float(da['time'].isel(time=nt).values) / 24 / 60 / 60
        fig, ax = plt.subplots(figsize=(3,3))
        cax = ax.pcolormesh(da["X"] * 1e-3,
                            -da["Z"],
                            da.isel(time=nt),
                            cmap=cmo.matter,
                            vmin=0,
                            vmax=4e-6)

        ax.invert_yaxis()
        ax.set_xlabel("Longitude (km)")
        ax.set_ylabel("Depth (m)")

        ax.set_title('t = {:.2f} days'.format(tdays))

        cb = fig.colorbar(cax, orientation="horizontal", pad=0.2, label="$\partial_z b$ (s$^{-2}$)")
        cb.formatter.set_useMathText(True)
        cb.formatter.set_powerlimits((0, 0))
        fig.tight_layout()

        fig_name = "toy_dbdz{:03d}.png".format(nt)
        fig.savefig(figure_path / "toy_model_frames" / fig_name, dpi=2 * dpi)
    
    for nt in range(da.sizes["time"]):
        plot_frame(nt)

# ffmpeg -framerate 15 -i toy_model_frames/toy_dbdz%03d.png -pix_fmt yuv420p Figure3_lower.mp4

if anim_figure4:

    Q_on_rho_path = processed_path / "standard/animations/PV_on_rho.zarr"
    Q_slice_path = processed_path / "standard/animations/PV_slices.zarr"
    da_Q_on_rho = xr.open_dataarray(Q_on_rho_path, engine="zarr")
    da_Q_slice = xr.open_dataarray(Q_slice_path, engine="zarr")

    def plot_frame(nt):
        cmo.curl.set_bad('grey')
        Qcmap = cmo.curl
        Qlim = 4e-11

        X_bigQ, Y_bigQ = da_Q_on_rho['XG'] * 1e-3, da_Q_on_rho['YG'] * 1e-3
        C_bigQ = da_Q_on_rho.isel(time=nt).values.squeeze()
        
        X, Z = da_Q_slice['XG'] * 1e-3, -da_Q_slice['Zl']
        
        fig = plt.figure(figsize=(text_width, 7))#23 * cm))

        gs = gridspec.GridSpec(4, 2,
                            width_ratios=[1, 1],
                            height_ratios=[1, 1, 1, 0.1]
                            )

        big_Q_ax = fig.add_subplot(gs[:3, 0])

        slice_ax3 = fig.add_subplot(gs[2, 1])
        slice_ax1 = fig.add_subplot(gs[0, 1], sharex=slice_ax3)
        slice_ax2 = fig.add_subplot(gs[1, 1], sharex=slice_ax3)

        plt.setp(slice_ax2.get_xticklabels(), visible=False)
        plt.setp(slice_ax1.get_xticklabels(), visible=False)

        slice_cbax = fig.add_subplot(gs[3, :])
        #big_cbax = fig.add_subplot(gs[0:, 0])

        big_Q_ax.set_title('$\gamma^n =$~28.04', usetex=True)
        slice_ax1.set_title('Equator')
        slice_ax2.set_title('250 km South')
        slice_ax3.set_title('500 km South')

        big_Q_ax.set_title('(a)', loc='left')
        slice_ax1.set_title('(b)', loc='left')
        slice_ax2.set_title('(c)', loc='left')
        slice_ax3.set_title('(d)', loc='left')


        big_Q_ax.set_xlabel('Longitude (km)')
        slice_ax3.set_xlabel('Longitude (km)')

        big_Q_ax.set_ylabel('Latitude (km)')
        slice_ax1.set_ylabel('Depth (m)')
        slice_ax2.set_ylabel('Depth (m)')
        slice_ax3.set_ylabel('Depth (m)')
        
        yticks = [0, 1000, 2000, 3000, 4000]
        slice_ax1.set_yticks(yticks)
        slice_ax2.set_yticks(yticks)
        slice_ax3.set_yticks(yticks)

        big_Q_cax = big_Q_ax.pcolormesh(X_bigQ, Y_bigQ, C_bigQ, cmap=Qcmap, shading='nearest',
                                        vmin=-Qlim, vmax=Qlim, rasterized=True)

        slice_ax1.pcolormesh(X, Z, da_Q_slice.isel(YG=0, time=nt), cmap=Qcmap,
                             shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)
        
        slice_ax2.pcolormesh(X, Z, da_Q_slice.isel(YG=1, time=nt), cmap=Qcmap,
                             shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)
        
        slice_ax3.pcolormesh(X, Z, da_Q_slice.isel(YG=2, time=nt), cmap=Qcmap,
                             shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)


        big_Q_ax.axhline(0, label='Equator', ls='-', c='k')
        big_Q_ax.axhline(-250, label='250 km South', ls=':', c='k')
        big_Q_ax.axhline(-500, label='500 km South', ls='-.', c='k')
        #big_Q_ax.axhline(-600, label='600 km south', ls='--', c='k')
        big_Q_ax.scatter(90, -250, label='Profile point', marker='o', c='magenta', linewidths=2)

        slice_ax2.axvline(90, c='magenta')

        slice_ax1.set_xlim(0, 300)
        slice_ax2.set_xlim(0, 300)
        slice_ax3.set_xlim(0, 300)


        tdays = float(da_Q_on_rho['time'].isel(time=nt).values) * 1e-9 / 24 / 60 / 60
        title = "t = {:.2f} days".format(tdays)
        fig.suptitle(title)

        big_Q_ax.set_aspect('equal')
        big_Q_ax.set_ylim(-1800, 500)

        slice_ax1.invert_yaxis()
        slice_ax2.invert_yaxis()
        
        slice_ax3.invert_yaxis()

        slice_cb = plt.colorbar(big_Q_cax, cax=slice_cbax,
                                orientation="horizontal")
        slice_cb.formatter.set_useMathText(True)
        slice_cb.set_label("$Q$ (s$^{-3})$", usetex=True)

        big_Q_ax.legend(loc='upper right')

        fig.tight_layout()

        fig_name = "full_pv_frames/Q{:03d}.png".format(nt)
        fig.savefig(figure_path / fig_name, dpi=2 * dpi)
        plt.close(fig)
        
    for nt in range(da_Q_on_rho.sizes["time"]):
        plot_frame(nt)
        
# ffmpeg -framerate 15 -i full_pv_frames/Q%03d.png -pix_fmt yuv420p Figure4.mp4


def plot_figure2and3(dbdz_path, overturning_path, zeta_y_path, fig_out_path,
                     suptitle):
    da_dbdz = xr.open_dataarray(dbdz_path, engine="zarr")
    ds_overturning = xr.open_dataset(overturning_path, engine="zarr")
    da_zeta_y_slice = xr.open_dataarray(zeta_y_path, engine="zarr")
    
    # da_dbdz = xr.open_dataarray(processed_path / f'dbdz_slice.zarr').sel(YC=-250e3, method='nearest')

    X, Z = da_dbdz['XC'] * 1e-3, -da_dbdz['Zl']
    
    fig = plt.figure(figsize=(text_width, 7))#8.5 * cm))
    
    width_ratios = [1, 1, 1, 1, 1, 1]
    height_ratios = [15, 1, 15, 1]
    
    gst = gridspec.GridSpec(4, 6,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios
                           )
    
    gsmt = gridspec.GridSpec(4, 6,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios
                           )
    
    gsmb = gridspec.GridSpec(4, 6,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios
                           )
    
    gsb = gridspec.GridSpec(4, 6,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios
                           )


    #gs = gridspec.GridSpec(4, 6,
    #                       width_ratios=[1, 1, 1, 1, 1, 1],
    #                       height_ratios=[15, 1, 15, 1]
     #                      )

    gst.update(wspace=1, hspace=1)
    gsmt.update(wspace=1, hspace=2)
    gsmb.update(wspace=4.2, hspace=3)
    gsb.update(wspace=2.5, hspace=1)


    ax1 = fig.add_subplot(gst[0, 0:2])
    ax2 = fig.add_subplot(gst[0, 2:4], sharey=ax1)
    ax3 = fig.add_subplot(gst[0, 4:6], sharey=ax1)
    ax4 = fig.add_subplot(gsb[2, 0:3])
    ax5 = fig.add_subplot(gsmb[2:4, 3:6])

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    cbax = plt.subplot(gsmt[1, :])
    cb_zeta = fig.add_subplot(gsb[3, 0:3])
    
    ax1.set_title('Equator')
    ax2.set_title('250 km South')
    ax3.set_title('500 km South')
    ax4.set_title("Zonal overturning")
    ax5.set_title("Staircases & overturning")

    ax1.set_title('(a)', loc='left')
    ax2.set_title('(b)', loc='left')
    ax3.set_title('(c)', loc='left')

    ax1.set_ylabel('Depth (m)')
    ax1.set_xlabel('Longitude (km)')
    ax2.set_xlabel('Longitude (km)')
    ax3.set_xlabel('Longitude (km)')

    cmo.matter.set_bad('grey')
    cax = ax1.pcolormesh(X, Z, da_dbdz.isel(YC=0), shading='nearest',
                         cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)
    
    ax2.pcolormesh(X, Z, da_dbdz.isel(YC=1), shading='nearest',
                   cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)
    
    ax3.pcolormesh(X, Z, da_dbdz.isel(YC=2), shading='nearest',
                   cmap=cmo.matter, vmin=0, vmax=7.5e-6, rasterized=True)

    ax1.invert_yaxis()

    ax1.set_xlim(0, 300)
    ax2.set_xlim(0, 300)
    ax3.set_xlim(0, 300)

    ax2.axvline(90, c='magenta')

    cb = plt.colorbar(cax, cax=cbax, orientation='horizontal')
    cb.formatter.set_useMathText(True)
    cb.set_label("$\partial_z$b (s$^{-2})$", usetex=True)
    yticks = [0, 1000, 2000, 3000, 4000]
    ax1.set_yticks(yticks)


    logging.info('Plotting overturning mechanisms')
        
    ax6 = ax5.twiny()

    # Right hand panel with staircase and zeta_y
    ln1 = ax5.plot(ds_overturning['rho'] - 1000, -ds_overturning['Z'],
                   ls='-', c='k', label='$\\gamma^n(z)$')
    
    ln2 = ax6.plot(ds_overturning['zeta_y'], -ds_overturning['Zl'],
                   ls='-', c='tab:orange', label='$\\xi_y(z)$')
    
    ax6.axvline(0, ls=':', c='grey')

    ax5.set_xlim(1027.9 - 1000, 1028.15 - 1000)
    ax5.set_ylim((ds_overturning['Depth'] - 16), 1500)

    ax5.set_ylabel('Depth (m)')
    ax5.set_xlabel('$\\gamma^n$ (kg$\,$m$^{-3}$)', usetex=True)
    ax6.set_xlabel('$\\xi_y$ (s$^{-1}$)', usetex=True)
    ax6.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    ax5.set_title('(e)', loc='left')
    ax5.set_title('Staircases & overturning')

    # # Left hand panel with zeta_y
    cmo.balance.set_bad('grey')
    zylim = 2.5e-3
    cax = ax4.pcolormesh(da_zeta_y_slice['XG'] * 1e-3,
                         -da_zeta_y_slice['Zl'],
                         da_zeta_y_slice,
                         shading='nearest',
                         cmap=cmo.balance,
                         vmin=-zylim, vmax=zylim,
                         rasterized=True)

    ax4.contour(da_dbdz['XC'] * 1e-3,
                -da_dbdz['Zl'],
                da_dbdz.sel(YC=-250e3, method='nearest'),
                levels=[2e-6], colors='k', linewidths=1.25,
                vmax=2.1e-6)
    
    ax4.invert_yaxis()
    ax4.axvline(90, c='magenta')
    ax4.set_xlim(0, 300)

    ax4.set_xlabel('Longitude (km)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_title('Zonal overturning')
    ax4.set_title('(d)', loc='left')

    ax4.set_ylim(3500, 1500)
    ax4.set_xlim(20, 180)
    # Colorbar
    cb = plt.colorbar(cax, cax=cb_zeta,
                      orientation='horizontal')
    cb.set_label("$\\xi_y$ (s$^{-1}$)", usetex=True)
    cb.formatter.set_useMathText(True)
    cb.formatter.set_powerlimits((0, 0))

    
    # Figure stuff
    plt.rc('text', usetex=True)
    ax5.legend(loc='lower left', handles=ln1 + ln2)
    plt.rc('text', usetex=False)
    
    fig.suptitle(suptitle)
    fig.tight_layout()
    
    fig.savefig(fig_out_path, dpi=dpi)






if figure2and3NS:
    dbdz_path = processed_path / "NS/dbdz_sliceNS.zarr"
    overturning_path = processed_path / "NS/overturningNS.zarr"
    zeta_y_path = processed_path / "NS/zeta_y_sliceNS.zarr"
    fig_out_path = figure_path / "FigureS4.pdf"
    suptitle = "No-slip lateral BC"

    plot_figure2and3(dbdz_path, overturning_path, zeta_y_path, fig_out_path,
                     suptitle)


if figure2and3_2km:
    dbdz_path = processed_path / "2km/dbdz_slice2km.zarr"
    overturning_path = processed_path / "2km/overturning2km.zarr"
    zeta_y_path = processed_path / "2km/zeta_y_slice2km.zarr"
    fig_out_path = figure_path / "FigureS2.pdf"
    suptitle = "2 km resolution"

    plot_figure2and3(dbdz_path, overturning_path, zeta_y_path, fig_out_path,
                     suptitle)




def plot_figure4(Q_on_rho_path, Q_slice_path, rho_slice_path, fig_out_path,
                 suptitle=None):
    logging.info('Plotting potential vorticity')
    
    cmo.curl.set_bad('grey')
    Qcmap = cmo.curl
    Qlim = 4e-11


    da_Q_on_rho = xr.open_dataarray(Q_on_rho_path, engine="zarr")
    X_bigQ, Y_bigQ = da_Q_on_rho['XG'] * 1e-3, da_Q_on_rho['YG'] * 1e-3
    C_bigQ = da_Q_on_rho.values.squeeze()
    
    da_Q_slice = xr.open_dataarray(Q_slice_path, engine="zarr")
    X, Z = da_Q_slice['XG'] * 1e-3, -da_Q_slice['Zl']
    
    da_rho = xr.open_dataarray(rho_slice_path, engine="zarr")
    
    fig = plt.figure(figsize=(text_width, 7))#23 * cm))

    gs = gridspec.GridSpec(4, 2,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1, 1, 0.1]
                           )

    big_Q_ax = fig.add_subplot(gs[:3, 0])

    slice_ax3 = fig.add_subplot(gs[2, 1])
    slice_ax1 = fig.add_subplot(gs[0, 1], sharex=slice_ax3)
    slice_ax2 = fig.add_subplot(gs[1, 1], sharex=slice_ax3)

    plt.setp(slice_ax2.get_xticklabels(), visible=False)
    plt.setp(slice_ax1.get_xticklabels(), visible=False)

    slice_cbax = fig.add_subplot(gs[3, :])
    #big_cbax = fig.add_subplot(gs[0:, 0])

    big_Q_ax.set_title('$\gamma^n =$~28.04', usetex=True)
    slice_ax1.set_title('Equator')
    slice_ax2.set_title('250 km South')
    slice_ax3.set_title('500 km South')

    big_Q_ax.set_title('(a)', loc='left')
    slice_ax1.set_title('(b)', loc='left')
    slice_ax2.set_title('(c)', loc='left')
    slice_ax3.set_title('(d)', loc='left')

    big_Q_ax.set_xlabel('Longitude (km)')
    slice_ax3.set_xlabel('Longitude (km)')

    big_Q_ax.set_ylabel('Latitude (km)')
    slice_ax1.set_ylabel('Depth (m)')
    slice_ax2.set_ylabel('Depth (m)')
    slice_ax3.set_ylabel('Depth (m)')
    
    yticks = [0, 1000, 2000, 3000, 4000]
    slice_ax1.set_yticks(yticks)
    slice_ax2.set_yticks(yticks)
    slice_ax3.set_yticks(yticks)

    big_Q_cax = big_Q_ax.pcolormesh(X_bigQ, Y_bigQ, C_bigQ, cmap=Qcmap,
                                    shading='nearest',
                                    vmin=-Qlim, vmax=Qlim, rasterized=True)

    slice_ax1.pcolormesh(X, Z, da_Q_slice.isel(YG=0), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)
    
    slice_ax2.pcolormesh(X, Z, da_Q_slice.isel(YG=1), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)
    
    slice_ax3.pcolormesh(X, Z, da_Q_slice.isel(YG=2), cmap=Qcmap, shading='nearest', vmin=-Qlim, vmax=Qlim, rasterized=True)

    slice_ax1.contour(da_rho["XC"] * 1e-3, -da_rho["Z"], da_rho.isel(YC=0),
                      levels=[1028.04], colors=["magenta"], linewidths=1.5, linestyles="dashed")
    
    slice_ax2.contour(da_rho["XC"] * 1e-3, -da_rho["Z"], da_rho.isel(YC=1),
                      levels=[1028.04], colors=["magenta"], linewidths=1.5, linestyles="dashed")
    
    slice_ax3.contour(da_rho["XC"] * 1e-3, -da_rho["Z"], da_rho.isel(YC=2),
                      levels=[1028.04], colors=["magenta"], linewidths=1.5, linestyles="dashed")
    
    big_Q_ax.axhline(0, label='Equator', ls='-', c='k')
    big_Q_ax.axhline(-250, label='250 km South', ls=':', c='k')
    big_Q_ax.axhline(-500, label='500 km South', ls='-.', c='k')
    big_Q_ax.axhline(1e12, label="$\\gamma^n =$ 28.04", ls="--", c="magenta")
    #big_Q_ax.axhline(-600, label='600 km south', ls='--', c='k')
    big_Q_ax.scatter(90, -250, label='Profile point', marker='o', c='magenta', linewidths=2)

    slice_ax2.axvline(90, c='magenta')

    slice_ax1.set_xlim(0, 300)
    slice_ax2.set_xlim(0, 300)
    slice_ax3.set_xlim(0, 300)

    big_Q_ax.set_aspect('equal')
    big_Q_ax.set_ylim(-1800, 500)

    slice_ax1.invert_yaxis()
    slice_ax2.invert_yaxis()
    slice_ax3.invert_yaxis()

    slice_cb = plt.colorbar(big_Q_cax, cax=slice_cbax,
                            orientation="horizontal")
    slice_cb.formatter.set_useMathText(True)
    slice_cb.set_label("$Q$ (s$^{-3}$)", usetex=True)


    plt.rc('text', usetex=True)
    big_Q_ax.legend(loc='upper right')
    plt.rc('text', usetex=False)

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    fig.savefig(fig_out_path, dpi=dpi)
    
    
if figure4:
    Q_on_rho_path = processed_path / "standard/Q_on_rho.zarr"
    Q_slice_path = processed_path / "standard/Q_slice.zarr"
    rho_slice_path = processed_path / "standard/rho_slice.zarr"
    fig_out_path = figure_path / "Figure4.pdf"
    
    plot_figure4(Q_on_rho_path, Q_slice_path, rho_slice_path, fig_out_path)
    
if figure4NS:
    Q_on_rho_path = processed_path / "NS/Q_on_rhoNS.zarr"
    Q_slice_path = processed_path / "NS/Q_sliceNS.zarr"
    rho_slice_path = processed_path / "NS/rho_sliceNS.zarr"
    fig_out_path = figure_path / "FigureS5.pdf"
    suptitle = "No-slip lateral BC"
    
    plot_figure4(Q_on_rho_path, Q_slice_path, rho_slice_path, fig_out_path,
                 suptitle)
    
if figure4_2km:
    Q_on_rho_path = processed_path / "2km/Q_on_rho2km.zarr"
    Q_slice_path = processed_path / "2km/Q_slice2km.zarr"
    rho_slice_path = processed_path / "2km/rho_slice2km.zarr"
    fig_out_path = figure_path / "FigureS3.pdf"
    suptitle = "2 km resolution"
    
    plot_figure4(Q_on_rho_path, Q_slice_path, rho_slice_path, fig_out_path,
                 suptitle)
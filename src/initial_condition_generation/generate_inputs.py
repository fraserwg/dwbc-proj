import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.linalg import solve
import xarray as xr
import xgcm
import MITgcmutils

run_names = ["NS", "2km", "500m"]
for run_name in run_names:
    logging.info(f"Making input files for {run_name}")
    logging.info('Setting paths')
    base_path = Path('/work/n01/n01/fwg/dwbc-proj')
    base_run_path = base_path / "data/raw/"
    if run_name != "":
        input_pf = base_run_path / f"run{run_name}/input"
    else:
        input_pf = base_run_path / f"run{run_name}/inputTEST"

    precision = 'float32'

    bathymetry_fn = 'bathy'
    uvel_fn = 'uvel'
    vvel_fn = 'vvel'
    Tinit_fn = 'Tinit'
    Tref_fn = 'Tref'
    Sref_fn = 'Sref'
    deltaZ_fn = 'deltaZ'
    Eta_fn = 'Eta'
    nv_fn = 'NVfile'
    sv_fn = 'SVfile'
    umask_fn = 'gammaU'
    vmask_fn = 'gammaV'
    Tmask_fn = 'gammaT'

    Tinit_ffn = input_pf / Tinit_fn
    Tref_ffn = input_pf / Tref_fn
    Sref_ffn = input_pf / Sref_fn
    deltaZ_ffn = input_pf / deltaZ_fn
    bathy_ffn = input_pf / bathymetry_fn
    uvel_ffn = input_pf / uvel_fn
    vvel_ffn = input_pf / vvel_fn
    Eta_ffn = input_pf / Eta_fn
    nv_ffn = input_pf / nv_fn
    sv_ffn = input_pf / sv_fn
    umask_ffn = input_pf / umask_fn
    vmask_ffn = input_pf / vmask_fn
    Tmask_ffn = input_pf / Tmask_fn

    input_pf.mkdir(parents=True, exist_ok=True)

    logging.info('Setting jet and model parameters')
    # Thermodynamics
    T_0 = 30
    alpha_T = 2e-4
    beta_S = None
    S = 0

    # Model domain
    Lx = 600e3
    Ly = 3000e3
    H = 4500

    if run_name == "2km":
        dx = 2e3
        dy = 2e3
    elif run_name == "500m":
        dx = 500
        dy = 500
    elif run_name == "" or run_name == "NS":
        dx = 1e3
        dy = 1e3
    else:
        raise ValueError("'run_name' not recognised")
    
    dz = 10

    nx = int(Lx / dx)
    ny = int(Ly / dy)
    nz = int(H / dz)

    # Jet parameters
    V0 = - 0.2
    xmid = 60e3
    zmid = - 2750
    sigmax = 30e3
    sigmaz = 500

    # Physical parameters
    beta_f = 2.3e-311
    f_0 = 0
    g = 9.81

    # Sponge parameters
    L_Nsponge_dim = 100e3
    L_Ssponge_dim = 300e3
    
    delta_Nsponge_dim = 5e3
    delta_Ssponge_dim = 10e3
    
    L_Nsponge = int(L_Nsponge_dim / dx)
    gammamax= 2e-5
    delta_Nsponge = int(delta_Nsponge_dim / dx)
    
    L_Ssponge = int(L_Ssponge_dim / dx)
    gammamax= 2e-5
    delta_Ssponge = int(delta_Ssponge_dim / dx)


    logging.info('Creating the model grid')
    drF = -dz * np.ones(nz)

    xg = np.linspace(0, Lx, nx)  # Coordinate of left, u points
    xc = xg + dx * 0.5  # Coordinate of right v, eta, rho and h points

    yg = np.linspace(0, Ly, ny) - 2000e3  # Coordinate of left, u points
    yc = yg + dy * 0.5  # Coordinate of right v, eta, rho and h points

    zu = np.cumsum(drF)  # This is the lower coordinate of the vertical cell faces, i.e. the w points
    zl = np.concatenate(([0], zu[:-1]))  # This is the upper coordinate of the vertical cell faces
    z = 0.5 * (zl + zu)  # Vertical coordiante of the velocity points


    ds_grid = xr.Dataset(coords={'XG': xg,
                                'XC': xc,
                                'YG': yg,
                                'YC': yc,
                                'Zu': zu,
                                'Zl': zl,
                                'Z': z})
    grid = xgcm.Grid(ds_grid,
                    periodic=['X'],
                    coords={'Y': {'left': 'YG', 'right': 'YC'},
                            'X': {'left': 'XG', 'center': 'XC'},
                            'Z': {'left': 'Zu', 'right': 'Zl', 'center': 'Z'}})

    ds_input = ds_grid.copy()
    ds_input.transpose(..., 'YC', 'XC')

    ds_input['deltaZ'] = dz * np.ones(nz)


    logging.info('Creating the model bathymetry')
    lambda_x = 1 / 50e3
    H0 = - H / (np.exp(-lambda_x * ds_grid['XC'][-1].values) - 1)
    ds_input['bathymetry'] = xr.zeros_like(ds_input['YC']) + H0 * (np.exp(-lambda_x * ds_grid['XC']) - 1)
        
    ds_input['bathymetry'][:, 0] = 0



    logging.info('Creating the velocity profile')
    def VVEL(Z, Y, X):
        VX = np.exp(-np.square(X - xmid) / 2 / np.square(sigmax))
        VZ = np.exp(-np.square(Z - zmid) / 2 / np.square(sigmaz))
        return V0 * VZ * xr.ones_like(Y) * VX 

    ds_input['VVEL'] = VVEL(ds_input['Z'], ds_input['YG'], ds_input['XC'])
    ds_input['NVfile'] = ds_input['VVEL'].isel(YG=-1)
    ds_input['SVfile'] = ds_input['VVEL'].isel(YG=0)

    ds_input['UVEL'] = xr.DataArray(np.zeros((nz, ny, nx)),
                                    dims={'Z': ds_input['Z'], 'YC': ds_input['YC'], 'XG': ds_input['XG']})
    ds_input['EUfile'] = ds_input['UVEL'].isel(XG=-1)


    logging.info('Creating the reference density profile')
    gamma_n_path = base_path / 'data/processed/climatological_gamman.nc'
    gamma_n_init = xr.open_dataset(gamma_n_path, decode_times=False)
    rho_ref = interp1d(gamma_n_init['depth'], gamma_n_init['mn_gamma_n'], bounds_error=False, fill_value='extrapolate')(ds_input['Z']) + 1000

    # Turn the original profile into a DataArray
    da_rho_ref = xr.DataArray(rho_ref, coords={'Z': ds_input['Z']}, dims=('Z'))

    # Set the depths at which to switch profiles
    z_therm = -150  # Depth at which the lienar surface profile ends
    z_bound = -1500  # Depth at which the linear deep profile starts

    surf_rho = da_rho_ref.sel(Z=slice(0, z_therm))
    mid_rho = da_rho_ref.sel(Z=slice(z_therm, z_bound))
    deep_rho = da_rho_ref.sel(Z=slice(z_bound, -H))

    # Use observations to linearly fit the deep profile
    k2, c2, _, _, _ = linregress(deep_rho['Z'], deep_rho)
    interp_deep_rho = k2 * deep_rho['Z'] + c2

    # Use observations to linearly fit the surface profile
    k1, c1, _, _, _ = linregress(surf_rho['Z'], surf_rho)
    interp_surf_rho = k1 * surf_rho['Z'] + c1

    # Fit a second order polynomial to the middle density profile
    # Match the density and its derivative at the boundary layer
    # Match the the density, but not its derivative at the the thermocline
    rho_therm = k1 * z_therm + c1
    rho_bound = k2 * z_bound + c2
    A = np.array([[0, 0, 1],[z_bound ** 2, z_bound, 1],[2 * z_bound, 1, 0]])
    bprime = np.array([rho_therm, rho_bound, k2])
    a, b, c = solve(A, bprime)  # rho = a * z ** 2 + b * z + c
    apher, bether, gammow = a, b, c
    interp_mid_rho = a * mid_rho['Z'] * mid_rho['Z']  + b * mid_rho['Z'] + c
    ds_input['rho_ref'] = xr.concat([interp_surf_rho, interp_mid_rho, interp_deep_rho], dim='Z')

    rho_0 = ds_input['rho_ref'][0].values

    logging.info('rho_0 = {}'.format(rho_0))

    ds_input['T_ref'] = (1 - ds_input['rho_ref'] / rho_0) / alpha_T + T_0
    ds_input['T_init'] = ds_input['T_ref'].broadcast_like(ds_input['YC']).broadcast_like(ds_input['XC'])

    ds_input['S_ref'] = xr.zeros_like(ds_input['T_ref'])


    logging.info('Setting up sponges')

    mid_Nsponge = L_Nsponge / 2
    Nsponge = np.arange(0, L_Nsponge + 1)
    gamma_Nsponge = (np.tanh((Nsponge - mid_Nsponge) / delta_Nsponge) - np.tanh(- mid_Nsponge / delta_Nsponge)) * gammamax / np.tanh(mid_Nsponge / delta_Nsponge) / 2

    mid_Ssponge = L_Ssponge / 2
    Ssponge = np.arange(0, L_Ssponge + 1)
    gamma_Ssponge = (np.tanh((Ssponge - mid_Ssponge) / delta_Ssponge) - np.tanh(- mid_Ssponge / delta_Ssponge)) * gammamax / np.tanh(mid_Ssponge / delta_Ssponge) / 2
    gamma_Ssponge = gamma_Ssponge[::-1]

    L_interior = int(Ly / dy - L_Ssponge - L_Nsponge - 2)
    gamma_interior = np.zeros(L_interior)
    gamma_whole = np.concatenate((gamma_Ssponge, gamma_interior, gamma_Nsponge))



    ds_input['gammaV'] = xr.DataArray(gamma_whole, dims=('YG')).broadcast_like(ds_input['VVEL']) #* ds_input['bool_land_mask']  # Should strictly be on a YG grid
    ds_input['gammaV'] = ds_input['gammaV'].transpose('Z', 'YG', 'XC')

    ds_input['gammaU'] = xr.DataArray(np.zeros((nz, ny, nx)),
                                    dims={'Z': ds_input['Z'], 'YC': ds_input['YC'], 'XG': ds_input['XG']})

    ds_input['gammaT'] = xr.DataArray(gamma_whole, dims=('YG')).broadcast_like(ds_input['VVEL'])


    logging.info('Saving input data')
    MITgcmutils.wrmds(str(uvel_ffn), ds_input['UVEL'].values, dataprec=precision)
    MITgcmutils.wrmds(str(vvel_ffn), ds_input['VVEL'].values, dataprec=precision)

    MITgcmutils.wrmds(str(nv_ffn), ds_input['NVfile'].values, dataprec=precision)
    MITgcmutils.wrmds(str(sv_ffn), ds_input['SVfile'].values, dataprec=precision)

    MITgcmutils.wrmds(str(bathy_ffn), ds_input['bathymetry'].values, dataprec=precision)

    MITgcmutils.wrmds(str(deltaZ_ffn), ds_input['deltaZ'].values, dataprec=precision)

    MITgcmutils.wrmds(str(Tinit_ffn), ds_input['T_init'].values, dataprec=precision)
    MITgcmutils.wrmds(str(Tref_ffn), ds_input['T_ref'].values, dataprec=precision)
    MITgcmutils.wrmds(str(Sref_ffn), ds_input['S_ref'].values, dataprec=precision)

    MITgcmutils.wrmds(str(umask_ffn), ds_input['gammaU'].values, dataprec=precision)
    MITgcmutils.wrmds(str(vmask_ffn), ds_input['gammaV'].values, dataprec=precision)
    MITgcmutils.wrmds(str(Tmask_ffn), ds_input['gammaT'].values, dataprec=precision)
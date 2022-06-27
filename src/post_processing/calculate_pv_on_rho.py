import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path
import os

logging.info('Importing third party python libraries')
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import xarray as xr
import xmitgcm
from xgcm import Grid
import f90nml

logging.info('Importing custom python libraries')
import pvcalc


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/dwbc-proj')

log_path = base_path / 'src/post_processing/.tmp/slurm-out'
dask_worker_path = base_path / 'src/post_processing/.tmp/dask-worker-space'

env_path = Path('/work/n01/n01/fwg/dwbc-proj/dwbc-proj/bin/activate')
run_path = base_path / 'data/raw/run'
processed_path = base_path / 'data/processed'
out_path = processed_path / 'PV_on_rho2'

# Check paths exist etc.
if not log_path.exists(): log_path.mkdir()
assert run_path.exists()


logging.info('Initialising the dask cluster')
# Set up the dask cluster
scluster = SLURMCluster(queue='standard',
                        project="n01-siAMOC",
                        job_cpu=128,
                        log_directory=log_path,
                        local_directory=dask_worker_path,
                        cores=64,
                        processes=8,  # Can change this
                        memory="256 GiB",
                        header_skip= ['#SBATCH --mem='],  
                        walltime="01:00:00",
                        death_timeout=60,
                        interface='hsn0',
                        job_extra=['--qos=standard'], #, '--reservation=shortqos'],
                        env_extra=['module load cray-python',
                                   'source {}'.format(str(env_path.absolute()))]
                       )

njobs = 8
client = Client(scluster)
scluster.scale(jobs=njobs)


logging.info('Reading in model parameters from the namelist')
with open(run_path / 'data') as data:
    data_nml = f90nml.read(data)

delta_t = data_nml['parm03']['deltat']
f0 = data_nml['parm01']['f0']
beta = data_nml['parm01']['beta']
no_slip_bottom = data_nml['parm01']['no_slip_bottom']
no_slip_sides = data_nml['parm01']['no_slip_sides']


logging.info('Reading in the model dataset')
ds = xmitgcm.open_mdsdataset(run_path,
                             prefix=['ZLevelVars', 'IntLevelVars'],
                             delta_t=delta_t,
                             geometry='cartesian',
                             chunks=600
                            )


logging.info('Calculating the potential vorticity')
grid = pvcalc.create_xgcm_grid(ds)
ds['drL'] = pvcalc.create_drL_from_dataset(ds)
ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])
ds['b'] = pvcalc.calculate_buoyancy(ds['rho'])

ds['zeta_x'], ds['zeta_y'], ds['zeta_z'] = pvcalc.calculate_curl_velocity(ds['UVEL'],
                                                                          ds['VVEL'],
                                                                          ds['WVEL'],
                                                                          ds,
                                                                          grid,no_slip_bottom,
                                                                          no_slip_sides
                                                                         )


ds['Q'] = pvcalc.calculate_C_potential_vorticity(ds['zeta_x'],
                                                 ds['zeta_y'],
                                                 ds['zeta_z'],
                                                 ds['b'],
                                                 ds,
                                                 grid,
                                                 beta,
                                                 f0
                                                 )


logging.info('Creating land masks')
ds['bool_land_mask'] = xr.where(-ds['Depth'] <= ds['Z'], 1, 0)
ds['nan_land_mask'] = xr.where(-ds['Depth'] <= ds['Z'], 1, np.nan)
slice_nan_land_mask = ds['nan_land_mask'].isel(YC=0).values

ds['bool_land_mask'].isel(YC=0).to_netcdf(processed_path / 'bool_land_mask.nc')

logging.info('Calculating potential vorticity on a density level')
Q_t = ds['Q']
rho_t = grid.interp(ds['rho'], ['X', 'Y', 'Z'], boundary='extend', to={'X': 'left', 'Y': 'left', 'Z': 'right'})
rho_t.name = 'rho'
target_rho_levels = ds['rhoRef'].sel(Z=[-2750], method='nearest').values  # -2750 m is the jet core

# Create a 2D mask which sets land points to nan
masked_rho = ds['rho'] * ds['nan_land_mask']
land_point_zeros = xr.where(masked_rho >= target_rho_levels, 1, 0).sum('Z') # Where zero, land, else water
target_rho_mask = xr.where(land_point_zeros == 0, np.nan, 1)

Q_on_rho = grid.transform(Q_t.chunk({'Zl': -1}),
                          'Z',
                          target_rho_levels,
                          target_data=rho_t.chunk({'Zl': -1}),
                          method='linear',
                         ) * grid.interp(target_rho_mask, ['X', 'Y'], boundary='extend')


logging.info('Saving potential vorticity on a density level dataset')

# We will save the data in blocks, 5 time steps at a time
ds_Q = Q_on_rho.chunk({'time': 1, 'YG': 3000, 'XG': 600, 'rho': 1}).to_dataset(name='Q')
time_block = 8


logging.info('Saving iters = 0 to {}'.format(time_block - 1))
ds_Q.isel(time=slice(0, time_block)).to_zarr(out_path)
start_iter = time_block
    
end_iter = int(ds_Q.sizes['time'])
for tn in range(start_iter, end_iter, time_block):
    ds_Qt = ds_Q.isel(time=slice(tn, tn + time_block))
    logging.info('Saving iters = {} to {}'.format(tn, tn + time_block - 1))
    ds_Qt.to_zarr(out_path, append_dim='time')
    scluster.scale(jobs=njobs)

#logging.info('Processing complete')
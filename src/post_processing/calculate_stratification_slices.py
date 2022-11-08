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
out_path = processed_path / 'stratification_slices.zarr'

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
                        processes=16,  # Can change this
                        memory="256 GiB",
                        header_skip= ['#SBATCH --mem='],  
                        walltime="01:00:00",
                        death_timeout=60,
                        interface='hsn0',
                        job_extra=['--qos=standard'], #, '--reservation=shortqos'],
                        env_extra=['module load cray-python',
                                   'source {}'.format(str(env_path.absolute()))]
                       )

njobs = 4
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
ds_full = xmitgcm.open_mdsdataset(run_path,
                                  prefix=['ZLevelVars'],
                                  delta_t=delta_t,
                                  geometry='cartesian',
                                  chunks=600
                                 )


logging.info('Calculating the stratification')
# Ylats = np.arange(-700e3, 600e3, 100e3)
Ylats = [-0, -250e3, -500e3]

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    ds = ds_full.sel(YC=Ylats, YG=Ylats,  method='nearest')
grid = pvcalc.create_xgcm_grid(ds)
ds['drL'] = pvcalc.create_drL_from_dataset(ds)
ds['rho'] = pvcalc.calculate_density(ds['RHOAnoma'], ds['rhoRef'])
ds['b'] = pvcalc.calculate_buoyancy(ds['rho'])

_, _, ds['db_dz'] = pvcalc.calculate_grad_buoyancy(ds['b'],
                                                   ds,
                                                   grid)

da_strat = ds['db_dz'] * grid.interp(ds['maskC'], 'Z', boundary='extend', to='right')
ds_strat = da_strat.chunk({'time': 10, 'YC': 3, 'XC': 600, 'Zl': 450}).to_dataset(name='db_dz')

logging.info('Saving stratification dataset')
ds_strat.to_zarr(out_path)

logging.info('Processing complete')
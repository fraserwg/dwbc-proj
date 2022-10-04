
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                )


logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import xmitgcm
import f90nml
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import zarr
import numpy as np


logging.info('Setting file path')
base_path = Path('/work/n01/n01/fwg/dwbc-proj')
assert base_path.exists()

env_path = base_path / 'dwbc-proj/bin/activate'
assert env_path.exists()

src_path = base_path / 'src'
assert src_path.exists()

log_path = src_path / '.tmp'
log_path.mkdir(exist_ok=True)

dask_worker_path = log_path / 'dask-worker-space'
dask_worker_path.mkdir(exist_ok=True)

run_name = 'run'
run_path = base_path / 'data/raw/' / run_name
out_path = run_path / (run_name + '.zarr')
assert run_path.exists()

logging.info('Initialising the dask cluster')
# Set up the dask cluster
scluster = SLURMCluster(queue='standard',
                        project="n01-GEOM",
                        job_cpu=128,
                        log_directory=log_path,
                        local_directory=dask_worker_path,
                        cores=64,
                        processes=16,  # Can change this
                        memory="512 GiB",
                        header_skip= ['#SBATCH --mem='],  
                        walltime="04:00:00",
                        death_timeout=60,
                        interface='hsn0',
                        job_extra=['--qos=highmem', '--partition=highmem'],
                        env_extra=['module load cray-python',
                                'source {}'.format(str(env_path.absolute()))]
                    )

client = Client(scluster)
scluster.scale(jobs=6)
logging.info(scluster)

logging.info('Reading in model parameters from the namelist')
with open(run_path / 'data') as data:
    data_nml = f90nml.read(data)
    
with open(run_path / 'data.diagnostics') as data:
    diag_nml = f90nml.read(data)

delta_t = data_nml['parm03']['deltat']
f0 = data_nml['parm01']['f0']
beta = data_nml['parm01']['beta']
no_slip_bottom = data_nml['parm01']['no_slip_bottom']
no_slip_sides = data_nml['parm01']['no_slip_sides']


logging.info('Reading in the model dataset')
ds = xmitgcm.open_mdsdataset(str(run_path),
                            prefix=['ZLevelVars', 'IntLevelVars'],
                            delta_t=delta_t,
                            geometry='cartesian',
                            chunks={'k': 50, 'k_l': 50})

def create_encoding_for_ds(ds, clevel):
    compressor = zarr.Blosc(cname="zstd", clevel=clevel, shuffle=2)
    enc = {x: {"compressor": compressor} for x in ds}
    return enc

logging.info('Creating compression encoding')
enc = create_encoding_for_ds(ds, 9)
logging.info('Saving to compressed zarr dataset')
logging.info(out_path)

iter_starts = int(diag_nml['diagnostics_list']['timephase'][0] / delta_t)

time_slice_size = 10
for it in np.arange(0, ds.dims['time'], time_slice_size) :
    logging.info('Compressing iters {} to {}'.format(ds['iter'].isel(time=it),
                                                     ds['iter'].isel(time=it + time_slice_size)))
    if ds['iter'].isel(time=it) != iter_starts: #it != 0:
        ds.isel(time=slice(it, it + time_slice_size)).to_zarr(out_path,
                                                              mode='a', append_dim='time')
    else:
        ds.isel(time=slice(it, it + time_slice_size)).to_zarr(out_path, 
                                                              mode='w', encoding=enc)
        
    iterations = ds['iter'].isel(time=slice(it, it + time_slice_size)).values
    
    logging.info(f'Removing iters {ds["iter"].isel(time=it)} to {ds["iter"].isel(time=it + time_slice_size)}')
    for iteration in iterations:
        for prefix in ['ZLevelVars', 'IntLevelVars']:
            for suffix in ['data', 'meta']:
                file_name = run_path / (prefix + '.{:010d}.'.format(iteration) + suffix)
                file_name.unlink()
            
    
logging.info('Compression complete, tidying up')

client.close()
scluster.close()
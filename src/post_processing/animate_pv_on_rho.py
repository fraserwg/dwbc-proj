import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path
import os

logging.info('Importing third party python libraries')
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import dask
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import xarray as xr
import f90nml

logging.info('Importing custom python libraries')


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/dwbc-proj')

log_path = base_path / 'src/post_processing/.tmp/slurm-out'
dask_worker_path = base_path / 'src/post_processing/.tmp/dask-worker-space'

env_path = Path('/work/n01/n01/fwg/dwbc-proj/dwbc-proj/bin/activate')
run_path = base_path / 'data/raw/run'
processed_path = base_path / 'data/processed'
in_path = processed_path / 'PV_on_rho'
frame_path = base_path / 'figures/PV_frames'

dpi = 342

# Check paths exist etc.
if not log_path.exists(): log_path.mkdir()
if not frame_path.exists(): frame_path.mkdir()
assert run_path.exists()
assert in_path.exists()


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
                        walltime="00:20:00",
                        death_timeout=60,
                        interface='hsn0',
                        job_extra=['--qos=short', '--reservation=shortqos'],
                        env_extra=['module load cray-python',
                                   'source {}'.format(str(env_path.absolute()))]
                       )

njobs = 1
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


logging.info('Reading in the PV dataset')
ds = xr.open_dataset(in_path, engine='zarr')

Qlim = 4e-11

@dask.delayed()
def plot_pv(nt):
    cmo.curl.set_bad('grey')
    Q = ds['Q'].isel(time=nt).squeeze()
    X, Y = ds['XG'] * 1e-3, ds['YG'] * 1e-3
    tdays = float(Q['time'].values) * 1e-9 / 24 / 60 / 60
    
    fig, ax = plt.subplots(figsize=(6, 12))
    cax = ax.pcolormesh(X, Y, Q,
                        shading='nearest',
                        cmap=cmo.curl,
                        vmin=-Qlim,
                        vmax=Qlim)
    cb = fig.colorbar(cax, ax=ax, label='Q (s$^{-3}$)', aspect=40)
    cb.formatter.set_useMathText(True)
    
    ax.set_xlabel('Longitude (km)')
    ax.set_ylabel('Lattitude (km)')
    ax.set_title('t = {:.2f} days'.format(tdays))
    ax.set_ylim(-1500, 500)
    ax.set_aspect('equal')
    #fig.tight_layout()
    
    fig.savefig(frame_path / 'pv{:03d}'.format(nt), dpi=dpi)
    plt.close(fig)

logging.info('Creating list of tasks')
collection = [plot_pv(nt) for nt in range(ds.dims['time'])]

logging.info('Computing tasks')
dask.compute(collection)

# ffmpeg -framerate 15 -i PV_frames/pv%03d.png -pix_fmt yuv420p out.mp4
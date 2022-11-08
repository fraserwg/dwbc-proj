from os import getcwd
from pathlib import Path

import numpy as np
from numba import njit
import xarray as xr

import shutil

base_path = (Path(getcwd()) / '../..').resolve()
assert base_path.exists()
processed_path = base_path /'data/processed'
env_path = base_path / 'dwbc-proj/bin/activate'
log_path = base_path / 'src/toy_model/.tmp/log.out'
dask_worker_path = base_path / 'src/toy_model/.tmp/'
out_path = processed_path / 'toy_strat_data_full.zarr'

print("base_path = Path('{}')".format(base_path))

dz = 2.5
dx = 1e3
nt = 10080 + 1

z_1d = np.arange(-600, 0 + dz, dz)
x_1d = np.arange(-50e3, 50e3 + dx, dx)

x, z = np.meshgrid(x_1d, z_1d)

Nsq_0 = 1e-6
Msq_0 = 0
kappa = 1e-5
dt = 240

kz = 2 * np.pi / 200
sigma = 25e3
sigma2 =  np.square(sigma)

psi = np.exp(-np.square(x)  / 2 / sigma2) * np.sin(kz * z)
w0 = - (x  / sigma2) * np.exp(-np.square(x) / 2 / sigma2) * np.sin(kz * z)
u0 = - kz * np.exp(-np.square(x) / 2 / sigma2) * np.cos(kz * z)

b0 = Nsq_0 * z + Msq_0 * x

@njit()
def forcing_2d(t, b):
    # db_dz
    b_plus = b.copy()
    b_plus[:-1] = b[1:]
    #np.roll(b, shift=-1, axis=0)
    b_minus = b.copy()
    b_minus[1:] = b[:-1]
    #np.roll(b, shift=1, axis=0)

    db_dz = (b_plus - b_minus) / (2 * dz)
    db_dz[-1, :] = db_dz[-2, :]
    db_dz[0, :] = db_dz[1, :]
    
    # d2b_dz2
    d2b_dz2 = (b_plus - 2 * b + b_minus) / dz / dz
    d2b_dz2[-1, :] = 0
    d2b_dz2[0, :] = 0    
    
    # db_dx
    b_plus = b.copy()
    b_plus[:, :-1] = b[:, 1:]
    #np.roll(b, shift=-1, axis=1)
    b_minus = b.copy()
    b_minus[:, 1:] = b[:, :-1]
    #np.roll(b, shift=1, axis=1)
    db_dx = (b_plus - b_minus) / (2 * dx)
    db_dx[:, -1] = db_dx[:, -2]
    db_dx[:, 0] = db_dx[:, 1]

    w = w0
    u = u0
    
    kappa_mask = np.where(db_dz > 0, kappa, kappa * 5e2)
    f = kappa_mask * d2b_dz2 - w * db_dz - u * db_dx
    return f

def db_dz_to_data_array(db_dz):
    da = xr.DataArray(db_dz[..., np.newaxis],
                      coords={'Z': z_1d, 'X': x_1d, 'time': [tn3]})
    return da

t0 = 0

tn3 = 0

if out_path.exists():
    shutil.rmtree(out_path)
                  
da = db_dz_to_data_array(Nsq_0 * np.ones_like(z))
da.to_dataset(name='db_dz').to_zarr(out_path)

b1 = b0 + dt * forcing_2d(t0, b0)
t1 = t0 + dt

b2 = b1 + dt * (3 / 2 * forcing_2d(t1, b1) + 1 / 2 * forcing_2d(t0, b0))
t2 = t1 + dt

@njit()
def ab3_iterate_2d(bn0, bn1, bn2, tn3):
    tn2 = tn3 - dt
    tn1 = tn2 - dt
    tn0 = tn2 - dt
    bn3 = bn2 + dt * (23 / 12 * forcing_2d(tn2, bn2) - 16 / 12 * forcing_2d
                      (tn1, bn1) + 5 / 12 * forcing_2d(tn0, bn0))
    return bn3


bn0, bn1, bn2 = b0, b1, b2
tn3 = t2 + dt
bn3 = bn2

b_plus = np.roll(bn3, shift=-1, axis=0)
b_minus = np.roll(bn3, shift=1, axis=0)
db_dz = (b_plus - b_minus) / (2 * dz)
db_dz[-1, :] = db_dz[-2, :]
db_dz[0, :] = db_dz[1, :]

for i in range(2, nt):
    bn3 = ab3_iterate_2d(bn0, bn1, bn2, tn3)
    bn0, bn1, bn2 = bn1, bn2, bn3
    tn3 += dt
    
    b_plus = np.roll(bn3, shift=-1, axis=0)
    b_minus = np.roll(bn3, shift=1, axis=0)
    db_dz = (b_plus - b_minus) / (2 * dz)
    db_dz[-1, :] = db_dz[-2, :]
    db_dz[0, :] = db_dz[1, :]


    if i % 90 == 0:
        da = db_dz_to_data_array(db_dz)
        da.to_dataset(name='db_dz').to_zarr(out_path, append_dim='time')

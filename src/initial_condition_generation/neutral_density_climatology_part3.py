import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO
                   )

logging.info('Importing standard python libraries')
from pathlib import Path

logging.info('Importing third party python libraries')
import xarray as xr
import gsw
import numpy as np
from scipy.io import loadmat


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/dwbc/dwbc-proj')
data_path = base_path / 'data'


logging.info('Loading subsetted stp')
ds_subset = xr.open_datset(data_path '/interim/subsetted_stp.nc')


logging.info('Loading gamma_n and converting to xr.Dataset')
gamma_n = loadmat('gamma_n.mat')['gamma_n'].squeeze()
ds_subset['gamma_n'] = (('depth', 'lat', 'lon'), gamma_n)
ds_subset['mn_gamma_n'] = ds_subset['gamma_n'].mean(['lat', 'lon'])


logging.info('Saving climatological gamma_n')
data_vars_to_remove = list(ds_subset.data_vars)
data_vars_to_remove.remove('s_an')
data_vars_to_remove.remove('t_an')
data_vars_to_remove.remove('pressure')
data_vars_to_remove.remove('mn_gamma_n')
data_vars_to_remove.remove('gamma_n')

ds_subset = ds_subset.drop_vars(data_vars_to_remove)

attrs = {'title': 'Neutral density in the Tropical Atlantic calculated from World Ocean Atlas 2018 1981-2010 1.00 degree data',
         'summary': 'Neutral density for use in models',
         'date_created': '08-04-2021',
         'author': 'Fraser W Goldsworth',
         'scripts_used': ['calc_gamma.m', 'DensityCalculator.ipynb']}

ds_subset.attrs = attrs

ds_subset.to_netcdf(data_path / 'processed/climatological_gamman.nc')

logging.info('Computation complete')
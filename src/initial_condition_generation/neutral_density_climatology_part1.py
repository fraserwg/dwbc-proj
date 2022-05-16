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
from scipy.io import savemat


logging.info('Setting paths')
base_path = Path('/work/n01/n01/fwg/dwbc/dwbc-proj')
data_path = base_path / 'data'


logging.info('Opening remote climatological dataset')
sal_opendap = 'https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa/salinity/decav81B0/1.00/woa18_decav81B0_s00_01.nc'
temp_opendap = 'https://www.ncei.noaa.gov/thredds-ocean/dodsC/ncei/woa/temperature/decav81B0/1.00/woa18_decav81B0_t00_01.nc'
ds = xr.open_mfdataset([sal_opendap, temp_opendap], decode_times=False)
ds['depth'] = ds['depth'] * -1
ds['pressure'] = gsw.p_from_z(ds['depth'], ds['lat'])


logging.info('Archiving climatological dataset')
ds.to_netcdf(data_path / 'raw/climatological_stp.nc')


logging.info('Subsetting climatological dataset')
ds_subset = ds.sel(lon=slice(-51, -30), lat=slice(-5, 5))


logging.info('Saving subsetted dataset')
ds_subset.to_netcdf(data_path / 'interim/subsetted_stp.nc')
savemat(data_path / 'interim/stp.mat',
        {'lat': ds_subset['lat'],
         'lon': ds_subset['lon'],
         't_an': ds_subset['t_an'],
         's_an': ds_subset['s_an'],
         'p': ds_subset['pressure']}
       )


logging.info('Copmutation complete. Please run part 2.')
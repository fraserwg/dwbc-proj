# dwbc-proj
This git repository contains code and post-processing scripts used to produce the data and figures in Goldsworth et al. (2022).

## Requirements
The file `requirements.txt` details the dependencies of this repository. A python virtual environment should easily be creatable from it using `pip`. 

## Data
All processed data and selected raw data can be downloaded from !Insert zenodo link!. 

## Code
The src directory contains MITgcm configuration files, initial condition generation codes and post-processing codes

### initial_condition_generation
To run the codes in this folder, you must first download and unzip the data folder into the `dwbc-proj` base folder.

The `neutral_density_climatology_partX` scripts produce a climatological neutral density profile on which the model's initial conditions are based. You will need a copy of the [neutral density matlab toolbox](https://www.teos-10.org/preteos10_software/neutral_density.html) installed.

The `generate_inputs.py` script generates the binary input files used by the model.

### mitgcm-models
This folder contains the MITgcm model configuration files. To build the model you will need to download a copy of the [MITgcm](https://github.com/MITgcm/MITgcm) – this study used checkpoint 68i. Build instructions are available [here](https://mitgcm.readthedocs.io/en/latest/). To run the model you will need the initial condition files which are contained in the data folder.

### post_processing
`pvcalc.py` is a library for calculating potential vorticity from the model.

The `subset_data.py` file reads in the raw binary model output, processes it and subsets it, and saves the outpt as netCDF files ready for plotting.

`plot_figures.py` reads in the processed data and creates the figures published in Goldsworth et al. (2022).

## Figures
This folder contains the `pdf` figures from the paper

## Reports
This folder contains a pre-print of Goldsworth et al. (2022).

## References
Goldsworth, F. W., Johnson, H. L., Marshall, D. P. (2022) *Density staircases generated by symmetric instability in a cross-equatorial deep western boundary current* (in preparation)

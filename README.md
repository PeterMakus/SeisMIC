# SeisMIC [![Build Status](https://github.com/PeterMakus/SeisMIC/actions/workflows/test_on_push.yml/badge.svg)](https://github.com/PeterMakus/SeisMIC/actions/workflows/test_on_push.yml?branch=main) [![Documentation Status](https://github.com/PeterMakus/SeisMIC/actions/workflows/deploy_gh_pages.yml/badge.svg)](https://github.com/PeterMakus/SeisMIC/actions/workflows/deploy_gh_pages.yml) [![License: EUPL v1.2](https://img.shields.io/badge/License-EUPL-1.2%20v3-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/introduction-eupl-licence) [![codecov](https://codecov.io/gh/PeterMakus/SeisMIC/branch/main/graph/badge.svg?token=DYVHODB6LN)](https://codecov.io/gh/PeterMakus/SeisMIC)

## Monitoring Velocity Changes using Ambient Seismic Noise
SeisMIC (**Seismological Monitoring using Interferometric Concepts**) is a python software that emerged from the miic library. **SeisMIC** provides functionality to apply some concepts of seismic interferometry to different data of elastic waves. Its main use case is the monitoring of temporal changes in a mediums Green's Function (i.e., monitoring of temporal velocity changes).

**SeisMIC** will handle the whole workflow to create velocity-change time-series including:
+ Downloading raw data
+ Adaptable preprocessing of the waveform data
+ Computating cross- and/or autocorrelations
+ Plotting tools for correlations
+ Database management of ambient seismic noise correlations
+ Adaptable postprocessing of correlations
+ Computation of velocity change time series
+ Plotting of dv time-series

**SeisMIC** handles correlations and data in an [ObsPy](https://github.com/obspy/obspy)-like manner.

## Installation of this package

Download this package via GitHub and install it via bash terminal (the few steps shown below) or using the graphical user interface

```bash
# Download via wget or web-browser
wget https://github.com/PeterMakus/SeisMIC/tree/main

# unzip the package
unzip main.zip

# Change directory to the same directory that this repo is in (i.e., same directory as setup.py)
cd SeisMIC-main  # That's the standard name the folder should have

# Create the conda environment and install dependencies
conda env create -f environment.yml

# Activate the conda environment
conda activate seismic

# Install your package
pip install -e .
```

## Getting started
Access SeisMIC's documentation [here](https://petermakus.github.io/SeisMIC/index.html).

SeisMIC comes with a few tutorials (Jupyter notebooks). You can find those in the `examples/` directory.

## Reporting Bugs / Contact the developers
This version is an early release. If you encounter any issues or unexpected behaviour, please [open an issue](https://github.com/PeterMakus/SeisMIC/issues/new) here on GitHub or [contact the developers](mailto:makus@gfz-potsdam.de).

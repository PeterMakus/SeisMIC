<img src="https://github.com/PeterMakus/SeisMIC/raw/main/docs/source/figures/seismic_logo_small.png" alt="SeisMIC logo" width="600"/>

[![Build Status](https://github.com/PeterMakus/SeisMIC/actions/workflows/pytest.yaml/badge.svg)](https://github.com/PeterMakus/SeisMIC/actions/workflows/pytest.yaml?branch=main)
[![Documentation Status](https://github.com/PeterMakus/SeisMIC/actions/workflows/deploy_gh_pages.yml/badge.svg)](https://github.com/PeterMakus/SeisMIC/actions/workflows/deploy_gh_pages.yml)
[![License: EUPL v1.2](https://img.shields.io/badge/license-EUPL--1.2-blue)](https://joinup.ec.europa.eu/collection/eupl/introduction-eupl-licence)
[![codecov](https://codecov.io/gh/PeterMakus/SeisMIC/branch/main/graph/badge.svg?token=DYVHODB6LN)](https://codecov.io/gh/PeterMakus/SeisMIC)
[![DOI](https://img.shields.io/badge/DOI-10.26443/seismica.v3i1.1099-blue)](https://doi.org/10.26443/seismica.v3i1.1099)
[![PyPI](https://img.shields.io/pypi/v/seismic)](https://pypi.org/project/seismic/)
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

## Monitoring Velocity Changes using Ambient Seismic Noise
SeisMIC (**Seismological Monitoring using Interferometric Concepts**) is a python software that emerged from the miic library. **SeisMIC** provides functionality to apply some concepts of seismic interferometry to different data of elastic waves. Its main use case is the monitoring of temporal changes in a mediums Green's Function (i.e., monitoring of temporal velocity changes).

<img src="https://github.com/PeterMakus/SeisMIC/raw/main/docs/source/figures/zhupanov_dv.png" alt="A velocity change time series" width="800"/>

**SeisMIC** will handle the whole workflow to create velocity-change time-series including:
+ Downloading raw data
+ Adaptable preprocessing of the waveform data
+ Computating cross- and/or autocorrelations
+ Plotting tools for correlations
+ Database management of ambient seismic noise correlations
+ Adaptable postprocessing of correlations
+ Computation of velocity change (dv/v) time series
+ Postprocessing of dv/v time series
+ Plotting of dv/v time-series
+ Inverting dv/v onto a spatial grid

**SeisMIC** handles correlations and data in an [ObsPy](https://github.com/obspy/obspy)-like manner.

## Installation of this package

### Installation from PyPi (pip install)
**SeisMIC** is  now deployed on PyPi and can simply be installed using:

```bash
# We recommend installing mpi4py from the conda-forge channel instead of PyPi
conda install -c conda-forge mpi4py

pip install seismic

# If you want to execute the tutorials, you will have to install jupyter as well
pip install jupyter
```
### Installation from Source Code
To obtain the lates features, you can install SeisMIC from its source code, available on GitHub.

**Developers should download the ``dev`` branch**

Download this package via GitHub and install it via bash terminal (the few steps shown below) or using the graphical user interface

```bash
# Download via wget or web-browser
wget https://github.com/PeterMakus/SeisMIC/archive/refs/heads/main.zip

# For developers download the dev branch
wget https://github.com/PeterMakus/SeisMIC/archive/refs/heads/dev.zip

# unzip the package
unzip main.zip  # or dev.zip

# Change directory to the same directory that this repo is in (i.e., same directory as setup.py)
cd SeisMIC-main  # That's the standard name the folder should have

# Create the conda environment and install dependencies
conda install -c conda-forge mpi4py
conda env create -f environment.yml

# Activate the conda environment
conda activate seismic

# Install the package in editable mode
pip install -e .

# If you want to execute the tutorials, you will have to install jupyter as well
pip install jupyter
```

## Getting started
Access SeisMIC's documentation [here](https://petermakus.github.io/SeisMIC/index.html).

SeisMIC comes with a few tutorials (Jupyter notebooks). You can find those in the `examples/` directory.

## Acknowledging the Use of SeisMIC in your Work
If you should use SeisMIC to create published scientific content please cite the SeisMIC paper: [Makus, P., & Sens-Schönfelder, C. (2024). SeisMIC-an Open Source Python Toolset to Compute Velocity Changes from Ambient Seismic Noise.](https://doi.org/10.26443/seismica.v3i1.1099).

## Reporting Bugs / Contact the developers
This version is an early release. If you encounter any issues or unexpected behaviour, please [open an issue](https://github.com/PeterMakus/SeisMIC/issues/new/choose) here on GitHub.

## Questions?
If you have any questions that do not require any changes in the source code, please use the [discussions feature](https://github.com/PeterMakus/SeisMIC/discussions)

## Contributing
Thank you for contributing to SeisMIC! Have a look at our [guidelines for contributors](https://github.com/PeterMakus/SeisMIC/blob/main/CONTRIBUTING.md)

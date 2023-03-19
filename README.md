<img src="https://github.com/PeterMakus/SeisMIC/tree/main/docs/source/figures/seismic_logo_small_nobg.png" alt="SeisMIC logo" width="600"/>

[![Build Status](https://github.com/PeterMakus/SeisMIC/actions/workflows/test_on_push.yml/badge.svg)](https://github.com/PeterMakus/SeisMIC/actions/workflows/test_on_push.yml?branch=main) [![Documentation Status](https://github.com/PeterMakus/SeisMIC/actions/workflows/deploy_gh_pages.yml/badge.svg)](https://github.com/PeterMakus/SeisMIC/actions/workflows/deploy_gh_pages.yml) [![License: EUPL v1.2](https://img.shields.io/badge/license-EUPL--1.2-blue)](https://joinup.ec.europa.eu/collection/eupl/introduction-eupl-licence) [![codecov](https://codecov.io/gh/PeterMakus/SeisMIC/branch/main/graph/badge.svg?token=DYVHODB6LN)](https://codecov.io/gh/PeterMakus/SeisMIC) [![DOI](https://img.shields.io/badge/DOI-10.5880%2FGFZ.2.4.2022.002-blue)](https://doi.org/10.5880/GFZ.2.4.2022.002)

## Monitoring Velocity Changes using Ambient Seismic Noise
SeisMIC (**Seismological Monitoring using Interferometric Concepts**) is a python software that emerged from the miic library. **SeisMIC** provides functionality to apply some concepts of seismic interferometry to different data of elastic waves. Its main use case is the monitoring of temporal changes in a mediums Green's Function (i.e., monitoring of temporal velocity changes).

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
+ INverting dv/v onto a spatial grid (currently only in the ``dev`` branch)

**SeisMIC** handles correlations and data in an [ObsPy](https://github.com/obspy/obspy)-like manner.

## Installation of this package

### Installation from PyPi (pip install)
**SeisMIC** is  now deployed on PyPi and can simply be installed using:

```bash
# We recommend installing mpi4py from the conda-forge channel instead of PyPi
conda install -c conda-forge mpi4py

pip install seismic
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
```

## Getting started
Access SeisMIC's documentation [here](https://petermakus.github.io/SeisMIC/index.html).

SeisMIC comes with a few tutorials (Jupyter notebooks). You can find those in the `examples/` directory.

## Acknowledging the Use of SeisMIC in your Work
If you should use SeisMIC to create published scientific content please cite [Makus, Peter; Sens-Schönfelder, Christoph (2022): Seismological Monitoring using Interferometric Concepts (SeisMIC). V. 0.1.27. GFZ Data Services. doi: 10.5880/GFZ.2.4.2022.002](https://doi.org/10.5880/GFZ.2.4.2022.002).

## Reporting Bugs / Contact the developers
This version is an early release. If you encounter any issues or unexpected behaviour, please [open an issue](https://github.com/PeterMakus/SeisMIC/issues/new) here on GitHub or [contact the developers](mailto:makus@gfz-potsdam.de).

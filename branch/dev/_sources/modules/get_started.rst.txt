Getting Started with SeisMIC
=============================

Download and Installation
-------------------------

Dependencies
++++++++++++

MPI
###

If you plan to compute correlations using MPI (e.g., because you want to use several cores), you will have to install an MPI.
Most likely, your system will come with a preinstalled MPI. Else, we recommend `Open MPI <https://www.open-mpi.org/>`_
(head there for installation instructions).

Python Dependencies
###################

**SeisMIC** has been tested on Python3.8, but should in principle also be compatible with Python3.7 and later.
If you follow the instructions below, it should not be necessary to install dependencies manually.
**SeisMIC** depends on the following modules:

- geographiclib
- h5py
- matplotlib
- mpi4py
- numpy
- obspy
- pip
- prov
- pytest
- pyyaml
- scipy
- flake8
- sphinx
- sphinx-rtd-theme
- sphinxcontrib-mermaid
- tqdm

Via PyPi
++++++++

***Not yet possible*** 

Via GitHub
++++++++++

You can download the latest version of SeisMIC from `GitHub <https://github.com/PeterMakus/SeisMIC>`_.

After downloading just run the following commands **in the repository's folder**:

.. code-blocK:: bash

    # Change directory to the same directory that this repo is in (i.e., same directory as setup.py)
    cd $PathToThisRepo$

    # Create the conda environment and install dependencies
    conda env create -f environment.yml

    # Activate the conda environment
    conda activate seismic

    # Install your package
    pip install -e .

.. note::

    While it is certainly recommendable to use a conda environment, you can also just install the package on your repository's python
    or in the currently active environment by executing the last command.

Tutorial
--------

Along with the source code **SeisMIC** is distributed with a Jupyter notebook that provides you with an easy example on how
to use the code in `examples/tutorial.ipynb`. Else, we recommend going throught this documentation.
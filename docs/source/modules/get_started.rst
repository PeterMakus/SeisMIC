Getting Started with SeisMIC
=============================

Download and Installation
-------------------------

Dependencies
++++++++++++

MPI
###

Multi-core operations in SeisMIC are supported using MPI. Therefore, you will have to install an MPI.
Most likely, your system will come with a preinstalled MPI. Else, we recommend `Open MPI <https://www.open-mpi.org/>`_
(head there for installation instructions).

Python Dependencies
###################

**SeisMIC** has been tested on Python 3.10 and 3.11 and we recommend using either of these version.
However, it should also be compatible with Python3.7 and later.
If you follow the instructions below, it should not be necessary to install dependencies manually.
**SeisMIC** depends on the following modules:

- geographiclib
- h5py
- matplotlib
- mpi4py (see note below)
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

.. note::

    We recommend using the precompiled version of mpi4py from conda-forge. This might differ on your system if you
    should experience problems with mpi4py, try a different precompiled version (for instance from PyPi).

.. note::

    Please make sure to install the same mpi version into your conda environment as you have installed on your system.
    For example, if you have installed OpenMPI on your system, you should install OpenMPI in your conda environment as well.

Via PyPi
++++++++

In general, we recommend using virtual environments (like conda or mamba).
You can install SeisMIC from PyPi simply by executing the following commands:

.. code-block:: bash

    # Install mpi4py, we recommend using the precompiled version from conda-forge
    conda install -c conda-forge mpi4py

    # Install SeisMIC
    pip install seismic

.. note::

    Developers should install SeisMIC from source code using GitHub's ``dev`` branch.
    You can also use this version if you want to explore the latest goodies.

Via GitHub
++++++++++

You can download the latest version of SeisMIC from `GitHub <https://github.com/PeterMakus/SeisMIC>`_.

After downloading just run the following commands **in the repository's folder**:

.. code-blocK:: bash

    # Download via wget or web-browser
    wget https://github.com/PeterMakus/SeisMIC/archive/refs/heads/main.zip

    # For developers download the dev branch
    wget https://github.com/PeterMakus/SeisMIC/archive/refs/heads/dev.zip

    # unzip the package
    unzip main.zip  # or dev.zip

    # Change directory to the same directory that this repo is in (i.e., same directory as setup.py)
    cd $PathToThisRepo$

    # Create the conda environment and install dependencies
    conda env create -f environment.yml

    # Activate the conda environment
    conda activate seismic

    # Install your package
    pip install -e .

    # optional: run the tests to see if your installation was successful
    pytest tests

.. note::

    While it is certainly recommendable to use a conda environment, you can also just install the package on your repository's python
    or in the currently active environment by executing the last command.

Tutorial
--------

Along with the source code **SeisMIC** is distributed with two Jupyter notebooks that provide you with an easy example on how
to use the code in `examples` on the `GitHub page <https://github.com/PeterMakus/SeisMIC>`_.
These tutorials will encompass more topics and functionalities than the documentation. If you simply want to see
a web version of those tutorials, you can find it at `tutorials <./tutorials>`_.

Aside from the tutorials, we recommend you continue reading this documentation.
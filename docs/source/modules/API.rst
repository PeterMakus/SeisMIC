.. _api-label:

``SeisMIIC`` API
================

.. toctree::
    :maxdepth: 3


miic3.trace_data
----------------

miic3.trace_data.waveform
+++++++++++++++++++++++++

Waveform download and handling of raw data

.. automodule:: miic3.trace_data.waveform
    :members:
    :show-inheritance:

miic3.correlate
---------------

miic3.correlate.correlate
+++++++++++++++++++++++++
Module to compute correlations 

.. automodule:: miic3.correlate.correlate
    :members:
    :show-inheritance:

miic3.correlate.stream
++++++++++++++++++++++
Module to handle and access correlations as pythonic objects

.. automodule:: miic3.correlate.stream
    :members:
    :show-inheritance:

miic3.correlate.stats
+++++++++++++++++++++
Managing headers

.. automodule:: miic3.correlate.stats
    :members:
    :show-inheritance:

miic3.correlate.preprocessing_stream
++++++++++++++++++++++++++++++++++++
Preprocessing functions that are executed on Obspy Streams

.. automodule:: miic3.correlate.preprocessing_stream
    :members:
    :show-inheritance:

miic3.correlate.preprocessing_td
++++++++++++++++++++++++++++++++
Preprocessing functions that are executed in Time Domain

.. automodule:: miic3.correlate.preprocessing_td
    :members:
    :show-inheritance:

miic3.correlate.preprocessing_fd
++++++++++++++++++++++++++++++++
Preprocessing functions that are executed in Frequency Domain

.. automodule:: miic3.correlate.preprocessing_fd
    :members:
    :show-inheritance:

miic3.db
--------

miic3.db.corr_hdf5
++++++++++++++++++
Save your Correlations in h5 format.

.. automodule:: miic3.db.corr_hdf5
    :members:
    :show-inheritance:

miic3.monitor
-------------

miic3.monitor.monitor
+++++++++++++++++++++
Compute seismic velocity changes

.. automodule:: miic3.monitor.monitor
    :members:
    :show-inheritance:

miic3.monitor.dv
++++++++++++++++
Module to handle velocity change objects. 

.. automodule:: miic3.monitor.dv
    :members:
    :show-inheritance:

miic3.monitor.post_corr_process
+++++++++++++++++++++++++++++++
Holds the postprocessing functions for CorrBulks.

.. automodule:: miic3.monitor.post_corr_process
    :members:
    :show-inheritance:

miic3.monitor.stretch_mod
+++++++++++++++++++++++++
Holds functions for the velocity estimation via stretching.

.. automodule:: miic3.monitor.stretch_mod
    :members:
    :show-inheritance:

miic3.plot
----------

miic3.plot.plot_correlation
+++++++++++++++++++++++++++
Plot Green's function estimations.

.. automodule:: miic3.plot.plot_correlation
    :members:
    :show-inheritance:

miic3.plot.plot_dv
++++++++++++++++++
Plot velocity change time series.

.. automodule:: miic3.plot.plot_dv
    :members:
    :show-inheritance:

miic3.plot.plt_spectra
+++++++++++++++++++++++
Plot noise spectra using Welch's method.

.. automodule:: miic3.plot.plt_spectra
    :members:
    :show-inheritance:

miic3.plot.plot_utils
+++++++++++++++++++++
Useful little plotting functions

.. automodule:: miic3.plot.plot_spectra
    :members:
    :show-inheritance:

miic3.utils
-----------

Collection of tools

miic3.utils.io
++++++++++++++
Load correlations produced by MIIC

.. automodule:: miic3.utils.io
    :members:
    :show-inheritance:

miic3.utils.fetch_func_from_str
+++++++++++++++++++++++++++++++
Import and Return a function

.. automodule:: miic3.utils.fetch_func_from_str
    :members:
    :show-inheritance:

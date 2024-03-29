.. _api-label:

API
===

seismic.trace_data
------------------

seismic.trace_data.waveform
+++++++++++++++++++++++++++

Waveform download and handling of raw data

.. automodule:: seismic.trace_data.waveform
    :members:
    :show-inheritance:

seismic.correlate
-----------------

seismic.correlate.correlate
+++++++++++++++++++++++++++
Module to compute correlations 

.. automodule:: seismic.correlate.correlate
    :members:
    :show-inheritance:

seismic.correlate.stream
++++++++++++++++++++++++
Module to handle and access correlations as pythonic objects

.. automodule:: seismic.correlate.stream
    :members:
    :show-inheritance:

seismic.correlate.stats
+++++++++++++++++++++++
Managing headers

.. automodule:: seismic.correlate.stats
    :members:
    :show-inheritance:

seismic.correlate.preprocessing_stream
++++++++++++++++++++++++++++++++++++++
Preprocessing functions that are executed on Obspy Streams

.. automodule:: seismic.correlate.preprocessing_stream
    :members:
    :show-inheritance:

seismic.correlate.preprocessing_td
++++++++++++++++++++++++++++++++++
Preprocessing functions that are executed in Time Domain

.. automodule:: seismic.correlate.preprocessing_td
    :members:
    :show-inheritance:

seismic.correlate.preprocessing_fd
++++++++++++++++++++++++++++++++++
Preprocessing functions that are executed in Frequency Domain

.. automodule:: seismic.correlate.preprocessing_fd
    :members:
    :show-inheritance:

seismic.db
----------

seismic.db.corr_hdf5
++++++++++++++++++++
Save your Correlations in h5 format.

.. automodule:: seismic.db.corr_hdf5
    :members:
    :show-inheritance:

seismic.monitor
---------------

seismic.monitor.monitor
+++++++++++++++++++++++
Compute seismic velocity changes

.. automodule:: seismic.monitor.monitor
    :members:
    :show-inheritance:

seismic.monitor.dv
++++++++++++++++++
Module to handle velocity change objects. 

.. automodule:: seismic.monitor.dv
    :members:
    :show-inheritance:

seismic.monitor.spatial
+++++++++++++++++++++++
Module to invert velocity change time series from a set of correlations to a spatial velocity field. Base on the algorithm proposed in
[Obermann et al. (2013)](https://doi.org/10.1002/2013JB010399).

.. automodule:: seismic.monitor.spatial
    :members:
    :show-inheritance:

seismic.monitor.wfc
+++++++++++++++++++
Module to analyse the waveform coherence.

.. automodule:: seismic.monitor.wfc
    :members:
    :show-inheritance:

seismic.monitor.post_corr_process
++++++++++++++++++++++++++++++++++
Holds the postprocessing functions for CorrBulks.

.. automodule:: seismic.monitor.post_corr_process
    :members:
    :show-inheritance:

seismic.monitor.stretch_mod
+++++++++++++++++++++++++++
Holds functions for the velocity estimation via stretching.

.. automodule:: seismic.monitor.stretch_mod
    :members:
    :show-inheritance:

seismic.monitor.trim
++++++++++++++++++++
Trim correlation matrices.

.. automodule:: seismic.monitor.trim
    :members:
    :show-inheritance:

seismic.plot
------------

seismic.plot.plot_correlation
+++++++++++++++++++++++++++++
Plot Green's function estimations.

.. automodule:: seismic.plot.plot_correlation
    :members:
    :show-inheritance:

seismic.plot.plot_dv
++++++++++++++++++++
Plot velocity change time series.

.. automodule:: seismic.plot.plot_dv
    :members:
    :show-inheritance:

seismic.plot.plot_multidv
+++++++++++++++++++++++++
Plot velocity change time series.

.. automodule:: seismic.plot.plot_multidv
    :members:
    :show-inheritance:

seismic.plot.plot_utils
+++++++++++++++++++++++
Useful little plotting functions

.. automodule:: seismic.plot.plot_utils
    :members:
    :show-inheritance:

seismic.plot.plot_spectrum
++++++++++++++++++++++++++
Plot spectrograms.

.. automodule:: seismic.plot.plot_spectrum
    :members:
    :show-inheritance:

seismic.plot.plot_wfc
+++++++++++++++++++++
Plot the waveform coherence

.. automodule:: seismic.plot.plot_wfc
    :members:
    :show-inheritance:

seismic.utils
-------------

Collection of tools

seismic.utils.io
++++++++++++++++
Load correlations produced by MIIC

.. automodule:: seismic.utils.io
    :members:
    :show-inheritance:

seismic.utils.fetch_func_from_str
+++++++++++++++++++++++++++++++++
Import and Return a function

.. automodule:: seismic.utils.fetch_func_from_str
    :members:
    :show-inheritance:

seismic.utils.miic_utils
++++++++++++++++++++++++
Set of smaller functions.

.. automodule:: seismic.utils.miic_utils
    :members:
    :show-inheritance:

seismic.utils.raw_analysis
++++++++++++++++++++++++++
Analysing raw waveform data.

.. automodule:: seismic.utils.raw_analysis
    :members:
    :show-inheritance:

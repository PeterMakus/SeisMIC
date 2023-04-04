
.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data] --> correlate(Correlation)
        correlate -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        monitor[Measure dv]:::active -->|save| dv{{DV}}
        click waveform "../trace_data.html" "trace_data"
        click correlate "../correlate.html" "correlate"
        click monitor "../monitor.html" "monitor"
        click corrdb "../corrdb.html" "CorrDB"
        click dv "../monitor/dv.html" "DV"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;

Compute Velocity Changes
------------------------

**SeisMIC** uses the stretching technique (Sens-Schönfelder & Wegler, 2006) to estimate a spatially homogeneous velocity change between the
stations (for Cross-Correlations) or in station vicinity (for inter-component or autocorrelations).
This technique is grounded upon the assumption that a homogeneous change in velocity will result in a homogeneous stretching of the
Green's function.

.. note::
    We are currently in the stage of implementing some other algorithms to estiamte dv/v.
    If you don't want to wait, it should be easy for you to implement
    a different algorithm yourself.

Arguments in ``params.yaml``
++++++++++++++++++++++++++++

To set the arguments for our velocity change computation, we will once again use the yaml file as shown before. Closer to the bottom
of this file, you will find a section with the key ``dv``:

.. code-block:: yaml
    :linenos:

    #### parameters for the estimation of time differences
    dv:
        # subfolder for storage of time difference results
        subdir : 'vel_change'

        # Plotting
        plot_vel_change : True

        ### Definition of calender time windows for the time difference measurements
        start_date : '2015-05-01 00:00:00.0'   # %Y-%m-%dT%H:%M:%S.%fZ'
        end_date : '2016-01-01 00:00:00.0'
        win_len : 86400                         # length of window in which EGFs are stacked
        date_inc : 86400                        # increment of measurements

        ### Frequencies
        freq_min : 4
        freq_max : 8

        ### Definition of lapse time window
        tw_start : 20     # lapse time of first sample [s]
        tw_len : 60       # length of window [s] Can be None if the whole (rest) of the coda should be used
        sides : 'both'   # options are left (for acausal), right (causal), both, or single (for active source experiments where the first sample is the trigger time)
        compute_tt : True  # Computes the travel time and adds it to tw_start (tt is 0 if not xstations). If true a rayleigh wave velocity has to be provided
        rayleigh_wave_velocity : 1  # rayleigh wave velocity in km/s, will be ignored if compute_tt=False
        

        ### Range to try stretching
        stretch_range : 0.03
        stretch_steps : 1001
    
        #### Reference trace extraction
        #  win_inc : Length in days of each reference time window to be used for trace extraction
        # If == 0, only one trace will be extracted
        # If > 0, mutliple reference traces will be used for the dv calculation
        # Can also be a list if the length of the windows should vary
        # See seismic.correlate.stream.CorrBulk.extract_multi_trace for details on arguments
        dt_ref : {'win_inc' : 0, 'method': 'mean', 'percentile': 50}

        # preprocessing on the correlation bulk or corr_stream before stretch estimation
        preprocessing: [
                        #{'function': 'pop_at_utcs', 'args': {'utcs': np.array([UTCDateTime()])},
                        {'function': 'smooth', 'args': {'wsize': 48, 'wtype': 'hanning', 'axis': 1}}
        ]

        # postprocessing of dv objects before saving and plotting
        postprocessing: [
                        {'function': 'smooth_sim_mat', 'args': {'win_len': 7, exclude_corr_below: 0}}
        ]

As you can see, there are only fairly few settings that can be changed, some of which are very obvious again:

+ ``subdir``: directory to save the dv objects in
+ ``plot_vel_change``: Set this to ``True`` if you would like your velocity changes to be plotted to a file in the fig folder.
+ ``start_date``, ``end_date``: Start and end of the time series (Note that it will result in nans if there are no correlations available.
+ ``win_len``, ``date_inc``: Length of each datapoint (i.e., correlation) in seconds and distance between the subsequent datapoints. ``win_len`` has to be at least equal to the length of one correlation. If it is longer, correlations will be stacked.
+ ``freq_min``, ``freq_max``: lower and upper frequencies for the bandpass filter.
+ ``preprocessing``: List of functions that will be applied to the correlations prior to the interferometry. You can feed in your own custom functions. ``smooth`` does just simply apply a moving window along one axis. Physical window length is ``win_inc``*48 + 2*(``win_len`` - ``win_inc``).
+ ``postprocessing``: Functions that are applied to the :py:class:`~seismic.monitor.dv.DV` object. Same logic as for ``preprocessing``

The other four parameters will influence the actual stretching:
+ ``tw`` is the time window in the coda which should be stretched and compared with the lapsed correlations.
+ ``stretch_range`` is the maximum absolute stretch to be tested
+  ``stretch_steps`` the number of increments that will be tested between the minimum and maximum stretching.
+  ``sides`` decides whether seismic will compare both sides of the correlation functions (positive and negative lag-times / causal and acausal) or just one (if set to *single*)
+  The ``compute_tt`` and ``rayleigh_wave_velocity`` are only relevant for cross-correlations. If set, SeisMIC will add the time of theoretical arrival to the ``tw_start`` parameter.

Computing the Reference Trace
=============================

``dt_ref`` is the parameter governing the computation of the reference trace. We can opt for a single reference trace for the whole period (``dt_ref['win_inc']=0``) or multiple reference traces.
Check out the docstring of :py:meth:`~seismic.correlate.stream.CorrBulk.extract_multi_trace` to learn more!

Start the Computation
+++++++++++++++++++++

Again, the procedure is fairly similar to startin the correlation. Velocity stretch estimates are computed by the
:py:class:`~seismic.monitor.monitor.Monitor` object. Once again, usage with mpi is possible. Your velocity stretch estimate
script could look something like this:

.. code-block:: python
    :caption: compute_dv.py
    :linenos:

    from seismic.monitor.monitor import Monitor

    yaml_f = '/home/pm/Documents/PhD/Chaku/params.yaml'
    m = Monitor(yaml_f)
    m.compute_velocity_change_bulk()

Again, you will only want to use the method :py:meth:`seismic.monitor.monitor.Monitor.compute_velocity_change_bulk`.
You can start the script using mpi:

.. code-block:: bash

    mpirun -n $number_of_cores$ python $path_to_file$/compute_dv.py+

.. note::

    :py:meth:`~seismic.monitor.monitor.Monitor.compute_velocity_change_bulk` is the multi-core equivalent of
    :py:meth:`~seismic.monitor.monitor.Monitor.compute_velocity_change`. The latter takes a particular `hdf5` file
    as input, whereas the former will estimate the velocity changes of all `hdf5` files that are defined by
    `co['subdir']` in the `params.yaml` file and fit the filters set in `net`. This also means that the process
    won't speed up any more if *number_of_cores* exceeds the number of hdf5 correlation files that you have computed previously.

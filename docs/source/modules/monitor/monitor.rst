
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

**SeisMIIC** uses the stretching technique (Sens-Schönfelder & Wegler, 2006) to estimate a spatially homogeneous velocity change between the
stations (for Cross-Correlations) or in station vicinity (for inter-component or autocorrelations).
This technique is grounded upon the assumption that a homogeneous change in velocity will result in a homogeneous stretching of the
Green's function.

Arguments in ``params.yaml``
++++++++++++++++++++++++++++

To set the arguments for our velocity change computation, we will once again use the yaml file as shown before. Right at the bottom
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
        freq_min : 0.1
        freq_max : 0.5

        ### Definition of lapse time window
        tw_start : 20     # lapse time of first sample [s]
        tw_len : 60       # length of window [s]
        
        ### Range to try stretching
        stretch_range : 0.03
        stretch_steps : 1001

As you can see, there are only fairly few settings that can be changed, some of which are very obvious again:

+ ``subdir``: directory to save the dv objects in
+ ``plot_vel_change``: Set this to ``True`` if you would like your velocity changes to be plotted to a file in the fig folder.
+ ``start_date``, ``end_date``: Start and end of the time series (Note that it will result in nans if there are no correlations available.
+ ``win_len``, ``date_inc``: Length of each datapoint in seconds and distance between the subsequent datapoints.
+ ``freq_min``, ``freq_max``: lower and upper frequencies for the bandpass filter.

The other four parameters will actually influence the actual stretching. ``tw`` is the time window which should be stretched and
compared with the lapsed correlations. ``stretch_range`` is the maximum absolute stretch to be tested and ``stretch_steps`` the number
of increments that will be tested between the minimum and maximum stretching.

Start the Computation
+++++++++++++++++++++

Again, the procedure is fairly similar to startin the correlation. Velocity stretch estimates are computed by the
:class:`~miic3.monitor.monitor.Monitor` object. Once again, usage with mpi is possible. Your velocity stretch estimate
script could look something like this:

.. code-block:: python
    :caption: compute_dv.py
    :linenos:

    from miic3.monitor.monitor import Monitor

    yaml_f = '/home/pm/Documents/PhD/Chaku/params.yaml'
    m = Monitor(yaml_f)
    m.compute_velocity_change_bulk()

Again, you will only want to use the method :meth:`miic3.monitor.monitor.Monitor.compute_velocity_change_bulk`.
You can start the script using mpi:

.. code-block:: bash

    mpirun -n $number_of_cores$ python $path_to_file$/compute_dv.py

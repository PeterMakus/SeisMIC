.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data] --> correlate(Correlation)
        correlate:::active -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        click waveform "../trace_data.html" "trace_data"
        click correlate "../correlate.html" "correlate"
        click monitor "../monitor.html" "monitor"
        click corrdb "../corrdb.html" "CorrDB"
        click dv "../monitor/dv.html" "DV"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;

Computing your first Noise Correlations
---------------------------------------

After having downloaded your data as shown `here <../trace_data/waveform.html#download-data>`_ (or retrieved seismic data any other way),
we are ready to compute our first noise correlations!
In **SeisMIC**, parameters for the *preprocessing*, the *correlation*, and, subsequently,
measuring the *change in seismic velocity* are provided as a *.yaml* file or as a python *dictionary*.
An example for such a *.yaml* file is shown below (and provided in the repository as ``params_example.yaml``).

Setting the parameters
++++++++++++++++++++++

.. code-block:: yaml
    :linenos:

    #### Project wide parameters
    # lowest level project directory
    proj_dir : '/path/to/project/root/'
    # Save the component combinations in separate hdf5 files
    # This is faster for multi-core if True
    # Set to False for compatibility with SeisMIC version < 0.5.0
    save_comps_separately: True
    # directory for logging information
    log_subdir : 'log'
    # levels:
    # 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
    log_level: 'WARNING'
    # folder for figures
    fig_subdir : 'figures'


    #### parameters that are network specific
    net:
        # list of stations used in the project
        # type: list of strings or string, wildcards allowed
        network : 'D0'
        station : ['BZG', 'ESO', 'KBG', 'KIR']

    #### parameters for correlation (emperical Green's function creation)
    co:
        # subdirectory of 'proj_dir' to store correlation
        # type: string
        subdir : 'corr'
        # times sequences to read for cliping or muting on stream basis
        # These should be long enough for the reference (e.g. the standard
        # deviation) to be rather independent of the parts to remove
        # type: string
        read_start : '2015-08-1 00:00:01.0'
        read_end : '2015-08-06 00:00:00.0'
        # type: float [seconds]
        # The longer the faster, but higher RAM usage.
        # Note that this is also the length of the correlation batches
        # that will be written (i.e., length that will be 
        # kept in memory before writing to disk)
        # If you are unsure, keep defaults
        read_len : 86398
        read_inc : 86400

        # New sampling rate in Hz. Note that it will try to decimate
        # if possible (i.e., there is an integer factor from the
        # native sampling_rate). Pay attention to Nyquist frequency for
        # later steps
        sampling_rate : 25
        # Remove the instrument response, will take substantially more time
        remove_response : True

        # Method to combine different traces
        # Options are: 'betweenStations', 'betweenComponents', 'autoComponents', 'allSimpleCombinations', or 'allCombinations'
        combination_method : 'betweenStations'
    
        # If you want only specific combinations to be computed enter them here
        # In the form [Net0-Net0.Stat0-Stat1]
        # This option will only be consider if combination_method == 'betweenStations'
        # Comment or set == None if not in use
        xcombinations : None

        # preprocessing of the original length time series
        # these function work on an obspy.Stream object given as first argument
        # and return an obspy.Stream object.
        # available preimplemented functions are in seismic.correlate.preprocessing_stream
        preProcessing : [
                        {'function':'seismic.correlate.preprocessing_stream.detrend_st',
                        'args':{'type':'linear'}},
                        {'function':'seismic.correlate.preprocessing_stream.cos_taper_st',
                        'args':{'taper_len': 100, # seconds
                                'lossless': True}}, # lossless tapering stitches additional data to trace, tapers, and removes the tapered ends after preprocessing
                        # This is intended as a "technical" bandpass filter to remove unphysical signals, i.e., frequencies you would not expect in the data
                        {'function':'seismic.correlate.preprocessing_stream.stream_filter',
                        'args':{'ftype':'bandpass',
                                'filter_option':{'freqmin':0.01, #0.01
                                                'freqmax':12}}}
                        # Mute data, when the amplitude is above a certain threshold
                        #{'function':'seismic.correlate.preprocessing_stream.stream_mute',
                        # 'args':{'taper_len':100,
                        #         'mute_method':'std_factor',
                        #         'mute_value':3}}
                        ]
        # subdivision of the read sequences for correlation
        # if this is set the stream processing will happen on the hourly subdivision. This has the
        # advantage that data that already exists will not need to be preprocessed again
        # On the other hand, computing a whole new database might be slower
        # Recommended to be set to True if:
        # a) You update your database and a lot of the data is already available (up to a magnitude faster)
        # b) corr_len is close to read_len
        # Is automatically set to False if no existing correlations are found
        preprocess_subdiv: True
        # type: presence of this key
        subdivision:
            # type: float [seconds]
            corr_inc : 3600
            corr_len : 3600
            # recombine these subdivisions
            # unused at the time
            # type: boolean
            recombine_subdivision : True
            # delete
            # type: booblean
            delete_subdivision : False

        # parameters for correlation preprocessing
        # Standard functions reside in seismic.correlate.preprocessing_td and preprocessing_fd, respectively
        corr_args : {'TDpreProcessing':[
                                        # detrend not recommended. Use preProcessing detrend_st instead (faster)
                                        # {'function':'seismic.correlate.preprocessing_td.detrend',
                                        # 'args':{'type':'linear'}},
                                    {'function':'seismic.correlate.preprocessing_td.TDfilter',
                                    'args':{'type':'bandpass','freqmin':2,'freqmax':8}},
                                        #{'function':'seismic.correlate.preprocessing_td.mute',
                                        # 'args':{'taper_len':100.,
                                            # 'threshold':1000, absolute threshold
                                        #         'std_factor':3,
                                        #         'filter':{'type':'bandpass','freqmin':2,'freqmax':4},
                                        #         'extend_gaps':True}},
                                    {'function':'seismic.correlate.preprocessing_td.clip',
                                        'args':{'std_factor':2.5}},
                                    {'function':'seismic.correlate.preprocessing_td.signBitNormalization',
                                        'args': {}}
                                    ],
                    # Standard functions reside in seismic.correlate.preprocessing_fd
                    'FDpreProcessing':[
                                        {'function':'seismic.correlate.preprocessing_fd.spectralWhitening',
                                        'args':{'joint_norm':False}},
                                        {'function':'seismic.correlate.preprocessing_fd.FDfilter',
                                        'args':{'flimit':[0.01,0.02,9,10]}}
                                        #  {'function':seismic.correlate.preprocessing_fd.FDsignBitNormalization,
                                        # 'args':{}}
                                        ],
                    'lengthToSave':100,
                    'center_correlation':True,      # make sure zero correlation time is in the center
                    'normalize_correlation':True,
                    'combinations':[]
                    }


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

        ### Definition of lapse time window, i.e. time window in the coda that is used for the dv/v estimate
        tw_start : 20     # lapse time of first sample [s]
        tw_len : 60       # length of window [s] Can be None if the whole (rest) of the coda should be used
        sides : 'both'   # options are left (for acausal), right (causal), both, or single (for active source experiments where the first sample is the trigger time)
        compute_tt : True  # Computes the travel time and adds it to tw_start (tt is 0 if not xstations). If true a rayleigh wave velocity has to be provided
        rayleigh_wave_velocity : 1  # rayleigh wave velocity in km/s, will be ignored if compute_tt=False
        

        ### Range to try stretching
        # maximum stretch factor
        stretch_range : 0.03
        # number of stretching increments
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
                        #{'function': 'pop_at_utcs', 'args': {'utcs': np.array([UTCDateTime()])},  # Used to remove correlations from certain times
                        {'function': 'smooth', 'args': {'wsize': 48, 'wtype': 'hanning', 'axis': 1}}
        ]

        # postprocessing of dv objects before saving and plotting
        postprocessing: [
                        {'function': 'smooth_sim_mat', 'args': {'win_len': 7, exclude_corr_below: 0}}
        ]

    #### parameters to compute the waveform coherence
    wfc:
    # subfolder for storage of time difference results
        subdir : 'wfc'

        ### Definition of calender time windows for the time difference measurements
        start_date : '2015-05-01 00:00:00.0'   # %Y-%m-%dT%H:%M:%S.%fZ'
        end_date : '2016-01-01 00:00:00.0'
        win_len : 86400                         # length of window in which EGFs are stacked
        date_inc : 86400                        # increment of measurements

        ### Frequencies
        # can be lists of same length
        freq_min : [0.0625, 0.09, 0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5]
        freq_max : [0.125, 0.18, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10]

        ### Definition of lapse time window
        # can be lists of same length or tw_start: List and tw_len: single value (will be applied to all)
        tw_start : [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10, 11.25, 12.5, 13.75, 15, 16.25, 17.5, 17.75, 20]     # lapse time of first sample [s]
        tw_len : 5       # length of window [s]
    
        #### Reference trace extraction
        #  win_inc : Length in days of each reference time window to be used for trace extraction
        # If == 0, only one trace will be extracted
        # If > 0, mutliple reference traces will be used for the dv calculation
        # Can also be a list if the length of the windows should vary
        # See seismic.correlate.stream.CorrBulk.extract_multi_trace for details on arguments
        dt_ref : {'win_inc' : 0, 'method': 'mean', 'percentile': 50}

        # preprocessing on the correlation bulk before stretch estimation
        preprocessing: [
                        {'function': 'smooth', 'args': {'wsize': 4, 'wtype': 'hanning', 'axis': 1}}
        ]

        ### SAVING
        # save components separately or only their average?
        save_comps: False




This might look a little intimidating at first glancec, but it is actually quite straight-forward.
To achieve a better understanding of what each of the parameters do, let's have a close look at them individually.

Project Wide Parameters
=======================

.. code-block:: yaml
    :linenos:

    #### Project wide parameters
    # lowest level project directory
    proj_dir : '/path/to/project/root/'
    # Save the component combinations in separate hdf5 files
    # This is faster for multi-core if True
    # Set to False for compatibility with SeisMIC version < 0.5.0
    save_comps_separately: True
    # directory for logging information
    log_subdir : 'log'
    # levels:
    # 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'
    log_level: 'WARNING'
    # folder for figures
    fig_subdir : 'figures'

Those are parameters that govern the logging and the file-structure. ``proj_dir`` is the root directory, we have chosen when initialising our 
:py:class:`~seismic.trace_data.waveform.Store_Client` as shown `here <../trace_data/waveform.html#download-data>`_ .
``fig_dir`` and ``log_dir`` are just subdirectories for figures and logs, respectively, and the log level decides how much will actually be logged.
In most cases, *WARNING* is the appropriate choice - everything below will start spitting out a lot of information.

Network Specific Parameters
===========================

.. code-block:: yaml
    :linenos:

    #### parameters that are network specific
    net:
        # list of stations used in the project
        # type: list of strings or string, wildcards allowed
        network : 'D0'
        station : ['BZG', 'ESO', 'KBG', 'KIR']

Here, we decide which data to use (i.e., which data the correlator will look for and read in). All parameters accept wildcards and can be strings or lists.

.. note::

    If both ``network`` and ``station`` are lists, they have to have the same length.
    The corresponding logic is: `[net0, net1, net2, net3]` and `[stat0, stat1, stat2, stat3]`
    and so on.

Correlation Arguments
=====================
This is the really juicy stuff and probably the part that will have the strongest influence on your results.
Let's start by getting the most obvious parameters out of the way:

.. code-block:: yaml
    :linenos:

    #### parameters for correlation (emperical Green's function creation)
    co:
        # subdirectory of 'proj_dir' to store correlation
        # type: string
        subdir : 'corr'
        # times sequences to read for cliping or muting on stream basis
        # These should be long enough for the reference (e.g. the standard
        # deviation) to be rather independent of the parts to remove
        # type: string
        read_start : '2015-08-1 00:00:01.0'
        read_end : '2015-08-06 00:00:00.0'
        # type: float [seconds]
        # The longer the faster, but higher RAM usage.
        # Note that this is also the length of the correlation batches
        # that will be written (i.e., length that will be
        # kept in memory before writing to disk)
        read_len : 86398
        read_inc : 86400

+ ``subdir`` The directory to save the correlations in (correlations are generally saved in hdf5 format).
+ ``read_start`` and ``read_end`` are the earliest and latest dates that you want to read
+ ``read_len`` the length that will be read in. Usually, you have one *mseed* file per day. **To avoid having to read several files, you will want that to be a bit less than a day**
+ ``read_inc`` is the increment between each reading interval

.. note::
    
    Neither ``read_len`` nor ``read_inc`` are deciding about the correlation length.

.. code-block:: yaml
    :linenos:

        # New sampling rate in Hz. Note that it will try to decimate
        # if possible (i.e., there is an integer factor from the
        # native sampling_rate)
        sampling_rate: 25
        # Remove the instrument response, will take substantially more time
        remove_response: False

        # Method to combine different traces
        combination_method : 'betweenStations'
  
        # If you want only specific combinations to be computed enter them here
        # In the form [Net0-Net0.Stat0-Stat1]
        # This option will only be consider if combination_method == 'betweenStations'
        # Comment or set == None if not in use
        xcombinations : None

+ ``Sampling_rate`` is the new sampling rate you will want your data to have. **SeisMIC** will take care of anti-alias filtering and determine whether data can be decimated.
+ ``remove_response`` if you want the data to be corrected for the instrument response, set this to ``True``. Instrument response removal in obspy is unfortunately quite expensive..
+ ``combination_method`` decides which components you will want to correlate. See :py:func:`~seismic.correlate.correlate.calc_cross_combis` for allowed options.
+ ``xcombinations`` If you want to save some computational resources and only compute specific combinations, use this function. If you want to limit the maximum distance between stations to cross-correlate you can use :py:meth:`seismic.correlate.correlate.Correlator.find_interstat_dist` to compute this parameter


Preprocessing Arguments
#######################

**SeisMIC** is coded in a manner that makes it easy for the user to pass custom preprocessing functions. Custom functions can be defined in the three parameters ``preProcessing``, ``TDpreProcessing``, and ``FDpreprocessing``.
All these parameters expect a ``list`` of ``dictionaries`` as input. Each dictionary must have the keys ``function`` and ``args``. The value for function is a string describing the complete import path of the preprocessing function in the form **'package.module.sobmodule.function'**.
``args`` is simply a keyword argument dictionary that will be passed to the function.

**SeisMIC** comes with a number of preprocessing functions. If you are creating a custom preprocessing function, it is probably a good idea to have a look at these first in order to understand the required syntax.
Preprocecssing is generally done in three steps:

**Preprocessing "on per stream basis"**
All functions here take an `obspy stream <https://docs.obspy.org/master/packages/autogen/obspy.core.stream.Stream.html>`_ as input and return the processed stream.
An over view of available stream preprocessing functions can  be found in :mod:`~seismic.correlate.preprocessing_stream`.


.. code-block:: yaml
    :linenos:

        # preprocessing of the original length time series
        # these function work on an obspy.Stream object given as first argument
        # and return an obspy.Stream object.
        preProcessing : [
                        {'function':'seismic.correlate.preprocessing_stream.detrend_st',
                        'args':{'type':'linear'}},
                        {'function':'seismic.correlate.preprocessing_stream.cos_taper_st',
                        'args':{'taper_len': 100, # seconds
                                'lossless': True}}, # lossless tapering stitches additional data to trace, tapers, and removes the tapered ends after preprocessing
                        # This is intended as a "technical" bandpass filter to remove unphysical signals, i.e., frequencies you would not expect in the data
                        {'function':'seismic.correlate.preprocessing_stream.stream_filter',
                        'args':{'ftype':'bandpass',
                                'filter_option':{'freqmin':0.01, #0.01
                                                'freqmax':12}}}
                        # Mute data, when the amplitude is above a certain threshold
                        #{'function':'seismic.correlate.preprocessing_stream.stream_mute',
                        # 'args':{'taper_len':100,
                        #         'mute_method':'std_factor',
                        #         'mute_value':3}}
                        ]


**Preprocessing on arrays in time and frequency domain**
The functions to use have to be provided in ``corr_args['TDpreProcecssing']`` and ``corr_args['FDpreProcecssing']``.
A custom function would need to take a matrix as input, where each column is one waveform in time or frequency domain.
Additionally, the ``args`` dictionary and a ``params`` dictionary will be passed.

.. code-block:: yaml
    :linenos:

    # parameters for correlation preprocessing
    # Standard functions reside in seismic.correlate.preprocessing_td
    corr_args : {'TDpreProcessing':[
                                    # detrend not recommended. Use preProcessing detrend_st instead (faster)
                                    # {'function':'seismic.correlate.preprocessing_td.detrend',
                                    # 'args':{'type':'linear'}},
                                   {'function':'seismic.correlate.preprocessing_td.TDfilter',
                                   'args':{'type':'bandpass','freqmin':2,'freqmax':8}},
                                    #{'function':'seismic.correlate.preprocessing_td.mute',
                                    # 'args':{'taper_len':100.,
                                           # 'threshold':1000, absolute threshold
                                    #         'std_factor':3,
                                    #         'filter':{'type':'bandpass','freqmin':2,'freqmax':4},
                                    #         'extend_gaps':True}},
                                   {'function':'seismic.correlate.preprocessing_td.clip',
                                    'args':{'std_factor':2.5}},
                                   {'function':'seismic.correlate.preprocessing_td.signBitNormalization',
                                    'args': {}}
                                   ],
                  # Standard functions reside in seismic.correlate.preprocessing_fd
                 'FDpreProcessing':[
                                    {'function':'seismic.correlate.preprocessing_fd.spectralWhitening',
                                     'args':{'joint_norm':False}},
                                    {'function':'seismic.correlate.preprocessing_fd.FDfilter',
                                     'args':{'flimit':[0.01,0.02,9,10]}}
                                    #  {'function':seismic.correlate.preprocessing_fd.FDsignBitNormalization,
                                    # 'args':{}}
                                    ],
                ...
                }

Arguments for the actual correlation
####################################

``Subdivision`` is the parameter that decides about the length and increment of the noise recordings to be preprocessed and correlated.
If ``recombine_subdivision=True``, the correlations will be stacked to ``read_len``.

+ ``LengthToSave`` is the length of each correlation function in seconds
+ ``Center_Correlation`` If True, zero-lag will always be in the middle of the function.
+ ``normalize_correlation``: Normalise the correlation by the absolute maximum?


.. code-block:: yaml
    :linenos:

        # subdivision of the read sequences for correlation
        # type: presence of this key
        subdivision:
            # type: float [seconds]
            corr_inc : 3600
            corr_len : 3600
            # recombine these subdivisions
            # unused at the time
            # type: boolean
            recombine_subdivision : True
            # delete
            # type: booblean
            delete_subdivision : False

        # parameters for correlation preprocessing
        # Standard functions reside in seismic.correlate.preprocessing_td
        corr_args : {'lengthToSave':100,
                    'center_correlation':True,      # make sure zero correlation time is in the center
                    'normalize_correlation':True,
                    'combinations':[]
                    }

The rest of the yaml file will be discussed at a later points. Now, let's actually start the computation!

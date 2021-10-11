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

Get Started (Compute your first Noise Correlations)
---------------------------------------------------

After having downloaded your data as shown `here <../trace_data/waveform.html#download-data>`_ (or retrieved seismic data any other way),
we are ready to compute our first noise correlation!
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
        # list of channels
        # type: list of strings
        # actually not in use (yet)
        # channels : ['HHZ','HHN','HHE']

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

        # New sampling rate in Hz. Note that it will try to decimate
        # if possible (i.e., there is an integer factor from the
        # native sampling_rate)
        sampling_rate: 25
        # Remove the instrument response, will take substantially more time
        remove_response: False

        # Method to combine different traces
        combination_method : 'betweenStations'

        # preprocessing of the original length time series
        # these function work on an obspy.Stream object given as first argument
        # and return an obspy.Stream object.
        preProcessing : [
                        {'function':'seismic.correlate.preprocessing_stream.detrend_st',
                        'args':{'type':'simple'}},
                        {'function':'seismic.correlate.preprocessing_stream.cos_taper_st',
                        'args':{'taper_len': 10, # seconds
                                'taper_at_masked': True}},
                        {'function':'seismic.correlate.preprocessing_stream.stream_filter',
                        'args':{'ftype':'bandpass',
                                'filter_option':{'freqmin':0.01, #0.01
                                                'freqmax':12.5}}}
                        ]
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
        # Taper the time windows with a 5% Hann taper on each side. If this is True,
        # the time windows will just simply be prolonged by the length of the taper,
        # so that no data is lost
        # This taper is probably obsolete
        taper: False

        # parameters for correlation preprocessing
        # Standard functions reside in seismic.correlate.preprocessing_td
        corr_args : {'TDpreProcessing':[
                                        {'function':'seismic.correlate.preprocessing_td.detrend',
                                        'args':{'type':'constant'}},
                                      {'function':'seismic.correlate.preprocessing_td.TDfilter',
                                      'args':{'type':'bandpass','freqmin':2,'freqmax':8}},
                                    # {'function':'seismic.correlate.preprocessing_td.taper',
                                    #  'args': {'type':'cosine_taper','p':0.02}},
                                        # {'function':'seismic.correlate.preprocessing_td.mute',
                                        # 'args':{'taper_len':100.,
                                        #        'threshold':1000, absolute threshold
                                        #         'std_factor':3,
                                        #         'filter':{'type':'bandpass','freqmin':2,'freqmax':4},
                                        #         'extend_gaps':True}},
                                    {'function':'seismic.correlate.preprocessing_td.clip',
                                        'args':{'std_factor':3}},
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

        # Component rotation (only possible if 'direct_output' is not in 'corr_args')
        # type: string ['NO', 'ZNE->ZRT', 'NE->RT']
        # Not used yet
        rotation : 'NO'


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
        stretch_steps : 1000

This might look a little intimidating at first glancec, but is actually quite straight-forward.
To achieve a better understanding of what each of the parameters do, let's have a close look at them individually.

Project Wide Parameters
=======================

.. code-block:: yaml
    :linenos:

    #### Project wide parameters
    # lowest level project directory
    proj_dir : '/path/to/project/root/'
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
        # list of channels
        # type: list of strings
        # actually not in use (yet)
        # channels : ['HHZ','HHN','HHE']

Here, we decide which data to use (i.e., which data the correlator will look for and read in). All parameters accept wildcards and can be strings or lists.

.. note::

    If both ``network`` and ``station`` are lists, they have to have the same length.

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

+ ``Sampling_rate`` is the new sampling rate you will want your data to have. **SeisMIC** will take care of anti-alias filtering and determine whether data can be decimated.
+ ``remove_response`` if you want the data to be corrected for the instrument response, set this to ``True``.
+ ``combination_method`` decides which components you will want to correlate. See :py:func:`~seismic.correlate.correlate.calc_cross_combis` for allowed options.


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
                        'args':{'type':'simple'}},
                        {'function':'seismic.correlate.preprocessing_stream.cos_taper_st',
                        'args':{'taper_len': 10, # seconds
                                'taper_at_masked': True}},
                        {'function':'seismic.correlate.preprocessing_stream.stream_filter',
                        'args':{'ftype':'bandpass',
                                'filter_option':{'freqmin':0.01,
                                                'freqmax':12.5}}}
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
                                    {'function':'seismic.correlate.preprocessing_td.detrend',
                                    'args':{'type':'constant'}},
                                  {'function':'seismic.correlate.preprocessing_td.TDfilter',
                                  'args':{'type':'bandpass','freqmin':2,'freqmax':8}},
                                # {'function':'seismic.correlate.preprocessing_td.taper',
                                #  'args': {'type':'cosine_taper','p':0.02}},
                                    # {'function':'seismic.correlate.preprocessing_td.mute',
                                    # 'args':{'taper_len':100.,
                                    #        'threshold':1000, absolute threshold
                                    #         'std_factor':3,
                                    #         'filter':{'type':'bandpass','freqmin':2,'freqmax':4},
                                    #         'extend_gaps':True}},
                                {'function':'seismic.correlate.preprocessing_td.clip',
                                    'args':{'std_factor':3}},
                                ],
                # Standard functions reside in seismic.correlate.preprocessing_fd
                'FDpreProcessing':[
                                    {'function':'seismic.correlate.preprocessing_fd.spectralWhitening',
                                    'args':{'joint_norm':False}},
                                    {'function':'seismic.correlate.preprocessing_fd.FDfilter',
                                    'args':{'flimit':[0.01,0.02,9,10]}}
                                    #  {'function':seismic.correlate.preprocessing_fd.FDsignBitNormalization,
                                    # 'args':{}}
                                    ]
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

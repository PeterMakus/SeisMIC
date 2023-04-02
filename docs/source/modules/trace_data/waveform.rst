
.. currentmodule:: seismic.trace_data.waveform

.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data]:::active --> correlate(Correlation)
        correlate -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        click waveform "../trace_data.html" "trace_data"
        click correlate "../correlate.html" "correlate"
        click monitor "../monitor.html" "monitor"
        click corrdb "../corrdb.html" "CorrDB"
        click dv "../monitor/dv.html" "DV"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;
    
    
Module to Download and Read Waveform Data
-----------------------------------------

The waveform module contains functions and classes that allow to read waveforms from a local database
or download waveform data from *FDSN servers* by using the `Obspy Mass Downloader <https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html>`_.
The downloaded data will then be written in a *seiscom data structure (SDS)*.
Both data download and management of the raw database are handled using a :class:`~seismic.trace_data.waveform.Store_Client` object.

Download Data
+++++++++++++
The recommended way to download continuous data for noise correlations is the method
:func:`~seismic.trace_data.waveform.Store_Client.download_waveforms_mdl`. To use this method, we first need to initialise a :class:`~seismic.trace_data.waveform.Store_Client` object.
The code block below shows and example for downloading data from the **IU.HRV** station.

.. code-block:: python
    :linenos:

    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime

    from seismic.trace_data.waveform import Store_Client

    root = '/path/to/project/'
    starttime = UTCDateTime(year=1990, julday=1)
    endtime = UTCDateTime.now()
    network = 'IU'
    station = 'HRV'

    # Note that you coul initiate the client object with
    # an Eida token if you should wish to download
    # restricted data.
    c = Client()
    sc = Store_Client(c, root, read_only=False)
    sc.download_waveforms_mdl(
        starttime, endtime, clients=[c], network=network,
        station=station, location='*', channel='*')

.. note::
    Instead of defining station and network codes, we could have used geographical coordinates
    to decide which stations to download. To do that, define the parameters ``minlon``, ``maxlon``,
    ``minlat``, and ``maxlat``.

.. note::
    :func:`~seismic.trace_data.waveform.Store_Client.download_waveforms_mdl` requires a list as input
    of clients.

Access Data
+++++++++++
.. note::
    If you just want to compute some noise correlations (i.e., Green's function estimations), you
    might not need this section.

Now, that we downloaded some waveform data, we might want to access it. This can as well be realised
using :class:`~seismic.trace_data.waveform.Store_Client`. In general, seismic data comes in two parts:
The waveform data in *.mseed* format and the Station Response information in *XML* format.
:class:`~seismic.trace_data.waveform.Store_Client` comes with several methods to fetch the two. There
are methods that will only read local data and methods that will check whether data is available locally
and, if not, download them from remote (i.e., the FDSN server that we defined when initialising the
:class:`~seismic.trace_data.waveform.Store_Client` object).

To access only already available data use:
    * response information:

      * :func:`~seismic.trace_data.waveform.Store_Client.read_inventory` to read all available response information.
      * :attr:`~seismic.trace_data.waveform.Store_Client.inventory` to access the pythonic inventory object
        (see `obspy's documentation <https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.Inventory.html>`_ for available methods)
    * waveform data:

      * :func:`~seismic.trace_data.waveform.Store_Client._load_local` to load local data

If you would like to download missing data, use the following methods:
    * Response Information:

      * :func:`~seismic.trace_data.waveform.Store_Client.select_inventory_or_load_remote`
    * waveform data:

      * :func:`~seismic.trace_data.waveform.Store_Client.get_waveforms`

Getting an overview of your database
====================================
Once your database grows in size, you might not exactly know anymore, which data you have available.
There are a couple of functions to help you:

    * :func:`~seismic.trace_data.waveform.Store_Client.get_available_stations` returns all codes of stations
        for which data is available (a network may or may not be defined).
    * :func:`~seismic.trace_data.waveform.Store_Client._get_times` returns the earliest and latest available
        starttimes for a certain station.


Feed in Data in a Different Way
+++++++++++++++++++++++++++++++
There may be scenarious, in which you will not need or won't be able to download waveform data
from an FDSN server. In such cases, it is easy to use your own mseed data to "mimic" SeisMIC's file
system structure.

You will need daily mseed files for each component of the seismometer.
If you need to convert/merge/split, you files we recommend using PyRocko or obspy.

Now you will have to sort your mseed files in the following way:

`path/to/project/mseed/{year}/{network}/{station}/{channel}.{sds_type}/{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}`

Where ``sds_type`` almost always corresponds to *D* (i.e., data). ``doy:03d`` is the "day of year"/Julian day in three digits.

For example
`path/to/project/mseed/2010/IU/HRV/BHZ.D/IU.HRV.00.BHZ.D.2010.001`
is the mseed file corresponding to a waveform recording from station *IU.HRV* channel *BHZ* on January 1st 2010.

.. note::
    The mseed files do not have file endings.

.. note::
    Files that do not correspond to this format won't be found by SeisMIC. Pay particular attention to
    always saving the mseed files for days 1 to 99 with leading zeros!


Station Inventories
===================
You might require response files either to provide station coordinates or to provide reponse information. SeisMIC reads *StationXML* files
these are saved in `/path/to/project/inventory/{network}.{station}.xml`. There should be exactly one file per station!
If you use SeisMIC's preimplemented methods to download data, station information will be downloaded automatically. Otherwise, add
it manually.





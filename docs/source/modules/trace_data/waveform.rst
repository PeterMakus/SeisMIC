
.. currentmodule:: miic3.trace_data.waveform

    
    
Module to Download and Read Waveform Data
-----------------------------------------

The waveform module contains functions and classes that allow to read waveforms from a local database
or download waveform data from *FDSN servers* by using the `Obspy Mass Downloader <https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html>`_.
The downloaded data will then be written in a *seiscom data structure (SDS)*.
Both data download and management of the raw database are handled using a :class:`~miic3.trace_data.waveform.Store_Client` object.

Download Data
+++++++++++++
The recommended way to download continuous data for noise correlations is the method
:func:`~miic3.trace_data.waveform.Store_Client.download_waveforms_mdl`. To use this method, we first need to initialise a :class:`~miic3.trace_data.waveform.Store_Client` object.
The code block below shows and example for downloading data from the **IU.HRV** station.

.. code-block:: python
    :linenos:

    from obspy.clients.fdsn import Client
    from obspy import UTCDateTime

    from miic3.trace_data.waveform import Store_Client

    root = '/path/to/project/'
    starttime = UTCDateTime(year=1990, julday=1)
    endtime = UTCDateTime.now()
    network = 'IU'
    station = 'HRV'


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
    :func:`~miic3.trace_data.waveform.Store_Client.download_waveforms_mdl` requires a list as input
    of clients.

Access Data
+++++++++++
.. note::
    If you just want to compute some noise correlations (i.e., Green's function estimations), you
    might not need this section.

Now, that we downloaded some waveform data, we might want to access it. This can as well be realised
using :class:`~miic3.trace_data.waveform.Store_Client`. In general, seismic data comes in two parts:
The waveform data in *.mseed* format and the Station Response information in *XML* format.
:class:`~miic3.trace_data.waveform.Store_Client` comes with several methods to fetch the two. There
are methods that will only read local data and methods that will check whether data is available locally
and, if not, download them from remote (i.e., the FDSN server that we defined when initialising the
:class:`~miic3.trace_data.waveform.Store_Client` object).

To access only already available data use:
    * response information:

      * :func:`~miic3.trace_data.waveform.Store_Client.read_inventory` to read all available response information.
      * :attr:`~miic3.trace_data.waveform.Store_Client.inventory` to access the pythonic inventory object
        (see `obspy's documentation <https://docs.obspy.org/packages/autogen/obspy.core.inventory.inventory.Inventory.html>`_ for available methods)
    * waveform data:

      * :func:`~miic3.trace_data.waveform.Store_Client._load_local` to load local data

If you would like to download missing data, use the following methods:
    * Response Information:

      * :func:`~miic3.trace_data.waveform.Store_Client.select_inventory_or_load_remote`
    * waveform data:

      * :func:`~miic3.trace_data.waveform.Store_Client.get_waveforms`

Getting an overview of your database
====================================
Once your database grows in size, you might not exactly know anymore, which data you have available.
There are a couple of functions to help you:

    * :func:`~miic3.trace_data.waveform.Store_Client.get_available_stations` returns all codes of stations
        for which data is available (a network may or may not be defined).
    * :func:`~miic3.trace_data.waveform.Store_Client._get_times` returns the earliest and latest available
        starttimes for a certain station.



.. autosummary::
    :toctree: autogen
    :nosignatures:
        
    Store_Client
    FS_Client
    read_from_filesystem
.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data] --> correlate(Correlation)
        correlate -->|save| corrdb[(CorrDB/hdf5)]
        corrdb:::active --> monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        click waveform "../trace_data.html" "trace_data"
        click correlate "../correlate.html" "correlate"
        click monitor "../monitor.html" "monitor"
        click corrdb "../corrdb.html" "CorrDB"
        click dv "../monitor/dv.html" "DV"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;

CorrelationDataBase and DBHandler objects
-----------------------------------------

In **SeisMIC**, correlations are stored in `HDF5 <https://www.hdfgroup.org/downloads/hdf5/>`_ container files.
This has the advantage of avoiding potential overhead caused by large amounts of correlation files.
SeisMIC's implementation relies on modified `h5py <https://www.h5py.org/>`_ classes. After computing your
correlations as shown in the earlier steps, they will be saved in one file *per station-combination*
(e.g., the file ``IU-TA.HRV-M58A`` holds the correlations of all components and locations of the two stations
with each other, whereas the file ``IU-IU.HRV-HRV`` holds all autocorrelations and intercomponent correlations
of the station ``IU.HRV``).

.. note::
    **SeisMIC** makes use of SEED-like station codes. The general logic of those codes is:
    ``net0-net1.stat0-stat1.loc0-loc1.ch0-ch1``. Here, 0 is the first station and 1 the second station.
    Correlations will always only be computed alphabetically. That is, **SeisMIC** will not compute a
    correlation for ``TA-IU.M58A-HRV`` but only for ``IU-TA.HRV-M58A``.

As a user, you will only ever be calling the :py:class:`~seismic.db.corr_hdf5.CorrelationDataBase` class.
The only function of this class is to return a :py:class:`~seismic.db.corr_hdf5.DBHandler`, which hold all the
"useful" functions. To call :py:class:`~seismic.db.corr_hdf5.CorrelationDataBase`, use a context manager like so:

>>> from seismic.db.corr_hdf5 import CorrelationDataBase
>>> with CorrelationDataBase('/path/to/myfile.h5') as cdb:
>>>     type(cdb)  # This is a DBHandler
<class 'seismic.db.corr_hdf5.DBHandler'>

.. warning::

    Do not call :py:class:`~seismic.db.corr_hdf5.DBHandler` directly! This might lead to unexpected behaviour or
    even dataloss due to corrupted hdf5 files.

.. warning::

    If you should for some reason decide to not use the context manager, you will have to close the hdf5 file
    with :py:meth:`seismic.db.corr_hdf5.DBHandler._close` to avoid corrupting your files!

Reading Correlations
++++++++++++++++++++

The most common usecase is probably that you will want to access correlations that **SeisMIC** computed
for you (as shown earlier). To do so, you can use the :py:meth:`~seismic.db.corr_hdf5.DBHandler.get_data`
method:

>>> from seismic.db.corr_hdf5 import CorrelationDataBase
>>> with CorrelationDataBase('/path/to/myfile.h5', mode='r') as cdb:
>>>     cst = cdb.get_data(
>>>         tag='subdivision', network='IU-IU', station='*', channel='??Z-??Z', corr_start=None, corr_end=None)
>>> # cst is a CorrStream object on that we can use our known methods
>>> print(type(cst))
<class 'seismic.correltea.stream.CorrStream'>
>>> #cst.count()
289

As you can see, we use station combination codes (described in the *note* box above) to identify data. All arguments accept wildcards.
The data we are loading are correlations from waveforms recorded between ``corr_start`` and ``corr_end``. Setting those arguments to ``None``
is similar to using a wildcard (i.e., load from earliest to latest available data).

Tags
####

**SeisMIC** uses the following standard tags:

1. ``subdivision``: Are the unstacked correlations with the the correlation length of ``corr_len`` as defined in our `yaml file <../correlate/get_started.html#setting-the-parameters>`_.
2. ``stack_$stacklen$`` : Is the standard tag for correlation stacks, where $stacklen$ should be replaced by the stack length in seconds.


Obtain correlation parameters
#############################

You might want to get the dictionary that you used to produce the correlations in the file. You can do that by using
:py:meth:`seismic.corr_hdf5.corrdb.DBHandler.get_corr_options`.

Getting an overview over available data
#######################################

Once you have a suffieciently large dataset, you might be confused about which data you have already produced.
In this case, **SeisMIC** offers several methods to make your life a little easier:

1. :py:meth:`seismic.corr_hdf5.corrdb.DBHandler.get_available_starttimes`: Returns a dictionary
   of available starttimes for your chosen network, station, and channel combinations (wildcards are allowed).
2. :py:meth:`seismic.corr_hdf5.corrdb.DBHandler.get_available_channels`:
   Returns the available channel combinations for a given station combination.
3. **Access the DBHandler like a dictionary**: Just like in h5py, it is possible to access the :py:class:`~seismic.db.corr_hdf5.corrdb.DBHandler` like a dictionary. The logic works as follows:
   dbh[tag][netcomb][statcomb][chacomb][corr_start][corr_end]

Following the logic of the structure above, we can get a list of all available tags as follows:

>>> print(list(dbh.keys()))
['stack_34798', 'subdivision']

Writing Correlations
++++++++++++++++++++

If you postprocess your correlations (e.g., stacking), you might want to save the data afterwards.
When writing to the correlation hdf5 files,
you will have to pay attention to a couple of particularities:

1. You need to provide a ``corr_options`` dictionary to be able to open the file with ``mode!=r``.
   If you don't provide a dictionary or your dictionary is different from the one used to produce the data, the code will raise an error.
   This is meant to prevent mixing of differently processed data.
2. You should consider using a sensible convention for your tags (if saving stacks, it's best to stick to the standard convention as discussed above).

.. code-block:: python
    :linenos:

    import yaml

    from seismic.db.corr_hdf5 import CorrelationDataBase
    from seismic.correlate.stream import CorrStream

    # For this example, we are just gonna create an empty CorrStream
    # Of course, this will not really add any data to the file
    cst = CorrStream()

    # Get your correlation dictionary
    with open('/path/to/my/params.yaml') as file:
        co = yaml.load(file, Loader=yaml.FullLoader)
    with CorrelationDataBase('/path/to/myfile.h5', mode='w', corr_options=co) as cdb:
        cdb.add_correlation(cst, tag='my_sensible_tag')

If there had been any data in our :py:class:`~seismic.correlate.stream.CorrStream`, we could retrieve it as shown above.
Network, station, and channel information are determined automatically from the :py:class:`~seismic.correlate.stream.CorrTrace` header.

Why won't SeisMIC allow me to an old file again?
################################################

Sometimes when re-executing your correlation workflow, you might encounter a ``PermissionError``. This happens whenever you change your processing
parameters in your *params.yaml* file under *corr*. Essentially, SeisMIC prevents you from mixing correlations that have been processed in different
ways. **So if you want to recompute your correlations with a new set of processing parameters, you need to write them to a different folder!**
    
    
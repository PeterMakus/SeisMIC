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

The Correlator Object
---------------------

The Object governing the actual computation is :class:`~miic3.correlate.correlate.Correlator`. It takes only two input arguments and the most important is the dictionary / yml file that we created in `the preceding step <./get_started.html#download-data>`_.
The second input argument is the :class:`~miic3.trace_data.waveform.Store_Client` we used to `download <../trace_data/waveform.html#download-data>`_ our data.

As Computing Ambient Noise Correlations is computationally very expensive, **SeisMIIC** can be used in conjuction with MPI (e.g., openMPI) to enable processing with several cores or on HPCs.

A script to start your correlation could look like this:

.. code-block:: python
    :caption: mycorrelation.py
    :linenos:

    from time import time

    from obspy.clients.fdsn import Client

    from miic3.correlate.correlate import Correlator
    from miic3.trace_data.waveform import Store_Client

    # Path to the paramter file we created in the step before
    params = '/path/to/my/params.yaml'
    # You don't have to set this (could be None)
    client = Client('MYFDSN')
    client.set_eida_token('/I/want/embargoed/data.txt)
    # root is the same as proj_dir in params.yaml
    root = '/path/to/project/'
    sc = Store_Client(client, root)

    c = Correlator(sc, options=params)
    print('Correlator initiated')
    x = time()
    st = c.pxcorr()
    print('Correlation finished after', time()-x, 'seconds')

This script can be iniated in bash using:

.. code-block:: bash

    mpirun -n $number_cores$ python mycorrelation.py

where ``$number_cores$`` is the number of cores you want to initialise. The only method of :class:`~miic3.correlate.correlate.Correlator` that you will want to use is :meth:`~miic3.correlate.correlate.Correlator.pxcorr()`, which does not require any (additional) input.
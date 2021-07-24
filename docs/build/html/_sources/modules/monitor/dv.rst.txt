.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data] --> correlate(Correlation)
        correlate -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        monitor[Measure dv] -->|save| dv{{DV}}:::active
        click waveform "../trace_data.html" "trace_data"
        click correlate "../correlate.html" "correlate"
        click monitor "../monitor.html" "monitor"
        click corrdb "../corrdb.html" "CorrDB"
        click dv "../monitor/dv.html" "DV"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;

Reading and Handling of DV objects
----------------------------------

**SeisMIIC** uses :class:`~miic3.monitor.dv.DV` objects to handle velocity changes. Those can for example saved, loaded, and plotted.

Read a DV Object from Disk
++++++++++++++++++++++++++

Most likely, you will want to ready the velocity changes that were computed in the previous step. **SeisMIIC** uses binary ``npz`` format to
store those. You will find the files in the folder that you have defined earlier (i.e., in the yaml file).
Load the object with :func:`miic3.monitor.dv.read_dv`. It only takes one argument: the path to the dv object.

.. code-block:: python

    from miic3.monitor.dv import read_dv

    dv = read_dv('/path/to/my/dv/DV-net0-net1.stat0-stat1.ch0-ch1.npz')

Plotting
++++++++

You can create a plot of the dv object with :meth:`miic3.monitor.dv.DV.plot`. The result will look like this:

.. image:: ../../figures/XN-XN_NEP06-NEP06_HHE-HHN_change.png
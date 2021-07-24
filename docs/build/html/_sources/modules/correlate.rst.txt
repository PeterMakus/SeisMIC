.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data] --> correlate(Correlation)
        correlate:::active -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        monitor[Measure dv] -->|save| dv{{DV}}
        click waveform "./trace_data.html" "trace_data"
        click correlate "./correlate.html" "correlate"
        click monitor "./monitor.html" "monitor"
        click corrdb "./corrdb.html" "CorrDB"
        click dv "./monitor/dv.html" "DV"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;

Compute and Handle Correlations
===============================

Compute Correlations
--------------------

.. toctree::
    :maxdepth: 3

    correlate/get_started
    correlate/correlator

Handle Correlations
-------------------

.. toctree::
    :maxdepth: 3

    correlate/stream

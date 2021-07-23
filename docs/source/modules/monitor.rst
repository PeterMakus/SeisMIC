.. mermaid::

    %%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
    graph LR
        waveform[Get Data] --> correlate(Correlation)
        correlate -->|save| corrdb[(CorrDB/hdf5)]
        corrdb --> monitor
        monitor[Measure dv]:::active -->|save| dv{{DV}}
        click waveform "./trace_data.html" "trace_data"
        click correlate "./correlate.html" "correlate"
        click monitor "./monitor.html" "monitor"
        click corrdb "./corrdb.html" "CorrDB"
        classDef active fill:#f666, stroke-width:4px, stroke:#f06;

Monitor Velocity Changes
========================

.. toctree::
    :maxdepth: 3

    monitor/monitor
    monitor/dv
